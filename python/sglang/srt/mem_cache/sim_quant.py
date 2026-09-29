"""Reference KV-quantization round trips in the BF16 write path.

The module applies offline-generated TurboQuant or OSCAR parameters before BF16 storage. It is
used to evaluate quantization accuracy when no native cache representation is available and does
not reduce memory use. It is inactive unless ``SGLANG_SIMQUANT_PATH`` is set.
"""
from __future__ import annotations

from typing import Optional

import torch

# Cache bundles by path so multiple configurations can coexist in one process.
_BUNDLES: dict = {}


def load_bundle(path: str, device) -> dict:
    """Load once and move the per-layer rotations to the compute device.

    The first load logs the active configuration for experiment provenance.
    """
    if path not in _BUNDLES:
        b = torch.load(path, map_location="cpu", weights_only=False)
        codec = str(b.get("codec", "turboquant"))
        print(
            f"[sim_quant] ACTIVE: codec={codec} K={int(b['k_bits'])}b V={int(b['v_bits'])}b "
            f"head_dim={int(b['head_dim'])} layers={int(b['n_layers'])} from {path} "
            f"(simulated: dequantized values stored as BF16, no memory saving)",
            flush=True,
        )
        loaded = {
            "codec": codec,
            "k_bits": int(b["k_bits"]),
            "v_bits": int(b["v_bits"]),
            "head_dim": int(b["head_dim"]),
            "n_layers": int(b["n_layers"]),
            "k_rotation": b["k_rotation"].to(device).float(),
            "v_rotation": b["v_rotation"].to(device).float(),
        }
        if codec == "turboquant":
            loaded["k_centroids"] = b["k_centroids"].to(device).float()
            loaded["v_centroids"] = b["v_centroids"].to(device).float()
        elif codec == "oscar_affine":
            # The mixed pool's scalar codec: rotate, percentile-clip the row,
            # per-group min-max affine at k_bits/v_bits. groups=1 is the served
            # per-(token, head) single scale; the bundle carries the knobs so a
            # bits/groups ladder is a bundle swap, not a code change.
            loaded["k_clip"] = float(b.get("k_clip", 1.0))
            loaded["v_clip"] = float(b.get("v_clip", 1.0))
            loaded["groups"] = int(b.get("groups", 1))
        else:
            raise ValueError(f"[sim_quant] unknown codec {codec!r} in {path}")
        _BUNDLES[path] = loaded
    return _BUNDLES[path]


@torch.no_grad()
def _roundtrip(x: torch.Tensor, Pi: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """TurboQuant MSE stage: unit-sphere normalise -> rotate -> nearest centroid -> undo.

    x is (..., D); the norm is per VECTOR (per token per head), matching MSECompressor.compress,
    which divides by ``torch.norm(flat, dim=-1) + 1e-8``. Getting this wrong -- normalising per
    tensor, or skipping it -- would mis-scale every centroid, since the centroids are solved for
    N(0, 1/D) and only match unit-norm rotated coordinates.
    """
    orig_dtype = x.dtype
    D = x.shape[-1]
    flat = x.reshape(-1, D).float()
    norms = flat.norm(dim=-1, keepdim=True)
    rotated = (flat / (norms + 1e-8)) @ Pi.T
    # The reference stores the per-vector norm as fp16 (`vec_norms.to(torch.float16)`) and reads it
    # back as fp32, so the norm is itself quantized. Keeping fp32 here would be simulating a
    # quantizer slightly better than the real one -- and that fp16 norm is exactly the 16/128 =
    # 0.125 bits/coord the rate accounting charges. The equivalence gate catches the difference as
    # 1-2 bf16 ULPs, which is how this was found.
    norms = norms.to(torch.float16).float()
    # Nearest centroid, chunked over tokens: the (N, D, levels) difference tensor is what OOMs the
    # reference implementation on long contexts, and a prefill chunk here can be 4096 tokens x heads.
    out = torch.empty_like(rotated)
    step = max(1, 2_000_000 // (D * centroids.numel()))
    for i in range(0, rotated.shape[0], step):
        seg = rotated[i : i + step]
        idx = (seg.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        out[i : i + step] = centroids[idx]
    return ((out @ Pi) * norms).reshape(x.shape).to(orig_dtype)


@torch.no_grad()
def _roundtrip_affine(
    x: torch.Tensor, R: torch.Tensor, bits: int, clip_ratio: float, groups: int
) -> torch.Tensor:
    """OSCAR scalar stage: rotate -> percentile clip -> per-group min-max intN -> undo.

    Matches the serving int2 writer's semantics (oscar_rotation_clip_int2_kv.py):
    x @ R into the calibrated basis, threshold = sort(|row|)[int(clip*D)] over the
    FULL head row (clip is per row even when the min-max scale is grouped), clamp,
    then affine min-max over each group of D/groups coords. R is orthogonal, so
    quantizing rotated coordinates and rotating back is exactly what attention
    computes against the stored rotated keys.
    """
    orig_dtype = x.dtype
    D = x.shape[-1]
    flat = x.reshape(-1, D).float() @ R
    if clip_ratio < 1.0:
        idx = min(max(int(clip_ratio * D), 0), D - 1)
        thr = flat.abs().sort(dim=-1).values[:, idx : idx + 1]
        flat = flat.clamp(-thr, thr)
    qmax = float(2**bits - 1)
    g = flat.reshape(-1, groups, D // groups)
    vmin = g.amin(-1, keepdim=True)
    vmax = g.amax(-1, keepdim=True)
    scale = (vmax - vmin).clamp_min(1e-8) / qmax
    zero = -vmin / scale
    q = torch.clamp(g / scale + zero + 0.5, 0, qmax).floor()
    deq = ((q - zero) * scale).reshape(-1, D)
    return (deq @ R.T).reshape(x.shape).to(orig_dtype)


_STATS = {"calls": 0, "k_err": 0.0, "k_sig": 0.0, "v_err": 0.0, "v_sig": 0.0, "layers": set(),
          "nonfinite_reports": 0, "nonfinite_writes": 0, "first_nonfinite_layer": None}
_LOG_EVERY = 2000


@torch.no_grad()
def apply_sim_quant(path: str, layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor):
    """Return (k, v) passed through the simulated quantizer. Shapes and dtypes are preserved."""
    b = load_bundle(path, cache_k.device)
    assert cache_k.shape[-1] == b["head_dim"], (
        f"sim-quant bundle is for head_dim {b['head_dim']}, got {cache_k.shape[-1]}"
    )
    assert layer_idx < b["n_layers"], (
        f"sim-quant bundle has {b['n_layers']} layers, got layer_idx {layer_idx}"
    )
    if b["codec"] == "oscar_affine":
        k = _roundtrip_affine(cache_k, b["k_rotation"][layer_idx], b["k_bits"],
                              b["k_clip"], b["groups"])
        v = _roundtrip_affine(cache_v, b["v_rotation"][layer_idx], b["v_bits"],
                              b["v_clip"], b["groups"])
    else:
        k = _roundtrip(cache_k, b["k_rotation"][layer_idx], b["k_centroids"])
        v = _roundtrip(cache_v, b["v_rotation"][layer_idx], b["v_centroids"])

    # Report running NMSE so logs confirm that the configured quantizer is active.
    #
    # Skipped while a CUDA graph is capturing: `.item()` is a device->host sync and raises
    # "operation not permitted when stream is capturing". The QUANTIZER itself is unaffected -- its
    # ops are recorded into the graph and so run on every replay -- but the Python-side counters
    # cannot, so these totals cover eager/prefill writes only and undercount decode. That is enough
    # to prove the values change; it is not a call count.
    if not torch.cuda.is_current_stream_capturing():
        s = _STATS
        s["calls"] += 1
        s["layers"].add(layer_idx)

        # Distinguish non-finite inputs from non-finite quantizer outputs and cap log volume.
        if s["nonfinite_writes"] > 200:
            raise RuntimeError(
                f"[sim_quant] ABORT: {s['nonfinite_writes']} writes with non-finite KV "
                f"(first seen layer {s['first_nonfinite_layer']}). The model diverged under this "
                f"quantizer; refusing to produce a score that would read as an accuracy result."
            )
        if s["nonfinite_reports"] < 12:
            k_in_bad = not bool(torch.isfinite(cache_k).all())
            v_in_bad = not bool(torch.isfinite(cache_v).all())
            k_out_bad = not bool(torch.isfinite(k).all())
            v_out_bad = not bool(torch.isfinite(v).all())
            if k_in_bad or v_in_bad or k_out_bad or v_out_bad:
                s["nonfinite_writes"] += 1
                if s["first_nonfinite_layer"] is None:
                    s["first_nonfinite_layer"] = layer_idx
                s["nonfinite_reports"] += 1
                nmax = cache_k.float().norm(dim=-1).max().item()
                print(
                    f"[sim_quant] NON-FINITE at layer {layer_idx} write {s['calls']}: "
                    f"in K={k_in_bad} V={v_in_bad} | out K={k_out_bad} V={v_out_bad} | "
                    f"max||k||={nmax:.1f} (fp16 norm store overflows above 65504)",
                    flush=True,
                )
        s["k_err"] += (k.float() - cache_k.float()).pow(2).sum().item()
        s["k_sig"] += cache_k.float().pow(2).sum().item()
        s["v_err"] += (v.float() - cache_v.float()).pow(2).sum().item()
        s["v_sig"] += cache_v.float().pow(2).sum().item()
        if s["calls"] % _LOG_EVERY == 0:
            print(
                # Read bit-widths from the bundle so diagnostic labels match the loaded artifact.
                f"[sim_quant] eager_writes={s['calls']} layers_seen={len(s['layers'])}/"
                f"{b['n_layers']} running NMSE: K({b['k_bits']}b) "
                f"{s['k_err'] / max(s['k_sig'], 1e-30):.6f} "
                f"V({b['v_bits']}b) {s['v_err'] / max(s['v_sig'], 1e-30):.6f}"
                f"  <- nonzero => not BF16",
                flush=True,
            )
    return k, v


def describe(path: str) -> str:
    b = torch.load(path, map_location="cpu", weights_only=False)
    kb, vb, D = int(b["k_bits"]), int(b["v_bits"]), int(b["head_dim"])
    codec = str(b.get("codec", "turboquant"))
    if codec == "oscar_affine":
        g = int(b.get("groups", 1))
        # 2 scale entries (scale+zero) per group, charged at fp32 like the pool's
        # SGLANG_MIXED_KV_SCALE_DTYPE default.
        bpe = ((D * kb + g * 64) / D + (D * vb + g * 64) / D) / 2
        return (f"sim-quant OSCAR-affine K={kb}b V={vb}b groups={g} "
                f"clip={b.get('k_clip')}/{b.get('v_clip')} head_dim={D} "
                f"-> {bpe:.3f} BPE if packed (SIMULATED: stores BF16, no memory saving)")
    bpe = ((D * kb + 16) / D + (D * vb + 16) / D) / 2
    return (f"sim-quant TurboQuant K={kb}b V={vb}b head_dim={D} "
            f"-> {bpe:.3f} BPE if packed (SIMULATED: stores BF16, no memory saving)")
