"""Group-VQ codebook bundle loader + torch-side encode helpers for the vq2
K quant tier of the unified mixed HP+int2 pool.

Bundle schema (produced by the ``exp/calibration/fit_vq_codebook.py`` trainer):

    forward   [L, H, D, D]   residual map: r = (k - mean) @ forward  (per head)
    inverse   [L, H, D, D]   recon map:    k_hat = r_hat @ inverse + mean
    mean      [L, H, D]
    codebooks {(l, h): list of NG tensors [K, G]}   (fp16 or fp8_e4m3fn)
    bounds    [(start, end, bits)] * NG   -- flat allocation, contiguous groups
    pertoken_norm bool -- per-token RMS scale on r before lookup

Engine storage convention: BOTH tiers hold the residual ``r`` (HP as bf16
rows, quant as VQ indices + per-token RMS scale). Queries are mapped with
``q @ inverse.T`` so ``q_m . r = q . (k - mean)``; the ``-q . mean`` term is
constant across keys for a given query, hence softmax-invariant, and it is
identical for both tiers. Models with a learned attention sink are the
exception: the sink is an extra denominator logit, not a real key, so it must
receive the same per-query ``-q . mean`` shift. Prefill adjusts it explicitly;
decode fuses the adjustment into unified stage 2.

The decode kernel reconstructs codewords from a packed int32 (4x fp8-e5m2
bytes, little-endian => byte i is coord i of the group; e5m2 because sm80
Triton only bitcasts fp8e5). Encode assigns against the *fp8-dequantized*
centroids so the encoder is a true nearest-neighbor for what the decoder
actually reconstructs.
"""

from __future__ import annotations

import functools
import logging
import os

import triton
import triton.language as tl
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def resolve_vq_fp8_fmt() -> str:
    """Resolve ``SGLANG_VQ_FP8_FMT`` to ``e5m2`` or ``e4m3``.

    ``e5m2`` is the default and supports A100-class devices. ``auto`` selects ``e4m3`` on
    compute capability 8.9 or newer. The selected format is recorded when the bundle is loaded.
    """
    import os

    want = os.environ.get("SGLANG_VQ_FP8_FMT", "e5m2").lower()
    if want not in ("auto", "e5m2", "e4m3"):
        raise ValueError(f"SGLANG_VQ_FP8_FMT must be auto|e5m2|e4m3, got {want!r}")
    if want == "auto":
        cap = torch.cuda.get_device_capability()
        return "e4m3" if cap >= (8, 9) else "e5m2"
    if want == "e4m3" and torch.cuda.get_device_capability() < (8, 9):
        raise ValueError(
            "SGLANG_VQ_FP8_FMT=e4m3 requires compute capability >= 8.9 "
            f"(this device is sm{''.join(map(str, torch.cuda.get_device_capability()))})"
        )
    return want


@dataclass
class VQCodebook:
    forward: torch.Tensor    # [L, H, D, D] bf16 -- k-side map (applied to k - mean)
    q_map: torch.Tensor      # [L, H, D, D] bf16 -- inverse.transpose(-1,-2), q-side map
    mean: torch.Tensor       # [L, H, D] bf16
    cb16: torch.Tensor       # [L, H, NG, K, G] fp16 -- fp8-dequantized centroids
    cb_sq: torch.Tensor      # [L, H, NG, K] fp32 -- 0.5 * ||c||^2 per centroid
    cb_packed: torch.Tensor  # [L, H, NG, K] int32 -- fp8 bytes packed little-endian
    num_groups: int          # NG
    codebook_size: int       # K
    group_dim: int           # G
    pertoken_norm: bool
    source_fp8_fmt: str | None      # fp8 format the bundle declares, if any
    centroid_resnap_rel_rmse: float  # error added by re-snapping at load
    fp8_fmt: str = "e5m2"    # "e5m2" | "e4m3" -- decode kernel must bitcast to match
    wide_g: bool = False      # G != 4 -- decode via plain fp16 gather, not packed-int32
    idx_dtype: torch.dtype = torch.uint8  # uint8 (K<=256) or int16 (K>256) index arena
    perm_rope: bool = False  # decode skips q_map entirely (see decode_attention.py's
    # PERM_ROPE branch) -- K is unweighted+rotated via freq_idx/w_perm to land on
    # plain original channels, and Q is read unmapped. Requires freq_idx/w_perm.
    identity_map: bool = False  # forward=inverse=I and mean=0; skip redundant transforms
    freq_idx: torch.Tensor = None  # [L, H, D//2] int64, only used when perm_rope
    w_perm: torch.Tensor = None    # [L, H, D] fp32, only used when perm_rope
    cb16_dec_pair_major: bool = False  # cb16_dec holds each centroid as [e0..e3 | o0..o3]
    # instead of pair-interleaved. A coordinate relabelling inside the group, so the quantiser
    # is unchanged; it lets the decode kernel halve at a register boundary instead of unpacking
    # evens from odds. See PM_FUSED in decode_attention.py.
    cb16_dec: torch.Tensor = None  # [L, H, NG, K, G] fp16 with 1/w folded in; decode reads
    # this instead of cb16 so PERM_ROPE needs no per-element unweight in the KV loop. Built
    # at load for perm_rope bundles only; see the fold note in load_vq_codebook.
    pool_heads_scale: bool = False  # pertoken_norm pools RMS across ALL KV heads, not
    # per-(token,head) independently like
    # the pool's own pertoken_norm. Decode needs no change (K_Scales_Zeros already stores
    # one value per (token,head); encode just writes the SAME pooled value to every
    # head's slot instead of an independent one). The arena stores one physical copy
    # using the serving configuration's scale dtype (FP16 for the published TaSQ setup).

    def layer(self, idx: int):
        return (
            self.forward[idx],
            self.q_map[idx],
            self.mean[idx],
            self.cb16[idx],
            self.cb_sq[idx],
            self.cb_packed[idx],
        )


def load_vq_codebook(
    path: str,
    *,
    layer_num: int,
    start_layer: int,
    head_num: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    head_start: int = 0,
    fold_decode: bool = False,
) -> VQCodebook:
    """``head_num`` is the pool's LOCAL head count; under tensor parallelism
    the bundle holds all global KV heads and ``head_start`` selects this
    rank's contiguous slice (Megatron-style sharding: rank r owns heads
    [r*local, (r+1)*local))."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    F = blob["forward"]
    inv = blob["inverse"]
    mean = blob["mean"]
    bounds = blob["bounds"]
    ptn = bool(blob.get("pertoken_norm", False))
    pool_heads_scale = bool(blob.get("pool_heads_scale", False))
    perm_rope = bool(blob.get("perm_rope", False))
    source_fp8_fmt = blob.get("fp8_fmt")

    L_total, H_total, D, D2 = F.shape
    assert D == D2 == head_dim, f"codebook head_dim {D} != pool head_dim {head_dim}"
    head_end = head_start + head_num
    assert head_end <= H_total and (head_start == 0 or H_total % head_num == 0), (
        f"codebook has {H_total} heads, rank wants [{head_start}, {head_end}) "
        f"— TP size must divide the KV head count (no replication support)"
    )
    H = head_num
    F = F[:, head_start:head_end]
    inv = inv[:, head_start:head_end]
    mean = mean[:, head_start:head_end]
    if perm_rope:
        freq_idx_full = blob["freq_idx"][:, head_start:head_end]
        w_perm_full = blob["w_perm"][:, head_start:head_end]
    end_layer = start_layer + layer_num
    assert end_layer <= L_total, (
        f"codebook has {L_total} layers, pool wants [{start_layer}, {end_layer})"
    )

    # Flat contiguous groups only (the stratified permutation is folded into
    # ``forward``); the decode kernel assumes uniform (K, G) across groups.
    starts = [s for (s, _e, _b) in bounds]
    ends = [e for (_s, e, _b) in bounds]
    NG = len(bounds)
    G = ends[0] - starts[0]
    assert starts == list(range(0, D, G)) and all(
        e - s == G for s, e in zip(starts, ends)
    ), f"vq2 requires uniform contiguous groups, got bounds={bounds[:4]}..."

    cbs = blob["codebooks"]
    K = cbs[(0, 0)][0].shape[0]
    cb = torch.empty((layer_num, H, NG, K, G), dtype=torch.float16)
    for l in range(layer_num):
        for h in range(H):
            entry = cbs[(start_layer + l, head_start + h)]
            for g in range(NG):
                c = entry[g]
                assert c.shape == (K, G), f"codebook ({l},{h},{g}) shape {c.shape}"
                cb[l, h, g] = c.to(torch.float16)

    # Snap centroids to their fp8 representation, then build both the packed
    # int32 decode view and the matching fp16 encode view from the SAME bytes.
    #
    # e5m2 has 2 mantissa bits; e4m3 has 3. Triton only admits fp8e4nv at
    # compute capability >= 8.9 (backends/nvidia/compiler.py gates it), so
    # sm80/A100 must stay on e5m2 -- that is what the A100 record was measured
    # with. On sm89+/sm90 e4m3 roughly halves the centroid representation error
    # (measured 5.51% -> 2.66% rel L2 on the gpqacc64k bundle).
    #
    # The loader's snap and the decode kernel's bitcast MUST agree: the encoder
    # assigns against the centroids the decoder reconstructs. fp8_fmt is carried
    # on VQCodebook and threaded to the kernel as a constexpr for exactly that.
    fmt = resolve_vq_fp8_fmt()
    cb_fp8 = cb.to(torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2)
    centroid_resnap_rel_rmse = float(
        (
            (cb_fp8.to(torch.float32) - cb.to(torch.float32)).pow(2).sum()
            / cb.to(torch.float32).pow(2).sum().clamp_min(1e-30)
        )
        .sqrt()
        .item()
    )
    if source_fp8_fmt is not None and centroid_resnap_rel_rmse > 0:
        logger.warning(
            "vq2: codebook %s declares fp8_fmt=%s but stores dequantized "
            "centroids; runtime %s packing re-snaps them (relative RMSE %.6f). "
            "Use a raw trained bundle to measure single-snap fidelity.",
            path,
            source_fp8_fmt,
            fmt.upper(),
            centroid_resnap_rel_rmse,
        )
    # Packed-int32 fp8 codewords (the bandwidth-optimized decode path) require
    # G == 4 -- 4 fp8 bytes pack losslessly into one int32 word, and the decode
    # kernel's byte-unpack/join is hand-unrolled for exactly that. For G != 4
    # (e.g. this project's own 8-dim CQ/TaSQ codebooks), skip the fp8-packed
    # path entirely rather than extending the byte-interleave trick to
    # multi-word groups: a wrong interleave would silently corrupt every
    # reconstructed vector, and there is no way to hand-verify Triton's
    # tl.join axis order without running it. The decode kernel instead does a
    # plain per-group fp16 gather from `cb16` for these ("wide-G") bundles --
    # slower (no fp8/shared-memory staging), but its correctness reduces to an
    # ordinary indexed load, and it also avoids the fp8-snap accuracy loss
    # this project's own centroids never asked for.
    wide_g = G != 4
    if wide_g:
        # [layer_num, 1] (not a bare [1]) so call sites that index
        # cb_packed[layer_idx] for every layer -- unconditionally, before
        # branching on wide_g -- stay in bounds; the value is never read.
        cb_packed = torch.zeros((layer_num, 1), dtype=torch.int32)
        cb16 = cb.to(torch.float16)
        centroid_resnap_rel_rmse = 0.0
        fmt = "none"
    else:
        cb_packed = (
            cb_fp8.contiguous().view(torch.int32).squeeze(-1).contiguous()
        )  # [L, H, NG, K]
        cb16 = cb_fp8.to(torch.float16)
    cb_sq = 0.5 * cb16.to(torch.float32).pow(2).sum(-1)  # [L, H, NG, K]
    # Index width: uint8 caps codebook_size at 256 (the G=4/bpc=1 arm,
    # K=16, fits trivially). This project's 8-dim CQ/TaSQ codebooks are K=1024
    # (10 bits, 1.25 bit/coord) -- indices >255 would silently wrap in uint8,
    # not error, so this must be sized from K rather than left at the old
    # hardcoded uint8.
    idx_dtype = torch.uint8 if K <= 256 else torch.int16

    Fl = F[start_layer:end_layer].to(dtype)
    q_map = inv[start_layer:end_layer].transpose(-1, -2).to(dtype).contiguous()
    mean_l = mean[start_layer:end_layer].to(dtype)
    freq_idx_l = freq_idx_full[start_layer:end_layer].to(device) if perm_rope else None
    w_perm_l = w_perm_full[start_layer:end_layer].to(torch.float32).to(device) if perm_rope else None

    # Identity detection: exact, on the dtype the kernels actually consume, so a
    # bundle that is identity only after rounding is NOT skipped.
    _D = Fl.shape[-1]
    _eye = torch.eye(_D, dtype=Fl.dtype, device=Fl.device)
    identity_map = bool(
        torch.equal(Fl, _eye.expand_as(Fl))
        and torch.equal(q_map, _eye.expand_as(q_map))
        and not mean_l.any()
    )
    if identity_map and os.environ.get("SGLANG_VQ_NO_IDENTITY_SKIP") == "1":
        # Debug escape ONLY -- restores the pre-2026-09-16 behaviour (multiply by I) so the
        # skip can be A/B'd against itself. Not a feature flag: the skip is bit-identical, so
        # there is no reason to run without it outside of that comparison.
        identity_map = False
        print("[vq] identity map detected but SGLANG_VQ_NO_IDENTITY_SKIP=1 -- NOT skipping",
              flush=True)
    if identity_map:
        print(
            "[vq] identity k/q map detected (forward == q_map == I, mean == 0): "
            "skipping both map kernels",
            flush=True,
        )

    out = VQCodebook(
        forward=Fl.to(device),
        q_map=q_map.to(device),
        identity_map=identity_map,
        mean=mean_l.to(device),
        cb16=cb16.to(device),
        cb_sq=cb_sq.to(device),
        cb_packed=cb_packed.to(device),
        num_groups=NG,
        codebook_size=K,
        group_dim=G,
        pertoken_norm=ptn,
        source_fp8_fmt=source_fp8_fmt,
        centroid_resnap_rel_rmse=centroid_resnap_rel_rmse,
        fp8_fmt=fmt,
        wide_g=wide_g,
        idx_dtype=idx_dtype,
        pool_heads_scale=pool_heads_scale,
        perm_rope=perm_rope,
        freq_idx=freq_idx_l,
        w_perm=w_perm_l,
    )
    logger.info(
        "vq2: loaded codebook %s (layers=%d heads=[%d,%d)/%d NG=%d K=%d "
        "G=%d ptn=%s fp8=%s source_fp8=%s resnap_rel_rmse=%.6f wide_g=%s idx_dtype=%s)",
        path,
        layer_num,
        head_start,
        head_end,
        H_total,
        NG,
        K,
        G,
        ptn,
        fmt,
        source_fp8_fmt,
        centroid_resnap_rel_rmse,
        wide_g,
        idx_dtype,
    )
    if fold_decode:
        _vq_fold_weights_into_decode_table(out)
    return out


def _vq_fold_weights_into_decode_table(vq: "VQCodebook") -> None:
    """Fold 1/w into a decode-side copy of the codebook, for perm+weight bundles.

    K SIDE ONLY, which is why the caller opts in rather than this keying on perm_rope alone:
    decode unweights K (its RoPE branch needs plain original channels) and never touches w on
    the V side, so folding a V codebook would scale V by 1/w with nothing to undo it. No V
    bundle sets perm_rope today, so the guard below would skip it anyway -- opting in makes
    that a decision rather than a coincidence.

    Decode reconstructs ``k = cb[idx] * s / w`` with w static per (layer, head, stored
    channel), so the division can be pre-applied to the centroids and the KV loop keeps only
    the multiply.

    The encode table is then RE-DERIVED from the folded one. That is the whole point: fp16
    ``cb/w`` is not exactly ``cb``, so an encoder scoring against the original table would
    pick a centroid that is not the centroid decode reconstructs -- the assignment and the
    reconstruction would describe different codebooks. Re-deriving makes both describe the
    same (slightly perturbed, equally valid) set, and cb_sq must be recomputed with it or the
    argmax mixes a new inner product with a stale norm.

    Verified as a quantiser, not just as arithmetic: held-out reconstruction distortion is
    unchanged to four decimals (exp/kernelopt/gate_fold16c.py, worst ratio 1.0000).
    """
    if not getattr(vq, "perm_rope", False) or getattr(vq, "w_perm", None) is None:
        return
    L, H, NG, K, G = vq.cb16.shape
    w5 = vq.w_perm.view(L, H, NG, 1, G)
    dec = (vq.cb16.to(torch.float32) / w5).to(torch.float16).contiguous()
    vq.cb16 = (dec.to(torch.float32) * w5).to(torch.float16).contiguous()
    vq.cb_sq = (0.5 * vq.cb16.to(torch.float32).pow(2).sum(-1)).contiguous()
    # Reorder the DECODE copy only: [e0,o0,e1,o1,...] -> [e0..e3, o0..o3]. The encode table
    # keeps original order, so assignments are bit-identical; the rotation operands then come
    # out of one G-wide load already halved (gate_pmfused.py).
    dec = torch.cat([dec[..., 0::2], dec[..., 1::2]], dim=-1).contiguous()
    vq.cb16_dec = dec
    vq.cb16_dec_pair_major = True
    logger.info(
        "vq2: folded 1/w into the decode codebook: %s %.0f MB",
        tuple(dec.shape), dec.numel() * dec.element_size() / 2**20,
    )


def vq_map_k(
    k: torch.Tensor, forward: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """r = (k - mean) @ forward, per head. k: [T, H, D] (any float dtype).
    Contiguous output: pool writers view the result as [-1, row_dim].

    Dispatches here rather than at the call sites (unlike QMAP) because there
    are two of them -- the decode aging flush and the prefill VQ write -- and a
    gate in each is a drift hazard for no gain.
    """
    from sglang.srt.environ import envs as _envs

    if _envs.SGLANG_VQ_OPT_KMAP.get():
        return vq_map_k_fused(k, forward, mean)
    kd = k.to(forward.dtype)
    return torch.einsum(
        "thd,hde->the", kd - mean.unsqueeze(0), forward
    ).contiguous()


@triton.jit
def _vq_qmap_kernel(
    Q, QMAP, MEAN, OUT,
    n_rows, QH,
    D: tl.constexpr, GRP: tl.constexpr, BLOCK_T: tl.constexpr,
    HAS_MEAN: tl.constexpr, PREC: tl.constexpr,
):
    """out[t, h*GRP+g, :] = (q[t, h*GRP+g, :] - mean[h]) @ q_map[h]

    One program handles BLOCK_T rows of a single KV head, reading and writing
    q in place-strided form. This replaces the view/permute/reshape/bmm/
    permute/reshape/contiguous chain with a single kernel: the profile showed
    the copies, not the GEMM, were the bulk of the cost.

    HAS_MEAN + GRP=1 makes the same kernel serve the K map, whose only
    differences are the centering term and the absent GQA group.
    """
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs_i = pid * BLOCK_T + tl.arange(0, BLOCK_T)      # index into T*GRP
    mask_i = offs_i < n_rows
    t = offs_i // GRP
    g = offs_i % GRP
    qh = h * GRP + g
    offs_d = tl.arange(0, D)

    base = t[:, None] * (QH * D) + qh[:, None] * D + offs_d[None, :]
    qv = tl.load(Q + base, mask=mask_i[:, None], other=0.0)
    if HAS_MEAN:
        qv = qv - tl.load(MEAN + h * D + offs_d)[None, :]
    m = tl.load(QMAP + h * D * D + offs_d[:, None] * D + offs_d[None, :])
    # PREC="ieee" on fp32 inputs: tl.dot would otherwise default to TF32 and
    # make the flag a silent numerics change (rel ~8e-4). fp16/bf16 are
    # bit-identical to the torch path either way.
    o = tl.dot(qv, m, input_precision=PREC)              # fp32 accumulate
    tl.store(OUT + base, o.to(qv.dtype), mask=mask_i[:, None])


def vq_map_q_fused(q: torch.Tensor, q_map: torch.Tensor) -> torch.Tensor:
    """Fused equivalent of vq_map_q (SGLANG_VQ_OPT_QMAP)."""
    T, QH, D = q.shape
    H = q_map.shape[0]
    GRP = QH // H
    qc = q.to(q_map.dtype)
    if not qc.is_contiguous():
        qc = qc.contiguous()
    out = torch.empty_like(qc)
    n_rows = T * GRP
    BLOCK_T = 16 if n_rows >= 16 else 16   # tl.dot needs >= 16 rows; mask covers the tail
    grid = (triton.cdiv(n_rows, BLOCK_T), H)
    _vq_qmap_kernel[grid](
        qc, q_map, qc, out, n_rows, QH,
        D=D, GRP=GRP, BLOCK_T=BLOCK_T, HAS_MEAN=False,
        PREC="ieee" if qc.dtype == torch.float32 else "tf32",
        num_warps=4, num_stages=2,
    )
    return out


def vq_map_k_fused(
    k: torch.Tensor, forward: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """Fused equivalent of vq_map_k (SGLANG_VQ_OPT_KMAP).

    The einsum form runs sub/bmm/contiguous as three kernels over the same
    small tensor; measured under CUDA-graph replay (the serving condition,
    where launch cost is already amortised) that is 2.5-2.9x the fused time
    across n=8..512.
    """
    T, H, D = k.shape
    mean = mean.to(forward.dtype)
    kc = k.to(forward.dtype)
    if not kc.is_contiguous():
        kc = kc.contiguous()
    out = torch.empty_like(kc)
    BLOCK_T = 16                     # tl.dot needs >= 16 rows; mask covers the tail
    grid = (triton.cdiv(T, BLOCK_T), H)
    _vq_qmap_kernel[grid](
        kc, forward, mean, out, T, H,
        D=D, GRP=1, BLOCK_T=BLOCK_T, HAS_MEAN=True,
        PREC="ieee" if kc.dtype == torch.float32 else "tf32",
        num_warps=4, num_stages=2,
    )
    return out


def vq_map_q(q: torch.Tensor, q_map: torch.Tensor) -> torch.Tensor:
    """q_m = q @ inverse.T per KV head, broadcast over the GQA group.

    q: [T, QH, D]; q_map: [H, D, D] with QH % H == 0. Batched-GEMM (bmm) form:
    ~1.5x faster than the einsum at decode (small T) and bit-identical; brings the
    per-head query map to OSCAR's shared-rotation GEMM efficiency.
    """
    T, QH, D = q.shape
    H = q_map.shape[0]
    grp = QH // H
    qd = (
        q.to(q_map.dtype)
        .view(T, H, grp, D)
        .permute(1, 0, 2, 3)
        .reshape(H, T * grp, D)
    )
    out = torch.bmm(qd, q_map).view(H, T, grp, D).permute(1, 0, 2, 3)
    return out.reshape(T, QH, D).contiguous()


def _vq_idx_dtype(K: int) -> torch.dtype:
    """uint8 fits codebook_size<=256 (the G=4 arm); wider codebooks
    (this project's G=8/K=1024 CQ/TaSQ bundles) need int16 -- silently
    wrapping indices past 255 in uint8 is a correctness bug, not an error."""
    return torch.uint8 if K <= 256 else torch.int16


# --------------------------------------------------------------------------- index packing
# A wide-G bundle has 16 groups of 10-bit indices per (token, kv-head). Stored one-per-int16
# that is 2.0 b/coord and six bits of every index are zero; as a little-endian bitstream it is
# 1.25 b/coord, five int32 words per row. The decode kernel unpacks with two gathered word
# loads and shifts -- ALU against what was a 2-byte load, and it buys the KV capacity.
#
# Gated on the GROUP COUNT, never on the index dtype. A previous attempt keyed on dtype, packed
# the int16 K arena and left the uint8 V arena raw, and the kernel -- which unpacks both --
# rejected the pair. K and V must be packed or not packed together.
VQ_PACK_NG = 16


def vq_idx_is_packed(num_groups: int) -> bool:
    return num_groups == VQ_PACK_NG


def vq_idx_bits(codebook_size: int) -> int:
    """Narrowest supported index width holding `codebook_size` codewords.

    Derived from the codebook, not configured, and it must agree with the decode kernel's
    own `_bits_for` -- the launcher asserts the arena width it computes this way, so a pool
    that packed everything at 10 bits made the K<=256 V arena (8 bits, 4 words) fail the
    check with a 5-word buffer.
    """
    for b in (8, 9, 10, 12):
        if codebook_size <= (1 << b):
            return b
    raise ValueError(f"codebook of {codebook_size} entries exceeds the supported widths")


def vq_pack_words(codebook_size: int, num_groups: int = VQ_PACK_NG) -> int:
    return (num_groups * vq_idx_bits(codebook_size) + 31) // 32


@triton.jit
def _vq_pack_kernel(IDX, OUT, n_rows,
                    BITS: tl.constexpr, NG: tl.constexpr, WORDS: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    mask = rows < n_rows
    for w in tl.static_range(0, WORDS):
        acc = tl.zeros([BLOCK], dtype=tl.int64)
        for g in tl.static_range(0, NG):
            bit = g * BITS
            gw = bit // 32
            off = bit % 32
            # a group lands in word w either wholly or, when it straddles, as the high part
            touches = (gw == w) or (gw + 1 == w and off + BITS > 32)
            if touches:
                v = tl.load(IDX + rows * NG + g, mask=mask, other=0).to(tl.int64) & (
                    (1 << BITS) - 1
                )
                if gw == w:
                    acc |= v << off
                else:
                    acc |= v >> (32 - off)
        acc &= 0xFFFFFFFF
        acc = tl.where(acc >= (1 << 31), acc - (1 << 32), acc)
        tl.store(OUT + rows * WORDS + w, acc.to(tl.int32), mask=mask)


def vq_pack_idx(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """[..., 16] integer indices -> [..., words] int32 bitstream at `bits` per index."""
    assert idx.shape[-1] == VQ_PACK_NG, f"packing is specialised to NG=16, got {idx.shape[-1]}"
    words = (VQ_PACK_NG * bits + 31) // 32
    flat = idx.contiguous().reshape(-1, VQ_PACK_NG)
    n = flat.shape[0]
    out = torch.empty((n, words), dtype=torch.int32, device=idx.device)
    if n == 0:
        return out.reshape(*idx.shape[:-1], words)
    if idx.is_cuda:
        BLOCK = 256
        _vq_pack_kernel[(triton.cdiv(n, BLOCK),)](
            flat.to(torch.int32), out, n,
            BITS=bits, NG=VQ_PACK_NG, WORDS=words, BLOCK=BLOCK,
        )
        return out.reshape(*idx.shape[:-1], words)
    return _vq_pack_idx_torch(idx, bits)


def _vq_pack_idx_torch(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """CPU reference for :func:`vq_pack_idx`; same bit layout."""
    words = (VQ_PACK_NG * bits + 31) // 32
    flat = idx.long().reshape(-1, VQ_PACK_NG)
    out = torch.zeros(flat.shape[0], words, dtype=torch.int64, device=idx.device)
    for g in range(VQ_PACK_NG):
        bit = g * bits
        w, off = bit // 32, bit % 32
        v = flat[:, g] & ((1 << bits) - 1)
        out[:, w] |= v << off
        if off + bits > 32:
            out[:, w + 1] |= v >> (32 - off)
    out &= 0xFFFFFFFF
    out = torch.where(out >= (1 << 31), out - (1 << 32), out)
    return out.to(torch.int32).reshape(*idx.shape[:-1], words)


def vq_unpack_idx(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """[..., words] int32 -> [..., 16] int64. Read side for the prefill prefix dequantise;
    decode unpacks inside the kernel instead."""
    words = packed.shape[-1]
    flat = packed.long().reshape(-1, words) & 0xFFFFFFFF
    out = torch.zeros(flat.shape[0], VQ_PACK_NG, dtype=torch.int64, device=packed.device)
    for g in range(VQ_PACK_NG):
        bit = g * bits
        w, off = bit // 32, bit % 32
        v = (flat[:, w] >> off) & ((1 << bits) - 1)
        if off + bits > 32:
            v = (v | (flat[:, w + 1] << (32 - off))) & ((1 << bits) - 1)
        out[:, g] = v
    return out.reshape(*packed.shape[:-1], VQ_PACK_NG)


def vq_encode(
    r: torch.Tensor,
    cb16: torch.Tensor,
    cb_sq: torch.Tensor,
    *,
    pertoken_norm: bool,
    pool_heads: bool = False,
    token_chunk: int = 2048,
):
    """Assign residual rows to nearest centroids.

    r: [T, H, D] (bf16/fp16/fp32); cb16: [H, NG, K, G]; cb_sq: [H, NG, K].
    Returns (idx [T, H, NG] uint8 or int16, scale fp32 [T, H]).
    (K<=256 -> indices fit uint8; wider codebooks use int16, see _vq_idx_dtype.)

    pool_heads=True computes ONE RMS scale per token, pooled over ALL KV heads
    and channels, and uses it for every head instead of the pool's own default
    of an independent scale per (token, head). See VQCodebook.pool_heads_scale.

    Nearest neighbor via argmax(<c, x> - 0.5 ||c||^2); token-chunked so the
    [T, H, NG, K] score tensor stays bounded.
    """
    T, H, D = r.shape
    NG, K, G = cb16.shape[1], cb16.shape[2], cb16.shape[3]
    idx_dtype = _vq_idx_dtype(K)
    rf = r.to(torch.float32)
    if pool_heads:
        scale = rf.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)  # [T,1,1]
        scale = scale.expand(T, H, 1)
        rn = (rf / scale).to(torch.float16)
        scale = scale.squeeze(-1)
    elif pertoken_norm:
        scale = rf.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        rn = (rf / scale).to(torch.float16)
        scale = scale.squeeze(-1)
    else:
        scale = torch.ones((T, H), dtype=torch.float32, device=r.device)
        rn = rf.to(torch.float16)
    rn = rn.view(T, H, NG, G)
    idx = torch.empty((T, H, NG), dtype=idx_dtype, device=r.device)
    for t0 in range(0, T, token_chunk):
        t1 = min(t0 + token_chunk, T)
        scores = torch.einsum("thgc,hgkc->thgk", rn[t0:t1], cb16).to(torch.float32)
        scores -= cb_sq.unsqueeze(0)
        idx[t0:t1] = scores.argmax(-1).to(idx_dtype)
    return idx, scale


@triton.jit
def _vq_encode_kernel(
    R, CB, CBSQ, SCALE, IDX, VALID,
    n_tok,
    stride_r_l, stride_r_n, stride_r_h,
    stride_s_l, stride_s_n,
    stride_i_l, stride_i_n, stride_i_h,
    H,
    NG: tl.constexpr, K: tl.constexpr, G: tl.constexpr, BLOCK_N: tl.constexpr,
    IDX_WIDE: tl.constexpr, HAS_VALID: tl.constexpr,
):
    """Nearest-centroid assign, fused -- never materialises the score tensor.

    grid = (L*H*NG, cdiv(n_tok, BLOCK_N)). Each program owns one
    (layer, head, group) and BLOCK_N tokens: it loads that group's [K, G]
    centroids once, scores the rows against them in registers, and writes the
    argmax straight to the uint8 index arena. The torch path instead builds an
    [L, n, H, NG, K] fp32 tensor (~76 MB at n=64) and makes three passes over
    it to produce ~0.5 MB of indices.

    The per-token RMS scale is precomputed by the caller (a cheap reduction)
    so it is not recomputed redundantly in all NG group-programs.
    """
    lhg = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = lhg % NG
    h = (lhg // NG) % H
    l = lhg // (NG * H)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_tok
    if HAS_VALID:
        # Decode-flush plans are shape-static (bs * flush_interval rows every step,
        # CUDA-graph-safe) but on most steps most rows are INVALID (trash-routed):
        # with synchronized requests, 7 of 8 steps carry zero valid rows, so the
        # encode was doing ~8x redundant work. Rows are request-major, so whole
        # programs are usually uniformly invalid -> exit before touching R or CB.
        vld = tl.load(VALID + offs_n, mask=mask_n, other=0)
        mask_n = mask_n & (vld != 0)
        if tl.max(vld, axis=0) == 0:
            return
    offs_g = tl.arange(0, G)

    x = tl.load(
        R + l * stride_r_l + offs_n[:, None] * stride_r_n + h * stride_r_h
        + (g * G + offs_g)[None, :],
        mask=mask_n[:, None], other=0.0,
    ).to(tl.float32)
    sc = tl.load(SCALE + l * stride_s_l + offs_n * stride_s_n + h,
                 mask=mask_n, other=1.0).to(tl.float32)
    x = x / sc[:, None]

    # K-chunked scoring: the original single-shot form materialised a
    # [BLOCK_N, K, G] broadcast intermediate, which at the g8 geometry
    # (K=1024, G=8) is 8x the K=256/G=4 design point and spills to local
    # memory -- profiled at 88% of ALL decode GPU time (152 ms/call) and the
    # cause of the catastrophic CQ/TaSQ prefill on small-L2 cards. Chunking
    # bounds the intermediate at [BLOCK_N, BK, G] while keeping every score's
    # arithmetic (and the first-max tie-break) bit-identical: scores are
    # computed with the same ops per entry, and the cross-chunk update uses
    # strict >, so the earliest index still wins ties exactly like a full
    # tl.argmax.
    BK: tl.constexpr = 128 if K > 128 else K
    best_val = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for k0 in tl.static_range(0, K, BK):
        offs_kc = k0 + tl.arange(0, BK)
        cbp = (CB + ((l * H + h) * NG + g) * (K * G)
               + offs_kc[:, None] * G + offs_g[None, :])
        c = tl.load(cbp).to(tl.float32)                              # [BK, G]
        csq = tl.load(CBSQ + ((l * H + h) * NG + g) * K + offs_kc).to(tl.float32)
        scores = tl.sum(x[:, None, :] * c[None, :, :], axis=2) - csq[None, :]
        loc_val = tl.max(scores, axis=1)
        loc_arg = tl.argmax(scores, axis=1).to(tl.int32) + k0
        upd = loc_val > best_val
        best_idx = tl.where(upd, loc_arg, best_idx)
        best_val = tl.where(upd, loc_val, best_val)
    best = best_idx
    if IDX_WIDE:
        store_val = best[:, None].to(tl.int16)
    else:
        store_val = best[:, None].to(tl.uint8)
    tl.store(
        IDX + l * stride_i_l + offs_n[:, None] * stride_i_n
        + h * stride_i_h + g,
        store_val, mask=mask_n[:, None],
    )


@triton.jit
def _vq_encode_dot_kernel(
    R, CB, CBSQ, SCALE, IDX, VALID,
    n_tok,
    stride_r_l, stride_r_n, stride_r_h,
    stride_s_l, stride_s_n,
    stride_i_l, stride_i_n, stride_i_h,
    H,
    NG: tl.constexpr, K: tl.constexpr, GP: tl.constexpr, G_REAL: tl.constexpr,
    BLOCK_N: tl.constexpr, BK: tl.constexpr,
    IDX_WIDE: tl.constexpr, HAS_VALID: tl.constexpr, HAS_SCALE: tl.constexpr,
):
    """Same assignment as :func:`_vq_encode_kernel`, scored as a matmul.

    The broadcast form ``tl.sum(x[:, None, :] * c[None, :, :], axis=2)`` never reaches the
    tensor cores and materialises a [BLOCK_N, BK, G] fp32 intermediate, which is why raising
    BLOCK_N made it slower (16 -> 1.90 ms, 32 -> 3.97, 64 -> 20.09): the intermediate grows
    with BLOCK_N and spills. As ``x @ c.T`` the intermediate is [BLOCK_N, BK], 8x smaller, so
    BK can cover the whole codebook in one chunk.

    tl.dot needs a contraction of at least 16 and G is 8, so CB is zero-padded to GP=16 on the
    host and x's padded lanes come from the load mask (other=0.0) -- no staging tensor. Padding
    contributes exactly zero to every score.

    NOT bit-identical to the broadcast form: tl.dot sums the padded contraction in the MMA's
    order, so near-ties can resolve to the other equally-valid argmax. Measured at 5
    disagreements in 6.3M assignments, every one between codewords whose float64 scores differ
    by <1.4e-07 (below fp32 epsilon). See exp/kernelopt/gate_encode.py.
    """
    pid0 = tl.program_id(0)
    g = pid0 % NG
    h = (pid0 // NG) % H
    l = pid0 // (NG * H)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_tok
    if HAS_VALID:
        vld = tl.load(VALID + offs_n, mask=mask_n, other=0)
        mask_n = mask_n & (vld != 0)
        if tl.max(vld, axis=0) == 0:
            return
    offs_g = tl.arange(0, GP)
    gmask = offs_g < G_REAL
    x = tl.load(
        R + l * stride_r_l + offs_n[:, None] * stride_r_n + h * stride_r_h
        + g * G_REAL + offs_g[None, :],
        mask=mask_n[:, None] & gmask[None, :], other=0.0,
    ).to(tl.float32)
    # HAS_SCALE=False: the bundle has neither pertoken_norm nor pool_heads_scale (CQ's K and V,
    # TaSQ's V), so the caller's scale is all ones and `x / 1.0` is a per-element divide plus a
    # load that compute nothing. decode_attention._is_const_one already elides exactly this on
    # the decode side; this is the encode side of the same thing.
    if HAS_SCALE:
        sc = tl.load(SCALE + l * stride_s_l + offs_n * stride_s_n + h,
                     mask=mask_n, other=1.0).to(tl.float32)
        x = x / sc[:, None]
    best_val = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for k0 in tl.static_range(0, K, BK):
        offs_kc = k0 + tl.arange(0, BK)
        c = tl.load(CB + ((l * H + h) * NG + g) * (K * GP)
                    + offs_kc[:, None] * GP + offs_g[None, :]).to(tl.float32)
        csq = tl.load(CBSQ + ((l * H + h) * NG + g) * K + offs_kc).to(tl.float32)
        # allow_tf32=False: the operands are fp32 here and TF32's 10-bit mantissa would move
        # far more assignments than the MMA reordering does.
        scores = tl.dot(x, tl.trans(c), allow_tf32=False) - csq[None, :]
        loc_val = tl.max(scores, axis=1)
        loc_arg = tl.argmax(scores, axis=1).to(tl.int32) + k0
        upd = loc_val > best_val
        best_idx = tl.where(upd, loc_arg, best_idx)
        best_val = tl.where(upd, loc_val, best_val)
    if IDX_WIDE:
        store_val = best_idx[:, None].to(tl.int16)
    else:
        store_val = best_idx[:, None].to(tl.uint8)
    tl.store(
        IDX + l * stride_i_l + offs_n[:, None] * stride_i_n + h * stride_i_h + g,
        store_val, mask=mask_n[:, None],
    )


_VQ_PAD_CACHE: dict = {}


def _padded_codebook(cb16: torch.Tensor) -> torch.Tensor:
    """[L,H,NG,K,G] -> [L,H,NG,K,16] fp16, zero-padded, cached per source tensor.

    fp16 rather than fp32: same indices, same speed, half the residency (~134 MB for a
    32-layer 8-kv-head K=1024 G=8 bundle instead of 268 MB).
    """
    key = (cb16.data_ptr(), tuple(cb16.shape))
    p = _VQ_PAD_CACHE.get(key)
    if p is None:
        L, H, NG, K, G = cb16.shape
        p = torch.zeros((L, H, NG, K, 16), dtype=torch.float16, device=cb16.device)
        p[..., :G] = cb16.to(torch.float16)
        _VQ_PAD_CACHE[key] = p
    return p


def vq_encode_fused(r, cb16, cb_sq, *, pertoken_norm: bool, pool_heads: bool = False,
                    valid: "torch.Tensor | None" = None):
    """Fused flush encode (SGLANG_VQ_OPT_FLUSH).

    r: [L, n, H, D]; cb16: [L, H, NG, K, G]; cb_sq: [L, H, NG, K].
    Returns (idx uint8 [L, n, H, NG], scale fp32 [L, n, H]) -- same contract as
    the torch path in vq_flush_k. pool_heads: see vq_encode.

    Scores through :func:`_vq_encode_dot_kernel` (tensor cores) when the geometry allows;
    ``SGLANG_VQ_ENCODE_DOT=0`` forces the original broadcast kernel. The index dtype contract
    is preserved either way -- returning int16 unconditionally would break the uint8 arenas of
    K<=256 bundles, which is the shape the V side uses.
    """
    L, n, H, D = r.shape
    NG, K, G = cb16.shape[2], cb16.shape[3], cb16.shape[4]
    idx_dtype = _vq_idx_dtype(K)
    rc = r.contiguous()
    rf = rc.to(torch.float32)
    if G == 8 and os.environ.get("SGLANG_VQ_ENCODE_DOT", "1") == "1":
        if pool_heads:
            scale = rf.pow(2).mean(dim=(2, 3)).sqrt().clamp_min(1e-8)
            scale = scale.unsqueeze(-1).expand(L, n, H).contiguous()
        elif pertoken_norm:
            scale = rf.pow(2).mean(-1).sqrt().clamp_min(1e-8).contiguous()
        else:
            # stride-0 view, not a materialised buffer: with HAS_SCALE=False the kernel never
            # reads it, and callers only need the documented [L, n, H] shape back.
            scale = torch.ones((), dtype=torch.float32, device=r.device).expand(L, n, H)
        has_scale = bool(pool_heads or pertoken_norm)
        idx = torch.empty((L, n, H, NG), dtype=idx_dtype, device=r.device)
        valid_i8 = (valid.to(torch.int8).contiguous() if valid is not None
                    else rf.new_zeros(1).to(torch.int8))
        BLOCK_N = 32
        grid = (L * H * NG, triton.cdiv(n, BLOCK_N))
        _vq_encode_dot_kernel[grid](
            rf, _padded_codebook(cb16), cb_sq.contiguous(), scale, idx, valid_i8,
            n,
            rf.stride(0), rf.stride(1), rf.stride(2),
            scale.stride(0), scale.stride(1),
            idx.stride(0), idx.stride(1), idx.stride(2),
            H, NG=NG, K=K, GP=16, G_REAL=G, BLOCK_N=BLOCK_N, BK=K,
            IDX_WIDE=(idx_dtype != torch.uint8),
            HAS_VALID=(valid is not None),
            HAS_SCALE=has_scale,
            num_warps=8, num_stages=2,
        )
        return idx, scale
    if pool_heads:
        scale = rf.pow(2).mean(dim=(2, 3)).sqrt().clamp_min(1e-8)     # [L, n]
        scale = scale.unsqueeze(-1).expand(L, n, H).contiguous()
    elif pertoken_norm:
        scale = rf.pow(2).mean(-1).sqrt().clamp_min(1e-8)             # [L, n, H]
    else:
        scale = torch.ones((L, n, H), dtype=torch.float32, device=r.device)
    idx = torch.empty((L, n, H, NG), dtype=idx_dtype, device=r.device)
    cbc, cbsqc = cb16.contiguous(), cb_sq.contiguous()
    BLOCK_N = 16
    grid = (L * H * NG, triton.cdiv(n, BLOCK_N))
    valid_i8 = valid.to(torch.int8).contiguous() if valid is not None else rf.new_zeros(1).to(torch.int8)
    _vq_encode_kernel[grid](
        rf, cbc, cbsqc, scale, idx, valid_i8,
        n,
        rf.stride(0), rf.stride(1), rf.stride(2),
        scale.stride(0), scale.stride(1),
        idx.stride(0), idx.stride(1), idx.stride(2),
        H, NG=NG, K=K, G=G, BLOCK_N=BLOCK_N,
        IDX_WIDE=(idx_dtype != torch.uint8),
        HAS_VALID=(valid is not None),
        num_warps=4, num_stages=2,
    )
    return idx, scale


def vq_encode_single(r, cb16_l, cb_sq_l, *, pertoken_norm, pool_heads: bool = False):
    """Single-layer adapter for :func:`vq_encode_fused`.

    Drop-in for ``vq_encode`` at the prefill/extend write sites, which work one
    layer at a time: r [T, H, D], cb16_l [H, NG, K, G], cb_sq_l [H, NG, K].
    The unfused path materialises a [chunk, H, NG, K] fp32 score tensor (537 MB
    at the 2048-token chunk) and makes three passes over it; the fused kernel
    keeps the reduction in registers.
    """
    idx, scale = vq_encode_fused(
        r.unsqueeze(0),
        cb16_l.unsqueeze(0),
        cb_sq_l.unsqueeze(0),
        pertoken_norm=pertoken_norm,
        pool_heads=pool_heads,
    )
    return idx.squeeze(0), scale.squeeze(0)


def vq_dequant(
    idx: torch.Tensor, scale: torch.Tensor, cb16: torch.Tensor
) -> torch.Tensor:
    """Reconstruct residual rows: idx [T, H, NG] uint8, scale [T, H],
    cb16 [H, NG, K, G] -> r_hat [T, H, NG*G] fp32."""
    T, H, NG = idx.shape
    G = cb16.shape[-1]
    h_ids = torch.arange(H, device=idx.device).view(1, H, 1)
    g_ids = torch.arange(NG, device=idx.device).view(1, 1, NG)
    cw = cb16[h_ids, g_ids, idx.long()]  # [T, H, NG, G]
    r_hat = cw.to(torch.float32).view(T, H, NG * G)
    return r_hat * scale.to(torch.float32).unsqueeze(-1)
