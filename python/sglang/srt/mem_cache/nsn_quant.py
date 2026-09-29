"""NSNQuant baseline integration for SGLang.

The implementation follows the reference Normalize-Shift-Normalize transform over fixed token
windows. It supports a BF16 round-trip path and a packed path selected by
``SGLANG_NSN_PACKED=1``. Value-side Hadamard transforms may be applied online or folded into the
projection weights. State is keyed by request slots, so serving uses ``--disable-radix-cache``.
The module is inactive unless ``SGLANG_NSN_PATH`` is set.
"""
from __future__ import annotations

import math
import os

import torch

_V_HADAMARD_FOLDED = os.environ.get("SGLANG_NSN_FOLD_V_HADAMARD") == "1"

# SGLANG_NSN_PACKED=1: also write the PACKED form to nsn_packed_store, so the served arm can
# hold 1.2383 b/ch instead of a bf16 reconstruction. Gated; unset and nothing below runs.
from sglang.srt.mem_cache import nsn_packed_store as _PS
from sglang.srt.mem_cache import nsn_hp_pool as _HPP

# High-precision recent tail, matching the OTHER methods' residual policy (CQ/TaSQ/Nova all run
# with PREFIX_TOKENS=0 RECENT_TOKENS=64 via serve_method.sh -> SGLANG_MIXED_KV_RECENT_TOKENS):
# the most recent RECENT tokens always stay raw; a 64-token NSN window is flushed only once ALL
# of its tokens have aged out of that recent region (buffer >= window_size + RECENT -> flush the
# oldest window_size). The raw tail therefore oscillates in [RECENT, RECENT + window_size), vs
# the other methods' [RECENT, RECENT + 8) -- NSN's window-joint mean/norm statistics force
# window_size flush granularity. Set to 0 for the reference NSNQuant behavior (flush the moment
# a window fills; raw tail [0, window_size)).
_RECENT_TOKENS = int(
    os.environ.get(
        "SGLANG_NSN_RECENT_TOKENS",
        os.environ.get("SGLANG_MIXED_KV_RECENT_TOKENS", "64"),
    )
)

# Rows evicted from the raw tail per flush. Eight matches the eviction granularity used by the
# other quantized methods.
#
# A window still cannot be TRANSFORMED before it is complete, but that never binds: evicting row
# t needs t <= n - RECENT, and t's window ends by t + ws - 1 < n whenever RECENT >= ws, which both
# served configurations (256 and 64) satisfy.
#
# A value of 0 selects whole-window eviction.
_EVICT_GRAN_ENV = int(os.environ.get("SGLANG_NSN_EVICT_GRAN", "8"))


def _gran(ws: int) -> int:
    return _EVICT_GRAN_ENV if _EVICT_GRAN_ENV > 0 else ws

# Attention-sink prefix, matching the other methods' PREFIX_TOKENS (their HP-prefix pool):
# the first PREFIX tokens of every request stay raw forever -- they never enter the window
# accumulator. Same semantics as the reference NSNQuant's KV_SINK. Default 0 (no sink),
# matching the standing sweep config.
_PREFIX_TOKENS = int(
    os.environ.get(
        "SGLANG_NSN_PREFIX_TOKENS",
        os.environ.get("SGLANG_MIXED_KV_PREFIX_TOKENS", "0"),
    )
)

_BUNDLES: dict = {}
_ACCUM: dict = {}  # (path, layer_idx, req_pool_idx) -> dict(k=[T,H,D] raw buffer, v=..., locs=[T])


def load_bundle(path: str, device) -> dict:
    if path not in _BUNDLES:
        b = torch.load(path, map_location="cpu", weights_only=False)
        assert b.get("codec") == "nsn", f"not an NSN bundle: {path}"
        loaded = {
            "n_bits": int(b["n_bits"]),
            "head_dim": int(b["head_dim"]),
            "window_size": int(b["window_size"]),
            "codebook": b["codebook"].to(device, dtype=torch.float16),  # [256, 8], already 4-bit-RTN-degraded
            "inv_freq": b["inv_freq"].to(device).float(),  # [head_dim/2]
        }
        # Bundles without an explicit declaration use the legacy float32 metadata format.
        want = {torch.float16: "float16", torch.bfloat16: "bfloat16", None: "float32"}[
            NORM2_STORE_DTYPE
        ]
        got = b.get("norm2_store_dtype", "float32")
        assert got == want, (
            f"bundle stores norm2 as {got} but nsn_quant.NORM2_STORE_DTYPE is {want}: the two "
            f"would reconstruct different values ({path})"
        )
        print(
            f"[nsn_quant] ACTIVE: 1-bit NSNQuant, window_size={loaded['window_size']} "
            f"head_dim={loaded['head_dim']} from {path} "
            f"(simulated: dequantized values stored as BF16, no memory saving)",
            flush=True,
        )
        _BUNDLES[path] = loaded
    return _BUNDLES[path]


def _hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(n)


_HAD_CACHE: dict = {}


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1]
    key = (D, x.device, x.dtype)
    if key not in _HAD_CACHE and torch.cuda.is_current_stream_capturing():
        # A Hadamard matrix first built inside CUDA-graph capture lives in that graph's
        # private pool and reads back as ZEROS from any eager call, which silently turns the
        # NSN encode into an all-zero window (2026-09-10). Prewarm every dtype instead.
        raise RuntimeError(
            f"_hadamard: first allocation for {key} inside CUDA-graph capture; build it "
            f"before capture (see memory_pool's NSN prewarm)."
        )
    if key not in _HAD_CACHE:
        _HAD_CACHE[key] = _hadamard_matrix(D, x.device, x.dtype)
    return x @ _HAD_CACHE[key].T


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(v, cos, sin):
    return v * cos + _rotate_half(v) * sin


def _rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor):
    freqs = positions.float()[..., None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rtn4_roundtrip(w: torch.Tensor, group_size: int) -> torch.Tensor:
    shape = w.shape
    flat = w.reshape(-1, group_size)
    w_max = flat.amax(dim=-1, keepdim=True)
    w_min = flat.amin(dim=-1, keepdim=True)
    scale = torch.clamp((w_max - w_min) / 15.0, min=1e-5)
    q = ((flat - w_min) / scale).clamp_(0, 15).round_()
    return (q * scale + w_min).reshape(shape)


def _nsn_transform(x: torch.Tensor, window_size: int):
    """x: [..., ws, D] -- [H, ws, D] for one window, [B, H, ws, D] for a batch of them.

    Shape-generic so the fused path can hand in a whole batch instead of looping window by
    window. Nothing here mixes rows across the leading axes: the norm reduction is over D, the
    mean over ws, and both RTN groups (ws for norm, 32 for mean) fall inside a single (.., H)
    window, so a batched call computes each window exactly as a lone call would.
    """
    *lead, ws, D = x.shape
    x_norm = x.norm(p=2, dim=-1, keepdim=True) / math.sqrt(D)
    # [..., ws, 1] -> [..., 1, ws] so the RTN group spans the window's own norms
    x_norm = _rtn4_roundtrip(x_norm.reshape(*lead, 1, ws), ws).reshape(*lead, ws, 1)
    x = x / x_norm

    x_mean = x.mean(dim=-2, keepdim=True)  # [H, 1, D]
    x_mean = _rtn4_roundtrip(x_mean, 32)
    x = x - x_mean

    x_norm2 = x.norm(p=2, dim=-1, keepdim=True) / math.sqrt(D)
    x = x / x_norm2
    return x, x_norm, x_mean, x_norm2


# The fused search uses argmin_c ||x-c||^2 == argmax_c (x.c - ||c||^2/2) and avoids
# materializing the full distance matrix.
_FAST_VQ = os.environ.get("SGLANG_NSN_FAST_VQ", "1") == "1"

# see the note at its use: one device->host copy per step, not per layer
_TOLIST_CACHE: dict = {}

# SGLANG_NSN_BATCH_ENCODE=1 (default): batch the packed window encodes of all requests due in
# one layer into a single call. See the note at the collection site.
_BATCH_ENC = os.environ.get("SGLANG_NSN_BATCH_ENCODE", "1") == "1"
_PEND: list = []
_COMMITQ: list = []

# SGLANG_NSN_COMPILE_WINDOW=1 (default): torch.compile the window encode. At the shape the
# server actually flushes -- ONE request's [8, 64, 128] window -- the encode is ~20 tiny torch
# ops and pure launch overhead: 1.68 ms for 65K elements, 47 ms/step across 28 layers. Compiled
# it is 0.73 ms (2.3x).
#
# Compilation may reassociate reductions, so this path is numerically equivalent but not
# necessarily bit-identical to the eager reference.
#
# Pad batches to a power of two to limit the number of compiled shapes.
_COMPILE_WIN = os.environ.get("SGLANG_NSN_COMPILE_WINDOW", "1") == "1"
# gather windows from the bf16 tier instead of keeping per-token copies (packed path only)
_HP_ACC_ENV = os.environ.get("SGLANG_NSN_HP_ACCUM", "1") == "1"
_CWIN: dict = {}


def _compiled_encode(nb):
    fn = _CWIN.get(nb)
    if fn is None:
        from sglang.srt.mem_cache import nsn_pack as _NP
        fn = (torch.compile(_NP.encode_window_k, dynamic=False),
              torch.compile(_NP.encode_window_v, dynamic=False))
        _CWIN[nb] = fn
    return fn


class _Slice:
    """One request's view of a batched PackedWindow, for commit_meta / commit_rows."""

    def __init__(self, p, i, nb):
        # `_rtn4_encode` returns its scale/min flattened over ALL leading axes (it never
        # reshapes them back), so a batched encode gives [B*H, 1] and [B*H*D/G, 1]. Fold the
        # batch axis back out before indexing, or slot i reads one scalar instead of a head's
        # worth -- which is exactly the "shape '[8]' is invalid for input of size 1" this hit.
        self.idx = p.idx[i]
        self.norm2 = p.norm2[i]
        self.norm_q = p.norm_q[i]
        self.norm_scale = p.norm_scale.reshape(nb, -1, 1)[i]
        self.norm_min = p.norm_min.reshape(nb, -1, 1)[i]
        self.mean_q = p.mean_q[i]
        self.mean_scale = p.mean_scale.reshape(nb, -1, 1)[i]
        self.mean_min = p.mean_min.reshape(nb, -1, 1)[i]


def _drain_pending(b, ws):
    """Encode every window collected for this layer in one call, then commit each."""
    from sglang.srt.mem_cache import nsn_pack as _NP

    pend, _PEND[:] = list(_PEND), []
    n = len(pend)
    nb = 1 << (n - 1).bit_length() if n > 1 else 1      # pad to a power of two: few graphs
    wk = torch.stack([e["win_k"] for e in pend] + [pend[-1]["win_k"]] * (nb - n))
    wv = torch.stack([e["win_v"] for e in pend] + [pend[-1]["win_v"]] * (nb - n))
    cos = torch.stack([e["cos"] for e in pend] + [pend[-1]["cos"]] * (nb - n))
    sin = torch.stack([e["sin"] for e in pend] + [pend[-1]["sin"]] * (nb - n))
    if _COMPILE_WIN:
        ek, evf = _compiled_encode(nb)
        pk = ek(wk, cos, sin, b["codebook"], ws)
        pv = evf(wv, b["codebook"], ws, _V_HADAMARD_FOLDED)
    else:
        pk = _NP.encode_window_k(wk, cos, sin, b["codebook"], ws)
        pv = _NP.encode_window_v(wv, b["codebook"], ws, _V_HADAMARD_FOLDED)
    for i, e in enumerate(pend):
        st, ls, ps = e["st"], e["ls"], e["ps"]
        st["pk"], st["pv"] = _Slice(pk, i, nb), _Slice(pv, i, nb)
        w_i = st.get("win", 0)
        if e["layer_idx"] == 0:
            st["wid"] = ps.alloc_window(e["r"])
        else:
            st["wid"] = ps.wins[e["r"]][w_i]
        ls.commit_meta(st["pk"], st["pv"], st["wid"])
        sl = e["sl"]
        _COMMITQ.append((ls, st["pk"], st["pv"], e["win_loc"][sl],
                         torch.arange(sl.start, sl.stop, device=e["win_loc"].device)))


def _vq_nearest(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    if _FAST_VQ:
        from sglang.srt.mem_cache.nsn_vq_fast import vq_nearest_fast

        return vq_nearest_fast(x, codebook, ieee=True)
    # The distance computation itself can stay fp32 for numerical stability (doesn't change
    # which index is nearest); what actually mattered was that `codebook` itself now holds the
    # reference's real 4-bit-RTN-degraded values (applied at bundle-build time), not the raw
    # undegraded codebook -- that was the actual source of the port's higher-resolution VQ.
    shape = x.shape
    idx = torch.cdist(x.reshape(-1, 8).float(), codebook.float()).argmin(dim=-1)
    return codebook[idx].reshape(shape)


NORM2_STORE_DTYPE = torch.float16  # set to None to retain float32 metadata


def _scale_adjust(nsn_res, vq_val, norm2):
    num = (nsn_res * nsn_res).sum(dim=-1, keepdim=True)
    den = (nsn_res * vq_val).sum(dim=-1, keepdim=True)
    out = norm2 * (num / den.clamp_min(1e-12))
    if NORM2_STORE_DTYPE is None:
        return out
    # Rounded to the storage precision, but the return dtype is unchanged so downstream ops keep
    # their accumulation width (returning fp16 would pull the 128-dim Hadamard into fp16 too).
    return out.to(NORM2_STORE_DTYPE).to(out.dtype)


@torch.no_grad()
def _transform_window_k(k_postrope, cos, sin, codebook, window_size):
    """k_postrope: [..., H, ws, D]; cos/sin: [..., ws, D]. Returns reconstructed K, same shape.

    unsqueeze(-3), not unsqueeze(0): inserts the head axis whether cos arrives as [ws, D] (one
    window) or [B, ws, D] (a batch), so both call shapes broadcast the same way.
    """
    cos_h, sin_h = cos.unsqueeze(-3), sin.unsqueeze(-3)
    k_prerope = _apply_rope(k_postrope, cos_h, -sin_h)
    x, norm, mean, norm2 = _nsn_transform(k_prerope, window_size)
    x = _apply_rope(x, cos_h, sin_h)
    x_had = _hadamard(x)

    vq_val = _vq_nearest(x_had, codebook)
    norm2 = _scale_adjust(x_had, vq_val, norm2)

    mean_full = mean.expand(*mean.shape[:-2], window_size, mean.shape[-1])
    mean_roped = _apply_rope(mean_full, cos_h, sin_h)

    recon_had = vq_val * norm2
    recon_roped_nohad = _hadamard(recon_had)
    return (recon_roped_nohad + mean_roped) * norm


@torch.no_grad()
def _transform_window_v(v, codebook, window_size, v_hadamard_folded=False):
    """v: [H, ws, D]. Returns reconstructed V, same shape/space.

    If v_hadamard_folded is True, the caller (nsn_v_fold.fold_v_hadamard_, applied to
    qkv_proj/o_proj at model-load time) has already rotated V into Hadamard space and will
    un-rotate via the folded o_proj, matching the reference ``rotate_v_proj`` and
    ``rotate_o_proj`` deployment. In that case ``v`` arrives in Hadamard space, so this function
    does not apply an explicit Hadamard transform; the result remains in the same space.
    """
    x, norm, mean, norm2 = _nsn_transform(v, window_size)
    x_had = x if v_hadamard_folded else _hadamard(x)

    vq_val = _vq_nearest(x_had, codebook)
    norm2 = _scale_adjust(x_had, vq_val, norm2)

    recon_had = vq_val * norm2
    recon_nohad = recon_had if v_hadamard_folded else _hadamard(recon_had)
    mean_full = mean.expand(*mean.shape[:-2], window_size, mean.shape[-1])
    return (recon_nohad + mean_full) * norm


@torch.no_grad()
def apply_nsn_quant(
    path: str,
    layer_idx: int,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    positions: torch.Tensor,
    req_pool_indices: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
):
    """cache_k/cache_v: [T, H, D] new tokens this call (T = batch decode size, or prefill chunk
    length for a single request). loc/positions/req_pool_indices: [T]. k_buffer/v_buffer: this
    layer's full pool storage tensors (same dtype as cache_k/cache_v), indexable by `loc` --
    passed in so this function can retroactively overwrite already-stored raw rows belonging to
    the SAME window once it completes (those rows were written as-is on an earlier call, before
    enough tokens had accumulated to run the transform).

    Returns (k, v) for the CURRENT batch of T tokens only, to be written normally by the existing
    `_set_kv_buffer_impl` call right after this -- exactly `sim_quant.py`'s contract. Any
    retroactive rewrite of OLDER tokens is done here, in place, on `k_buffer`/`v_buffer` directly.
    """
    b = load_bundle(path, cache_k.device)
    ws = b["window_size"]
    D = b["head_dim"]
    assert cache_k.shape[-1] == D
    # All three index tensors must be PER-TOKEN and aligned with cache_k's rows. SGLang's
    # forward_batch.req_pool_indices is per-REQUEST [bs]; the caller must expand it (see
    # triton_backend.py's repeat_interleave over extend_seq_lens). Passing it unexpanded made
    # every extend call silently accumulate only its first bs rows -- the bug that left ~92% of
    # KV uncompressed across all pre-2026-08-22 NSN sweeps.
    assert req_pool_indices.numel() == positions.numel() == loc.numel() == cache_k.shape[0], (
        f"per-token index tensors misaligned: req_pool_indices={req_pool_indices.numel()} "
        f"positions={positions.numel()} loc={loc.numel()} tokens={cache_k.shape[0]}"
    )

    T, H, _ = cache_k.shape
    out_k = cache_k.clone()
    out_v = cache_v.clone()

    # Three device->host copies, and this hook runs once per LAYER: 84 syncs per decode step
    # on a 28-layer model, each draining the GPU queue. The three tensors are identical for
    # every layer of a step, so only the first layer pays. Layer 0 always runs first in a
    # forward pass, which is what makes the refresh point safe; the shape guard catches any
    # path where that assumption does not hold and falls back to copying.
    _hp = _HPP.hp()
    _HP_ACC = _HP_ACC_ENV and _hp is not None
    _c = _TOLIST_CACHE
    if layer_idx == 0 or _c.get("n") != (req_pool_indices.shape[0], loc.shape[0]):
        _c["n"] = (req_pool_indices.shape[0], loc.shape[0])
        _c["req"] = req_pool_indices.tolist()
        _c["pos"] = positions.tolist()
        _c["loc"] = loc.tolist()
    req_ids, pos_list, loc_list = _c["req"], _c["pos"], _c["loc"]

    # Group this call's rows by request (usually 1 row/request at decode, many at prefill).
    by_req: dict = {}
    for i, r in enumerate(req_ids):
        by_req.setdefault(r, []).append(i)

    for r, idxs in by_req.items():
        if _PREFIX_TOKENS > 0:
            # Sink prefix: rows at positions < PREFIX pass through raw (out_k/out_v are
            # already unmodified copies) and are permanently excluded from windowing.
            idxs = [i for i in idxs if pos_list[i] >= _PREFIX_TOKENS]
            if not idxs:
                continue
        key = (path, layer_idx, r)
        st = _ACCUM.get(key)
        # Request slots are reused. Reset on any positional discontinuity so buffered rows from
        # a previous request or server warmup cannot enter the next window's statistics.
        expected_next = st["pos"][-1] + 1 if st and st["pos"] else 0
        if st is None or pos_list[idxs[0]] != expected_next:
            st = {"k": [], "v": [], "loc": [], "pos": [], "ev": 0, "win": 0}
            _ACCUM[key] = st
            if _PS.ENABLED and layer_idx == 0:
                # The slot is being reused by a different sequence: its committed prefix is
                # gone, so the read path must stop attending to it and its windows go back.
                _PS.store().release(r)

        for i in idxs:
            # Clone eager-path rows so views do not retain the full input batch. The packed path
            # instead gathers raw rows from the BF16 tier when the window is flushed.
            if not _HP_ACC:
                st["k"].append(cache_k[i].clone())
                st["v"].append(cache_v[i].clone())
            st["loc"].append(loc_list[i])
            st["pos"].append(pos_list[i])
        st.setdefault("ev", 0)
        # With the shrunk bf16 tier, PREFILL defers flushing: the stock extend kernel reads the
        # whole span out of that tier, so every row of the prompt must still hold an hp row.
        # A prefill call carries many tokens for one request; a decode call carries one. The
        # backlog drains on the first decode steps, keeping prefill attention exact.
        _defer = _HPP.hp() is not None and len(idxs) > 1
        while (not _defer) and len(st["loc"]) >= _RECENT_TOKENS + st["ev"] + _gran(ws):
            ev = st["ev"]
            # The whole-WINDOW transform runs on every `gran`-row eviction, so a 64-row window
            # is transformed 64/gran = 8 times to commit 8 rows at a time -- the flush
            # redundancy. With the packed store the reconstruction is never read, so encode the
            # window once and reuse it for subsequent evictions.
            # Here the window is touched ONCE, at ev==0, and later evictions only scatter the
            # already-encoded rows.
            _packed_only = _hp is not None
            win_loc = torch.tensor(st["loc"][:ws], device=cache_k.device, dtype=torch.long)
            if (not _packed_only) or ev == 0:
                if _HP_ACC:
                    # gather the window's raw rows straight out of the bf16 tier
                    _rows = _hp.lookup(win_loc).long()
                    win_k = k_buffer[_rows].transpose(0, 1)       # [H, ws, D]
                    win_v = v_buffer[_rows].transpose(0, 1)
                else:
                    win_k = torch.stack(st["k"][:ws], dim=0).transpose(0, 1)
                    win_v = torch.stack(st["v"][:ws], dim=0).transpose(0, 1)
                win_pos = torch.tensor(st["pos"][:ws], device=cache_k.device,
                                       dtype=torch.long)
                cos, sin = _rope_cos_sin(win_pos, b["inv_freq"])
                cos, sin = cos.to(cache_k.dtype), sin.to(cache_k.dtype)

            if not _packed_only:
                recon_k = _transform_window_k(win_k, cos, sin, b["codebook"],
                                              ws).to(cache_k.dtype)
                recon_v = _transform_window_v(win_v, b["codebook"], ws,
                                              _V_HADAMARD_FOLDED).to(cache_v.dtype)
                recon_k = recon_k.transpose(0, 1)  # [ws, H, D]
                recon_v = recon_v.transpose(0, 1)

            # Overwrite the pool directly for every row of this window (most of these rows were
            # written raw on earlier calls; the last `len(idxs)` rows overlap the CURRENT call's
            # `loc`s, which are also covered by the `out_k`/`out_v` return below -- both paths end
            # up writing the identical reconstructed value to those slots, so this is redundant
            # but not incorrect for the overlap).
            # Only the rows leaving the raw tail NOW are written; the rest of the window stays
            # raw in the pool and in st["k"], and is written by later iterations. Slicing the
            # WHOLE-window transform is what keeps each row bit-identical to the old behaviour --
            # the values do not depend on how many of them are committed at a time. The window is
            # re-transformed once per eviction rather than cached: this is the eager oracle, and
            # recomputing from rows that are still raw is simpler than holding the packed form
            # (which is what the fused path must do, since it reads its raw rows back from the
            # pool and would otherwise re-read its own output).
            gran = _gran(ws)
            sl = slice(ev, ev + gran)
            if _hp is None and (_PS.KEEP_BF16 or not _PS.ENABLED):
                k_buffer.index_copy_(0, win_loc[sl], recon_k[sl])
                v_buffer.index_copy_(0, win_loc[sl], recon_v[sl])
            # No release: the hp tier is a per-request ring indexed by position, so a row is
            # reused exactly when it stops being raw. That is what makes the write path
            # capturable -- there is no allocator left to sync on.

            if _PS.ENABLED:
                # Same whole-window transform, kept in its PACKED form instead of reconstructed.
                # Encoded once at the window's first commit and reused for the rest of its
                # slices, matching the recon path's rule that a row's value must not depend on
                # how many rows commit at a time.
                ps = _PS.store()
                if ps.cfg is None:
                    # live windows are bounded by nslots/ws; the margin covers windows whose
                    # rows are partly still raw
                    _margin = int(os.environ.get("SGLANG_NSN_PACKED_WIN_MARGIN", "512"))
                    # NOT k_buffer.shape[0]: with the hp tier that is the small bf16 row
                    # count, while slot ids span the full (packed-rate) token budget. Sizing
                    # the store by the buffer would index out of bounds the moment the
                    # allocator hands out a slot above the hp row count.
                    _hp0 = _HPP.hp()
                    _nslots = (_hp0.map.shape[0] if _hp0 is not None
                               else k_buffer.shape[0])
                    ps.configure(
                        nslots=_nslots,
                        nwin=k_buffer.shape[0] // ws + _margin,
                        KVH=cache_k.shape[1], D=D, ws=ws,
                        device=cache_k.device, meta_dtype=cache_k.dtype,
                    )
                ls = ps.layer(layer_idx)
                if _BATCH_ENC and ev == 0:
                    # Defer: one encode per (request, layer) is ~15 kernels on a 65K-element
                    # window, so a flush step fires ~224 of them and is launch-bound (measured
                    # 19.9 of the hook's 27.2 ms/step). encode_window_* takes arbitrary leading
                    # dims, so every request due in THIS layer can go through one call. Same
                    # values because the pipeline does not mix rows across leading axes.
                    _PEND.append(dict(st=st, r=r, ls=ls, ps=ps, win_k=win_k, win_v=win_v,
                                      cos=cos, sin=sin, win_loc=win_loc, sl=sl,
                                      layer_idx=layer_idx))
                elif ev == 0:
                    from sglang.srt.mem_cache import nsn_pack as _NP  # circular at module scope
                    st["pk"] = _NP.encode_window_k(win_k, cos, sin, b["codebook"], ws)
                    st["pv"] = _NP.encode_window_v(win_v, b["codebook"], ws,
                                                   _V_HADAMARD_FOLDED)
                    # One arena id per window, SHARED by every layer: `st` is keyed per layer,
                    # so the id cannot live there. Layer 0 allocates and appends to wins[r];
                    # the other layers of the same forward pass look it up by ordinal, which is
                    # identical across layers because they all see the same tokens and run the
                    # same flush schedule.
                    w_i = st.get("win", 0)
                    if layer_idx == 0:
                        st["wid"] = ps.alloc_window(r)
                    else:
                        st["wid"] = ps.wins[r][w_i]
                    ls.commit_meta(st["pk"], st["pv"], st["wid"])
                if not (_BATCH_ENC and ev == 0):
                    # The deferred prefill backlog drains in ONE decode call, so this loop can
                    # reach ev>0 before the batch collected at ev==0 has been encoded. Force
                    # the drain when that happens; steady-state decode does one eviction per
                    # call and never takes this branch.
                    if st.get("pk") is None and _PEND:
                        _drain_pending(b, ws)
                    _COMMITQ.append((ls, st["pk"], st["pv"], win_loc[sl],
                                     torch.arange(ev, ev + gran, device=win_loc.device)))
                if layer_idx == 0:
                    ps.advance(r, gran)

            if not _packed_only:
                # Rows committed in THIS call that are also among the tokens being written.
                # Cannot happen once the packed store is on: a committed row is at least
                # RECENT tokens old, and prefill (the only call carrying many tokens) defers.
                for j in range(ev, ev + gran):
                    loc_j = st["loc"][j]
                    if loc_j in loc_list:
                        out_k[loc_list.index(loc_j)] = recon_k[j]
                        out_v[loc_list.index(loc_j)] = recon_v[j]

            st["ev"] = ev + gran
            if st["ev"] < ws:
                continue
            st["ev"] = 0
            if _PS.ENABLED:
                st["win"] = st.get("win", 0) + 1
                st["pk"] = st["pv"] = None
            if not _HP_ACC:
                st["k"] = st["k"][ws:]
                st["v"] = st["v"][ws:]
            st["loc"] = st["loc"][ws:]
            st["pos"] = st["pos"][ws:]
            if not st["loc"]:
                # nothing buffered: drop the slot so completed requests stop occupying _ACCUM
                # (entries are otherwise never freed -- there is no request-finished hook here).
                _ACCUM.pop(key, None)

    if _PEND:
        _drain_pending(b, ws)
    if _COMMITQ:
        q, _COMMITQ[:] = list(_COMMITQ), []
        _ls = q[0][0]
        _ls.commit_rows_batched([(a, bb, c, d) for _, a, bb, c, d in q])

    return out_k, out_v
