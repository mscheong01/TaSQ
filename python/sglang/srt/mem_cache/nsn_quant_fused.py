"""CUDA-graph-capturable NSNQuant write path.

Per-window transforms match the reference implementation. GPU-resident request rings replace
host-side accumulation, and shape-static flushes route inactive writes to a reserved row. Decode
processes at most one completed window per request and step; eager prefill drains completed
windows in batches. Prefix, recent-window, and discontinuity-reset semantics are preserved.
"""
from __future__ import annotations

import os

import torch
import triton

from sglang.srt.mem_cache import nsn_packed_store as _PS
from sglang.srt.mem_cache import nsn_encode_kernel as _EK
from sglang.srt.mem_cache.nsn_pack import (
    PackedWindow,
    _maybe_compile,
    decode_window,
    encode_window_k as _encode_window_k,
    encode_window_v as _encode_window_v,
)

# dynamic=True, not the plain _maybe_compile. The encode is B=1 whenever the slack is large
# (RECENT=256 -> 200 steps), which is what nemotron and the reasoning cells run, so a static
# compile looked sufficient. At RECENT=64 -- every non-reasoning cell -- the slack is 8, the
# slack<B fallback below fires for capture batches of 12 and 16, and the encode then meets a new
# shape INSIDE CUDA-graph capture: "Cannot call CUDAGeneratorImpl::current_seed during CUDA graph
# capture", i.e. the server does not boot at all. One dynamic graph covers every batch.
# (Only reached with SGLANG_NSN_TRITON_ENCODE=0; the default encoder is nsn_encode_kernel.)
_COMPILE_ENC = os.environ.get("SGLANG_NSN_COMPILE_ENCODE") == "1"
encode_window_k = (torch.compile(_encode_window_k, dynamic=True) if _COMPILE_ENC
                   else _encode_window_k)
encode_window_v = (torch.compile(_encode_window_v, dynamic=True) if _COMPILE_ENC
                   else _encode_window_v)


def _decode_kv(ck_idx, ck_n2, ck_nq, ck_nsc, ck_nmin, ck_mq, ck_msc, ck_mmin,
               cv_idx, cv_n2, cv_nq, cv_nsc, cv_nmin, cv_mq, cv_msc, cv_mmin,
               sel, cs, sn, codebook, D, dtype, folded, arith, ws):
    """Slice both caches and decode both, as one unit so the fuser can see across them.

    Separately these were 0.29 ms of gathers plus 0.54 ms of decode per flush at B=32; the gathers
    feed the decode directly and there is no reason for them to be a separate pass.
    """
    B, G = sel.shape
    H = ck_idx.shape[1]
    e = sel[:, None, :, None]

    def _sl(idx, n2, nq, nsc, nmin, mq, msc, mmin, kind, fold):
        return PackedWindow(
            kind,
            idx.gather(2, e.expand(B, H, G, idx.shape[3])),
            n2.gather(2, e.expand(B, H, G, 1)),
            nq.gather(2, sel[:, None, :].expand(B, H, G)),
            nsc.reshape(-1, 1), nmin.reshape(-1, 1),
            mq, msc.reshape(-1, 1), mmin.reshape(-1, 1),
            32, G, D, dtype, fold, norm2_arith_dtype=arith, hadamard_ws=ws,
        )

    rk = decode_window(_sl(ck_idx, ck_n2, ck_nq, ck_nsc, ck_nmin, ck_mq, ck_msc, ck_mmin,
                           "k", False), codebook, cs, sn)
    rv = decode_window(_sl(cv_idx, cv_n2, cv_nq, cv_nsc, cv_nmin, cv_mq, cv_msc, cv_mmin,
                           "v", folded), codebook)
    return rk.permute(0, 2, 1, 3).contiguous(), rv.permute(0, 2, 1, 3).contiguous()


# dynamic=True: the encode is always one window so its shape is fixed and prewarm's single call
# compiles it, but the decode's leading dim is the decode batch, and SGLang captures a graph per
# batch size in [1, 2, 4, 8, ...]. With a static compile the second size would recompile INSIDE
# capture, which fails outright -- "Cannot call CUDAGeneratorImpl::current_seed during CUDA graph
# capture". One dynamic graph covers them all, and prewarm still triggers it off the capture
# stream.
_decode_kv_c = (torch.compile(_decode_kv, dynamic=True)
                if os.environ.get("SGLANG_NSN_COMPILE_ENCODE") == "1" else _decode_kv)
from sglang.srt.mem_cache.nsn_quant import (
    _PREFIX_TOKENS,
    _gran,
    _RECENT_TOKENS,
    _V_HADAMARD_FOLDED,
    _rope_cos_sin,
    _transform_window_k,
    _transform_window_v,
    load_bundle,
)

MAX_SLOTS = int(os.environ.get("SGLANG_NSN_FUSED_MAX_SLOTS", "512"))
# _nsn_book stores COMMITTED + slot for slot up to MAX_SLOTS-1, but committed_t is sized by
# nsn_packed_store's own MAXREQ. Both default to 512; raising one alone is an unmasked
# out-of-bounds store, so refuse rather than corrupt.
_PACKED_MAXREQ = int(os.environ.get("SGLANG_NSN_PACKED_MAXREQ", "512"))
if MAX_SLOTS > _PACKED_MAXREQ:
    raise RuntimeError(
        f"SGLANG_NSN_FUSED_MAX_SLOTS={MAX_SLOTS} exceeds SGLANG_NSN_PACKED_MAXREQ="
        f"{_PACKED_MAXREQ}; the bookkeeping kernel would write committed_t out of bounds.")
CACHE_SCRATCH = MAX_SLOTS  # packed-cache sink for stores that are masked off
# Eviction granularity is shared with the reference path through ``_gran``.
_DEBUG_AUDIT = os.environ.get("SGLANG_NSN_FUSED_AUDIT")
if _PS.ENABLED and int(os.environ.get("SGLANG_NSN_EVICT_GRAN", "8")) != 8:
    raise RuntimeError("The packed NSNQuant path requires SGLANG_NSN_EVICT_GRAN=8.")


def _audit(st: "_LayerState", layer_idx: int, where: str):
    """Eager-only ring invariant check: positions within each slot's live window must be
    strictly consecutive. Logs violations to $SGLANG_NSN_FUSED_AUDIT."""
    cap = st.cap
    bad = []
    cnts = st.cnt.tolist()
    starts = st.start.tolist()
    for s_ in range(MAX_SLOTS - 1):
        n = cnts[s_]
        if n <= 1:
            continue
        idx = (starts[s_] + torch.arange(n, device=st.pos.device)) % cap
        pos = st.pos[s_, idx]
        d = pos[1:] - pos[:-1]
        if bool((d != 1).any()):
            j = int(torch.nonzero(d != 1)[0])
            bad.append((s_, n, int(pos[j]), int(pos[j + 1]), j))
    if bad:
        with open(_DEBUG_AUDIT, "a") as f:
            f.write(f"layer={layer_idx} at={where} violations={bad[:5]}\n")
TRASH_LOC = 0  # SGLang token pools reserve row 0 as padding; safe write-sink for invalid rows


class _LayerState:
    """GPU-resident replacement for nsn_quant._ACCUM, one per layer.

    `start` addresses the first row of the CURRENT WINDOW, not the first raw row: rows are
    evicted EVICT_GRAN at a time but the window they belong to only rolls once all ws of them
    are gone. `ev` counts how many of the current window are already in the pool as
    reconstructions, so raw tail = cnt - ev.

    The packed cache exists because eviction and transformability are no longer simultaneous.
    `_flush_once` reads the window's RAW rows back out of the pool; once the first GRAN rows have
    been overwritten with reconstructions those raw values are gone, so a later re-transform of
    the same window would read its own output. The window is therefore encoded ONCE, when its
    first eviction comes due, and the remaining evictions decode slices of that. Storing it
    packed rather than as bf16 reconstructions is what makes it affordable: 512 slots x 28 layers
    is 77 MB packed against 917 MB dequantised.
    """

    def __init__(self, cap: int, device, H: int, D: int, ws: int, dtype):
        self.cap = cap
        self.loc = torch.zeros((MAX_SLOTS, cap), dtype=torch.int64, device=device)
        self.pos = torch.full((MAX_SLOTS, cap), -1, dtype=torch.int64, device=device)
        self.start = torch.zeros(MAX_SLOTS, dtype=torch.int64, device=device)
        self.cnt = torch.zeros(MAX_SLOTS, dtype=torch.int64, device=device)
        self.ev = torch.zeros(MAX_SLOTS, dtype=torch.int64, device=device)
        self.enc = torch.zeros(MAX_SLOTS, dtype=torch.bool, device=device)
        self.ws, self.H, self.D = ws, H, D
        u8, f16 = torch.uint8, torch.float16
        mg = D // 32
        # one row past the addressable slots: unwanted stores land here and are never read
            # (nothing ever evicts from CACHE_SCRATCH -- it is not a real request slot).
        def _cache():
            return {
                "idx":   torch.zeros((MAX_SLOTS + 1, H, ws, D // 8), dtype=u8, device=device),
                "norm2": torch.zeros((MAX_SLOTS + 1, H, ws, 1), dtype=f16, device=device),
                "nq":    torch.zeros((MAX_SLOTS + 1, H, ws), dtype=u8, device=device),
                "nsc":   torch.zeros((MAX_SLOTS + 1, H, 1), dtype=dtype, device=device),
                "nmin":  torch.zeros((MAX_SLOTS + 1, H, 1), dtype=dtype, device=device),
                "mq":    torch.zeros((MAX_SLOTS + 1, H, 1, D), dtype=u8, device=device),
                "msc":   torch.zeros((MAX_SLOTS + 1, H, mg, 1), dtype=dtype, device=device),
                "mmin":  torch.zeros((MAX_SLOTS + 1, H, mg, 1), dtype=dtype, device=device),
            }
        self.ck, self.cv = _cache(), _cache()


_STATE: dict = {}  # (path, layer_idx) -> _LayerState


@torch.no_grad()
def prewarm(path: str, layer_num: int, device, dtype, head_num: int = 8,
            head_dim: int = None) -> None:
    """Initialize bundles, rings, constants, and workspaces before CUDA graph capture."""
    b = load_bundle(path, device)
    ws = b["window_size"]
    H, D = int(head_num), int(head_dim if head_dim is not None else b["head_dim"])
    for l in range(layer_num):
        _get_state(path, l, ws, device, H, D, dtype)
    win_k = torch.zeros((1, ws, H, D), device=device, dtype=dtype)
    win_v = torch.zeros((1, ws, H, D), device=device, dtype=dtype)
    win_pos = torch.arange(ws, device=device)[None, :]
    _transform_batched(win_k, win_v, win_pos, b, dtype)
    # the live path is encode/decode now, not _transform_batched -- warm those allocations too
    cos, sin = _rope_cos_sin(win_pos, b["inv_freq"])
    cos, sin = cos.to(dtype), sin.to(dtype)
    for _nb in (1, 2, 4):
        _wk = win_k.expand(_nb, -1, -1, -1).contiguous()
        _wv = win_v.expand(_nb, -1, -1, -1).contiguous()
        _cs3, _sn3 = _rope_cos_sin(win_pos.expand(_nb, -1), b["inv_freq"])
        encode_window_k(_wk.permute(0, 2, 1, 3), _cs3.to(dtype), _sn3.to(dtype), b["codebook"], ws)
        encode_window_v(_wv.permute(0, 2, 1, 3), b["codebook"], ws, _V_HADAMARD_FOLDED)
    _pk = encode_window_k(win_k.permute(0, 2, 1, 3), cos, sin, b["codebook"], ws)
    _pv = encode_window_v(win_v.permute(0, 2, 1, 3), b["codebook"], ws, _V_HADAMARD_FOLDED)
    EVICT_GRAN_WARM = _gran(ws)
    _sel = torch.zeros((1, EVICT_GRAN_WARM), dtype=torch.int64, device=device)
    _st0 = _get_state(path, 0, ws, device)
    _arith = torch.promote_types(b["codebook"].dtype, dtype)
    _cs, _sn = _rope_cos_sin(win_pos[:, : _gran(ws)], b["inv_freq"])
    decode_window(_cache_slice(_st0.ck, torch.zeros(1, dtype=torch.int64, device=device),
                               _sel, "k", D, dtype, arith=_arith),
                  b["codebook"], _cs.to(dtype), _sn.to(dtype))
    decode_window(_cache_slice(_st0.cv, torch.zeros(1, dtype=torch.int64, device=device),
                               _sel, "v", D, dtype, _V_HADAMARD_FOLDED, arith=_arith),
                  b["codebook"])
    # warm the compiled decode off the capture stream, at two batch sizes so the dynamic shape
    # is generalised before any graph capture sees it
    _cbk = b["codebook"]
    _arith = torch.promote_types(_cbk.dtype, dtype)
    for _b in (1, 2):
        _sel2 = torch.zeros((_b, EVICT_GRAN_WARM), dtype=torch.int64, device=device)
        _sl2 = torch.zeros(_b, dtype=torch.int64, device=device)
        _cs2, _sn2 = _rope_cos_sin(_sel2, b["inv_freq"])
        K, V = _st0.ck, _st0.cv
        _decode_kv_c(
            K["idx"][_sl2], K["norm2"][_sl2], K["nq"][_sl2], K["nsc"][_sl2], K["nmin"][_sl2],
            K["mq"][_sl2], K["msc"][_sl2], K["mmin"][_sl2],
            V["idx"][_sl2], V["norm2"][_sl2], V["nq"][_sl2], V["nsc"][_sl2], V["nmin"][_sl2],
            V["mq"][_sl2], V["msc"][_sl2], V["mmin"][_sl2],
            _sel2, _cs2.to(dtype), _sn2.to(dtype), _cbk, D, dtype, _V_HADAMARD_FOLDED, _arith, ws)
    # and the bookkeeping pair, at two batch sizes so the dynamic shape is settled pre-capture
    for _b in (1, 2):
        _sl3 = torch.zeros(_b, dtype=torch.int64, device=device)
        _ar3 = torch.arange(ws, device=device)
        _prep_c(_st0.cnt, _st0.ev, _st0.enc, _st0.start, _sl3, _ar3, _st0.cap,
                _RECENT_TOKENS, _gran(ws), ws)
        _commit_c(_st0.cnt, _st0.ev, _st0.enc, _st0.start, _sl3,
                  torch.zeros(_b, dtype=torch.bool, device=device), _gran(ws), ws, _st0.cap)
    torch.cuda.synchronize()
    print(f"[nsn_quant_fused] prewarmed: {layer_num} layer states (H={H} D={D} ws={ws} "
          f"evict_gran={_gran(ws)}{' (=ws, legacy schedule)' if _gran(ws) == ws else ''}) "
          f"+ transform caches on the default stream (pre-capture)",
          flush=True)


def _get_state(path: str, layer_idx: int, ws: int, device, H=None, D=None,
               dtype=None) -> _LayerState:
    key = (path, layer_idx)
    st = _STATE.get(key)
    if st is None:
        # Lazy allocation is the original behaviour and is kept: prewarm exists to move it OFF
        # the capture stream, not to make it mandatory (validate_nsn_fused.py drives the hook
        # directly, with no pool and so no prewarm). Allocating during capture is the thing that
        # actually corrupts, so that is what raises.
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "nsn_quant_fused state would be allocated during CUDA-graph capture -- prewarm() "
                "must run on the default stream first (see its docstring; boot-dependent "
                "reconstruction corruption otherwise)."
            )
        if H is None or D is None or dtype is None:
            raise RuntimeError("_get_state needs H/D/dtype to size the packed window cache")
        # cap must hold cnt's ceiling: prefill drains to < ws + RECENT + 1; a decode append
        # before its drain adds 1 more. Margin doubles that for safety; ring indexing wraps.
        # Ring capacity sets how many windows the prefill can accumulate before it must drain,
        # and therefore how many go through one batched transform: at the old 2*(ws+RECENT+2) it
        # was four, i.e. seven calls per 2048-token request per layer. Sized from NFULL_PAD
        # instead -- the drain's chunk is (cap - 2*ws - RECENT), and it needs NFULL_PAD*ws of it.
        cap = NFULL_PAD * ws + _RECENT_TOKENS + 2 * ws + 64
        st = _LayerState(cap, device, H, D, ws, dtype)
        _STATE[key] = st
    return st


def _append(st: _LayerState, slots, pos, loc, first_layer=False):
    """Append one row per slot (slots unique -- the decode case). Pure tensor ops.

    Mirrors the reference's discontinuity reset: a slot whose incoming pos is not
    last_pos + 1 (or with an empty buffer and pos != anything -- empty accepts any pos,
    matching the reference where a fresh accumulator starts wherever the first row lands)
    drops its buffered rows first.
    """
    cap = st.cap
    last_idx = (st.start[slots] + st.cnt[slots] - 1) % cap
    has = st.cnt[slots] > 0
    expected = torch.where(has, st.pos[slots, last_idx] + 1, pos)
    reset = pos != expected
    st.cnt[slots] = torch.where(reset, torch.zeros_like(st.cnt[slots]), st.cnt[slots])
    # ``ev`` belongs to the current window and must reset with ``cnt``.
    st.ev[slots] = torch.where(reset, torch.zeros_like(st.ev[slots]), st.ev[slots])
    st.enc[slots] = st.enc[slots] & ~reset
    if _PS.ENABLED and first_layer:
        # ``committed_t`` is shared across layers and advanced by layer 0 alone.
        ps = _PS.store()
        if ps.committed_t is not None:
            ps.committed_t[slots] = torch.where(
                reset, torch.zeros_like(ps.committed_t[slots]), ps.committed_t[slots])
    # (start need not move on reset; cnt=0 makes old contents unreachable, and with ev=0 the
    #  next window is based at `start` again, which is where the first append lands)
    tail = (st.start[slots] + st.cnt[slots]) % cap
    st.loc[slots, tail] = loc
    st.pos[slots, tail] = pos
    st.cnt[slots] = st.cnt[slots] + 1


def _bookkeep(cnt, ev, enc, start, cap, recent, gran, ws, ar_ws, slots_col):
    """due / fresh / ring indices / urgency, as one fusible block.

    Individually these are a dozen tiny elementwise kernels on [B] tensors -- 0.22 ms for
    _flush_candidates plus 0.24 ms for _advance plus 0.10 ms for the urgency pick, at B=32, which
    is a quarter of the flush and none of it arithmetic that matters. They are separated only by
    having been written as separate functions.
    """
    due = cnt >= (recent + ev + gran)
    fresh = due & (ev == 0)
    idx = (start[:, None] + ar_ws[None, :]) % cap
    urgency = torch.where((cnt >= ws) & ~enc, cnt, torch.full_like(cnt, -1))
    return due, fresh, idx, urgency


def _flush_candidates(st: _LayerState, slots, ws: int, require_enc: bool = True,
                      compiled: bool = True):
    """The current window (ws rows from `start`) per slot, plus the eviction decision.

    A row may leave the raw tail once it is older than RECENT; rows go out EVICT_GRAN at a time,
    matching the other methods' flush_interval. The next GRAN rows are rows [ev, ev+GRAN) of the
    window, and the newest of them has aged out when

        cnt - 1 - (ev + GRAN - 1) >= RECENT   <=>   cnt >= RECENT + ev + GRAN

    Window completeness never has to be checked separately: evicting row t needs t <= n - RECENT,
    and t's window ends by t + ws - 1 <= n - RECENT + ws - 1 < n whenever RECENT >= ws. Both
    configurations this project serves (RECENT=256 and RECENT=64) satisfy that, and a simulation
    over n=1..5000 finds zero steps where completeness rather than RECENT was binding.
    """
    due_raw, due_enc, ev, idx, urgency = (_prep_c if compiled else _prep_t)(
        st.cnt, st.ev, st.enc, st.start, slots,
        torch.arange(ws, device=slots.device), st.cap,
        _RECENT_TOKENS, _gran(ws), ws)
    due = due_enc if require_enc else due_raw
    fresh = due & (ev == 0)          # window not encoded yet -> its rows are still raw in the pool
    win_loc = st.loc[slots[:, None].expand_as(idx), idx]
    win_pos = st.pos[slots[:, None].expand_as(idx), idx]
    st._urgency = urgency            # the encode picker reuses this instead of recomputing it
    return due, fresh, win_loc, win_pos


def _prep_t(cnt_all, ev_all, enc_all, start_all, slots, ar_ws, cap, recent, gran, ws):
    """due / ring indices / urgency in one block.

    Separately these were a dozen tiny elementwise kernels on [B] tensors: 0.23 ms in
    _flush_candidates, 0.25 ms in _advance and 0.10 ms for the urgency pick at B=32, so 30% of a
    2.81 ms flush spent on bookkeeping rather than on arithmetic that matters. They were separate
    only because they were written as separate functions.
    """
    cnt, ev, enc = cnt_all[slots], ev_all[slots], enc_all[slots]
    due_raw = cnt >= (recent + ev + gran)
    idx = (start_all[slots][:, None] + ar_ws[None, :]) % cap
    urgency = torch.where((cnt >= ws) & ~enc, cnt, torch.full_like(cnt, -1))
    return due_raw, due_raw & enc, ev, idx, urgency


def _commit_t(cnt_all, ev_all, enc_all, start_all, slots, due, gran, ws, cap):
    ev = torch.where(due, ev_all[slots] + gran, ev_all[slots])
    roll = ev >= ws
    start_all[slots] = torch.where(roll, (start_all[slots] + ws) % cap, start_all[slots])
    cnt_all[slots] = torch.where(roll, cnt_all[slots] - ws, cnt_all[slots])
    ev_all[slots] = torch.where(roll, torch.zeros_like(ev), ev)
    enc_all[slots] = enc_all[slots] & ~roll


# dynamic=True and prewarmed, for the same reason as _decode_kv_c: the leading dim is the decode
# batch, SGLang captures one graph per batch size, and a recompile inside capture is fatal
# ("Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture").
_COMPILE_ON = os.environ.get("SGLANG_NSN_COMPILE_ENCODE") == "1"
_prep_c = torch.compile(_prep_t, dynamic=True) if _COMPILE_ON else _prep_t
_commit_c = torch.compile(_commit_t, dynamic=True) if _COMPILE_ON else _commit_t


def _advance(st: _LayerState, slots, due, ws: int, compiled: bool = True):
    """Charge the eviction, and roll to the next window once all ws rows of this one are out."""
    (_commit_c if compiled else _commit_t)(
        st.cnt, st.ev, st.enc, st.start, slots, due, _gran(ws), ws, st.cap)


def _cache_store(cache: dict, slots, p: PackedWindow, fresh):
    """Write this window's packed form for the slots that just encoded it; leave the rest.

    Unwanted rows are routed to a reserved scratch slot rather than masked in place. Masking cost
    a gather, a `where` and a scatter for each of eight fields -- 0.55 ms of a 4.48 ms flush at
    B=32, for slots that were not being written at all. Sending them somewhere harmless makes it
    one scatter per field. Same pattern the pool itself uses with TRASH_LOC.
    """
    H = p.norm_q.shape[1]
    mg = cache["msc"].shape[2]
    dst = torch.where(fresh, slots, torch.full_like(slots, CACHE_SCRATCH))
    cache["idx"][dst] = p.idx
    cache["norm2"][dst] = p.norm2
    cache["mq"][dst] = p.mean_q
    cache["nq"][dst] = p.norm_q
    cache["nsc"][dst] = p.norm_scale.reshape(-1, H, 1)
    cache["nmin"][dst] = p.norm_min.reshape(-1, H, 1)
    cache["msc"][dst] = p.mean_scale.reshape(-1, H, mg, 1)
    cache["mmin"][dst] = p.mean_min.reshape(-1, H, mg, 1)


def _cache_slice(cache: dict, slots, sel, kind: str, D: int, dtype, folded=False,
                 arith=torch.float32, had_ws: int = None) -> PackedWindow:
    """PackedWindow over just the GRAN rows named by `sel` ([B, GRAN] row indices in the window).

    Slicing is exact rather than approximate: idx/norm2/norm_q are per row, and norm's
    (scale, min) is per WINDOW, so decoding n rows with group_size=n reproduces the same
    q*scale+min those rows get in a full-window decode. `mean` is per window and broadcasts.
    """
    B, G = sel.shape
    H = cache["idx"].shape[1]
    ci, cn2, cnq = cache["idx"][slots], cache["norm2"][slots], cache["nq"][slots]
    e = sel[:, None, :, None]
    idx = ci.gather(2, e.expand(B, H, G, ci.shape[3]))
    n2 = cn2.gather(2, e.expand(B, H, G, 1))
    nq = cnq.gather(2, sel[:, None, :].expand(B, H, G))
    return PackedWindow(
        kind, idx, n2, nq,
        cache["nsc"][slots].reshape(-1, 1), cache["nmin"][slots].reshape(-1, 1),
        cache["mq"][slots], cache["msc"][slots].reshape(-1, 1),
        cache["mmin"][slots].reshape(-1, 1),
        32, G, D, dtype, folded, norm2_arith_dtype=arith,
        # run the Hadamard at the full window height -- see PackedWindow.hadamard_ws
        hadamard_ws=had_ws if had_ws is not None else cache["idx"].shape[2],
    )


def _transform_batched_c(win_k, win_v, win_pos, bundle, dtype):
    """Compiled wrapper for the eager prefill's whole-window transform.

    Prefill is 28 windows per request per layer and, unlike the decode path, it is not captured,
    so the only thing standing between it and the fuser is that nobody had wrapped it. Same drift
    as the compiled encode -- the schedule axis of validate_nsn_fused.py stays exact.
    """
    return _transform_batched(win_k, win_v, win_pos, bundle, dtype)


@torch.no_grad()
def _transform_batched(win_k, win_v, win_pos, bundle, dtype):
    """win_k/win_v: [B, ws, H, D] raw rows; win_pos: [B, ws]. Returns recon [B, ws, H, D].

    Operations are independent across the leading window axis, so batching preserves the
    per-window computation while avoiding a Python loop.
    """
    B, ws, H, D = win_k.shape
    cos, sin = _rope_cos_sin(win_pos, bundle["inv_freq"])  # [B, ws, D]
    cos, sin = cos.to(dtype), sin.to(dtype)
    kh = win_k.permute(0, 2, 1, 3)  # [B, H, ws, D]
    vh = win_v.permute(0, 2, 1, 3)
    rk = _transform_window_k(kh, cos, sin, bundle["codebook"], ws).to(dtype)
    rv = _transform_window_v(vh, bundle["codebook"], ws, _V_HADAMARD_FOLDED).to(dtype)
    return rk.permute(0, 2, 1, 3).contiguous(), rv.permute(0, 2, 1, 3).contiguous()


# Pad to a fixed size so the compiled prefill transform uses one graph shape.
NFULL_PAD = int(os.environ.get("SGLANG_NSN_NFULL_PAD", "32"))
_COMPILE_ENC_STATIC = os.environ.get("SGLANG_NSN_COMPILE_ENCODE") == "1"
_tb_c = (torch.compile(_transform_batched, dynamic=False)
         if os.environ.get("SGLANG_NSN_COMPILE_ENCODE") == "1" else _transform_batched)


@torch.no_grad()
def _flush_eager(st, slots, bundle, k_buffer, v_buffer, dtype, layer_idx=-1):
    """Drain completed prefill windows through the packed cache.

    Prefill is eager, so it may branch and commit all due rows together. Partially committed
    windows are decoded from their packed representation rather than reread from overwritten
    BF16 rows.
    """
    ws = bundle["window_size"]
    gran = _gran(ws)
    cb = bundle["codebook"]
    arith = torch.promote_types(cb.dtype, dtype)
    due, fresh, win_loc, win_pos = _flush_candidates(st, slots, ws, require_enc=False,
                                                     compiled=False)
    idx = torch.nonzero(due).flatten()
    if idx.numel() == 0:
        return due

    # host loop over the due slots -- a prefill call carries one request in the common case, and
    # the eager path already loops per slot for the append.
    ar_ws = torch.arange(ws, device=slots.device)
    for b_i in idx.tolist():
        sl = slots[b_i : b_i + 1]
        wl, wp = win_loc[b_i : b_i + 1], win_pos[b_i : b_i + 1]
        ev = int(st.ev[sl])
        cnt = int(st.cnt[sl])

        # Several whole windows can be due at once when the append has run ahead of the drain.
        # Transform them in ONE call instead of one per window: prefill is ~25k window transforms
        # (32 requests x 28 windows x 28 layers) and every one of them was running at B=1, which
        # is essentially all of its 70 s against fp16's 3.2 s. Values are unchanged -- this is
        # still the whole-window transform with several windows stacked on the batch axis.
        if ev == 0 and _PS.ENABLED:
            # SGLANG_NSN_PACKED prefill: every whole window past RECENT goes to the packed
            # store in one batched encode (padded to NFULL_PAD windows so the compiled encoder
            # sees one shape). Nothing bf16 is written unless KEEP_BF16.
            nfull = min((cnt - _RECENT_TOKENS) // ws, NFULL_PAD)
            if nfull >= 1:
                ps, ls = _packed_layer(st, layer_idx, ws, slots.device, st.ck["nsc"].dtype)
                ridx = (int(st.start[sl]) + torch.arange(nfull * ws, device=slots.device)) % st.cap
                ml = st.loc[sl.expand(nfull * ws), ridx].reshape(nfull, ws)
                mp = st.pos[sl.expand(nfull * ws), ridx].reshape(nfull, ws)
                # pad only for the compiled (static-shape) encoder; eager runs the true count
                # (a 445-token prompt has 5 windows -- padding to 32 was 6x the encode work)
                NP = NFULL_PAD if _COMPILE_ENC_STATIC else nfull
                pad = NP - nfull
                ml_c = torch.cat([ml, torch.full((pad, ws), TRASH_LOC, dtype=ml.dtype,
                                                 device=ml.device)]) if pad else ml
                mp_c = torch.cat([mp, torch.zeros((pad, ws), dtype=mp.dtype,
                                                  device=mp.device)]) if pad else mp
                mrow = _hrow(ml_c)
                if _EK.ENABLED:
                    mpos = mp_c.clamp_min(0)
                    fk = _EK.scratch_cache(NP, st.H, ws, st.D, dtype, slots.device)
                    fv = _EK.scratch_cache(NP + 1, st.H, ws, st.D, dtype, slots.device)
                    adst = torch.arange(NP, device=slots.device)
                    _EK.encode_kv_windows_into(k_buffer, v_buffer, mrow, mpos, bundle["inv_freq"], cb,
fk, fv, adst, not _V_HADAMARD_FOLDED)
                    kf, vf = fk, fv
                else:
                    cos, sin = _rope_cos_sin(mp_c.clamp_min(0), bundle["inv_freq"])
                    cos, sin = cos.to(dtype), sin.to(dtype)
                    pk = encode_window_k(k_buffer[mrow].permute(0, 2, 1, 3), cos, sin, cb, ws)
                    pv = encode_window_v(v_buffer[mrow].permute(0, 2, 1, 3), cb, ws,
                                         _V_HADAMARD_FOLDED)
                    kf, vf = _pw_fields(pk), _pw_fields(pv)
                sel_all = ar_ws[None, :].expand(NP, ws)
                wid = _wid_for(ps, ls, sl.expand(NP), mp_c[:, 0], ws,
                               due=torch.arange(NP, device=slots.device) < nfull)
                if _EK.ENABLED:
                    _EK.commit_into(ls, kf, vf, adst, sel_all, ml_c.reshape(-1), wid)
                else:
                    _commit_packed(ls, kf, vf, sel_all, ml_c.reshape(-1), wid, sl)
                if layer_idx == 0:
                    ps.committed_t[sl] += nfull * ws
                if _PS.KEEP_BF16:
                    rk, rv = _tb_c(k_buffer[mrow], v_buffer[mrow], mp_c.clamp_min(0), bundle, dtype)
                    dst = _hrow(ml.reshape(-1))
                    k_buffer.index_copy_(0, dst, rk[:nfull].reshape(-1, *rk.shape[2:]).to(k_buffer.dtype))
                    v_buffer.index_copy_(0, dst, rv[:nfull].reshape(-1, *rv.shape[2:]).to(v_buffer.dtype))
                st.start[sl] = (st.start[sl] + nfull * ws) % st.cap
                st.cnt[sl] = st.cnt[sl] - nfull * ws
                st.enc[sl] = False
                cnt = int(st.cnt[sl])
                base = (int(st.start[sl]) + ar_ws) % st.cap
                wl = st.loc[sl.expand(ws), base][None, :]
                wp = st.pos[sl.expand(ws), base][None, :]
        elif ev == 0:
            nfull = (cnt - _RECENT_TOKENS) // ws
            if nfull >= 2:
                nfull = min(nfull, NFULL_PAD)
                ridx = (int(st.start[sl]) + torch.arange(nfull * ws, device=slots.device)) % st.cap
                ml = st.loc[sl.expand(nfull * ws), ridx].reshape(nfull, ws)
                mp = st.pos[sl.expand(nfull * ws), ridx].reshape(nfull, ws)
                if nfull < NFULL_PAD:
                    pad = NFULL_PAD - nfull
                    ml_c = torch.cat([ml, torch.full((pad, ws), TRASH_LOC,
                                                     dtype=ml.dtype, device=ml.device)])
                    mp_c = torch.cat([mp, torch.zeros((pad, ws), dtype=mp.dtype,
                                                      device=mp.device)])
                else:
                    ml_c, mp_c = ml, mp
                _mr = _hrow(ml_c)
                rk, rv = _tb_c(k_buffer[_mr], v_buffer[_mr], mp_c.clamp_min(0), bundle, dtype)
                rk, rv = rk[:nfull], rv[:nfull]
                dst = _hrow(ml.reshape(-1))
                k_buffer.index_copy_(0, dst, rk.reshape(-1, *rk.shape[2:]).to(k_buffer.dtype))
                v_buffer.index_copy_(0, dst, rv.reshape(-1, *rv.shape[2:]).to(v_buffer.dtype))
                st.start[sl] = (st.start[sl] + nfull * ws) % st.cap
                st.cnt[sl] = st.cnt[sl] - nfull * ws
                st.enc[sl] = False
                cnt = int(st.cnt[sl])
                base = (int(st.start[sl]) + ar_ws) % st.cap
                wl = st.loc[sl.expand(ws), base][None, :]
                wp = st.pos[sl.expand(ws), base][None, :]

        nout = min(((cnt - _RECENT_TOKENS - ev) // gran) * gran, ws - ev)
        if nout <= 0:
            continue
        if ev == 0 and nout == ws and not _PS.ENABLED:
            # The whole window leaves at once and every row of it is still raw, so there is
            # nothing for the cache to carry between evictions: transform once and write, which
            # is exactly what the unpatched code did. Same values as the encode/decode route --
            # decode(encode(x)) is bit-equal to the transform (verified for G in {8,16,32} x
            # {bf16,fp16}) -- at half the work. This is the common case during prefill at
            # RECENT=256, and skipping the round trip is what keeps prefill at the original cost
            # rather than 1.8x it.
            _wr = _hrow(wl)
            rk, rv = _tb_c(k_buffer[_wr], v_buffer[_wr], wp.clamp_min(0), bundle, dtype)
            dst = _hrow(wl.reshape(-1))
            k_buffer.index_copy_(0, dst, rk.reshape(-1, *rk.shape[2:]).to(k_buffer.dtype))
            v_buffer.index_copy_(0, dst, rv.reshape(-1, *rv.shape[2:]).to(v_buffer.dtype))
            st.start[sl] = (st.start[sl] + ws) % st.cap
            st.cnt[sl] = st.cnt[sl] - ws
            st.ev[sl] = 0
            st.enc[sl] = False
            continue

        if ev == 0:  # window newly due: every row of it is still raw in the pool
            cos, sin = _rope_cos_sin(wp.clamp_min(0), bundle["inv_freq"])
            cos, sin = cos.to(dtype), sin.to(dtype)
            _wr = _hrow(wl)
            if _EK.ENABLED:
                wpos = wp.clamp_min(0)
                _EK.encode_kv_windows_into(k_buffer, v_buffer, _wr, wpos, bundle["inv_freq"], cb, st.ck, st.cv,
        sl, not _V_HADAMARD_FOLDED)
            else:
                pk = encode_window_k(k_buffer[_wr].permute(0, 2, 1, 3), cos, sin, cb, ws)
                pv = encode_window_v(v_buffer[_wr].permute(0, 2, 1, 3), cb, ws, _V_HADAMARD_FOLDED)
                one = torch.ones(1, dtype=torch.bool, device=slots.device)
                _cache_store(st.ck, sl, pk, one)
                _cache_store(st.cv, sl, pv, one)
            st.enc[sl] = True

        sel = torch.arange(ev, ev + nout, device=slots.device)[None, :]
        if _PS.ENABLED:
            ps, ls = _packed_layer(st, layer_idx, ws, slots.device, st.ck["nsc"].dtype)
            wid = _wid_for(ps, ls, sl, wp[:, 0], ws)
            if _EK.ENABLED:
                _EK.commit_into(ls, st.ck, st.cv, sl, sel, wl.gather(1, sel).reshape(-1), wid)
            else:
                _commit_packed(ls, _slot_fields(st.ck, sl), _slot_fields(st.cv, sl), sel,
                               wl.gather(1, sel).reshape(-1), wid, sl)
            if layer_idx == 0:
                ps.committed_t[sl] += nout
        if _PS.ENABLED and not _PS.KEEP_BF16:
            ev += nout
            if ev >= ws:
                st.start[sl] = (st.start[sl] + ws) % st.cap
                st.cnt[sl] = st.cnt[sl] - ws
                st.ev[sl] = 0
                st.enc[sl] = False
            else:
                st.ev[sl] = ev
            continue
        sk = _cache_slice(st.ck, sl, sel, "k", st.D, dtype, arith=arith, had_ws=ws)
        sv = _cache_slice(st.cv, sl, sel, "v", st.D, dtype, _V_HADAMARD_FOLDED,
                          arith=arith, had_ws=ws)
        cs, sn = _rope_cos_sin(wp.gather(1, sel).clamp_min(0), bundle["inv_freq"])
        rk = decode_window(sk, cb, cs.to(dtype), sn.to(dtype)).permute(0, 2, 1, 3)
        rv = decode_window(sv, cb).permute(0, 2, 1, 3)
        dst = _hrow(wl.gather(1, sel).reshape(-1))
        k_buffer.index_copy_(0, dst, rk.reshape(-1, *rk.shape[2:]).to(k_buffer.dtype))
        v_buffer.index_copy_(0, dst, rv.reshape(-1, *rv.shape[2:]).to(v_buffer.dtype))

        ev += nout
        if ev >= ws:
            st.start[sl] = (st.start[sl] + ws) % st.cap
            st.cnt[sl] = st.cnt[sl] - ws
            st.ev[sl] = 0
            st.enc[sl] = False
        else:
            st.ev[sl] = ev
    return due



MAXW = int(os.environ.get("SGLANG_NSN_PACKED_MAXWIN_PER_REQ", "64"))


def _hrow(loc):
    """Pool slot(s) -> bf16-tier row(s). Identity without the packed tier; through the ring's
    slot map with it (pool slot 0 / TRASH_LOC resolves to the ring's trash row)."""
    from sglang.srt.mem_cache import nsn_hp_pool as _HPP

    h = _HPP.hp()
    return h.lookup(loc).long() if h is not None else loc


def _packed_layer(st, layer_idx, ws, device, meta_dtype):
    """The packed store's LayerStore, configuring the store on first use (fused path)."""
    from sglang.srt.mem_cache import nsn_hp_pool as _HPP

    ps = _PS.store()
    if ps.cfg is None:
        ring = _HPP.hp()
        # At least one metadata window per request is required; zero would alias every request to
        # the trash entry.
        if MAXW < 1:
            raise RuntimeError(
                f"SGLANG_NSN_PACKED_MAXWIN_PER_REQ={MAXW}: the packed NSN window arena needs at "
                f"least 1 window per request. Windows are position-aligned (ordinal = pos // "
                f"{ws}), so a request served at context C needs ceil(C / {ws}) of them -- 640 at "
                f"the 40960 the reasoning protocol uses. Set it from the SERVED CONTEXT; note "
                f"that an unset shell variable in an arithmetic expansion silently yields 0.")
        ps.configure(nslots=ring.map.shape[0], nwin=ring.n_pre * MAXW, KVH=st.H, D=st.D,
                     ws=ws, device=device, meta_dtype=meta_dtype)
    return ps, ps.layer(layer_idx)


def _pw_fields(p: PackedWindow):
    """A PackedWindow (batched, [nw, H, ws, ...]) as the same dict layout st.ck/st.cv use."""
    H = p.norm_q.shape[1]
    mg = p.mean_q.shape[-1] // 32          # mean groups per head (MEAN_GROUP=32), as _cache_store
    return {"idx": p.idx, "norm2": p.norm2, "nq": p.norm_q,
            "nsc": p.norm_scale.reshape(-1, H, 1), "nmin": p.norm_min.reshape(-1, H, 1),
            "mq": p.mean_q, "msc": p.mean_scale.reshape(-1, H, mg, 1),
            "mmin": p.mean_min.reshape(-1, H, mg, 1)}


def _win_ordinal(win_pos0, ws):
    """Window arena ordinal = the window's first position // ws (position-aligned windows,
    PREFIX=0), which is exactly what the read path indexes: compact[req] * MAXW + pos // ws."""
    return (win_pos0.clamp_min(0) // ws).clamp_max(MAXW - 1)


@torch.no_grad()
def _commit_packed(ls, kf, vf, sel, dst, wid, req_slots):
    """Scatter packed window fields into the store.

    kf/vf: field dicts over nw windows ([nw, H, ws, ...] per-token fields, [nw, H, ...] window
    metadata). sel [nw, n]: the rows (within the window) leaving the raw tail now. dst [nw*n]:
    their pool slots, TRASH_LOC for rows that must not land. wid [nw]: window arena rows, the
    trash row for windows not due. Pure gather/scatter on device tensors -- capturable.
    """
    from sglang.srt.mem_cache.nsn_pack import pack_nibbles

    nw, n = sel.shape
    H = ls.KVH
    e = sel[:, None, :, None]
    ik = kf["idx"].gather(2, e.expand(nw, H, n, kf["idx"].shape[3]))
    iv = vf["idx"].gather(2, e.expand(nw, H, n, vf["idx"].shape[3]))
    n2k = kf["norm2"].gather(2, e.expand(nw, H, n, 1))
    n2v = vf["norm2"].gather(2, e.expand(nw, H, n, 1))
    e3 = sel[:, None, :].expand(nw, H, n)
    nqk = kf["nq"].gather(2, e3)
    nqv = vf["nq"].gather(2, e3)
    ls.idx_k[dst] = ik.permute(0, 2, 1, 3).reshape(nw * n, H, -1)
    ls.idx_v[dst] = iv.permute(0, 2, 1, 3).reshape(nw * n, H, -1)
    ls.n2[dst, :, 0] = n2k.permute(0, 2, 1, 3).reshape(nw * n, H).to(torch.float16)
    ls.n2[dst, :, 1] = n2v.permute(0, 2, 1, 3).reshape(nw * n, H).to(torch.float16)
    ls.nrm8[dst] = (nqk.permute(0, 2, 1).reshape(nw * n, H)
                    | (nqv.permute(0, 2, 1).reshape(nw * n, H) << 4))
    dt = ls.nsc.dtype
    ls.nsc[wid, :, 0] = kf["nsc"].reshape(nw, H).to(dt)
    ls.nsc[wid, :, 1] = kf["nmin"].reshape(nw, H).to(dt)
    ls.nsc[wid, :, 2] = vf["nsc"].reshape(nw, H).to(dt)
    ls.nsc[wid, :, 3] = vf["nmin"].reshape(nw, H).to(dt)
    ls.mk_q[wid] = pack_nibbles(kf["mq"].reshape(nw, H, -1))
    ls.mv_q[wid] = pack_nibbles(vf["mq"].reshape(nw, H, -1))
    ls.mk_s[wid, :, :, 0] = kf["msc"].reshape(nw, H, -1).to(dt)
    ls.mk_s[wid, :, :, 1] = kf["mmin"].reshape(nw, H, -1).to(dt)
    ls.mv_s[wid, :, :, 0] = vf["msc"].reshape(nw, H, -1).to(dt)
    ls.mv_s[wid, :, :, 1] = vf["mmin"].reshape(nw, H, -1).to(dt)


def _slot_fields(cache, slots):
    return {k: cache[k][slots] for k in ("idx", "norm2", "nq", "nsc", "nmin", "mq", "msc", "mmin")}


def _wid_for(ps, ls, slots, win_pos0, ws, due=None):
    from sglang.srt.mem_cache import nsn_hp_pool as _HPP

    comp = _HPP.hp().compact[slots]
    wid = ps.wid_arith(comp, _win_ordinal(win_pos0, ws), MAXW)
    if due is not None:
        wid = torch.where(due, wid, torch.full_like(wid, ls.nsc.shape[0] - 1))
    return wid


@torch.no_grad()
def _decode_step_kernels(st, slots, pos, lc, bundle, cache_k, cache_v, k_buffer, v_buffer,
                         layer_idx):
    """The packed decode step as ~10 launches: bookkeeping kernel, pre-write, fused encoder,
    commit kernel. Same state transitions and same stored bits as _append + _flush_once."""
    from sglang.srt.mem_cache import nsn_hp_pool as _HPP

    ws = bundle["window_size"]
    gran = _gran(ws)
    B = slots.shape[0]
    slack = _RECENT_TOKENS - ws + gran - 1   # pick steps between a roll and the next due
    NENC = B if slack <= 0 else min(B, max(1, -(-B // slack)))
    ps, ls = _packed_layer(st, layer_idx, ws, slots.device, st.ck["nsc"].dtype)
    ring = _HPP.hp()
    due, sel, dst, wid, lrow, prow, ppos, cdst, spos = _EK.decode_bookkeeping(
        st, slots, pos, lc, ring.map, ring.compact, ps.committed_t, ws, _RECENT_TOKENS, gran,
        NENC, ls.nsc.shape[0] - 1, MAX_SLOTS - 1, MAXW, layer_idx == 0, TRASH_LOC,
        CACHE_SCRATCH, cache_k.contiguous(), cache_v.contiguous(), k_buffer, v_buffer,
        prefix=_PREFIX_TOKENS)
    cb = bundle["codebook"]
    _EK.encode_kv_windows_into(k_buffer, v_buffer, prow, ppos, bundle["inv_freq"], cb, st.ck, st.cv,
        cdst, not _V_HADAMARD_FOLDED)
    _EK.commit_into(ls, st.ck, st.cv, slots, sel, dst, wid)
    if _PS.KEEP_BF16:
        # audit mode: the SAME served path, plus the bf16 reconstruction of the rows just
        # committed, so a full-span bf16 read can serve as the reference for the packed read
        _write_recon_rows(st, slots, sel, dst, spos, bundle, k_buffer, v_buffer, cache_k.dtype)
    return due


@torch.no_grad()
def _write_recon_rows(st, slots, sel, dst, spos, bundle, k_buffer, v_buffer, dtype):
    cb = bundle["codebook"]
    arith = torch.promote_types(cb.dtype, dtype)
    cs, sn = _rope_cos_sin(spos.clamp_min(0), bundle["inv_freq"])
    K, V = st.ck, st.cv
    rk, rv = _decode_kv(
        K["idx"][slots], K["norm2"][slots], K["nq"][slots], K["nsc"][slots], K["nmin"][slots],
        K["mq"][slots], K["msc"][slots], K["mmin"][slots],
        V["idx"][slots], V["norm2"][slots], V["nq"][slots], V["nsc"][slots], V["nmin"][slots],
        V["mq"][slots], V["msc"][slots], V["mmin"][slots],
        sel, cs.to(dtype), sn.to(dtype), cb, st.D, dtype, _V_HADAMARD_FOLDED, arith, st.ws)
    drow = _hrow(dst.reshape(-1))
    k_buffer.index_copy_(0, drow, rk.reshape(-1, *rk.shape[2:]).to(k_buffer.dtype))
    v_buffer.index_copy_(0, drow, rv.reshape(-1, *rv.shape[2:]).to(v_buffer.dtype))


@torch.no_grad()
def _prefill_packed(st, s, p, l, bundle, k_buffer, v_buffer, layer_idx,
                    src_k=None, src_v=None, src_idx=None):
    """Prefill of one request slot `s` (rows p/l, positions/pool slots, in order) straight into
    the packed store. Every bookkeeping quantity is host-side arithmetic after three syncs
    (reset check, start, cnt), so the whole call is ~40 launches instead of the generic drain's
    ~250 with a sync per iteration. Same windows, same rows, same bits as _flush_eager.

    src_k/src_v/src_idx: THIS CALL's K/V and the rows of them belonging to this request, in
    position order. Windows built entirely out of this call are encoded straight from those.

    The BF16 tier is a per-request ring. Windows fully contained in this call use ``src_k`` and
    ``src_v`` directly so long prompts cannot overwrite their own source rows before encoding.
    Only windows that begin before this call fall back to the ring.
    """
    from sglang.srt.mem_cache import nsn_hp_pool as _HPP

    ws = bundle["window_size"]
    gran = _gran(ws)
    cb = bundle["codebook"]
    inv_freq = bundle["inv_freq"]
    dev = p.device
    cap = st.cap
    ps, ls = _packed_layer(st, layer_idx, ws, dev, st.ck["nsc"].dtype)
    ring = _HPP.hp()
    comp = int(ring.compact[s])
    # The discontinuity reset comes FIRST, before the PREFIX filter can return early. A prompt
    # that is entirely sink rows (shorter than PREFIX, on a slot recycled from a longer request)
    # leaves n == 0; filtering first meant returning with the previous occupant's cnt/ev/enc and
    # committed_t still in place. Nothing read them -- the read clamps nq to 0 while seq <=
    # prefix, and the first decode token re-triggers the kernel's own reset -- but that is two
    # unrelated clamps holding up a stale state, not a guarantee.
    if p.numel() == 0:
        return          # nothing addressed to this slot in this call: touch no state
    p_all = p
    cnt = int(st.cnt[s])
    # `committed_t` is the READ path's split point and is shared by every layer, while this
    # function runs once PER LAYER. Only layer 0 may touch it: layer 0 adds what this call
    # commits, and layers 1..N-1 would otherwise clear that contribution again (each of them
    # sees its own cnt == 0 on a request's first call). The bug made prefill contribute
    # nothing -- committed_t stayed 0, so the raw tail was the whole prompt, and with the
    # tier being a ring the tail wrapped onto itself: two positions sharing one row, which
    # is exactly the degenerate repeated-token output (gsm8k 8-shot 0/4, 2026-09-10).
    _first = layer_idx == 0
    if cnt > 0:
        last = int(st.pos[s, (int(st.start[s]) + cnt - 1) % cap])
        if int(p_all[0]) != last + 1:
            cnt = 0
            st.cnt[s] = 0
            st.ev[s] = 0
            st.enc[s] = False
            if _first:
                ps.committed_t[s] = 0
    else:
        st.ev[s] = 0
        st.enc[s] = False
        if _first:
            ps.committed_t[s] = 0
    if _PREFIX_TOKENS > 0:
        # sink rows never enter a window; they stay raw in the tier's fixed prefix rows.
        # src_idx indexes THIS CALL's K/V and must be filtered with them -- dropping only
        # p/l left the window encode reading rows shifted by the number of sink tokens
        # (caught by test_fused_packed_batch at PREFIX=64: 640 rows wrong per request).
        m = p >= _PREFIX_TOKENS
        p, l = p[m], l[m]
        if src_idx is not None:
            src_idx = src_idx[m]
    n = p.numel()
    if n == 0:
        return
    start = int(st.start[s])
    ev = int(st.ev[s])
    sl = s.reshape(1) if torch.is_tensor(s) else torch.tensor([s], device=dev)
    trash_w = ls.nsc.shape[0] - 1
    committed = 0

    def _win(k0, nwin):
        ridx = (start + torch.arange(k0, k0 + nwin * ws, device=dev)) % cap
        return (st.loc[s, ridx].reshape(nwin, ws), st.pos[s, ridx].reshape(nwin, ws))

    p0 = int(p[0]) if (src_idx is not None and p.numel()) else None

    def _src(wp, wl):
        """(K, V, rows) to encode the window (positions wp, pool slots wl) from."""
        if p0 is not None:
            off = wp - p0
            if bool(((off >= 0) & (off < src_idx.numel())).all()):
                return src_k, src_v, src_idx[off]
        return k_buffer, v_buffer, _hrow(wl)

    def _wid(pos0):
        return comp * MAXW + (pos0.clamp_min(0) // ws).clamp_max(MAXW - 1)

    step = max(ws, ((cap - 2 * ws - _RECENT_TOKENS) // ws) * ws)
    encoded = False   # st.ck/st.cv hold the current head window (see (c))
    for c0 in range(0, n, step):
        encoded = False
        c1 = min(c0 + step, n)
        idx = (start + cnt + torch.arange(c1 - c0, device=dev)) % cap
        st.loc[s, idx] = l[c0:c1]
        st.pos[s, idx] = p[c0:c1]
        cnt += c1 - c0
        # (a) finish a partially evicted window (encoded earlier, ev rows already out)
        if ev > 0:
            nout = min(((cnt - _RECENT_TOKENS - ev) // gran) * gran, ws - ev)
            if nout > 0:
                wl, wp = _win(0, 1)
                sel = torch.arange(ev, ev + nout, device=dev)[None, :]
                _EK.commit_into(ls, st.ck, st.cv, sl, sel, wl.gather(1, sel).reshape(-1),
                                _wid(wp[:, 0]))
                if _PS.KEEP_BF16:
                    _write_recon_rows(st, sl, sel, wl.gather(1, sel), wp.gather(1, sel), bundle,
                                      k_buffer, v_buffer, st.ck["nsc"].dtype)
                committed += nout
                ev += nout
                if ev >= ws:
                    start = (start + ws) % cap
                    cnt -= ws
                    ev = 0
        # (b) whole windows past RECENT, one batched encode + one commit
        if ev == 0:
            nfull = (cnt - _RECENT_TOKENS) // ws
            while nfull >= 1:
                nb = min(nfull, NFULL_PAD)
                ml, mp = _win(0, nb)
                fk = _EK.scratch_cache(nb, st.H, ws, st.D, st.ck["nsc"].dtype, dev)
                fv = _EK.scratch_cache(nb + 1, st.H, ws, st.D, st.ck["nsc"].dtype, dev)
                adst = torch.arange(nb, device=dev)
                sk_, sv_, mrow = _src(mp, ml)
                _EK.encode_kv_windows_into(sk_, sv_, mrow, mp.clamp_min(0), inv_freq, cb, fk, fv,
        adst, not _V_HADAMARD_FOLDED)
                sel_all = torch.arange(ws, device=dev)[None, :].expand(nb, ws)
                _EK.commit_into(ls, fk, fv, adst, sel_all, ml.reshape(-1), _wid(mp[:, 0]))
                if _PS.KEEP_BF16:
                    rk, rv = _transform_batched(sk_[mrow], sv_[mrow], mp.clamp_min(0),
                                                bundle, st.ck["nsc"].dtype)
                    dr = _hrow(ml.reshape(-1))
                    k_buffer.index_copy_(0, dr, rk.reshape(-1, *rk.shape[2:]).to(k_buffer.dtype))
                    v_buffer.index_copy_(0, dr, rv.reshape(-1, *rv.shape[2:]).to(v_buffer.dtype))
                committed += nb * ws
                start = (start + nb * ws) % cap
                cnt -= nb * ws
                nfull -= nb
        # (c) a partial window: encode once into the slot cache, commit what has aged out.
        # The encode runs as soon as the head window is complete, even when nothing has aged out
        # yet (nout == 0). Encoding only on nout > 0 left a complete-but-unencoded window behind
        # when the prefill ended with cnt - RECENT < gran: decode's `due` requires `enc`, so that
        # request's first commit waited for an urgency pick and ran one step late, holding one
        # extra granule of tokens in bf16 for that step.
        if ev == 0:
            nout = min(((cnt - _RECENT_TOKENS) // gran) * gran, ws)
            if cnt >= ws:
                wl, wp = _win(0, 1)
                sk_, sv_, wr = _src(wp, wl)
                _EK.encode_kv_windows_into(sk_, sv_, wr, wp.clamp_min(0), inv_freq, cb, st.ck, st.cv,
        sl, not _V_HADAMARD_FOLDED)
                encoded = True
                if nout > 0:
                    sel = torch.arange(0, nout, device=dev)[None, :]
                    _EK.commit_into(ls, st.ck, st.cv, sl, sel, wl.gather(1, sel).reshape(-1),
                                    _wid(wp[:, 0]))
                    if _PS.KEEP_BF16:
                        _write_recon_rows(st, sl, sel, wl.gather(1, sel), wp.gather(1, sel), bundle,
                                          k_buffer, v_buffer, st.ck["nsc"].dtype)
                    committed += nout
                    ev = nout
                    if ev >= ws:
                        start = (start + ws) % cap
                        cnt -= ws
                        ev = 0
                        encoded = False
    st.start[s] = start
    st.cnt[s] = cnt
    st.ev[s] = ev
    st.enc[s] = ev > 0 or encoded
    if layer_idx == 0 and committed:
        ps.committed_t[s] += committed
@torch.no_grad()
def _flush_once(st, slots, bundle, k_buffer, v_buffer, dtype, layer_idx=-1):
    """Encode the current window if it is newly due, then evict EVICT_GRAN rows from it.

    Shape-static throughout, so it stays CUDA-graph capturable: the encode runs every step for
    every slot and is committed only where `fresh`; the decode always produces GRAN rows and is
    written only where `due`, everything else routed to the reserved trash row.

    Cost against the previous whole-window flush: the encode is the same work as the old
    _transform_batched call, and the decode is GRAN/ws = 1/8 of one, so ~1.1x. The 64x redundancy
    (evaluating a window per slot per step when few are due) remains because the flush is captured.
    """
    ws = bundle["window_size"]
    due, fresh, win_loc, win_pos = _flush_candidates(st, slots, ws)
    cb = bundle["codebook"]
    arith = torch.promote_types(cb.dtype, dtype)

    # ---- encode at most NENC windows per step, most urgent first.
    #
    # Encode only the most urgent complete windows. A completed window has additional steps before
    # eviction, so inactive slots do not need to be encoded on every step.
    #
    # Select complete, unencoded windows by urgency. Batch order may change between calls, so a
    # positional cursor would not provide fair progress across request slots.
    #
    # `enc` gates eviction, so a window that somehow missed its turn waits rather than being
    # written from a cache that does not hold it.
    B = slots.shape[0]
    slack = _RECENT_TOKENS - ws + _gran(ws) - 1   # see _decode_step_kernels
    NENC = B if slack <= 0 else min(B, max(1, -(-B // slack)))
    urgency = st._urgency
    want_all = urgency >= 0
    pick = urgency.topk(NENC).indices                       # [NENC]
    psl = slots[pick]
    want = want_all[pick]
    ploc = torch.where(want[:, None], win_loc[pick],
                       torch.full_like(win_loc[pick], TRASH_LOC))
    prow = _hrow(ploc)
    if _EK.ENABLED:
        # fused Triton encoder (nsn_encode_kernel): 6 launches for K and V, written straight
        # into the per-slot cache; unwanted windows land on the scratch slot like _cache_store
        cdst = torch.where(want, psl, torch.full_like(psl, CACHE_SCRATCH))
        ppos = win_pos[pick].clamp_min(0)
        _EK.encode_kv_windows_into(k_buffer, v_buffer, prow, ppos, bundle["inv_freq"], cb, st.ck, st.cv,
        cdst, not _V_HADAMARD_FOLDED)
    else:
        cos, sin = _rope_cos_sin(win_pos[pick].clamp_min(0), bundle["inv_freq"])
        cos, sin = cos.to(dtype), sin.to(dtype)
        pk = encode_window_k(k_buffer[prow].permute(0, 2, 1, 3), cos, sin, cb, ws)
        pv = encode_window_v(v_buffer[prow].permute(0, 2, 1, 3), cb, ws, _V_HADAMARD_FOLDED)
        _cache_store(st.ck, psl, pk, want)
        _cache_store(st.cv, psl, pv, want)
    st.enc[psl] = st.enc[psl] | want

    # ---- decode and write just the rows leaving the raw tail now
    sel = (st.ev[slots][:, None]
           + torch.arange(_gran(ws), device=slots.device)[None, :]).clamp_max(ws - 1)
    loc_sel = win_loc.gather(1, sel)
    dst = torch.where(due[:, None], loc_sel,
                      torch.full_like(loc_sel, TRASH_LOC)).reshape(-1)

    if _PS.ENABLED:
        # SGLANG_NSN_PACKED: this file already HOLDS the packed form on device (st.ck/st.cv);
        # the bf16 decode below only exists to turn it back into bf16 for the pool. Scatter the
        # packed fields into the store instead -- same shape-static form, same TRASH routing,
        # so the flush stays CUDA-graph capturable.
        ps, ls = _packed_layer(st, layer_idx, ws, slots.device, st.ck["nsc"].dtype)
        wid = _wid_for(ps, ls, slots, win_pos[:, 0], ws, due)
        if _EK.ENABLED:
            _EK.commit_into(ls, st.ck, st.cv, slots, sel, dst, wid)
        else:
            _commit_packed(ls, _slot_fields(st.ck, slots), _slot_fields(st.cv, slots),
                           sel, dst, wid, slots)
        if layer_idx == 0:
            ps.committed_t[slots] += due.to(ps.committed_t.dtype) * sel.shape[1]
        if not _PS.KEEP_BF16:
            _advance(st, slots, due, ws)
            return due

    cs, sn = _rope_cos_sin(win_pos.gather(1, sel).clamp_min(0), bundle["inv_freq"])
    K, V = st.ck, st.cv
    rk, rv = _decode_kv_c(
        K["idx"][slots], K["norm2"][slots], K["nq"][slots], K["nsc"][slots], K["nmin"][slots],
        K["mq"][slots], K["msc"][slots], K["mmin"][slots],
        V["idx"][slots], V["norm2"][slots], V["nq"][slots], V["nsc"][slots], V["nmin"][slots],
        V["mq"][slots], V["msc"][slots], V["mmin"][slots],
        sel, cs.to(dtype), sn.to(dtype), cb, st.D, dtype, _V_HADAMARD_FOLDED, arith, ws)

    drow = _hrow(dst)
    k_buffer.index_copy_(0, drow, rk.reshape(-1, *rk.shape[2:]).to(k_buffer.dtype))
    v_buffer.index_copy_(0, drow, rv.reshape(-1, *rv.shape[2:]).to(v_buffer.dtype))
    _advance(st, slots, due, ws)
    return due


@torch.no_grad()
def apply_nsn_quant_fused(
    path: str,
    layer_idx: int,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    positions: torch.Tensor,
    req_pool_indices: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    is_decode: Optional[bool] = None,
):
    """Drop-in replacement for nsn_quant.apply_nsn_quant (same contract). The caller writes
    the returned (k, v) for the current rows; window flushes overwrite older pool rows here.

    Decode (T == number of distinct slots, one row each; the captured path): fully tensor ops.
    Prefill (multi-row slots; eager only): per-slot serial append via a host loop over the
    (bounded) number of rows, then a drain loop -- no .tolist(), no per-request dicts.
    """
    b = load_bundle(path, cache_k.device)
    ws = b["window_size"]
    # H/D from the pool view, so the hook can allocate its own state when nothing prewarmed it
    st = _get_state(path, layer_idx, ws, cache_k.device,
                    k_buffer.shape[1], k_buffer.shape[2], cache_k.dtype)
    T = cache_k.shape[0]
    dtype = cache_k.dtype

    # Flushes below read raw window rows back FROM THE POOL (the reference reads them from its
    # host-side accumulator clones instead). Rows of the CURRENT call are not in the pool yet --
    # the caller writes them after we return -- so a window that includes any current row (every
    # prefill flush; decode flushes when RECENT == 0) would otherwise read stale pool contents.
    # Pre-write the raw rows now; the caller's later write of the returned rows is value-
    # identical (raw for untouched rows, reconstruction re-read below for flushed ones).
    _lrow = _hrow(loc.long())
    k_buffer.index_copy_(0, _lrow, cache_k.to(k_buffer.dtype))
    v_buffer.index_copy_(0, _lrow, cache_v.to(v_buffer.dtype))

    keep = positions >= _PREFIX_TOKENS if _PREFIX_TOKENS > 0 else None

    capturing = torch.cuda.is_current_stream_capturing()
    # Decode fast path. The caller knows which forward mode it is in -- forward_decode versus
    # forward_extend -- so take it from there and keep the branch IDENTICAL between the graph
    # runner's pre-capture warmup and the capture itself. Deriving it from the data instead is
    # what made this path capture-incompatible; see _rows_unique.
    is_decode = _rows_unique(req_pool_indices, capturing) if is_decode is None else is_decode
    if is_decode:
        slots, pos, lc = req_pool_indices.long(), positions.long(), loc.long()
        # CUDA-graph PADDING rows must not reach real slots: when a decode batch is padded to
        # a capture bucket (bs != raw_bs), the graph runner zeroes out_cache_loc and refreshes
        # only the first raw_bs entries of req_pool_indices/positions -- the padded tail keeps
        # STALE values from an earlier step (a real slot + an old position). `loc == TRASH_LOC`
        # identifies padding because row 0 is reserved. Route padding and prefix rows to the
        # scratch slot so they cannot modify request state.
        # A slot index past the ring would scribble outside it, silently. MAX_SLOTS is a
        # compile-time bound (env SGLANG_NSN_FUSED_MAX_SLOTS), not read from the pool -- the
        # plain MHATokenToKVPool carries no request-slot count -- so check rather than assume.
        # Skipped under capture: the check syncs, and capture replays fixed slot ids anyway.
        if not capturing:
            _mx = int(req_pool_indices.max()) if req_pool_indices.numel() else 0
            if _mx >= MAX_SLOTS - 1:
                raise RuntimeError(
                    f"nsn_quant_fused: request slot {_mx} >= MAX_SLOTS-1 ({MAX_SLOTS - 1}); the "
                    f"last slot is reserved as scratch. Raise SGLANG_NSN_FUSED_MAX_SLOTS."
                )
        if _PS.ENABLED and _EK.ENABLED:
            # drop-routing, append, candidates, encode picks, commit and advance -- fused
            _decode_step_kernels(st, req_pool_indices.long(), positions.long(), loc.long(), b,
                                 cache_k, cache_v, k_buffer, v_buffer, layer_idx)
            return cache_k, cache_v
        drop = lc == TRASH_LOC
        if keep is not None:
            drop = drop | ~keep
        scratch = torch.full_like(slots, MAX_SLOTS - 1)
        slots = torch.where(drop, scratch, slots)
        pos = torch.where(drop, torch.full_like(pos, -(10**9)), pos)
        lc = torch.where(drop, torch.full_like(lc, TRASH_LOC), lc)
        _append(st, slots, pos, lc, first_layer=layer_idx == 0)
        _flush_once(st, slots, b, k_buffer, v_buffer, dtype, layer_idx)
        # NOTE on the returned rows: with RECENT_TOKENS > 0 a just-appended row is never part
        # of the flushed window (it sits in the recent tail), so returning the raw row matches
        # the reference. With RECENT == 0 the row that completes a window IS flushed; the pool
        # gets the reconstruction via index_copy_ above, and the caller's subsequent write of
        # the raw row would clobber it -- so re-read the (possibly reconstructed) rows back.
        if _RECENT_TOKENS == 0:
            return k_buffer[_lrow].clone(), v_buffer[_lrow].clone()
        return cache_k, cache_v

    # ---- prefill (eager) ----
    if _DEBUG_AUDIT and layer_idx == 0:
        _audit(st, layer_idx, "prefill-entry")
    slots_all, pos_all, loc_all = req_pool_indices.long(), positions.long(), loc.long()
    if _PS.ENABLED and _EK.ENABLED:
        m = loc_all != TRASH_LOC
        call_rows = torch.nonzero(m).flatten()   # row of cache_k/cache_v per kept token
        _sk, _sv = cache_k.contiguous(), cache_v.contiguous()
        slots_all, pos_all, loc_all = slots_all[m], pos_all[m], loc_all[m]
        for s_ in torch.unique(slots_all).tolist():
            m2 = slots_all == s_
            _prefill_packed(st, s_, pos_all[m2], loc_all[m2], b, k_buffer, v_buffer,
                            layer_idx, src_k=_sk, src_v=_sv, src_idx=call_rows[m2])
        if _PS.KEEP_BF16:                # audit mode: hand back the reconstruction where it exists
            return k_buffer[_lrow].to(dtype).clone(), v_buffer[_lrow].to(dtype).clone()
        return cache_k, cache_v          # rows stay raw in the bf16 tier (pre-written above)
    # Exclude padding and prefix rows before inserting positions into the ring.
    m = loc_all != TRASH_LOC
    if keep is not None:
        m = m & keep
    slots_all, pos_all, loc_all = slots_all[m], pos_all[m], loc_all[m]
    if slots_all.numel():
        uniq = [int(x) for x in torch.unique(slots_all).tolist()]
        rows = {}
        for s in uniq:
            m2 = slots_all == s
            rows[s] = (pos_all[m2], loc_all[m2])

        # Discontinuity check, once per slot. This path has its own (the decode path's lives in
        # _append), and it must clear `ev` with `cnt`: the abandoned window's eviction count
        # would otherwise carry into the next one.
        cap = st.cap
        for s in uniq:
            p, _l = rows[s]
            has = st.cnt[s] > 0
            last_idx = (st.start[s] + st.cnt[s] - 1) % cap
            expected = st.pos[s, last_idx] + 1 if bool(has) else int(p[0])
            if int(p[0]) != int(expected):
                st.cnt[s] = 0
                st.ev[s] = 0
                if (layer_idx == 0 and _PS.ENABLED
                        and _PS.store().committed_t is not None):
                    _PS.store().committed_t[s] = 0   # shared across layers; see _prefill_packed

        # Append in bounded chunks to avoid ring wrap, and drain across slots to batch transforms.
        maxn = max(rows[s][0].numel() for s in uniq)
        # Keep enough ring capacity for the window currently being drained.
        step = max(ws, ((st.cap - 2 * ws - _RECENT_TOKENS) // ws) * ws)
        for c0 in range(0, maxn, step):
            active = [s for s in uniq if rows[s][0].numel() > c0]
            for s in active:
                p, l = rows[s]
                c1 = min(c0 + step, p.numel())
                idx = (st.start[s] + st.cnt[s]
                       + torch.arange(c1 - c0, device=p.device)) % cap
                st.loc[s, idx] = l[c0:c1]
                st.pos[s, idx] = p[c0:c1]
                st.cnt[s] = st.cnt[s] + (c1 - c0)
            act = torch.tensor(active, device=cache_k.device, dtype=torch.int64)
            _drained = 0
            while True:
                # require_enc=False, like _flush_eager itself: the eager path encodes inline, so
                # gating the LOOP on a flag only the decode path maintains ended the drain early
                # and left prompt rows raw that the reference had quantised.
                due, _fresh, _wl, _wp = _flush_candidates(st, act, ws, require_enc=False,
                                                          compiled=False)
                if not bool(due.any()):
                    break
                _flush_eager(st, act, b, k_buffer, v_buffer, dtype, layer_idx)
                _drained += 1
    # The pool now holds exactly what each current row should return: the reconstruction for
    # rows inside a flushed window, the pre-written raw value otherwise.
    return k_buffer[_lrow].to(dtype).clone(), v_buffer[_lrow].to(dtype).clone()


def _rows_unique(req_pool_indices: torch.Tensor, capturing: bool) -> bool:
    """Fallback only. Prefer the caller's `is_decode`, which is what the attention backend
    already knows.

    This predicate is why NSN alone needed explicit pre-capture warming. It short-circuits to
    True while capturing and genuinely checks otherwise, and the graph runner's pre-capture
    warmup uses a dummy batch whose req_pool_indices are all ZEROS
    (cuda_graph_runner.py:166) -- so for bs > 1 the warmup took a different branch from the
    capture, the decode kernels were never launched outside a capture, and Triton's binary
    load landed inside one. Every other arm runs the same kernels in warmup and capture and so
    needs nothing. With `is_decode` passed down, so does NSN.
    """
    if capturing:
        return True  # capture uses decode-shaped dummy batches; uniqueness holds by contract
    return req_pool_indices.unique().numel() == req_pool_indices.numel()
