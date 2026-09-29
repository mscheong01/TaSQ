"""Triton glue for the two-tier packed read (nsn_packed_read.two_tier_decode).

The read's index building (`_tier_indices` twice, the hp-row lookup, the window-arena map,
the split point) and the LSE merge were ~55 torch launches per layer per decode step -- with
the attention kernels themselves being only a few launches. Two kernels replace them:
``read_indices`` (one program with static shapes) and ``merge``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _read_index(R2T, REQ, SEQ, COMMITTED, MAP, COMPACT,
                P_IND, P_PTR, R_IND, R_PTR, NQ, WMAP,
                B, r2t_stride, maxw, W, prefix,
                BLOCK_B: tl.constexpr, BLOCK_W: tl.constexpr, WS: tl.constexpr,
                NW: tl.constexpr, TRASH_ROW: tl.constexpr):
    b = tl.arange(0, BLOCK_B)
    mb = b < B
    req = tl.load(REQ + b, mask=mb, other=0).to(tl.int64)
    seq = tl.load(SEQ + b, mask=mb, other=0).to(tl.int32)
    # packed span = positions [prefix, prefix + nq); raw = [0, prefix) and [prefix + nq, seq)
    nq = tl.minimum(tl.load(COMMITTED + req, mask=mb, other=0).to(tl.int32),
                    tl.maximum(seq - prefix, 0))
    nq = tl.where(mb, nq, 0)
    seq = tl.where(mb, seq, 0)
    ln_r = seq - nq
    # compact (ragged) layouts, as the attention kernels expect: indptr = exclusive prefix sums
    cp = tl.cumsum(nq, axis=0)
    cr = tl.cumsum(ln_r, axis=0)
    p0 = cp - nq
    r0 = cr - ln_r
    tl.store(P_PTR + b + 1, cp, mask=mb)
    tl.store(R_PTR + b + 1, cr, mask=mb)
    tl.store(P_PTR, 0)
    tl.store(R_PTR, 0)
    tl.store(NQ + b, nq, mask=mb)
    comp = tl.load(COMPACT + req, mask=mb, other=0).to(tl.int64)
    w = tl.arange(0, NW)
    tl.store(WMAP + b[:, None] * NW + w[None, :], (comp[:, None] * maxw + w[None, :]).to(tl.int32),
             mask=mb[:, None])
    for c in range(0, W, BLOCK_W):
        j = c + tl.arange(0, BLOCK_W)
        inr = mb[:, None] & (j[None, :] < seq[:, None])
        slot = tl.load(R2T + req[:, None] * r2t_stride + j[None, :], mask=inr, other=0).to(tl.int64)
        pk_end = prefix + nq[:, None]
        mp = inr & (j[None, :] >= prefix) & (j[None, :] < pk_end)
        tl.store(P_IND + p0[:, None] + (j[None, :] - prefix), slot.to(tl.int32), mask=mp)
        mr = inr & ((j[None, :] < prefix) | (j[None, :] >= pk_end))
        row = tl.load(MAP + slot, mask=mr, other=TRASH_ROW)
        ridx = j[None, :] - tl.where(j[None, :] >= pk_end, nq[:, None], 0)
        tl.store(R_IND + r0[:, None] + ridx, row.to(tl.int32), mask=mr)


@triton.jit
def _merge(OPK, LPK, ORAW, LRAW, NQ, OUT, H: tl.constexpr, D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    nq = tl.load(NQ + b)
    l1 = tl.load(LPK + b * H + h)
    l1 = tl.where(nq > 0, l1, float("-inf"))
    l2 = tl.load(LRAW + b * H + h)
    m = tl.maximum(l1, l2)
    # Both tiers empty -> m is -inf and exp(-inf - -inf) is NaN, so the output would be NaN
    # rather than the zeros every other reduction in this stack produces. Unreachable in the
    # served path (RAW_FOLD=1 does not call this, and the raw tier always holds the current
    # token) but this is the one reduction here without the guard, so give it one.
    empty = m == float("-inf")
    m = tl.where(empty, 0.0, m)
    w1 = tl.where(empty, 0.0, tl.exp(l1 - m))
    w2 = tl.where(empty, 0.0, tl.exp(l2 - m))
    o1 = tl.load(OPK + (b * H + h) * D + d)
    o2 = tl.load(ORAW + (b * H + h) * D + d).to(tl.float32)
    den = w1 + w2
    out = tl.where(den > 0, (o1 * w1 + o2 * w2) / tl.where(den > 0, den, 1.0), 0.0)
    tl.store(OUT + (b * H + h) * D + d, out.to(OUT.dtype.element_ty))


_BUF: dict = {}


def _bufs(B, W, nw, device, tag=""):
    key = (B, W, nw, device, tag)
    if key not in _BUF:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"nsn_read_glue._bufs: first allocation of {key} inside CUDA-graph capture; "
                "memory_pool prewarming must cover this (B, W, tag).")
        i32 = torch.int32
        _BUF[key] = dict(
            p_ind=torch.zeros(B * W + 1, dtype=i32, device=device),
            r_ind=torch.zeros(B * W + 1, dtype=i32, device=device),
            p_ptr=torch.zeros(B + 1, dtype=i32, device=device),
            r_ptr=torch.zeros(B + 1, dtype=i32, device=device),
            nq=torch.zeros(B, dtype=i32, device=device),
            # row stride = the kernel's power-of-two NW (nsn_fused reads shape[1] as the stride)
            wmap=torch.zeros(B, triton.next_power_of_2(nw), dtype=i32, device=device),
            mean_idx=torch.zeros(B, dtype=i32, device=device),
        )
    return _BUF[key]


@torch.no_grad()
def read_indices(r2t, req_idx, seq, committed_t, hp_map, compact, W, ws, maxw, trash_row,
                 prefix=0, tag=""):
    """Build compact indices, tier boundaries, and the window map in one launch.

    ``W`` is the static context bound and ``maxw`` is the per-request arena capacity. ``tag``
    selects a separate buffer set for asynchronous audit reads.
    """
    B = req_idx.numel()
    nw = min((W + ws - 1) // ws, maxw)
    bf = _bufs(B, W, nw, req_idx.device, tag)
    _read_index[(1,)](r2t, req_idx, seq, committed_t, hp_map, compact,
                      bf["p_ind"], bf["p_ptr"], bf["r_ind"], bf["r_ptr"], bf["nq"], bf["wmap"],
                      B, r2t.stride(0), maxw, W, prefix,
                      BLOCK_B=max(2, triton.next_power_of_2(B)), BLOCK_W=256, WS=ws,
                      NW=triton.next_power_of_2(nw), TRASH_ROW=trash_row, num_warps=4)
    return bf, nw


@torch.no_grad()
def merge_into(o, o_pk, lse_pk, o_raw, lse_raw, nq):
    B, H, D = o.shape
    _merge[(B, H)](o_pk, lse_pk, o_raw, lse_raw, nq, o, H=H, D=D, num_warps=1)
    return o
