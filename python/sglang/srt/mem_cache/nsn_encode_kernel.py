"""Fused NSN window encoder for the served write path (Triton).

``nsn_pack.encode_window_{k,v}`` uses many small torch operations per window. This module
implements the same computation as three kernels per side:

    stage1  per (window, head): un-RoPE (K), row norm -> RTN4(ws) -> divide, column mean ->
            RTN4(32) -> subtract, row norm2 -> divide. Writes the normalised tile (scratch) and
            every metadata field straight into the fused state's cache layout.
    stage2  per (window, head): re-RoPE (K), Hadamard (bf16 tl.dot, fp32 accumulate).
    stage3  per 32 vectors: nearest centroid (fp32 IEEE dot, argmax), scale-adjusted norm2.

Elementwise operations follow the torch path's BF16 rounding; reductions and VQ search accumulate
in fp32. FMA contraction is disabled to preserve intermediate BF16 rounding. Reduction order may
still differ at quantization boundaries. ``SGLANG_NSN_TRITON_ENCODE=0`` restores the torch path.
"""
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ENABLED = os.environ.get("SGLANG_NSN_TRITON_ENCODE", "1") == "1"
NUM_WARPS = int(os.environ.get("SGLANG_NSN_ENC_WARPS", "16"))
S3_WARPS = int(os.environ.get("SGLANG_NSN_ENC_S3_WARPS", "4"))
S3_RB = int(os.environ.get("SGLANG_NSN_ENC_S3_RB", "2"))
RTN4_LEVELS = 15.0


@triton.jit
def _bf(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rtn4_scale_min(v, LEVELS: tl.constexpr):
    # torch: scale = clamp((amax - amin) / 15, min=1e-5), each op rounding to bf16
    mx = tl.max(v, axis=0)
    mn = tl.min(v, axis=0)
    sc = _bf(_bf(mx - mn) / LEVELS)
    sc = _bf(tl.maximum(sc, 1e-5))
    return sc, mn


@triton.jit
def _rtn4_code(v, sc, mn, LEVELS: tl.constexpr):
    # torch: ((flat - min) / scale).clamp_(0, 15).round_()   (round half to even)
    q = _bf(_bf(v - mn) / sc)
    q = tl.minimum(tl.maximum(q, 0.0), LEVELS)
    return libdevice.rint(q)


@triton.jit
def _nsn_stage1(XK, XV, ROWS, POS, INVF, XN, N2P, DST, NWH,
                K_NQ, K_NSC, K_NMIN, K_MQ, K_MSC, K_MMIN,
                V_NQ, V_NSC, V_NMIN, V_MQ, V_MSC, V_MMIN,
                H: tl.constexpr, WS: tl.constexpr, D: tl.constexpr, MG: tl.constexpr,
                LEVELS: tl.constexpr):
    # one launch for both sides: programs [0, nw*H) are K, [nw*H, 2*nw*H) are V
    side = tl.program_id(0) // NWH
    wh = tl.program_id(0) % NWH
    w = wh // H
    h = wh % H
    r = tl.arange(0, WS)
    c = tl.arange(0, D)
    HD: tl.constexpr = D // 2
    DG: tl.constexpr = D // MG
    rows = tl.load(ROWS + w * WS + r).to(tl.int64)
    off = rows[:, None] * (H * D) + h * D + c[None, :]
    if side == 0:
        x = tl.load(XK + off).to(tl.float32)
    else:
        x = tl.load(XV + off).to(tl.float32)
    if side == 0:
        # rotate_half(x)[c] = c < D/2 ? -x[c + D/2] : x[c - D/2]
        cp = (c + HD) % D
        sgn = tl.where(c < HD, -1.0, 1.0)
        xr = tl.load(XK + rows[:, None] * (H * D) + h * D + cp[None, :]).to(tl.float32) * sgn[None, :]
        pos = tl.load(POS + w * WS + r).to(tl.float32)
        invf = tl.load(INVF + (c % HD))
        f = pos[:, None] * invf[None, :]
        cs = _bf(libdevice.cos(f))
        sn = _bf(libdevice.sin(f))
        # caller passes -sin for the un-rotation; negation is exact
        x = _bf(_bf(x * cs) + _bf(xr * (-sn)))
    # row norm / sqrt(D), RTN4 over the ws norms of this head
    SQD: tl.constexpr = D ** 0.5   # NSN normalises by sqrt(head_dim); folded at compile time
    n = _bf(_bf(tl.sqrt(tl.sum(x * x, axis=1))) / SQD)
    sc, mn = _rtn4_scale_min(n, LEVELS)
    q = _rtn4_code(n, sc, mn, LEVELS)
    nd = _bf(_bf(q * sc) + mn)
    x = _bf(x / nd[:, None])
    # column mean over the window, RTN4 in groups of MG channels
    m = _bf(tl.sum(x, axis=0) / WS)                       # [D]
    m2 = tl.reshape(m, [DG, MG])
    gmx = tl.max(m2, axis=1)
    gmn = tl.min(m2, axis=1)
    gsc = _bf(_bf(gmx - gmn) / LEVELS)
    gsc = _bf(tl.maximum(gsc, 1e-5))
    qm = _bf(_bf(m2 - gmn[:, None]) / gsc[:, None])
    qm = libdevice.rint(tl.minimum(tl.maximum(qm, 0.0), LEVELS))
    md2 = _bf(_bf(qm * gsc[:, None]) + gmn[:, None])
    md = tl.reshape(md2, [D])
    qmf = tl.reshape(qm, [D])
    x = _bf(x - md[None, :])
    n2 = _bf(_bf(tl.sqrt(tl.sum(x * x, axis=1))) / SQD)
    x = _bf(x / n2[:, None])
    # scratch (side-major): normalised tile and pre-adjust norm2
    swh = side * NWH + wh
    tl.store(XN + (swh * WS + r[:, None]) * D + c[None, :], x.to(tl.bfloat16))
    tl.store(N2P + swh * WS + r, n2)
    # metadata straight into the cache layout at slot dst
    dst = tl.load(DST + w).to(tl.int64)
    g = tl.arange(0, DG)
    if side == 0:
        tl.store(K_NQ + (dst * H + h) * WS + r, q.to(tl.uint8))
        tl.store(K_NSC + dst * H + h, sc.to(tl.bfloat16))
        tl.store(K_NMIN + dst * H + h, mn.to(tl.bfloat16))
        tl.store(K_MQ + (dst * H + h) * D + c, qmf.to(tl.uint8))
        tl.store(K_MSC + (dst * H + h) * DG + g, gsc.to(tl.bfloat16))
        tl.store(K_MMIN + (dst * H + h) * DG + g, gmn.to(tl.bfloat16))
    else:
        tl.store(V_NQ + (dst * H + h) * WS + r, q.to(tl.uint8))
        tl.store(V_NSC + dst * H + h, sc.to(tl.bfloat16))
        tl.store(V_NMIN + dst * H + h, mn.to(tl.bfloat16))
        tl.store(V_MQ + (dst * H + h) * D + c, qmf.to(tl.uint8))
        tl.store(V_MSC + (dst * H + h) * DG + g, gsc.to(tl.bfloat16))
        tl.store(V_MMIN + (dst * H + h) * DG + g, gmn.to(tl.bfloat16))


@triton.jit
def _nsn_stage2(XN, POS, INVF, HAD, XH, NWH,
                H: tl.constexpr, WS: tl.constexpr, D: tl.constexpr, HAD_V: tl.constexpr):
    side = tl.program_id(0) // NWH
    wh = tl.program_id(0) % NWH
    w = wh // H
    r = tl.arange(0, WS)
    c = tl.arange(0, D)
    HD: tl.constexpr = D // 2
    swh = side * NWH + wh
    base = XN + (swh * WS + r[:, None]) * D
    x = tl.load(base + c[None, :]).to(tl.float32)
    if side == 0:
        cp = (c + HD) % D
        sgn = tl.where(c < HD, -1.0, 1.0)
        xr = tl.load(base + cp[None, :]).to(tl.float32) * sgn[None, :]
        pos = tl.load(POS + w * WS + r).to(tl.float32)
        invf = tl.load(INVF + (c % HD))
        f = pos[:, None] * invf[None, :]
        cs = _bf(libdevice.cos(f))
        sn = _bf(libdevice.sin(f))
        x = _bf(_bf(x * cs) + _bf(xr * sn))
    if (side == 0) | HAD_V:
        k = tl.arange(0, D)
        hm = tl.load(HAD + k[:, None] * D + c[None, :])          # [D, D] bf16 (symmetric)
        x = tl.dot(x.to(tl.bfloat16), hm).to(tl.float32)        # bf16 GEMM, fp32 acc, bf16 out
        x = _bf(x)
    tl.store(XH + (swh * WS + r[:, None]) * D + c[None, :], x.to(tl.bfloat16))


@triton.jit
def _nsn_stage3(XH, N2P, CBP, CB, DST, K_IDX, K_N2, V_IDX, V_N2, NWH,
                H: tl.constexpr, WS: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
                KC: tl.constexpr, RB: tl.constexpr):
    # one program: RB rows of one (side, window, head) = RB * D/G vectors
    NV: tl.constexpr = RB * (D // G)
    pid = tl.program_id(0)
    PPW: tl.constexpr = WS // RB                 # programs per (window, head)
    swh = pid // PPW
    rb = pid % PPW
    side = swh // NWH
    wh = swh % NWH
    w = wh // H
    h = wh % H
    v = tl.arange(0, NV)
    kk = tl.arange(0, 16)                        # G padded to tl.dot's minimum K
    kmask = kk < G
    row0 = rb * RB
    tile = XH + (swh * WS + row0) * D             # contiguous RB x D = NV x G
    x = tl.load(tile + v[:, None] * G + kk[None, :], mask=kmask[None, :], other=0.0).to(tl.float32)
    cc = tl.arange(0, KC)
    cbp = tl.load(CBP + kk[:, None] * KC + cc[None, :])        # [16, KC] fp32, rows >= G are 0
    sc = tl.dot(x, cbp, input_precision="ieee")                # x . c
    cbsq = tl.load(CBP + 16 * KC + cc)                          # 0.5*|c|^2 appended as row 16
    sc = sc - cbsq[None, :]
    idx = tl.argmax(sc, axis=1)                                # [NV]
    # scale adjust: num = sum(bf16(x*x)) per row (bf16 accumulate-once), den = sum(x * cb[idx]) fp32
    cbv = tl.load(CB + idx[:, None] * G + kk[None, :], mask=kmask[None, :], other=0.0).to(tl.float32)
    num_v = tl.sum(_bf(x * x), axis=1)                          # [NV] partial over G
    den_v = tl.sum(x * cbv, axis=1)
    num_r = _bf(tl.sum(tl.reshape(num_v, [RB, D // G]), axis=1))   # torch: bf16 sum -> bf16
    den_r = tl.sum(tl.reshape(den_v, [RB, D // G]), axis=1)          # fp32 (bf16 x fp16 -> fp32)
    rr = tl.arange(0, RB)
    n2p = tl.load(N2P + swh * WS + row0 + rr)
    den_r = tl.maximum(den_r, 1e-12)
    out = n2p * (num_r / den_r)
    dst = tl.load(DST + w).to(tl.int64)
    if side == 0:
        tl.store(K_N2 + (dst * H + h) * WS + row0 + rr, out.to(tl.float16))
        tl.store(K_IDX + ((dst * H + h) * WS + row0) * (D // G) + v, idx.to(tl.uint8))
    else:
        tl.store(V_N2 + (dst * H + h) * WS + row0 + rr, out.to(tl.float16))
        tl.store(V_IDX + ((dst * H + h) * WS + row0) * (D // G) + v, idx.to(tl.uint8))


_CACHE: dict = {}


def scratch_cache(nw, H, ws, D, dtype, device):
    """A field dict in the fused state's cache layout with `nw` slots (prefill's batched
    whole-window encode writes here and commits straight from it)."""
    key = ("fields", nw, H, ws, D, dtype, device)
    if key not in _CACHE:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"nsn_encode_kernel.scratch_cache: first allocation of {key} inside CUDA-graph "
                "capture; memory_pool prewarming must cover this window count.")
        u8, f16 = torch.uint8, torch.float16
        z = lambda *shape, dt: torch.zeros(shape, dtype=dt, device=device)
        _CACHE[key] = {
            "idx": z(nw, H, ws, D // 8, dt=u8), "norm2": z(nw, H, ws, 1, dt=f16),
            "nq": z(nw, H, ws, dt=u8), "nsc": z(nw, H, 1, dt=dtype),
            "nmin": z(nw, H, 1, dt=dtype), "mq": z(nw, H, 1, D, dt=u8),
            "msc": z(nw, H, D // 32, 1, dt=dtype), "mmin": z(nw, H, D // 32, 1, dt=dtype)}
    return _CACHE[key]


def _consts(codebook, D, device):
    # Keyed on (D, device), NOT on the codebook's data_ptr: a bundle reloaded into a fresh
    # tensor made a new key, and if that first happened INSIDE CUDA-graph capture the packed
    # codebook and the Hadamard matrix were allocated in that graph's private pool. Reading
    # them from an eager call then yields ZEROS -- no error, but the encoder emits an all-zero
    # window, the packed store stays empty while committed_t advances, and generation collapses
    # to a repeated token (2026-09-10, the 0.0 sweep cells). Allocating during capture is now
    # refused outright so this can never be silent again.
    key = ("consts", D, device)
    if key not in _CACHE and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "nsn_encode_kernel._consts: first allocation inside CUDA-graph capture. These "
            "constants must be built before capture (see memory_pool's packed-store prewarm)."
        )
    if key not in _CACHE:
        KC, G = codebook.shape
        cb32 = codebook.float()
        cbp = torch.zeros(17, KC, device=device, dtype=torch.float32)
        cbp[:G] = cb32.t()
        cbp[16] = 0.5 * (cb32 * cb32).sum(-1)
        h = torch.ones((1, 1), device=device, dtype=torch.bfloat16)
        while h.shape[0] < D:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        had = (h / math.sqrt(D)).contiguous()     # bf16, exactly torch's cached matrix
        _CACHE[key] = (cbp.contiguous(), codebook.contiguous(), had)
    return _CACHE[key]


def stage1_scratch(nw, H, ws, D, dev):
    """Stage 1's normalised-tile scratch. Shared between captured decode (NENC windows) and
    eager prefill (1 window, NFULL_PAD for the batched path), so a first allocation on the
    capture stream would hand eager code a graph-pool tensor. memory_pool
    prewarms every window count that can occur; refuse the silent path."""
    key = ("scratch2", nw, H, ws, D, dev)
    if key not in _CACHE:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"nsn_encode_kernel.stage1_scratch: first allocation of {key} inside CUDA-graph "
                "capture; memory_pool prewarming must cover this window count.")
        xn = torch.empty(2, nw, H, ws, D, device=dev, dtype=torch.bfloat16)
        _CACHE[key] = (xn, torch.empty_like(xn),
                       torch.empty(2, nw, H, ws, device=dev, dtype=torch.float32))
    return _CACHE[key]


@torch.no_grad()
def encode_kv_windows_into(kbuf, vbuf, rows, pos, inv_freq, codebook, ck, cv, dst, had_v):
    """Encode nw windows of K and V ([nrows, H, D] bf16 buffers) in 3 launches and write them into
    the fused state's field dicts ck / cv at slots `dst` [nw]. rows/pos: [nw, ws] int64."""
    nw, ws = rows.shape
    H, D = kbuf.shape[1], kbuf.shape[2]
    G = codebook.shape[1]
    MG = D // ck["msc"].shape[2]
    dev = kbuf.device
    cbp, cb, had = _consts(codebook, D, dev)
    xn, xh, n2p = stage1_scratch(nw, H, ws, D, dev)
    rows = rows.contiguous()
    pos = pos.contiguous()
    dst = dst.contiguous()
    NWH = nw * H
    _nsn_stage1[(2 * NWH,)](kbuf, vbuf, rows, pos, inv_freq, xn, n2p, dst, NWH,
                            ck["nq"], ck["nsc"], ck["nmin"], ck["mq"], ck["msc"], ck["mmin"],
                            cv["nq"], cv["nsc"], cv["nmin"], cv["mq"], cv["msc"], cv["mmin"],
                            H=H, WS=ws, D=D, MG=MG, LEVELS=RTN4_LEVELS,
                            num_warps=NUM_WARPS, enable_fp_fusion=False)
    _nsn_stage2[(2 * NWH,)](xn, pos, inv_freq, had, xh, NWH, H=H, WS=ws, D=D, HAD_V=had_v,
                            num_warps=NUM_WARPS, enable_fp_fusion=False)
    RB = S3_RB
    _nsn_stage3[(2 * NWH * (ws // RB),)](xh, n2p, cbp, cb, dst, ck["idx"], ck["norm2"],
                                         cv["idx"], cv["norm2"], NWH,
                                         H=H, WS=ws, D=D, G=G, KC=codebook.shape[0], RB=RB,
                                         num_warps=S3_WARPS, enable_fp_fusion=False)


# ----------------------------------------------------------------------------------------------
# Commit: per-slot cache fields -> packed store, one launch (replaces _slot_fields + _commit_packed:
# ~60 gather/scatter launches per layer per step).
# ----------------------------------------------------------------------------------------------
@triton.jit
def _nsn_commit(SL, SEL, DST, WID,
                K_IDX, K_N2, K_NQ, K_NSC, K_NMIN, K_MQ, K_MSC, K_MMIN,
                V_IDX, V_N2, V_NQ, V_NSC, V_NMIN, V_MQ, V_MSC, V_MMIN,
                S_IDXK, S_IDXV, S_N2, S_NRM8, S_NSC, S_MKQ, S_MVQ, S_MKS, S_MVS,
                H: tl.constexpr, WS: tl.constexpr, D: tl.constexpr, NG: tl.constexpr,
                DG: tl.constexpr, G: tl.constexpr, GP: tl.constexpr):
    b = tl.program_id(0) // H
    h = tl.program_id(0) % H
    sl = tl.load(SL + b).to(tl.int64)
    # ---- per-token fields for the G rows leaving now
    g = tl.arange(0, GP)                                             # G padded to a power of 2
    gm = g < G
    rows = tl.load(SEL + b * G + g, mask=gm, other=0).to(tl.int64)   # rows within the window
    dst = tl.load(DST + b * G + g, mask=gm, other=0).to(tl.int64)    # pool slots (trash-routed)
    j = tl.arange(0, NG)
    m2 = gm[:, None] & (j[None, :] < NG)
    src_k = K_IDX + ((sl * H + h) * WS + rows[:, None]) * NG + j[None, :]
    src_v = V_IDX + ((sl * H + h) * WS + rows[:, None]) * NG + j[None, :]
    out = (dst[:, None] * H + h) * NG + j[None, :]
    tl.store(S_IDXK + out, tl.load(src_k, mask=m2, other=0), mask=m2)
    tl.store(S_IDXV + out, tl.load(src_v, mask=m2, other=0), mask=m2)
    n2k = tl.load(K_N2 + (sl * H + h) * WS + rows, mask=gm, other=0.0)
    n2v = tl.load(V_N2 + (sl * H + h) * WS + rows, mask=gm, other=0.0)
    tl.store(S_N2 + (dst * H + h) * 2, n2k.to(tl.float16), mask=gm)
    tl.store(S_N2 + (dst * H + h) * 2 + 1, n2v.to(tl.float16), mask=gm)
    nqk = tl.load(K_NQ + (sl * H + h) * WS + rows, mask=gm, other=0)
    nqv = tl.load(V_NQ + (sl * H + h) * WS + rows, mask=gm, other=0)
    tl.store(S_NRM8 + dst * H + h, nqk | (nqv << 4), mask=gm)
    # ---- window metadata (idempotent; not-due windows are routed to the trash row by WID)
    wid = tl.load(WID + b).to(tl.int64)
    dt = S_NSC.dtype.element_ty
    tl.store(S_NSC + (wid * H + h) * 4 + 0, tl.load(K_NSC + sl * H + h).to(dt))
    tl.store(S_NSC + (wid * H + h) * 4 + 1, tl.load(K_NMIN + sl * H + h).to(dt))
    tl.store(S_NSC + (wid * H + h) * 4 + 2, tl.load(V_NSC + sl * H + h).to(dt))
    tl.store(S_NSC + (wid * H + h) * 4 + 3, tl.load(V_NMIN + sl * H + h).to(dt))
    c2 = tl.arange(0, D // 2)
    mk_lo = tl.load(K_MQ + (sl * H + h) * D + 2 * c2)
    mk_hi = tl.load(K_MQ + (sl * H + h) * D + 2 * c2 + 1)
    tl.store(S_MKQ + (wid * H + h) * (D // 2) + c2, (mk_lo & 15) | ((mk_hi & 15) << 4))
    mv_lo = tl.load(V_MQ + (sl * H + h) * D + 2 * c2)
    mv_hi = tl.load(V_MQ + (sl * H + h) * D + 2 * c2 + 1)
    tl.store(S_MVQ + (wid * H + h) * (D // 2) + c2, (mv_lo & 15) | ((mv_hi & 15) << 4))
    q = tl.arange(0, DG)
    tl.store(S_MKS + ((wid * H + h) * DG + q) * 2, tl.load(K_MSC + (sl * H + h) * DG + q).to(dt))
    tl.store(S_MKS + ((wid * H + h) * DG + q) * 2 + 1, tl.load(K_MMIN + (sl * H + h) * DG + q).to(dt))
    tl.store(S_MVS + ((wid * H + h) * DG + q) * 2, tl.load(V_MSC + (sl * H + h) * DG + q).to(dt))
    tl.store(S_MVS + ((wid * H + h) * DG + q) * 2 + 1, tl.load(V_MMIN + (sl * H + h) * DG + q).to(dt))


@torch.no_grad()
def commit_into(ls, kf, vf, sl, sel, dst, wid):
    """Scatter rows `sel` [nw, G] of the windows at cache slots `sl` [nw] (fields kf/vf, cache
    layout) to pool slots `dst` [nw*G] and arena rows `wid` [nw] of LayerStore `ls`. One launch."""
    nw, G = sel.shape
    H, WS = kf["nq"].shape[1], kf["nq"].shape[2]
    D = kf["mq"].shape[-1]
    NG = kf["idx"].shape[-1]
    DG = kf["msc"].shape[2]
    _nsn_commit[(nw * H,)](
        sl.contiguous(), sel.contiguous(), dst.contiguous(), wid.contiguous(),
        kf["idx"], kf["norm2"], kf["nq"], kf["nsc"], kf["nmin"], kf["mq"], kf["msc"], kf["mmin"],
        vf["idx"], vf["norm2"], vf["nq"], vf["nsc"], vf["nmin"], vf["mq"], vf["msc"], vf["mmin"],
        ls.idx_k, ls.idx_v, ls.n2, ls.nrm8, ls.nsc, ls.mk_q, ls.mv_q, ls.mk_s, ls.mv_s,
        H=H, WS=WS, D=D, NG=NG, DG=DG, G=G, GP=triton.next_power_of_2(G), num_warps=1)


# ----------------------------------------------------------------------------------------------
# Bookkeeping: _append + _flush_candidates + urgency pick + _advance + committed_t, one launch.
# Replaces ~80 [B]-sized elementwise launches per layer per step.
# ----------------------------------------------------------------------------------------------
@triton.jit
def _nsn_book(SLOTS, POS, LOC, CNT, EV, ENC, START, RPOS, RLOC, MAP, COMPACT, COMMITTED,
              O_DUE, O_SEL, O_DST, O_WID, O_LROW, O_PROW, O_PPOS, O_CDST, O_SPOS,
              CK, CV, KBUF, VBUF,
              B, cap, recent, gran, nwin_trash, scratch_slot, maxw, prefix,
              BLOCK: tl.constexpr, WS: tl.constexpr, GRAN: tl.constexpr, NENC: tl.constexpr,
              FIRST: tl.constexpr, TRASH_LOC: tl.constexpr, CACHE_SCRATCH: tl.constexpr,
              HD: tl.constexpr, HD_P: tl.constexpr, PREWRITE: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < B
    slot = tl.load(SLOTS + i, mask=m, other=scratch_slot).to(tl.int64)
    pos = tl.load(POS + i, mask=m, other=-1000000000).to(tl.int64)
    loc = tl.load(LOC + i, mask=m, other=TRASH_LOC).to(tl.int64)
    # padding rows (loc == TRASH) and PREFIX (attention-sink) rows never enter a window; the
    # sink rows stay raw in the bf16 tier, at fixed rows the ring never reuses
    drop = (loc == TRASH_LOC) | (pos < prefix)
    slot = tl.where(drop, scratch_slot, slot)
    pos = tl.where(drop, -1000000000, pos)
    # bf16-tier row of the incoming token; the raw row is written here too (the pool writes
    # it again after the hook -- same bytes), so the flush below can read a complete window
    lrow = tl.load(MAP + loc, mask=m, other=0).to(tl.int64)
    tl.store(O_LROW + i, lrow, mask=m)
    if PREWRITE:
        # HD_P is HD padded to a power of two: tl.arange needs that, but KVH*D is not one for
        # every model (phi-4 has 10 KV heads, so HD = 1280). The mask below was already written
        # against the real HD, so padding the span is all that was missing.
        e = tl.arange(0, HD_P)
        mm = m[:, None] & (e[None, :] < HD)
        tl.store(KBUF + lrow[:, None] * HD + e[None, :], tl.load(CK + i[:, None] * HD + e[None, :], mask=mm, other=0.0), mask=mm)
        tl.store(VBUF + lrow[:, None] * HD + e[None, :], tl.load(CV + i[:, None] * HD + e[None, :], mask=mm, other=0.0), mask=mm)
    # ---- _append
    cnt = tl.load(CNT + slot, mask=m, other=0)
    ev = tl.load(EV + slot, mask=m, other=0)
    enc = tl.load(ENC + slot, mask=m, other=0) != 0
    start = tl.load(START + slot, mask=m, other=0)
    last = (start + cnt - 1 + cap) % cap
    lastpos = tl.load(RPOS + slot * cap + last, mask=m, other=0)
    expected = tl.where(cnt > 0, lastpos + 1, pos)
    reset = pos != expected
    cnt = tl.where(reset, 0, cnt)
    ev = tl.where(reset, 0, ev)
    enc = enc & (~reset)
    tail = (start + cnt) % cap
    tl.store(RLOC + slot * cap + tail, loc, mask=m)
    tl.store(RPOS + slot * cap + tail, pos, mask=m)
    tl.debug_barrier()   # win_loc/win_pos below may read the row just stored, from another lane
    cnt = cnt + 1
    # ---- _flush_candidates (require_enc) + urgency
    due = (cnt >= recent + ev + gran) & enc & m
    urg = tl.where((cnt >= WS) & (~enc) & m, cnt, -1)
    k = tl.arange(0, WS)
    ridx = (start[:, None] + k[None, :]) % cap
    win_loc = tl.load(RLOC + slot[:, None] * cap + ridx, mask=m[:, None], other=TRASH_LOC)
    win_pos = tl.load(RPOS + slot[:, None] * cap + ridx, mask=m[:, None], other=0)
    # rows leaving now: [B, GRAN]
    g = tl.arange(0, GRAN)
    sel = tl.minimum(ev[:, None] + g[None, :], WS - 1)
    tl.store(O_SEL + i[:, None] * GRAN + g[None, :], sel, mask=m[:, None])
    sidx = (start[:, None] + sel) % cap
    loc_sel = tl.load(RLOC + slot[:, None] * cap + sidx, mask=m[:, None], other=TRASH_LOC)
    pos_sel = tl.load(RPOS + slot[:, None] * cap + sidx, mask=m[:, None], other=0)
    tl.store(O_SPOS + i[:, None] * GRAN + g[None, :], pos_sel, mask=m[:, None])
    dst = tl.where(due[:, None], loc_sel, TRASH_LOC)
    tl.store(O_DST + i[:, None] * GRAN + g[None, :], dst, mask=m[:, None])
    # window arena row: compact[req] * maxw + first_pos // WS, trash unless due
    comp = tl.load(COMPACT + slot, mask=m, other=0).to(tl.int64)
    first = tl.sum(tl.where(k[None, :] == 0, win_pos, 0), axis=1)
    ordn = tl.minimum(tl.maximum(first, 0) // WS, maxw - 1)
    wid = tl.where(due, comp * maxw + ordn, nwin_trash)
    tl.store(O_WID + i, wid, mask=m)
    tl.store(O_DUE + i, due.to(tl.int8), mask=m)
    # ---- encode picks: NENC most urgent complete-and-unencoded windows
    picked = i < 0
    for e in tl.static_range(NENC):
        pk = tl.argmax(urg, axis=0)
        onehot = i == pk
        # MAX, not sum: the neutral element here is -1 (urg is -1 for a slot with nothing to
        # encode), so a sum would return urg[pk] - (BLOCK-1) and test `cnt >= BLOCK-1` rather
        # than `cnt >= WS`. Invisible below a decode batch of 65 and then silent: the window is
        # never picked, `due` needs `enc`, and the bf16 residual grows to BLOCK-1 tokens --
        # inconsistent residual policies, and ring self-overwrite once BLOCK reaches the ring
        # length. The three sibling reductions below use 0 as their neutral element and are
        # correct as sums.
        want = tl.max(tl.where(onehot, urg, -1), axis=0) >= 0
        psl = tl.sum(tl.where(onehot, slot, 0), axis=0)
        cdst = tl.where(want, psl, CACHE_SCRATCH)
        tl.store(O_CDST + e, cdst)
        prow_loc = tl.sum(tl.where(onehot[:, None], win_loc, 0), axis=0)          # [WS]
        prow_loc = tl.where(want, prow_loc, TRASH_LOC)
        tl.store(O_PROW + e * WS + k, tl.load(MAP + prow_loc).to(tl.int64))
        ppos = tl.sum(tl.where(onehot[:, None], win_pos, 0), axis=0)
        tl.store(O_PPOS + e * WS + k, tl.maximum(ppos, 0))
        picked = picked | (onehot & want)
        urg = tl.where(onehot, -2, urg)
    enc = enc | picked
    # ---- _advance
    ev2 = tl.where(due, ev + gran, ev)
    roll = ev2 >= WS
    start = tl.where(roll, (start + WS) % cap, start)
    cnt = tl.where(roll, cnt - WS, cnt)
    ev2 = tl.where(roll, 0, ev2)
    enc = enc & (~roll)
    tl.store(CNT + slot, cnt, mask=m)
    tl.store(EV + slot, ev2, mask=m)
    tl.store(ENC + slot, enc.to(tl.int8), mask=m)
    tl.store(START + slot, start, mask=m)
    if FIRST:
        c0 = tl.load(COMMITTED + slot, mask=m, other=0)
        c0 = tl.where(reset, 0, c0)
        c0 = tl.where(due, c0 + gran, c0)
        tl.store(COMMITTED + slot, c0.to(tl.int32), mask=m)


def _assert_shapes(kvh, D, recent, ws):
    """Preconditions the kernels rely on but cannot express.

    KVH*D used to be required to be a power of two, because _nsn_book spanned the pool row with
    tl.arange(0, HD); that rejected phi-4, which has 10 KV heads. The span is padded now, so
    only the flush-schedule precondition is left."""
    if recent < ws:
        raise RuntimeError(
            f"nsn_encode_kernel: RECENT_TOKENS={recent} < window_size={ws}. The flush schedule "
            "assumes a window is complete before any of its rows age out; below that the "
            "encoder reads ring rows that have not been written yet.")


@torch.no_grad()
def decode_bookkeeping(st, slots, pos, loc, hp_map, compact, committed_t, ws, recent, gran,
                       nenc, nwin_trash, scratch_slot, maxw, first_layer, trash_loc,
                       cache_scratch, cache_k=None, cache_v=None, k_buffer=None, v_buffer=None,
                       prefix=0):
    """One launch: append the incoming rows, decide evictions, pick windows to encode, advance.
    Returns (due[B] bool, sel[B,gran], dst[B*gran], wid[B], lrow[B], prow[nenc,ws], ppos[nenc,ws],
    cdst[nenc]) -- everything the encoder, the pre-write and the commit need."""
    B = slots.numel()
    dev = slots.device
    i64 = torch.int64
    due = torch.empty(B, dtype=torch.int8, device=dev)
    sel = torch.empty(B, gran, dtype=i64, device=dev)
    dst = torch.empty(B * gran, dtype=i64, device=dev)
    wid = torch.empty(B, dtype=i64, device=dev)
    lrow = torch.empty(B, dtype=i64, device=dev)
    prow = torch.empty(nenc, ws, dtype=i64, device=dev)
    ppos = torch.empty(nenc, ws, dtype=i64, device=dev)
    cdst = torch.empty(nenc, dtype=i64, device=dev)
    spos = torch.empty(B, gran, dtype=i64, device=dev)
    pre = cache_k is not None
    if pre:
        assert cache_k.is_contiguous() and k_buffer.is_contiguous() and cache_k.dtype == k_buffer.dtype
        HD = k_buffer.shape[1] * k_buffer.shape[2]
        _assert_shapes(k_buffer.shape[1], k_buffer.shape[2], recent, ws)
    else:
        cache_k = cache_v = k_buffer = v_buffer = slots
        HD = 1
    _nsn_book[(1,)](slots, pos, loc, st.cnt, st.ev, st.enc, st.start, st.pos, st.loc, hp_map,
                    compact, committed_t, due, sel, dst, wid, lrow, prow, ppos, cdst, spos,
                    cache_k, cache_v, k_buffer, v_buffer,
                    B, st.cap, recent, gran, nwin_trash, scratch_slot, maxw, prefix,
                    BLOCK=max(2, triton.next_power_of_2(B)), WS=ws, GRAN=gran, NENC=nenc,
                    FIRST=first_layer, TRASH_LOC=trash_loc, CACHE_SCRATCH=cache_scratch,
                    HD=HD, HD_P=triton.next_power_of_2(HD), PREWRITE=pre, num_warps=4)
    return due, sel, dst, wid, lrow, prow, ppos, cdst, spos
