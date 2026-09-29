#!/usr/bin/env python3
"""Fused attention over packed NSNQuant KV state.

The kernel combines VQ reconstruction, RoPE, online softmax, and value accumulation without
materializing reconstructed KV tensors or the full attention matrix. Value reconstruction uses
separate VQ and mean accumulators, followed by one Hadamard transform. For Keys, expanding the
position-dependent mean term gives

    mean_term(t) = sum_j cos[t,j]*A_j + sin[t,j]*B_j
    A = q_lo*m_lo + q_hi*m_hi        B = q_hi*m_lo - q_lo*m_hi

where ``A`` and ``B`` are constant within a window tile. This form evaluates the mean term as a
single matrix product. The Key VQ term is position-independent because NSNQuant applies the
Hadamard transform to Keys and the corresponding transform to queries.
"""
# import sys   # (path hack from the local clone, not needed in the package)
import torch, triton, triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import _unified_stage2



@triton.jit
def _unpack_mean4(Q_, S_, mbase, sbase, offs_j, D2: tl.constexpr, G32: tl.constexpr):
    """One window mean out of the 4-bit arena: byte i holds coord 2i in its low nibble and
    2i+1 in its high (pack_nibbles' order), dequantised by a scale/min per G32 coords.
    Returns the two halves the rope adjoint wants, ([0, D2), [D2, D)).

    Dequantization uses fp32 because Triton does not reproduce the reference BF16 arithmetic
    bit-for-bit here; fp32 more directly reconstructs the stored 4-bit value.
    """
    b_lo = tl.load(Q_ + mbase + offs_j // 2)
    c_lo = tl.where(offs_j % 2 == 0, b_lo & 0x0F, (b_lo >> 4) & 0x0F)
    b_hi = tl.load(Q_ + mbase + (D2 + offs_j) // 2)
    c_hi = tl.where((D2 + offs_j) % 2 == 0, b_hi & 0x0F, (b_hi >> 4) & 0x0F)
    g_lo = offs_j // G32
    g_hi = (D2 + offs_j) // G32
    s_lo = tl.load(S_ + sbase + g_lo * 2)
    m_lo_ = tl.load(S_ + sbase + g_lo * 2 + 1)
    s_hi = tl.load(S_ + sbase + g_hi * 2)
    m_hi_ = tl.load(S_ + sbase + g_hi * 2 + 1)
    # Dequantize stored codes in fp32; see the function docstring.
    return (c_lo.to(tl.float32) * s_lo.to(tl.float32) + m_lo_.to(tl.float32),
            c_hi.to(tl.float32) * s_hi.to(tl.float32) + m_hi_.to(tl.float32))


@triton.jit
def _nsn_fused_stage1(
    Hq,          # [B, H, D]        query, Hadamard-rotated once
    Q,           # [B, H, D]        query, raw (for the A/B mean coefficients)
    K_Idx,       # [NSLOT, KVH, NG] uint8
    V_Idx,
    CB,          # [KVH, KC, G]     fp32/fp16 centroids
    S1, S2,      # [NSLOT, KVH]     s1 = norm*norm2, s2 = norm  (K side)  (PACK_SC=False)
    N2, NRM,     # [NSLOT, KVH]     V side per-token scalars              (PACK_SC=False)
    Scal,        # [NSLOT, KVH, 4]  the same four, interleaved            (PACK_SC=True)
    # --- NATIVE: nsn_packed_store's own layout, read with nothing materialised
    N2P,         # [NSLOT, KVH, 2]  fp16 norm2, (K, V)
    Nrm8,        # [NSLOT, KVH]     uint8, K norm code in the low nibble | V in the high
    Nsc,         # [NWIN, KVH, 4]   k_scale, k_min, v_scale, v_min
    MKQ, MVQ,    # [NWIN, KVH, D/2] uint8, the 4-bit window mean, nibble-packed over D
    MKS, MVS,    # [NWIN, KVH, D/G32, 2]  its scale and min
    WinMap,      # [B, NWMAX]       int32: window ordinal -> arena row
    MeanK,       # [NREQ, NW, KVH, D]  pre-RoPE window mean, K
    MeanV,       # [NREQ, NW, KVH, D]  window mean, V, as the reference stores it
    CosSin,      # [T, D]           cos in [:D2], sin in [D2:]  (TRIG_ONFLY=False only)
    InvFreq,     # [D2]             rope inverse frequencies     (TRIG_ONFLY=True)
    MeanIdx,     # [B]
    kv_indptr, kv_indices,
    Qr, RK, RV, r_indptr, r_indices,     # RAW: unscaled q (model dtype), bf16 K/V tier, raw-tail
    sqr_b, sqr_h, sk_t, sk_h, sv_t, sv_h, sm_scale,   # indices (hp rows) and strides
    pos_off,     # absolute position of packed index 0 (= PREFIX): windows and RoPE are positional
    Att_Out, Att_Lse, num_kv_splits,
    sq_b, sq_h,
    si_t, si_h,
    ss_t,
    sm_r, sm_w, sm_h,
    scs,
    so_b, so_h, so_s,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr, KVH_: tl.constexpr,
    D: tl.constexpr, D2: tl.constexpr, NG: tl.constexpr, G: tl.constexpr, KC: tl.constexpr,
    WS: tl.constexpr, NW,
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, MIN_BLOCK_KV: tl.constexpr,
    DOT_F16: tl.constexpr, TRIG_ONFLY: tl.constexpr, FOLD_K: tl.constexpr,
    NO_SCALARS: tl.constexpr, PACK_SC: tl.constexpr, FOLD_MEAN: tl.constexpr,
    NATIVE: tl.constexpr, G32: tl.constexpr, NWMAX: tl.constexpr,
    RAW: tl.constexpr, SPLITS_VQ: tl.constexpr, RAW_SPLITS: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, D)
    offs_j = tl.arange(0, D2)
    offs_g = tl.arange(0, NG)
    offs_gc = tl.arange(0, G)

    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    kv_splits = tl.load(num_kv_splits + cur_batch)
    rb = tl.load(MeanIdx + cur_batch).to(tl.int64)

    kv_len_per_split = tl.cdiv(tl.cdiv(seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, seq_len)
    is_raw = False
    if RAW:
        # Programs [SPLITS_VQ, SPLITS_VQ + RAW_SPLITS) read the RAW bf16 tail of this request
        # (positions [nq, seq), rows of the ring tier) with plain attention, into the same
        # split-partial layout: stage 2 merges them with the packed splits, so the two-tier
        # read is one launch pair instead of two attention calls plus an LSE merge.
        is_raw = split_kv_id >= SPLITS_VQ
        if is_raw:
            r_start = tl.load(r_indptr + cur_batch)
            r_len = tl.load(r_indptr + cur_batch + 1) - r_start
            r_per = tl.cdiv(tl.cdiv(r_len, RAW_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
            split_kv_start = r_per * (split_kv_id - SPLITS_VQ)
            split_kv_end = tl.minimum(split_kv_start + r_per, r_len)
            kv_start = r_start

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    accm = tl.zeros([BLOCK_H, D], dtype=tl.float32)   # unused when FOLD_MEAN

    if RAW and is_raw and (split_kv_end > split_kv_start):
        qr = tl.load(Qr + cur_batch * sqr_b + cur_head[:, None] * sqr_h + offs_d[None, :],
                     mask=mask_h[:, None], other=0.0)                    # [BLOCK_H, D] model dtype
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            row = tl.load(r_indices + kv_start + offs_n, mask=mask_n, other=0).to(tl.int64)
            kt = tl.load(RK + row[:, None] * sk_t + cur_kv_head * sk_h + offs_d[None, :],
                         mask=mask_n[:, None], other=0.0)               # [BLOCK_N, D]
            qk = tl.dot(qr, tl.trans(kt)).to(tl.float32) * sm_scale
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))
            vt = tl.load(RV + row[:, None] * sv_t + cur_kv_head * sv_h + offs_d[None, :],
                         mask=mask_n[:, None], other=0.0)
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            accm *= re_scale[:, None]
            # raw rows are NOT Hadamard-rotated: they belong in the un-rotated half (accm)
            accm += tl.dot(p.to(vt.dtype), vt).to(tl.float32)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
    elif split_kv_end > split_kv_start:
        cb_base = CB + cur_kv_head.to(tl.int64) * (KC * G)
        hq = tl.load(Hq + cur_batch * sq_b + cur_head[:, None] * sq_h + offs_d[None, :],
                     mask=mask_h[:, None], other=0.0).to(tl.float32)      # [BLOCK_H, D]
        q_lo = tl.load(Q + cur_batch * sq_b + cur_head[:, None] * sq_h + offs_j[None, :],
                       mask=mask_h[:, None], other=0.0).to(tl.float32)
        q_hi = tl.load(Q + cur_batch * sq_b + cur_head[:, None] * sq_h + D2 + offs_j[None, :],
                       mask=mask_h[:, None], other=0.0).to(tl.float32)
        mean_base = MeanK + rb * sm_r + cur_kv_head * sm_h
        mv_base = MeanV + rb * sm_r + cur_kv_head * sm_h
        if TRIG_ONFLY:
            # angle(n) = (start_n + j)*invf = base*invf + j*invf. The j half is the same for
            # every tile, batch and head-block, so its cos/sin are computed ONCE here instead
            # of BLOCK_N x D2 libdevice calls per iteration -- CQ's TRIG_HOIST, 1.04-1.30x
            # there. It also removes the [BLOCK_N, D] CosSin load, which is a third of this
            # kernel's per-tile bytes and comes from a 2 MB table (the codebook is 8 KB).
            inv_f = tl.load(InvFreq + offs_j).to(tl.float32)
            ang_j = tl.arange(0, BLOCK_N)[:, None].to(tl.float32) * inv_f[None, :]
            cos_j = tl.cos(ang_j)
            sin_j = tl.sin(ang_j)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0).to(tl.int64)

            # ---- K: VQ term. No trig -- NSN rotates the query, not the key.
            ik = tl.load(K_Idx + kv_loc[:, None] * si_t + cur_kv_head * si_h + offs_g[None, :],
                         mask=mask_n[:, None], other=0).to(tl.int32)
            kf = tl.load(cb_base + ik[:, :, None] * G + offs_gc[None, None, :],
                         mask=mask_n[:, None, None], other=0.0).to(tl.float32)
            ktile = tl.reshape(kf, (BLOCK_N, NG * G))                     # [BLOCK_N, D]

            # ---- K: mean term, collected on cos/sin (one dot, same shape as the VQ dot)
            w = tl.minimum((start_n + pos_off) // WS, NW - 1)
            if NATIVE:
                # The arena hands out window rows from a free list, so a request's windows are
                # NOT contiguous: the ordinal -> row map is an argument, not arithmetic.
                wid = tl.load(WinMap + cur_batch * NWMAX + w).to(tl.int32)
                ksc = tl.load(Nsc + wid * (KVH_ * 4) + cur_kv_head * 4 + 0)
                kmn = tl.load(Nsc + wid * (KVH_ * 4) + cur_kv_head * 4 + 1)
                vsc = tl.load(Nsc + wid * (KVH_ * 4) + cur_kv_head * 4 + 2)
                vmn = tl.load(Nsc + wid * (KVH_ * 4) + cur_kv_head * 4 + 3)
                mbase = wid * (KVH_ * (D // 2)) + cur_kv_head * (D // 2)
                sbase = wid * (KVH_ * (D // G32) * 2) + cur_kv_head * (D // G32) * 2
                m_lo, m_hi = _unpack_mean4(MKQ, MKS, mbase, sbase, offs_j, D2, G32)
            else:
                m_lo = tl.load(mean_base + w * sm_w + offs_j).to(tl.float32)
                m_hi = tl.load(mean_base + w * sm_w + D2 + offs_j).to(tl.float32)
            ab = tl.join(q_lo * m_lo[None, :] + q_hi * m_hi[None, :],
                         q_hi * m_lo[None, :] - q_lo * m_hi[None, :])
            ab = tl.reshape(tl.permute(ab, (0, 2, 1)), (BLOCK_H, D))      # [A | B]
            if TRIG_ONFLY:
                ang_b = (start_n + pos_off).to(tl.float32) * inv_f
                cos_b = tl.cos(ang_b)
                sin_b = tl.sin(ang_b)
                cs = tl.join(cos_b[None, :] * cos_j - sin_b[None, :] * sin_j,
                             sin_b[None, :] * cos_j + cos_b[None, :] * sin_j)
                cs = tl.reshape(tl.permute(cs, (0, 2, 1)), (BLOCK_N, D))
                cs = tl.where(mask_n[:, None], cs, 0.0)
            else:
                cs = tl.load(CosSin + offs_n[:, None] * scs + offs_d[None, :],
                             mask=mask_n[:, None], other=0.0).to(tl.float32)  # [BLOCK_N, D]

            # Timing-only ablation that removes the four per-token scalar gathers.
            if NO_SCALARS:
                s1 = tl.full([BLOCK_N], 1.0, tl.float32)
                s2 = s1
            elif NATIVE:
                # norm is 4 bits sharing one byte with the other side's, dequantised by the
                # window's scale/min loaded above; norm2 is the adjacent fp16 pair. Two token
                # loads, off the stored form, nothing materialised.
                nb = tl.load(Nrm8 + kv_loc * KVH_ + cur_kv_head, mask=mask_n, other=0)
                # fp32, bit-exact against torch's fp32; see _unpack_mean4's note
                nrm_k = ((nb & 0x0F).to(tl.float32) * ksc.to(tl.float32)
                         + kmn.to(tl.float32))
                nrm_v = (((nb >> 4) & 0x0F).to(tl.float32) * vsc.to(tl.float32)
                         + vmn.to(tl.float32))
                n2k = tl.load(N2P + kv_loc * (KVH_ * 2) + cur_kv_head * 2,
                              mask=mask_n, other=0.0).to(tl.float32)
                n2v_ = tl.load(N2P + kv_loc * (KVH_ * 2) + cur_kv_head * 2 + 1,
                               mask=mask_n, other=0.0).to(tl.float32)
                s1 = nrm_k * n2k
                s2 = nrm_k
            elif PACK_SC:
                # Four scalars per (token, kv head) from four SEPARATE [NSLOT, KVH] tensors is
                # four gather streams to four distant addresses. CQ gathers none (const-1,
                # elided), and ablate_scalars.py showed these four ARE the whole remaining gap:
                # stubbing them out took NSN from 2.07x CQ to 1.02x at B=128. Interleaved into
                # [NSLOT, KVH, 4] they are 16 adjacent bytes, read as ONE vector load and taken
                # apart with tl.split -- one instruction and one sector instead of four of each.
                # Storage order is (s1, s2, n2, nrm), so after reshaping to [BLOCK_N, 2, 2] the
                # first split yields {s1, n2} and the second {s2, nrm}.
                sc4 = tl.load(Scal + (kv_loc * ss_t + cur_kv_head * 4)[:, None]
                              + tl.arange(0, 4)[None, :],
                              mask=mask_n[:, None], other=0.0).to(tl.float32)
                sc_a, sc_b = tl.split(tl.reshape(sc4, (BLOCK_N, 2, 2)))
                s1, n2p = tl.split(sc_a)
                s2, nrp = tl.split(sc_b)
            else:
                s1 = tl.load(S1 + kv_loc * ss_t + cur_kv_head, mask=mask_n, other=0.0).to(tl.float32)
                s2 = tl.load(S2 + kv_loc * ss_t + cur_kv_head, mask=mask_n, other=0.0).to(tl.float32)
            if FOLD_K:
                # qk = s1*dot(hq, ktile^T) + s2*dot(ab, cs^T). The per-token scalars land on the
                # RESULT's columns, so folding them into the tiles instead lets the two dots
                # become one of twice the contraction depth:
                #     dot([hq | ab], [s1*ktile | s2*cs]^T)
                # Same FLOPs, one instruction sequence and one accumulator. CQ's K side is two
                # dots of depth 64 where ours is two of depth 128, so this is the term that
                # differs; measured rather than assumed (FOLD_K is an ablation switch).
                lhs = tl.join(hq, ab)
                lhs = tl.reshape(tl.permute(lhs, (0, 2, 1)), (BLOCK_H, 2 * D))
                rhs = tl.join(ktile * s1[:, None], cs * s2[:, None])
                rhs = tl.reshape(tl.permute(rhs, (0, 2, 1)), (BLOCK_N, 2 * D))
                if DOT_F16:
                    qk = tl.dot(lhs.to(tl.float16), tl.trans(rhs).to(tl.float16))
                else:
                    qk = tl.dot(lhs, tl.trans(rhs), input_precision="ieee")
            else:
                if DOT_F16:
                    qk = tl.dot(hq.to(tl.float16), tl.trans(ktile).to(tl.float16))
                    qm = tl.dot(ab.to(tl.float16), tl.trans(cs).to(tl.float16))
                else:
                    qk = tl.dot(hq, tl.trans(ktile), input_precision="ieee")
                    qm = tl.dot(ab, tl.trans(cs), input_precision="ieee")
                qk = qk * s1[None, :] + qm * s2[None, :]
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            # ---- V: one accumulator. n2_t*cb[iv_t] + Hmean_v[w], with H folded into the mean.
            iv = tl.load(V_Idx + kv_loc[:, None] * si_t + cur_kv_head * si_h + offs_g[None, :],
                         mask=mask_n[:, None], other=0).to(tl.int32)
            vf = tl.load(cb_base + iv[:, :, None] * G + offs_gc[None, None, :],
                         mask=mask_n[:, None, None], other=0.0).to(tl.float32)
            vtile = tl.reshape(vf, (BLOCK_N, NG * G))
            if NO_SCALARS:
                n2 = tl.full([BLOCK_N], 1.0, tl.float32)
            elif NATIVE:
                n2 = n2v_
            elif PACK_SC:
                n2 = n2p
            else:
                n2 = tl.load(N2 + kv_loc * ss_t + cur_kv_head, mask=mask_n, other=0.0).to(tl.float32)
            # The V mean is kept in its OWN accumulator rather than folded into the tile as
            # H(mean_v). Folding needs the mean stored already-rotated, and H's first row is
            # all-ones, so H(mean) carries a large DC term that 4-bit RTN quantises badly --
            # measured 2.8e-02 against the reference's 2.0e-03. Here mean_v is stored exactly as
            # the reference stores it and the single H moves to the caller, applied to the VQ
            # half of the output only. Per tile this is a rank-1 update, not a dot.
            if NATIVE:
                mv_lo, mv_hi = _unpack_mean4(MVQ, MVS, mbase, sbase, offs_j, D2, G32)
                mvw = tl.join(mv_lo, mv_hi)
                mvw = tl.reshape(tl.permute(mvw, (1, 0)), (D,))
            else:
                mvw = tl.load(mv_base + w * sm_w + offs_d).to(tl.float32)
            if FOLD_MEAN:
                # MeanV then holds H(mean_v): one accumulator, no rank-1 update, no 2D-wide
                # stage-2 partial. Requires the pool to store the ROTATED mean, which costs
                # 0.25 b/ch at fp16 against 0.078 for the reference's 4-bit form (4-bit of the
                # rotated mean is not an option -- H's all-ones first row makes a DC outlier
                # that RTN4 mangles). Memory/speed trade, both measured in ablate_foldmean.py.
                vtile = vtile * n2[:, None] + mvw[None, :]
            else:
                vtile = vtile * n2[:, None]
            if NO_SCALARS:
                nr = tl.full([BLOCK_N], 1.0, tl.float32)
            elif NATIVE:
                nr = nrm_v
            elif PACK_SC:
                nr = nrp
            else:
                nr = tl.load(NRM + kv_loc * ss_t + cur_kv_head, mask=mask_n, other=0.0).to(tl.float32)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            # nrm scales the PROBABILITIES, not the V rows: BLOCK_H*BLOCK_N multiplies instead of
            # BLOCK_N*D. e_sum must keep the UNSCALED p or the softmax denominator is wrong --
            # the same trap CQ documents for VSCALE_ON_P.
            p_dot = p * nr[None, :]
            acc *= re_scale[:, None]
            if not FOLD_MEAN:
                accm *= re_scale[:, None]
            if DOT_F16:
                acc += tl.dot(p_dot.to(tl.float16), vtile.to(tl.float16))
            else:
                acc += tl.dot(p_dot, vtile, input_precision="ieee")
            if not FOLD_MEAN:
                accm += tl.sum(p_dot, 1)[:, None] * mvw[None, :]
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    # Stored UNCONDITIONALLY, outside the `if`, so a split with no work writes its own zero /
    # -inf instead of leaving the caller to memset the whole partial buffer. The caller's
    # `torch.zeros` cost B*H*splits*D per call -- 335 MB at B=128 -- and made this kernel
    # superlinear in batch (4x batch -> 5.3x time) where CQ and bf16 were linear. e_sum is 0
    # exactly when the split was empty, and 0/0 would be NaN, so the divide is guarded.
    denom = tl.where(e_sum > 0.0, e_sum, 1.0)
    off_o = (cur_batch * so_b + cur_head[:, None] * so_h + split_kv_id * so_s
             + offs_d[None, :])
    tl.store(Att_Out + off_o, acc / denom[:, None], mask=mask_h[:, None])
    if not FOLD_MEAN:
        tl.store(Att_Out + off_o + D, accm / denom[:, None], mask=mask_h[:, None])
    off_l = (cur_batch * so_b + cur_head * so_h + split_kv_id * so_s) // (D if FOLD_MEAN else 2 * D)
    tl.store(Att_Lse + off_l,
             tl.where(e_sum > 0.0, e_max + tl.log(denom), float("-inf")), mask=mask_h)


_NKS: dict = {}


def _nks_for(B, splits, device):
    key = (B, splits, device)
    if key not in _NKS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"nsn_fused_kernel._nks_for: first allocation of {key} inside CUDA-graph "
                "capture; memory_pool prewarming must cover this batch size.")
        _NKS[key] = torch.full((B,), splits, device=device, dtype=torch.int32)
    return _NKS[key]


def nsn_fused(hq, q, idx_k, idx_v, cb, s1, s2, n2, nrm, mean_k, mean_v, cos_sin,
              kv_indptr, kv_indices, mean_idx, max_len, ws, splits=16, dot_f16=True,
              block_n=16, block_h=4, hadamard=None, inv_freq=None, fold_k=False,
              no_scalars=False, scal=None, fold_mean=False, return_lse=False,
              native=None, nw_span=None, raw=None, pos_offset=0):
    """One fused pass + CQ's stage 2. Returns [B, H, D]; the caller applies the final Hadamard
    (pass `hadamard` to have it done here). mean_k is pre-RoPE; mean_v is ALREADY H-rotated."""
    B, H, D = q.shape
    KVH, NG = idx_k.shape[1], idx_k.shape[2]
    KC, G = cb.shape[-2], cb.shape[-1]
    kvg = H // KVH
    # NATIVE reads nsn_packed_store's own layout: 4-bit norm sharing a byte with the other
    # side's, fp16 norm2, and the 4-bit window mean, all dequantised in-kernel. Materialising
    # them would cost more traffic per step than the packed store saves.
    nat = native is not None
    pack_sc = (scal is not None) and not nat
    if not pack_sc:
        scal = s1
    if nat:
        # the arena is flat, so the span's window count is passed rather than read off a mean
        assert nw_span is not None, "native mode needs nw_span"
        if mean_k is None:
            mean_k = mean_v = s1 = s2 = n2 = nrm = scal = idx_k
        if cos_sin is None:
            cos_sin = idx_k          # unused when TRIG_ONFLY
        n2p, nrm8 = native["n2"], native["nrm8"]
        nsc, mkq, mvq = native["nsc"], native["mk_q"], native["mv_q"]
        mks, mvs, winmap = native["mk_s"], native["mv_s"], native["win_map"]
        nwmax = winmap.shape[1]
    else:
        n2p = nrm8 = nsc = mkq = mvq = mks = mvs = winmap = s1
        nwmax = 1
    NW = nw_span if nat else mean_k.shape[1]
    # BLOCK_H must DIVIDE kv_group_num, or cover it whole. CQ ships BLOCK_H=4 and the head->KV
    # mapping `cur_head_id // cdiv(kvg, BLOCK_H)` is only right in those two regimes: at kvg=6
    # a block of 4 straddles two KV heads (heads 4-7 with heads 6,7 belonging to KV head 1) and
    # the kernel reads the wrong store -- silently, 5.7e-01 relative, not a crash. This is the
    # tl.arange also forces a power of two, so at kvg=6 the kernel covers the whole group and
    # masks the padding lanes.
    if kvg <= 16:
        block_h = triton.next_power_of_2(kvg)
    else:
        block_h = max(d for d in (1, 2, 4, 8, 16) if kvg % d == 0)
    assert kvg % block_h == 0 or block_h >= kvg
    assert ws % block_n == 0, f"ws={ws} must be a multiple of BLOCK_N={block_n}"
    trig_onfly = inv_freq is not None
    if not trig_onfly:
        inv_freq = q.new_zeros(1)
    W = D if fold_mean else 2 * D
    # RAW fold: raw-tail programs appended to the split axis (see the kernel)
    raw_on = raw is not None
    raw_splits = raw["splits"] if raw_on else 0
    total_splits = splits + raw_splits
    if raw_on:
        Qr, RK, RV, r_ptr, r_ind = raw["q"], raw["k"], raw["v"], raw["indptr"], raw["indices"]
        assert not fold_mean, "raw fold uses the un-rotated (mean) half"
    else:
        Qr = RK = RV = q
        r_ptr = r_ind = kv_indptr
    o = torch.empty((B, H, W), device=q.device, dtype=torch.float32)
    # No memset here: the kernel stores every split, empty ones included (see its epilogue).
    # `decode_attention_fwd_vq2_unified` instead fills the LSE with -inf from the host, which
    # costs a pass over the buffer on every call.
    att_out = torch.empty((B, H, total_splits, W), device=q.device, dtype=torch.float32)
    att_lse = torch.empty((B, H, total_splits), device=q.device, dtype=torch.float32)
    nks = _nks_for(B, splits, q.device)
    grid = (B, triton.cdiv(H, min(block_h, kvg)), total_splits)
    _nsn_fused_stage1[grid](
        hq, q, idx_k, idx_v, cb, s1, s2, n2, nrm, scal,
        n2p, nrm8, nsc, mkq, mvq, mks, mvs, winmap,
        mean_k, mean_v, cos_sin, inv_freq,
        mean_idx,
        kv_indptr, kv_indices,
        Qr, RK, RV, r_ptr, r_ind,
        Qr.stride(0), Qr.stride(1), RK.stride(0), RK.stride(1), RV.stride(0), RV.stride(1),
        float(raw["sm_scale"]) if raw_on else 1.0,
        int(pos_offset),
        att_out, att_lse, nks,
        q.stride(0), q.stride(1),
        idx_k.stride(0), idx_k.stride(1),
        scal.stride(0) if pack_sc else s1.stride(0),
        mean_k.stride(0), mean_k.stride(1), mean_k.stride(2),
        cos_sin.stride(0),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        kv_group_num=kvg, q_head_num=H, KVH_=KVH, D=D, D2=D // 2, NG=NG, G=G, KC=KC, WS=ws, NW=NW,
        BLOCK_N=block_n, BLOCK_H=block_h, MIN_BLOCK_KV=block_n, DOT_F16=dot_f16,
        TRIG_ONFLY=trig_onfly, FOLD_K=fold_k, NO_SCALARS=no_scalars, PACK_SC=pack_sc,
        FOLD_MEAN=fold_mean, NATIVE=nat, G32=32, NWMAX=nwmax,
        RAW=raw_on, SPLITS_VQ=splits, RAW_SPLITS=raw_splits,
        num_warps=1, num_stages=1,
    )
    # A two-tier read needs this tier's LSE so the raw tail can be merged against it.
    olse = torch.empty((B, H), device=q.device, dtype=torch.float32) if return_lse else None
    _unified_stage2(att_out, att_lse, o, total_splits=total_splits, output_lse=olse)
    ovq = o[..., :D]
    om = None if fold_mean else o[..., D:]
    if hadamard is None:
        return ovq, om
    hv = hadamard(ovq.reshape(B * H, 1, D).contiguous()).reshape(B, H, D)
    out = hv if fold_mean else hv + om
    return (out, olse) if return_lse else out
