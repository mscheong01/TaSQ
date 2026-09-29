# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import functools
import json
import logging
import math
import os

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip
from sglang.srt.environ import envs

_is_hip = is_hip()

logger = logging.getLogger(__name__)


_MIN_BLOCK_KV = 32


def _get_scale_group_size(head_dim: int, scales_zeros) -> int:
    """Return the per-group head-dim span for quantized KV scales.

    ``scales_zeros`` has last-dim layout ``[scale_0, zero_0, scale_1, zero_1, ...]``,
    i.e. ``2 * num_groups`` entries. Returns ``head_dim // num_groups``; when the
    cache uses a single scale/zero pair per head this equals ``head_dim``.
    """
    num_groups = scales_zeros.shape[-1] // 2
    if head_dim % num_groups != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by quant scale groups ({num_groups})"
        )
    return head_dim // num_groups


def _get_shared_kv_scale_group_size(
    Lk: int, Lv: int, k_scales_zeros, v_scales_zeros
) -> int:
    """Return the shared configured INT2 KV group size.

    K and V may have different head dims in MLA/DPE-style layouts, so the
    scalar one-group case can report different per-tensor group sizes. Once
    either side is actually grouped, the configured group size must match.
    """
    k_group_size = _get_scale_group_size(Lk, k_scales_zeros)
    v_group_size = _get_scale_group_size(Lv, v_scales_zeros)
    k_grouped = k_group_size < Lk
    v_grouped = v_group_size < Lv

    if (k_grouped or v_grouped) and k_group_size != v_group_size:
        raise ValueError(
            "INT2 KV cache requires K and V to use the same quant group size "
            f"when grouped, got K={k_group_size}, V={v_group_size}"
        )
    return k_group_size if (k_grouped or v_grouped) else max(k_group_size, v_group_size)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    logit_cap,
    xai_temperature_len=-1,
):
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
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

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    logit_cap,
    xai_temperature_len=-1,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


@triton.jit
def _fwd_grouped_kernel_stage1_prerope(
    Q,
    K_Buffer,    # [cache_size, num_kv_heads, L] -- stored PRE-RoPE
    V_Buffer,
    CosSin,      # [max_pos, L] fp32 -- cache_sin_cache layout: [cos(L/2), sin(L/2)]
    Full_Seq_Len,  # [batch] int -- this request's TOTAL sequence length so far
    FreqIdx,     # [H, L//2] int64, only read when PERM_ROPE (dummy otherwise)
    WPerm,       # [H, L] fp32, only read when PERM_ROPE (dummy otherwise)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_cs_pos,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    L: tl.constexpr,
    LV: tl.constexpr,
    PERM_ROPE: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
):
    """HP/exact-tier stage-1 for pre-RoPE caches: K_Buffer holds RAW (un-rotated)
    K, matching this project's own CQ/TaSQ reference (KVQuantQuantizer stores
    key_states before RoPE, see src/models/qwen3.py's post_rope=False branch --
    the residual/exact tier is pre-RoPE too, not just the quantized bulk).
    RoPE is instead applied here, per cached row, using that row's own
    absolute sequence position -- standard NeoX rotate-half, matching
    sglang's own RotaryEmbedding.forward_native exactly (cos/sin from the
    model's own cos_sin_cache, so this cannot drift from whatever
    base/scaling config the model uses).

    HP indices are scattered in position order, so with PREFIX_TOKENS>0 the tier
    is two bands: the sink rows at [0, PREFIX_TOKENS) followed by the recent tail
    at [full_len - recent, full_len). Rows in the first band are at ``offs_n``
    itself; the rest keep the ``Full_Seq_Len``-derived offset, which is exact for
    a contiguous tail of any length. PREFIX_TOKENS==0 makes the first branch
    unreachable, preserving the no-prefix execution path.

    A dedicated copy of :func:`_fwd_grouped_kernel_stage1` rather than a flag
    on it: that function is the exact-tier kernel for every OTHER method
    (the VQ arm, quarot, turboquant), and must not change behavior for
    them.
    """
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

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < LV
    offs_half = tl.arange(0, BLOCK_D // 2)
    mask_half = offs_half < (L // 2)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)
    # Offset for the recent band: its rows are the last (hp_len - PREFIX_TOKENS)
    # positions of the sequence. Sink rows bypass this, see abs_pos below.
    pos_offset = tl.load(Full_Seq_Len + cur_batch) - cur_batch_seq_len

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        if PERM_ROPE:
            # HP/exact tier for perm+weight methods (this project's TaSQ):
            # the VQ pool's own _prepare_hp_kv_tensors applies `forward` (permute+
            # weight) before writing HP rows too, same as the quant tier --
            # for identity `forward` (CQ) that is a no-op, which is why this
            # branch is new here and the non-perm path below still matches
            # CQ exactly. Same fix as the quant-tier vqwide kernel: unweight
            # (WPerm) -> rotate with the ORIGINAL per-pair frequency index
            # (FreqIdx) -> no reweight, landing on plain original channels;
            # Q is gathered at those same channels, never mapped.
            offs_pair = tl.arange(0, BLOCK_D // 2)
            mask_pair = offs_pair < (L // 2)
            fidx = tl.load(FreqIdx + cur_kv_head * (L // 2) + offs_pair, mask=mask_pair, other=0).to(tl.int64)
            w_even = tl.load(WPerm + cur_kv_head * L + 2 * offs_pair, mask=mask_pair, other=1.0).to(tl.float32)
            w_odd = tl.load(WPerm + cur_kv_head * L + 2 * offs_pair + 1, mask=mask_pair, other=1.0).to(tl.float32)

            offs_q1 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + fidx[None, :]
            offs_q2 = offs_q1 + L // 2
            q1 = tl.load(Q + offs_q1, mask=(mask_h[:, None]) & (mask_pair[None, :]), other=0.0).to(tl.float32)
            q2 = tl.load(Q + offs_q2, mask=(mask_h[:, None]) & (mask_pair[None, :]), other=0.0).to(tl.float32)
        else:
            # q1/q2: the two NeoX rotate-half halves of the (already-RoPE'd, at
            # the query's own position) query -- loaded as two separate halves
            # rather than one full load + slice, since Triton tensors don't
            # support mid-axis integer/slice indexing after the fact (only fresh
            # loads with computed offsets).
            offs_q1 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_half[None, :]
            offs_q2 = offs_q1 + L // 2
            q1 = tl.load(Q + offs_q1, mask=(mask_h[:, None]) & (mask_half[None, :]), other=0.0).to(tl.float32)
            q2 = tl.load(Q + offs_q2, mask=(mask_h[:, None]) & (mask_half[None, :]), other=0.0).to(tl.float32)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=mask_n,
                other=0,
            ).to(tl.int64)

            k_base = kv_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh

            # Sink rows use their absolute prefix positions.
            abs_pos = tl.where(offs_n < PREFIX_TOKENS, offs_n, pos_offset + offs_n)

            if PERM_ROPE:
                k1 = tl.load(
                    K_Buffer + k_base + (2 * offs_pair)[None, :],
                    mask=mask_n[:, None] & mask_pair[None, :], other=0.0,
                ).to(tl.float32)
                k2 = tl.load(
                    K_Buffer + k_base + (2 * offs_pair + 1)[None, :],
                    mask=mask_n[:, None] & mask_pair[None, :], other=0.0,
                ).to(tl.float32)

                cs_row = tl.load(
                    CosSin + abs_pos[:, None] * stride_cs_pos + fidx[None, :],
                    mask=mask_n[:, None] & mask_pair[None, :], other=0.0,
                ).to(tl.float32)
                sn_row = tl.load(
                    CosSin + abs_pos[:, None] * stride_cs_pos + (L // 2 + fidx)[None, :],
                    mask=mask_n[:, None] & mask_pair[None, :], other=0.0,
                ).to(tl.float32)
                k1u = k1 / w_even[None, :]
                k2u = k2 / w_odd[None, :]
                o1 = k1u * cs_row - k2u * sn_row
                o2 = k2u * cs_row + k1u * sn_row
            else:
                k1 = tl.load(
                    K_Buffer + k_base + offs_half[None, :],
                    mask=mask_n[:, None] & mask_half[None, :], other=0.0,
                ).to(tl.float32)
                k2 = tl.load(
                    K_Buffer + k_base + (L // 2 + offs_half)[None, :],
                    mask=mask_n[:, None] & mask_half[None, :], other=0.0,
                ).to(tl.float32)

                cs_row = tl.load(
                    CosSin + abs_pos[:, None] * stride_cs_pos + offs_half[None, :],
                    mask=mask_n[:, None] & mask_half[None, :],
                    other=0.0,
                ).to(tl.float32)
                sn_row = tl.load(
                    CosSin
                    + abs_pos[:, None] * stride_cs_pos
                    + (L // 2 + offs_half)[None, :],
                    mask=mask_n[:, None] & mask_half[None, :],
                    other=0.0,
                ).to(tl.float32)
                # NeoX rotate-half: o1 = k1*cos - k2*sin, o2 = k2*cos + k1*sin.
                # q . k_rot = q1.o1 + q2.o2 -- computed as two partial dots
                # instead of concatenating o1/o2 into one BLOCK_D-wide row
                # (Triton has no `torch.cat`; `tl.join` interleaves rather
                # than concatenates, which would silently scramble channel
                # order here).
                o1 = k1 * cs_row - k2 * sn_row
                o2 = k2 * cs_row + k1 * sn_row

            qk = tl.dot(q1.to(tl.float16), tl.trans(o1).to(tl.float16))
            qk += tl.dot(q2.to(tl.float16), tl.trans(o2).to(tl.float16))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & mask_n[None, :], qk, float("-inf")
            )

            v = tl.load(
                V_Buffer
                + kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :],
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // LV
        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd_prerope(
    q,
    k_buffer,
    v_buffer,
    cos_sin_cache,
    full_seq_len,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
    freq_idx=None,
    w_perm=None,
    prefix_tokens=0,
):
    """Launcher for :func:`_fwd_grouped_kernel_stage1_prerope`. See that
    kernel's docstring -- this is the HP/exact-tier counterpart of
    :func:`_decode_grouped_att_m_fwd_quant_vqwide`'s pre-RoPE handling.

    full_seq_len: [batch] int -- each request's total sequence length so far
    (prefill + generated tokens up to and including this step); combined with
    this tier's own per-request length (from kv_indptr) to get each cached
    row's absolute position.

    prefix_tokens: SGLANG_MIXED_KV_PREFIX_TOKENS. Rows below it are the always-exact
    sink band and are rotated at their own index.

    freq_idx/w_perm != None activates PERM_ROPE (this project's TaSQ): the
    HP/exact buffer holds permuted+weighted (not raw) K, same as the quant
    tier -- see the kernel docstring."""
    perm_rope = freq_idx is not None
    if not perm_rope:
        freq_idx = q.new_zeros(1, dtype=torch.int64)
        w_perm = q.new_zeros(1)
    L = k_buffer.shape[-1]
    LV = v_buffer.shape[-1]
    assert L % 2 == 0, "RoPE rotate-half requires an even head_dim"

    BLOCK_D = triton.next_power_of_2(L)
    BLOCK_DV = triton.next_power_of_2(LV)
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK = int(os.environ.get("SGL_PREROPE_BLOCK_N", 16))
    BLOCK_H = int(os.environ.get("SGL_PREROPE_BLOCK_H", 4))
    num_warps = int(os.environ.get("SGL_PREROPE_NUM_WARPS", 2))
    num_stages = int(os.environ.get("SGL_PREROPE_NUM_STAGES", 1))

    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        max_kv_splits,
    )

    _fwd_grouped_kernel_stage1_prerope[grid](
        q,
        k_buffer,
        v_buffer,
        cos_sin_cache,
        full_seq_len,
        freq_idx,
        w_perm,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        cos_sin_cache.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        L=L,
        LV=LV,
        PERM_ROPE=perm_rope,
        PREFIX_TOKENS=int(prefix_tokens),
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    O_lse,
    v_scale,
    kv_indptr,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum * v_scale,
        mask=mask_d,
    )
    if WRITE_LSE:
        # Per-seq log-sum-exp = e_max + log(e_sum). O_lse has shape
        # [bs, num_heads]; batch stride = stride_obs // Lv = num_heads.
        tl.store(O_lse + cur_batch * (stride_obs // Lv) + cur_head, e_max + tl.log(e_sum))


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_scale,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
    sinks=None,
    output_lse=None,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None
    WRITE_LSE = output_lse is not None

    extra_kargs = {}
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        output_lse,
        v_scale,
        kv_indptr,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=HAS_SINK,
        WRITE_LSE=WRITE_LSE,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
        output_lse=output_lse,
    )


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
        output_lse=output_lse,
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    k_scale,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale * k_scale,
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            output_lse=output_lse,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale * k_scale,
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            output_lse=output_lse,
        )


def decode_attention_fwd_quantized(
    q,
    k_buffer,  # Quantized INT2 packed uint8
    v_buffer,  # Quantized INT2 packed uint8
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    kv_dtype,  # must be "int2"
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    """
    Attention forward with INT2 quantized KV cache.
    Dispatches between MHA and GQA/MQA paths based on ``kv_group_num``.
    """
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]
    assert kv_dtype == "int2", f"Only int2 quant KV is supported, got {kv_dtype}"

    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        decode_attention_fwd_normal_quant_int2(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            output_lse=output_lse,
        )
    else:
        decode_attention_fwd_grouped_quant_int2(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            output_lse=output_lse,
        )


# ---------------------------------------------------------------------------
# INT2 quantized decode attention kernels
# ---------------------------------------------------------------------------
# INT2 packs 4 values per byte (2-bit crumbs).  Storage is head_dim // 4
# packed uint8 bytes.  Unpacking uses masks 0x03, shifts >> 2, >> 4, >> 6.
# ---------------------------------------------------------------------------


@triton.jit
def _fwd_kernel_stage1_quant_int2(
    Q,
    K_Buffer,  # Quantized INT2 [cache_size, num_heads, head_dim//4] uint8 (packed)
    V_Buffer,  # Quantized INT2 [cache_size, num_heads, head_dim//4] uint8 (packed)
    K_Scales_Zeros,  # [cache_size, num_heads, 2*k_groups] float32, interleaved scale/zero pairs
    V_Scales_Zeros,  # [cache_size, num_heads, 2*v_groups] float32
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,
    stride_sz_kh,
    stride_sz_vbs,
    stride_sz_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    K_GROUPED: tl.constexpr = GROUP_SIZE < Lk
    V_GROUPED: tl.constexpr = GROUP_SIZE < Lv

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    # For INT2, work with 4 quarters separately
    acc_q0 = tl.zeros([BLOCK_DV // 4], dtype=tl.float32)
    acc_q1 = tl.zeros([BLOCK_DV // 4], dtype=tl.float32)
    acc_q2 = tl.zeros([BLOCK_DV // 4], dtype=tl.float32)
    acc_q3 = tl.zeros([BLOCK_DV // 4], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        offs_d_quarter = tl.arange(0, BLOCK_DMODEL // 4)
        mask_d_quarter = offs_d_quarter < (Lk // 4)

        q_main = tl.load(Q + off_q, mask=mask_d, other=0.0)
        # Split Q into 4 quarters
        q_q0 = tl.where(mask_d_quarter, tl.gather(q_main, offs_d_quarter, 0), 0.0)
        idx_q1 = (Lk // 4) + offs_d_quarter
        q_q1 = tl.where(mask_d_quarter, tl.gather(q_main, idx_q1, 0), 0.0)
        idx_q2 = 2 * (Lk // 4) + offs_d_quarter
        q_q2 = tl.where(mask_d_quarter, tl.gather(q_main, idx_q2, 0), 0.0)
        idx_q3 = 3 * (Lk // 4) + offs_d_quarter
        q_q3 = tl.where(mask_d_quarter, tl.gather(q_main, idx_q3, 0), 0.0)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load packed INT2 K (uint8, 4 values per byte)
            offs_d_packed = tl.arange(0, BLOCK_DMODEL // 4)
            mask_d_packed = offs_d_packed < (Lk // 4)

            offs_buf_k_packed = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d_packed[None, :]
            )
            k_quant_packed = tl.load(
                K_Buffer + offs_buf_k_packed,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                other=0,
            )

            # Load scales and zeros for K
            if K_GROUPED:
                offs_group_k_q0 = offs_d_packed // GROUP_SIZE
                offs_group_k_q1 = (offs_d_packed + (Lk // 4)) // GROUP_SIZE
                offs_group_k_q2 = (offs_d_packed + 2 * (Lk // 4)) // GROUP_SIZE
                offs_group_k_q3 = (offs_d_packed + 3 * (Lk // 4)) // GROUP_SIZE
                safe_group_k_q0 = tl.where(mask_d_packed, offs_group_k_q0, 0)
                safe_group_k_q1 = tl.where(mask_d_packed, offs_group_k_q1, 0)
                safe_group_k_q2 = tl.where(mask_d_packed, offs_group_k_q2, 0)
                safe_group_k_q3 = tl.where(mask_d_packed, offs_group_k_q3, 0)
                offs_sz_k = (
                    kv_loc[:, None] * stride_sz_kbs + cur_kv_head * stride_sz_kh
                )
                k_scale_q0 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q0[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=1.0,
                )
                k_zero_q0 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q0[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=0.0,
                )
                k_scale_q1 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q1[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=1.0,
                )
                k_zero_q1 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q1[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=0.0,
                )
                k_scale_q2 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q2[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=1.0,
                )
                k_zero_q2 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q2[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=0.0,
                )
                k_scale_q3 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q3[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=1.0,
                )
                k_zero_q3 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q3[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=0.0,
                )
                # Dequantize INT2 K inline: unpack 4 crumbs and dequantize per-group.
                k_q0 = (
                    ((k_quant_packed & 0x03).to(tl.float32) - k_zero_q0)
                    * k_scale_q0
                ).to(q_q0.dtype)
                k_q1 = (
                    (((k_quant_packed >> 2) & 0x03).to(tl.float32) - k_zero_q1)
                    * k_scale_q1
                ).to(q_q0.dtype)
                k_q2 = (
                    (((k_quant_packed >> 4) & 0x03).to(tl.float32) - k_zero_q2)
                    * k_scale_q2
                ).to(q_q0.dtype)
                k_q3 = (
                    (((k_quant_packed >> 6) & 0x03).to(tl.float32) - k_zero_q3)
                    * k_scale_q3
                ).to(q_q0.dtype)
            else:
                offs_sz_k_1d = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
                k_scale_1d = tl.load(
                    K_Scales_Zeros + offs_sz_k_1d + 0,
                    mask=offs_n < split_kv_end,
                    other=1.0,
                )
                k_zero_1d = tl.load(
                    K_Scales_Zeros + offs_sz_k_1d + 1,
                    mask=offs_n < split_kv_end,
                    other=0.0,
                )
                k_q0 = (
                    ((k_quant_packed & 0x03).to(tl.float32) - k_zero_1d[:, None])
                    * k_scale_1d[:, None]
                ).to(q_q0.dtype)
                k_q1 = (
                    (((k_quant_packed >> 2) & 0x03).to(tl.float32) - k_zero_1d[:, None])
                    * k_scale_1d[:, None]
                ).to(q_q0.dtype)
                k_q2 = (
                    (((k_quant_packed >> 4) & 0x03).to(tl.float32) - k_zero_1d[:, None])
                    * k_scale_1d[:, None]
                ).to(q_q0.dtype)
                k_q3 = (
                    (((k_quant_packed >> 6) & 0x03).to(tl.float32) - k_zero_1d[:, None])
                    * k_scale_1d[:, None]
                ).to(q_q0.dtype)

            # Compute QK from 4 partial dot products
            qk = (
                tl.sum(q_q0[None, :] * k_q0, 1)
                + tl.sum(q_q1[None, :] * k_q1, 1)
                + tl.sum(q_q2[None, :] * k_q2, 1)
                + tl.sum(q_q3[None, :] * k_q3, 1)
            )
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            # Load packed INT2 V
            offs_dv_packed = tl.arange(0, BLOCK_DV // 4)
            mask_dv_packed = offs_dv_packed < (Lv // 4)

            offs_buf_v_packed = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv_packed[None, :]
            )
            v_quant_packed = tl.load(
                V_Buffer + offs_buf_v_packed,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                other=0,
            )

            # Load scales and zeros for V
            if V_GROUPED:
                offs_group_v_q0 = offs_dv_packed // GROUP_SIZE
                offs_group_v_q1 = (offs_dv_packed + (Lv // 4)) // GROUP_SIZE
                offs_group_v_q2 = (offs_dv_packed + 2 * (Lv // 4)) // GROUP_SIZE
                offs_group_v_q3 = (offs_dv_packed + 3 * (Lv // 4)) // GROUP_SIZE
                safe_group_v_q0 = tl.where(mask_dv_packed, offs_group_v_q0, 0)
                safe_group_v_q1 = tl.where(mask_dv_packed, offs_group_v_q1, 0)
                safe_group_v_q2 = tl.where(mask_dv_packed, offs_group_v_q2, 0)
                safe_group_v_q3 = tl.where(mask_dv_packed, offs_group_v_q3, 0)
                offs_sz_v = (
                    kv_loc[:, None] * stride_sz_vbs + cur_kv_head * stride_sz_vh
                )
                v_scale_q0 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q0[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=1.0,
                )
                v_zero_q0 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q0[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=0.0,
                )
                v_scale_q1 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q1[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=1.0,
                )
                v_zero_q1 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q1[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=0.0,
                )
                v_scale_q2 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q2[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=1.0,
                )
                v_zero_q2 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q2[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=0.0,
                )
                v_scale_q3 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q3[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=1.0,
                )
                v_zero_q3 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q3[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=0.0,
                )
                # Dequantize INT2 V inline: unpack 4 crumbs per-group.
                v_q0 = (
                    ((v_quant_packed & 0x03).to(tl.float32) - v_zero_q0)
                    * v_scale_q0
                ).to(q_q0.dtype)
                v_q1 = (
                    (((v_quant_packed >> 2) & 0x03).to(tl.float32) - v_zero_q1)
                    * v_scale_q1
                ).to(q_q0.dtype)
                v_q2 = (
                    (((v_quant_packed >> 4) & 0x03).to(tl.float32) - v_zero_q2)
                    * v_scale_q2
                ).to(q_q0.dtype)
                v_q3 = (
                    (((v_quant_packed >> 6) & 0x03).to(tl.float32) - v_zero_q3)
                    * v_scale_q3
                ).to(q_q0.dtype)
            else:
                offs_sz_v_1d = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
                v_scale_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 0,
                    mask=offs_n < split_kv_end,
                    other=1.0,
                )
                v_zero_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 1,
                    mask=offs_n < split_kv_end,
                    other=0.0,
                )
                v_q0 = (
                    ((v_quant_packed & 0x03).to(tl.float32) - v_zero_1d[:, None])
                    * v_scale_1d[:, None]
                ).to(q_q0.dtype)
                v_q1 = (
                    (((v_quant_packed >> 2) & 0x03).to(tl.float32) - v_zero_1d[:, None])
                    * v_scale_1d[:, None]
                ).to(q_q0.dtype)
                v_q2 = (
                    (((v_quant_packed >> 4) & 0x03).to(tl.float32) - v_zero_1d[:, None])
                    * v_scale_1d[:, None]
                ).to(q_q0.dtype)
                v_q3 = (
                    (((v_quant_packed >> 6) & 0x03).to(tl.float32) - v_zero_1d[:, None])
                    * v_scale_1d[:, None]
                ).to(q_q0.dtype)

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)

            # Accumulate separately for 4 quarters
            acc_q0 *= re_scale
            acc_q1 *= re_scale
            acc_q2 *= re_scale
            acc_q3 *= re_scale
            acc_q0 += tl.sum(p[:, None] * v_q0, 0)
            acc_q1 += tl.sum(p[:, None] * v_q1, 0)
            acc_q2 += tl.sum(p[:, None] * v_q2, 0)
            acc_q3 += tl.sum(p[:, None] * v_q3, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        # Store 4 quarters separately
        # Quarter 0: indices [0, Lv//4)
        offs_dv_q0 = tl.arange(0, BLOCK_DV // 4)
        mask_dv_quarter = offs_dv_q0 < (Lv // 4)
        offs_mid_o_q0 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_q0
        )
        tl.store(
            Att_Out + offs_mid_o_q0,
            acc_q0 / e_sum,
            mask=mask_dv_quarter,
        )

        # Quarter 1: indices [Lv//4, Lv//2)
        offs_dv_q1 = tl.arange(0, BLOCK_DV // 4)
        offs_mid_o_q1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_q1
            + Lv // 4
        )
        tl.store(
            Att_Out + offs_mid_o_q1,
            acc_q1 / e_sum,
            mask=mask_dv_quarter,
        )

        # Quarter 2: indices [Lv//2, 3*Lv//4)
        offs_dv_q2 = tl.arange(0, BLOCK_DV // 4)
        offs_mid_o_q2 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_q2
            + 2 * (Lv // 4)
        )
        tl.store(
            Att_Out + offs_mid_o_q2,
            acc_q2 / e_sum,
            mask=mask_dv_quarter,
        )

        # Quarter 3: indices [3*Lv//4, Lv)
        offs_dv_q3 = tl.arange(0, BLOCK_DV // 4)
        offs_mid_o_q3 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_q3
            + 3 * (Lv // 4)
        )
        tl.store(
            Att_Out + offs_mid_o_q3,
            acc_q3 / e_sum,
            mask=mask_dv_quarter,
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_grouped_kernel_stage1_quant_int2(
    Q,
    K_Buffer,  # Quantized INT2 [cache_size, num_heads, head_dim//4] uint8 (packed)
    V_Buffer,  # Quantized INT2 [cache_size, num_heads, head_dim//4] uint8 (packed)
    K_Scales_Zeros,  # [cache_size, num_heads, 2*groups] float32, interleaved scale/zero pairs
    V_Scales_Zeros,  # [cache_size, num_heads, 2*groups] float32
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,  # K scales_zeros stride for cache
    stride_sz_kh,  # K scales_zeros stride for head
    stride_sz_vbs,  # V scales_zeros stride for cache
    stride_sz_vh,  # V scales_zeros stride for head
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    L: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    GROUPED: tl.constexpr = GROUP_SIZE < L
    FAST: tl.constexpr = (BLOCK_D // 4) >= GROUP_SIZE

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < L

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    # Use 4 separate accumulators for INT2 quarters
    acc_q0 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q1 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q2 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q3 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        offs_d_q0 = tl.arange(0, BLOCK_D // 4)
        offs_d_q1 = tl.arange(BLOCK_D // 4, 2 * (BLOCK_D // 4))
        offs_d_q2 = tl.arange(2 * (BLOCK_D // 4), 3 * (BLOCK_D // 4))
        offs_d_q3 = tl.arange(3 * (BLOCK_D // 4), BLOCK_D)
        mask_d_quarter = offs_d_q0 < (L // 4)

        q_main = tl.load(
            Q + offs_q,
            mask=(mask_h[:, None]) & (mask_d[None, :]),
            other=0.0,
        )

        q_q0 = tl.where(
            (mask_h[:, None]) & (mask_d_quarter[None, :]),
            tl.gather(
                q_main,
                tl.broadcast_to(offs_d_q0[None, :], [BLOCK_H, BLOCK_D // 4]),
                1,
            ),
            0.0,
        )
        q_q1 = tl.where(
            (mask_h[:, None]) & (mask_d_quarter[None, :]),
            tl.gather(
                q_main,
                tl.broadcast_to(offs_d_q1[None, :], [BLOCK_H, BLOCK_D // 4]),
                1,
            ),
            0.0,
        )
        q_q2 = tl.where(
            (mask_h[:, None]) & (mask_d_quarter[None, :]),
            tl.gather(
                q_main,
                tl.broadcast_to(offs_d_q2[None, :], [BLOCK_H, BLOCK_D // 4]),
                1,
            ),
            0.0,
        )
        q_q3 = tl.where(
            (mask_h[:, None]) & (mask_d_quarter[None, :]),
            tl.gather(
                q_main,
                tl.broadcast_to(offs_d_q3[None, :], [BLOCK_H, BLOCK_D // 4]),
                1,
            ),
            0.0,
        )

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load packed INT2 K in transposed format for efficient dot product
            offs_d_packed = tl.arange(0, BLOCK_D // 4)
            mask_d_packed = offs_d_packed < (L // 4)

            offs_buf_k_packed = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d_packed[:, None]
            )
            k_packed = tl.load(
                K_Buffer + offs_buf_k_packed,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d_packed[:, None]),
                other=0,
            )

            # Load K scales and zeros for dequantization
            if GROUPED:
                # When GROUP_SIZE divides into the per-quarter dim
                # (BLOCK_D // 4), use the fast per-group-load + broadcast
                # path. Otherwise (group spans multiple quarters), fall back
                # to the per-element load.
                if FAST:
                    NUM_GROUPS_QUARTER: tl.constexpr = (BLOCK_D // 4) // GROUP_SIZE
                    offs_grp_k = tl.arange(0, NUM_GROUPS_QUARTER)
                    offs_grp_k_q1 = (BLOCK_D // 4) // GROUP_SIZE + offs_grp_k
                    offs_grp_k_q2 = 2 * (BLOCK_D // 4) // GROUP_SIZE + offs_grp_k
                    offs_grp_k_q3 = 3 * (BLOCK_D // 4) // GROUP_SIZE + offs_grp_k
                    offs_sz_k = (
                        kv_loc[None, :] * stride_sz_kbs + cur_kv_head * stride_sz_kh
                    )
                    k_scale_q0_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k[:, None],
                        mask=offs_n[None, :] < split_kv_end, other=1.0,
                    )
                    k_zero_q0_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k[:, None] + 1,
                        mask=offs_n[None, :] < split_kv_end, other=0.0,
                    )
                    k_scale_q1_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q1[:, None],
                        mask=offs_n[None, :] < split_kv_end, other=1.0,
                    )
                    k_zero_q1_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q1[:, None] + 1,
                        mask=offs_n[None, :] < split_kv_end, other=0.0,
                    )
                    k_scale_q2_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q2[:, None],
                        mask=offs_n[None, :] < split_kv_end, other=1.0,
                    )
                    k_zero_q2_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q2[:, None] + 1,
                        mask=offs_n[None, :] < split_kv_end, other=0.0,
                    )
                    k_scale_q3_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q3[:, None],
                        mask=offs_n[None, :] < split_kv_end, other=1.0,
                    )
                    k_zero_q3_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q3[:, None] + 1,
                        mask=offs_n[None, :] < split_kv_end, other=0.0,
                    )
                    # Broadcast per-group across GROUP_SIZE dims via reshape.
                    k_scale_q0 = tl.reshape(
                        tl.broadcast_to(k_scale_q0_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_zero_q0 = tl.reshape(
                        tl.broadcast_to(k_zero_q0_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_scale_q1 = tl.reshape(
                        tl.broadcast_to(k_scale_q1_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_zero_q1 = tl.reshape(
                        tl.broadcast_to(k_zero_q1_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_scale_q2 = tl.reshape(
                        tl.broadcast_to(k_scale_q2_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_zero_q2 = tl.reshape(
                        tl.broadcast_to(k_zero_q2_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_scale_q3 = tl.reshape(
                        tl.broadcast_to(k_scale_q3_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_zero_q3 = tl.reshape(
                        tl.broadcast_to(k_zero_q3_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                else:
                    # Fallback: group spans multiple quarters. Each quarter is
                    # entirely within a single group, so just load 1 (scale,
                    # zero) per (quarter, token) and broadcast across all dims.
                    offs_sz_k_1d = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
                    grp_q0: tl.constexpr = (0 * (BLOCK_D // 4)) // GROUP_SIZE
                    grp_q1: tl.constexpr = (1 * (BLOCK_D // 4)) // GROUP_SIZE
                    grp_q2: tl.constexpr = (2 * (BLOCK_D // 4)) // GROUP_SIZE
                    grp_q3: tl.constexpr = (3 * (BLOCK_D // 4)) // GROUP_SIZE
                    k_scale_q0_t = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q0,
                                           mask=offs_n < split_kv_end, other=1.0)
                    k_zero_q0_t  = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q0 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    k_scale_q1_t = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q1,
                                           mask=offs_n < split_kv_end, other=1.0)
                    k_zero_q1_t  = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q1 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    k_scale_q2_t = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q2,
                                           mask=offs_n < split_kv_end, other=1.0)
                    k_zero_q2_t  = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q2 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    k_scale_q3_t = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q3,
                                           mask=offs_n < split_kv_end, other=1.0)
                    k_zero_q3_t  = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q3 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    k_scale_q0 = tl.broadcast_to(k_scale_q0_t[None, :], (BLOCK_D // 4, BLOCK_N))
                    k_zero_q0  = tl.broadcast_to(k_zero_q0_t[None, :],  (BLOCK_D // 4, BLOCK_N))
                    k_scale_q1 = tl.broadcast_to(k_scale_q1_t[None, :], (BLOCK_D // 4, BLOCK_N))
                    k_zero_q1  = tl.broadcast_to(k_zero_q1_t[None, :],  (BLOCK_D // 4, BLOCK_N))
                    k_scale_q2 = tl.broadcast_to(k_scale_q2_t[None, :], (BLOCK_D // 4, BLOCK_N))
                    k_zero_q2  = tl.broadcast_to(k_zero_q2_t[None, :],  (BLOCK_D // 4, BLOCK_N))
                    k_scale_q3 = tl.broadcast_to(k_scale_q3_t[None, :], (BLOCK_D // 4, BLOCK_N))
                    k_zero_q3  = tl.broadcast_to(k_zero_q3_t[None, :],  (BLOCK_D // 4, BLOCK_N))
                # Cast scales/zeros to q's dtype ONCE so the per-element dequant
                # below stays entirely in bf16 (saves 2 fp32↔bf16 casts per crumb).
                k_scale_q0 = k_scale_q0.to(q_q0.dtype)
                k_zero_q0  = k_zero_q0.to(q_q0.dtype)
                k_scale_q1 = k_scale_q1.to(q_q0.dtype)
                k_zero_q1  = k_zero_q1.to(q_q0.dtype)
                k_scale_q2 = k_scale_q2.to(q_q0.dtype)
                k_zero_q2  = k_zero_q2.to(q_q0.dtype)
                k_scale_q3 = k_scale_q3.to(q_q0.dtype)
                k_zero_q3  = k_zero_q3.to(q_q0.dtype)
                # Dequantize INT2 K inline: unpack 4 crumbs per-group.
                # k_packed shape: [BLOCK_D//4, BLOCK_N] (transposed)
                k_q0 = ((k_packed & 0x03).to(q_q0.dtype) - k_zero_q0) * k_scale_q0
                k_q1 = (((k_packed >> 2) & 0x03).to(q_q0.dtype) - k_zero_q1) * k_scale_q1
                k_q2 = (((k_packed >> 4) & 0x03).to(q_q0.dtype) - k_zero_q2) * k_scale_q2
                k_q3 = (((k_packed >> 6) & 0x03).to(q_q0.dtype) - k_zero_q3) * k_scale_q3
            else:
                offs_sz_k_1d = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
                k_scale_1d = tl.load(
                    K_Scales_Zeros + offs_sz_k_1d + 0,
                    mask=offs_n < split_kv_end,
                    other=1.0,
                ).to(q_q0.dtype)
                k_zero_1d = tl.load(
                    K_Scales_Zeros + offs_sz_k_1d + 1,
                    mask=offs_n < split_kv_end,
                    other=0.0,
                ).to(q_q0.dtype)
                k_q0 = (
                    (k_packed & 0x03).to(q_q0.dtype) - k_zero_1d[None, :]
                ) * k_scale_1d[None, :]
                k_q1 = (
                    ((k_packed >> 2) & 0x03).to(q_q0.dtype) - k_zero_1d[None, :]
                ) * k_scale_1d[None, :]
                k_q2 = (
                    ((k_packed >> 4) & 0x03).to(q_q0.dtype) - k_zero_1d[None, :]
                ) * k_scale_1d[None, :]
                k_q3 = (
                    ((k_packed >> 6) & 0x03).to(q_q0.dtype) - k_zero_1d[None, :]
                ) * k_scale_1d[None, :]

            # Compute QK as ONE fused MMA instead of 4 small ones by stacking
            # the 4 dequantized quarters into a contiguous D axis.
            # The int2 unpack assigns crumb i to original dim positions
            # [i*L//4, (i+1)*L//4), so concatenating q0|q1|q2|q3 along
            # D reconstructs the natural K layout.
            #
            # We use tl.join (which adds a new last axis) + tl.reshape to
            # interleave: [BLOCK_D//4, BLOCK_N] -> [4, BLOCK_D//4, BLOCK_N]
            # via two binary joins -> permute -> reshape to [BLOCK_D, BLOCK_N].
            k_01 = tl.join(k_q0, k_q1)        # [BLOCK_D//4, BLOCK_N, 2]
            k_23 = tl.join(k_q2, k_q3)        # [BLOCK_D//4, BLOCK_N, 2]
            k_full = tl.join(k_01, k_23)      # [BLOCK_D//4, BLOCK_N, 2, 2]
            k_full = tl.reshape(k_full, (BLOCK_D // 4, BLOCK_N, 4))
            k_full = tl.permute(k_full, (2, 0, 1))      # [4, BLOCK_D//4, BLOCK_N]
            k_full = tl.reshape(k_full, (BLOCK_D, BLOCK_N))

            q_01 = tl.join(q_q0, q_q1)        # [BLOCK_H, BLOCK_D//4, 2]
            q_23 = tl.join(q_q2, q_q3)
            q_full = tl.join(q_01, q_23)      # [BLOCK_H, BLOCK_D//4, 2, 2]
            q_full = tl.reshape(q_full, (BLOCK_H, BLOCK_D // 4, 4))
            q_full = tl.permute(q_full, (0, 2, 1))      # [BLOCK_H, 4, BLOCK_D//4]
            q_full = tl.reshape(q_full, (BLOCK_H, BLOCK_D))

            qk = tl.dot(q_full, k_full)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # Load packed INT2 V and dequantize. V layout: [BLOCK_N, BLOCK_D//4]
            offs_d_packed_v = tl.arange(0, BLOCK_D // 4)
            offs_buf_v_packed = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_d_packed_v[None, :]
            )
            v_packed = tl.load(
                V_Buffer + offs_buf_v_packed,
                mask=(offs_n[:, None] < split_kv_end) & (offs_d_packed_v[None, :] < (L // 4)),
                other=0,
            )

            # Load V scales and zeros for dequantization
            if GROUPED:
                if FAST:
                    # Distinct name from the K block above: Triton >=3.6 rejects re-declaring
                    # the same tl.constexpr twice in one kernel body
                    # ("constexpr cannot be reassigned"), which older Triton allowed.
                    NUM_GROUPS_QUARTER_V: tl.constexpr = (BLOCK_D // 4) // GROUP_SIZE
                    offs_grp_v = tl.arange(0, NUM_GROUPS_QUARTER_V)
                    offs_grp_v_q1 = (BLOCK_D // 4) // GROUP_SIZE + offs_grp_v
                    offs_grp_v_q2 = 2 * (BLOCK_D // 4) // GROUP_SIZE + offs_grp_v
                    offs_grp_v_q3 = 3 * (BLOCK_D // 4) // GROUP_SIZE + offs_grp_v
                    offs_sz_v = (
                        kv_loc[:, None] * stride_sz_vbs + cur_kv_head * stride_sz_vh
                    )
                    v_scale_q0_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v[None, :],
                        mask=offs_n[:, None] < split_kv_end, other=1.0,
                    )
                    v_zero_q0_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v[None, :] + 1,
                        mask=offs_n[:, None] < split_kv_end, other=0.0,
                    )
                    v_scale_q1_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q1[None, :],
                        mask=offs_n[:, None] < split_kv_end, other=1.0,
                    )
                    v_zero_q1_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q1[None, :] + 1,
                        mask=offs_n[:, None] < split_kv_end, other=0.0,
                    )
                    v_scale_q2_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q2[None, :],
                        mask=offs_n[:, None] < split_kv_end, other=1.0,
                    )
                    v_zero_q2_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q2[None, :] + 1,
                        mask=offs_n[:, None] < split_kv_end, other=0.0,
                    )
                    v_scale_q3_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q3[None, :],
                        mask=offs_n[:, None] < split_kv_end, other=1.0,
                    )
                    v_zero_q3_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q3[None, :] + 1,
                        mask=offs_n[:, None] < split_kv_end, other=0.0,
                    )
                    v_scale_q0 = tl.reshape(
                        tl.broadcast_to(v_scale_q0_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER_V, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_zero_q0 = tl.reshape(
                        tl.broadcast_to(v_zero_q0_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER_V, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_scale_q1 = tl.reshape(
                        tl.broadcast_to(v_scale_q1_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER_V, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_zero_q1 = tl.reshape(
                        tl.broadcast_to(v_zero_q1_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER_V, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_scale_q2 = tl.reshape(
                        tl.broadcast_to(v_scale_q2_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER_V, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_zero_q2 = tl.reshape(
                        tl.broadcast_to(v_zero_q2_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER_V, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_scale_q3 = tl.reshape(
                        tl.broadcast_to(v_scale_q3_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER_V, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_zero_q3 = tl.reshape(
                        tl.broadcast_to(v_zero_q3_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER_V, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                else:
                    # Fallback: group spans multiple quarters.
                    offs_sz_v_1d = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
                    v_grp_q0: tl.constexpr = (0 * (BLOCK_D // 4)) // GROUP_SIZE
                    v_grp_q1: tl.constexpr = (1 * (BLOCK_D // 4)) // GROUP_SIZE
                    v_grp_q2: tl.constexpr = (2 * (BLOCK_D // 4)) // GROUP_SIZE
                    v_grp_q3: tl.constexpr = (3 * (BLOCK_D // 4)) // GROUP_SIZE
                    v_scale_q0_t = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q0,
                                           mask=offs_n < split_kv_end, other=1.0)
                    v_zero_q0_t  = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q0 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    v_scale_q1_t = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q1,
                                           mask=offs_n < split_kv_end, other=1.0)
                    v_zero_q1_t  = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q1 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    v_scale_q2_t = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q2,
                                           mask=offs_n < split_kv_end, other=1.0)
                    v_zero_q2_t  = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q2 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    v_scale_q3_t = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q3,
                                           mask=offs_n < split_kv_end, other=1.0)
                    v_zero_q3_t  = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q3 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    v_scale_q0 = tl.broadcast_to(v_scale_q0_t[:, None], (BLOCK_N, BLOCK_D // 4))
                    v_zero_q0  = tl.broadcast_to(v_zero_q0_t[:, None],  (BLOCK_N, BLOCK_D // 4))
                    v_scale_q1 = tl.broadcast_to(v_scale_q1_t[:, None], (BLOCK_N, BLOCK_D // 4))
                    v_zero_q1  = tl.broadcast_to(v_zero_q1_t[:, None],  (BLOCK_N, BLOCK_D // 4))
                    v_scale_q2 = tl.broadcast_to(v_scale_q2_t[:, None], (BLOCK_N, BLOCK_D // 4))
                    v_zero_q2  = tl.broadcast_to(v_zero_q2_t[:, None],  (BLOCK_N, BLOCK_D // 4))
                    v_scale_q3 = tl.broadcast_to(v_scale_q3_t[:, None], (BLOCK_N, BLOCK_D // 4))
                    v_zero_q3  = tl.broadcast_to(v_zero_q3_t[:, None],  (BLOCK_N, BLOCK_D // 4))
                # Cast V scales/zeros to q's dtype ONCE so per-element dequant
                # below stays in bf16 (saves 2 fp32↔bf16 casts per crumb).
                v_scale_q0 = v_scale_q0.to(q_q0.dtype)
                v_zero_q0  = v_zero_q0.to(q_q0.dtype)
                v_scale_q1 = v_scale_q1.to(q_q0.dtype)
                v_zero_q1  = v_zero_q1.to(q_q0.dtype)
                v_scale_q2 = v_scale_q2.to(q_q0.dtype)
                v_zero_q2  = v_zero_q2.to(q_q0.dtype)
                v_scale_q3 = v_scale_q3.to(q_q0.dtype)
                v_zero_q3  = v_zero_q3.to(q_q0.dtype)
                # Dequantize INT2 V inline: unpack 4 crumbs per-group.
                v_q0 = ((v_packed & 0x03).to(q_q0.dtype) - v_zero_q0) * v_scale_q0
                v_q1 = (((v_packed >> 2) & 0x03).to(q_q0.dtype) - v_zero_q1) * v_scale_q1
                v_q2 = (((v_packed >> 4) & 0x03).to(q_q0.dtype) - v_zero_q2) * v_scale_q2
                v_q3 = (((v_packed >> 6) & 0x03).to(q_q0.dtype) - v_zero_q3) * v_scale_q3
            else:
                offs_sz_v_1d = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
                v_scale_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 0,
                    mask=offs_n < split_kv_end,
                    other=1.0,
                ).to(q_q0.dtype)
                v_zero_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 1,
                    mask=offs_n < split_kv_end,
                    other=0.0,
                ).to(q_q0.dtype)
                v_q0 = (
                    (v_packed & 0x03).to(q_q0.dtype) - v_zero_1d[:, None]
                ) * v_scale_1d[:, None]
                v_q1 = (
                    ((v_packed >> 2) & 0x03).to(q_q0.dtype) - v_zero_1d[:, None]
                ) * v_scale_1d[:, None]
                v_q2 = (
                    ((v_packed >> 4) & 0x03).to(q_q0.dtype) - v_zero_1d[:, None]
                ) * v_scale_1d[:, None]
                v_q3 = (
                    ((v_packed >> 6) & 0x03).to(q_q0.dtype) - v_zero_1d[:, None]
                ) * v_scale_1d[:, None]

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            # Scale existing accumulators
            acc_q0 *= re_scale[:, None]
            acc_q1 *= re_scale[:, None]
            acc_q2 *= re_scale[:, None]
            acc_q3 *= re_scale[:, None]

            # Accumulate attention-weighted V for 4 quarters
            acc_q0 += tl.dot(p.to(v_q0.dtype), v_q0)
            acc_q1 += tl.dot(p.to(v_q1.dtype), v_q1)
            acc_q2 += tl.dot(p.to(v_q2.dtype), v_q2)
            acc_q3 += tl.dot(p.to(v_q3.dtype), v_q3)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # Store 4 quarters separately to indices [k*L//4, (k+1)*L//4)
        offs_dv = tl.arange(0, BLOCK_D // 4)
        mask_dv_quarter = offs_dv < (L // 4)
        base_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        tl.store(
            Att_Out + base_mid_o + offs_dv[None, :],
            acc_q0 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )
        tl.store(
            Att_Out + base_mid_o + (offs_dv + L // 4)[None, :],
            acc_q1 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )
        tl.store(
            Att_Out + base_mid_o + (offs_dv + 2 * (L // 4))[None, :],
            acc_q2 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )
        tl.store(
            Att_Out + base_mid_o + (offs_dv + 3 * (L // 4))[None, :],
            acc_q3 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // L

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_att_m_fwd_quant_int2(
    q,
    k_buffer,  # Quantized INT2 (packed)
    v_buffer,  # Quantized INT2 (packed)
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    """
    INT2 quantized KV cache attention wrapper (MHA).
    Dequantizes KV cache on-the-fly inside the kernel.
    """
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    # For INT2, the buffer stores packed values (head_dim//4)
    # But we need to work with the actual head_dim
    Lk = k_buffer.shape[-1] * 4  # Unpack to get real dimension
    Lv = v_buffer.shape[-1] * 4

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)
    group_size = _get_shared_kv_scale_group_size(
        Lk, Lv, k_scales_zeros, v_scales_zeros
    )

    _fwd_kernel_stage1_quant_int2[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        GROUP_SIZE=group_size,
    )


def _decode_grouped_att_m_fwd_quant_int2(
    q,
    k_buffer,  # Quantized INT2 (packed)
    v_buffer,  # Quantized INT2 (packed)
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    """
    INT2 quantized KV cache attention wrapper (GQA/MQA).
    Dequantizes KV cache on-the-fly inside the kernel.

    Tuning history (Qwen3-8B 32Q/8KV, head_dim=128, bs=1, seq=80k, H100):
      knobs                                   | mean ms (seq=80k bs=1)
      ----------------------------------------+----------------------
      BLOCK_N=32  BLOCK_H=16 W=4 S=2 (legacy) | 0.650
      BLOCK_N=32  BLOCK_H=16 W=4 S=3          | 0.165  (+splits=32 default)
      BLOCK_N=128 BLOCK_H=8  W=4 S=3 (current)| 0.096  ← 1.74x over previous tune
    Bigger BLOCK_N amortizes the per-iteration dependency chain (load packed
    crumb → mask/shift → cast → sub zero → mul scale → tl.dot) over more KV
    tokens; smaller BLOCK_H lowers register pressure so more blocks fit per SM.
    """
    # For INT2, k_buffer is packed, so actual head dim is 4x the last dimension.
    # K and V share the same head dim in this path (no MLA/DPE split).
    L = k_buffer.shape[-1] * 4
    assert v_buffer.shape[-1] * 4 == L, "INT2 KV cache requires Lk == Lv"
    BLOCK_D = triton.next_power_of_2(L)
    group_size = _get_shared_kv_scale_group_size(
        L, L, k_scales_zeros, v_scales_zeros
    )

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    MAX_KV_SPLITS = max_kv_splits

    # Tile heuristic
    #
    # Narrow heads (gpt-oss: head_dim=64) return half the payload per gather,
    # so the head_dim=128 ladder below collapses at batch: its bs>=16 branch
    # picks a single-warp 32-token tile. Measured on H100, gpt-oss 64Q/8KV,
    # 30k ctx, us at bs = 1 / 4 / 16 / 58:
    #     head_dim=128 ladder:   83 / 153 / 658 / 2676
    #     this ladder:           80 / 118 / 275 / 1216   (up to 2.4x)
    # head_dim=128 keeps its own selections unchanged.
    if L <= 64 and kv_group_num <= 8:
        if batch >= 32:
            _bn_default, _bh_default, _nw_default = 128, 8, 2
        elif batch >= 16:
            _bn_default, _bh_default, _nw_default = 64, 8, 1
        else:
            _bn_default, _bh_default, _nw_default = 128, 8, 4
    elif kv_group_num <= 8:
        if batch >= 16:
            _bn_default, _bh_default, _nw_default = 32, 4, 1
        elif batch >= 4:
            _bn_default, _bh_default, _nw_default = 64, 8, 2
        else:
            _bn_default, _bh_default, _nw_default = 128, 8, 4
    else:
        _bn_default = 128
        _bh_default = 16 if batch >= 16 else 8
        _nw_default = 4
    BLOCK = int(os.environ.get("SGL_INT2_BLOCK_N", _bn_default))
    BLOCK_H = int(os.environ.get("SGL_INT2_BLOCK_H", _bh_default))
    num_warps = int(os.environ.get("SGL_INT2_NUM_WARPS", _nw_default))
    num_stages = int(os.environ.get("SGL_INT2_NUM_STAGES", 3))

    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    _fwd_grouped_kernel_stage1_quant_int2[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=num_stages,
        L=L,
        GROUP_SIZE=group_size,
        **extra_kargs,
    )


def decode_attention_fwd_normal_quant_int2(
    q,
    k_buffer,  # Quantized INT2
    v_buffer,  # Quantized INT2
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    """
    Normal (MHA) attention forward with INT2 quantized KV cache.
    Dequantizes on-the-fly inside the kernel, avoiding global memory writes.
    """
    # Stage 1: Compute attention scores and accumulate values
    _decode_att_m_fwd_quant_int2(
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
        xai_temperature_len,
    )
    # For INT2, v_buffer is packed (quarter size), but stage2 needs full dimension
    # o has the correct output dimension
    v_buf_for_stage2 = o

    # Stage 2: Reduce across KV splits and compute final output
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale=1.0,
        v_buffer=v_buf_for_stage2,
        kv_indptr=kv_indptr,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sinks=sinks,
        output_lse=output_lse,
    )


def decode_attention_fwd_grouped_quant_int2(
    q,
    k_buffer,  # Quantized INT2
    v_buffer,  # Quantized INT2
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    """
    Grouped (GQA/MQA) attention forward with INT2 quantized KV cache.
    Dequantizes on-the-fly inside the kernel, avoiding global memory writes.
    """
    # Stage 1: Compute attention scores and accumulate values
    _decode_grouped_att_m_fwd_quant_int2(
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
        xai_temperature_len,
    )
    # For INT2, v_buffer is packed (quarter size), but stage2 needs full dimension
    # o has the correct output dimension
    v_buf_for_stage2 = o

    # Stage 2: Reduce across KV splits and compute final output
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale=1.0,
        v_buffer=v_buf_for_stage2,
        kv_indptr=kv_indptr,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sinks=sinks,
        output_lse=output_lse,
    )


@triton.jit
def _fwd_kernel_stage2_unified(
    Mid_O,
    Mid_O_1,
    O,
    O_lse,
    sink_ptr,
    sink_q_ptr,
    sink_mean_ptr,
    v_scale,
    sink_sm_scale,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_sink_qb,
    stride_sink_qh,
    stride_sink_mh,
    TOTAL_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DQ: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    Lv: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SHIFT_VQ_SINK: tl.constexpr,
):
    """Tier-agnostic stage-2 reduction.

    Iterates over ``TOTAL_SPLITS`` splits of the shared scratch buffer and
    accumulates only those with a finite LSE (stage-1 writes -inf into
    unfilled splits before it runs; valid stage-1 programs overwrite with the
    true LSE). Unlike :func:`_fwd_kernel_stage2`, this kernel does not depend
    on ``kv_indptr`` / ``num_kv_splits`` for split-boundary math — the scratch
    itself carries all the information.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv

    for split_id in range(0, TOTAL_SPLITS):
        tlogic = tl.load(Mid_O_1 + offs_logic + split_id * stride_mid_os // Lv)
        if tlogic > -float("inf"):
            tv = tl.load(
                Mid_O + offs_v + split_id * stride_mid_os, mask=mask_d, other=0.0
            )
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # Learned attention sinks (gpt-oss): the sink contributes exp(sink) to
    # the softmax denominator only — same formulation as the stock
    # _fwd_kernel_stage2. Applied only when at least one split was valid
    # (e_max finite), so the empty-seq guard below keeps its semantics.
    if HAS_SINK:
        if e_sum > 0.0:
            cur_sink = tl.load(sink_ptr + cur_head)
            if SHIFT_VQ_SINK:
                offs_q = tl.arange(0, BLOCK_DQ)
                q = tl.load(
                    sink_q_ptr
                    + cur_batch * stride_sink_qb
                    + cur_head * stride_sink_qh
                    + offs_q,
                    mask=offs_q < QK_HEAD_DIM,
                    other=0.0,
                ).to(tl.float32)
                kv_head = cur_head // KV_GROUP_NUM
                mean = tl.load(
                    sink_mean_ptr + kv_head * stride_sink_mh + offs_q,
                    mask=offs_q < QK_HEAD_DIM,
                    other=0.0,
                ).to(tl.float32)
                cur_sink -= tl.sum(q * mean, axis=0) * sink_sm_scale
            e_sum += tl.exp(cur_sink - e_max)

    # Guard against e_sum == 0 (all splits were -inf -> empty seq row).
    # Without this, acc / e_sum yields NaN in o. Match the empty-seq policy
    # of _fwd_kernel_stage2 (store zeros, LSE = -inf).
    safe_e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    out = tl.where(e_sum > 0.0, acc / safe_e_sum * v_scale, 0.0)
    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        out,
        mask=mask_d,
    )
    if WRITE_LSE:
        lse_out = tl.where(e_sum > 0.0, e_max + tl.log(safe_e_sum), -float("inf"))
        tl.store(
            O_lse + cur_batch * (stride_obs // Lv) + cur_head,
            lse_out,
        )


def _unified_stage2(
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    o: torch.Tensor,
    total_splits: int,
    output_lse=None,
    sinks=None,
    sink_q=None,
    sink_mean=None,
    sink_sm_scale=1.0,
):
    batch, head_num = o.shape[0], o.shape[1]
    Lv = o.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)
    shift_vq_sink = sink_q is not None
    assert shift_vq_sink == (sink_mean is not None)
    qk_head_dim = sink_q.shape[-1] if shift_vq_sink else 1
    block_dq = triton.next_power_of_2(qk_head_dim)
    kv_group_num = (
        sink_q.shape[1] // sink_mean.shape[0] if shift_vq_sink else 1
    )
    grid = (batch, head_num)
    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}
    _fwd_kernel_stage2_unified[grid](
        attn_logits,
        attn_lse,
        o,
        output_lse,
        sinks,
        sink_q,
        sink_mean,
        1.0,
        sink_sm_scale,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        sink_q.stride(0) if shift_vq_sink else 0,
        sink_q.stride(1) if shift_vq_sink else 0,
        sink_mean.stride(0) if shift_vq_sink else 0,
        TOTAL_SPLITS=int(total_splits),
        BLOCK_DV=BLOCK_DV,
        BLOCK_DQ=block_dq,
        QK_HEAD_DIM=qk_head_dim,
        KV_GROUP_NUM=kv_group_num,
        Lv=Lv,
        WRITE_LSE=output_lse is not None,
        HAS_SINK=sinks is not None,
        SHIFT_VQ_SINK=shift_vq_sink,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_int2_unified(
    q,
    hp_k_buffer,
    hp_v_buffer,
    quant_k_buffer,
    quant_v_buffer,
    quant_k_scales_zeros,
    quant_v_scales_zeros,
    o,
    hp_kv_indptr,
    hp_kv_indices,
    quant_kv_indptr,
    quant_kv_indices,
    attn_logits,
    attn_lse,
    hp_num_kv_splits,
    quant_num_kv_splits,
    hp_max_kv_splits,
    quant_max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    """Unified HP + int2 decode attention: 2 stage-1 launches + 1 stage-2.

    Scratch layout (allocated by caller; pre-filled with ``-inf`` for LSE so
    that the tier-agnostic stage-2 can skip unused splits):

        attn_logits : [bs, num_heads, hp_max_kv_splits + quant_max_kv_splits, v_head_dim]
        attn_lse    : [bs, num_heads, hp_max_kv_splits + quant_max_kv_splits]

    The HP stage-1 writes splits ``[0, hp_max_kv_splits)``; the quant stage-1
    writes splits ``[hp_max_kv_splits, hp_max_kv_splits + quant_max_kv_splits)``.
    Stage-2 then reduces over the entire split range in a single launch — no
    ``merge_state`` post-process.
    """
    total_splits = hp_max_kv_splits + quant_max_kv_splits
    assert attn_logits.shape[2] == total_splits, (
        f"attn_logits split dim ({attn_logits.shape[2]}) must equal hp_max_kv_splits "
        f"({hp_max_kv_splits}) + quant_max_kv_splits ({quant_max_kv_splits})"
    )

    # Unused splits (smaller sequences that don't use every split) retain a
    # prior call's values because stage-1 early-exits without writing. Reset
    # LSE to -inf so the unified stage-2 correctly skips them.
    attn_lse.fill_(float("-inf"))

    # HP and quant each see their own slice of the shared scratch. Strides on
    # the sliced views are identical to the full tensor so per-split writes
    # continue to address the correct memory.
    hp_logits = attn_logits[:, :, :hp_max_kv_splits, :]
    hp_lse = attn_lse[:, :, :hp_max_kv_splits]
    quant_logits = attn_logits[:, :, hp_max_kv_splits:, :]
    quant_lse = attn_lse[:, :, hp_max_kv_splits:]

    kv_group_num = q.shape[1] // hp_k_buffer.shape[1]

    if hp_kv_indices.numel() > 0:
        if kv_group_num == 1:
            _decode_att_m_fwd(
                q,
                hp_k_buffer,
                hp_v_buffer,
                hp_logits,
                hp_lse,
                hp_kv_indptr,
                hp_kv_indices,
                hp_num_kv_splits,
                hp_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
            )
        else:
            _decode_grouped_att_m_fwd(
                q,
                hp_k_buffer,
                hp_v_buffer,
                hp_logits,
                hp_lse,
                hp_kv_indptr,
                hp_kv_indices,
                hp_num_kv_splits,
                hp_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
            )

    if quant_kv_indices.numel() > 0:
        if kv_group_num == 1:
            _decode_att_m_fwd_quant_int2(
                q,
                quant_k_buffer,
                quant_v_buffer,
                quant_k_scales_zeros,
                quant_v_scales_zeros,
                quant_logits,
                quant_lse,
                quant_kv_indptr,
                quant_kv_indices,
                quant_num_kv_splits,
                quant_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
            )
        else:
            _decode_grouped_att_m_fwd_quant_int2(
                q,
                quant_k_buffer,
                quant_v_buffer,
                quant_k_scales_zeros,
                quant_v_scales_zeros,
                quant_logits,
                quant_lse,
                quant_kv_indptr,
                quant_kv_indices,
                quant_num_kv_splits,
                quant_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
            )

    _unified_stage2(
        attn_logits,
        attn_lse,
        o,
        total_splits=total_splits,
        sinks=sinks,
    )
    return o


@triton.jit
def _vq_fp8_byte_to_f16(b, FP8E4: tl.constexpr):
    """Reinterpret one packed fp8 byte as fp16.

    FP8E4 selects e4m3 (fp8e4nv) over e5m2. It must match the format the
    loader snapped the centroids to (vq_codebook.resolve_vq_fp8_fmt) --
    encoder and decoder have to agree on the same bytes. fp8e4nv is only
    admitted by Triton at compute capability >= 8.9, so sm80 stays on e5m2.
    """
    if FP8E4:
        return b.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float16)
    return b.to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)


@triton.jit
def _fwd_grouped_kernel_stage1_quant_vq2(
    Q,
    K_Idx,       # uint8 [cache_size(+1), num_kv_heads, NG] VQ group indices
    CB,          # int32 [num_kv_heads, NG, KC] packed fp8-e5m2 codewords
    V_Buffer,    # uint8 [cache_size, num_kv_heads, head_dim//4] packed INT2
    V_Idx,       # uint8 [cache_size(+1), num_kv_heads, NG_V] VQ V indices (V_VQ only; else dummy)
    CB_V,        # int32 [num_kv_heads, NG_V, KC_V] packed fp8 V codewords (V_VQ only; else dummy)
    K_Scales_Zeros,  # [cache_size(+1), num_kv_heads, 2]: slot 0 = ptn scale
    V_Scales_Zeros,  # [cache_size, num_kv_heads, 2]: int2 affine, or slot 0 = V ptn scale when V_VQ
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_ki_bs,
    stride_ki_h,
    stride_buf_vbs,
    stride_buf_vh,
    stride_vi_bs,
    stride_vi_h,
    stride_sz_kbs,
    stride_sz_kh,
    stride_sz_vbs,
    stride_sz_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    L: tl.constexpr,
    NG: tl.constexpr,
    KC: tl.constexpr,
    NG_V: tl.constexpr,
    KC_V: tl.constexpr,
    V_VQ: tl.constexpr,
    FP8E4: tl.constexpr,
):
    """Group-VQ K + INT2 V stage-1 (GQA). K gather core ported from an
    earlier fused VQ8 prototype kernel: one int32 load per
    4-coord codeword, 4x fp8-e4m3 bitcast planes, join-interleave to
    [BLOCK_N, L]. K_hat = cb[idx] * ptn_scale; scores fold the per-token
    scale into the qk columns after the dot. V dequant + online softmax +
    quarter accumulators match the int2 kernel's single-scale path.
    """
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

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < L

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_q0 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q1 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q2 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q3 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q_main = tl.load(
            Q + offs_q,
            mask=(mask_h[:, None]) & (mask_d[None, :]),
            other=0.0,
        ).to(tl.float16)

        offs_g = tl.arange(0, NG)
        cb_head_base = CB + cur_kv_head.to(tl.int64) * (NG * KC)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=mask_n,
                other=0,
            ).to(tl.int64)

            # K: gather VQ indices -> packed codewords -> fp8e5m2 planes.
            isel = tl.load(
                K_Idx
                + kv_loc[:, None] * stride_ki_bs
                + cur_kv_head * stride_ki_h
                + offs_g[None, :],
                mask=mask_n[:, None],
                other=0,
            ).to(tl.int32)
            cw = tl.load(
                cb_head_base + offs_g[None, :] * KC + isel,
                mask=mask_n[:, None],
                other=0,
            ).to(tl.int32)
            p0 = _vq_fp8_byte_to_f16(cw & 0xFF, FP8E4)
            p1 = _vq_fp8_byte_to_f16((cw >> 8) & 0xFF, FP8E4)
            p2 = _vq_fp8_byte_to_f16((cw >> 16) & 0xFF, FP8E4)
            p3 = _vq_fp8_byte_to_f16((cw >> 24) & 0xFF, FP8E4)
            # join-interleave: flatten order over the two joined axes is
            # (p0, p1, p2, p3) per group == natural coord order
            # (little-endian int32 packing puts coord i in byte i).
            kg = tl.reshape(
                tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BLOCK_N, NG * 4)
            )

            qk = tl.dot(q_main, tl.trans(kg))  # [BLOCK_H, BLOCK_N] fp32

            # Fold the per-token ptn RMS scale into the score columns:
            # q . (cb * s) == (q . cb) * s.
            k_scale = tl.load(
                K_Scales_Zeros
                + kv_loc * stride_sz_kbs
                + cur_kv_head * stride_sz_kh,
                mask=mask_n,
                other=1.0,
            ).to(tl.float32)
            qk = qk * k_scale[None, :]
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & mask_n[None, :], qk, float("-inf")
            )

            if V_VQ:
                # V: group-VQ, single per-token ptn scale (slot 0). Gather the
                # per-group index -> packed fp8-e5m2 codeword -> 4 planes. With
                # group_dim=4 the planes ARE the v_q0..v_q3 quarters (byte m ==
                # the coord going to output position g+m*NG_V), so no join is
                # needed. Codebook groups are trained strided to match this.
                offs_gv = tl.arange(0, NG_V)
                cbv_head_base = CB_V + cur_kv_head.to(tl.int64) * (NG_V * KC_V)
                iselv = tl.load(
                    V_Idx
                    + kv_loc[:, None] * stride_vi_bs
                    + cur_kv_head * stride_vi_h
                    + offs_gv[None, :],
                    mask=mask_n[:, None],
                    other=0,
                ).to(tl.int32)
                cwv = tl.load(
                    cbv_head_base + offs_gv[None, :] * KC_V + iselv,
                    mask=mask_n[:, None],
                    other=0,
                ).to(tl.int32)
                v_scale_1d = tl.load(
                    V_Scales_Zeros + kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh,
                    mask=mask_n,
                    other=1.0,
                ).to(tl.float16)
                v_q0 = _vq_fp8_byte_to_f16(cwv & 0xFF, FP8E4) * v_scale_1d[:, None]
                v_q1 = _vq_fp8_byte_to_f16((cwv >> 8) & 0xFF, FP8E4) * v_scale_1d[:, None]
                v_q2 = _vq_fp8_byte_to_f16((cwv >> 16) & 0xFF, FP8E4) * v_scale_1d[:, None]
                v_q3 = _vq_fp8_byte_to_f16((cwv >> 24) & 0xFF, FP8E4) * v_scale_1d[:, None]
            else:
                # V: packed INT2, single-scale affine (matches the int2 kernel's
                # non-grouped branch).
                offs_d_packed_v = tl.arange(0, BLOCK_D // 4)
                offs_buf_v_packed = (
                    kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + offs_d_packed_v[None, :]
                )
                v_packed = tl.load(
                    V_Buffer + offs_buf_v_packed,
                    mask=mask_n[:, None] & (offs_d_packed_v[None, :] < (L // 4)),
                    other=0,
                )
                offs_sz_v_1d = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
                v_scale_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 0, mask=mask_n, other=1.0
                ).to(tl.float16)
                v_zero_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 1, mask=mask_n, other=0.0
                ).to(tl.float16)
                v_q0 = ((v_packed & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]
                v_q1 = (((v_packed >> 2) & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]
                v_q2 = (((v_packed >> 4) & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]
                v_q3 = (((v_packed >> 6) & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            acc_q0 *= re_scale[:, None]
            acc_q1 *= re_scale[:, None]
            acc_q2 *= re_scale[:, None]
            acc_q3 *= re_scale[:, None]

            acc_q0 += tl.dot(p.to(v_q0.dtype), v_q0)
            acc_q1 += tl.dot(p.to(v_q1.dtype), v_q1)
            acc_q2 += tl.dot(p.to(v_q2.dtype), v_q2)
            acc_q3 += tl.dot(p.to(v_q3.dtype), v_q3)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_dv = tl.arange(0, BLOCK_D // 4)
        mask_dv_quarter = offs_dv < (L // 4)
        base_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        tl.store(
            Att_Out + base_mid_o + offs_dv[None, :],
            acc_q0 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )
        tl.store(
            Att_Out + base_mid_o + (offs_dv + L // 4)[None, :],
            acc_q1 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )
        tl.store(
            Att_Out + base_mid_o + (offs_dv + 2 * (L // 4))[None, :],
            acc_q2 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )
        tl.store(
            Att_Out + base_mid_o + (offs_dv + 3 * (L // 4))[None, :],
            acc_q3 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // L

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# Build the opt-in CUDA vq2 stage-1 at IMPORT time. sglang sizes the KV pool
# from a free-memory probe during model init; an extension loaded any later
# (backend __init__, or lazily on first decode) has its CUDA module memory
# unaccounted, and a long-context prefill then OOMs mid-run.
try:
    from sglang.srt.layers.attention.triton_ops import vq2_cuda_stage1 as _vq2_cuda

    if _vq2_cuda.enabled():
        _vq2_cuda.prebuild()
except Exception as _e:  # never let the opt-in path break the Triton one
    logger.warning("vq2 CUDA stage-1 unavailable, using Triton: %s", _e)
    _vq2_cuda = None


@functools.lru_cache(maxsize=1)
def _vq2_tuned_table():
    """Load the offline tuning table once, if SGL_VQ2_CONFIG_JSON is set."""
    path = os.environ.get("SGL_VQ2_CONFIG_JSON")
    if not path:
        return None
    try:
        with open(path) as f:
            t = json.load(f)
        return t.get("geometry", {}), {int(k): v for k, v in t["configs"].items()}
    except Exception as e:
        logger.warning("vq2: ignoring SGL_VQ2_CONFIG_JSON=%s: %s", path, e)
        return None


def _vq2_tuned_config(batch, kv_group_num, L, NG, KC):
    """Config for the largest tuned batch bucket <= batch, if the geometry matches."""
    t = _vq2_tuned_table()
    if t is None:
        return None
    geo, cfgs = t
    if (geo.get("head_dim"), geo.get("ng"), geo.get("kc")) != (L, NG, KC):
        return None            # tuned for a different model; use the ladder
    # kv_group_num sets how much work each kv head does, so a table tuned at a
    # different group size is not transferable -- and (head_dim, ng, kc) alone
    # does not distinguish them. A gpt-oss table tuned at h_q=32 (group 4) was
    # silently applied at the real group 8 and cost vq2_triton ~40% at bs=210.
    geo_group = (geo.get("h_q") or 0) // (geo.get("h_kv") or 1)
    if geo_group and geo_group != kv_group_num:
        return None
    buckets = [b for b in sorted(cfgs) if b <= batch] or [min(cfgs)]
    return cfgs[buckets[-1]]


def _decode_grouped_att_m_fwd_quant_vq2(
    q,
    k_idx_buffer,      # uint8 [cache_size(+1), num_kv_heads, NG]
    cb_packed,         # int32 [num_kv_heads, NG, KC]
    v_buffer,          # uint8 packed INT2
    k_scales_zeros,    # ptn scale in slot 0
    v_scales_zeros,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
    cb_packed_v=None,      # int32 [num_kv_heads, NG_V, KC_V] -> group-VQ V tier
    v_idx_buffer=None,     # uint8 [cache_size(+1), num_kv_heads, NG_V]
):
    """vq2 stage-1 launcher (GQA). Defaults from an offline VQ8 tuning sweep
    (A100, Qwen3-8B: BLOCK_N=64, warps=2, stages=2 optimal at 32K and 128K);
    overridable via SGL_VQ2_* env vars, mirroring the SGL_INT2_* knobs.
    When cb_packed_v/v_idx_buffer are given, V is decoded as group-VQ too
    (v_scales_zeros slot 0 must hold the per-token V ptn scale); else int2 V."""
    L = v_buffer.shape[-1] * 4
    NG = k_idx_buffer.shape[-1]
    KC = cb_packed.shape[-1]
    # Same pure resolver the loader used to snap the centroids, so the decode
    # bitcast and the stored bytes cannot disagree (it depends only on env +
    # device capability, and is lru_cached).
    from sglang.srt.mem_cache.vq_codebook import resolve_vq_fp8_fmt

    FP8E4 = resolve_vq_fp8_fmt() == "e4m3"
    V_VQ = cb_packed_v is not None
    if V_VQ:
        NG_V = v_idx_buffer.shape[-1]
        KC_V = cb_packed_v.shape[-1]
        v_idx_arg, cb_v_arg = v_idx_buffer, cb_packed_v
        stride_vi_bs, stride_vi_h = v_idx_buffer.stride(0), v_idx_buffer.stride(1)
        assert NG_V * 4 == L, f"vq2 V expects NG_V*4 == head_dim: NG_V={NG_V}, L={L}"
    else:
        # dummies (unread when V_VQ=False); reuse the K arena's shapes/strides.
        NG_V, KC_V = NG, KC
        v_idx_arg, cb_v_arg = k_idx_buffer, cb_packed
        stride_vi_bs, stride_vi_h = k_idx_buffer.stride(0), k_idx_buffer.stride(1)
    assert cb_packed.shape[-2] == NG and NG * 4 == L, (
        f"vq2 kernel expects NG*G == head_dim: NG={NG}, L={L}"
    )
    assert k_scales_zeros.shape[-1] == 2 and v_scales_zeros.shape[-1] == 2, (
        "vq2 decode requires single-scale K/V layouts"
    )

    # Opt-in CUDA stage-1 (SGLANG_VQ2_CUDA=1): same maths, but the codebook is
    # staged in shared memory instead of gathered from global. Measured 0.94-1.05x
    # int2 and 0.75-0.83x this Triton kernel across four shapes; see
    # vq2_cuda_stage1.py. Default OFF, and `supports()` gates every assumption it
    # bakes in, so the Triton path below stays the one everything else uses.
    if _vq2_cuda is not None and _vq2_cuda.enabled() and _vq2_cuda.supports(
        q, k_idx_buffer, cb_packed, v_buffer, k_scales_zeros, v_scales_zeros,
        att_out, att_lse, kv_indptr, kv_indices, num_kv_splits,
        logit_cap, xai_temperature_len, V_VQ,
    ):
        _vq2_cuda.launch(
            q, k_idx_buffer, cb_packed, v_buffer, k_scales_zeros, v_scales_zeros,
            att_out, att_lse, kv_indptr, kv_indices, num_kv_splits,
            max_kv_splits, sm_scale,
        )
        return

    BLOCK_D = triton.next_power_of_2(L)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_idx_buffer.shape[1]

    # F2 (2026-07): the gather kernel is decode-latency-bound; a live sweep found
    # BLOCK_N=128/warps=4/stages=3 + a high split-K cap (--triton-attention-num-kv-splits
    # 32, set per-arm in serve_oscar.sh) cut 32K decode ~14% (21.4->18.4 ms/tok),
    # halving the vq2-vs-int2 gap. The old 64/2/2 defaults badly under-parallelized it.
    #
    # F3 (2026-07-27): those were single-config, tuned at small batch. int2's
    # launcher has always scaled its config with batch; vq2 never did, and that
    # is most of why the vq2-vs-int2 gap WIDENS with batch (1.14x at bs=1 ->
    # 1.63x at bs=64). Measured on H100 at the served shape (ctx=30k, engine-
    # derived splits, logs/bench_stage1.py): at bs=64 the fixed 128/8/4 runs
    # 2158 us vs 1944 us for 64/8/2 -- 1.11x on the kernel, ~1.05x end-to-end.
    # vq2 wants a LARGER config than int2 at the same batch: its codebook gather
    # is a dependent load that needs more warps in flight to hide, so int2's
    # 32/4/1 large-batch tier measures *worse* here (2171 us) despite fitting
    # more CTAs/SM.
    # F4 (2026-07-27): retuned after reading the compiled PTX. The codebook
    # gather lowers to 32 x cp.async.ca.shared.global per token (source line
    # ~2709), i.e. Triton already stages it through shared -- just once per
    # TOKEN rather than once per CTA. That makes num_stages, the cp.async
    # pipeline depth, a first-class knob rather than a minor one. The previous
    # sweep capped it at 3 and so missed the optimum: at stages=3 BLOCK_H=8
    # wins, but at stages>=4 BLOCK_H=4 wins -- and BLOCK_H=4 exactly matches
    # kv_group_num, so the qk tile carries no padded rows.
    #
    # Measured at ctx=30k, H100, against the int2 kernel (bench_stage1.py):
    #   bs=1    105.3 -> 91.1 us   (1.09x -> 0.94x int2)
    #   bs=4    148.7 -> 144.2     (1.02x -> 0.99x)
    #   bs=16   507.2 -> 399.8     (1.34x -> 1.06x)
    #   bs=32           -> 956.2   (         0.98x)
    #   bs=64  1947.9 -> 1836.7    (1.18x -> 1.11x)
    # Tier boundaries are interpolated; bs=2 and bs=8 were not measured.
    if batch >= 32:
        _bn_default, _bh_default, _nw_default, _ns_default = 64, 4, 2, 4
    elif batch >= 16:
        _bn_default, _bh_default, _nw_default, _ns_default = 32, 8, 1, 4
    elif batch >= 4:
        _bn_default, _bh_default, _nw_default, _ns_default = 128, 4, 4, 4
    else:
        _bn_default, _bh_default, _nw_default, _ns_default = 128, 4, 4, 2
    # Optional offline-tuned table (SGL_VQ2_CONFIG_JSON), produced by
    # kernel_study/tune_vq2_config.py. Runtime @triton.autotune is unusable here:
    # it benchmarks on first sight of a key, and it cannot run under CUDA-graph
    # capture, which is where sglang first calls this kernel. The optimum is
    # measured to depend on batch and geometry but NOT on context length, and
    # sglang buckets batch to its graph sizes, so a small offline table suffices.
    _tuned = _vq2_tuned_config(batch, kv_group_num, L, NG, KC)
    if _tuned is not None:
        _bn_default, _bh_default, _nw_default, _ns_default = _tuned

    BLOCK = int(os.environ.get("SGL_VQ2_BLOCK_N", _bn_default))
    BLOCK_H = int(os.environ.get("SGL_VQ2_BLOCK_H", _bh_default))
    num_warps = int(os.environ.get("SGL_VQ2_NUM_WARPS", _nw_default))
    num_stages = int(os.environ.get("SGL_VQ2_NUM_STAGES", _ns_default))

    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        max_kv_splits,
    )

    _fwd_grouped_kernel_stage1_quant_vq2[grid](
        q,
        k_idx_buffer,
        cb_packed,
        v_buffer,
        v_idx_arg,
        cb_v_arg,
        k_scales_zeros,
        v_scales_zeros,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_idx_buffer.stride(0),
        k_idx_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        stride_vi_bs,
        stride_vi_h,
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        L=L,
        NG=NG,
        KC=KC,
        NG_V=NG_V,
        KC_V=KC_V,
        V_VQ=V_VQ,
        FP8E4=FP8E4,
        num_warps=num_warps,
        num_stages=num_stages,
    )


IDX_BITS = 10          # default width; per-side widths come from the codebook sizes
PACK_WORDS = 5         # ceil(16*10/32) -- the default-width bitstream arena


def _bits_for(kc):
    """Narrowest supported index width holding `kc` codewords.

    Restricted to 8/9/10/12 because the split layout needs (BITS-8) to divide 8, so a
    high field never straddles a byte. 8 is the interesting one: the arena becomes a
    plain uint8 row and the unpack disappears entirely.
    """
    for b in (8, 9, 10, 12):
        if kc <= (1 << b):
            return b
    raise ValueError(f"codebook of {kc} entries exceeds the supported index widths")


def _pack_words(bits, ng=16):
    return (ng * bits + 31) // 32


@triton.jit
def _unpack10(IdxPtr, row_off, offs_g, mask_n, BITS: tl.constexpr):
    """Gather 10-bit codes for groups `offs_g` from a packed int32 arena.

    row_off: [BLOCK_N] int64 element offset of each row's word 0 (already includes
    kv_loc * stride_bs + head * stride_h). Returns [BLOCK_N, len(offs_g)] int64.
    """
    bit = offs_g * BITS
    wv = bit // 32
    off = bit % 32
    lo = tl.load(IdxPtr + row_off[:, None] + wv[None, :], mask=mask_n[:, None], other=0)
    strad = (off + BITS) > 32
    hi = tl.load(
        IdxPtr + row_off[:, None] + (wv + 1)[None, :],
        mask=mask_n[:, None] & strad[None, :],
        other=0,
    )
    lo_bits = tl.minimum(32 - off, BITS)
    lo_part = (lo >> off[None, :]) & (((1 << lo_bits) - 1))[None, :]
    sh = tl.where(strad, 32 - off, 0)
    v = lo_part | ((hi << sh[None, :]) & ((1 << BITS) - 1))
    return v.to(tl.int32)



# --- launch-time environment, read ONCE -------------------------------------------------
# Keyed by (storage pointer, last-dim width): the .all() below is a device sync, so it must
# run once per arena, never per decode step.
_SCALE_ONE_CACHE: dict = {}


def _is_const_one(t):
    """True when `t` is the stride-0 all-ones scale arena that noscale_patch installs.

    Under SGLANG_NOSCALE the ptn-scale arena for a pertoken_norm=False bundle (CQ K and V,
    TaSQ V) is a [L,1,1,n] tensor of 1.0 expanded across tokens and heads, so the kernel loads
    the same 1.0 for every lane and multiplies by it -- a global load plus a multiply per tile
    per iteration that provably cannot change the result. Detected by the zero strides on the
    token and head axes, then confirmed by value ONCE per storage (the .all() is a sync, so it
    must not run per decode step).
    """
    if t is None or t.dim() < 2 or t.stride(0) != 0 or t.stride(1) != 0:
        return False
    key = (t.data_ptr(), t.shape[-1])
    hit = _SCALE_ONE_CACHE.get(key)
    if hit is None:
        hit = bool((t == 1.0).all().item())
        _SCALE_ONE_CACHE[key] = hit
    return hit



@triton.jit
def _unpack_split(IdxPtr, Idx32, row_off, offs_g, mask_n, BITS: tl.constexpr,
                  NG_ALL: tl.constexpr):
    """Byte-split index unpack: low 8 bits as a contiguous byte per group, high bits after.

    Layout per (token, kv-head) row, in a uint8 arena of width NG_ALL + ceil(NG_ALL*HI/8):
        bytes [0, NG_ALL)      : low 8 bits of each group's code, one byte per group
        bytes [NG_ALL, ...)    : the remaining HI = BITS-8 bits, HI per group, little-endian

    Versus the 10-bit bitstream this replaces: the low half becomes ONE contiguous byte gather
    over the group axis (no bit arithmetic at all), and the high half never straddles a byte
    because HI in (1, 2, 4) divides 8 -- so the straddle load, the second word load and the
    conditional shift all disappear. Storage is unchanged at BITS=10 (16+4 = 20 bytes = the
    same 5 int32 words) and strictly smaller at BITS=9 (18 B) and BITS=8 (16 B, no high half
    at all, which is exactly the VQ pool's uint8 V side).
    """
    lo = tl.load(
        IdxPtr + row_off[:, None] + offs_g[None, :], mask=mask_n[:, None], other=0
    ).to(tl.int32)
    if BITS == 8:
        return lo.to(tl.int32)
    HI: tl.constexpr = BITS - 8
    # The high bits of ALL NG groups occupy NG*HI/8 bytes -- 4 bytes at NG=16, BITS=10 -- so
    # they are read as ONE int32 word per row through Idx32 (the same storage viewed as int32;
    # the row stride is a multiple of 4, so the word offset is exact). Gathering them per group
    # off the byte pointer instead issued 16 loads for 4 distinct bytes and made the split
    # layout ~2% SLOWER than the bitstream it was meant to beat.
    hw = tl.load(Idx32 + (row_off // 4)[:, None] + (NG_ALL // 4), mask=mask_n[:, None], other=0)
    hv = (hw >> (offs_g * HI)[None, :]) & ((1 << HI) - 1)
    return (lo | (hv << 8)).to(tl.int32)


@triton.jit
def _unpack_idx(IdxPtr, Idx32, row_off, offs_g, mask_n, BITS: tl.constexpr,
                NG_ALL: tl.constexpr, IDX_SPLIT: tl.constexpr):
    """IDX_SPLIT: 0 = 10-bit bitstream, 1 = byte-split, 2 = raw int16 (no packing at all).

    Mode 2 exists so that "what does packing cost?" can be answered with EVERY other feature
    held fixed. The obvious alternative -- comparing against vqwide_fold.py -- would confound
    packing with TRIG_HOIST, which that copy does not have.
    """
    if IDX_SPLIT == 1:
        return _unpack_split(IdxPtr, Idx32, row_off, offs_g, mask_n, BITS, NG_ALL)
    if IDX_SPLIT == 2:
        return tl.load(
            IdxPtr + row_off[:, None] + offs_g[None, :], mask=mask_n[:, None], other=0
        ).to(tl.int32)
    return _unpack10(IdxPtr, row_off, offs_g, mask_n, BITS)


@triton.jit
def _fwd_grouped_kernel_stage1_quant_vqwide(
    Q,
    K_Idx,       # packed: [cache_size(+1), num_kv_heads, PACK_WORDS] int32
    CB16,        # fp16 [num_kv_heads, NG, KC, G] K centroids (folded or not)
    K_Idx32,     # same storage as K_Idx viewed as int32 (split layout only)
    V_Idx32,     # same storage as V_Idx viewed as int32 (split layout only)
    V_Idx,       # packed: [cache_size(+1), num_kv_heads, PACK_WORDS] int32 (V_VQ only)
    CB16_V,      # fp16 [num_kv_heads, NG_V, KC_V, G_V] (V_VQ only)
    V_Buffer,    # uint8 [cache_size, num_kv_heads, LV // (8 // V_PACK_BITS)] plane-packed
                 # scalar V -- read only when not V_VQ (dummy otherwise)
    K_Scales_Zeros,  # [cache_size(+1), num_kv_heads, 1 or 2]: slot 0 = K ptn scale
    V_Scales_Zeros,  # V_VQ: slot 0 = V ptn scale. else: [scale, zero] affine pair
    CosSin,
    InvFreq,
    FreqIdx,
    WPerm,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_ki_bs,
    stride_ki_h,
    stride_vi_bs,
    stride_vi_h,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,
    stride_sz_kh,
    stride_sz_vbs,
    stride_sz_vh,
    stride_cs_pos,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    pos_offset,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    L: tl.constexpr,
    LV: tl.constexpr,
    NG: tl.constexpr,
    KC: tl.constexpr,
    G: tl.constexpr,
    NG_V: tl.constexpr,
    KC_V: tl.constexpr,
    G_V: tl.constexpr,
    PRE_ROPE: tl.constexpr,
    PERM_ROPE: tl.constexpr,
    W_FOLDED: tl.constexpr,
    PAIR_MAJOR: tl.constexpr,
    PAIR_FUSED: tl.constexpr,
    PM_FUSED: tl.constexpr,
    VSCALE_ON_P: tl.constexpr,
    TRIG_ONFLY: tl.constexpr,
    TRIG_HOIST: tl.constexpr,
    SCALE_K_ONE: tl.constexpr,
    SCALE_V_ONE: tl.constexpr,
    RECIP_W: tl.constexpr,
    RANK_MAJOR: tl.constexpr,
    NO_ROT: tl.constexpr,
    IDX_SPLIT: tl.constexpr,
    BITS_K: tl.constexpr,
    BITS_V: tl.constexpr,
    V_VQ: tl.constexpr,
    V_PACK_BITS: tl.constexpr,
    Q_PERMUTED: tl.constexpr,
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

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < LV

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        if V_VQ:
            offs_gv = tl.arange(0, NG_V)
            offs_gcv = tl.arange(0, G_V)
        # Centroid table addressing. Shipped layout is GROUP-major [H, NG, KC, G]: group g's
        # rows live in their own KC*G block, so the 16 groups a tile touches are 16 separate
        # 16 KB regions = a 256 KB working set against a 128 KB L1. RANK-major [H, KC, NG, G]
        # puts row r of EVERY group adjacent, so after frequency sorting the hot ranks of all
        # groups form one contiguous run (top-128 x 16 groups x 8 x 2B = 32 KB, L1-resident).
        # Pure re-addressing of a permuted table -- bit-identical.
        cb_head_base = CB16 + cur_kv_head.to(tl.int64) * (NG * KC * G)
        if V_VQ:
            cbv_head_base = CB16_V + cur_kv_head.to(tl.int64) * (NG_V * KC_V * G_V)

        if PERM_ROPE:
            G_HALF: tl.constexpr = G // 2
            offs_g_all = tl.arange(0, NG)
            if PAIR_MAJOR:
                # decode table stores each centroid row as [even coords | odd coords],
                # so a pair's two halves are CONTIGUOUS 8-byte runs instead of two
                # stride-2 gathers over the same 16-byte row. Same values, same lanes.
                offs_ep = tl.arange(0, G_HALF)
                offs_op = G_HALF + tl.arange(0, G_HALF)
            else:
                offs_ep = 2 * tl.arange(0, G_HALF)
                offs_op = offs_ep + 1
            offs_pair = tl.arange(0, NG * G_HALF)

            fidx = tl.load(FreqIdx + cur_kv_head * (NG * G_HALF) + offs_pair).to(tl.int64)
            if TRIG_ONFLY:
                inv_f_h = tl.load(InvFreq + fidx).to(tl.float32)
                if TRIG_HOIST:
                    # angle(n) = (pos_offset + start_n + j) * invf = base*invf + j*invf.
                    # The j*invf half is the SAME for every tile, batch and head-block, so
                    # its cos/sin are computed once here instead of BLOCK_N x 64 accurate
                    # libdevice sinf/cosf per loop iteration.
                    ang_j = tl.arange(0, BLOCK_N)[:, None].to(tl.float32) * inv_f_h[None, :]
                    cos_j = tl.cos(ang_j)
                    sin_j = tl.sin(ang_j)
            w_even = tl.load(WPerm + cur_kv_head * L + 2 * offs_pair).to(tl.float32)
            w_odd = tl.load(WPerm + cur_kv_head * L + 2 * offs_pair + 1).to(tl.float32)
            if RECIP_W:
                # w is loop-invariant, so the unfold can be a reciprocal computed ONCE here
                # plus a multiply per iteration, instead of a divide per element per
                # iteration. Matters because the alternative to the unfold is the FOLDED
                # table, which is fp32 (128 MB) where the unfolded one is fp16 (64 MB) --
                # halving the centroid-gather bytes is worth more than the divide cost if
                # the divide stops being per-iteration work.
                iw_even = 1.0 / w_even
                iw_odd = 1.0 / w_odd

            q_channel = offs_pair if Q_PERMUTED else fidx
            offs_q1 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + q_channel[None, :]
            offs_q2 = offs_q1 + L // 2
            q1 = tl.load(Q + offs_q1, mask=mask_h[:, None], other=0.0).to(tl.float32)
            q2 = tl.load(Q + offs_q2, mask=mask_h[:, None], other=0.0).to(tl.float32)
        else:
            NG_HALF: tl.constexpr = NG // 2
            offs_g1 = tl.arange(0, NG_HALF)
            offs_g2 = NG_HALF + tl.arange(0, NG_HALF)
            offs_gc = tl.arange(0, G)
            offs_g_both = tl.arange(0, NG)

            offs_half = tl.arange(0, NG_HALF * G)
            if TRIG_ONFLY:
                inv_f_h = tl.load(InvFreq + offs_half).to(tl.float32)
                if TRIG_HOIST:
                    ang_j = tl.arange(0, BLOCK_N)[:, None].to(tl.float32) * inv_f_h[None, :]
                    cos_j = tl.cos(ang_j)
                    sin_j = tl.sin(ang_j)
            offs_q1 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_half[None, :]
            offs_q2 = offs_q1 + NG_HALF * G
            q1 = tl.load(Q + offs_q1, mask=mask_h[:, None], other=0.0).to(tl.float32)
            q2 = tl.load(Q + offs_q2, mask=mask_h[:, None], other=0.0).to(tl.float32)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=mask_n,
                other=0,
            ).to(tl.int64)
            k_row_off = kv_loc * stride_ki_bs + cur_kv_head * stride_ki_h

            if PERM_ROPE:
                isel = _unpack_idx(K_Idx, K_Idx32, k_row_off, offs_g_all, mask_n, BITS_K, NG, IDX_SPLIT)
                if PM_FUSED:
                    # The decode table holds every centroid with its 8 coordinates reordered
                    # to [e0..e3 | o0..o3]. Permuting the axes of an 8-D space preserves every
                    # distance, so the quantiser is the same object relabelled -- assignments,
                    # centroids and distortion are untouched and nothing is retrained. Only the
                    # read order changes, and the encode table stays in original order.
                    #
                    # Why this beats both alternatives: the memory shape is the SAME single
                    # G-wide contiguous load the interleaved path uses, but the split now falls
                    # at the register midpoint. Eight fp16 occupy four 32-bit registers
                    # [01][23][45][67], so halving is registers {0,1} vs {2,3} -- a renaming --
                    # where the interleaved split has to unpack the low and high half of all
                    # four. PAIR_MAJOR used the same table but read it as two 8-byte loads and
                    # lost on memory shape instead (1.23-1.33x).
                    #
                    # Measured at real index skew, ctx=4096: interleaved 1.158x vs cq,
                    # this 1.011x -- it recovers 14.7 of the 15.8 points, and gate_pmfused.py
                    # shows it bit-identical to the interleaved path in every mode.
                    kf = tl.load(
                        cb_head_base
                        + (offs_g_all[None, :, None] * G + isel[:, :, None] * (NG * G)
                           if RANK_MAJOR else
                           offs_g_all[None, :, None] * (KC * G) + isel[:, :, None] * G)
                        + tl.arange(0, G)[None, None, :],
                        mask=mask_n[:, None, None], other=0.0,
                    ).to(tl.float32)
                    kf = tl.reshape(kf, (BLOCK_N, NG, 2, G_HALF))
                    k1, k2 = tl.split(tl.permute(kf, (0, 1, 3, 2)))
                elif PAIR_FUSED:
                    # One contiguous G-wide load per group, split into the pair halves
                    # in registers, instead of two G_HALF-wide gathers. The stored
                    # layout is already pair-interleaved (coords 2j / 2j+1 are a RoPE
                    # pair), so the trailing [G_HALF, 2] axes ARE the contiguous run and
                    # tl.split peels evens/odds apart at no memory cost. The two source
                    # loads were STRIDE-2 over the row and so could not be merged; fusing
                    # drops 16 ld instructions (one per group: 48 -> 32) and buys
                    # 1.16-1.19x on a kernel that is issue-bound, not bandwidth-bound.
                    # PAIR_MAJOR failed here because it kept two narrow loads.
                    kf = tl.load(
                        cb_head_base
                        + (offs_g_all[None, :, None, None] * G + isel[:, :, None, None] * (NG * G)
                           if RANK_MAJOR else
                           offs_g_all[None, :, None, None] * (KC * G) + isel[:, :, None, None] * G)
                        + (2 * tl.arange(0, G_HALF))[None, None, :, None]
                        + tl.arange(0, 2)[None, None, None, :],
                        mask=mask_n[:, None, None, None], other=0.0,
                    ).to(tl.float32)
                    k1, k2 = tl.split(kf)
                else:
                    k1 = tl.load(
                        cb_head_base + (offs_g_all[None, :, None] * G + isel[:, :, None] * (NG * G)
                                        if RANK_MAJOR else
                                        offs_g_all[None, :, None] * (KC * G) + isel[:, :, None] * G)
                        + offs_ep[None, None, :],
                        mask=mask_n[:, None, None], other=0.0,
                    ).to(tl.float32)
                    k2 = tl.load(
                        cb_head_base + (offs_g_all[None, :, None] * G + isel[:, :, None] * (NG * G)
                                        if RANK_MAJOR else
                                        offs_g_all[None, :, None] * (KC * G) + isel[:, :, None] * G)
                        + offs_op[None, None, :],
                        mask=mask_n[:, None, None], other=0.0,
                    ).to(tl.float32)
                k1 = tl.reshape(k1, (BLOCK_N, NG * G_HALF))
                k2 = tl.reshape(k2, (BLOCK_N, NG * G_HALF))

                abs_pos = pos_offset + offs_n
                if TRIG_ONFLY and TRIG_HOIST:
                    # only the 64-wide base angle is transcendental now; the tile is
                    # rebuilt with the angle-addition identity (6 FMAs on [BLOCK_N, 64]).
                    ang_b = (pos_offset + start_n).to(tl.float32) * inv_f_h
                    cos_b = tl.cos(ang_b)
                    sin_b = tl.sin(ang_b)
                    cs_row = cos_b[None, :] * cos_j - sin_b[None, :] * sin_j
                    sn_row = sin_b[None, :] * cos_j + cos_b[None, :] * sin_j
                elif TRIG_ONFLY:
                    ang = abs_pos[:, None].to(tl.float32) * inv_f_h[None, :]
                    cs_row = tl.cos(ang)
                    sn_row = tl.sin(ang)
                else:
                    cs_row = tl.load(
                        CosSin + abs_pos[:, None] * stride_cs_pos + fidx[None, :],
                        mask=mask_n[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    sn_row = tl.load(
                        CosSin + abs_pos[:, None] * stride_cs_pos + (L // 2 + fidx)[None, :],
                        mask=mask_n[:, None],
                        other=0.0,
                    ).to(tl.float32)
                if W_FOLDED:
                    k1u = k1
                    k2u = k2
                elif RECIP_W:
                    k1u = k1 * iw_even[None, :]
                    k2u = k2 * iw_odd[None, :]
                else:
                    k1u = k1 / w_even[None, :]
                    k2u = k2 / w_odd[None, :]
                if NO_ROT:
                    # WRONG OUTPUT BY CONSTRUCTION -- timing probe only. The first cut set
                    # cs_row/sn_row to constant tensors, which left the four FMAs running and
                    # ADDED register pressure, so the probe came out slower than the code it
                    # was meant to be cheaper than. Skipping the application is what actually
                    # removes the work.
                    k1 = k1u
                    k2 = k2u
                else:
                    k1 = k1u * cs_row - k2u * sn_row
                    k2 = k2u * cs_row + k1u * sn_row
            else:
                if PAIR_FUSED:
                    # Kept for the record, but a NO-OP: here the halves are whole GROUPS
                    # (0..NG/2-1 vs NG/2..NG-1) whose rows are already contiguous, so this
                    # compiles to byte-identical PTX and the launcher disables it. See
                    # study_pairfused_ptx.py.
                    isel = _unpack_idx(K_Idx, K_Idx32, k_row_off, offs_g_both, mask_n, BITS_K, NG, IDX_SPLIT)
                    kf = tl.load(
                        cb_head_base + (offs_g_both[None, :, None] * G + isel[:, :, None] * (NG * G)
                                        if RANK_MAJOR else
                                        offs_g_both[None, :, None] * (KC * G) + isel[:, :, None] * G)
                        + offs_gc[None, None, :],
                        mask=mask_n[:, None, None], other=0.0,
                    ).to(tl.float32)
                    kf = tl.reshape(kf, (BLOCK_N, 2, NG_HALF * G))
                    k1, k2 = tl.split(tl.permute(kf, (0, 2, 1)))
                else:
                    isel1 = _unpack_idx(K_Idx, K_Idx32, k_row_off, offs_g1, mask_n, BITS_K, NG, IDX_SPLIT)
                    isel2 = _unpack_idx(K_Idx, K_Idx32, k_row_off, offs_g2, mask_n, BITS_K, NG, IDX_SPLIT)
                    k1 = tl.load(
                        cb_head_base + (offs_g1[None, :, None] * G + isel1[:, :, None] * (NG * G)
                                        if RANK_MAJOR else
                                        offs_g1[None, :, None] * (KC * G) + isel1[:, :, None] * G)
                        + offs_gc[None, None, :],
                        mask=mask_n[:, None, None], other=0.0,
                    ).to(tl.float32)
                    k2 = tl.load(
                        cb_head_base + (offs_g2[None, :, None] * G + isel2[:, :, None] * (NG * G)
                                        if RANK_MAJOR else
                                        offs_g2[None, :, None] * (KC * G) + isel2[:, :, None] * G)
                        + offs_gc[None, None, :],
                        mask=mask_n[:, None, None], other=0.0,
                    ).to(tl.float32)
                    k1 = tl.reshape(k1, (BLOCK_N, NG_HALF * G))
                    k2 = tl.reshape(k2, (BLOCK_N, NG_HALF * G))

                if PRE_ROPE:
                    abs_pos = pos_offset + offs_n
                    if TRIG_ONFLY and TRIG_HOIST:
                        ang_b = (pos_offset + start_n).to(tl.float32) * inv_f_h
                        cos_b = tl.cos(ang_b)
                        sin_b = tl.sin(ang_b)
                        cs_row = cos_b[None, :] * cos_j - sin_b[None, :] * sin_j
                        sn_row = sin_b[None, :] * cos_j + cos_b[None, :] * sin_j
                    elif TRIG_ONFLY:
                        ang = abs_pos[:, None].to(tl.float32) * inv_f_h[None, :]
                        cs_row = tl.cos(ang)
                        sn_row = tl.sin(ang)
                    else:
                        cs_row = tl.load(
                            CosSin + abs_pos[:, None] * stride_cs_pos + offs_half[None, :],
                            mask=mask_n[:, None],
                            other=0.0,
                        ).to(tl.float32)
                        sn_row = tl.load(
                            CosSin
                            + abs_pos[:, None] * stride_cs_pos
                            + (NG_HALF * G + offs_half)[None, :],
                            mask=mask_n[:, None],
                            other=0.0,
                        ).to(tl.float32)
                    o1 = k1 * cs_row - k2 * sn_row
                    o2 = k2 * cs_row + k1 * sn_row
                    k1, k2 = o1, o2

            qk = tl.dot(q1.to(tl.float16), tl.trans(k1).to(tl.float16))
            qk += tl.dot(q2.to(tl.float16), tl.trans(k2).to(tl.float16))

            if not SCALE_K_ONE:
                k_scale = tl.load(
                    K_Scales_Zeros
                    + kv_loc * stride_sz_kbs
                    + cur_kv_head * stride_sz_kh,
                    mask=mask_n,
                    other=1.0,
                ).to(tl.float32)
                qk = qk * k_scale[None, :]
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & mask_n[None, :], qk, float("-inf")
            )

            if V_VQ:
                v_row_off = kv_loc * stride_vi_bs + cur_kv_head * stride_vi_h
                iselv = _unpack_idx(V_Idx, V_Idx32, v_row_off, offs_gv, mask_n, BITS_V, NG_V, IDX_SPLIT)
                vg = tl.load(
                    cbv_head_base
                    + (offs_gv[None, :, None] * G_V + iselv[:, :, None] * (NG_V * G_V)
                       if RANK_MAJOR else
                       offs_gv[None, :, None] * (KC_V * G_V) + iselv[:, :, None] * G_V)
                    + offs_gcv[None, None, :],
                    mask=mask_n[:, None, None],
                    other=0.0,
                ).to(tl.float32)
                vg = tl.reshape(vg, (BLOCK_N, NG_V * G_V))
                if not SCALE_V_ONE:
                    v_scale = tl.load(
                        V_Scales_Zeros
                        + kv_loc * stride_sz_vbs
                        + cur_kv_head * stride_sz_vh,
                        mask=mask_n,
                        other=1.0,
                    ).to(tl.float32)
                    if not VSCALE_ON_P:
                        vg = vg * v_scale[:, None]
            else:
                # Scalar V (OSCAR min/max affine), plane-packed at V_PACK_BITS bits
                # per coordinate: coord d lives at byte (d % plane_dv), field
                # (d // plane_dv). Same convention the writers use, which is the
                # QUARTERED one at 2 bits -- NOT four consecutive coords per byte.
                # Getting that backwards reads plausible-looking garbage rather than
                # failing, so it is stated here as well as at the two write sites.
                #
                # The G=4 packed kernel (_fwd_grouped_kernel_stage1_quant_vq2) needs
                # its V as four quarter-planes to match its accumulators; here the
                # accumulator is a dense [BLOCK_N, LV] tile feeding one tl.dot, so
                # the unpack is a single elementwise expression over offs_dv.
                V_PACK_DEN: tl.constexpr = 8 // V_PACK_BITS
                plane_dv: tl.constexpr = LV // V_PACK_DEN
                v_packed = tl.load(
                    V_Buffer
                    + kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + (offs_dv % plane_dv)[None, :],
                    mask=mask_n[:, None] & mask_dv[None, :],
                    other=0,
                )
                v_shift = (offs_dv // plane_dv) * V_PACK_BITS
                v_scale_a = tl.load(
                    V_Scales_Zeros + kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh,
                    mask=mask_n,
                    other=1.0,
                ).to(tl.float32)
                v_zero_a = tl.load(
                    V_Scales_Zeros + kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh + 1,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
                vg = (
                    ((v_packed >> v_shift[None, :]) & ((1 << V_PACK_BITS) - 1)).to(tl.float32)
                    - v_zero_a[:, None]
                ) * v_scale_a[:, None]

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            # acc[h,d] = sum_n p[h,n] * vg[n,d] * s[n] -- scaling the PROBABILITIES instead of
            # the V rows is the same sum with BLOCK_H*BLOCK_N multiplies instead of BLOCK_N*LV
            # (256 vs 2048 at the served geometry). The softmax DENOMINATOR must keep using the
            # UNSCALED p: folding s into p before `tl.sum(p, 1)` makes e_sum sum(p*s) and the
            # output is then wrong by ~2e-2 (measured, gate_vscale_on_p.py first cut).
            if VSCALE_ON_P and not SCALE_V_ONE and V_VQ:
                # Scalar V carries a zero-point, so `sum_n p*(q - z)*s` does not
                # factor into a per-probability scale; the fold is VQ-only.
                p_dot = p * v_scale[None, :]
            else:
                p_dot = p
            acc *= re_scale[:, None]
            acc += tl.dot(p_dot.to(tl.float16), vg.to(tl.float16))

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // LV
        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


_VQWIDE_INVFREQ_CACHE: dict = {}


def _decode_grouped_att_m_fwd_quant_vqwide(
    q,
    k_idx_buffer,   # PACKED [cache_size(+1), num_kv_heads, PACK_WORDS] int32
    cb16,           # [num_kv_heads, NG, KC, G] fp16 (folded table when w_folded=True)
    v_idx_buffer,   # PACKED
    cb16_v,
    k_scales_zeros,  # [..., 1] (slot-dropped) or [..., 2] (legacy)
    v_scales_zeros,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
    cos_sin_cache=None,
    pos_offset=0,
    freq_idx=None,
    w_perm=None,
    w_folded=False,
    pair_major_table=False,
    v_buffer=None,
    v_pack_bits=2,
    q_permuted=False,
):
    pre_rope = cos_sin_cache is not None
    perm_rope = freq_idx is not None
    assert not perm_rope or pre_rope, "PERM_ROPE requires PRE_ROPE (cos_sin_cache)"
    # cb16_v=None selects the scalar-V tier: plane-packed int2/int1 + per-(token,
    # head) affine, the same V the G=4 packed kernel has always been able to serve.
    # Until this branch existed a wide-G K codebook forced VQ-V, which is why the
    # 1-bit Nova rebuild could not use OSCAR's own V quantiser (see the boot assert
    # in unified_kv_pool).
    v_vq = cb16_v is not None
    assert v_vq or v_buffer is not None, (
        "scalar-V mode (cb16_v=None) requires v_buffer"
    )
    assert v_pack_bits in (1, 2), f"v_pack_bits must be 1 or 2, got {v_pack_bits}"
    assert not q_permuted or perm_rope, "Permuted Q requires a TaSQ frequency map"
    # No flags. Every optimisation below was gated and adopted, so it is simply the path:
    #   on-the-fly + hoisted trig   gate_trighoist.py, excess SSE <= 3.8e-08, 1.04-1.30x
    #   fused pair load             gate_pairfused.py, bit-identical (PERM_ROPE only -- on
    #                               the plain pre-RoPE path the halves are already whole
    #                               contiguous groups and the fused form is a no-op)
    #   constant-scale elision      gate_scaleone.py, bit-identical; a data property of the
    #                               scale tensors, so it is detected, not configured
    #   10-bit packed index arenas  gate_idxpack_decode.py, bit-identical in all six modes
    trig_onfly = pre_rope
    trig_hoist = pre_rope
    pair_fused = perm_rope
    scale_k_one = _is_const_one(k_scales_zeros)
    scale_v_one = _is_const_one(v_scales_zeros)
    recip_w = False
    # Layout/experiment knobs that were measured and refuted; fixed so the kernel keeps one
    # specialisation. PAIR_MAJOR: two 8-byte loads where the fused form does one 16-byte
    # (1.25x slower at real index skew). RANK_MAJOR, NO_ROT, VSCALE_ON_P, RECIP_W: see
    # exp/kernelopt/README.md.
    pair_major = False
    pm_fused = bool(pair_major_table) and perm_rope
    no_rot = False
    vscale_on_p = False
    if trig_onfly:
        key = id(cos_sin_cache)
        inv_freq = _VQWIDE_INVFREQ_CACHE.get(key)
        if inv_freq is None:
            half = cos_sin_cache.shape[1] // 2
            row1 = cos_sin_cache[1].float()
            inv_freq = torch.atan2(row1[half:], row1[:half]).contiguous()
            _VQWIDE_INVFREQ_CACHE[key] = inv_freq
    else:
        inv_freq = q.new_zeros(1, dtype=torch.float32)
    if not pre_rope:
        cos_sin_cache = q.new_zeros(1, 1)
    if not perm_rope:
        freq_idx = q.new_zeros(1, dtype=torch.int64)
        w_perm = q.new_zeros(1)
    # NG can no longer be read off the idx buffer (its last dim is PACK_WORDS) --
    # it comes from the codebook, whose layout packing does not change.
    rank_major = False
    # the two layouts transpose the middle axes: [H, NG, KC, G] vs [H, KC, NG, G]
    if rank_major:
        KC, NG, G = cb16.shape[1], cb16.shape[2], cb16.shape[3]
    else:
        NG, KC, G = cb16.shape[1], cb16.shape[2], cb16.shape[3]
    if v_vq:
        if rank_major:
            KC_V, NG_V, G_V = cb16_v.shape[1], cb16_v.shape[2], cb16_v.shape[3]
        else:
            NG_V, KC_V, G_V = cb16_v.shape[1], cb16_v.shape[2], cb16_v.shape[3]
        LV = NG_V * G_V
    else:
        # Unread constexprs, but they still size tl.arange() in the prologue, so
        # they must be legal (power-of-two, >= 1) rather than 0. The tensors handed
        # in as CB16_V/V_Idx are the K ones for the same reason: Triton needs a
        # typed pointer per parameter even when the branch is compiled out.
        NG_V, KC_V, G_V = 16, 16, 8
        LV = v_buffer.shape[-1] * (8 // v_pack_bits)
        cb16_v = cb16
        v_idx_buffer = k_idx_buffer
    # Derived HERE, before the arena shape checks that consume them -- putting this after those
    # checks is what made the first cut of this patch fail every gate with UnboundLocalError.
    bits_k = _bits_for(KC)
    bits_v = _bits_for(KC_V) if v_vq else 8
    # 0 bitstream (int32 x PACK_WORDS) | 1 byte-split (uint8) | 2 raw (int16 x NG)
    idx_split = {torch.uint8: 1, torch.int16: 2}.get(k_idx_buffer.dtype, 0)
    # int32 views of the SAME storage; only read when IDX_SPLIT, but must be valid tensors
    # either way because Triton needs a typed pointer for every parameter.
    k_idx32 = k_idx_buffer.view(torch.int32) if idx_split == 1 else k_idx_buffer
    v_idx32 = v_idx_buffer.view(torch.int32) if idx_split == 1 else v_idx_buffer
    if idx_split == 2:
        assert k_idx_buffer.shape[-1] == NG, (
            "raw layout expects [..., NG] int16 index arenas"
        )
    if idx_split == 1:
        # split layout: uint8 [..., NG + ceil(NG*(BITS-8)/8)]
        want = NG + (NG * (bits_k - 8) + 7) // 8
        assert k_idx_buffer.shape[-1] == want, (
            f"split K arena width {k_idx_buffer.shape[-1]} != {want} for NG={NG} BITS={IDX_BITS}"
        )
    assert idx_split or k_idx_buffer.dtype == torch.int32 and k_idx_buffer.shape[-1] == _pack_words(bits_k, NG), (
        f"idxpack launcher requires a packed int32 [..., {PACK_WORDS}] K arena, got "
        f"{k_idx_buffer.dtype} {tuple(k_idx_buffer.shape)}"
    )
    if v_vq:
        if idx_split == 2:
            assert v_idx_buffer.shape[-1] == NG_V, (
                "raw layout expects [..., NG_V] int16 index arenas"
            )
        if idx_split == 1:
            assert v_idx_buffer.dtype == torch.uint8, "K and V arenas must use the same layout"
            want_v = NG_V + (NG_V * (bits_v - 8) + 7) // 8
            assert v_idx_buffer.shape[-1] == want_v, (
                f"split V arena width {v_idx_buffer.shape[-1]} != {want_v} for NG_V={NG_V}"
            )
        assert idx_split or (
            v_idx_buffer.dtype == torch.int32 and v_idx_buffer.shape[-1] == _pack_words(bits_v, NG_V)
        ), f"V arena layout does not match K's, got {v_idx_buffer.dtype} {tuple(v_idx_buffer.shape)}"
        assert NG_V == 16, "packing layout is specialised to NG=16"
    else:
        assert v_buffer.dtype == torch.uint8, (
            f"scalar V arena must be uint8, got {v_buffer.dtype}"
        )
        assert v_scales_zeros.shape[-1] == 2, (
            "scalar V needs the [scale, zero] affine pair; got "
            f"{v_scales_zeros.shape[-1]} slot(s)"
        )
    assert NG == 16, "packing layout is specialised to NG=16"
    # NB: the strides handed to the kernel are in ELEMENTS of the arena dtype -- int32 words
    # for the bitstream layout, BYTES for the split layout -- and _unpack_split indexes in
    # bytes, so the two stay consistent without a special case.
    L = NG * G
    # LV is set above -- from the V codebook when V_VQ, from the packed V arena width
    # otherwise.
    assert k_scales_zeros.shape[-1] in (1, 2) and v_scales_zeros.shape[-1] in (1, 2)
    # The scale arena is all-ones only for a ptn-free VQ bundle; the scalar tier's
    # arena holds (scale, zero) pairs and the kernel must always read them.
    scale_v_one = scale_v_one and v_vq
    if not v_vq:
        v_buffer_arg = v_buffer
    else:
        v_buffer_arg = k_idx_buffer  # dummy typed pointer; unread when V_VQ

    BLOCK_D = triton.next_power_of_2(L)
    BLOCK_DV = triton.next_power_of_2(LV)
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_idx_buffer.shape[1]

    BLOCK = 16
    BLOCK_H = 4
    num_warps = 1
    num_stages = 1
    # Empirically selected for the wide-VQ decode kernel across the serving shapes.
    _maxnreg = None

    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        max_kv_splits,
    )


    _fwd_grouped_kernel_stage1_quant_vqwide[grid](
        q,
        k_idx_buffer,
        cb16,
        k_idx32,
        v_idx32,
        v_idx_buffer,
        cb16_v,
        v_buffer_arg,
        k_scales_zeros,
        v_scales_zeros,
        cos_sin_cache,
        inv_freq,
        freq_idx,
        w_perm,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_idx_buffer.stride(0),
        k_idx_buffer.stride(1),
        v_idx_buffer.stride(0),
        v_idx_buffer.stride(1),
        v_buffer_arg.stride(0),
        v_buffer_arg.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        cos_sin_cache.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        pos_offset,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        L=L,
        LV=LV,
        NG=NG,
        KC=KC,
        G=G,
        NG_V=NG_V,
        KC_V=KC_V,
        G_V=G_V,
        PRE_ROPE=pre_rope,
        PERM_ROPE=perm_rope,
        W_FOLDED=w_folded,
        PAIR_MAJOR=pair_major,
        PAIR_FUSED=pair_fused and not pm_fused,
        PM_FUSED=pm_fused,
        VSCALE_ON_P=vscale_on_p,
        TRIG_ONFLY=trig_onfly,
        TRIG_HOIST=trig_hoist,
        SCALE_K_ONE=scale_k_one,
        SCALE_V_ONE=scale_v_one,
        RECIP_W=recip_w,
        RANK_MAJOR=rank_major,
        NO_ROT=bool(no_rot),
        IDX_SPLIT=idx_split,
        BITS_K=bits_k,
        BITS_V=bits_v,
        V_VQ=v_vq,
        V_PACK_BITS=v_pack_bits,
        Q_PERMUTED=q_permuted,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=_maxnreg,
    )


def decode_attention_fwd_vq2_unified(
    q,
    hp_k_buffer,
    hp_v_buffer,
    quant_k_idx_buffer,
    cb_packed,
    quant_v_buffer,
    quant_k_scales_zeros,
    quant_v_scales_zeros,
    o,
    hp_kv_indptr,
    hp_kv_indices,
    quant_kv_indptr,
    quant_kv_indices,
    attn_logits,
    attn_lse,
    hp_num_kv_splits,
    quant_num_kv_splits,
    hp_max_kv_splits,
    quant_max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    sink_q=None,
    sink_mean=None,
    xai_temperature_len=-1,
    cb_packed_v=None,
    quant_v_idx_buffer=None,
    wide_g=False,
    cb16=None,          # decode-side table: the 1/w-folded copy when the bundle
                        # has one, in which case w_folded must be True
    k_cb_folded=False,
    k_cb_pair_major=False,
    cb16_v=None,
    cos_sin_cache=None,
    full_seq_len=None,
    pos_offset=0,
    prefix_tokens=0,
    freq_idx=None,
    w_perm=None,
    v_pack_bits=2,
    q_permuted=None,
):
    """Unified HP + vq2 decode attention: identical structure to
    :func:`decode_attention_fwd_int2_unified`, with the quant stage-1 swapped
    for the group-VQ K kernel. HP stage-1 and the tier-agnostic stage-2 are
    reused unchanged. When cb_packed_v/quant_v_idx_buffer are given, the quant
    tier decodes V as group-VQ too (the VQ-V ablation); else int2 V.

    wide_g=True routes the quant tier through the plain-gather vqwide kernel
    (group_dim != 4 codebooks, e.g. this project's own CQ/TaSQ); requires cb16.
    V follows the same rule as the packed path: cb16_v/quant_v_idx_buffer select
    VQ-V, and their absence selects the scalar tier out of ``quant_v_buffer`` at
    ``v_pack_bits`` bits per coordinate (2 = int2 crumbs, 1 = int1).

    cos_sin_cache != None activates pre-RoPE mode for BOTH tiers (this
    project's CQ/TaSQ convention -- the cache holds raw pre-RoPE K
    everywhere, not just in the quant tier): the HP tier routes through
    ``_decode_grouped_att_m_fwd_prerope`` (needs ``full_seq_len``, see that
    function) and the quant tier's rotation uses ``pos_offset`` (the fixed
    PREFIX_TOKENS constant -- the quant tier always starts at that absolute
    position, unlike the HP tier's per-request offset)."""
    pre_rope = cos_sin_cache is not None
    total_splits = hp_max_kv_splits + quant_max_kv_splits
    assert attn_logits.shape[2] == total_splits

    attn_lse.fill_(float("-inf"))

    hp_logits = attn_logits[:, :, :hp_max_kv_splits, :]
    hp_lse = attn_lse[:, :, :hp_max_kv_splits]
    quant_logits = attn_logits[:, :, hp_max_kv_splits:, :]
    quant_lse = attn_lse[:, :, hp_max_kv_splits:]

    kv_group_num = q.shape[1] // hp_k_buffer.shape[1]
    if kv_group_num == 1:
        raise NotImplementedError(
            "vq2 decode currently supports GQA/MQA (kv_group_num > 1) only."
        )

    if hp_kv_indices.numel() > 0:
        if pre_rope:
            assert full_seq_len is not None, "pre-RoPE mode requires full_seq_len"
            _decode_grouped_att_m_fwd_prerope(
                q,
                hp_k_buffer,
                hp_v_buffer,
                cos_sin_cache,
                full_seq_len,
                hp_logits,
                hp_lse,
                hp_kv_indptr,
                hp_kv_indices,
                hp_num_kv_splits,
                hp_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
                freq_idx=freq_idx,
                w_perm=w_perm,
                prefix_tokens=prefix_tokens,
            )
        else:
            _decode_grouped_att_m_fwd(
                q,
                hp_k_buffer,
                hp_v_buffer,
                hp_logits,
                hp_lse,
                hp_kv_indptr,
                hp_kv_indices,
                hp_num_kv_splits,
                hp_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
            )

    if quant_kv_indices.numel() > 0:
        if wide_g:
            assert cb16 is not None, "wide_g=True requires cb16"
            assert (cb16_v is None) == (quant_v_idx_buffer is None), (
                "cb16_v and quant_v_idx_buffer select VQ-V together; passing one "
                "without the other would silently read the wrong V arena"
            )
            _decode_grouped_att_m_fwd_quant_vqwide(
                q if q_permuted is None else q_permuted,
                quant_k_idx_buffer,
                cb16,
                quant_v_idx_buffer,
                cb16_v,
                quant_k_scales_zeros,
                quant_v_scales_zeros,
                quant_logits,
                quant_lse,
                quant_kv_indptr,
                quant_kv_indices,
                quant_num_kv_splits,
                quant_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
                cos_sin_cache=cos_sin_cache if pre_rope else None,
                pos_offset=pos_offset,
                freq_idx=freq_idx,
                w_perm=w_perm,
                w_folded=k_cb_folded,
                pair_major_table=k_cb_pair_major,
                v_buffer=quant_v_buffer,
                v_pack_bits=v_pack_bits,
                q_permuted=q_permuted is not None,
            )
        else:
            _decode_grouped_att_m_fwd_quant_vq2(
                q,
                quant_k_idx_buffer,
                cb_packed,
                quant_v_buffer,
                quant_k_scales_zeros,
                quant_v_scales_zeros,
                quant_logits,
                quant_lse,
                quant_kv_indptr,
                quant_kv_indices,
                quant_num_kv_splits,
                quant_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
                cb_packed_v=cb_packed_v,
                v_idx_buffer=quant_v_idx_buffer,
            )

    from sglang.srt.environ import envs as _envs

    _unified_stage2(
        attn_logits,
        attn_lse,
        o,
        total_splits=total_splits,
        sinks=sinks,
        sink_q=sink_q,
        sink_mean=sink_mean,
        sink_sm_scale=sm_scale,
    )
    return o
