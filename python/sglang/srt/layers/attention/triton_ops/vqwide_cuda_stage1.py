"""CUDA stage-1 for WIDE-group (G=8) vq decode attention with per-group shared-memory
codebook staging — the g8/10-bit analogue of vq2_cuda_stage1.py.

Why a separate kernel: the g8 CQ/TaSQ codebooks are 1024x8 fp16 = 16 KB PER GROUP
(256 KB per head), so vq2_cuda's whole-table staging cannot hold them (SM shared
memory is ~100 KB). But the score decomposes per group (score = sum_g q~_g . k^_g),
so this kernel runs three in-block phases per (batch, kv_head, split):

  phase A  for g in 0..NG-1: stage K table_g (16 KB) once, accumulate every split
           token's partial score into a shared qk buffer;
  phase B  exact softmax over the split (all scores resident -> no online rescale);
  phase C  for g in 0..NG-1: stage V table_g, accumulate the weighted sum.

RoPE: the stored K is pre-RoPE (CQ) or permuted+weighted pre-RoPE (TaSQ). Rotating K
would couple channels ACROSS groups (pair (c, c+64) lives in groups g and g+8), so the
rotation is moved to the query side instead:

    q~_c(pos) = q_c * cos_c(pos) + q_{c+64} * sin_c(pos)          (c <  64)
    q~_c(pos) = q_c * cos_{c-64}(pos) - q_{c-64} * sin_{c-64}(pos) (c >= 64)

which reproduces q . RoPE(k^) exactly while keeping the K side group-local. For
PERM_ROPE (TaSQ) the stored pair {2p, 2p+1} of group g maps to original channels
{f, f+64} with f = freq_idx[h][4g+p]; the kernel unweights by w_perm and gathers the
matching q~ terms — pairs never leave their group, so the same three-phase structure
holds.

Correctness contract: must match `_decode_grouped_att_m_fwd_quant_vqwide` (Triton) and
a torch reference on att_out AND att_lse for all
three modes. fp32 score/output accumulation throughout (no half2 fast path in v1).

Split-length cap: everything for the split lives in shared memory, so the launcher
refuses splits longer than SPLIT_CAP tokens (supports() gate -> caller falls back to
the Triton kernel). With the engine default of 48 kv-splits this admits ~49k context.

Opt-in via SGLANG_VQWIDE_CUDA=1; `supports()` gates geometry exactly like vq2_cuda.
"""
from __future__ import annotations

import functools
import os

import torch

# (NG, KC, L, KVG) — Qwen3-4B / Llama-3.1-8B share this after GQA sharding at TP=1.
_SUPPORTED_GEOMS = {(16, 1024, 128, 4)}
_SPLIT_CAP = 1024
_MIN_BLOCK_KV = 32

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

#ifndef VQW_THR
#define VQW_THR 256
#endif
#ifndef VQW_GEOM_NG
#define VQW_GEOM_NG 16
#endif
#ifndef VQW_GEOM_KC
#define VQW_GEOM_KC 1024
#endif
#ifndef VQW_GEOM_L
#define VQW_GEOM_L 128
#endif
#ifndef VQW_GEOM_KVG
#define VQW_GEOM_KVG 4
#endif
#ifndef VQW_SPLIT_CAP
#define VQW_SPLIT_CAP 1024
#endif
#define NGw   VQW_GEOM_NG
#define KCw   VQW_GEOM_KC
#define Lw    VQW_GEOM_L
#define KVGw  VQW_GEOM_KVG
#define Gw    (Lw / NGw)          /* 8 */
#define HALFw (Lw / 2)            /* 64 */
#define PPGw  (Gw / 2)            /* RoPE/perm pairs per group: 4 */
#define CAPw  VQW_SPLIT_CAP
#define VQW_MINBK 32

__device__ __forceinline__ float q2f(const __half v)        { return __half2float(v); }
__device__ __forceinline__ float q2f(const __nv_bfloat16 v) { return __bfloat162float(v); }

// MODE: 0 = post-RoPE (no rotation in kernel), 1 = PRE_ROPE (CQ), 2 = PERM_ROPE (TaSQ)
template <typename QT, typename IT, int THR, int MODE>
__global__ __launch_bounds__(THR) void vqwide_stage1(
    const QT*      __restrict__ Q,        // [B, H_kv*KVG, L]
    const int16_t* __restrict__ K_Idx,    // [slots, H_kv, NG]
    const __half*  __restrict__ K_CB,     // [H_kv, NG, KC, G]
    const int16_t* __restrict__ V_Idx,    // [slots, H_kv, NG]
    const __half*  __restrict__ V_CB,     // [H_kv, NG, KC, G]
    const float*   __restrict__ K_SZ,     // [slots, H_kv, 2] scale at [...,0]
    const float*   __restrict__ V_SZ,
    const float*   __restrict__ CosSin,   // [max_pos, L] cos|sin halves (fp32), MODE>=1
    const long*    __restrict__ FreqIdx,  // [H_kv, HALF] (MODE==2)
    const float*   __restrict__ WPerm,    // [H_kv, L]    (MODE==2)
    const int*     __restrict__ kv_indptr,
    const IT*      __restrict__ kv_indices,
    const int*     __restrict__ num_kv_splits,
    float*         __restrict__ Att_Out,  // strided slice, fp32
    float*         __restrict__ Att_Lse,
    float sm_scale, int pos_offset,
    long s_q_b, long s_q_h,
    long s_ki_b, long s_ki_h, long s_vi_b, long s_vi_h,
    long s_ksz_b, long s_ksz_h, long s_vsz_b, long s_vsz_h,
    long s_cs_p,
    long s_o_b, long s_o_h, long s_o_s,
    long s_l_b, long s_l_h, long s_l_s)
{
    const int bb = blockIdx.x, kvh = blockIdx.y, bs = blockIdx.z;
    const int tid = threadIdx.x;

    extern __shared__ __align__(16) char smem_raw[];
    // layout:  [ table 16KB | idx 2*CAP*NG int16 (K then reused for V) is too big at
    //            CAP=1024*16*2B=32KB | qk CAP*KVG fp32 16KB | q 2KB | pos-side small ]
    __half*   tbl_s = reinterpret_cast<__half*>(smem_raw);                 // KC*G fp16 = 16KB
    int16_t*  idx_s = reinterpret_cast<int16_t*>(tbl_s + KCw * Gw);        // CAP*NG = 32KB
    float*    qk_s  = reinterpret_cast<float*>(idx_s + CAPw * NGw);        // CAP*KVG = 16KB
    float*    q_s   = qk_s + CAPw * KVGw;                                  // KVG*L fp32 = 2KB
    float*    w_s   = q_s + KVGw * Lw;                                     // L fp32 (MODE==2)
    long*     fi_s  = reinterpret_cast<long*>(w_s + Lw);                   // HALF longs (MODE==2)
    float*    red_s = reinterpret_cast<float*>(fi_s + HALFw);              // THR/32*KVG reduce pad

    const int kv_start = kv_indptr[bb];
    const int seq_len  = kv_indptr[bb + 1] - kv_start;
    const int splits   = num_kv_splits[bb];
    const int per = ((seq_len + splits - 1) / splits + VQW_MINBK - 1) / VQW_MINBK * VQW_MINBK;
    const int s_start = per * bs;
    const int s_end   = min(s_start + per, seq_len);
    if (s_end <= s_start) return;
    const int n_tok = s_end - s_start;   // <= CAPw, guaranteed by launcher

    // ---- stage q (fp32) and TaSQ tables ----
    for (int i = tid; i < KVGw * Lw; i += THR) {
        const int h = i / Lw, c = i % Lw;
        q_s[i] = q2f(Q[(long)bb * s_q_b + (long)(kvh * KVGw + h) * s_q_h + c]);
    }
    if (MODE == 2) {
        for (int i = tid; i < Lw; i += THR)   w_s[i]  = WPerm[(long)kvh * Lw + i];
        for (int i = tid; i < HALFw; i += THR) fi_s[i] = FreqIdx[(long)kvh * HALFw + i];
    }
    for (int i = tid; i < n_tok * KVGw; i += THR) qk_s[i] = 0.f;

    // ---- stage K index rows (one coalesced 32B row per token) ----
    for (int i = tid; i < n_tok * NGw; i += THR) {
        const int n = i / NGw, g = i % NGw;
        const long loc = (long)kv_indices[kv_start + s_start + n];
        idx_s[n * NGw + g] = K_Idx[loc * s_ki_b + (long)kvh * s_ki_h + g];
    }
    __syncthreads();

    // ---- phase A: per-group K table staging + score accumulation ----
    const __half* kcb_h = K_CB + (long)kvh * (NGw * KCw * Gw);
    for (int g = 0; g < NGw; ++g) {
        __syncthreads();
        const __half* src = kcb_h + (long)g * (KCw * Gw);
        for (int i = tid; i < KCw * Gw; i += THR) tbl_s[i] = src[i];
        __syncthreads();

        for (int n = tid; n < n_tok; n += THR) {
            const int idx = (int)idx_s[n * NGw + g];
            const __half* cw = tbl_s + idx * Gw;
            float part[KVGw];
#pragma unroll
            for (int h = 0; h < KVGw; ++h) part[h] = 0.f;

            if (MODE == 0) {
#pragma unroll
                for (int j = 0; j < Gw; ++j) {
                    const float kv = __half2float(cw[j]);
                    const int c = g * Gw + j;
#pragma unroll
                    for (int h = 0; h < KVGw; ++h)
                        part[h] = fmaf(q_s[h * Lw + c], kv, part[h]);
                }
            } else if (MODE == 1) {
                // q~ pre-rope: channel c pairs with c^HALF, cos/sin index c % HALF
                const int apos = pos_offset + s_start + n;
                const float* cs = CosSin + (long)apos * s_cs_p;
#pragma unroll
                for (int j = 0; j < Gw; ++j) {
                    const float kv = __half2float(cw[j]);
                    const int c = g * Gw + j;
                    const int f = c & (HALFw - 1);
                    const float co = cs[f], si = cs[HALFw + f];
                    const float sgn = (c < HALFw) ? 1.f : -1.f;
                    const int cp = c ^ HALFw;   // partner channel
#pragma unroll
                    for (int h = 0; h < KVGw; ++h) {
                        const float qt = q_s[h * Lw + c] * co + sgn * q_s[h * Lw + cp] * si;
                        part[h] = fmaf(qt, kv, part[h]);
                    }
                }
            } else {
                // PERM_ROPE: stored pair {2p, 2p+1} of this group -> original channels
                // {f, f+HALF}; unweight then rotate at frequency f, exactly the Triton
                // kernel's k1u*cs - k2u*sn / k2u*cs + k1u*sn against q at f / f+HALF.
                const int apos = pos_offset + s_start + n;
                const float* cs = CosSin + (long)apos * s_cs_p;
#pragma unroll
                for (int p = 0; p < PPGw; ++p) {
                    const int sc = g * Gw + 2 * p;      // stored channels sc, sc+1
                    const long f = fi_s[g * PPGw + p];  // original channel f, f+HALF
                    const float k1u = __half2float(cw[2 * p])     / w_s[sc];
                    const float k2u = __half2float(cw[2 * p + 1]) / w_s[sc + 1];
                    const float co = cs[f], si = cs[HALFw + f];
                    const float r1 = k1u * co - k2u * si;   // lands on channel f
                    const float r2 = k2u * co + k1u * si;   // lands on channel f+HALF
#pragma unroll
                    for (int h = 0; h < KVGw; ++h)
                        part[h] = fmaf(q_s[h * Lw + f], r1,
                                  fmaf(q_s[h * Lw + HALFw + f], r2, part[h]));
                }
            }
#pragma unroll
            for (int h = 0; h < KVGw; ++h) qk_s[n * KVGw + h] += part[h];
        }
    }
    __syncthreads();

    // ---- apply K scale * sm_scale ----
    for (int n = tid; n < n_tok; n += THR) {
        const long loc = (long)kv_indices[kv_start + s_start + n];
        const float ksc = K_SZ[loc * s_ksz_b + (long)kvh * s_ksz_h] * sm_scale;
#pragma unroll
        for (int h = 0; h < KVGw; ++h) qk_s[n * KVGw + h] *= ksc;
    }
    __syncthreads();

    // ---- phase B: exact softmax per head over the split ----
    __shared__ float m_sh[KVGw], l_sh[KVGw];
    const int lane = tid & 31, warp = tid >> 5;
    constexpr int NW = THR / 32;
#pragma unroll
    for (int h = 0; h < KVGw; ++h) {
        float m = -INFINITY;
        for (int n = tid; n < n_tok; n += THR) m = fmaxf(m, qk_s[n * KVGw + h]);
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
        if (lane == 0) red_s[h * NW + warp] = m;
        __syncthreads();
        if (tid == 0) {
            float bm = red_s[h * NW];
            for (int w = 1; w < NW; ++w) bm = fmaxf(bm, red_s[h * NW + w]);
            m_sh[h] = bm;
        }
        __syncthreads();
        float l = 0.f;
        const float mh = m_sh[h];
        for (int n = tid; n < n_tok; n += THR) {
            const float e = __expf(qk_s[n * KVGw + h] - mh);
            qk_s[n * KVGw + h] = e;          // qk buffer now holds p (unnormalised)
            l += e;
        }
        for (int o = 16; o; o >>= 1) l += __shfl_down_sync(0xffffffff, l, o);
        if (lane == 0) red_s[h * NW + warp] = l;
        __syncthreads();
        if (tid == 0) {
            float bl = 0.f;
            for (int w = 0; w < NW; ++w) bl += red_s[h * NW + w];
            l_sh[h] = bl;
        }
        __syncthreads();
    }

    // ---- fold V per-token scale into p once (v^ = cb * vscale) ----
    for (int n = tid; n < n_tok; n += THR) {
        const long loc = (long)kv_indices[kv_start + s_start + n];
        const float vsc = V_SZ[loc * s_vsz_b + (long)kvh * s_vsz_h];
#pragma unroll
        for (int h = 0; h < KVGw; ++h) qk_s[n * KVGw + h] *= vsc;
    }
    // ---- restage V index rows over the K ones ----
    __syncthreads();
    for (int i = tid; i < n_tok * NGw; i += THR) {
        const int n = i / NGw, g = i % NGw;
        const long loc = (long)kv_indices[kv_start + s_start + n];
        idx_s[n * NGw + g] = V_Idx[loc * s_vi_b + (long)kvh * s_vi_h + g];
    }
    __syncthreads();

    // ---- phase C: per-group V staging + weighted sum ----
    // Each thread owns ONE (head, out-channel-within-group) pair per group iteration:
    // THR=256 = KVG(4) * 8 * 8 warpsets... simpler: thread t covers head h = t / 64,
    // channel j = (t % 64) % Gw, token stride = (t % 64) / Gw over a 8-token comb.
    // v1 keeps it simple: threads stride tokens, accumulate per-thread partial
    // out[KVG][Gw] in registers, block-reduce per group at the end of the group loop.
    const __half* vcb_h = V_CB + (long)kvh * (NGw * KCw * Gw);
    float* out_base = Att_Out + (long)bb * s_o_b + (long)bs * s_o_s;
    for (int g = 0; g < NGw; ++g) {
        __syncthreads();
        const __half* src = vcb_h + (long)g * (KCw * Gw);
        for (int i = tid; i < KCw * Gw; i += THR) tbl_s[i] = src[i];
        __syncthreads();

        float acc[KVGw][Gw];
#pragma unroll
        for (int h = 0; h < KVGw; ++h)
#pragma unroll
            for (int j = 0; j < Gw; ++j) acc[h][j] = 0.f;

        for (int n = tid; n < n_tok; n += THR) {
            const int idx = (int)idx_s[n * NGw + g];
            const __half* cw = tbl_s + idx * Gw;
            float p[KVGw];
#pragma unroll
            for (int h = 0; h < KVGw; ++h) p[h] = qk_s[n * KVGw + h];
#pragma unroll
            for (int j = 0; j < Gw; ++j) {
                const float vv = __half2float(cw[j]);
#pragma unroll
                for (int h = 0; h < KVGw; ++h) acc[h][j] = fmaf(p[h], vv, acc[h][j]);
            }
        }
        // block-reduce acc into att_out[..., g*Gw + j]
#pragma unroll
        for (int h = 0; h < KVGw; ++h) {
#pragma unroll
            for (int j = 0; j < Gw; ++j) {
                float v = acc[h][j];
                for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
                if (lane == 0) red_s[warp] = v;
                __syncthreads();
                if (tid == 0) {
                    float t = 0.f;
                    for (int w = 0; w < NW; ++w) t += red_s[w];
                    const long qh = (long)kvh * KVGw + h;
                    out_base[qh * s_o_h + g * Gw + j] = t / l_sh[h];
                }
                __syncthreads();
            }
        }
    }

    if (tid < KVGw) {
        const long qh = (long)kvh * KVGw + tid;
        Att_Lse[(long)bb * s_l_b + qh * s_l_h + (long)bs * s_l_s] =
            m_sh[tid] + logf(l_sh[tid]);
    }
}

void vqwide_stage1_init() {
    const int sm = (int)(sizeof(__half) * KCw * Gw
                         + sizeof(int16_t) * CAPw * NGw
                         + sizeof(float) * CAPw * KVGw
                         + sizeof(float) * KVGw * Lw
                         + sizeof(float) * Lw
                         + sizeof(long) * HALFw
                         + sizeof(float) * (VQW_THR / 32) * KVGw + 256);
#define REG(QT, IT, MODE) TORCH_CHECK(cudaFuncSetAttribute(                     \
        vqwide_stage1<QT, IT, VQW_THR, MODE>,                                   \
        cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess,        \
        "vqwide_cuda: cudaFuncSetAttribute failed for ", sm, " bytes")
    REG(__half, long, 0); REG(__half, long, 1); REG(__half, long, 2);
    REG(__nv_bfloat16, long, 0); REG(__nv_bfloat16, long, 1); REG(__nv_bfloat16, long, 2);
    REG(__half, int, 0); REG(__half, int, 1); REG(__half, int, 2);
    REG(__nv_bfloat16, int, 0); REG(__nv_bfloat16, int, 1); REG(__nv_bfloat16, int, 2);
#undef REG
}

void vqwide_stage1_cuda(torch::Tensor q, torch::Tensor k_idx, torch::Tensor k_cb,
                        torch::Tensor v_idx, torch::Tensor v_cb,
                        torch::Tensor k_sz, torch::Tensor v_sz,
                        torch::Tensor cos_sin, torch::Tensor freq_idx,
                        torch::Tensor w_perm,
                        torch::Tensor kv_indptr, torch::Tensor kv_indices,
                        torch::Tensor splits, torch::Tensor att_out,
                        torch::Tensor att_lse, int64_t n_splits, double sm_scale,
                        int64_t pos_offset, int64_t mode) {
    const int B = q.size(0), H_KV = k_idx.size(1);
    dim3 grid(B, H_KV, n_splits);
    const int sm = (int)(sizeof(__half) * KCw * Gw
                         + sizeof(int16_t) * CAPw * NGw
                         + sizeof(float) * CAPw * KVGw
                         + sizeof(float) * KVGw * Lw
                         + sizeof(float) * Lw
                         + sizeof(long) * HALFw
                         + sizeof(float) * (VQW_THR / 32) * KVGw + 256);
#define ARGS(IT)                                                               \
        k_idx.data_ptr<int16_t>(),                                             \
        reinterpret_cast<const __half*>(k_cb.data_ptr()),                      \
        v_idx.data_ptr<int16_t>(),                                             \
        reinterpret_cast<const __half*>(v_cb.data_ptr()),                      \
        k_sz.data_ptr<float>(), v_sz.data_ptr<float>(),                        \
        cos_sin.data_ptr<float>(), freq_idx.data_ptr<long>(),                  \
        w_perm.data_ptr<float>(), kv_indptr.data_ptr<int>(),                   \
        reinterpret_cast<const IT*>(kv_indices.data_ptr()),                    \
        splits.data_ptr<int>(),                                                \
        att_out.data_ptr<float>(), att_lse.data_ptr<float>(),                  \
        (float)sm_scale, (int)pos_offset,                                      \
        q.stride(0), q.stride(1),                                              \
        k_idx.stride(0), k_idx.stride(1), v_idx.stride(0), v_idx.stride(1),    \
        k_sz.stride(0), k_sz.stride(1), v_sz.stride(0), v_sz.stride(1),        \
        cos_sin.stride(0),                                                     \
        att_out.stride(0), att_out.stride(1), att_out.stride(2),               \
        att_lse.stride(0), att_lse.stride(1), att_lse.stride(2)
#define GO(QT, IT, MODE) do {                                                  \
        vqwide_stage1<QT, IT, VQW_THR, MODE>                                   \
            <<<grid, VQW_THR, sm, at::cuda::getCurrentCUDAStream()>>>(         \
                reinterpret_cast<const QT*>(q.data_ptr()), ARGS(IT));          \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    } while (0)
#define PICK(QT, IT) do {                                                      \
        if (mode == 0) GO(QT, IT, 0);                                          \
        else if (mode == 1) GO(QT, IT, 1);                                     \
        else GO(QT, IT, 2);                                                    \
    } while (0)
    const bool i64 = kv_indices.scalar_type() == at::kLong;
    if (q.scalar_type() == at::kBFloat16) {
        if (i64) PICK(__nv_bfloat16, long); else PICK(__nv_bfloat16, int);
    } else {
        if (i64) PICK(__half, long); else PICK(__half, int);
    }
#undef PICK
#undef GO
#undef ARGS
}
"""


@functools.lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load_inline

    m = load_inline(
        name="sgl_vqwide_stage1_cuda",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "void vqwide_stage1_cuda(torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " int64_t, double, int64_t, int64_t);\n"
            "void vqwide_stage1_init();"
        ),
        cuda_sources=_CUDA_SRC,
        functions=["vqwide_stage1_cuda", "vqwide_stage1_init"],
        extra_cuda_cflags=["-O3"],
    )
    m.vqwide_stage1_init()
    return m


def supports(ng, kc, l, kvg, seq_len_max, n_splits):
    if (ng, kc, l, kvg) not in _SUPPORTED_GEOMS:
        return False
    per = ((seq_len_max + n_splits - 1) // n_splits + _MIN_BLOCK_KV - 1) \
        // _MIN_BLOCK_KV * _MIN_BLOCK_KV
    return per <= _SPLIT_CAP


def run(q, k_idx, k_cb, v_idx, v_cb, k_sz, v_sz, kv_indptr, kv_indices,
        num_kv_splits, n_splits, att_out, att_lse, sm_scale,
        cos_sin_cache=None, pos_offset=0, freq_idx=None, w_perm=None):
    dev = q.device
    mode = 0 if cos_sin_cache is None else (2 if freq_idx is not None else 1)
    dummy_f = torch.zeros(1, 1, device=dev, dtype=torch.float32)
    dummy_l = torch.zeros(1, 1, device=dev, dtype=torch.int64)
    _ext().vqwide_stage1_cuda(
        q, k_idx, k_cb.contiguous(), v_idx, v_cb.contiguous(),
        k_sz.contiguous(), v_sz.contiguous(),
        cos_sin_cache.float().contiguous() if cos_sin_cache is not None else dummy_f,
        freq_idx.contiguous() if freq_idx is not None else dummy_l,
        w_perm.float().contiguous() if w_perm is not None else dummy_f,
        kv_indptr.int(), kv_indices, num_kv_splits.int(),
        att_out, att_lse, n_splits, float(sm_scale), int(pos_offset), mode,
    )
