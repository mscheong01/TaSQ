"""CUDA stage-1 for vq2 decode attention: the codebook lives in shared memory.

The Triton kernel (`_fwd_grouped_kernel_stage1_quant_vq2`) gathers each codeword
straight from global memory. Nsight Compute shows that costs 392 M L1 sectors at
74% of L1 peak against int2's 94 M at 30%, and *sectors per request* are nearly
identical between the two kernels (19.4 vs 17.8) -- so the penalty is sector
VOLUME, not address divergence. 32 lanes fetching 4 B each out of a 32 KB table
cost ~24 sectors per warp instruction however the table is addressed, which is
why cache hints, eviction policy, group-major layout and launch-config tuning all
measured as no-ops or regressions.

Shared memory has no sectors. Staging the 32 KB table there drops global load
sectors to 37.4 M and closes the gap to int2:

    shape           int2      Triton vq2      this kernel
    bs=64 ctx=30k   1602 us   1942 (1.21x)    1604 us  (1.00x int2, 0.83x Triton)
    bs=32 ctx=60k   1593 us   1934 (1.21x)    1503 us  (0.94x int2, 0.78x Triton)
    bs=16 ctx=30k    368 us    501 (1.36x)     375 us  (1.02x int2, 0.75x Triton)
    bs=64 ctx=8k     402 us    536 (1.33x)     423 us  (1.05x int2, 0.79x Triton)

Full derivation, ablations and the list of measured dead ends are in the
kernel-study notes accompanying the paper.

Opt-in via SGLANG_VQ2_CUDA=1. `supports()` gates every assumption the kernel
bakes in; anything else falls back to Triton, so this can only ever affect the
exact configuration it was validated on.

Scores accumulate in fp16 (half2), which is 2.3e-03 relative on qk but only
3.2e-04 on the stage-1 output because softmax damps it. Set SGLANG_VQ2_CUDA_FP32=1
for fp32 accumulation at ~13% lower throughput.
"""
from __future__ import annotations

import functools
import os

import torch

# Geometry this kernel is specialised for. Qwen3-8B and Qwen3-4B-Thinking-2507
# both match exactly; anything else takes the Triton path.
# Geometries the parameterized source is validated for. Constraints baked
# into the kernel: KC == 256 (uint8 indices), G == 4 (packed int32 codewords),
# NG == L/4, KVG % 4 == 0 (float4 p chunks), L % 64 == 0 (uint4 V staging),
# L <= 128 (one packed byte per lane).
#   (32, 256, 128, 4)  Qwen3 dense family
#   (16, 256, 64, 8)   gpt-oss-20b (64 q / 8 kv heads, head_dim 64)
_SUPPORTED_GEOMS = {(32, 256, 128, 4), (16, 256, 64, 8)}
_DEFAULT_GEOM = (32, 256, 128, 4)
_MIN_BLOCK_KV = 32

_CUDA_SRC = r"""
#include <torch/extension.h>
#ifndef VQ2_THR
#define VQ2_THR 128
#endif
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

// Geometry: overridable at build time (-DNG=.. etc) so one source serves
// every (NG, KC, L, KVG) in SUPPORTED_GEOMS. Derived invariants: a packed V
// byte always holds 4 two-bit quarters with dim = byte + (L/4)*quarter, so
// the quarter loops stay 4; what varies is bytes/token, index words, and
// float4 chunks of p per token.
// The overrides arrive as VQ2_GEOM_* (bare -DL=.. would clobber template
// parameters named L inside torch headers, which are included above).
#ifdef VQ2_GEOM_NG
#define NG VQ2_GEOM_NG
#else
#define NG 32
#endif
#ifdef VQ2_GEOM_KC
#define KC VQ2_GEOM_KC
#else
#define KC 256
#endif
#ifdef VQ2_GEOM_L
#define L VQ2_GEOM_L
#else
#define L 128
#endif
#ifdef VQ2_GEOM_KVG
#define KVG VQ2_GEOM_KVG
#else
#define KVG 4
#endif
#define MINBK 32
#define NBY (L / 4)      // packed V bytes per (token, head); lane < NBY owns one
#define NIW (NG / 4)     // int32 index words per (token, head)
#define KV4 (KVG / 4)    // float4 chunks of p per token

__device__ __forceinline__ float q2f(const __half v)          { return __half2float(v); }
__device__ __forceinline__ float q2f(const __nv_bfloat16 v)   { return __bfloat162float(v); }

// Two e5m2 bytes -> one half2 in ONE instruction: prmt builds [0, b0, 0, b1],
// which is exactly the pair of <<8 shifts that decode e5m2 to fp16 (they share
// sign and a 5-bit exponent with the same bias). The scalar path needs eight.
__device__ __forceinline__ __half2 unpack2(unsigned int cw, unsigned int sel) {
    union { unsigned int u; __half2 h; } c;
    c.u = __byte_perm(cw, 0u, sel);
    return c.h;
}
#define SEL_LO 0x1404u
#define SEL_HI 0x3424u

// kv_indices is int64 in the engine (mixed_quant_kv_indices) but int32 in
// the standalone benchmark, so the index type is a template parameter
// rather than a per-call conversion copy of a seq_lens_sum-sized tensor.
template <typename QT, typename IT, int THR, bool H2>
__global__ __launch_bounds__(THR) void vq2_stage1(
    const QT*      __restrict__ Q,
    const uint8_t* __restrict__ K_Idx,
    const int*     __restrict__ CB,
    const uint8_t* __restrict__ V_Buf,
    const float*   __restrict__ K_SZ,
    const float*   __restrict__ V_SZ,
    const int*     __restrict__ kv_indptr,
    const IT*      __restrict__ kv_indices,
    const int*     __restrict__ num_kv_splits,
    float*         __restrict__ Att_Out,
    float*         __restrict__ Att_Lse,
    float sm_scale,
    long s_q_b, long s_q_h,
    long s_ki_b, long s_ki_h, long s_vb_b, long s_vb_h,
    long s_ksz_b, long s_ksz_h, long s_vsz_b, long s_vsz_h,
    long s_o_b, long s_o_h, long s_o_s,
    long s_l_b, long s_l_h, long s_l_s)
{
    constexpr int NW = THR / 32;
    constexpr int BLOCK_N = THR;
    const int bb = blockIdx.x, kvh = blockIdx.y, bs = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    extern __shared__ __align__(16) char smem_raw[];
    int*     cb_s  = reinterpret_cast<int*>(smem_raw);              // 32 KB
    uint8_t* vpk_s = reinterpret_cast<uint8_t*>(cb_s + NG * KC);
    // acc_s only lives in the epilogue, so it reuses the V staging region.
    constexpr int AREG = NW * KVG * 4 * 32 * 4;
    constexpr int VREG = BLOCK_N * NBY > AREG ? BLOCK_N * NBY : AREG;
    float*   acc_s = reinterpret_cast<float*>(vpk_s);
    __half2* q2_s  = reinterpret_cast<__half2*>(vpk_s + VREG);
    float4*  q4_s  = reinterpret_cast<float4*>(q2_s + KVG * NG * 2);
    float4*  qk_s  = reinterpret_cast<float4*>(q4_s + KVG * NG);
    __shared__ float redm[KVG][NW], reds[KVG][NW], m_s[KVG], l_s[KVG];

    const int kv_start = kv_indptr[bb];
    const int seq_len  = kv_indptr[bb + 1] - kv_start;
    const int splits   = num_kv_splits[bb];
#ifdef VQ2_TRACE
    if (bb == 0 && kvh == 0 && bs == 0 && tid == 0) {
        static __device__ unsigned int ncall = 0;
        const unsigned int c = atomicAdd(&ncall, 1u);
        if ((c % 512u) == 0u)
            printf("[vq2_cuda TRACE] call=%u seq_len=%d splits=%d gridz=%d\n",
                   c, seq_len, splits, gridDim.z);
    }
#endif
    const int per = ((seq_len + splits - 1) / splits + MINBK - 1) / MINBK * MINBK;
    const int s_start = per * bs;
    const int s_end   = min(s_start + per, seq_len);
    if (s_end <= s_start) return;   // Triton leaves these cells untouched too

    const int* cb_head = CB + (long)kvh * (NG * KC);
    for (int i = tid; i < NG * KC; i += THR) cb_s[i] = cb_head[i];
    for (int i = tid; i < KVG * NG; i += THR) {
        const int h = i / NG, g = i % NG;
        const QT* qp = Q + (long)bb * s_q_b + (long)(kvh * KVG + h) * s_q_h + g * 4;
        const float a = q2f(qp[0]), b = q2f(qp[1]), c = q2f(qp[2]), d = q2f(qp[3]);
        q2_s[2 * i + 0] = __floats2half2_rn(a, b);
        q2_s[2 * i + 1] = __floats2half2_rn(c, d);
        q4_s[i] = make_float4(a, b, c, d);
    }
    if (tid < KVG) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }

    // PV assignment: warp-strided tokens, and a lane owns the four quarters of
    // ONE packed V byte -- dims {lane, lane+32, lane+64, lane+96} are exactly
    // that byte's four 2-bit fields. One byte plus one float4 of p feeds 16
    // FMAs, so PV costs 2 shared loads/token instead of 8.
    float acc16[KVG][4], corrl[KVG];
#pragma unroll
    for (int h = 0; h < KVG; ++h) {
        corrl[h] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc16[h][j] = 0.f;
    }
    __syncthreads();

    for (int base = s_start; base < s_end; base += BLOCK_N) {
        const int nmax = min(BLOCK_N, s_end - base);
        const int n = base + tid;
        const bool ok = n < s_end;
        const long loc = ok ? kv_indices[kv_start + n] : 0;

        const uint8_t* ip = K_Idx + loc * s_ki_b + (long)kvh * s_ki_h;
        unsigned int iw[NIW];
#pragma unroll
        for (int wq = 0; wq < NIW / 4; ++wq) {
            const uint4 t = *reinterpret_cast<const uint4*>(ip + 16 * wq);
            iw[4 * wq + 0] = t.x; iw[4 * wq + 1] = t.y;
            iw[4 * wq + 2] = t.z; iw[4 * wq + 3] = t.w;
        }

        __half2 a2[KVG];
        float    af[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            a2[h] = __floats2half2_rn(0.f, 0.f);
            af[h] = 0.f;
        }
#ifndef VQ2_REP
#define VQ2_REP 1
#endif
        for (int rep = 0; rep < VQ2_REP; ++rep)
        // FULLY unrolled: the score accumulators are a serial dependency chain,
        // and at `unroll 4` there is almost no ILP. Worth 1843 -> 1630 us.
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int idx = (iw[g >> 2] >> ((g & 3) * 8)) & 0xFF;
            const unsigned int cw = cb_s[g * KC + idx];
            const __half2 k01 = unpack2(cw, SEL_LO);
            const __half2 k23 = unpack2(cw, SEL_HI);
            if (H2) {
#pragma unroll
                for (int h = 0; h < KVG; ++h) {
                    // The two half2 are adjacent, so ONE 8 B load gets both.
                    // Two separate __half2 loads cost 256 shared loads/token
                    // against the fp32 path's 128, which is why half2 loses
                    // without this.
                    const uint2 qq =
                        *reinterpret_cast<const uint2*>(&q2_s[(h * NG + g) * 2]);
                    union { unsigned int u; __half2 h2; } c0, c1;
                    c0.u = qq.x; c1.u = qq.y;
                    a2[h] = __hfma2(c0.h2, k01, a2[h]);
                    a2[h] = __hfma2(c1.h2, k23, a2[h]);
                }
            } else {
                const float k0 = __low2float(k01), k1 = __high2float(k01);
                const float k2 = __low2float(k23), k3 = __high2float(k23);
#pragma unroll
                for (int h = 0; h < KVG; ++h) {
                    const float4 qv = q4_s[h * NG + g];
                    af[h] = fmaf(qv.x, k0, af[h]);
                    af[h] = fmaf(qv.y, k1, af[h]);
                    af[h] = fmaf(qv.z, k2, af[h]);
                    af[h] = fmaf(qv.w, k3, af[h]);
                }
            }
        }
        const float ksc = ok ? K_SZ[loc * s_ksz_b + (long)kvh * s_ksz_h + 0] : 0.f;
        const float2 vsz = ok
            ? *reinterpret_cast<const float2*>(V_SZ + loc * s_vsz_b + (long)kvh * s_vsz_h)
            : make_float2(0.f, 0.f);
        float qkv[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            const float v = H2 ? (__low2float(a2[h]) + __high2float(a2[h])) : af[h];
            qkv[h] = ok ? (v * ksc * sm_scale) : -INFINITY;
        }

        const uint8_t* vp = V_Buf + loc * s_vb_b + (long)kvh * s_vb_h;
        uint4 vv[NBY / 16];
#pragma unroll
        for (int wq = 0; wq < NBY / 16; ++wq)
            vv[wq] = *reinterpret_cast<const uint4*>(vp + 16 * wq);
        __syncthreads();
#pragma unroll
        for (int wq = 0; wq < NBY / 16; ++wq)
            *reinterpret_cast<uint4*>(vpk_s + tid * NBY + 16 * wq) = vv[wq];

        float mloc[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            mloc[h] = qkv[h];
            for (int o = 16; o; o >>= 1)
                mloc[h] = fmaxf(mloc[h], __shfl_down_sync(0xffffffff, mloc[h], o));
            if (lane == 0) redm[h][warp] = mloc[h];
        }
        __syncthreads();

        float mnew[KVG], rsc[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            float bmax = redm[h][0];
#pragma unroll
            for (int w = 1; w < NW; ++w) bmax = fmaxf(bmax, redm[h][w]);
            mnew[h] = fmaxf(m_s[h], bmax);
            rsc[h] = (m_s[h] == -INFINITY) ? 0.f : __expf(m_s[h] - mnew[h]);
            // corr is accumulated below, so it must be rescaled BEFORE this
            // block contributes -- rescaling after silently scales the new term.
            corrl[h] *= rsc[h];
#pragma unroll
            for (int j = 0; j < 4; ++j) acc16[h][j] *= rsc[h];
        }

        // The (scale, zero) affine folds out of the PV inner loop entirely:
        //   sum p*((q2-z)*s) = sum (p*s)*q2 - sum (p*s)*z
        float pv4[KVG], sloc[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            const float e = (qkv[h] == -INFINITY) ? 0.f : __expf(qkv[h] - mnew[h]);
            sloc[h] = e;                       // l uses the plain exp
            pv4[h] = e * vsz.x;
            corrl[h] += pv4[h] * vsz.y;
        }
#pragma unroll
        for (int c = 0; c < KV4; ++c)
            qk_s[tid * KV4 + c] = make_float4(pv4[4 * c + 0], pv4[4 * c + 1],
                                              pv4[4 * c + 2], pv4[4 * c + 3]);
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            for (int o = 16; o; o >>= 1)
                sloc[h] += __shfl_down_sync(0xffffffff, sloc[h], o);
            if (lane == 0) reds[h][warp] = sloc[h];
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            float sm = reds[h][0];
#pragma unroll
            for (int w = 1; w < NW; ++w) sm += reds[h][w];
            if (tid == 0) { m_s[h] = mnew[h]; l_s[h] = l_s[h] * rsc[h] + sm; }
        }

        // One warp covers TPW tokens at once. NBY = L/4 bytes belong to a token,
        // so at L=128 that is exactly a warp (TPW=1, the original mapping). At
        // L=64 only 16 lanes would own a byte and the upper half of every warp
        // would issue predicated-off instructions for the whole PV phase --
        // measured as 2.67x int2's instruction count at gpt-oss geometry.
        // Splitting the warp across TPW tokens keeps all 32 lanes live.
        constexpr int TPW = (NBY >= 32) ? 1 : (32 / NBY);
        const int sub = lane / NBY;   // which of the TPW tokens this lane serves
        const int lb = lane % NBY;    // byte owned within that token
#pragma unroll 4
        for (int nl = warp * TPW + sub; nl < nmax; nl += NW * TPW) {
            const unsigned int byte = vpk_s[nl * NBY + lb];
#pragma unroll
            for (int c = 0; c < KV4; ++c) {
                const float4 pp = qk_s[nl * KV4 + c];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float q2 = (float)((byte >> (2 * j)) & 0x3);
                    acc16[4 * c + 0][j] = fmaf(pp.x, q2, acc16[4 * c + 0][j]);
                    acc16[4 * c + 1][j] = fmaf(pp.y, q2, acc16[4 * c + 1][j]);
                    acc16[4 * c + 2][j] = fmaf(pp.z, q2, acc16[4 * c + 2][j]);
                    acc16[4 * c + 3][j] = fmaf(pp.w, q2, acc16[4 * c + 3][j]);
                }
            }
        }
    }

    __syncthreads();
#pragma unroll
    for (int h = 0; h < KVG; ++h)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc_s[((warp * KVG + h) * 4 + j) * 32 + lane] = acc16[h][j];
#pragma unroll
    for (int h = 0; h < KVG; ++h) {
        float c = corrl[h];
        for (int o = 16; o; o >>= 1) c += __shfl_down_sync(0xffffffff, c, o);
        if (lane == 0) reds[h][warp] = c;
    }
    __syncthreads();
    // each warp reduces heads {warp, warp+NW, ...}; lanes >= NBY held zeros
    const int ol = lane;
    for (int oh = warp; oh < KVG; oh += NW) {
        float corr = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) corr += reds[oh][w];
        const long qh = (long)kvh * KVG + oh;
        // Real strides: the caller passes a SLICE of a [bs, heads,
        // total_splits, L] scratch buffer, so the head stride is
        // total_splits*L, not n_splits*L.
        float* op = Att_Out + (long)bb * s_o_b + qh * s_o_h + (long)bs * s_o_s;
        const float inv = 1.f / l_s[oh];
        if (ol < NBY) {
            // With TPW tokens per warp, output position `ol` was accumulated by
            // lanes ol, ol+NBY, ... -- one per token slot -- so the epilogue sums
            // over sub-groups as well as warps. TPW=1 reproduces the old sum.
            constexpr int TPWo = (NBY >= 32) ? 1 : (32 / NBY);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float t = 0.f;
#pragma unroll
                for (int w = 0; w < NW; ++w)
#pragma unroll
                    for (int s = 0; s < TPWo; ++s)
                        t += acc_s[((w * KVG + oh) * 4 + j) * 32 + s * NBY + ol];
                op[j * NBY + ol] = (t - corr) * inv;
            }
        }
        if (ol == 0)
            Att_Lse[(long)bb * s_l_b + qh * s_l_h + (long)bs * s_l_s] =
                m_s[oh] + logf(l_s[oh]);
    }
}

// Registering the >48 KB dynamic-shared opt-in must happen OUTSIDE CUDA graph
// capture. Doing it lazily on first launch put it inside sglang's capture, the
// launch failed, the failure was never checked, and the graph recorded a no-op:
// decode then ran 8x "faster" while attending nothing. Called once from Python
// at import.
void vq2_stage1_init() {
    constexpr int THR = VQ2_THR;
    const int NWv = THR / 32;
    const int areg = NWv * KVG * 4 * 32 * 4;
    const int vreg = THR * NBY > areg ? THR * NBY : areg;
    const int sm = (int)(sizeof(int) * NG * KC + vreg
                         + sizeof(__half2) * KVG * NG * 2
                         + sizeof(float4) * KVG * NG
                         + sizeof(float) * THR * KVG);
#define REG(QT, IT, H2) TORCH_CHECK(cudaFuncSetAttribute(                        \
        vq2_stage1<QT, IT, THR, H2>,                                            \
        cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess,        \
        "vq2_cuda: cudaFuncSetAttribute failed for ", sm, " bytes shared")
    REG(__half, int, true);            REG(__half, int, false);
    REG(__half, long, true);           REG(__half, long, false);
    REG(__nv_bfloat16, int, true);     REG(__nv_bfloat16, int, false);
    REG(__nv_bfloat16, long, true);    REG(__nv_bfloat16, long, false);
#undef REG
}

void vq2_stage1_cuda(torch::Tensor q, torch::Tensor k_idx, torch::Tensor cb,
                     torch::Tensor v_buf, torch::Tensor k_sz, torch::Tensor v_sz,
                     torch::Tensor kv_indptr, torch::Tensor kv_indices,
                     torch::Tensor splits, torch::Tensor att_out,
                     torch::Tensor att_lse, int64_t n_splits, double sm_scale,
                     int64_t fp32) {
    constexpr int THR = VQ2_THR;
    const int B = q.size(0), H_KV = k_idx.size(1);
    dim3 grid(B, H_KV, n_splits);
    const int NWv = THR / 32;
    const int areg = NWv * KVG * 4 * 32 * 4;
    const int vreg = THR * NBY > areg ? THR * NBY : areg;
    const int sm = (int)(sizeof(int) * NG * KC + vreg
                         + sizeof(__half2) * KVG * NG * 2
                         + sizeof(float4) * KVG * NG
                         + sizeof(float) * THR * KVG);
#define ARGS(IT)                                                               \
        k_idx.data_ptr<uint8_t>(), cb.data_ptr<int>(),                         \
        v_buf.data_ptr<uint8_t>(), k_sz.data_ptr<float>(),                     \
        v_sz.data_ptr<float>(), kv_indptr.data_ptr<int>(),                     \
        reinterpret_cast<const IT*>(kv_indices.data_ptr()),                    \
        splits.data_ptr<int>(),                                                \
        att_out.data_ptr<float>(), att_lse.data_ptr<float>(),                  \
        (float)sm_scale, q.stride(0), q.stride(1),                             \
        k_idx.stride(0), k_idx.stride(1), v_buf.stride(0), v_buf.stride(1),    \
        k_sz.stride(0), k_sz.stride(1), v_sz.stride(0), v_sz.stride(1),        \
        att_out.stride(0), att_out.stride(1), att_out.stride(2),               \
        att_lse.stride(0), att_lse.stride(1), att_lse.stride(2)
// The opt-in attribute is a host-side property of the function, not stream
// work, so it is set once rather than per launch -- sglang captures the decode
// path into CUDA graphs, and there is no reason to re-issue it under capture.
// MUST launch on PyTorch's CURRENT stream, not the legacy default stream.
// sglang captures decode into CUDA graphs on a side stream; a kernel launched
// on the default stream is simply not captured, so the graph contained NO
// quant stage-1 at all. Decode then skipped attention entirely and looked
// 1.6-8.3x "faster" (profiler: the 282 ms of stage-1 just vanished), while
// eager-mode tests still passed because the default stream syncs implicitly.
#define GO(QT, IT, H2) do {                                                    \
        vq2_stage1<QT, IT, THR, H2>                                            \
            <<<grid, THR, sm, at::cuda::getCurrentCUDAStream()>>>(             \
                reinterpret_cast<const QT*>(q.data_ptr()), ARGS(IT));          \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    } while (0)
#define PICK_H2(QT, IT) do {                                                   \
        if (fp32) GO(QT, IT, false); else GO(QT, IT, true);                    \
    } while (0)
    const bool i64 = kv_indices.scalar_type() == at::kLong;
    if (q.scalar_type() == at::kBFloat16) {
        if (i64) PICK_H2(__nv_bfloat16, long); else PICK_H2(__nv_bfloat16, int);
    } else {
        if (i64) PICK_H2(__half, long); else PICK_H2(__half, int);
    }
#undef PICK_H2
#undef GO
#undef ARGS
}
"""


@functools.lru_cache(maxsize=4)
def _ext(geom=_DEFAULT_GEOM):
    from torch.utils.cpp_extension import load_inline

    m = _build(load_inline, geom)
    m.vq2_stage1_init()      # must run outside CUDA graph capture
    return m


def _build(load_inline, geom=_DEFAULT_GEOM):
    ng, kc, l, kvg = geom
    return load_inline(
        name="sgl_vq2_stage1_cuda"
             + (f"_g{ng}x{kc}x{l}x{kvg}" if geom != _DEFAULT_GEOM else "")
             + (f"_t{_thr()}" if _thr() != 128 else "")
             + ("_trace" if os.environ.get("SGLANG_VQ2_CUDA_TRACE") == "1" else "")
             + ("_rep" + os.environ["SGLANG_VQ2_CUDA_REP"]
                if os.environ.get("SGLANG_VQ2_CUDA_REP") else ""),
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "void vq2_stage1_cuda(torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " int64_t, double, int64_t);\n"
            "void vq2_stage1_init();"
        ),
        cuda_sources=_CUDA_SRC,
        functions=["vq2_stage1_cuda", "vq2_stage1_init"],
        extra_cuda_cflags=(["-O3", "--use_fast_math", f"-DVQ2_THR={_thr()}",
                            f"-DVQ2_GEOM_NG={ng}", f"-DVQ2_GEOM_KC={kc}",
                            f"-DVQ2_GEOM_L={l}", f"-DVQ2_GEOM_KVG={kvg}"]
                           + (["-DVQ2_TRACE"]
                              if os.environ.get("SGLANG_VQ2_CUDA_TRACE") == "1"
                              else [])
                           + ([f"-DVQ2_REP={os.environ['SGLANG_VQ2_CUDA_REP']}"]
                              if os.environ.get("SGLANG_VQ2_CUDA_REP")
                              else [])),
        verbose=False,
    )


def prebuild() -> None:
    """Compile and load the extension NOW.

    sglang profiles free device memory to size the KV pool during model init,
    before the first decode. Loading the extension lazily (on the first decode,
    i.e. during CUDA-graph capture) puts its CUDA module memory OUTSIDE that
    measurement, so the pool is sized as if it were free and a long-context cell
    OOMs mid-run. Called from TritonAttnBackend.__init__.

    SGLANG_VQ2_CUDA_GEOM="16,256,64,8" prebuilds a non-default geometry too;
    without it a non-default model would first compile DURING capture.
    """
    _ext()
    extra = os.environ.get("SGLANG_VQ2_CUDA_GEOM")
    if extra:
        geom = tuple(int(x) for x in extra.split(","))
        assert geom in _SUPPORTED_GEOMS, f"unsupported vq2 CUDA geometry {geom}"
        _ext(geom)


def _thr() -> int:
    """Threads (== BLOCK_N tokens) per stage-1 block.

    128 is the Qwen-tuned default. Narrow-head geometries stage the same
    NG*KC codebook in shared memory but return half the payload per gather,
    so the tradeoff between occupancy and per-block reuse moves.
    """
    return int(os.environ.get("SGLANG_VQ2_CUDA_THR", "128"))


def enabled() -> bool:
    return os.environ.get("SGLANG_VQ2_CUDA", "0") == "1"


def supports(q, k_idx_buffer, cb_packed, v_buffer, k_scales_zeros,
             v_scales_zeros, att_out, att_lse, kv_indptr, kv_indices,
             num_kv_splits, logit_cap, xai_temperature_len, v_vq: bool) -> bool:
    """Every assumption the kernel bakes in. Anything unmet -> Triton."""
    if v_vq or logit_cap > 0 or xai_temperature_len > 0:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    L = v_buffer.shape[-1] * 4
    ng, kc = k_idx_buffer.shape[-1], cb_packed.shape[-1]
    kvg = q.shape[1] // k_idx_buffer.shape[1]
    if (ng, kc, L, kvg) not in _SUPPORTED_GEOMS:
        return False
    # e5m2 only: the prmt decode is a <<8 per byte, which is e5m2's exact bit
    # layout. e4m3 would need a different (and slower) conversion.
    from sglang.srt.mem_cache.vq_codebook import resolve_vq_fp8_fmt

    if resolve_vq_fp8_fmt() != "e5m2":
        return False
    if k_scales_zeros.shape[-1] != 2 or v_scales_zeros.shape[-1] != 2:
        return False
    # Exact dtypes the extension reinterprets. kv_indices may be int32 or int64;
    # everything else is fixed. Checked rather than assumed, because a mismatch
    # surfaces as a torch data_ptr<T>() type error mid CUDA-graph capture.
    if (k_idx_buffer.dtype != torch.uint8 or v_buffer.dtype != torch.uint8
            or cb_packed.dtype != torch.int32
            or k_scales_zeros.dtype != torch.float32
            or v_scales_zeros.dtype != torch.float32
            or att_out.dtype != torch.float32 or att_lse.dtype != torch.float32
            or kv_indptr.dtype != torch.int32
            or num_kv_splits.dtype != torch.int32
            or kv_indices.dtype not in (torch.int32, torch.int64)):
        return False
    if att_out.stride(-1) != 1 or att_lse.stride(-1) != 1:
        return False
    return att_out.shape[-1] == v_buffer.shape[-1] * 4


_announced = False


def launch(q, k_idx_buffer, cb_packed, v_buffer, k_scales_zeros, v_scales_zeros,
           att_out, att_lse, kv_indptr, kv_indices, num_kv_splits,
           max_kv_splits, sm_scale):
    global _announced
    if not _announced:
        # One line, once: without it an inactive flag and a null result look
        # identical in the benchmark output.
        print(f"[vq2_cuda] CUDA stage-1 ACTIVE (q={q.dtype}, "
              f"fp32_acc={os.environ.get('SGLANG_VQ2_CUDA_FP32', '0')})", flush=True)
        _announced = True
    geom = (k_idx_buffer.shape[-1], cb_packed.shape[-1],
            v_buffer.shape[-1] * 4, q.shape[1] // k_idx_buffer.shape[1])
    _ext(geom).vq2_stage1_cuda(
        q, k_idx_buffer, cb_packed, v_buffer, k_scales_zeros, v_scales_zeros,
        kv_indptr, kv_indices, num_kv_splits, att_out, att_lse,
        int(max_kv_splits), float(sm_scale),
        1 if os.environ.get("SGLANG_VQ2_CUDA_FP32", "0") == "1" else 0,
    )
