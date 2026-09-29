#!/usr/bin/env python3
"""Fused nearest-centroid search for NSNQuant encoding.

The standard identity

    argmin_c ||x - c||^2  ==  argmax_c ( x.c - ||c||^2 / 2 )

avoids materializing the full distance matrix. ``ieee=True`` uses fp32 accumulation; finite-
precision near ties may resolve differently from ``torch.cdist``.
"""
import torch, triton, triton.language as tl


@triton.jit
def _argmin_kernel(X, CB, CBSQ, Out, N,
                   G: tl.constexpr, KC: tl.constexpr, BLOCK_K: tl.constexpr,
                   BLOCK_N: tl.constexpr, IEEE: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)          # G padded to tl.dot's 16-element minimum
    mask_k = offs_k < G
    offs_c = tl.arange(0, KC)

    x = tl.load(X + offs_n[:, None] * G + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    cb = tl.load(CB + offs_c[None, :] * G + offs_k[:, None],
                 mask=mask_k[:, None], other=0.0).to(tl.float32)     # [BLOCK_K, KC]
    if IEEE:
        sc = tl.dot(x, cb, input_precision="ieee")
    else:
        sc = tl.dot(x.to(tl.float16), cb.to(tl.float16))
    sc = sc - 0.5 * tl.load(CBSQ + offs_c)[None, :]
    tl.store(Out + offs_n, tl.argmax(sc, axis=1).to(tl.int32), mask=mask_n)


def vq_indices_fast(x, codebook, ieee=True, block_n=16):
    """x [..., G] -> int32 indices [...]; codebook [KC, G]."""
    G = codebook.shape[-1]
    KC = codebook.shape[-2]
    flat = x.reshape(-1, G)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    cb = codebook.contiguous()
    N = flat.shape[0]
    out = torch.empty(N, device=x.device, dtype=torch.int32)
    cbsq = (cb.float() ** 2).sum(-1).contiguous()
    _argmin_kernel[(triton.cdiv(N, block_n),)](
        flat, cb, cbsq, out, N,
        G=G, KC=KC, BLOCK_K=max(16, triton.next_power_of_2(G)),
        BLOCK_N=block_n, IEEE=ieee, num_warps=4,
    )
    return out.reshape(x.shape[:-1])


def vq_nearest_fast(x, codebook, ieee=True):
    """Drop-in for `nsn_quant._vq_nearest`: returns the centroid VALUES, x's shape."""
    idx = vq_indices_fast(x.reshape(-1, codebook.shape[-1]), codebook, ieee=ieee)
    return codebook[idx.long()].reshape(x.shape)
