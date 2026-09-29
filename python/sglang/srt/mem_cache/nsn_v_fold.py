"""Fold NSNQuant's value-side Hadamard transform into projection weights.

This matches the reference ``rotate_v_proj`` and ``rotate_o_proj`` operations while removing the
runtime transform. Call ``fold_v_hadamard_`` after loading all local attention weight shards;
tensor parallelism retains complete head-dimension blocks on each rank.
"""
from __future__ import annotations

import math

import torch


def _hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(n)


@torch.no_grad()
def fold_v_hadamard_(attn_module, head_dim: int) -> None:
    """attn_module: a LlamaAttention/Qwen3Attention-style module with `.qkv_proj`, `.o_proj`,
    `.q_size`, `.kv_size` attributes (all in this rank's LOCAL, already-TP-sharded units)."""
    qkv = attn_module.qkv_proj
    q_size, kv_size = attn_module.q_size, attn_module.kv_size
    w = qkv.weight  # [q_size + 2*kv_size, hidden_size] (local shard)
    device, dtype = w.device, w.dtype
    had = _hadamard_matrix(head_dim, device, dtype)

    v_slice = w[q_size + kv_size : q_size + 2 * kv_size, :]
    shape = v_slice.shape
    v_flat = v_slice.transpose(0, 1).reshape(-1, head_dim)  # rotate along OUTPUT (per-head) dim
    v_flat = v_flat @ had.T
    v_new = v_flat.reshape(shape[1], shape[0]).transpose(0, 1).contiguous()
    w[q_size + kv_size : q_size + 2 * kv_size, :] = v_new

    if qkv.bias is not None:
        b = qkv.bias
        b_v = b[q_size + kv_size : q_size + 2 * kv_size]
        b_v_new = (b_v.reshape(-1, head_dim) @ had.T).reshape(b_v.shape)
        b[q_size + kv_size : q_size + 2 * kv_size] = b_v_new

    # o_proj: RowParallelLinear, INPUT dim sharded per-rank -- this rank's weight already holds
    # only its local heads' worth of input columns, contiguous per head_dim block.
    ow = attn_module.o_proj.weight  # [hidden_size, num_heads_local * head_dim]
    o_shape = ow.shape
    ow_flat = ow.reshape(-1, head_dim)
    ow_flat = ow_flat @ had.T
    ow.copy_(ow_flat.reshape(o_shape))
