"""Shared helpers for int2 quantized KV prefill.

These utilities were originally inline methods on :class:`TritonAttnBackend`.
They're factored out here so other attention backends (notably FA3) can reuse
the same rotation (Hadamard for MHA-int2, Oscar for unified HP+int2) and
HP+int2 aware dequantization pipeline, keeping a single source of truth for
int2 prefill semantics.

Callers are expected to drive the higher level flow themselves:

  1. Call :func:`prepare_quantized_extend_qkv` before writing KV to the pool.
     Pass the returned ``pre_rotated_k`` / ``pre_rotated_v`` to
     ``set_kv_buffer(..., already_hadamard_transformed=True)`` so the pool does
     not rotate again.
  2. Call :func:`dequantize_prefix_kv` to materialize contiguous per-token K/V
     for the cached prefix (any mix of HP and int2 tiers is handled).
  3. Concatenate the dequantized prefix with the newly rotated extend K/V and
     run ``flash_attn_varlen_func`` (or equivalent).
  4. Call :func:`apply_inverse_v_rotation` on the attention output if the V
     path was rotated (Hadamard self-inverse, or ``result @ R_v.T`` for
     Oscar).

The module is intentionally framework-light: it only depends on torch +
the JIT Hadamard kernel (``sglang.jit_kernel.hadamard``, with an optional
``fast_hadamard_transform`` fast path) and on the pool surface
(``get_raw_key_buffer`` / ``get_key_scales_zeros`` / ...) that both
``MHATokenToKVPool`` and ``UnifiedInt2HPKVPool`` already implement.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional, Tuple

from sglang.srt.environ import envs
import torch
import triton
import triton.language as tl

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    from sglang.jit_kernel.hadamard import hadamard_transform

from sglang.srt.mem_cache.kv_quant_kernels import (
    _get_num_scale_groups,
    dequantize_kv_int2_triton,
)
from sglang.srt.mem_cache.vq_codebook import (
    vq_idx_bits,
    vq_idx_is_packed,
    vq_map_k,
    vq_map_q,
    vq_unpack_idx,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


def apply_segmented_hadamard_transform(
    tensor: torch.Tensor,
    hadamard_order: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Segmented (blockwise) FWHT along the last dim, matching the fused int2
    write / decode kernels.

    ``hadamard_order`` defaults to ``envs.HADAMARD_ORDER.get()`` when
    unspecified. ``out_dtype``, when set, casts the input before transforming
    (used by callers that want to upcast bf16 → fp32 internally before the
    rotation, then cast back). The returned tensor has the same shape as
    ``tensor`` and a dtype determined by ``hadamard_transform``'s rules
    (typically the input dtype after the optional cast). Self-inverse with
    the ``1/sqrt(order)`` pre-normalization.

    Single canonical implementation; previously duplicated in three places
    (memory_pool.py, this file, and inline in triton_backend.py).
    """
    # Naive-INT2 baseline reproduction: SGLANG_INT2_NO_HADAMARD makes the plain-int2
    # rotation an identity everywhere (decode-Q, output, prefill K/V), so the int2 path
    # quantizes raw K/V with no rotation. Only affects the plain (non-OSCAR) int2 path.
    if envs.SGLANG_INT2_NO_HADAMARD.get():
        return tensor if out_dtype is None else tensor.to(out_dtype)
    if hadamard_order is None:
        hadamard_order = envs.HADAMARD_ORDER.get()
    if out_dtype is not None:
        tensor = tensor.to(out_dtype)
    return hadamard_transform(
        tensor.view(
            *tensor.shape[:-1],
            tensor.shape[-1] // hadamard_order,
            hadamard_order,
        )
        / math.sqrt(hadamard_order)
    ).view_as(tensor)


# Backward-compatible alias for module-internal callers.
_apply_segmented_hadamard_transform = apply_segmented_hadamard_transform


def _pool_uses_oscar_rotation(kv_pool) -> bool:
    """True for the mixed HP+int2 unified pool that loads per-layer Oscar
    rotation matrices. Non-oscar int2 pools (MHA without mixed HP+int2
    storage) implicitly use the legacy segmented Hadamard rotation.
    """
    return getattr(kv_pool, "_R_k", None) is not None


def resolve_layer_quant_pool(kv_pool, layer_id: int):
    """Resolve the pool that actually stores this layer's KV and whether it
    is int2-quantized.

    Returns ``(quant_pool, eff_layer_id)`` where ``quant_pool`` is the pool
    to run the quantized attention path against (``None`` if this layer's KV
    is not quantized) and ``eff_layer_id`` is the layer id in that pool's
    own numbering.

    For hybrid-SWA models (gpt-oss) the top-level pool is ``SWAKVPool``,
    which fans layers out to two inner pools; only the full-attention inner
    pool may be quantized, and its layers are re-indexed densely (global odd
    layer ids -> 0..N_full-1), matching the dense layer axis of the
    calibration bundles (rotations/codebooks are built per layer_map.json).
    """
    layers_mapping = getattr(kv_pool, "layers_mapping", None)
    if layers_mapping is not None:  # hybrid SWA wrapper
        idx, is_swa = layers_mapping[layer_id]
        inner = kv_pool.swa_kv_pool if is_swa else kv_pool.full_kv_pool
        if getattr(inner, "dtype", None) == "int2":
            return inner, idx
        return None, layer_id
    if getattr(kv_pool, "dtype", None) == "int2":
        return kv_pool, layer_id
    return None, layer_id


def _apply_oscar_rotation(tensor: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply ``tensor @ R`` along the last dim. Returns a contiguous tensor in
    ``R.dtype``. Works for shapes ``[*, head_dim]``.
    """
    return (tensor.to(R.dtype) @ R).contiguous()


def adjust_vq_attention_sinks(
    q: torch.Tensor,
    sinks: torch.Tensor,
    mean: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Move learned sink logits into the mean-centered VQ score space.

    VQ stores ``(k - mean) @ forward``, so mapped Q/K logits are shifted by
    ``-sm_scale * q.dot(mean)`` relative to the model's original logits.
    That shift is softmax-invariant among real keys, but a learned attention
    sink must receive the same per-query shift to preserve its probability.
    """
    token_num, q_head_num, head_dim = q.shape
    kv_head_num = mean.shape[0]
    assert q_head_num % kv_head_num == 0
    group_num = q_head_num // kv_head_num
    q_grouped = q.to(torch.float32).view(
        token_num, kv_head_num, group_num, head_dim
    )
    mean_shift = (
        (q_grouped * mean.to(torch.float32).view(1, kv_head_num, 1, head_dim))
        .sum(dim=-1)
        .reshape(token_num, q_head_num)
    )
    return sinks.to(torch.float32).unsqueeze(0) - mean_shift * sm_scale


def disable_vq_attention_sink_correction() -> bool:
    return os.environ.get("SGLANG_VQ_DISABLE_SINK_CORRECTION", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def prepare_quantized_extend_qkv(
    kv_pool,
    layer: "RadixAttention",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_already_hadamard_transformed: bool = False,
    kv_already_hadamard_transformed: bool = False,
    layer_id_override: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Apply the pool's active rotation to Q/K/V for int2 extend.

    Returns the (possibly) rotated ``q, k, v`` tensors and a flag indicating
    whether the attention output must be inverse-rotated afterwards. For
    non-int2 pools this is a no-op.

    ``layer_id_override`` carries the pool-local layer id when ``kv_pool`` is
    an inner pool of a hybrid-SWA wrapper (dense re-indexed layers).
    """
    need_v_inverse = False
    kv_dtype = kv_pool.dtype
    if kv_dtype != "int2":
        return q, k, v, need_v_inverse

    eff_layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
    if _pool_uses_oscar_rotation(kv_pool):
        layer_idx = eff_layer_id - kv_pool.start_layer
        R_k = kv_pool._R_k[layer_idx]
        R_v = kv_pool._R_v[layer_idx]
        # debug (SGLANG_DEBUG_KV_WRITE_SUM=1): one line per call from the ONE
        # shared rotation site -- shows whether the two prefill backends feed
        # different inputs, resolve different layer ids, or pick different R.
        import os as _os
        if _os.environ.get("SGLANG_DEBUG_KV_WRITE_SUM") == "1":
            print(f"[prep] gl={layer.layer_id} eff={eff_layer_id} idx={layer_idx} "
                  f"kin={k.float().sum().item():.6e} qin={q.float().sum().item():.6e} "
                  f"Rk={R_k.float().abs().sum().item():.6e} "
                  f"aht_q={q_already_hadamard_transformed} aht_kv={kv_already_hadamard_transformed}",
                  flush=True)
        v_rotation_absorbed = bool(
            getattr(layer, "oscar_v_rotation_absorbed", False)
        )
        if getattr(kv_pool, "vq_enabled", False):
            # vq2 storage space: K residual r = (k - mean) @ forward per
            # head; q mapped with inverse.T so both tiers' scores share the
            # same softmax-invariant -q.mean shift. V keeps the R_v path.
            vq = kv_pool._vq
            # identity_map (CQ): forward == q_map == I, mean == 0, so both maps are
            # no-ops. Skipping is bit-identical. See VQCodebook.identity_map.
            if not q_already_hadamard_transformed and not vq.identity_map:
                q = vq_map_q(q, vq.q_map[layer_idx])
            if not kv_already_hadamard_transformed:
                if vq.identity_map:
                    k = k.contiguous()
                else:
                    k = vq_map_k(
                        k, vq.forward[layer_idx], vq.mean[layer_idx]
                    ).contiguous()
                if v_rotation_absorbed:
                    v = v.to(R_v.dtype).contiguous()
                else:
                    v = _apply_oscar_rotation(v, R_v)
            need_v_inverse = True
            return q, k, v, need_v_inverse
        if not q_already_hadamard_transformed:
            q = _apply_oscar_rotation(q, R_k)
        if not kv_already_hadamard_transformed:
            k = _apply_oscar_rotation(k, R_k)
            if v_rotation_absorbed:
                v = v.to(R_v.dtype).contiguous()
            else:
                v = _apply_oscar_rotation(v, R_v)
        need_v_inverse = True
        return q, k, v, need_v_inverse

    if not q_already_hadamard_transformed:
        q = _apply_segmented_hadamard_transform(q)
    if not kv_already_hadamard_transformed:
        k = _apply_segmented_hadamard_transform(k)
        v = _apply_segmented_hadamard_transform(v)
    need_v_inverse = True
    return q, k, v, need_v_inverse


@triton.jit
def _mixed_prefix_dequant_kernel(
    prefix_indices_ptr,
    quant_ptr,
    scales_zeros_ptr,
    hp_ptr,
    out_ptr,
    num_tokens,
    num_heads,
    head_dim: tl.constexpr,
    quant_stride_token: tl.constexpr,
    quant_stride_head: tl.constexpr,
    quant_stride_dim: tl.constexpr,
    sz_stride_token: tl.constexpr,
    sz_stride_head: tl.constexpr,
    sz_stride_dim: tl.constexpr,
    hp_stride_token: tl.constexpr,
    hp_stride_head: tl.constexpr,
    hp_stride_dim: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    PACK_BITS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_DIM)
    dim_mask = offs < head_dim

    slot = tl.load(prefix_indices_ptr + token_idx)
    is_hp = slot >= HP_OFFSET
    # Plane-packed: PACK_DEN coords share a byte, coord d at byte (d % plane_dim),
    # field (d // plane_dim). PACK_BITS=2 is the historical int2 crumb layout
    # (plane_dim = head_dim // 4); PACK_BITS=1 is the int1 arena (head_dim // 8).
    PACK_DEN: tl.constexpr = 8 // PACK_BITS
    plane_dim = head_dim // PACK_DEN

    byte_offsets = offs % plane_dim
    packed = tl.load(
        quant_ptr
        + slot * quant_stride_token
        + head_idx * quant_stride_head
        + byte_offsets * quant_stride_dim,
        mask=(~is_hp) & dim_mask,
        other=0,
    )
    shift = (offs // plane_dim) * PACK_BITS
    q = ((packed >> shift) & ((1 << PACK_BITS) - 1)).to(tl.float32)

    group_ids = offs // GROUP_SIZE
    scale = tl.load(
        scales_zeros_ptr
        + slot * sz_stride_token
        + head_idx * sz_stride_head
        + (group_ids * 2) * sz_stride_dim,
        mask=(~is_hp) & dim_mask,
        other=1.0,
    ).to(tl.float32)
    zero = tl.load(
        scales_zeros_ptr
        + slot * sz_stride_token
        + head_idx * sz_stride_head
        + (group_ids * 2 + 1) * sz_stride_dim,
        mask=(~is_hp) & dim_mask,
        other=0.0,
    ).to(tl.float32)
    quant_val = (q - zero) * scale

    hp_slot = slot - HP_OFFSET
    hp_val = tl.load(
        hp_ptr
        + hp_slot * hp_stride_token
        + head_idx * hp_stride_head
        + offs * hp_stride_dim,
        mask=is_hp & dim_mask,
        other=0.0,
    )
    out_val = tl.where(is_hp, hp_val, quant_val)
    tl.store(
        out_ptr
        + token_idx * out_stride_token
        + head_idx * out_stride_head
        + offs * out_stride_dim,
        out_val,
        mask=(token_idx < num_tokens) & (head_idx < num_heads) & dim_mask,
    )


def _mixed_prefix_dequantize_tensor(
    prefix_indices: torch.Tensor,
    quantized: torch.Tensor,
    scales_zeros: torch.Tensor,
    hp: torch.Tensor,
    hp_offset: int,
    head_dim: int,
    model_dtype: torch.dtype,
    pack_bits: int = 2,
) -> torch.Tensor:
    num_tokens = prefix_indices.shape[0]
    num_heads = quantized.shape[1]
    out = torch.empty(
        (num_tokens, num_heads, head_dim),
        dtype=model_dtype,
        device=prefix_indices.device,
    )
    if num_tokens == 0:
        return out
    num_groups = _get_num_scale_groups(scales_zeros)
    group_size = head_dim // num_groups
    grid = (num_tokens, num_heads)
    _mixed_prefix_dequant_kernel[grid](
        prefix_indices,
        quantized,
        scales_zeros,
        hp,
        out,
        num_tokens,
        num_heads,
        head_dim,
        quantized.stride(0),
        quantized.stride(1),
        quantized.stride(2),
        scales_zeros.stride(0),
        scales_zeros.stride(1),
        scales_zeros.stride(2),
        hp.stride(0),
        hp.stride(1),
        hp.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HP_OFFSET=int(hp_offset),
        GROUP_SIZE=group_size,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        PACK_BITS=pack_bits,
        num_warps=4,
        num_stages=1,
    )
    return out


def _vq_prefix_dequantize_k(
    prefix_indices: torch.Tensor,
    kv_pool,
    layer_id: int,
    model_dtype: torch.dtype,
    token_chunk: int = 16384,
) -> torch.Tensor:
    """vq2 analogue of the K half of ``_mixed_prefix_dequantize_tensor``:
    HP slots pass through; quant slots reconstruct ``cb[idx] * ptn_scale``
    (in the stored residual space). Token-chunked to bound temporaries."""
    layer_idx = layer_id - kv_pool.start_layer
    vq = kv_pool._vq
    cb16 = vq.cb16[layer_idx]  # [H, NG, K, G] fp16
    hp = kv_pool.get_hp_key_buffer(layer_id)
    idx_buf = kv_pool.get_raw_key_buffer(layer_id)  # uint8 [slots+1, H, NG]
    sz = kv_pool.get_key_scales_zeros(layer_id)
    hp_off = int(kv_pool.hp_global_offset)
    trash = int(kv_pool.quant_size)
    H, NG, K, G = cb16.shape
    D = NG * G

    T = prefix_indices.shape[0]
    out = torch.empty((T, H, D), dtype=model_dtype, device=prefix_indices.device)
    h_ids = torch.arange(H, device=prefix_indices.device).view(1, H, 1)
    g_ids = torch.arange(NG, device=prefix_indices.device).view(1, 1, NG)
    # NOTE: this function returns STORED-space rows for every tier (quant recon AND exact
    # HP rows -- both hold the vq residual r = (k - mean) @ forward for perm_rope bundles).
    # The perm_rope stored->natural inversion happens in dequantize_prefix_kv AFTER the
    # exact-chunked-prefill shadow swap, so shadow rows (also stored space) get the same
    # treatment; see the caller.
    for t0 in range(0, T, token_chunk):
        t1 = min(t0 + token_chunk, T)
        slots = prefix_indices[t0:t1]
        is_hp = slots >= hp_off
        quant_slot = torch.where(is_hp, torch.full_like(slots, trash), slots)
        hp_slot = torch.where(is_hp, slots - hp_off, torch.zeros_like(slots))
        # the arena is a 10-bit bitstream for NG=16 bundles; decode unpacks in-kernel,
        # this read side does it here
        raw = idx_buf[quant_slot]
        idx = (vq_unpack_idx(raw, vq_idx_bits(K)) if raw.shape[-1] != NG
               else raw.long())  # [t, H, NG]
        # fp32 for the centroid*scale product, matching vq_dequant and the
        # decode kernel's fp32 accumulate. In fp16 this is the ONLY place a
        # residual row is materialised at reduced precision, and it feeds
        # attention logits directly.
        scale = sz[quant_slot][..., 0].to(torch.float32)  # [t, H]
        r_hat = cb16[h_ids, g_ids, idx].view(t1 - t0, H, D).to(torch.float32)
        r_hat = r_hat * scale.unsqueeze(-1)
        hp_rows = hp[hp_slot].to(torch.float32)
        out[t0:t1] = torch.where(is_hp.view(-1, 1, 1), hp_rows, r_hat).to(model_dtype)
    return out


def _vq_prefix_dequantize_v(
    prefix_indices: torch.Tensor,
    idx_buf: torch.Tensor,      # uint8 [slots+1, H, NG_V] VQ V indices
    sz: torch.Tensor,           # [slots+1, H, 2], slot 0 = ptn scale
    hp: torch.Tensor,           # [hp_slots, H, D] bf16 (R_v space)
    cb16: torch.Tensor,         # [H, NG_V, K, G] fp16 (one layer)
    inv_perm: torch.Tensor,     # strided perm inverse: permuted -> natural coords
    hp_offset: int,
    trash: int,
    model_dtype: torch.dtype,
    token_chunk: int = 16384,
) -> torch.Tensor:
    """VQ-V analogue of the V half of ``dequantize_prefix_kv``. The arena
    holds codebook indices in STRIDED-group order (pool applies _vq_v_perm
    before encode; the decode kernel's plane layout matches), so the
    reconstruction must gather contiguous groups then inverse-permute back
    to natural coords: v_hat = unperm(cb[idx]) * ptn_scale. Raw-tensor
    signature so the integrity gate can exercise it without a pool."""
    H, NGV, K, G = cb16.shape
    D = NGV * G
    T = prefix_indices.shape[0]
    out = torch.empty((T, H, D), dtype=model_dtype, device=prefix_indices.device)
    h_ids = torch.arange(H, device=prefix_indices.device).view(1, H, 1)
    g_ids = torch.arange(NGV, device=prefix_indices.device).view(1, 1, NGV)
    for t0 in range(0, T, token_chunk):
        t1 = min(t0 + token_chunk, T)
        slots = prefix_indices[t0:t1]
        is_hp = slots >= hp_offset
        quant_slot = torch.where(is_hp, torch.full_like(slots, trash), slots)
        hp_slot = torch.where(is_hp, slots - hp_offset, torch.zeros_like(slots))
        raw = idx_buf[quant_slot]
        idx = vq_unpack_idx(raw, vq_idx_bits(K)) if raw.shape[-1] != NGV else raw.long()
        scale = sz[quant_slot][..., 0].to(torch.float16)
        v_hat_p = cb16[h_ids, g_ids, idx].view(t1 - t0, H, D)  # permuted order
        v_hat = v_hat_p[..., inv_perm] * scale.unsqueeze(-1)
        hp_rows = hp[hp_slot].to(torch.float16)
        out[t0:t1] = torch.where(is_hp.view(-1, 1, 1), hp_rows, v_hat).to(model_dtype)
    return out


def dequantize_prefix_kv(
    kv_pool,
    layer_id: int,
    prefix_indices: torch.Tensor,
    model_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize the prefix slots referenced by ``prefix_indices`` into dense
    ``[num_tokens, head_num, head_dim]`` tensors in ``model_dtype``.

    Supports both ``MHATokenToKVPool`` (all slots are int2) and
    ``UnifiedInt2HPKVPool`` (some slots are HP, others are int2; classified
    by ``slot >= hp_global_offset`` -- HP slot ids start at exactly
    ``hp_global_offset``).

    Grouped scales (``scales.shape[-1] > 2``) are handled by
    ``dequantize_kv_int2_triton`` internally.
    """
    device = prefix_indices.device
    if prefix_indices.numel() == 0:
        return (
            torch.empty(
                (0, kv_pool.head_num, kv_pool.head_dim),
                dtype=model_dtype,
                device=device,
            ),
            torch.empty(
                (0, kv_pool.head_num, kv_pool.v_head_dim),
                dtype=model_dtype,
                device=device,
            ),
        )

    prefix_indices = prefix_indices.to(torch.int64)
    if (
        getattr(kv_pool, "mixed_kv_enabled", None) is not None
        and kv_pool.mixed_kv_enabled()
    ):
        assert kv_pool.dtype == "int2", (
            f"Unsupported quantized KV dtype: {kv_pool.dtype}"
        )
        if getattr(kv_pool, "vq_enabled", False):
            prefix_k = _vq_prefix_dequantize_k(
                prefix_indices, kv_pool, layer_id, model_dtype
            )
        else:
            prefix_k = _mixed_prefix_dequantize_tensor(
                prefix_indices,
                kv_pool.get_raw_key_buffer(layer_id),
                kv_pool.get_key_scales_zeros(layer_id),
                kv_pool.get_hp_key_buffer(layer_id),
                kv_pool.hp_global_offset,
                kv_pool.head_dim,
                model_dtype,
            )
        if getattr(kv_pool, "vq_v_enabled", False):
            # VQ-V: the V arena holds codebook indices, NOT packed int2 —
            # the int2 crumb dequant would silently decode garbage V for the
            # whole quantized prefix during chunked prefill.
            layer_idx = layer_id - kv_pool.start_layer
            ngv = kv_pool._vq_v.num_groups
            if kv_pool._vq_v.wide_g:
                # The encode side skips the strided permutation for wide-G
                # codebooks (unified_kv_pool._vq_write_v / the flush twin:
                # ``r_v_in = r_v if vq.wide_g else r_v[..., self._vq_v_perm]``),
                # so the arena already holds natural-order groups. Applying the
                # inverse perm here anyway scrambled all V channels of every
                # quantized prefix row read during chunked prefill.
                inv_perm = torch.arange(
                    kv_pool.v_head_dim, device=kv_pool._vq_v_perm.device
                )
            else:
                inv_perm = torch.empty_like(kv_pool._vq_v_perm)
                inv_perm[kv_pool._vq_v_perm] = torch.arange(
                    kv_pool.v_head_dim, device=kv_pool._vq_v_perm.device
                )
            prefix_v = _vq_prefix_dequantize_v(
                prefix_indices,
                kv_pool.get_raw_value_buffer(layer_id),
                kv_pool.get_value_scales_zeros(layer_id),
                kv_pool.get_hp_value_buffer(layer_id),
                kv_pool._vq_v.cb16[layer_idx],
                inv_perm,
                int(kv_pool.hp_global_offset),
                int(kv_pool.quant_size),
                model_dtype,
            )
        else:
            prefix_v = _mixed_prefix_dequantize_tensor(
                prefix_indices,
                kv_pool.get_raw_value_buffer(layer_id),
                kv_pool.get_value_scales_zeros(layer_id),
                kv_pool.get_hp_value_buffer(layer_id),
                kv_pool.hp_global_offset,
                kv_pool.v_head_dim,
                model_dtype,
                pack_bits=getattr(kv_pool, "_v_int_bits", 2),
            )
        if getattr(kv_pool, "shadow_enabled", False) and kv_pool.shadow_active():
            # Exact chunked prefill: quant slots written by earlier chunks of
            # a still-prefilling request carry a bf16 shadow of their exact
            # stored-space rows. Prefer those over the reconstructions so a
            # chunked prefill matches a single-chunk prefill numerically.
            # Covers K and V in ALL quant flavors (VQ or int2) since the swap
            # happens after reconstruction. Shape-static: unregistered slots
            # map to -1 and route to the shadow trash row.
            layer_idx = layer_id - kv_pool.start_layer
            trash_slot = int(kv_pool.quant_size)
            q_safe = torch.where(
                prefix_indices < trash_slot,
                prefix_indices,
                torch.full_like(prefix_indices, trash_slot),
            )
            rows = kv_pool._shadow_map[q_safe].to(torch.int64)
            has_shadow = rows >= 0
            s_trash = kv_pool._shadow_k.shape[1] - 1
            rows_safe = torch.where(
                has_shadow, rows, torch.full_like(rows, s_trash)
            )
            mask = has_shadow.view(-1, 1, 1)
            side = envs.SGLANG_MIXED_KV_SHADOW_SIDE.get()
            if side in ("kv", "k"):
                prefix_k = torch.where(
                    mask,
                    kv_pool._shadow_k[layer_idx][rows_safe].to(model_dtype),
                    prefix_k,
                )
            if side in ("kv", "v"):
                prefix_v = torch.where(
                    mask,
                    kv_pool._shadow_v[layer_idx][rows_safe].to(model_dtype),
                    prefix_v,
                )
        vq_obj = getattr(kv_pool, "_vq", None)
        if (
            getattr(kv_pool, "vq_enabled", False)
            and vq_obj is not None
            and getattr(vq_obj, "perm_rope", False)
        ):
            # TaSQ: every K row above (quant recon, exact HP, AND exact-chunked-prefill
            # shadow rows) is in vq STORED space (r = (k - mean) @ forward; the HP write
            # path runs vq_map_k too). Map back to NATURAL pre-RoPE space via the bundle's
            # documented recon map ("k_hat = r_hat @ inverse + mean", inverse = q_map^T) so
            # the caller's NeoX rotation and stored-space remap
            # is valid. Placed AFTER the shadow swap deliberately -- doing it inside
            # _vq_prefix_dequantize_k left shadow rows un-inverted (double-transformed
            # downstream). CQ is a no-op case (forward=identity, mean=0); the VQ pool never sets
            # perm_rope and keeps stored space to match its mapped query.
            li = layer_id - kv_pool.start_layer
            inv_map = vq_obj.q_map[li].transpose(-1, -2).to(torch.float32)
            mean_nat = vq_obj.mean[li].to(torch.float32).unsqueeze(0)
            prefix_k = (
                torch.einsum("thd,hde->the", prefix_k.to(torch.float32), inv_map)
                + mean_nat
            ).to(model_dtype)
        return (prefix_k, prefix_v)

    raw_k = kv_pool.get_raw_key_buffer(layer_id)[prefix_indices]
    raw_v = kv_pool.get_raw_value_buffer(layer_id)[prefix_indices]
    scales_k = kv_pool.get_key_scales_zeros(layer_id)[prefix_indices]
    scales_v = kv_pool.get_value_scales_zeros(layer_id)[prefix_indices]
    assert kv_pool.dtype == "int2", (
        f"Unsupported quantized KV dtype: {kv_pool.dtype}"
    )
    return (
        dequantize_kv_int2_triton(raw_k, scales_k, kv_pool.head_dim, model_dtype),
        dequantize_kv_int2_triton(raw_v, scales_v, kv_pool.v_head_dim, model_dtype),
    )


def apply_inverse_v_rotation(
    result: torch.Tensor,
    kv_pool,
    layer: "RadixAttention",
    need_v_inverse: bool,
    layer_id_override: Optional[int] = None,
) -> torch.Tensor:
    """Apply the inverse V rotation on an attention output tensor, when
    required. Hadamard is self-inverse (segmented FWHT), Oscar inverts as
    ``result @ R_v.T`` in ``R_v``'s dtype.

    ``result`` must have shape ``[..., v_head_dim]``; callers should reshape
    beforehand if their output is stored flattened.
    """
    if not need_v_inverse or kv_pool.dtype != "int2":
        return result
    if _pool_uses_oscar_rotation(kv_pool):
        eff_layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        layer_idx = eff_layer_id - kv_pool.start_layer
        if getattr(kv_pool, "_R_v_identity", False):
            return result  # R_v == I: result @ I.T == result
        R_v = kv_pool._R_v[layer_idx]
        return (result.to(R_v.dtype) @ R_v.T).contiguous()
    return _apply_segmented_hadamard_transform(result)


@triton.jit
def _build_prefix_indices_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    prefix_lens_ptr,
    prefix_indptr_ptr,
    out_ptr,
    req_to_token_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx)
    prefix_len = tl.load(prefix_lens_ptr + req_idx)
    out_start = tl.load(prefix_indptr_ptr + req_idx)
    mask = offs < prefix_len
    slots = tl.load(
        req_to_token_ptr + req_pool_idx * req_to_token_stride + offs,
        mask=mask,
        other=0,
    ).to(tl.int64)
    tl.store(out_ptr + out_start + offs, slots, mask=mask)


def _cpu_int_list(values) -> Optional[list[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            return None
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]

def build_prefix_indices_from_req_to_token(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_seqlens_cpu=None,
) -> torch.Tensor:
    """Gather valid prefix slot ids per request as a flat 1-D tensor by
    reading ``req_to_token`` directly (no page striding / page_size division).

    Required by the unified mixed HP+int2 prefill path. HP slot ids in that
    pool are encoded ``HP_OFFSET + hp_page_id`` and adjacent positions can
    map to arbitrary HP page ids, so the page-table-based reconstruction
    used by :func:`build_prefix_indices_from_page_table` returns garbled HP
    slot ids and reads the wrong rows out of ``hp_k_buffer`` / ``hp_v_buffer``.

    Inputs:
        req_to_token       : int32 [max_req_slots, max_context_len]
        req_pool_indices   : int64 [bs] -- per-request row index
        cache_seqlens      : int32 or int64 [bs] -- valid prefix length

    Returns:
        flat int64 1-D tensor of slot ids, in (request, position) order
        consistent with what the FA varlen kernel expects when concatenated
        with the freshly written extend slots.
    """
    device = req_to_token.device
    bs = req_pool_indices.shape[0]
    if bs == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    cache_seqlens_cpu = _cpu_int_list(cache_seqlens_cpu)
    if cache_seqlens_cpu is None:
        raise ValueError(
            "build_prefix_indices_from_req_to_token requires CPU prefix lengths "
            "to avoid CUDA boolean-index synchronization"
        )
    total_prefix = sum(cache_seqlens_cpu)
    out = torch.empty((total_prefix,), dtype=torch.int64, device=device)
    if total_prefix == 0:
        return out
    prefix_lens_cpu = torch.tensor(cache_seqlens_cpu, dtype=torch.int32)
    prefix_indptr_cpu = torch.empty((bs + 1,), dtype=torch.int32)
    prefix_indptr_cpu[0] = 0
    prefix_indptr_cpu[1:] = torch.cumsum(prefix_lens_cpu, dim=0)
    prefix_lens = prefix_lens_cpu.to(device, non_blocking=True)
    prefix_indptr = prefix_indptr_cpu.to(device, non_blocking=True)
    max_prefix_len = max(cache_seqlens_cpu)
    _build_prefix_indices_kernel[(bs,)](
        req_to_token,
        req_pool_indices.to(torch.int64),
        prefix_lens,
        prefix_indptr,
        out,
        req_to_token.stride(0),
        BLOCK_SIZE=triton.next_power_of_2(max_prefix_len),
        num_warps=8,
        num_stages=1,
    )
    return out
