"""
Unified HP + int2 KV cache pool.

Quant arena: paged with ``N_Q`` slots per page. HP arena: shared HP-prefix
pool (paged) followed by per-request HP-recent ring slabs. Slot id namespace
is flat (``[0, num_quant_pages*N_Q)`` quant, ``[HP_OFFSET, ...)`` HP), and
kernels dispatch by ``slot >= HP_OFFSET``.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.QuantKernel.fused_hadamard_int2_kv import (
    quantized_set_kv_int2_pretransformed_triton,
)
from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
    _launch_single_clip_int2,
    quantized_set_kv_int2_oscar_rotate_k_clip_triton,
    quantized_set_kv_int2_pretransformed_clip_triton,
)
from sglang.srt.mem_cache.kv_quant_kernels import _get_num_scale_groups
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    OscarRotationConfig,
    _set_kv_buffer_impl,
    get_tensor_size_bytes,
    load_oscar_rotation_config,
    load_oscar_rotations,
)
from sglang.srt.mem_cache.vq_codebook import (
    load_vq_codebook,
    vq_encode,
    vq_encode_single,
    vq_idx_bits,
    vq_idx_is_packed,
    vq_map_k,
    vq_pack_idx,
    vq_pack_words,
)

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


@triton.jit
def _set_mixed_hp_buffer_kernel(
    src_ptr,
    dst_ptr,
    loc_ptr,
    num_tokens,
    row_dim: tl.constexpr,
    src_stride_token: tl.constexpr,
    src_stride_dim: tl.constexpr,
    dst_stride_loc: tl.constexpr,
    dst_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    loc = tl.load(loc_ptr + token_idx)
    is_hp = loc >= HP_OFFSET
    hp_loc = loc - HP_OFFSET
    mask = is_hp & (token_idx < num_tokens) & (offs < row_dim)
    vals = tl.load(
        src_ptr + token_idx * src_stride_token + offs * src_stride_dim,
        mask=mask,
        other=0.0,
    )
    tl.store(
        dst_ptr + hp_loc * dst_stride_loc + offs * dst_stride_dim,
        vals,
        mask=mask,
    )


def _resolve_torch_dtype(name: str, *, kind: str) -> torch.dtype:
    """Map a friendly dtype name (``bf16``/``bfloat16``/``fp16``/``half``/``fp32``)
    to the corresponding ``torch.dtype``. ``kind`` is only used in the error
    message ("scale" / "HP" / etc.) so the caller's intent surfaces in the
    failure mode.
    """
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(
        f"Unsupported {kind} dtype: {name}. Expected bf16/fp16/fp32."
    )


def resolve_scale_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="scale")


def resolve_hp_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="HP")


def compute_page_geometry(hp_dtype: torch.dtype) -> Tuple[int, int]:
    """Return ``(N_H, N_Q)`` for int2 + ``hp_dtype``.

    ``N_Q`` is the int2 page size used by the paged quant allocator (and
    ``--page-size``). ``N_H`` is retained as ``1`` for documentation and for
    legacy callers, but no longer carries the LCM byte-equivalence invariant —
    HP and quant arenas are decoupled allocations under the slab design.
    """
    hp_itemsize = torch.empty(0, dtype=hp_dtype).element_size()
    return 1, 4 * hp_itemsize


def compute_recent_ring_size(hp_recent_tokens: int, n_q: int) -> int:
    # Decode reserves the next HP slot before the flush plan releases the
    # oldest N_Q slots. Keep one transient slot beyond the maximum live
    # occupancy so the allocation on a flush step cannot overwrite the
    # oldest slot's full->SWA mapping before that mapping is released.
    return int(hp_recent_tokens) + int(n_q)


class UnifiedInt2HPKVPool(KVCache):
    """Unified HP + int2 MHA KV cache.

    The pool exposes:
      * ``k_buffer[l]``, ``v_buffer[l]``           – quant (int2 packed uint8) views
      * ``hp_k_buffer[l]``, ``hp_v_buffer[l]``     – HP (``hp_dtype``) views
      * ``k_scales_zeros[l]``, ``v_scales_zeros[l]`` – per-group scales+zeros in
        ``scale_dtype`` (bf16/fp16/fp32)

    The quant and HP views alias the same byte arena. Callers must treat a
    physical page as homogeneous (either tier) at any given time; this invariant
    is enforced by the ``UnifiedInt2HPKVAllocator`` that hands out slot ids into
    these views.
    """

    def __init__(
        self,
        num_quant_pages: int,
        hp_dtype: torch.dtype,
        hp_prefix_tokens: int,
        hp_recent_tokens: int,
        dtype: str,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_req_slots: int,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        model_dtype: Optional[torch.dtype] = None,
        kv_cache_quant_group_size: Optional[int] = None,
        scale_dtype: torch.dtype = torch.bfloat16,
        num_hp_prefix_slots: int = 0,
    ):
        assert dtype == "int2", (
            "UnifiedInt2HPKVPool supports only int2 quant tier; got %s" % dtype
        )
        # Work around KVCache.__init__ dtype validation: it stores ``dtype`` as
        # a string and sets ``store_dtype=torch.uint8`` for int2.
        super().__init__(
            size=num_quant_pages,  # used by base class for sizing heuristics only
            page_size=1,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            model_dtype=model_dtype,
        )

        self.num_quant_pages = int(num_quant_pages)
        self.hp_dtype = hp_dtype
        self.scale_dtype = scale_dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.hp_prefix_tokens = int(hp_prefix_tokens)
        self.hp_recent_tokens = int(hp_recent_tokens)
        self.kv_cache_quant_group_size = kv_cache_quant_group_size

        self.N_H, self.N_Q = compute_page_geometry(hp_dtype)
        self.hp_recent_ring_size = compute_recent_ring_size(
            self.hp_recent_tokens, self.N_Q
        )
        if num_hp_prefix_slots > 0 and num_hp_prefix_slots % self.N_Q != 0:
            num_hp_prefix_slots = (
                (num_hp_prefix_slots + self.N_Q - 1) // self.N_Q * self.N_Q
            )
        self.num_hp_prefix_slots = int(num_hp_prefix_slots)
        self.slab_size = self.hp_recent_ring_size  # back-compat alias
        self._hp_offset = self.num_quant_pages * self.N_Q
        self._hp_recent_base = self.num_hp_prefix_slots

        # Window sizes must be N_Q-aligned so radix tree (page_size=N_Q) and
        # the flush kernel land on page boundaries.
        if self.hp_prefix_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_PREFIX_TOKENS ({self.hp_prefix_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        if self.hp_recent_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_RECENT_TOKENS ({self.hp_recent_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        # Flush every N_Q decode steps demotes exactly N_Q HP-recent slots
        # into one quant page. Per-request counter (initialized at admission
        # to (hp_recent+N_Q-1)-H_0) keeps every flush whole-page.
        self.flush_interval = self.N_Q
        self.max_req_slots = int(max_req_slots)
        self._flush_counter = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )
        self._next_slab_offset = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )

        # Forward-done event stashed each iteration; consumed by
        # ``wait_pending_forward`` inside the flush apply phase.
        self._pending_forward_done = None

        # Grouping for quantization.
        self.k_quant_group_size, self.k_num_scale_groups = self._resolve_quant_grouping(
            self.head_dim, "K"
        )
        self.v_quant_group_size, self.v_num_scale_groups = self._resolve_quant_grouping(
            self.v_head_dim, "V"
        )
        assert self.head_dim % 4 == 0, (
            f"head_dim={self.head_dim} must be divisible by 4 for int2 packing"
        )
        assert self.v_head_dim % 4 == 0, (
            f"v_head_dim={self.v_head_dim} must be divisible by 4 for int2 packing"
        )

        # vq2 K quant tier: group-VQ indices (uint8) + per-token RMS scale
        # instead of int2 affine. Activated by SGLANG_VQ_CODEBOOK_PATH; V and
        # all allocator/flush bookkeeping stay on the int2 path.
        vq_path = envs.SGLANG_VQ_CODEBOOK_PATH.get()
        self.vq_enabled = bool(vq_path)
        self._vq = None
        # Under tensor parallelism head_num is the per-rank count while the
        # codebook bundle holds all global KV heads; rank r owns the
        # contiguous slice starting at r * head_num (Megatron sharding).
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        vq_head_start = get_tensor_model_parallel_rank() * self.head_num
        if self.vq_enabled:
            assert self.k_num_scale_groups == 1, (
                "vq2 K tier requires single-scale layout "
                "(--kv-cache-quant-group-size unset)"
            )
            self._vq = load_vq_codebook(
                vq_path,
                layer_num=self.layer_num,
                start_layer=self.start_layer,
                head_num=self.head_num,
                head_dim=self.head_dim,
                device=torch.device(self.device),
                dtype=self.hp_dtype,
                head_start=vq_head_start,
                fold_decode=True,   # K side: see _vq_fold_weights_into_decode_table
            )

        # vq2 V quant tier (additive ablation): group-VQ V indices (uint8) +
        # per-token RMS scale in the R_v=U_S basis instead of int2 affine.
        # Activated by SGLANG_VQ_V_CODEBOOK_PATH; unset keeps OSCAR scalar-INT2
        # V untouched. Storage mirrors the K tier; the flush-encode + stage-2
        # decode gather are not built yet, so a set env fails fast at the end of
        # __init__ (below) rather than half-building the V path.
        vq_v_path = envs.SGLANG_VQ_V_CODEBOOK_PATH.get()
        self.vq_v_enabled = bool(vq_v_path)
        self._vq_v = None
        if self.vq_v_enabled:
            assert self.v_num_scale_groups == 1, (
                "vq2 V tier requires single-scale layout "
                "(--kv-cache-quant-group-size unset)"
            )
            self._vq_v = load_vq_codebook(
                vq_v_path,
                layer_num=self.layer_num,
                start_layer=self.start_layer,
                head_num=self.head_num,
                head_dim=self.v_head_dim,
                device=torch.device(self.device),
                dtype=self.hp_dtype,
                head_start=vq_head_start,
            )
            # Strided coord permutation matching the decode kernel's fixed quarter
            # layout: codebook group g holds V-coords {g, g+NG, g+2NG, g+3NG}.
            # Reorder V by this before the contiguous vq_encode so groups line up;
            # the kernel places the planes back at those coords naturally.
            _ngv = self._vq_v.num_groups
            self._vq_v_perm = torch.tensor(
                [g + m * _ngv for g in range(_ngv) for m in range(self._vq_v.group_dim)],
                dtype=torch.long, device=torch.device(self.device),
            )

        if self.vq_enabled and self._vq.wide_g and self.vq_v_enabled:
            # VQ-V is no longer MANDATORY in wide mode: the vqwide decode kernel now
            # carries the scalar-V branch the packed G==4 path always had, so a
            # wide-G K codebook can be served alongside OSCAR's own int V. It is
            # still the case that IF a V codebook is given it must match K's group
            # dim, since one kernel instance serves both.
            assert self._vq_v.wide_g and self._vq_v.group_dim == self._vq.group_dim, (
                f"K codebook group_dim={self._vq.group_dim} must match V "
                f"codebook group_dim={self._vq_v.group_dim} for the vqwide kernel"
            )

        # Physical V storage width, resolved BEFORE the arenas are built (the oscar
        # config block below runs after _create_arenas and is too late to size a
        # buffer). 1 makes the 1-bit V arm's ALLOCATED size equal its nominal
        # accounting (see SGLANG_V_INT_BITS); 2 is the historical int2 crumb arena
        # every other arm uses and is left bit-identical.
        self._v_int_bits: int = int(envs.SGLANG_V_INT_BITS.get())
        assert self._v_int_bits in (1, 2), (
            f"SGLANG_V_INT_BITS must be 1 or 2, got {self._v_int_bits}"
        )
        if self._v_int_bits == 1:
            _v_max_q = int(envs.SGLANG_V_INT_MAX_Q.get())
            assert _v_max_q == 1, (
                "SGLANG_V_INT_BITS=1 requires SGLANG_V_INT_MAX_Q=1: a 1-bit field "
                f"holds 2 levels, but MAX_Q={_v_max_q} asks for {_v_max_q + 1}. "
                "Storing the wider code would alias silently rather than fail."
            )
            assert not envs.SGLANG_LLOYD_MAX.get(), (
                "SGLANG_V_INT_BITS=1 is defined for the uniform min/max path only"
            )
            assert self.v_head_dim % 8 == 0, (
                f"int1 V packing needs v_head_dim % 8 == 0, got {self.v_head_dim}"
            )
            assert self.v_num_scale_groups == 1, (
                "int1 V requires the single-scale layout "
                "(--kv-cache-quant-group-size == v_head_dim, or unset)"
            )
            assert not self.vq_v_enabled, (
                "SGLANG_V_INT_BITS applies to the scalar (int) V tier; unset "
                "SGLANG_VQ_V_CODEBOOK_PATH to use it"
            )
        # Coordinates per packed byte in the V arena.
        self._v_pack_den: int = 8 // self._v_int_bits

        self._create_arenas()

        # Log the configured V tier so experiment provenance records the active path.
        if self.vq_v_enabled:
            # Code width belongs to the bundle here, not to this setting, so the
            # arena size is all this line can honestly report for the VQ tier.
            logger.info(
                "UnifiedInt2HPKVPool: V tier = vq2 codebook (K=%d, G=%d) | "
                "V arena %d element(s) per (token, head)",
                self._vq_v.codebook_size, self._vq_v.group_dim,
                self.v_buffer[0].shape[-1],
            )
        else:
            # Nominal code width is what the QUANTISER uses (levels = max_q + 1);
            # allocated is what the ARENA holds. They differ whenever a 1-bit code is
            # stored in int2 crumbs, which is precisely the simulated-width case this
            # setting exists to close -- so both are printed, never just one.
            _max_q = int(envs.SGLANG_V_INT_MAX_Q.get())
            nominal_code = max(1, (_max_q + 1 - 1).bit_length())
            meta = 2 * 16  # (scale, zero), accounted at the bf16 the NovaKV row uses
            logger.info(
                "UnifiedInt2HPKVPool: V tier = scalar int%d (max_q=%d, %d levels) | "
                "V arena %d B per (token, head) | nominal %.4f bit/coord "
                "(code %d + scale/zero %.4f at 16-bit) | as-allocated %.4f bit/coord",
                self._v_int_bits, _max_q, _max_q + 1,
                self.v_head_dim // self._v_pack_den,
                nominal_code + meta / self.v_head_dim,
                nominal_code, meta / self.v_head_dim,
                self._v_int_bits + meta / self.v_head_dim,
            )

        # Cached attributes used by the rest of the stack.
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = None
        self.row_dim = self.head_num * self.head_dim  # for store_cache helpers
        self.same_kv_dim = self.head_dim == self.v_head_dim

        # Exact chunked-prefill shadow (SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL):
        # bf16 copies of the quant-tier rows written by a request's non-final
        # prefill chunks, read back by ``dequantize_prefix_kv`` so later
        # chunks of the same prompt attend to exact K/V. Rows live in the
        # STORED space (K residual / R_v-rotated V) -- exactly what the HP
        # tier holds -- so substitution is a straight row swap. Keyed by
        # quant slot id via ``_shadow_map`` (-1 = no shadow); buffers grow
        # per chunk and are dropped wholesale once no request is mid-chunk.
        self.shadow_enabled = envs.SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL.get()
        self._shadow_map: Optional[torch.Tensor] = None
        self._shadow_k: Optional[torch.Tensor] = None  # [L, rows+1, H, Dk]
        self._shadow_v: Optional[torch.Tensor] = None  # [L, rows+1, H, Dv]
        self._shadow_next_row = 0
        self._shadow_rid_slots: dict = {}   # rid -> list[slot tensor]
        self._shadow_release_rids: set = set()
        if self.shadow_enabled:
            self._shadow_map = torch.full(
                (self.quant_size + 1,), -1, dtype=torch.int32, device=self.device
            )

        # Oscar rotation + clip. Per-layer orthogonal matrices [head_dim,
        # head_dim] / [v_head_dim, v_head_dim] are loaded in ``hp_dtype`` so
        # the ``rows @ R`` pre-pass and ``result @ R.T`` inverse are plain
        # bf16 GEMMs.
        self._oscar_cfg: OscarRotationConfig = load_oscar_rotation_config()
        self._k_clip_ratio: float = self._oscar_cfg.k_clip_ratio
        self._v_clip_ratio: float = self._oscar_cfg.v_clip_ratio
        self._lloyd_max: bool = envs.SGLANG_LLOYD_MAX.get()
        self._v_int_max_q: int = int(envs.SGLANG_V_INT_MAX_Q.get())
        self._R_k: torch.Tensor = load_oscar_rotations(
            self._oscar_cfg.k_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
        )
        self._R_v: torch.Tensor = load_oscar_rotations(
            self._oscar_cfg.v_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.v_head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
        )
        logger.info(
            "UnifiedInt2HPKVPool: Oscar rotation enabled (k_clip=%.4f v_clip=%.4f lloyd_max=%s)",
            self._k_clip_ratio,
            self._v_clip_ratio,
            self._lloyd_max,
        )

        # Identity rotations are a no-op, and both CQ and TaSQ ship them that way: every layer of
        # k_rotation_qqt_r_h_pbr.pt and v_rotation_sst_r_h_pbr.pt is exactly I for the bundles this
        # project serves (the OSCAR rotation slots exist for arms that learn one). Left in, each
        # costs a GEMM per layer per step on the write path plus a cast/GEMM/cast/copy on the
        # decode output path -- the 36 extra cutlass launches per step visible in the decode
        # traces. Detected once here, exactly, on the dtype the kernels consume.
        _eye_k = torch.eye(self._R_k.shape[-1], dtype=self._R_k.dtype, device=self._R_k.device)
        _no_skip = os.environ.get("SGLANG_VQ_NO_IDENTITY_SKIP") == "1"
        self._R_k_identity: bool = (not _no_skip) and bool(
            torch.equal(self._R_k, _eye_k.expand_as(self._R_k))
        )
        _eye_v = torch.eye(self._R_v.shape[-1], dtype=self._R_v.dtype, device=self._R_v.device)
        self._R_v_identity: bool = (not _no_skip) and bool(
            torch.equal(self._R_v, _eye_v.expand_as(self._R_v))
        )
        if self._R_k_identity or self._R_v_identity:
            logger.info(
                "UnifiedInt2HPKVPool: identity OSCAR rotation detected (R_k=%s R_v=%s) -- skipping",
                self._R_k_identity, self._R_v_identity,
            )


        hp_total_slots = (
            self.num_hp_prefix_slots
            + self.max_req_slots * self.hp_recent_ring_size
        )
        self._finalize_allocation_log(hp_total_slots)
        hp_itemsize = torch.empty(0, dtype=self.hp_dtype).element_size()
        hp_bytes = (
            hp_total_slots
            * self.layer_num
            * self.head_num
            * (self.head_dim + self.v_head_dim)
            * hp_itemsize
        )
        logger.info(
            "UnifiedInt2HPKVPool: HP arena reserves %.2f GB "
            "(hp_prefix_pool_slots=%d, max_req_slots=%d, recent_ring=%d "
            "= R=%d + N_Q-1=%d, P=%d, layers=%d, head_num=%d, "
            "head_dim+v_head_dim=%d, hp_dtype=%s)",
            hp_bytes / GB,
            self.num_hp_prefix_slots,
            self.max_req_slots,
            self.hp_recent_ring_size,
            self.hp_recent_tokens,
            self.N_Q - 1,
            self.hp_prefix_tokens,
            self.layer_num,
            self.head_num,
            self.head_dim + self.v_head_dim,
            str(self.hp_dtype),
        )

        if self.vq_v_enabled and not self.vq_enabled:
            # VQ-V rides on the K-VQ prefill/decode path (both quant tiers share
            # the vq2 kernel + the vq2 prefill branch), so it requires VQ-K.
            raise ValueError(
                "SGLANG_VQ_V_CODEBOOK_PATH requires SGLANG_VQ_CODEBOOK_PATH "
                "(VQ-V is decoded by the vq2 kernel, which also does VQ-K)."
            )

    # -- Configuration accessors -------------------------------------------

    def mixed_kv_enabled(self) -> bool:
        return True

    @property
    def quant_format(self) -> str:
        """K quant tier format: "int2" (affine) or "vq2" (group-VQ indices).
        The pool's ``dtype`` stays "int2" either way so the existing int2
        gates (allocator, prefill seam, decode dispatch) keep working; format
        divergence is handled at the call sites that touch the K arena."""
        return "vq2" if self.vq_enabled else "int2"

    @property
    def v_quant_format(self) -> str:
        """V quant tier format: "int2" (OSCAR affine, default) or "vq2"
        (group-VQ indices in the R_v basis). Independent of the K tier so the
        INT2-V vs VQ-V ablation is a single-env-var flip."""
        return "vq2" if self.vq_v_enabled else "int2"

    def stash_pending_forward(self, event) -> None:
        """Record the most recent forward-stream completion event.

        Called once per iteration from the scheduler. The event is consumed
        by :meth:`wait_pending_forward` at the apply boundary inside
        ``_alloc_for_decode_mixed``.
        """
        self._pending_forward_done = event

    def wait_pending_forward(self) -> None:
        """Order the current stream after the stashed forward-done event.

        Must be issued *before* the apply phase of the flush
        (``gpu_flush_int2_apply``), since the remap kernel writes
        ``req_to_token`` at positions the previous forward's attention is
        concurrently reading. Pre-apply work (allocator free, plan kernel)
        runs ahead of this wait so its host syncs don't block on the
        previous forward.
        """
        if self._pending_forward_done is None:
            return
        torch.cuda.current_stream().wait_event(self._pending_forward_done)
        self._pending_forward_done = None

    @property
    def hp_global_offset(self) -> int:
        return self._hp_offset

    @property
    def hp_size(self) -> int:
        return (
            self.num_hp_prefix_slots
            + self.max_req_slots * self.hp_recent_ring_size
        )

    @property
    def quant_size(self) -> int:
        return self.num_quant_pages * self.N_Q

    @property
    def hp_prefix_pool_slots(self) -> int:
        return self.num_hp_prefix_slots

    @property
    def hp_recent_base(self) -> int:
        """First HP-buffer index reserved for per-req recent slabs."""
        return self._hp_recent_base

    def release_req_slab(self, req_pool_idx) -> None:
        # Reset the per-req HP-recent cursor and flush counter so the next
        # request taking over ``req_pool_idx`` starts clean.
        if isinstance(req_pool_idx, torch.Tensor):
            idx = req_pool_idx.to(self._next_slab_offset.device).to(torch.int64)
            if idx.numel() == 0:
                return
            self._next_slab_offset[idx] = 0
            self._flush_counter[idx] = 0
        else:
            i = int(req_pool_idx)
            self._next_slab_offset[i] = 0
            self._flush_counter[i] = 0

    def _resolve_quant_grouping(self, head_dim: int, tensor_name: str) -> tuple[int, int]:
        group_size = (
            head_dim
            if self.kv_cache_quant_group_size is None
            else self.kv_cache_quant_group_size
        )
        if group_size <= 0:
            raise ValueError(
                f"{tensor_name} kv_cache_quant_group_size must be positive, got {group_size}"
            )
        if head_dim % group_size != 0:
            raise ValueError(
                f"{tensor_name} head_dim ({head_dim}) must be divisible by "
                f"kv_cache_quant_group_size ({group_size})"
            )
        return group_size, head_dim // group_size

    # -- Arena construction ------------------------------------------------

    def _create_arenas(self):
        # HP arena layout: [shared prefix pool] [per-req recent slab 0]
        # [per-req recent slab 1] ... Quant arena is paged with N_Q slots
        # per page; scales/zeros are quant-only.
        hp_total_slots = (
            self.num_hp_prefix_slots
            + self.max_req_slots * self.hp_recent_ring_size
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                if self.vq_enabled:
                    # K quant tier = VQ indices. One stacked arena with
                    # per-layer views (so the flush-time VQ encode gathers /
                    # scatters all layers in single index ops), plus one
                    # trailing "trash" row at index ``quant_size`` that
                    # shape-static writes divert HP-tier / invalid rows to.
                    ng = self._vq.num_groups
                    # VQ group indices are in [0, codebook_size); with K<=256
                    # (the G=4/bpc=1 arm) they fit in uint8 -> 2.0
                    # b/coord of index memory. Wider codebooks (this project's
                    # G=8/K=1024 CQ/TaSQ bundles, 10-bit indices) need int16 --
                    # int8 would silently wrap, not error.
                    # NG=16 bundles store the row as a 10-bit bitstream instead: 1.25 b/coord,
                    # five int32 words. The decode kernel unpacks in-register; the prefill
                    # prefix dequantise uses vq_unpack_idx. Gated on the group count so K and V
                    # always agree -- see vq_codebook's packing note.
                    self._vq_k_idx_big = torch.zeros(
                        (
                            self.layer_num,
                            self.num_quant_pages * self.N_Q + 1,
                            self.head_num,
                            vq_pack_words(self._vq.codebook_size, ng)
                            if vq_idx_is_packed(ng) else ng,
                        ),
                        dtype=(torch.int32 if vq_idx_is_packed(ng)
                               else self._vq.idx_dtype),
                        device=self.device,
                    )
                    self.k_buffer = [
                        self._vq_k_idx_big[l] for l in range(self.layer_num)
                    ]
                else:
                    self.k_buffer = [
                        torch.zeros(
                            (self.num_quant_pages * self.N_Q, self.head_num, self.head_dim // 4),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                if self.vq_v_enabled:
                    # V quant tier = VQ indices (uint8), stacked per-layer with a
                    # trailing trash row, mirroring the K vq2 arena.
                    ng_v = self._vq_v.num_groups
                    self._vq_v_idx_big = torch.zeros(
                        (
                            self.layer_num,
                            self.num_quant_pages * self.N_Q + 1,
                            self.head_num,
                            vq_pack_words(self._vq_v.codebook_size, ng_v)
                            if vq_idx_is_packed(ng_v) else ng_v,
                        ),
                        dtype=(torch.int32 if vq_idx_is_packed(ng_v)
                               else self._vq_v.idx_dtype),
                        device=self.device,
                    )
                    self.v_buffer = [
                        self._vq_v_idx_big[l] for l in range(self.layer_num)
                    ]
                else:
                    self.v_buffer = [
                        torch.zeros(
                            (
                                self.num_quant_pages * self.N_Q,
                                self.head_num,
                                self.v_head_dim // self._v_pack_den,
                            ),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                if self.vq_enabled:
                    # Slot 0 holds the per-token ptn RMS scale; slot 1 stayed 0 only to keep the
                    # int2 [*, H, 2] stride. Wide-G bundles drop it: their decode launcher
                    # accepts a 1-slot arena, and the G=4 vq2 launcher -- which asserts 2 --
                    # is never reached for them. +1 trash row, matching k_buffer.
                    self._vq_k_sz_big = torch.zeros(
                        (
                            self.layer_num,
                            self.num_quant_pages * self.N_Q + 1,
                            self.head_num,
                            1 if self._vq.wide_g else 2,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    self.k_scales_zeros = [
                        self._vq_k_sz_big[l] for l in range(self.layer_num)
                    ]
                else:
                    self.k_scales_zeros = [
                        torch.zeros(
                            (self.num_quant_pages * self.N_Q, self.head_num, 2 * self.k_num_scale_groups),
                            dtype=self.scale_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                if self.vq_v_enabled:
                    # Slot 0 = per-token ptn RMS scale; slot 1 stays 0 (same
                    # [*, H, 2] single-scale layout as int2), +1 trash row.
                    self._vq_v_sz_big = torch.zeros(
                        (
                            self.layer_num,
                            self.num_quant_pages * self.N_Q + 1,
                            self.head_num,
                            1 if self._vq_v.wide_g else 2,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    self.v_scales_zeros = [
                        self._vq_v_sz_big[l] for l in range(self.layer_num)
                    ]
                else:
                    self.v_scales_zeros = [
                        torch.zeros(
                            (self.num_quant_pages * self.N_Q, self.head_num, 2 * self.v_num_scale_groups),
                            dtype=self.scale_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                if self.vq_enabled:
                    # Stacked so the flush-time VQ encode gathers all layers'
                    # HP K rows in one index_select.
                    self._vq_hp_k_big = torch.zeros(
                        (
                            self.layer_num,
                            hp_total_slots,
                            self.head_num,
                            self.head_dim,
                        ),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    self.hp_k_buffer = [
                        self._vq_hp_k_big[l] for l in range(self.layer_num)
                    ]
                else:
                    self.hp_k_buffer = [
                        torch.zeros(
                            (hp_total_slots, self.head_num, self.head_dim),
                            dtype=self.hp_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                if self.vq_v_enabled:
                    # Stacked so the flush-time VQ encode gathers all layers' HP
                    # V rows in one index_select (mirrors _vq_hp_k_big).
                    self._vq_hp_v_big = torch.zeros(
                        (
                            self.layer_num,
                            hp_total_slots,
                            self.head_num,
                            self.v_head_dim,
                        ),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    self.hp_v_buffer = [
                        self._vq_hp_v_big[l] for l in range(self.layer_num)
                    ]
                else:
                    self.hp_v_buffer = [
                        torch.zeros(
                            (hp_total_slots, self.head_num, self.v_head_dim),
                            dtype=self.hp_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        # The ptn-scale arenas are mostly redundant: a pertoken_norm=False bundle stores 1.0
        # everywhere, and a pool_heads_scale bundle stores the same value across all heads.
        # Collapse them to a broadcast view over a small base -- this is also what lets decode
        # elide the scale load and multiply for the const case (see _is_const_one).
        self._vq_k_sz_mode = self._vq_sz_mode(getattr(self, "_vq", None))
        self._vq_v_sz_mode = self._vq_sz_mode(getattr(self, "_vq_v", None))
        if getattr(self, "vq_enabled", False):
            self._vq_compress_sz("_vq_k_sz_big", "k_scales_zeros", self._vq_k_sz_mode)
        else:
            self._vq_k_sz_mode = None
        if getattr(self, "vq_v_enabled", False):
            self._vq_compress_sz("_vq_v_sz_big", "v_scales_zeros", self._vq_v_sz_mode)
        else:
            self._vq_v_sz_mode = None

        # Cached device pointer arrays for the fused decode-flush kernel. The
        # flush kernel loops over layers inside the kernel, so it needs
        # per-layer base pointers as an int64 GPU tensor. Strides are identical
        # across layers (we enforce that below); the flush kernel reads the
        # single set at launch time via tl.constexpr.
        def _base_ptrs(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [t.data_ptr() for t in tensors],
                dtype=torch.int64,
                device=self.device,
            )

        self._flush_hp_k_ptrs = _base_ptrs(self.hp_k_buffer)
        self._flush_hp_v_ptrs = _base_ptrs(self.hp_v_buffer)
        self._flush_quant_k_ptrs = _base_ptrs(self.k_buffer)
        self._flush_quant_v_ptrs = _base_ptrs(self.v_buffer)
        self._flush_k_sz_ptrs = _base_ptrs(self.k_scales_zeros)
        self._flush_v_sz_ptrs = _base_ptrs(self.v_scales_zeros)

        # Strides (elements, not bytes) for each kind of buffer. The arenas are
        # all contiguous, so every layer shares the same strides; assert to be
        # safe.
        def _strides(t: torch.Tensor) -> tuple:
            return (int(t.stride(0)), int(t.stride(1)), int(t.stride(2)))

        hp_k_stride = _strides(self.hp_k_buffer[0])
        hp_v_stride = _strides(self.hp_v_buffer[0])
        q_k_stride = _strides(self.k_buffer[0])
        q_v_stride = _strides(self.v_buffer[0])
        k_sz_stride = _strides(self.k_scales_zeros[0])
        v_sz_stride = _strides(self.v_scales_zeros[0])
        for l in range(self.layer_num):
            assert _strides(self.hp_k_buffer[l]) == hp_k_stride
            assert _strides(self.hp_v_buffer[l]) == hp_v_stride
            assert _strides(self.k_buffer[l]) == q_k_stride
            assert _strides(self.v_buffer[l]) == q_v_stride
            assert _strides(self.k_scales_zeros[l]) == k_sz_stride
            assert _strides(self.v_scales_zeros[l]) == v_sz_stride

        self._flush_hp_k_stride = hp_k_stride
        self._flush_hp_v_stride = hp_v_stride
        self._flush_quant_k_stride = q_k_stride
        self._flush_quant_v_stride = q_v_stride
        self._flush_k_sz_stride = k_sz_stride
        self._flush_v_sz_stride = v_sz_stride

    # -- KVCache interface -------------------------------------------------

    def get_kv_size_bytes(self):
        k = sum(get_tensor_size_bytes(t) for t in self.k_buffer)
        k += sum(get_tensor_size_bytes(s) for s in self.k_scales_zeros)
        k += sum(get_tensor_size_bytes(t) for t in self.hp_k_buffer)
        v = sum(get_tensor_size_bytes(t) for t in self.v_buffer)
        v += sum(get_tensor_size_bytes(s) for s in self.v_scales_zeros)
        v += sum(get_tensor_size_bytes(t) for t in self.hp_v_buffer)
        return k, v

    def _layer_index(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        # Triton backend asks for the quant view in the mixed path; HP view is
        # accessed via ``get_hp_key_buffer``.
        return self.k_buffer[self._layer_index(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_raw_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[self._layer_index(layer_id)]

    def get_raw_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_key_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.k_scales_zeros[self._layer_index(layer_id)]

    def get_value_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.v_scales_zeros[self._layer_index(layer_id)]

    def get_hp_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_k_buffer[self._layer_index(layer_id)]

    def get_hp_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_v_buffer[self._layer_index(layer_id)]

    def get_raw_kv_buffer(self, layer_id: int):
        idx = self._layer_index(layer_id)
        return {
            "k_buffer": self.k_buffer[idx],
            "v_buffer": self.v_buffer[idx],
            "k_scales_zeros": self.k_scales_zeros[idx],
            "v_scales_zeros": self.v_scales_zeros[idx],
            "dtype": "int2",
        }

    def _split_global_locs(self, loc: torch.Tensor):
        loc64 = loc.to(torch.int64)
        hp_mask = loc64 >= self._hp_offset
        quant_loc = loc64[~hp_mask]
        hp_loc_global = loc64[hp_mask] - self._hp_offset
        return quant_loc, hp_loc_global, hp_mask

    def _rotate_kv_inplace(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the per-layer Oscar rotation ``rows @ R`` to HP K/V tiles.

        Returns tensors in ``self.hp_dtype`` ready to be stored or packed.
        ``R_k`` / ``R_v`` are ``[head_dim, head_dim]`` bf16 on the KV device,
        loaded in ``__init__``.
        """
        idx = self._layer_index(layer_id)
        if self.vq_enabled:
            # vq2 storage space for K is the per-head residual
            # r = (k - mean) @ forward; queries are mapped with inverse.T so
            # both tiers' scores share the same (softmax-invariant) -q.mean
            # shift. V stays on the per-layer R_v path.
            if self._vq.identity_map:
                # forward == I and mean == 0: (k - 0) @ I == k. Same bits, one kernel fewer
                # per layer per step. See VQCodebook.identity_map.
                k_hp = cache_k.to(self.hp_dtype)
            else:
                k_hp = vq_map_k(
                    cache_k, self._vq.forward[idx], self._vq.mean[idx]
                ).to(self.hp_dtype)
        else:
            k_hp = (
                cache_k.to(self.hp_dtype)
                if self._R_k_identity
                else cache_k.to(self.hp_dtype) @ self._R_k[idx]
            )
        if v_rotation_absorbed:
            v_hp = cache_v.to(self.hp_dtype)
        else:
            v_hp = (
                cache_v.to(self.hp_dtype)
                if self._R_v_identity
                else cache_v.to(self.hp_dtype) @ self._R_v[idx]
            )
        return k_hp, v_hp

    def _prepare_hp_kv_tensors(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_rotated: bool,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the Oscar rotation to HP K/V and cast to ``hp_dtype``.
        ``already_rotated`` skips the rotation pre-pass.
        """
        if already_rotated:
            return cache_k.to(self.hp_dtype), cache_v.to(self.hp_dtype)
        return self._rotate_kv_inplace(
            layer_id, cache_k, cache_v, v_rotation_absorbed
        )

    def _set_hp_kv_buffer(
        self,
        layer_id: int,
        hp_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)
        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.hp_k_buffer[idx],
            self.hp_v_buffer[idx],
            hp_loc,
            row_dim=self.row_dim,
            store_dtype=self.hp_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def _set_quant_kv_buffer_extend(
        self,
        layer_id: int,
        quant_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_hadamard_transformed: bool,
        mixed_hp_offset: Optional[int] = None,
        v_rotation_absorbed: bool = False,
    ):
        """Prefill/extend-only: rotate (oscar R) + optional per-row clip +
        int2-pack + write quant slots.

        Decode-time flushes go through the dedicated GPU flush kernel
        (see ``gpu_flush_int2``); this method is *not* used for those.
        """
        idx = self._layer_index(layer_id)
        clip_on = self._k_clip_ratio > 0.0 or self._v_clip_ratio > 0.0

        if self.vq_enabled:
            if not already_hadamard_transformed:
                cache_k, cache_v = self._rotate_kv_inplace(
                    layer_id, cache_k, cache_v, v_rotation_absorbed
                )
            else:
                cache_k = cache_k.to(self.hp_dtype)
                cache_v = cache_v.to(self.hp_dtype)
            if self.vq_v_enabled:
                # V uses group-VQ (the VQ-V ablation); cache_v is already in R_v space.
                self._vq_write_v(idx, quant_loc, cache_v, hp_offset=mixed_hp_offset)
            else:
                # V keeps the int2 single-scale clip path unchanged.
                _launch_single_clip_int2(
                    cache_v,
                    quant_loc,
                    self.v_buffer[idx],
                    self.v_scales_zeros[idx],
                    self._v_clip_ratio,
                    hp_global_offset=mixed_hp_offset,
                    lloyd_max=self._lloyd_max,
                    max_q=self._v_int_max_q,
                    pack_bits=self._v_int_bits,
                )
            self._vq_write_k(idx, quant_loc, cache_k, hp_offset=mixed_hp_offset)
            return

        # Fused rotate(K) + clip(KV) + quantize(KV) + set(KV). Skips the
        # standalone ``K @ R_k`` GEMM and its bf16 staging tensor by doing
        # the rotation inside the int2 pack kernel via ``tl.dot``. V must
        # already be in R_v space (rotation absorbed) — the kernel does
        # not rotate V. Requires single-scale layout (num_groups == 1) for
        # both K and V scales/zeros.
        if envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.get():
            assert v_rotation_absorbed, (
                "V rotation must be absorbed for fused oscar K-rotation + clip + quant + set"
            )

        use_fused_rotate = (
            envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.get()
            and not already_hadamard_transformed
            and v_rotation_absorbed
            and clip_on
            and _get_num_scale_groups(self.k_scales_zeros[idx]) == 1
            and _get_num_scale_groups(self.v_scales_zeros[idx]) == 1
        )
        if use_fused_rotate:
            quantized_set_kv_int2_oscar_rotate_k_clip_triton(
                cache_k.to(self.hp_dtype),
                cache_v.to(self.hp_dtype),
                self._R_k[idx],
                quant_loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                self._k_clip_ratio,
                self._v_clip_ratio,
                hp_global_offset=mixed_hp_offset,
            )
            return

        if not already_hadamard_transformed:
            cache_k, cache_v = self._rotate_kv_inplace(
                layer_id, cache_k, cache_v, v_rotation_absorbed
            )
        else:
            cache_k = cache_k.to(self.hp_dtype)
            cache_v = cache_v.to(self.hp_dtype)

        if not clip_on:
            quantized_set_kv_int2_pretransformed_triton(
                cache_k,
                cache_v,
                quant_loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                hp_global_offset=mixed_hp_offset,
            )
            return

        quantized_set_kv_int2_pretransformed_clip_triton(
            cache_k,
            cache_v,
            quant_loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scales_zeros[idx],
            self.v_scales_zeros[idx],
            self._k_clip_ratio,
            self._v_clip_ratio,
            hp_global_offset=mixed_hp_offset,
            lloyd_max=self._lloyd_max,
            v_max_q=self._v_int_max_q,
        )

    @staticmethod
    def _vq_sz_mode(vq):
        """How compressible this bundle's ptn-scale arena is.

        "const"    pertoken_norm=False -- the scale is 1.0 for every row, so the arena is one
                   broadcast value. Decode then elides the load and the multiply entirely
                   (the kernel detects the stride-0 view; see _is_const_one).
        "headpool" pool_heads_scale -- one scale per (layer, token) shared by every head, so
                   the head axis is a broadcast.
        None       genuine per-(token, head) scales; store them.
        """
        if vq is None:
            return None
        if not getattr(vq, "pertoken_norm", False):
            return "const"
        if getattr(vq, "pool_heads_scale", False):
            return "headpool"
        return None

    def _vq_compress_sz(self, big_attr, list_attr, mode):
        """Replace a materialised scale arena with a broadcast view over a small base.

        The expanded view is what readers (and the decode launcher) see; writes go to
        ``<big_attr>_phys``. Keeping the base alive matters -- an expanded view does not own
        its storage.
        """
        old = getattr(self, big_attr, None)
        if old is None or mode is None:
            return
        L, slots, H, n_last = old.shape
        setattr(self, big_attr, None)
        setattr(self, list_attr, None)
        del old
        base = (torch.ones((L, 1, 1, n_last), dtype=self.scale_dtype, device=self.device)
                if mode == "const" else
                torch.zeros((L, slots, 1, n_last), dtype=self.scale_dtype, device=self.device))
        exp = base.expand(L, slots, H, n_last)
        setattr(self, big_attr + "_phys", base)
        setattr(self, big_attr, exp)
        setattr(self, list_attr, [exp[l] for l in range(L)])

    def _vq_store_scale(self, big_attr, mode, dim, loc, scale, layer_idx=None):
        """Write per-token scales honouring the arena's compression.

        const:    nothing to write -- every row of the arena is already 1.0.
        headpool: the value is identical across heads, so only the head-0 slice is stored.
        """
        if mode == "const":
            return
        tgt = getattr(self, big_attr + "_phys" if mode == "headpool" else big_attr)
        if layer_idx is not None:
            tgt = tgt[layer_idx]
        n_last = tgt.shape[-1]
        if mode == "headpool":
            # `scale` is [T, H] on the prefill path and [L, n, H] on the flush path, so the
            # head axis is the LAST one -- the slot axis only appears on the arena side, which
            # is why the target is [.., 1, n_last] while the value is [.., 1].
            scale = scale.narrow(-1, 0, 1)
        sz = torch.zeros((*scale.shape, n_last), dtype=self.scale_dtype, device=scale.device)
        sz[..., 0] = scale.to(self.scale_dtype)
        tgt.index_copy_(dim, loc, sz)

    @staticmethod
    def _vq_idx_for_store(idxs: torch.Tensor, vq) -> torch.Tensor:
        """Encoder output -> arena representation.

        Keyed on the group count, matching the allocation, so an arena and the rows written
        into it can never disagree about the layout. The previous in-tree attempt keyed the
        two on different things and produced a pool whose K arena was packed and whose V
        arena was not.
        """
        if not vq_idx_is_packed(vq.num_groups):
            return idxs
        return vq_pack_idx(idxs, vq_idx_bits(vq.codebook_size))

    def _vq_write_k(
        self,
        layer_idx: int,
        loc: torch.Tensor,
        r_k: torch.Tensor,
        hp_offset: Optional[int],
    ) -> None:
        """VQ-encode already-mapped K residual rows and scatter into the index
        arena. Rows whose ``loc`` is HP-tier (>= hp_offset) are diverted to the
        trash row so the whole write stays shape-static (no host sync)."""
        if r_k.shape[0] == 0:
            return
        trash = self.quant_size
        loc64 = loc.to(torch.int64)
        if hp_offset is not None:
            loc_safe = torch.where(
                loc64 < int(hp_offset), loc64, torch.full_like(loc64, trash)
            )
        else:
            loc_safe = loc64
        vq = self._vq
        _enc = vq_encode_single if envs.SGLANG_VQ_OPT_PREFILL.get() else vq_encode
        idxs, scale = _enc(
            r_k,
            vq.cb16[layer_idx],
            vq.cb_sq[layer_idx],
            pertoken_norm=vq.pertoken_norm,
            pool_heads=vq.pool_heads_scale,
        )
        self._vq_k_idx_big[layer_idx].index_copy_(
            0, loc_safe, self._vq_idx_for_store(idxs, vq)
        )
        self._vq_store_scale("_vq_k_sz_big", self._vq_k_sz_mode, 0, loc_safe, scale,
                             layer_idx=layer_idx)

    def _vq_write_v(
        self,
        layer_idx: int,
        loc: torch.Tensor,
        r_v: torch.Tensor,
        hp_offset: Optional[int],
    ) -> None:
        """VQ-encode already-R_v-space V rows and scatter into the V index arena.
        Mirrors ``_vq_write_k`` but strided-reorders coords first so the kernel's
        fixed quarter layout lines up (see ``_vq_v_perm``) -- wide-G (G != 4)
        codebooks skip this: the vqwide kernel gathers each group's coords
        contiguously, with no quarter layout to line up against."""
        if r_v.shape[0] == 0:
            return
        trash = self.quant_size
        loc64 = loc.to(torch.int64)
        if hp_offset is not None:
            loc_safe = torch.where(
                loc64 < int(hp_offset), loc64, torch.full_like(loc64, trash)
            )
        else:
            loc_safe = loc64
        vq = self._vq_v
        r_v_in = r_v if vq.wide_g else r_v[..., self._vq_v_perm]
        _enc = vq_encode_single if envs.SGLANG_VQ_OPT_PREFILL.get() else vq_encode
        idxs, scale = _enc(
            r_v_in,
            vq.cb16[layer_idx],
            vq.cb_sq[layer_idx],
            pertoken_norm=vq.pertoken_norm,
            pool_heads=vq.pool_heads_scale,
        )
        self._vq_v_idx_big[layer_idx].index_copy_(
            0, loc_safe, self._vq_idx_for_store(idxs, vq)
        )
        self._vq_store_scale("_vq_v_sz_big", self._vq_v_sz_mode, 0, loc_safe, scale,
                             layer_idx=layer_idx)

    def vq_flush_k(self, plan) -> None:
        """Decode-flush companion of the fused int2 flush: VQ-encode the K
        rows being demoted (all layers at once) and scatter indices + ptn
        scales into the quant arenas. The int2 flush kernel handles V and the
        req_to_token remap; it is launched with ``k_vq=True`` so its K pack is
        skipped. All ops are shape-static (invalid rows route to the trash
        row), so this is safe in the eager scheduler path with no host sync.
        """
        valid = plan.valid_mask.to(torch.bool)
        src = plan.src_hp_slot.clamp_min(0)
        trash = self.quant_size
        dst = torch.where(
            valid,
            plan.dst_quant_slots,
            torch.full_like(plan.dst_quant_slots, trash),
        )
        r = self._vq_hp_k_big.index_select(1, src)  # [L, n, H, D]
        L, n, H, D = r.shape
        vq = self._vq
        if envs.SGLANG_VQ_OPT_FLUSH.get():
            from sglang.srt.mem_cache.vq_codebook import vq_encode_fused

            idxs, scales = vq_encode_fused(
                r, vq.cb16, vq.cb_sq, pertoken_norm=vq.pertoken_norm,
                pool_heads=vq.pool_heads_scale, valid=valid,
            )
            self._vq_k_idx_big.index_copy_(
                1, dst, self._vq_idx_for_store(idxs, vq)
            )
            self._vq_store_scale("_vq_k_sz_big", self._vq_k_sz_mode, 1, dst, scales)
            return

        idxs = torch.empty(
            (L, n, H, vq.num_groups), dtype=vq.idx_dtype, device=r.device
        )
        scales = torch.empty((L, n, H), dtype=torch.float32, device=r.device)
        # Chunk over the (small) flush-token axis to bound the [L, c, H, NG, K]
        # score tensor (~76 MB fp32 at c=8 for Qwen3-8B).
        for t0 in range(0, n, 8):
            t1 = min(t0 + 8, n)
            rf = r[:, t0:t1].to(torch.float32)
            if vq.pool_heads_scale:
                sc = rf.pow(2).mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1e-8)
                sc = sc.expand(L, t1 - t0, H, 1)
                rn = (rf / sc).to(torch.float16)
                scales[:, t0:t1] = sc.squeeze(-1)
            elif vq.pertoken_norm:
                sc = rf.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
                rn = (rf / sc).to(torch.float16)
                scales[:, t0:t1] = sc.squeeze(-1)
            else:
                rn = rf.to(torch.float16)
                scales[:, t0:t1] = 1.0
            rn = rn.view(L, t1 - t0, H, vq.num_groups, vq.group_dim)
            sco = torch.einsum("lnhgc,lhgkc->lnhgk", rn, vq.cb16).to(
                torch.float32
            )
            sco -= vq.cb_sq.unsqueeze(1)
            idxs[:, t0:t1] = sco.argmax(-1).to(vq.idx_dtype)
        self._vq_k_idx_big.index_copy_(
                1, dst, self._vq_idx_for_store(idxs, vq)
            )
        self._vq_store_scale("_vq_k_sz_big", self._vq_k_sz_mode, 1, dst, scales)


    def vq_flush_v(self, plan) -> None:
        """VQ-V companion of vq_flush_k: VQ-encode the demoted V rows (all layers)
        and scatter into the V index arena. The flush kernel is launched with
        ``v_vq=True`` so its V int2 pack is skipped. V rows are strided-reordered
        (``_vq_v_perm``) so the contiguous .view(NG,G) matches the kernel layout."""
        valid = plan.valid_mask.to(torch.bool)
        src = plan.src_hp_slot.clamp_min(0)
        trash = self.quant_size
        dst = torch.where(
            valid,
            plan.dst_quant_slots,
            torch.full_like(plan.dst_quant_slots, trash),
        )
        r = self._vq_hp_v_big.index_select(1, src)  # [L, n, H, D]
        vq = self._vq_v
        if not vq.wide_g:
            r = r[..., self._vq_v_perm]              # strided -> contiguous groups
        L, n, H, D = r.shape
        if envs.SGLANG_VQ_OPT_FLUSH.get():
            # Same fused nearest-centroid kernel as vq_flush_k (this V path was
            # still on the unfused einsum+argmax torch loop -- a [L, 8, H, NG, K]
            # fp32 score tensor per chunk EVERY decode step; at the g8 geometry
            # that was a top-5 profile entry). valid= skips the trash-routed rows
            # the shape-static plan carries on most steps.
            from sglang.srt.mem_cache.vq_codebook import vq_encode_fused

            idxs, scales = vq_encode_fused(
                r, vq.cb16, vq.cb_sq, pertoken_norm=vq.pertoken_norm,
                pool_heads=getattr(vq, "pool_heads_scale", False), valid=valid,
            )
            self._vq_v_idx_big.index_copy_(
                1, dst, self._vq_idx_for_store(idxs, vq)
            )
            self._vq_store_scale("_vq_v_sz_big", self._vq_v_sz_mode, 1, dst, scales)
            return
        idxs = torch.empty(
            (L, n, H, vq.num_groups), dtype=vq.idx_dtype, device=r.device
        )
        scales = torch.empty((L, n, H), dtype=torch.float32, device=r.device)
        for t0 in range(0, n, 8):
            t1 = min(t0 + 8, n)
            rf = r[:, t0:t1].to(torch.float32)
            if vq.pertoken_norm:
                sc = rf.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
                rn = (rf / sc).to(torch.float16)
                scales[:, t0:t1] = sc.squeeze(-1)
            else:
                rn = rf.to(torch.float16)
                scales[:, t0:t1] = 1.0
            rn = rn.view(L, t1 - t0, H, vq.num_groups, vq.group_dim)
            sco = torch.einsum("lnhgc,lhgkc->lnhgk", rn, vq.cb16).to(torch.float32)
            sco -= vq.cb_sq.unsqueeze(1)
            idxs[:, t0:t1] = sco.argmax(-1).to(vq.idx_dtype)
        self._vq_v_idx_big.index_copy_(
                1, dst, self._vq_idx_for_store(idxs, vq)
            )
        self._vq_store_scale("_vq_v_sz_big", self._vq_v_sz_mode, 1, dst, scales)

    def _set_mixed_hp_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)

        def _launch(src: torch.Tensor, dst: torch.Tensor):
            src2 = src.reshape(src.shape[0], -1)
            dst2 = dst.reshape(dst.shape[0], -1)
            row_dim = src2.shape[1]
            if row_dim == 0 or src2.shape[0] == 0:
                return
            block_row = min(1024, triton.next_power_of_2(row_dim))
            grid = (src2.shape[0], triton.cdiv(row_dim, block_row))
            _set_mixed_hp_buffer_kernel[grid](
                src2,
                dst2,
                loc,
                src2.shape[0],
                row_dim,
                src2.stride(0),
                src2.stride(1),
                dst2.stride(0),
                dst2.stride(1),
                HP_OFFSET=int(self._hp_offset),
                BLOCK_ROW=block_row,
                num_warps=4,
                num_stages=1,
            )

        _launch(cache_k, self.hp_k_buffer[idx])
        _launch(cache_v, self.hp_v_buffer[idx])

    # ---- Exact chunked-prefill shadow -------------------------------------

    def shadow_active(self) -> bool:
        # Skip under graph capture: shadow buffers change identity every
        # chunk, so a captured gather would bake a stale pointer. Falls back
        # to plain dequantization there (the pre-fix behavior).
        return (
            self.shadow_enabled
            and self._shadow_k is not None
            and not torch.cuda.is_current_stream_capturing()
        )

    def shadow_register(self, slots: torch.Tensor, rid) -> None:
        """Mark ``slots`` (quant tier) as belonging to a request that is mid
        chunked prefill: their exact rows are captured at write time and
        preferred over dequantization until ``shadow_mark_release(rid)`` +
        the next ``shadow_step_release`` drop them."""
        n = int(slots.numel())
        if n == 0:
            return
        slots64 = slots.to(torch.int64)
        s0 = self._shadow_next_row
        self._shadow_map[slots64] = torch.arange(
            s0, s0 + n, dtype=torch.int32, device=slots64.device
        )

        def _grow(buf: Optional[torch.Tensor], dim: int) -> torch.Tensor:
            new = torch.empty(
                (self.layer_num, s0 + n + 1, self.head_num, dim),
                dtype=self.hp_dtype,
                device=self.device,
            )
            if buf is not None and s0 > 0:
                new[:, :s0] = buf[:, :s0]
            return new

        self._shadow_k = _grow(self._shadow_k, self.head_dim)
        self._shadow_v = _grow(self._shadow_v, self.v_head_dim)
        self._shadow_next_row = s0 + n
        self._shadow_rid_slots.setdefault(rid, []).append(slots64)

    def shadow_mark_release(self, rid) -> None:
        """Defer the actual release to the next alloc step: the marking batch
        is the request's FINAL chunk, whose forward pass still reads the
        shadow. Any subsequent alloc implies that forward completed."""
        if rid in self._shadow_rid_slots:
            self._shadow_release_rids.add(rid)

    def shadow_step_release(self, batch_rids=None) -> None:
        """Alloc-time hook: drop shadows whose owners finished their final
        chunk. ``batch_rids`` (extend batches only) additionally drops owners
        absent from the batch -- an aborted chunked request never marks
        release, and the scheduler always re-includes a live chunked request
        in the next prefill batch."""
        if not self._shadow_rid_slots:
            self._shadow_release_rids.clear()
            return
        to_release = set(self._shadow_release_rids)
        if batch_rids is not None:
            to_release.update(
                rid for rid in self._shadow_rid_slots if rid not in batch_rids
            )
        if not to_release:
            return
        for rid in to_release:
            for s in self._shadow_rid_slots.pop(rid, []):
                self._shadow_map[s] = -1
            self._shadow_release_rids.discard(rid)
        if not self._shadow_rid_slots:
            # Rows of partially-released owners leak until this wholesale
            # reset; bounded by one in-flight prompt since at most one
            # request is mid-chunk at a time.
            self._shadow_k = None
            self._shadow_v = None
            self._shadow_next_row = 0

    def _shadow_write(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k_hp: torch.Tensor,
        cache_v_hp: torch.Tensor,
    ) -> None:
        """Mirror stored-space rows of registered quant slots into the shadow.
        Shape-static: HP-tier and unregistered locs route to the trailing
        trash row (duplicate trash indices are fine -- same pattern as the
        VQ index scatter)."""
        idx = self._layer_index(layer_id)
        loc64 = loc.to(torch.int64)
        trash_slot = self.quant_size
        q_safe = torch.where(
            loc64 < trash_slot, loc64, torch.full_like(loc64, trash_slot)
        )
        rows = self._shadow_map[q_safe].to(torch.int64)
        s_trash = self._shadow_k.shape[1] - 1
        rows_safe = torch.where(rows >= 0, rows, torch.full_like(rows, s_trash))
        self._shadow_k[idx].index_copy_(0, rows_safe, cache_k_hp.to(self.hp_dtype))
        self._shadow_v[idx].index_copy_(0, rows_safe, cache_v_hp.to(self.hp_dtype))

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        already_hadamard_transformed: bool = False,
        is_decode: bool = False,
    ):
        """Write K/V to the unified pool.

        ``is_decode`` selects the path:
            * ``True`` -- single-token decode write. Caller fills ``loc`` with
              valid HP slot ids (from ``allocator.alloc_hp_recent``); we
              write only the HP buffer, with no boolean masking (safe under
              CUDA-graph capture).
            * ``False`` (extend / prefill) -- mixed write: int2 quant slots get
              the rotated+clipped pack, HP slots get the bf16 row.

        Callers must pass ``is_decode`` based on
        ``forward_batch.forward_mode.is_decode_or_idle()``. Capture state is
        *not* a reliable proxy: piecewise CUDA graph captures parts of prefill
        (would mis-route to the HP-only branch) and ``--disable-cuda-graph``
        runs decode eagerly (would mis-route to the quant+HP branch).
        """
        if loc.numel() == 0:
            return

        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        # SGLANG_DEBUG_KV_WRITE_SUM=1: checksum every extend write's INPUTS at
        # the pool boundary. Two servers fed identical requests must print
        # identical lines if their prefill paths hand the pool the same data;
        # a diff pinpoints the first divergent (layer, tensor). Debug aid for
        # the fa3-vs-triton quant prefill divergence; remove when resolved.
        import os as _os
        if _os.environ.get("SGLANG_DEBUG_KV_WRITE_SUM") == "1" and not is_decode:
            print(f"[kvw] L{layer_id} n={loc.numel()} aht={already_hadamard_transformed} "
                  f"k={cache_k.float().sum().item():.6e} v={cache_v.float().sum().item():.6e} "
                  f"loc={int(loc.to(torch.int64).sum().item())}", flush=True)
        v_rotation_absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))

        if is_decode:
            hp_local = loc.to(torch.int64) - self._hp_offset
            cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
                layer_id,
                cache_k,
                cache_v,
                already_hadamard_transformed,
                v_rotation_absorbed,
            )
            self._set_hp_kv_buffer(layer_id, hp_local, cache_k_hp, cache_v_hp)
            return

        self._set_quant_kv_buffer_extend(
            layer_id,
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            mixed_hp_offset=int(self._hp_offset),
            v_rotation_absorbed=v_rotation_absorbed,
        )
        cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
            layer_id,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            v_rotation_absorbed,
        )
        self._set_mixed_hp_kv_buffer(layer_id, loc, cache_k_hp, cache_v_hp)
        if self.shadow_active():
            # Exact chunked-prefill: the rows just computed for the HP write
            # are the stored-space truth for EVERY tier; capture the ones
            # landing in registered quant slots.
            self._shadow_write(layer_id, loc, cache_k_hp, cache_v_hp)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        # Moves only make sense within the same tier. The allocator ensures
        # that; callers should split by tier before calling here.
        tgt_q, tgt_hp, tgt_mask = self._split_global_locs(tgt_loc)
        src_q, src_hp, src_mask = self._split_global_locs(src_loc)
        assert torch.equal(tgt_mask, src_mask), (
            "move_kv_cache requires src/tgt tiers to match"
        )
        for l in range(self.layer_num):
            if tgt_q.numel() > 0:
                self.k_buffer[l][tgt_q] = self.k_buffer[l][src_q]
                self.v_buffer[l][tgt_q] = self.v_buffer[l][src_q]
                self.k_scales_zeros[l][tgt_q] = self.k_scales_zeros[l][src_q]
                self.v_scales_zeros[l][tgt_q] = self.v_scales_zeros[l][src_q]
            if tgt_hp.numel() > 0:
                self.hp_k_buffer[l][tgt_hp] = self.hp_k_buffer[l][src_hp]
                self.hp_v_buffer[l][tgt_hp] = self.hp_v_buffer[l][src_hp]

    def get_cpu_copy(self, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")
