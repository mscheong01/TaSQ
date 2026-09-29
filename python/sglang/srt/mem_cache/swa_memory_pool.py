import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils import is_npu

_is_npu = is_npu()

if _is_npu:
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


class SWAKVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        token_to_kv_pool_class: KVCache = MHATokenToKVPool,
        full_dtype=None,
        full_kv_pool_override: Optional[KVCache] = None,
        **kwargs,
    ):
        # ``full_dtype`` lets the full-attention inner pool store a different
        # (quantized, e.g. "int2") dtype while SWA layers stay in the model
        # dtype: sliding-window layers hold at most window_size tokens, so
        # quantizing them buys nothing and the quant decode kernels have no
        # windowed variant.
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.full_dtype = full_dtype if full_dtype is not None else dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.start_layer = 0
        self.page_size = page_size
        self.swa_loc = None

        kwargs["page_size"] = page_size
        kwargs["enable_memory_saver"] = False
        kwargs["head_num"] = head_num
        kwargs["head_dim"] = head_dim
        kwargs["device"] = device
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        self.swa_kv_pool = token_to_kv_pool_class(
            size=size_swa,
            dtype=dtype,
            layer_num=self.swa_layer_nums,
            **kwargs,
        )
        # set_kv_buffer below forwards a POOL-LOCAL layer id to each inner pool, so the
        # SWA and full-attention pools both count 0..N-1. Anything keyed on that id (the
        # sim-quant bundle) would otherwise apply one set of per-layer constants to two
        # different layers. Mark the SWA pool so it can opt out; the full pool is the one
        # the oscar_int2/vq2 arms quantize.
        self.swa_kv_pool._is_swa_inner_pool = True
        kwargs.pop("swa_head_num", None)
        kwargs.pop("swa_head_dim", None)
        kwargs.pop("swa_v_head_dim", None)
        if full_kv_pool_override is not None:
            # The unified mixed HP+int2 pool has its own constructor surface
            # (quant pages, HP windows, ...), so the model runner builds it
            # and hands the instance in rather than going through the shared
            # (size, dtype, layer_num) construction below.
            self.full_kv_pool = full_kv_pool_override
        else:
            self.full_kv_pool = token_to_kv_pool_class(
                size=size,
                dtype=self.full_dtype,
                layer_num=self.full_layer_nums,
                **kwargs,
            )
        # {layer_id: (index, is_swa_layer)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            f"SWAKVPool mem usage: {self.mem_usage:.2f} GB, swa size: {self.size_swa}, full size: {self.size}"
        )

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    # --- Mixed HP+int2 delegation (full pool = UnifiedInt2HPKVPool) -------
    # Callers gate on ``mixed_kv_enabled()`` before touching the hp_*
    # attributes, so the AttributeError on a plain full pool is unreachable.
    def mixed_kv_enabled(self):
        fn = getattr(self.full_kv_pool, "mixed_kv_enabled", None)
        return fn is not None and fn()

    @property
    def hp_prefix_tokens(self):
        return self.full_kv_pool.hp_prefix_tokens

    @property
    def hp_recent_tokens(self):
        return self.full_kv_pool.hp_recent_tokens

    @property
    def hp_global_offset(self):
        return self.full_kv_pool.hp_global_offset

    @property
    def flush_interval(self):
        return self.full_kv_pool.flush_interval

    @property
    def N_Q(self):
        return self.full_kv_pool.N_Q

    def release_req_slab(self, req_pool_idx):
        release_slab = getattr(self.full_kv_pool, "release_req_slab", None)
        if release_slab is not None:
            release_slab(req_pool_idx)

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        return (
            full_kv_data_ptrs,
            full_kv_data_lens,
            full_kv_item_lens,
        )

    def get_state_buf_infos(self):
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        return swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens

    def get_key_buffer(self, layer_id: int):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def set_swa_loc(self, loc: torch.Tensor):
        self.swa_loc = loc

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None

        # Note: kv_indices could have -1 values (from alloc_extend), which will be mapped to -1
        # since the last item of full_to_swa_index_mapping is -1.
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        already_hadamard_transformed: bool = False,
        is_decode: bool = False,
    ):
        # The OSCAR fork extended the standard pools' set_kv_buffer with the
        # two kwargs above and the attention backends pass them; this wrapper
        # (hybrid-SWA models, e.g. gpt-oss) must accept and forward them or
        # every SWA-model boot dies on TypeError.
        layer_id = layer.layer_id
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            if self.swa_loc is not None:
                loc = self.swa_loc
            else:
                if self.full_to_swa_index_mapping is not None:
                    loc = self.translate_loc_from_full_to_swa(loc)

            self.swa_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
                already_hadamard_transformed=already_hadamard_transformed,
                is_decode=is_decode,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
                already_hadamard_transformed=already_hadamard_transformed,
                is_decode=is_decode,
            )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)
        tgt_loc_swa = self.translate_loc_from_full_to_swa(tgt_loc)
        src_loc_swa = self.translate_loc_from_full_to_swa(src_loc)
        self.swa_kv_pool.move_kv_cache(tgt_loc_swa, src_loc_swa)

    def get_cpu_copy(self, indices):
        # For SWA, we need to copy KV cache from both full and SWA pools
        # The indices are for the full pool, and we use mapping to get SWA indices
        full_kv_cpu = self.full_kv_pool.get_cpu_copy(indices)

        # Get SWA indices through the mapping
        # Note: SWA allocation always creates 1:1 mapping, so no need to filter
        if self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            swa_kv_cpu = self.swa_kv_pool.get_cpu_copy(swa_indices)
        else:
            swa_kv_cpu = None

        return {"full": full_kv_cpu, "swa": swa_kv_cpu}

    def load_cpu_copy(self, kv_cache_cpu, indices):
        # Load KV cache back from CPU to both full and SWA pools
        # Note: indices here are NEW indices (newly allocated), different from get_cpu_copy indices
        full_kv_cpu = kv_cache_cpu["full"]
        swa_kv_cpu = kv_cache_cpu["swa"]

        # Load full KV cache to the new indices
        self.full_kv_pool.load_cpu_copy(full_kv_cpu, indices)

        # Load SWA KV cache if it exists
        if swa_kv_cpu is not None and self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            self.swa_kv_pool.load_cpu_copy(swa_kv_cpu, swa_indices)


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: SWAKVPool,
        need_sort: bool,
    ):
        assert isinstance(kvcache, SWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.page_size = page_size

        full_pool_is_mixed = (
            getattr(kvcache.full_kv_pool, "mixed_kv_enabled", None) is not None
            and kvcache.full_kv_pool.mixed_kv_enabled()
        )
        if full_pool_is_mixed:
            # Full side: the unified paged-quant + HP allocator over the
            # mixed pool. SWA side: a plain page-1 allocator — SWA slot ids
            # only ever live in ``full_to_swa_index_mapping`` and the SWA
            # buffers, so nothing requires them to be N_Q-page aligned even
            # though the scheduler-facing page_size is N_Q.
            from sglang.srt.mem_cache.unified_kv_allocator import (
                UnifiedInt2HPKVAllocator,
            )

            fp = kvcache.full_kv_pool
            self.full_attn_allocator = UnifiedInt2HPKVAllocator(
                num_quant_pages=fp.num_quant_pages,
                quant_tokens_per_page=fp.N_Q,
                hp_prefix_tokens=fp.hp_prefix_tokens,
                hp_recent_tokens=fp.hp_recent_tokens,
                hp_recent_ring_size=fp.hp_recent_ring_size,
                max_req_slots=fp.max_req_slots,
                num_hp_prefix_slots=fp.num_hp_prefix_slots,
                dtype=dtype,
                hp_dtype=fp.hp_dtype,
                device=device,
                kvcache=fp,
                need_sort=need_sort,
                scheduler_size=size + fp.num_hp_prefix_slots,
            )
            self.swa_attn_allocator = TokenToKVPoolAllocator(
                size_swa,
                dtype,
                device,
                kvcache.swa_kv_pool,
                need_sort,
            )
        elif page_size == 1:
            self.full_attn_allocator = TokenToKVPoolAllocator(
                size,
                dtype,
                device,
                kvcache.full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = TokenToKVPoolAllocator(
                size_swa,
                dtype,
                device,
                kvcache.swa_kv_pool,
                need_sort,
            )
        else:
            if _is_npu:
                PagedTokenToKVPoolAllocatorClass = NPUPagedTokenToKVPoolAllocator
            else:
                PagedTokenToKVPoolAllocatorClass = PagedTokenToKVPoolAllocator
            self.full_attn_allocator = PagedTokenToKVPoolAllocatorClass(
                size,
                page_size,
                dtype,
                device,
                kvcache.full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = PagedTokenToKVPoolAllocatorClass(
                size_swa,
                page_size,
                dtype,
                device,
                kvcache.swa_kv_pool,
                need_sort,
            )
        # Note: append one more item of value -1 in the end so -1 maps to -1.
        # It is needed for the last_loc in alloc_extend, where the first full_last_loc
        # is -1, and we need to map it to swa_last_loc -1 as well.
        # For a mixed full pool the id space spans quant slots, HP-prefix
        # slots and the per-request HP-recent rings; size the mapping over
        # that whole range so any full slot id can be translated.
        if full_pool_is_mixed:
            fp = kvcache.full_kv_pool
            mapping_len = (
                fp.num_quant_pages * fp.N_Q
                + fp.num_hp_prefix_slots
                + fp.max_req_slots * fp.hp_recent_ring_size
            )
        else:
            mapping_len = size + self.page_size
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    mapping_len,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

        self.need_sort = need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        self.clear()
        self._kvcache = kvcache
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)

    def available_size(self):
        return min(
            self.full_attn_allocator.available_size(),
            self.swa_attn_allocator.available_size(),
        )

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    @property
    def size(self):
        return min(self._size_full, self._size_swa)

    @property
    def size_swa(self):
        return self._size_swa

    @property
    def size_full(self):
        return self._size_full

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self._kvcache.full_to_swa_index_mapping is not None
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)

    def alloc(self, need_size: int):
        assert self.page_size == 1
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        if _is_npu:
            self.full_to_swa_index_mapping[alloc_full_indices.to(torch.int64)] = (
                alloc_swa_indices.to(torch.int64)
            )
        else:
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        assert self.page_size > 1
        num_tokens = extend_num_tokens + len(seq_lens) * self.page_size
        if num_tokens > self.full_attn_allocator.available_size():
            return None
        if num_tokens > self.swa_attn_allocator.available_size():
            return None

        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            swa_last_loc,
            extend_num_tokens,
        )
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        if _is_npu:
            self.full_to_swa_index_mapping[alloc_full_indices.to(torch.int64)] = (
                alloc_swa_indices.to(torch.int64)
            )
        else:
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        assert self.page_size > 1
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, swa_last_loc
        )

        if alloc_full_indices is None or alloc_swa_indices is None:
            return None

        if _is_npu:
            self.full_to_swa_index_mapping[alloc_full_indices.to(torch.int64)] = (
                alloc_swa_indices.to(torch.int64)
            )
        else:
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # NOTE: the API is not idempotent.
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def alloc_swa_for(self, full_index: torch.Tensor):
        """Pair one SWA slot to each already-allocated full slot.

        ``alloc``/``alloc_extend`` allocate *both* tiers together, which is how
        they keep the 1:1 invariant. The mixed-KV path cannot use them: it
        allocates its full slots itself across three tiers (hp-prefix / quant /
        hp-recent), so calling them would double-allocate the full side. This is
        the other half of that contract for such callers -- it establishes the
        same invariant (one SWA slot per full-tier token, recorded in
        ``full_to_swa_index_mapping``) in one place, so no caller has to reach
        into ``swa_attn_allocator`` and risk breaking it from outside.

        Returns None when the SWA tier is exhausted, matching ``alloc``.
        Mirrors ``free_swa``.

        Any SWA slot still mapped to one of these full ids is released FIRST.
        That is what keeps release exactly-once without a second bookkeeper:
        ``free_swa`` may only be called from ``SWARadixCache.evict``, which
        decrements ``swa_evictable_size_`` in the same step -- free it anywhere
        else and the tree keeps counting slots the allocator has already handed
        back, so ``available + evictable`` exceeds capacity. Releasing on reuse
        instead pins the release to a real ownership transfer: a full slot is
        only re-handed-out after whoever owned it (tree via evict, or the
        request itself) has given it up, so the counter is already correct.
        """
        n = full_index.numel()
        if n == 0:
            return full_index.new_empty((0,), dtype=torch.int64)
        stale = self.full_to_swa_index_mapping[full_index]
        stale = stale[stale > 0]
        if stale.numel() > 0:
            self.swa_attn_allocator.free(stale)
        swa_index = self.swa_attn_allocator.alloc(n)
        if swa_index is None:
            # The stale slots above are already back in the free list. Leaving
            # the rows pointing at them would alias whichever tokens get those
            # slots next, and a later sweep over these ids (free_swa at
            # eviction or teardown) would release them a second time.
            self.full_to_swa_index_mapping[full_index] = 0
            return None
        self.full_to_swa_index_mapping[full_index] = swa_index
        return swa_index

    def move_swa(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        """Retarget SWA ownership when a token's full slot changes tier.

        The mixed-KV flush demotes a token from its HP-recent slot to a quant
        slot and rewrites ``req_to_token`` to the new id. Only the HP slot was
        ever paired with an SWA slot (``alloc_swa_for`` on the decode path), so
        without this the token's SWA stays stranded on the vacated HP id while
        the quant id -- the one the radix tree caches and hands to the next
        request -- owns nothing. A tree node built from those ids is counted in
        ``swa_evictable_size_`` but backed by no slot, and ``match_prefix``
        serves it as SWA-valid.

        A pure relabel: no alloc, no net change in occupancy. Any SWA still
        mapped to a destination id is released first, on the same
        release-on-reuse contract as ``alloc_swa_for`` -- quant slots are
        recycled through the full tier's free list, which does not clear the
        mapping.
        """
        if src_index.numel() == 0:
            return
        stale = self.full_to_swa_index_mapping[dst_index]
        stale = stale[stale > 0]
        if stale.numel() > 0:
            self.swa_attn_allocator.free(stale)
        self.full_to_swa_index_mapping[dst_index] = self.full_to_swa_index_mapping[
            src_index
        ]
        self.full_to_swa_index_mapping[src_index] = 0

    def free_swa(self, free_index: torch.Tensor):
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[free_index] = 0

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        self.full_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def clear(self):
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
