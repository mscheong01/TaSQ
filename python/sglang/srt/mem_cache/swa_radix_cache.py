from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the hybrid (full and SWA) KV cache.
"""

import heapq
import os
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
    maybe_bigram_convert,
    mixed_kv_detect,
    mixed_kv_match_cap,
    mixed_kv_slack_insert_limit,
    mixed_kv_tail_to_drop,
    mixed_kv_with_quant_slack,
    page_align_keys,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.utils import convert_to_bigram_key

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

import logging

logger = logging.getLogger(__name__)


class TreeNode:

    counter = 0
    swa_uuid_counter = 1
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        # swa_tombstone is used to indicate the kv indices have been freed for swa layers
        self.swa_tombstone = False
        # invariant: for any node, if swa_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, swa_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= swa_lock_ref.
        self.full_lock_ref = 0
        self.swa_lock_ref = 0
        # Tree-owned irreducible SWA tail (SGLANG_SWA_KEEP_PREFIX_TAIL): the
        # trailing live window a cached prefix needs to stay matchable. Distinct
        # from a request lock: it has no owner and no dec path. Pinned tokens
        # are charged to swa_protected_size_, and the SWA LRU getters skip
        # pinned nodes, so SWA-only eviction cannot select them (the fallback
        # pass and FULL-tier eviction still can -- _delete_leaf handles the
        # accounting either way).
        self.swa_pinned = False
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        # store the host indices of KV cache
        self.host_value = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.swa_prev = None
        self.swa_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
        self.swa_uuid = None

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def gen_swa_uuid() -> int:
    TreeNode.swa_uuid_counter += 1
    return TreeNode.swa_uuid_counter


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, is_swa_list: bool = False):
        self.is_swa_list = is_swa_list
        if self.is_swa_list:
            self.prv = "swa_prev"
            self.nxt = "swa_next"
            self.lock_ref = "swa_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Resetting swa tombstone node in swa lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            # for swa lru list, only reset non-tombstone nodes
            if not self.is_swa_list or not node.swa_tombstone:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Inserting swa tombstone node in swa lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Removing swa tombstone node from swa lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self, ignore_pin: bool = False) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False, ignore_pin=ignore_pin)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True, ignore_pin: bool = False
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        # Pinned nodes (keep-prefix-tail) are not eviction candidates; only the
        # liveness fallback pass walks through them (ignore_pin=True). This
        # also keeps sanity_check_evictable_size consistent with the counter,
        # which excludes pinned tokens.
        while getattr(x, self.lock_ref) > 0 or (
            self.is_swa_list and x.swa_pinned and not ignore_pin
        ):
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += len(node.value)
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: "SWARadixCache"):
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.
        """
        try:
            if self.is_swa_list:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru_plus_1 = len(self.cache) + 1
            # heapify based on last_access_time
            heapq.heapify(nodes)
            # the root node is not in the lru list
            assert (
                len(nodes) == len(self.cache) + 1
            ), f"len(nodes): {len(nodes)} != len(self.cache) + 1: {len(self.cache) + 1}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.is_swa_list=}, x: {x.id=} != x_lru: {x_lru.id=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"
                assert (
                    x_lru.swa_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.swa_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.is_swa_list:
                evictable_size = tree_cache.swa_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.is_swa_list=}, total nodes: {total_nodes}, total lru plus 1: {total_lru_plus_1}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            msg = f"SWA Radix tree sanity check failed, ping @hanming-lu: {e}"
            logger.error(msg)
            raise Exception(msg)


class SWARadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.is_eagle = params.is_eagle

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.is_eagle:
            self.key_convert_fn = convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key

        if params.enable_metrics:
            self.init_metrics_collector()

        self.sliding_window_size = params.sliding_window_size

        # Mixed-KV: tree caches the shared HP-prefix pool + quant slots;
        # per-request HP-recent slots are excluded by ``mixed_kv_tail_to_drop``
        # (correctness -- they alias across requests, see below) and
        # ``match_prefix`` is capped via ``_mixed_kv_match_cap_overhead``.
        # Mirrors RadixCache; this cache silently lacked all of it, which let
        # HP-recent slot ids into the tree. Those ids are deterministic per
        # req_pool_idx (unified_kv_allocator._recent_slab_base_slots), so a
        # finished request's cached node aliased the LIVE KV of whichever
        # request next took that slot -- wrong reads, plus a
        # ``swa_evictable_size_`` over-count when the new owner's
        # ``_evict_swa`` released slots this tree still counted.
        (
            self._mixed_kv_enabled,
            self._mixed_kv_hp_prefix_tokens,
            self._mixed_kv_match_cap_overhead,
        ) = mixed_kv_detect(self.token_to_kv_pool_allocator)
        # Read once: this is consulted per eviction candidate.
        self._swa_keep_prefix_tail = envs.SGLANG_SWA_KEEP_PREFIX_TAIL.get()
        # A protected tail must survive the match, not just exist: the match key
        # ends up to 2 pages before the insert end (init_next_round_input drops
        # the last token, both ends are page-floored, and the mixed-KV match cap
        # and insert-side tail drop cancel), and a match ending inside the tail
        # SPLITS it -- so keeping exactly ``sliding_window_size`` leaves the exit
        # checkpoint short and the prefix still dies. Keep two extra pages.
        self._swa_keep_target = (
            (self.sliding_window_size or 0) + 2 * self.page_size
        )
        # Diagnostic: see _swa_evict_report.
        self._swa_trace = envs.SGLANG_SWA_EVICT_TRACE.get()
        self._swa_trace_n = 0
        self._swa_trace_cap = envs.SGLANG_SWA_EVICT_TRACE_CAP.get()
        self._swa_fallback_warned = False
        self._swa_ev = {'tombstone': 0, 'trim': 0, 'trim_tok': 0, 'delete': 0,
                        'skip_internal': 0, 'pin': 0, 'fallback_swa': 0}
        # TEMPORARY: see _dbg_check_backed / assert_mixed_kv_swa_invariants.
        self._dbg_swa = bool(
            self._mixed_kv_enabled and os.environ.get("SGLANG_CHECK_MIXED_KV_SWA")
        )
        self._dbg_n = 0                        # phantom reports emitted so far
        self._dbg_ins = ("-", "-", "-", "-")   # last _insert_helper context
        self._dbg_probe_max = 0                # tree high-water mark seen

        self.reset()

    ##### Public API #####

    def supports_swa(self) -> bool:
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for SWARadixCache"
        return True

    def swa_evict_tail_reserve(self, committed_len: int) -> int:
        # Mixed-KV inserts stop ``mixed_kv_tail_to_drop`` short of
        # page_floor(seq_len), so the stock frontier (seq_len - window - page)
        # lands ABOVE the insert boundary instead of below it. That inverts the
        # invariant _evict_swa exists to hold: the tail this cache is about to
        # insert has already had its SWA released. Reserving the same span
        # restores it -- costs one hp-recent window of SWA slots per live
        # request, and keeps every inserted node backed by real SWA.
        if not self._mixed_kv_enabled:
            return 0
        return mixed_kv_tail_to_drop(
            self.token_to_kv_pool_allocator, self.page_size, committed_len
        )

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.full_lock_ref = 1
        self.root_node.swa_lock_ref = 1
        self.full_evictable_size_ = 0
        self.swa_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.swa_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(is_swa_list=False)
        self.swa_lru_list = LRUList(is_swa_list=True)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            params: MatchPrefixParams containing key.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """

        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        prev_prefix_len = params.prev_prefix_len
        swa_evicted_seqlen = params.swa_evicted_seqlen

        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)

        key, value = maybe_bigram_convert(self.is_eagle, key, value)

        prefix_len = self._insert_helper(
            self.root_node, key, value, prev_prefix_len, swa_evicted_seqlen
        )
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        """Cache request when it finishes."""
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = self.key_convert_fn(token_ids)
        keys = page_align_keys(keys, self.page_size)
        page_aligned_len = len(keys)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(
            keys[:page_aligned_len],
            req.extra_key,
            is_bigram=self.is_eagle,
        )
        old_prefix_len = req.cache_protected_len

        # Mixed-KV: TRIM the tail rather than refuse to insert.
        #
        # RadixCache returns early here and lets cache_unfinished_req own the
        # tree. That does not transfer: cache_unfinished_req is skipped once a
        # request joins decoding_reqs (scheduler_output_processor_mixin.py:193),
        # so on a chunked prefill the FINAL partial chunk only ever reaches the
        # tree through this method. Returning early cost 30,000 -> 24,312 cached
        # (= 3*8192 - trim) at 30k, and the warm-pass TTFT ratio went 0.068 ->
        # 0.252, failing the throughput harness's warm_took gate.
        #
        # So keep the original insert, minus the per-request HP-recent band --
        # which is the actual bug -- and stop short of any request-owned partial
        # quant page. The lock handling below is unchanged, so this does not
        # reintroduce the lock_ref=0-leaf-under-retract hazard that motivated
        # RadixCache's early return (that came from its re-match + relock dance,
        # which this method never had).
        insert_len = page_aligned_len
        if self._mixed_kv_enabled:
            trim = mixed_kv_tail_to_drop(
                self.token_to_kv_pool_allocator, self.page_size, page_aligned_len
            )
            insert_len = mixed_kv_slack_insert_limit(
                req, max(0, page_aligned_len - trim)
            )
            if insert_len != page_aligned_len:
                radix_key = RadixKey(
                    keys[:insert_len], req.extra_key, is_bigram=self.is_eagle
                )
                values = values[:insert_len]

        # Radix Cache takes one ref in memory pool
        # Note: the insert function already frees the overlapped kv_indices
        if is_insert and insert_len > 0:
            self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    prev_prefix_len=old_prefix_len,
                    swa_evicted_seqlen=req.swa_evicted_seqlen,
                )
            )
            free_from = max(old_prefix_len, insert_len)
        else:
            free_from = old_prefix_len

        # Everything the tree did not take is still request-owned. Includes the
        # trimmed HP-recent band and the unaligned tail; the quant slack rides
        # along so a partial page is released with the page it belongs to.
        self.token_to_kv_pool_allocator.free(
            mixed_kv_with_quant_slack(req, kv_indices[free_from:])
            if self._mixed_kv_enabled
            else kv_indices[free_from:page_aligned_len]
        )
        if not self._mixed_kv_enabled:
            # free the unaligned tail
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])

        # Remove req slot release the cache lock
        self.dec_lock_ref(
            req.last_node, DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock)
        )

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        keys = self.key_convert_fn(token_ids)
        keys = page_align_keys(keys, self.page_size)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)
        old_prefix_len = req.cache_protected_len

        # Mixed-KV: drop the per-request HP-recent tail (the live request still
        # owns those slots -- they stay addressable through the prefix_indices
        # rebuild below), then stop short of any request-owned partial quant
        # page. Clamping the INSERT KEY is what bounds the frees too: unlike
        # RadixCache's, this class's ``_insert_helper`` frees overlapped indices
        # internally, and only walks as far as the key it is given. That is why
        # RadixCache's explicit post-insert dup free is deliberately NOT ported
        # here -- it would be a straight double free.
        insert_keys = keys
        if self._mixed_kv_enabled:
            trim = mixed_kv_tail_to_drop(
                self.token_to_kv_pool_allocator, self.page_size, len(keys)
            )
            if trim > 0:
                insert_keys = keys[: len(keys) - trim]
            insert_limit = mixed_kv_slack_insert_limit(req, len(insert_keys))
            if insert_limit < len(insert_keys):
                insert_keys = insert_keys[:insert_limit]
            if len(insert_keys) != len(keys):
                radix_key = RadixKey(
                    insert_keys, req.extra_key, is_bigram=self.is_eagle
                )

        # Radix Cache takes one ref in memory pool
        # Note: the insert function already frees the overlapped kv_indices
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values[: len(insert_keys)],
                prev_prefix_len=old_prefix_len,
            )
        )
        new_prefix_len = result.prefix_len

        # The prefix indices could be updated, reuse it.
        if self._mixed_kv_enabled:
            # Match the FULL (untrimmed) key so ``cache_protected_len`` never
            # regresses -- ``bypass_mixed_kv_cap`` is required, since capping
            # this internal match desyncs it from the admission set and the
            # reconstruction below then silently truncates and leaks slots.
            # It must still stop before a request-owned partial quant page.
            match_len = mixed_kv_slack_insert_limit(req, len(keys))
            match_result = self.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        keys[:match_len], req.extra_key, is_bigram=self.is_eagle
                    ),
                    bypass_mixed_kv_cap=True,
                )
            )
        else:
            match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        if self._mixed_kv_enabled:
            # Only advance the protected region; never regress. Note the two
            # asserts used on the stock path are both reachable-false here:
            # ``old_prefix_len`` can exceed the slack-clamped match_len, and
            # ``_match_prefix_helper``'s ``best_value_len`` truncation can
            # return fewer indices than were just inserted when a tombstone
            # leaves < sliding_window_size non-tombstone tokens behind it.
            full_match_len = len(new_indices)
            if full_match_len > old_prefix_len:
                # Reclaim our-own slots at positions that became tree-covered
                # only AFTER our insert (via a sibling's node). We are about to
                # overwrite req_to_token with the tree's ids there, so the
                # originals would otherwise leak. Clamped to the slack boundary
                # so the free cannot cross into a page the request still owns.
                extra_free_start = max(old_prefix_len, new_prefix_len, len(insert_keys))
                extra_free_end = mixed_kv_slack_insert_limit(req, full_match_len)
                if extra_free_start < extra_free_end:
                    self.token_to_kv_pool_allocator.free(
                        kv_indices[extra_free_start:extra_free_end]
                    )
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(old_prefix_len, full_match_len)),
                    new_indices[old_prefix_len:],
                )
                req.cache_protected_len = full_match_len
                self.dec_lock_ref(
                    req.last_node,
                    DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
                )
                result = self.inc_lock_ref(new_last_node)
                req.last_node = new_last_node
                req.swa_uuid_for_lock = result.swa_uuid_for_lock

            # `req.prefix_indices` is used by `PrefillAdder::add_chunked_req`.
            # When cache_protected_len did not advance it can exceed
            # len(new_indices); slicing would then silently drop slot ids.
            protected_len = req.cache_protected_len
            if protected_len <= len(new_indices):
                protected_indices = new_indices[:protected_len]
            else:
                protected_indices = kv_indices[:protected_len].to(dtype=torch.int64)

            if protected_len < len(kv_indices):
                req.prefix_indices = torch.cat(
                    [protected_indices, kv_indices[protected_len:]]
                )
            else:
                req.prefix_indices = protected_indices
            return

        assert old_prefix_len <= len(new_indices), f"{old_prefix_len=}, {new_indices=}"
        assert new_prefix_len <= len(new_indices), f"{new_prefix_len=}, {new_indices=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),
            new_indices[old_prefix_len:],
        )

        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(
            req.last_node, DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock)
        )
        result = self.inc_lock_ref(new_last_node)
        swa_uuid_for_lock = result.swa_uuid_for_lock

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node
        req.swa_uuid_for_lock = swa_uuid_for_lock

    def assert_mixed_kv_swa_invariants(self) -> None:
        """TEMPORARY diagnostic: the two structural claims of the mixed-KV fix.

        Tests them directly rather than inferring them from an accuracy number
        (the scheduler's SWA assertion is not a usable proxy -- it stayed silent
        through a run that lost 0.52 accuracy).

        1. No tree node owns an HP-recent slot id. Those ids are per-req_pool_idx
           and get re-issued to the next occupant, so a tree node holding one
           aliases another request's live KV.
        2. Every NON-tombstone tree token owns a live SWA slot. match_prefix
           hands non-tombstone nodes to the next request as SWA-valid; if the
           slot was already released, its sliding layers read
           full_to_swa_index_mapping == 0. This is the one that costs accuracy,
           and it is invisible to (1).

        Gated behind SGLANG_CHECK_MIXED_KV_SWA; drop once verified.
        """
        if not self._mixed_kv_enabled:
            return
        alloc = getattr(self.token_to_kv_pool_allocator, "full_attn_allocator", None)
        if alloc is None or not hasattr(alloc, "_hp_recent_offset"):
            return
        lo = int(alloc._hp_recent_offset)
        hi = lo + int(alloc.max_req_slots) * int(alloc.hp_recent_ring_size)
        mapping = self.token_to_kv_pool_allocator.full_to_swa_index_mapping

        stack = [self.root_node]
        nodes = tokens = live = phantom = 0
        while stack:
            node = stack.pop()
            if node.value is not None and len(node.value) > 0:
                nodes += 1
                tokens += len(node.value)
                bad = ((node.value >= lo) & (node.value < hi)).sum().item()
                assert bad == 0, (
                    f"HP-recent slot id in radix tree: {bad} of {len(node.value)} "
                    f"node values fall in [{lo}, {hi}); these alias across requests"
                )
                if not node.swa_tombstone:
                    live += len(node.value)
                    gone = int((mapping[node.value] == 0).sum().item())
                    phantom += gone
                    assert gone == 0, (
                        f"non-tombstone node {node.id} holds {gone} of "
                        f"{len(node.value)} tokens whose SWA slot was already "
                        "released; match_prefix would serve them as SWA-valid"
                    )
            stack.extend(node.children.values())

        counted = self.swa_evictable_size_ + self.swa_protected_size_
        assert counted == live, (
            f"SWA counter desync: swa_evictable_size_ + swa_protected_size_ "
            f"= {counted}, walked non-tombstone tokens = {live}"
        )

        # Report the high-water mark. Without this the gate is unfalsifiable: a
        # silently-skipped check and an empty tree both look like "passed".
        if tokens > self._dbg_probe_max:
            self._dbg_probe_max = tokens
            logger.info(
                "[SWA tree probe] %d nodes / %d tokens, %d non-tombstone, "
                "%d phantom-SWA, 0 in HP-recent range [%d, %d)",
                nodes, tokens, live, phantom, lo, hi,
            )

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_swa_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #swa_tokens: {total_swa_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        full_num_tokens = params.num_tokens
        swa_num_tokens = params.swa_num_tokens
        full_num_evicted = 0
        swa_num_evicted = 0
        if full_num_tokens > 0:
            # get the least recently used leaf node that is not locked
            x = self.full_lru_list.get_leaf_lru_no_lock()

            while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
                assert (
                    x != self.root_node
                ), f"root node should not exist in full lru list, {x.id=}"
                assert x.full_lock_ref == 0, f"node is in use, {x.id=}"

                # 1. free node kv indices, evict full and swa tokens
                self.token_to_kv_pool_allocator.free(x.value)
                full_num_evicted += len(x.value)
                swa_num_evicted += len(x.value)

                # 2. get the next leaf, update the lru lists
                x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
                self.full_lru_list.remove_node(x)
                self.swa_lru_list.remove_node(x)

                # 3. delete the leaf node
                self._delete_leaf(x)

                # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
                x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
                full_num_evicted += leaf_full_num_evicted

                # 5. if parent has no more children, it is a leaf. It is possible that this node is lru, so
                # we need to get the first leaf node in the lru list
                if len(x.parent.children) == 0:
                    x_next = self.full_lru_list.get_leaf_lru_no_lock()

                x = x_next

        if swa_num_evicted < swa_num_tokens:
            s, f = self._evict_swa_pass(
                swa_num_tokens - swa_num_evicted,
                protect=self._swa_keep_prefix_tail,
            )
            swa_num_evicted += s
            full_num_evicted += f
            if swa_num_evicted < swa_num_tokens and self._swa_keep_prefix_tail:
                # Liveness fallback. Callers size their request against
                # swa_evictable_size_, and alloc raises RuntimeError (killing
                # the server) if evict under-delivers -- protection may never
                # turn a satisfiable request into a failure. Reaching here means
                # even one trailing window per cached entry does not fit, i.e.
                # the tier is genuinely undersized; deleting entries is then the
                # only option left, exactly the old behavior.
                s, f = self._evict_swa_pass(
                    swa_num_tokens - swa_num_evicted, protect=False
                )
                swa_num_evicted += s
                full_num_evicted += f
                self._swa_ev['fallback_swa'] += s
                if s > 0 and not self._swa_fallback_warned:
                    self._swa_fallback_warned = True
                    logger.warning(
                        "SWA keep-prefix-tail fallback fired: the protected "
                        "sweep could not reclaim %d tokens, so cache entries "
                        "are being deleted. The SWA tier is too small to hold "
                        "one trailing window per cached prefix.",
                        swa_num_tokens,
                    )

        self.update_eviction_metrics(full_num_evicted + swa_num_evicted, start_time)
        if swa_num_tokens > 0:
            self._swa_evict_report(swa_num_tokens, swa_num_evicted)
        return EvictResult(
            num_tokens_evicted=full_num_evicted, swa_num_tokens_evicted=swa_num_evicted
        )

    def _evict_swa_pass(self, target: int, protect: bool) -> Tuple[int, int]:
        """One LRU sweep of the SWA tier. Returns (swa_evicted, full_evicted).

        ``protect=True`` preserves every cached prefix's ability to match:
        internal nodes are skipped when tombstoning them would drop some leaf's
        trailing live run under ``_swa_keep_target``, and leaves are trimmed to
        that target and PINNED -- never deleted. Pinned tokens move to
        ``swa_protected_size_`` and the LRU getters skip pinned nodes, so the
        tree stops advertising them as evictable and later sweeps never
        re-visit them. Once every leaf is pinned, a protected sweep reclaims
        nothing more; the caller then falls back to ``protect=False``, which
        walks through pins (``ignore_pin``) and restores the original
        destructive behavior.
        """
        swa_evicted = 0
        full_evicted = 0
        ignore_pin = not protect
        # get the least recently used node that is not locked, doesn't have to be a leaf
        x = self.swa_lru_list.get_lru_no_lock(ignore_pin=ignore_pin)

        while swa_evicted < target and (self.swa_lru_list.in_list(x)):
            assert not x.swa_tombstone, f"duplicate swa tombstone node, {x.id=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.swa_lock_ref == 0, f"node is in use by swa kv indices, {x.id=}"

            # Capture the successor BEFORE mutating: a trim's _split_node pulls
            # x out of the SWA LRU and re-inserts both halves at MRU, so a
            # get_prev_no_lock(x) afterwards would walk from the wrong end of
            # the list and cut this eviction pass short. MRU re-insertions land
            # ahead of the walk, so each node is visited at most once per sweep.
            x_next = self.swa_lru_list.get_prev_no_lock(x, ignore_pin=ignore_pin)

            if len(x.children) > 0:
                if protect and self._swa_tombstone_would_orphan(x):
                    # Inside some cached prefix's trailing window. Tombstoning
                    # here would not shorten those matches, it would end them
                    # (see _swa_tombstone_would_orphan). Leave it live and
                    # carry on down the LRU; the reclaimable SWA is above.
                    self._swa_ev['skip_internal'] += 1
                    x = x_next
                    continue

                if x.swa_pinned:
                    # Fallback only (protected sweeps never see pinned nodes).
                    # A pinned tail that later gained children -- a longer
                    # duplicate prompt extended past it -- is internal now, and
                    # _tombstone_internal_node accounts in evictable terms, so
                    # unpin first.
                    self._swa_unpin(x)

                # 1. an internal node, free swa tokens.
                self.token_to_kv_pool_allocator.free_swa(x.value)
                swa_evicted += len(x.value)

                # 2. update the lru list
                self.swa_lru_list.remove_node(x)

                # 3. tombstone the node
                self._tombstone_internal_node(x)
                self._swa_ev['tombstone'] += 1
            else:
                assert (
                    x.full_lock_ref == 0
                ), f"leaf node with full lock must also have swa lock, {x.id=}"

                if protect:
                    trimmed = self._swa_trim_leaf_to_window(x)
                    if trimmed > 0:
                        # Reclaimed this entry's SWA down to its trailing window
                        # WITHOUT evicting it. The full tier is untouched and
                        # the prefix still matches, so this is strictly better
                        # than deletion. The tail is pinned by the trim.
                        swa_evicted += trimmed
                        self._swa_ev['trim'] += 1
                        self._swa_ev['trim_tok'] += trimmed
                    else:
                        # Already at (or under) the keep target: this leaf IS
                        # some prefix's irreducible tail. Deleting it would
                        # gain a window's worth of slots at the cost of the
                        # entire entry -- the full-to-zero cliff this flag
                        # exists to prevent. Pin it instead; the fallback pass
                        # may still delete it if the tier truly cannot hold
                        # the tails.
                        self._swa_pin(x)
                        self._swa_ev['pin'] += 1
                    x = x_next
                    continue

                # 1. a leaf node, free full and swa tokens
                self.token_to_kv_pool_allocator.free(x.value)
                full_evicted += len(x.value)
                swa_evicted += len(x.value)

                # 2. update the lru lists
                self.full_lru_list.remove_node(x)
                self.swa_lru_list.remove_node(x)

                self._swa_ev['delete'] += 1
                # 3. delete the leaf node (pin-aware accounting inside)
                self._delete_leaf(x)

                # 4. Iteratively delete tombstone leaves to maintain invariant
                # that leaf nodes are not tombstone
                _, tombstone_full_evicted = self._iteratively_delete_tombstone_leaf(x)
                full_evicted += tombstone_full_evicted

            x = x_next

        return swa_evicted, full_evicted

    def _swa_pin(self, node: TreeNode) -> None:
        assert not node.swa_pinned and node.swa_lock_ref == 0
        node.swa_pinned = True
        self.swa_evictable_size_ -= len(node.value)
        self.swa_protected_size_ += len(node.value)

    def _swa_unpin(self, node: TreeNode) -> None:
        assert node.swa_pinned
        node.swa_pinned = False
        if node.swa_lock_ref == 0:
            self.swa_evictable_size_ += len(node.value)
            self.swa_protected_size_ -= len(node.value)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        """
        Increment the lock reference count for the node. Returns the swa_uuid_for_lock, which needs
        to be passed to dec_lock_ref.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the swa_lock_ref for nodes between the [last node, swa_uuid_for_lock], inclusive.
        """
        if self.disable:
            return IncLockRefResult()

        swa_lock_size = 0
        swa_uuid_for_lock = None
        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1

            # lock swa if we have not reached the sliding window size.
            # When we reach the sliding window size, we will set the swa_uuid_for_lock.
            # caller needs to pass the swa_uuid_for_lock to dec_lock_ref
            if swa_lock_size < self.sliding_window_size:
                assert (
                    not node.swa_tombstone
                ), f"inc_lock_swa on swa_tombstone node, {node.id=}"
                # A pinned node's tokens are already in protected.
                if node.swa_lock_ref == 0 and not node.swa_pinned:
                    self.swa_evictable_size_ -= len(node.value)
                    self.swa_protected_size_ += len(node.value)
                node.swa_lock_ref += 1
                swa_lock_size += len(node.value)
                if swa_lock_size >= self.sliding_window_size:
                    if node.swa_uuid is None:
                        node.swa_uuid = gen_swa_uuid()
                    swa_uuid_for_lock = node.swa_uuid
            node = node.parent
        return IncLockRefResult(swa_uuid_for_lock=swa_uuid_for_lock)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the swa_lock_ref for nodes between the [last node, swa_uuid_for_lock], inclusive.
        If swa_uuid_for_lock is None, it unlocks to the root, exclusive.
        """
        swa_uuid_for_lock = params.swa_uuid_for_lock if params is not None else None

        if self.disable:
            return DecLockRefResult()

        dec_lock_swa = True
        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1

            if dec_lock_swa:
                assert (
                    not node.swa_tombstone
                ), f"dec_lock_ref on swa_tombstone node, {node.id=}"
                assert (
                    node.swa_lock_ref > 0
                ), f"dec_lock_ref on node with {node.swa_lock_ref=}, {node.id=}"

                # A pinned node's tokens stay in protected after unlock.
                if node.swa_lock_ref == 1 and not node.swa_pinned:
                    self.swa_evictable_size_ += len(node.value)
                    self.swa_protected_size_ -= len(node.value)
                node.swa_lock_ref -= 1
                if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:
                    dec_lock_swa = False

            node = node.parent

        return DecLockRefResult()

    def sanity_check(self):
        self.full_lru_list.sanity_check(self)
        self.swa_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and swa_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def swa_evictable_size(self) -> int:
        return self.swa_evictable_size_

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and swa_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def swa_protected_size(self) -> int:
        # protected size refers to the size of the swa cache that is locked
        return self.swa_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        full_evictable_size = self.full_evictable_size()
        swa_evictable_size = self.swa_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
            f"SWA LRU list evictable size: {self.swa_lru_list.sanity_check_evictable_size()}\n"
        )

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        SWA prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without swa tombstone,
        or 2. the number of matching tokens from the matched node to the last swa tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = self.get_child_key_fn(key)

        value = []
        # for path connected to root without tombstone, always match, so set to inf
        match_len_since_tombstone = float("inf")
        best_value_len = 0
        best_last_node = node
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]

            if child.swa_tombstone:
                # update best_value_len and best_last_node if needed
                if match_len_since_tombstone >= self.sliding_window_size:
                    best_value_len = len(value)
                    best_last_node = node
                # reset match_len_since_tombstone if we hit a tombstone node
                match_len_since_tombstone = 0

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                if not new_node.swa_tombstone:
                    match_len_since_tombstone += len(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                if not child.swa_tombstone:
                    match_len_since_tombstone += len(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if match_len_since_tombstone >= self.sliding_window_size:
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:
        """Preprocess the key before matching."""
        key = params.key
        key, _ = maybe_bigram_convert(self.is_eagle, key)

        if self.disable or len(key) == 0:
            return None

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        # Never match into the HP-recent band: those slot ids are per-request
        # and aliased, so sharing them across requests is a wrong read.
        # Internal callers (``cache_unfinished_req``'s post-insert
        # sibling-coverage match) pass ``bypass_mixed_kv_cap=True``; capping
        # that one desyncs the ``cache_protected_len`` admission set.
        #
        # SWA-specific: ``_match_prefix_helper``'s ``best_value_len`` gate needs
        # >= sliding_window_size non-tombstone tokens after the last tombstone,
        # so shortening the key can drop a marginal match to zero rather than
        # merely shorten it. Correct either way -- but it is the first thing to
        # check if the prefix-cache hit rate regresses.
        if self._mixed_kv_enabled and len(key) > 0 and not params.bypass_mixed_kv_cap:
            cap = mixed_kv_match_cap(
                len(key),
                self._mixed_kv_hp_prefix_tokens,
                self._mixed_kv_match_cap_overhead,
                self.page_size,
            )
            if cap < len(key):
                key = key[:cap]
            if len(key) == 0:
                return None

        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the matched result."""
        node_update = last_node
        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows swa to evict nodes closer to root first
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        self.swa_lru_list.reset_node_and_parents_mru(node_update, self.root_node)

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.swa_tombstone = child.swa_tombstone
        # Both halves of a pinned tail stay pinned: the pin protects the band
        # [end - keep_target, end), and a match ending mid-tail splits inside
        # it. Token counters are unaffected (both halves stay in protected).
        new_node.swa_pinned = child.swa_pinned
        new_node.full_lock_ref = child.full_lock_ref
        new_node.swa_lock_ref = child.swa_lock_ref
        new_node.key = child.key[:split_len]
        assert len(new_node.key) > 0, f"new_node.key should not be empty"
        new_node.value = child.value[:split_len].clone()
        # parent inherits the swa_uuid from child for swa lock ref
        new_node.swa_uuid = child.swa_uuid
        child.swa_uuid = None
        # child time should be later than parent's time for swa tombstone
        child.last_access_time = get_last_access_time()

        # remove the child from the lru lists because it is being split
        self.full_lru_list.remove_node(child)
        if not new_node.swa_tombstone:
            self.swa_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        assert len(child.key) > 0, f"child.key should not be empty"
        child.value = child.value[split_len:].clone()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if not new_node.swa_tombstone:
            self.swa_lru_list.insert_mru(new_node)
            self.swa_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        update_kv_after_len: int,
        swa_evicted_seqlen: int = 0,
    ) -> int:
        # Update the last access time from root to leaf, so that
        # swa will tombstone the node closer to root first
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            if not node.swa_tombstone:
                self.swa_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        self._dbg_ins = (update_kv_after_len, swa_evicted_seqlen, 0, len(key))
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if not node.swa_tombstone:
                self.swa_lru_list.reset_node_mru(node)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            # if tombstone after update_kv_after_len, update node.value to be the input value.
            # This is needed because it is possible that the last sliding window size tokens
            # contains tombstone. If this is the case and we don't update the kv value, then
            # the prefill prefix matching will stuck.
            if update_kv_after_len < total_prefix_length + prefix_len:
                # For page_size > 1 and chunked prefill case, update_kv_after_len may be not page-aligned due to a trailing partial page
                # (kept in the request but not inserted into the radix tree) appended to prefix_indices.
                if node.swa_tombstone:
                    assert (
                        node.swa_lock_ref == 0
                    ), f"tombstone swa_lock_ref should always be 0, {node.full_lock_ref=}, {node.swa_lock_ref=}, {node.id=}"
                    assert (
                        swa_evicted_seqlen % self.page_size == 0
                    ), f"swa_evicted_seqlen must be page aligned, {swa_evicted_seqlen=}, {self.page_size=}"
                    if swa_evicted_seqlen <= total_prefix_length:
                        # Branch 1: all swa tokens of value[:prefix_len] are not evicted, so we can insert it to the tree directly.
                        # Free full tokens in the original tree node.
                        self.token_to_kv_pool_allocator.free(node.value[:prefix_len])
                        # Overwrite the new value in request to the tree node.
                        node.value = value[:prefix_len].clone()
                        node.swa_tombstone = False
                        self.swa_lru_list.insert_mru(node)
                        self.swa_evictable_size_ += len(node.value)
                        self._dbg_check_backed(node, "untombstone_b1")
                    elif swa_evicted_seqlen < total_prefix_length + prefix_len:
                        # Branch 2: part of swa tokens of value[:prefix_len] are evicted, so we need to split the node and insert the value to new node.
                        start_update_idx = swa_evicted_seqlen - total_prefix_length
                        self.token_to_kv_pool_allocator.free(
                            node.value[start_update_idx:prefix_len]
                        )
                        self._split_node(node.key, node, start_update_idx)
                        # Here node is the new node after split, so we can overwrite the value to the new node.
                        # The old node is still swa tombstone and the full token is not freed.
                        node.value = value[start_update_idx:prefix_len].clone()
                        self.token_to_kv_pool_allocator.free(value[:start_update_idx])
                        node.swa_tombstone = False
                        self.swa_lru_list.insert_mru(node)
                        self.swa_evictable_size_ += len(node.value)
                        self._dbg_check_backed(node, "untombstone_b2")
                    else:
                        # Branch 3: all swa tokens of value[:prefix_len] are evicted, so we don't need to update the node.
                        self.token_to_kv_pool_allocator.free(value[:prefix_len])
                else:
                    # The node is not tombstone, so we don't need to update the node.
                    self.token_to_kv_pool_allocator.free(value[:prefix_len])

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            self._dbg_ins = (
                update_kv_after_len,
                swa_evicted_seqlen,
                total_prefix_length,
                len(key),
            )

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            # Layout: |--- total_prefix_length ---|--- len(key) ---|
            #         ^                           ^                ^
            #         0              total_prefix_length     total_length
            #
            # Cases based on swa_evicted_seqlen position:
            # 1. swa_evicted_seqlen <= total_prefix_length:
            #    Already handled in the while loop above. All of len(key) is non-tombstone.
            # 2. total_prefix_length < swa_evicted_seqlen < total_length:
            #    Split: [total_prefix_length, swa_evicted_seqlen) as tombstone,
            #           [swa_evicted_seqlen, total_length) as non-tombstone.
            # 3. swa_evicted_seqlen >= total_length:
            #    All remaining tokens are evicted. Free value and return without
            #    creating a node (leaf nodes must not be tombstone).
            #    Note: the -page_size fix in _evict_swa prevents this case from
            #    occurring in normal operation. This check is a defensive guard
            #    against unexpected eviction states from other code paths.
            #
            #    ``>=``, not ``==``: a caller that shortens the insert key can
            #    push the watermark PAST total_length, and the equality test
            #    then fell through to the non-tombstone _add_new_node below --
            #    crediting swa_evictable_size_ for tokens whose SWA slots were
            #    already released. That is not just a miscount: match_prefix
            #    hands those slots to the next request, whose sliding layers
            #    then read full_to_swa_index_mapping == 0. Mixed-KV hit this on
            #    every finished request (the tail trim shortens the key by
            #    exactly the span _evict_swa had just released).
            if swa_evicted_seqlen >= total_prefix_length + len(key):
                self.token_to_kv_pool_allocator.free(value)
                return total_prefix_length

            if (
                swa_evicted_seqlen > total_prefix_length
                and swa_evicted_seqlen < total_prefix_length + len(key)
            ):
                swa_tombstone_len = swa_evicted_seqlen - total_prefix_length
                node = self._add_new_node(
                    node,
                    key[:swa_tombstone_len],
                    value[:swa_tombstone_len],
                    swa_tombstone=True,
                )
                key = key[swa_tombstone_len:]
                value = value[swa_tombstone_len:]

            self._add_new_node(node, key, value, swa_tombstone=False)
        return total_prefix_length

    def _add_new_node(
        self,
        parent: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        swa_tombstone: bool = False,
    ) -> TreeNode:
        assert len(key) > 0, f"key should not be empty"
        new_node = TreeNode()
        new_node.parent = parent
        new_node.key = key
        new_node.value = value.clone()
        new_node.swa_tombstone = swa_tombstone
        parent.children[self.get_child_key_fn(key)] = new_node
        self.full_lru_list.insert_mru(new_node)
        self.full_evictable_size_ += len(value)
        if not swa_tombstone:
            self.swa_lru_list.insert_mru(new_node)
            self.swa_evictable_size_ += len(value)
            self._dbg_check_backed(new_node, "_add_new_node")
        return new_node

    def _dbg_check_backed(self, node: TreeNode, site: str) -> None:
        """TEMPORARY: report a node credited to swa_evictable_size_ whose SWA is gone.

        Fires at the CREATION site, with the insert parameters that produced it
        -- the tree-walk probe only sees the aftermath, by which point the
        originating call is long gone. Drop with the rest of the probe.
        """
        if not self._dbg_swa or len(node.value) == 0:
            return
        mapping = self.token_to_kv_pool_allocator.full_to_swa_index_mapping
        gone = int((mapping[node.value] == 0).sum().item())
        if gone:
            self._dbg_n += 1
            if self._dbg_n <= 25:
                logger.error(
                    "[SWA phantom @%s] node=%d len=%d unbacked=%d | insert ctx "
                    "update_kv_after_len=%s swa_evicted_seqlen=%s total_prefix=%s "
                    "keylen=%s | first_vals=%s",
                    site, node.id, len(node.value), gone,
                    *self._dbg_ins,
                    node.value[:8].tolist(),
                )

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.swa_tombstone and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.swa_lock_ref == 0
            ), f"tombstone swa_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.swa_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens
            self.token_to_kv_pool_allocator.free(node.parent.value)
            full_num_evicted += len(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            not node.swa_tombstone
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"
        self.full_evictable_size_ -= len(node.key)
        # A pinned tail can be deleted by FULL-tier eviction or the SWA
        # fallback pass; its tokens live in protected, not evictable.
        if node.swa_pinned:
            self.swa_protected_size_ -= len(node.key)
        else:
            self.swa_evictable_size_ -= len(node.key)

    def _swa_evict_report(self, requested: int, evicted: int) -> None:
        """TEMPORARY (SGLANG_SWA_EVICT_TRACE): what did an evict pass actually do?

        Two models of SWA retention have now fit the aggregate measurements and
        then failed a prediction, so this records the primitive actions instead
        of inferring them: how many internal nodes were tombstoned, how many
        leaves trimmed vs deleted outright, and what the tree thinks it holds.
        A pass that reports ``deleted`` >> ``trimmed`` is destroying cache
        entries to reclaim slots the matcher never needed.
        """
        if not self._swa_trace:
            return
        self._swa_trace_n += 1
        if self._swa_trace_n > self._swa_trace_cap:
            return
        logger.info(
            "[swa-evict] want=%d got=%d | tombstoned=%d trimmed=%d(%d tok) "
            "deleted=%d fallback_swa=%d | evictable=%d protected=%d "
            "skip_internal=%d pinned=%d",
            requested, evicted, self._swa_ev["tombstone"], self._swa_ev["trim"],
            self._swa_ev["trim_tok"], self._swa_ev["delete"],
            self._swa_ev["fallback_swa"],
            self.swa_evictable_size_, self.swa_protected_size_,
            self._swa_ev["skip_internal"], self._swa_ev["pin"],
        )

    def _min_live_run_below(self, node: TreeNode, limit: int) -> int:
        """Min over leaves L below ``node`` of live tokens on the path (node, L].

        Capped at ``limit`` and early-terminated there, so the walk descends at
        most ``limit`` tokens' worth of nodes per branch -- about one node for
        the chain-shaped trees a long-prompt workload builds.

        Branches containing a tombstone deeper than ``node`` are skipped: the
        matcher measures a leaf's run from the *last* tombstone above it, so
        those leaves are unaffected by whether ``node`` is tombstoned.
        """
        if not node.children:
            return 0  # ``node`` is itself a leaf: nothing follows it
        best = limit
        for child in node.children.values():
            if child.swa_tombstone:
                continue
            n = len(child.value)
            if n < limit and child.children:
                n = min(n + self._min_live_run_below(child, limit - n), limit)
            if n < best:
                best = n
                if best <= 0:
                    break
        return best

    def _swa_tombstone_would_orphan(self, node: TreeNode) -> bool:
        """Would tombstoning ``node`` leave some leaf with too short a live run?

        ``_match_prefix_helper`` only returns a match while
        ``match_len_since_tombstone >= sliding_window_size``, so a leaf whose
        trailing live run falls below the window stops matching altogether --
        the cache entry is destroyed, not shortened. Everything ABOVE a prefix's
        trailing window is still free to tombstone, which is where essentially
        all the reclaimable SWA is.
        """
        return self._min_live_run_below(node, self._swa_keep_target) < (
            self._swa_keep_target
        )

    def _swa_trim_leaf_to_window(self, leaf: TreeNode) -> int:
        """Free a cached prefix's SWA down to its trailing window. Returns freed.

        A leaf is where a cached prefix's retained SWA actually sits: ``evict``
        never tombstones leaves (the tree's invariant is that a leaf is not a
        tombstone), so its only way to reclaim that SWA has been to delete the
        whole entry -- throwing away a perfectly good full-tier prefix to
        reclaim sliding-window slots the matcher never needed.

        Split it instead. ``head`` keeps everything but the last
        ``_swa_keep_target`` tokens and becomes a tombstoned INTERNAL node;
        ``tail`` keeps the target and stays a live leaf. A match walking this
        path resets ``match_len_since_tombstone`` at ``head`` and then
        accumulates >= ``sliding_window_size`` over ``tail`` even after the
        match key (which ends up to two pages before the insert end) splits it,
        so ``best_value_len`` still reaches the end -- the prefix matches in
        full, at a cost of ~one window of SWA slots instead of its length.

        Returns 0 when the leaf is already at or under the target, and the
        caller decides what to do with it (protected pass: skip).
        """
        keep = self._swa_keep_target
        n = len(leaf.value)
        # Splitting must leave BOTH sides non-empty, and must actually pay for
        # itself -- a split that frees less than a page is churn.
        if n <= keep + self.page_size:
            return 0
        split_len = n - keep
        if self.page_size > 1:
            split_len = (split_len // self.page_size) * self.page_size
            if split_len <= 0:
                return 0

        head = self._split_node(leaf.key, leaf, split_len)
        assert head.children, "split head must have the tail as a child"
        # ``head`` is live at this point (it inherits the leaf's flags), so it is
        # in the SWA LRU. Take it out before tombstoning: tombstoned nodes are
        # not members of that list.
        self.token_to_kv_pool_allocator.free_swa(head.value)
        self.swa_lru_list.remove_node(head)
        self._tombstone_internal_node(head)
        # The tail is now some prefix's irreducible live window. Left as an
        # ordinary LRU candidate, a later sweep would select it, fail to trim
        # it (too short), and the fallback would delete the entry -- the exact
        # deferred-kill that sank the first version of this fix. Pin it.
        self._swa_pin(leaf)
        return len(head.value)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        node.swa_tombstone = True
        self.swa_evictable_size_ -= len(node.key)

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.swa_tombstone
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if not cur_node.swa_tombstone:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                current_node.id,
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"sr={current_node.swa_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"sll={self.swa_lru_list.in_list(current_node)}",
                f"ts={current_node.swa_tombstone}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_swa_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if not current_node.swa_tombstone:
                total_swa_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_swa_size
