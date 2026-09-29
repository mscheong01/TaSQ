from __future__ import annotations

import logging
import os
import time
import warnings
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache
from sglang.srt.observability.metrics_collector import QueueCount
from sglang.srt.utils.common import ceil_align, raise_error_or_warn
from sglang.srt.utils.request_logger import disable_request_logging
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerRuntimeCheckerMixin:
    def _session_held_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_tokens()
        return 0

    def _session_held_full_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_full_tokens()
        return 0

    def _session_held_swa_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_swa_tokens()
        return 0

    def _session_held_req_count(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_req_count()
        return 0

    def _get_token_info(self: Scheduler):
        allocator = self.token_to_kv_pool_allocator
        available_size = allocator.available_size()
        # mixed-KV: allocator.size includes the shared HP-prefix pool.
        total_capacity = getattr(allocator, "size", self.max_total_num_tokens)
        evictable_size = self.tree_cache.evictable_size()
        num_used = total_capacity - (available_size + evictable_size)
        token_usage = num_used / total_capacity if total_capacity else 0.0
        return num_used, token_usage, available_size, evictable_size

    def _get_mamba_token_info(self: Scheduler):
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_pool.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_mamba_radix_cache else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size
        return (
            full_num_used,
            mamba_num_used,
            full_token_usage,
            mamba_usage,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        )

    def _get_swa_token_info(self: Scheduler):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        # mixed-KV full pool: the unified allocator's ``size`` includes the
        # shared HP-prefix pool, and ``full_available_size()`` counts its free
        # HP-prefix slots too, so the capacity must match (mirrors
        # ``_get_token_info``). For a plain full allocator ``.size`` equals
        # ``full_tokens_per_layer``, so this is a no-op for stock SWA models.
        full_allocator = getattr(
            self.token_to_kv_pool_allocator, "full_attn_allocator", None
        )
        full_capacity = getattr(
            full_allocator, "size", self.full_tokens_per_layer
        )
        full_num_used = full_capacity - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        # TEMPORARY diagnostic, own flag on purpose. It must NOT ride on
        # SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY: that check compares
        # available_size (quant + HP-prefix pooled slots) against
        # max_total_num_tokens (quant only), so on a mixed-KV pool it fires on
        # the first scheduler pass with an empty tree -- structurally
        # incompatible, unrelated to anything the tree does. Drop this block
        # once the HP-recent claim is verified.
        if os.environ.get("SGLANG_CHECK_MIXED_KV_SWA"):
            checker = getattr(self.tree_cache, "assert_mixed_kv_swa_invariants", None)
            if checker is not None:
                checker()
        # These two terms come from different sources of truth: available_size is
        # physical SWA slots held by the allocator, evictable_size is a TOKEN
        # count maintained by the radix tree (swa_radix_cache: `len(node.value)`).
        # They only agree while "one full-tier token owns one SWA slot" holds.
        # The mixed-KV quant path breaks that -- only HP-tier tokens own slots --
        # and the tree then reports the full tier's tokens as SWA-evictable, so
        # the sum exceeds capacity and this goes negative. That state is not
        # cosmetic: eviction is driven by these sizes, so the tree stops
        # reclaiming and the quant arena fills until alloc_quant fails.
        # It went unnoticed for a whole benchmark grid because nothing checked.
        if swa_num_used < 0:
            # Which side is wrong? The two candidates need opposite fixes, and
            # the overshoot magnitude cannot distinguish them (it only reflects
            # when the check happened to run). Recount from the allocator's own
            # state -- only on the failing path, so the hot path is untouched:
            #
            #   dups > 0                 -> a slot is in the free list twice, so
            #                               `available` is inflated: a FREE path
            #                               is releasing something twice.
            #   dups == 0 and held+avail
            #     == capacity            -> the allocator is self-consistent, so
            #                               `evictable` is inflated: the TREE is
            #                               counting tokens that own no slot.
            #   held+avail < capacity    -> slots orphaned (leaked): mapped over
            #                               without being freed.
            detail = ""
            try:
                pool = self.token_to_kv_pool_allocator
                swa_alloc = pool.swa_attn_allocator
                mapping = pool.full_to_swa_index_mapping
                free_all = torch.cat(
                    (swa_alloc.free_pages, swa_alloc.release_pages)
                )
                n_free = int(free_all.numel())
                n_uniq = int(torch.unique(free_all).numel())
                held = int((mapping > 0).sum().item())
                cap = int(self.swa_tokens_per_layer)
                detail = (
                    f" | RECOUNT: free_list={n_free} unique={n_uniq} "
                    f"dups={n_free - n_uniq} mapped_held={held} "
                    f"held+avail={held + n_free} capacity={cap} "
                    f"delta={held + n_free - cap} "
                    f"tree_evictable={swa_evictable_size}"
                )
            except Exception as e:  # diagnostic must never mask the assertion
                detail = f" | RECOUNT unavailable: {type(e).__name__}: {e}"
            raise AssertionError(
                "SWA accounting invariant violated: available + evictable "
                f"({swa_available_size} + {swa_evictable_size}) exceeds capacity "
                f"({self.swa_tokens_per_layer}) by {-swa_num_used}. The radix "
                "tree's token-based SWA count no longer matches slots actually "
                "held." + detail
            )
        full_token_usage = full_num_used / full_capacity if full_capacity else 0.0
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def _check_hybrid_memory(self: Scheduler):
        (
            full_num_used,
            swa_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = self._get_swa_token_info()
        session_held_full = self._session_held_full_tokens()
        session_held_swa = self._session_held_swa_tokens()

        # Streaming sessions hold tree locks during idle, so tree-protected
        # tokens must be accounted for alongside session-held tokens.
        full_protected = self.tree_cache.full_protected_size()
        swa_protected = self.tree_cache.swa_protected_size()
        full_leaked = full_num_used - full_protected - session_held_full
        swa_leaked = swa_num_used - swa_protected - session_held_swa
        memory_leak = full_leaked != 0 or swa_leaked != 0
        token_msg = (
            f"{full_leaked=}, {swa_leaked=}\n"
            f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {full_protected=}, {session_held_full=}\n"
            f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {swa_protected=}, {session_held_swa=}\n"
        )
        return memory_leak, token_msg

    def _check_mamba_memory(self: Scheduler):
        (
            full_num_used,
            mamba_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        ) = self._get_mamba_token_info()
        session_held = self._session_held_tokens()
        memory_leak = (
            full_num_used != self.tree_cache.full_protected_size() + session_held
            or mamba_num_used != self.tree_cache.mamba_protected_size()
        )
        if memory_leak:
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            free_mamba_pages = set(
                self.req_to_token_pool.mamba_pool.free_slots.tolist()
            )
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(self.req_to_token_pool.mamba_pool.size))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}, leaked_full_pages={leaked_full_pages if len(leaked_full_pages) > 0 else None}, leaked_mamba_pages={leaked_mamba_pages if len(leaked_mamba_pages) > 0 else None}\n"
            )
        else:
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}\n"
            )
        return memory_leak, token_msg

    def _check_radix_cache_memory(self: Scheduler):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        session_held = self._session_held_tokens()
        allocator = self.token_to_kv_pool_allocator
        total_capacity = getattr(allocator, "size", self.max_total_num_tokens)
        memory_leak = (available_size + evictable_size) != (
            total_capacity - protected_size - session_held
        )
        token_msg = (
            f"{total_capacity=}, {available_size=}, {evictable_size=}, "
            f"{protected_size=}, {session_held=}\n"
        )
        if memory_leak:
            kvc = self.token_to_kv_pool_allocator.get_kvcache()
            if getattr(kvc, "mixed_kv_enabled", lambda: False)():
                token_msg += self._mixed_kv_leak_debug_msg()
        return memory_leak, token_msg

    def _mixed_kv_leak_debug_msg(self: Scheduler) -> str:
        allocator = self.token_to_kv_pool_allocator

        def summarize_req(req, source: str) -> str:
            slack = getattr(req, "mixed_kv_quant_slack_indices", None)
            return (
                f"{source}:rid={req.rid},pool={req.req_pool_idx},"
                f"kv={req.kv_committed_len}/{req.kv_allocated_len},"
                f"freed={req.kv_committed_freed}/{req.kv_overallocated_freed},"
                f"chunked={req.is_chunked},retracted={req.is_retracted},"
                f"lens(origin/output/fill/prefix/protected)="
                f"{len(req.origin_input_ids)}/{len(req.output_ids)}/"
                f"{len(req.fill_ids)}/{len(req.prefix_indices)}/"
                f"{req.cache_protected_len},"
                f"slack={0 if slack is None else slack.numel()}"
            )

        summaries = []
        seen = set()

        def add_req(req, source: str):
            if req is None or id(req) in seen:
                return
            seen.add(id(req))
            if (
                req.req_pool_idx is not None
                or req.kv_committed_len
                or req.kv_allocated_len
                or getattr(req, "mixed_kv_quant_slack_indices", None) is not None
                and req.mixed_kv_quant_slack_indices.numel() > 0
            ):
                summaries.append(summarize_req(req, source))

        for source, batch in (
            ("running", getattr(self, "running_batch", None)),
            ("last", getattr(self, "last_batch", None)),
            ("cur", getattr(self, "cur_batch", None)),
        ):
            if batch is not None:
                for req in batch.reqs:
                    add_req(req, source)

        add_req(getattr(self, "chunked_req", None), "chunked")

        for req in getattr(self, "waiting_queue", []):
            add_req(req, "waiting")

        for i, item in enumerate(getattr(self, "result_queue", [])):
            batch = item[0]
            for req in batch.reqs:
                add_req(req, f"result{i}")

        owner_msg = "; ".join(summaries[:32])
        if len(summaries) > 32:
            owner_msg += f"; ... +{len(summaries) - 32} more"
        if not owner_msg:
            owner_msg = "none"

        return (
            "MIXED_KV_IDLE_LEAK "
            f"allocator=({allocator.debug_print()}), "
            f"free_pages={allocator.free_pages.numel()}, "
            f"release_pages={allocator.release_pages.numel()}, "
            f"tokens_in_use={getattr(allocator, '_tokens_in_use', 'n/a')}, "
            f"owners=[{owner_msg}]\n"
        )

    def _get_batch_uncached_size(self: Scheduler, batch: ScheduleBatch) -> int:
        ret = 0
        for req in batch.reqs:
            assert req.kv_committed_freed == req.kv_overallocated_freed
            uncached_len = 0
            if not req.kv_committed_freed:
                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0
                uncached_len = allocated_len - req.cache_protected_len

            ret += uncached_len

        return ret

    def self_check_during_busy(self: Scheduler):
        current_batch: ScheduleBatch = self.last_batch

        if current_batch is None:
            return

        spec_topk = self.server_args.speculative_eagle_topk or 1
        if spec_topk > 1:
            warnings.warn(
                "Runtime memory check (busy) is not supported when speculation topk > 1."
            )
            return

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()

        uncached_size = self._get_batch_uncached_size(current_batch)

        if (
            current_batch.forward_mode.is_extend()
            and self.running_batch is not None
            and not self.running_batch.is_empty()
        ):
            uncached_size += self._get_batch_uncached_size(self.running_batch)

        if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get() > 1:
            log_msg = f"[Mem Check (BUSY)] {available_size=}, {evictable_size=}, {protected_size=}, {uncached_size=}"
            logger.info(log_msg)

        session_held = self._session_held_tokens()
        total_tokens = (
            available_size
            + evictable_size
            + protected_size
            + uncached_size
            + session_held
        )
        assert (
            total_tokens == self.max_total_num_tokens
        ), f"Mem Leak Detected! {total_tokens=} vs {self.max_total_num_tokens=}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        session_req_count = self._session_held_req_count()
        if len(self.req_to_token_pool.free_slots) + session_req_count != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def check_memory(self: Scheduler):
        if self.is_hybrid_swa:
            memory_leak, token_msg = self._check_hybrid_memory()
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            memory_leak, token_msg = self._check_mamba_memory()
        else:
            memory_leak, token_msg = self._check_radix_cache_memory()

        if memory_leak:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_memory_leak_warnings",
                msg,
            )

        self._check_req_pool()

        if (
            self.current_scheduler_metrics_enabled
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid_swa:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            elif self.is_hybrid_ssm:
                (
                    num_used,
                    _,
                    token_usage,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_mamba_token_info()
            else:
                num_used, token_usage, _, _ = self._get_token_info()

            priority_enabled = self.enable_priority_scheduling
            self.stats.num_running_reqs = QueueCount.from_reqs(
                self.running_batch.reqs, priority_enabled
            )
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_bootstrap_queue.queue, priority_enabled
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_inflight_queue, priority_enabled
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_transfer_queue.queue, priority_enabled
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self: Scheduler):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()

    def self_check_during_idle(self: Scheduler):
        if self.enable_hisparse and self.hisparse_coordinator.has_ongoing_staging():
            return
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if len(self.disagg_prefill_inflight_queue) > 0:
                return
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)
            if queue_size:
                return

        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing or disable_request_logging():
            return ""
        if scheduler.is_hybrid_swa:
            _, info_msg = scheduler._check_hybrid_memory()
        elif scheduler.is_hybrid_ssm and scheduler.tree_cache.supports_mamba():
            _, info_msg = scheduler._check_mamba_memory()
        else:
            _, info_msg = scheduler._check_radix_cache_memory()
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n"
            f"{info_msg}"
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: getattr(scheduler, "forward_ct", 0),
        is_active=lambda: scheduler.is_initializing
        or getattr(scheduler, "cur_batch", None) is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
