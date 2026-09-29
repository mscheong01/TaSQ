"""Memory pool configurators for profiling and sizing KV cache pools.

Each model architecture has its own configurator that computes pool sizes
from available GPU memory using a unified coeff+bias model:

    available_bytes = max_tokens * coeff + bias
    max_tokens = (available_bytes - bias) / coeff

Two entry points, same core computation:
- calculate_pool_sizes(available_bytes, page_size): profiling path
- calculate_pool_sizes_from_max_tokens(max_tokens, page_size): constraint path
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import get_nsa_index_head_dim, is_deepseek_nsa
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.mem_cache.unified_kv_pool import (
    compute_page_geometry,
    resolve_hp_dtype,
    resolve_scale_dtype,
)
from sglang.srt.utils.common import is_float4_e2m1fn_x2


def _attention_supports_mixed_kv(server_args) -> bool:
    """Return True when the active attention-backend configuration can run
    the unified HP+int2 KV pool end-to-end.

    The pool's read-path (dequant + rotation + int2 decode) lives in the
    Triton backend today. The FA3 prefill backend gained a rotation-aware
    int2 prefill path in
    ``python/sglang/srt/layers/attention/quantized_kv_prefill.py``, so the
    hybrid ``prefill=fa3 + decode=triton`` configuration is supported too.
    Any other combination stays on the plain ``MHATokenToKVPool`` int2 path.

    ``prefill_attention_backend`` and ``decode_attention_backend`` can both
    be unset (``None``), in which case the model runner falls back to
    ``attention_backend`` for both roles (see
    ``ServerArgs.get_attention_backends``). Apply the same resolution here
    so partially-specified configurations (e.g. only ``--attention-backend
    triton``, or only ``--prefill-attention-backend fa3`` with
    ``--attention-backend triton``) are recognized correctly.
    """
    ab = getattr(server_args, "attention_backend", None)
    pab = getattr(server_args, "prefill_attention_backend", None) or ab
    dab = getattr(server_args, "decode_attention_backend", None) or ab
    if pab == "triton" and dab == "triton":
        return True
    if pab == "fa3" and dab == "triton":
        return True
    return False


def _nsn_packed_cell_bytes(mr, model_config, num_layers, tp_size):
    """Bytes per token for the NSN packed store, or None when it is not active.

    1.2383 b/ch per side: 1.0 index + 0.125 norm2
    + 0.0312 norm + 0.0625 window mean + 0.0156 mean scale/min + 0.0039 norm scale/min.
    Both sides, all layers. The raw bf16 ring tier is NOT charged here -- it is a fixed
    allocation that does not scale with the token budget; `_nsn_hp_tier_bytes` takes it off
    the available memory instead (calculate_pool_sizes).
    """
    import os

    from sglang.srt.environ import envs

    if os.environ.get("SGLANG_NSN_PACKED") != "1" or not envs.SGLANG_NSN_PATH.get():
        return None
    bits_per_channel_per_side = 1.2383
    channels = model_config.get_num_kv_heads(tp_size) * model_config.head_dim
    return int(2 * channels * bits_per_channel_per_side / 8 * num_layers)


def _nsn_hp_tier_bytes(mr, model_config, num_layers, tp_size):
    """Fixed bytes of the packed arm's bf16 ring tier (K and V, all layers), or 0."""
    if _nsn_packed_cell_bytes(mr, model_config, num_layers, tp_size) is None:
        return 0
    import os

    from sglang.srt.mem_cache import nsn_hp_pool as _HPP

    n_pre = int(os.environ.get("SGLANG_NSN_HP_PREASSIGN", "128"))   # same default as nsn_hp_pool
    rows = n_pre * _HPP.rows_per_request() + 1
    kvh = model_config.get_num_kv_heads(tp_size)
    D = model_config.head_dim
    tier = rows * kvh * D * 2 * 2 * num_layers
    # window-metadata arena (nsn_packed_store.LayerStore: nsc, mk_q/mv_q, mk_s/mv_s), also fixed:
    # n_pre * MAXW windows per layer
    maxw = int(os.environ.get("SGLANG_NSN_PACKED_MAXWIN_PER_REQ", "64"))
    # Windows are position-aligned at NSNQuant's defining window_size of 64 (nsn_quant.py's
    # header), so a request served at context C occupies ceil(C / 64) of them and MAXW below
    # that silently ALIASES the tail onto the last window instead of failing -- see the guard in
    # nsn_quant_fused._packed_layer for what that does to the metadata. Checked here as well
    # because this runs at boot, before a single token is served, and because the value reaching
    # the server is computed in shell (run_sweep.sh: (NSN_CTX + 63) / 64), where an unset
    # variable yields 0 rather than an error.
    _ctx = getattr(model_config, "context_len", None)
    if _ctx and maxw * 64 < int(_ctx):
        raise RuntimeError(
            f"SGLANG_NSN_PACKED_MAXWIN_PER_REQ={maxw} covers {maxw * 64} tokens per request, but "
            f"this server is configured for a context of {int(_ctx)}. The packed NSN arm would "
            f"alias every position past {maxw * 64} onto its last window and serve corrupted "
            f"means/scales. Set it to at least {-(-int(_ctx) // 64)}.")
    per_win = kvh * (4 * 2 + 2 * (D // 2) + 2 * (D // 32) * 2 * 2)
    arena = (n_pre * maxw + 1) * per_win * num_layers
    # The fused hook's per-layer state is charged separately, by _nsn_fused_state_bytes: it is
    # not packed-only, because the allocation is gated on FUSED rather than PACKED.
    return tier + arena


def _nsn_fused_state_bytes(model_config, num_layers, tp_size, dtype_bytes=2):
    """Fixed bytes of the fused hook's per-layer state (nsn_quant_fused._LayerState), or 0.

    Account for this state whenever the fused hook is enabled, independently of whether packed
    storage is also enabled.
    """
    import os

    from sglang.srt.environ import envs   # function-scoped, as everywhere else in this file

    if not envs.SGLANG_NSN_PATH.get() or os.environ.get("SGLANG_NSN_FUSED", "1") != "1":
        return 0
    kvh = model_config.get_num_kv_heads(tp_size)
    D = model_config.head_dim
    slots = int(os.environ.get("SGLANG_NSN_FUSED_MAX_SLOTS", "512"))
    ws = 64          # bundle["window_size"]; not settable by env, 64 for every bundle here
    # RECENT has the same MIXED_KV fallback the runtime uses (nsn_quant), or the charge and the
    # allocation disagree by ~50 MB for a run configured through the shared variable.
    recent = int(os.environ.get("SGLANG_NSN_RECENT_TOKENS",
                                os.environ.get("SGLANG_MIXED_KV_RECENT_TOKENS", "64")))
    nfull = int(os.environ.get("SGLANG_NSN_NFULL_PAD", "32"))
    cap = nfull * ws + recent + 2 * ws + 64
    # loc + pos int64 rings; start/cnt/ev int64 scalars; enc bool
    rings = 2 * slots * cap * 8 + 3 * slots * 8 + slots
    def _fields(n):
        return n * kvh * (
            ws * (D // 8)                  # idx    u8
            + ws * 2                       # norm2  f16
            + ws                           # nq     u8
            + 2 * dtype_bytes              # nsc, nmin  (cache dtype)
            + D                            # mq     u8
            + 2 * (D // 32) * dtype_bytes  # msc, mmin  (cache dtype)
        )

    state = (rings + 2 * _fields(slots + 1)) * num_layers

    # The encode scratch, which is NOT per layer: one set per server, keyed by window count.
    # Prefill step (b) calls scratch_cache(nb) and stage1_scratch(nb) for every nb up to
    # NFULL_PAD, and (nb+1) for the V dict, so the whole range is resident once a long prompt
    # has gone through; decode's counts are a subset of it. Uncharged, this was ~300 MB the
    # pool grew into -- which is why qwen3_4b_think NIAH at 65536 booted with 5.02 GB free and
    # then died asking for 322 MiB mid-prefill, while the int2 arms came up at the same
    # mem-fraction (2026-09-11).
    nfull = int(os.environ.get("SGLANG_NSN_NFULL_PAD", "32"))
    scratch = 0
    for n in range(1, nfull + 2):
        # stage1_scratch: xn and xh are [2, n, H, ws, D] bf16; n2p is [2, n, H, ws] fp32
        scratch += 2 * n * kvh * ws * D * 2 * 2 + 2 * n * kvh * ws * 4
        scratch += _fields(n)                                      # scratch_cache
    return state + scratch


@dataclass
class MemoryPoolConfig:
    """Resolved memory pool config, shared between target and draft workers."""

    max_total_num_tokens: int
    max_running_requests: Optional[int] = None
    full_max_total_num_tokens: Optional[int] = None
    swa_max_total_num_tokens: Optional[int] = None

    mem_fraction_static: Optional[float] = None

    def __post_init__(self):
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _resolve_quant_group_count(head_dim: int, group_size: Optional[int]) -> int:
    effective_group_size = head_dim if group_size is None else group_size
    if effective_group_size <= 0:
        raise ValueError(
            f"kv_cache_quant_group_size must be positive, got {effective_group_size}"
        )
    if head_dim % effective_group_size != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by "
            f"kv_cache_quant_group_size ({effective_group_size})"
        )
    return head_dim // effective_group_size


def _get_int_kv_bytes_per_head_pair(
    k_head_dim: int,
    v_head_dim: int,
    kv_cache_dtype: str,
    group_size: Optional[int],
    scale_dtype_bytes: int = 4,
) -> int:
    """Bytes per *quant-token* per head-pair for the int2 KV cache.

    Used by both the non-mixed path and the mixed path: the scheduler's
    ``max_total_num_tokens`` is denominated in quant tokens (= slot ids on the
    int2 tier), and the mixed allocator's ``size`` = ``(num_pages - 1) * N_Q``
    is also in quant tokens, so the leak check
    ``size - available - evictable - protected`` closes in a single unit.
    """
    assert kv_cache_dtype == "int2", (
        f"Only int2 quant KV is supported, got {kv_cache_dtype}"
    )
    pack_factor = 4
    k_groups = _resolve_quant_group_count(k_head_dim, group_size)
    v_groups = _resolve_quant_group_count(v_head_dim, group_size)
    packed_k_bytes = k_head_dim // pack_factor
    packed_v_bytes = v_head_dim // pack_factor
    # Interleaved (scale, zero) per group in ``scale_dtype``.
    scales_zeros_bytes = 2 * scale_dtype_bytes * (k_groups + v_groups)
    return packed_k_bytes + packed_v_bytes + scales_zeros_bytes


def _get_unified_mixed_kv_bytes_per_quant_token(
    k_head_dim: int,
    v_head_dim: int,
    hp_dtype_bytes: int,
    scale_dtype_bytes: int,
    k_groups: int,
    v_groups: int,
    n_q: int,
) -> int:
    """Shared-arena bytes per *quant token* in the unified pool.

    A page of the shared K/V arena is sized to host either 1 HP token or N_Q
    quant tokens; its byte size is therefore ``(head_dim + v_head_dim) *
    hp_dtype_bytes``. Per quant token that is this over ``N_Q``. Scales/zeros
    live in a parallel arena (one entry per quant slot) and are included
    here so the scheduler's cell-size matches the actual allocator size.
    """
    arena_bytes_per_page = (k_head_dim + v_head_dim) * hp_dtype_bytes
    arena_bytes_per_quant_token = arena_bytes_per_page // n_q
    scales_zeros_bytes_per_quant_token = (
        2 * scale_dtype_bytes * (k_groups + v_groups)
    )
    return arena_bytes_per_quant_token + scales_zeros_bytes_per_quant_token


class MemoryPoolConfigurator:
    """Base class for memory pool configurators.

    Subclasses compute pool sizes for their architecture via coeff+bias model.
    Both entry points return MemoryPoolConfig (with max_running_requests=None,
    to be filled by the consumer).
    """

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        """Profiling path: compute pool sizes from available bytes."""
        raise NotImplementedError

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Constraint path: recalculate pool sizes from a constrained max_tokens."""
        raise NotImplementedError


class DefaultPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for standard models: MHA, MLA, NSA, FP4.

    coeff = cell_size (bytes per token across all layers)
    bias = 0
    """

    def __init__(self, mr: ModelRunner):
        # Determine effective number of layers for KV cache
        if mambaish := mr.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if mr.start_layer <= i < mr.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = mr.num_effective_layers

        self._cell_size = self._compute_cell_size(mr, num_layers)
        _tp = get_attention_tp_size()
        self._nsn_fixed_bytes = _nsn_hp_tier_bytes(
            mr, mr.model_config, num_layers, _tp
        ) + _nsn_fused_state_bytes(mr.model_config, num_layers, _tp)

        # DFLASH: scale cell_size to account for draft model KV cache
        if mr.spec_algorithm.is_dflash() and not mr.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                scale_kv_cell_size_per_token_for_dflash,
            )

            draft_num_layers = getattr(mr, "dflash_draft_num_layers", None)
            if (
                draft_num_layers is not None
                and int(draft_num_layers) > 0
                and int(num_layers) > 0
            ):
                self._cell_size = scale_kv_cell_size_per_token_for_dflash(
                    target_cell_size_per_token=self._cell_size,
                    target_num_layers=int(num_layers),
                    draft_num_layers=int(draft_num_layers),
                )

    def _compute_cell_size(self, mr: ModelRunner, num_layers: int) -> int:
        """Compute per-token KV cache cost in bytes. Subclasses can override."""
        # args to config cell size
        from sglang.srt.environ import envs

        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype
        kv_quant_group_size = getattr(
            mr.server_args, "kv_cache_quant_group_size", None
        )
        tp_size = get_attention_tp_size()

        # Mixed HP+quant pool: the shared arena bytes per logical token are
        # ``(head_dim + v_head_dim) * hp_dtype_bytes`` (== 1 HP token worth).
        # Plus the scales/zeros arena sized for every page's quant view.
        enable_mixed_kv = (
            kv_cache_dtype == "int2"
            and envs.SGLANG_ENABLE_MIXED_KV_WINDOWS.get()
            and _attention_supports_mixed_kv(mr.server_args)
            and not mr.is_hybrid_swa
            and mr.server_args.disaggregation_mode in (None, "null")
            and mr.server_args.speculative_algorithm is None
        )

        if kv_cache_dtype == "int2":
            scale_dtype = resolve_scale_dtype(envs.SGLANG_MIXED_KV_SCALE_DTYPE.get())
            scale_bytes = torch.empty(0, dtype=scale_dtype).element_size()
        else:
            scale_bytes = None

        if enable_mixed_kv:
            hp_dtype = resolve_hp_dtype(envs.SGLANG_MIXED_KV_HP_DTYPE.get())
            hp_dtype_bytes = torch.empty(0, dtype=hp_dtype).element_size()
            _, n_q = compute_page_geometry(hp_dtype)
            k_groups = _resolve_quant_group_count(
                model_config.head_dim, kv_quant_group_size
            )
            v_groups = _resolve_quant_group_count(
                model_config.v_head_dim, kv_quant_group_size
            )
            # max_total_num_tokens is denominated in *quant tokens* (slot ids
            # on the int2 tier). This matches the unified allocator's
            # scheduler-facing ``size = (num_pages - 1) * N_Q``.
            bytes_per_head = _get_unified_mixed_kv_bytes_per_quant_token(
                model_config.head_dim,
                model_config.v_head_dim,
                hp_dtype_bytes,
                scale_bytes,
                k_groups,
                v_groups,
                n_q,
            )
            # NOTE: vq2 needs NO adjustment here, and adding one is a bug.
            # ``_get_unified_mixed_kv_bytes_per_quant_token`` already charges
            # K *and* V -- its ``(head_dim + v_head_dim) * hp_bytes // n_q``
            # term is the packed int2 K+V cost written as a page-sharing
            # identity. In vq2 mode ``_create_arenas`` allocates the uint8 VQ
            # index arena in *place of* the int2 K arena (not in addition to
            # it), and at G=4 both cost NG = head_dim/4 = 32 B/(token, head).
            # A previous ``bytes_per_head += head_dim // 4`` double-counted K:
            # it charged 112 B where 80 B is allocated, so the pool was sized
            # at 80/112 = 0.714 of the budget and ~15 GB of HBM went unused.
            kv_size = None
        elif kv_cache_dtype == "int2":
            bytes_per_head = _get_int_kv_bytes_per_head_pair(
                model_config.head_dim,
                model_config.v_head_dim,
                kv_cache_dtype,
                kv_quant_group_size,
                scale_bytes,
            )
            kv_size = None
        else:
            kv_size = torch._utils._element_size(kv_cache_dtype)
            bytes_per_head = None

        if mr.use_mla_backend:
            cell_size = (
                (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                * num_layers
                * kv_size
            )
            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                        // scale_block_size
                    )
                    * num_layers
                    * kv_size
                )

            # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
            if is_deepseek_nsa(model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            if bytes_per_head is not None:
                cell_size = (
                    model_config.get_num_kv_heads(tp_size) * bytes_per_head * num_layers
                )
            elif _nsn_packed_cell_bytes(mr, model_config, num_layers, tp_size) is not None:
                # SGLANG_NSN_PACKED: the token budget is set by the PACKED store, not by a bf16
                # pool. Without this branch it would fall through to the BF16 sizing expression.
                cell_size = _nsn_packed_cell_bytes(
                    mr, model_config, num_layers, tp_size
                )
            else:
                cell_size = (
                    model_config.get_num_kv_heads(tp_size)
                    * (model_config.head_dim + model_config.v_head_dim)
                    * num_layers
                    * kv_size
                )

            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                n = model_config.get_num_kv_heads(tp_size)
                k = model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * num_layers * 2 * kv_size) // scale_block_size
                )

        return cell_size

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        # SGLANG_NSN_PACKED: the bf16 ring tier is allocated alongside the packed store and
        # does not scale with tokens, so it comes off the top before the per-token division.
        available_bytes = max(available_bytes - getattr(self, "_nsn_fixed_bytes", 0), 0)
        max_total_num_tokens = available_bytes // self._cell_size
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)


class HybridSWAPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for hybrid sliding window attention models (Gemma2, Command-R, MiMo).

    Splits available memory between full attention and SWA pools.
    Does NOT inherit DefaultPoolConfigurator — different coeff model.
    """

    def __init__(self, mr: ModelRunner):
        # Local import mirrors DefaultPoolConfigurator (envs is function-scoped
        # in this module, not module-level).
        from sglang.srt.environ import envs

        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype
        kv_quant_group_size = getattr(
            mr.server_args, "kv_cache_quant_group_size", None
        )
        tp_size = get_attention_tp_size()

        if (
            kv_cache_dtype == "int2"
            and kv_quant_group_size is not None
        ):
            raise ValueError(
                "--kv-cache-quant-group-size is only supported for the "
                "full-attention Triton int2 KV cache path and is not supported "
                "with hybrid SWA models"
            )

        enable_mixed_kv = (
            kv_cache_dtype == "int2"
            and envs.SGLANG_ENABLE_MIXED_KV_WINDOWS.get()
            and _attention_supports_mixed_kv(mr.server_args)
            and mr.server_args.disaggregation_mode in (None, "null")
            and mr.server_args.speculative_algorithm is None
        )

        if kv_cache_dtype == "int2":
            # Hybrid int2 quantizes only the full-attention pool; SWA layers
            # stay in the model dtype (see model_runner_kv_cache_mixin), so
            # size them at model-dtype bytes.
            if enable_mixed_kv:
                scale_dtype = resolve_scale_dtype(
                    envs.SGLANG_MIXED_KV_SCALE_DTYPE.get()
                )
                scale_bytes = torch.empty(0, dtype=scale_dtype).element_size()
                hp_dtype = resolve_hp_dtype(envs.SGLANG_MIXED_KV_HP_DTYPE.get())
                hp_dtype_bytes = torch.empty(0, dtype=hp_dtype).element_size()
                _, n_q = compute_page_geometry(hp_dtype)
                k_groups = _resolve_quant_group_count(
                    model_config.head_dim, kv_quant_group_size
                )
                v_groups = _resolve_quant_group_count(
                    model_config.v_head_dim, kv_quant_group_size
                )
                bytes_per_head = _get_unified_mixed_kv_bytes_per_quant_token(
                    model_config.head_dim,
                    model_config.v_head_dim,
                    hp_dtype_bytes,
                    scale_bytes,
                    k_groups,
                    v_groups,
                    n_q,
                )
                # Same double-count as the non-hybrid path (see the NOTE above
                # _get_unified_mixed_kv_bytes_per_quant_token's caller): the
                # base term already charges K+V, and _create_arenas allocates
                # the VQ index arena *in place of* the int2 K arena. Charging
                # head_dim//4 again undersized the full tier -- on gpt-oss it
                # cost vq2 5.95% of its pool (2,166,760 vs int2's 2,303,896)
                # and left 3.04 GB of HBM unused.
                full_per_token = (
                    model_config.get_num_kv_heads(tp_size) * bytes_per_head
                )
            else:
                full_per_token = model_config.get_num_kv_heads(tp_size) * (
                    _get_int_kv_bytes_per_head_pair(
                        model_config.head_dim,
                        model_config.v_head_dim,
                        kv_cache_dtype,
                        kv_quant_group_size,
                    )
                )
            swa_elt_size = torch._utils._element_size(mr.dtype)
            swa_per_token = (
                model_config.get_swa_num_kv_heads(tp_size)
                * (model_config.swa_head_dim + model_config.swa_v_head_dim)
                * swa_elt_size
            )
            kv_size = None
        else:
            kv_size = torch._utils._element_size(kv_cache_dtype)
            full_per_token = (
                model_config.get_num_kv_heads(tp_size)
                * (model_config.head_dim + model_config.v_head_dim)
                * kv_size
            )
            swa_per_token = (
                model_config.get_swa_num_kv_heads(tp_size)
                * (model_config.swa_head_dim + model_config.swa_v_head_dim)
                * kv_size
            )

        self._full_layers_num = len(model_config.full_attention_layer_ids)
        self._swa_layers_num = len(model_config.swa_attention_layer_ids)
        assert (
            self._swa_layers_num > 0
        ), "Hybrid SWA model must have at least one SWA layer"

        self._swa_full_tokens_ratio = mr.server_args.swa_full_tokens_ratio
        # Absolute SWA-tier size, overriding both the derived size and the
        # ratio when > 0. See envs.SGLANG_SWA_POOL_TOKENS.
        self._swa_fixed_tokens = envs.SGLANG_SWA_POOL_TOKENS.get()

        # Only needed by _check_swa_admission_floor.
        self._context_len = model_config.context_len

        # Full layer per-token memory (bytes)
        self._full_per_token = full_per_token

        # SWA layer per-token memory (bytes)
        self._swa_per_token = swa_per_token

        # Bytes per token of max_total_num_tokens.
        #
        # Hybrid (full_layers > 0): max_total = full_tokens, so cell_size accounts
        # for both pools: F*nf + r*S*ns (where swa_tokens = full_tokens * r).
        #
        # All-SWA (full_layers == 0): max_total = swa_tokens directly. The ratio
        # is meaningless here -- there is no full pool to relate to, and every
        # token beyond the sliding window can be evicted. So cell_size = S*ns,
        # with no ratio factor applied.
        if self._full_layers_num == 0:
            self._cell_size = self._swa_per_token * self._swa_layers_num
        else:
            self._cell_size = (
                self._full_per_token * self._full_layers_num
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * self._swa_layers_num
            )

    # A derived default lived here and was REMOVED, 2026-07-31. It sized the
    # tier at context_len + chunked_prefill_size + R*(2*window + hp_recent_ring)
    # on the reasoning that ``inc_lock_ref`` stops taking swa_lock_ref after
    # ``sliding_window_size``, so a cached prefix pins only its last window and
    # everything older is tombstonable.
    #
    # The premise is true; the conclusion is false. swa_lock_ref governs whether
    # ``evict`` may RECLAIM a node, not whether its slots are HELD. Under the
    # STOCK evictor, retaining a cached prefix in a matchable state costs one
    # SWA slot per token of it, because the evictor deletes the prefix's
    # trailing-window leaf under pressure and the match then collapses to zero
    # rather than shortening. Measured on gpt-oss at 90k: a 144,560-token tier
    # dropped cached_tokens_median to 0 at bs=4 (bf16 and quant alike) and
    # admitted only 6 of 71 requests at bs=71. SGLANG_SWA_KEEP_PREFIX_TAIL
    # (swa_radix_cache.py) fixes the evictor and brings the per-prefix cost
    # down to ~(window + ring); with it a small pinned tier is correct, but the
    # derived default here was never re-landed -- sizing stays pin-or-ratio.
    #
    # So the tier cannot be sized from liveness while the radix tree accounts
    # SWA in tokens. The real fix is to stop allocating SWA for tokens the
    # sliding layers can never read. Until then SGLANG_SWA_POOL_TOKENS stays
    # an explicit knob and the ratio stays the default.

    def _pinned_swa_tokens(self, page_size: int) -> int | None:
        """Page-aligned SGLANG_SWA_POOL_TOKENS, or None if unset.

        Not applied on the all-SWA path: there is no full tier to trade against,
        so the pool *is* the SWA tier and pinning it would just discard memory.
        """
        if self._swa_fixed_tokens <= 0 or self._full_layers_num == 0:
            return None
        return max(page_size, (self._swa_fixed_tokens // page_size) * page_size)

    def _resolve_swa_tokens(self, page_size: int) -> tuple[int | None, str]:
        """SWA-tier size taken off the top of the budget, and what chose it.

        Precedence: explicit SGLANG_SWA_POOL_TOKENS > ratio (returns None, and
        the caller applies swa_full_tokens_ratio to the full tier as upstream
        does). A derived middle tier was tried and removed -- see the note above
        ``_pinned_swa_tokens``.
        """
        pinned = self._pinned_swa_tokens(page_size)
        if pinned is not None:
            return pinned, f"fixed={self._swa_fixed_tokens}"
        return None, f"ratio={self._swa_full_tokens_ratio}"

    def _check_swa_admission_floor(
        self, swa_tokens: int, full_tokens: int, sized_by: str
    ) -> None:
        """Refuse a tier that cannot admit a ``context_len`` prompt.

        Without this the server boots happily and then rejects every long
        request at ``validate_input_length`` -- or, under
        ``--allow-auto-truncate``, quietly shortens it while the caller still
        believes it sent the full prompt. Both are far harder to diagnose than a
        boot error, and neither is affected by chunked prefill: the length check
        runs on the whole prompt before any chunking.

        Scoped narrowly on purpose:

        * only when this code chose the size -- the stock ratio path keeps
          upstream's behaviour of clamping ``max_req_input_len`` to the pool;
        * only against the SWA tier. A *full* tier below ``context_len`` is
          upstream's long-standing "small pool, long context" configuration and
          is not something this change introduces, so it is left alone.
        """
        if sized_by.startswith("ratio="):
            return
        if swa_tokens >= self._context_len or full_tokens < self._context_len:
            return
        raise ValueError(
            f"sliding-window KV tier ({swa_tokens} tokens, {sized_by}) caps "
            f"max_req_input_len at {swa_tokens - 6} even though the full tier "
            f"holds {full_tokens}, below --context-length {self._context_len}. "
            f"Requests longer than that are rejected at admission, before "
            f"chunked prefill sees them. Raise SGLANG_SWA_POOL_TOKENS / "
            f"--mem-fraction-static, or lower --context-length."
        )

    def _solve_pool_sizes(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Core computation: split max_total_num_tokens into full/swa pool sizes."""

        def align_page_size(x: int) -> int:
            return (x // page_size) * page_size

        if self._full_layers_num == 0:
            # All-SWA: no full pool, max_total = actual SWA pool size.
            # Ratio is not applied -- see __init__ comment.
            swa_tokens = align_page_size(max_total_num_tokens)
            logger.info(
                f"Use sliding window memory pool (all SWA). "
                f"swa_layer_tokens={swa_tokens}"
            )
            return MemoryPoolConfig(
                max_total_num_tokens=swa_tokens,
                full_max_total_num_tokens=0,
                swa_max_total_num_tokens=swa_tokens,
            )

        # Hybrid: full_tokens = max_total_num_tokens, swa_tokens = full_tokens * ratio
        full_tokens = align_page_size(max_total_num_tokens)
        swa_tokens, sized_by = self._resolve_swa_tokens(page_size)
        if swa_tokens is None:
            swa_tokens = align_page_size(
                int(full_tokens * self._swa_full_tokens_ratio)
            )
        self._check_swa_admission_floor(swa_tokens, full_tokens, sized_by)

        full_gib = full_tokens * self._full_per_token * self._full_layers_num / (1 << 30)
        swa_gib = swa_tokens * self._swa_per_token * self._swa_layers_num / (1 << 30)
        logger.info(
            f"Use sliding window memory pool. "
            f"full_layer_tokens={full_tokens} ({full_gib:.2f} GiB), "
            f"swa_layer_tokens={swa_tokens} ({swa_gib:.2f} GiB, {sized_by})"
        )

        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        swa_tokens, sized_by = self._resolve_swa_tokens(page_size)
        if swa_tokens is None:
            max_total_num_tokens = int(available_bytes // self._cell_size)
            return self._solve_pool_sizes(max_total_num_tokens, page_size)

        # The SWA tier is an absolute size, so it comes off the budget first and
        # every remaining byte goes to the full tier -- rather than both tiers
        # scaling together through _cell_size's ratio term.
        swa_bytes = swa_tokens * self._swa_per_token * self._swa_layers_num
        full_bytes = available_bytes - swa_bytes
        if full_bytes <= 0:
            raise ValueError(
                f"the sliding-window tier ({sized_by}) needs "
                f"{swa_bytes / (1 << 30):.2f} GiB but only "
                f"{available_bytes / (1 << 30):.2f} GiB is available for KV; "
                f"lower SGLANG_SWA_POOL_TOKENS / --context-length / "
                f"--max-running-requests, or raise --mem-fraction-static"
            )
        max_total_num_tokens = int(
            full_bytes // (self._full_per_token * self._full_layers_num)
        )
        return self._solve_pool_sizes(max_total_num_tokens, page_size)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._solve_pool_sizes(max_total_num_tokens, page_size)


def create_memory_pool_configurator(
    mr: ModelRunner,
) -> MemoryPoolConfigurator:
    """Factory: select the right configurator for the model architecture."""
    if mr.is_hybrid_swa:
        return HybridSWAPoolConfigurator(mr)
    # Future: MambaPoolConfigurator
    return DefaultPoolConfigurator(mr)
