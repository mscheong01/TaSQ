import os
import subprocess
import warnings
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from typing import Any


@contextmanager
def temp_set_env(*, allow_sglang: bool = False, **env_vars: Any):
    """Temporarily set environment variables, restoring originals on exit.

    By default, SGLANG_*/SGL_* keys are rejected — use ``Envs`` descriptors
    for those.  Pass ``allow_sglang=True`` only for special env vars that
    intentionally bypass ``environ.py``.
    """
    if not allow_sglang:
        for key in env_vars:
            if key.startswith("SGLANG_") or key.startswith("SGL_"):
                raise ValueError("temp_set_env should not be used for sglang env vars")

    backup = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self.default

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class Envs:
    # fmt: off

    # Model & File Download
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    SGLANG_SORT_WEIGHT_FILES = EnvBool(False)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())

    # Logging Options
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_DISABLE_REQUEST_LOGGING = EnvBool(False)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)

    # SGLang CI
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    SGLANG_CUDA_COREDUMP_DIR = EnvStr("/tmp/sglang_cuda_coredumps")
    SGLANG_TEST_MAX_RETRY = EnvInt(None)

    # Constrained Decoding (Grammar)
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)


    # Test & Debug
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    IS_H200 = EnvBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(False)
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("multinomial")
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)
    SGLANG_ENABLE_MIXED_KV_WINDOWS = EnvBool(False)
    SGLANG_MIXED_KV_PREFIX_TOKENS = EnvInt(32)
    SGLANG_MIXED_KV_RECENT_TOKENS = EnvInt(128)
    SGLANG_MIXED_KV_HP_DTYPE = EnvStr("bfloat16")
    SGLANG_MIXED_KV_SCALE_DTYPE = EnvStr("float32")
    # Shared HP-prefix pool size (in HP slot units; rounded up to N_Q).
    # 0 = use the default of ``max_running_requests * P * 16``.
    SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS = EnvInt(0)
    # Absolute size of the sliding-window KV tier, in tokens. 0 = derive it
    # (HybridSWAPoolConfigurator._derived_swa_tokens) on a mixed-KV pool, or
    # fall back to --swa-full-tokens-ratio as upstream does.
    #
    # The ratio scales the SWA tier with the FULL tier's token count, a quantity
    # the sliding window has no relation to: SWA layers read at most
    # `sliding_window` tokens per request, so their need is an absolute number.
    # The mismatch is severe once the full tier is quantized, and self-defeating
    # -- compressing the full tier fits more full-tier tokens, so 0.8*full GROWS
    # the SWA tier and eats the saving. On gpt-oss-20b (12 bf16 sliding layers at
    # 24,576 B/token) the default 0.8 gave the SWA tier 24.1 GiB of a 27.9 GiB
    # pool to hold ~0.4 GiB of live state, leaving the quantized full tier --
    # the one the batch size actually depends on -- 3.8 GiB.
    #
    # Precedence: this pin > --swa-full-tokens-ratio. Whatever the source, the
    # tier must be >= --context-length or long prompts are rejected at admission
    # (tp_worker caps max_req_input_len at min(swa, full) - 6), and with
    # SGLANG_SWA_KEEP_PREFIX_TAIL it should also cover one in-flight prefill
    # chunk plus ~(window + ring) per concurrent request.
    SGLANG_SWA_POOL_TOKENS = EnvInt(0)
    # Keep every cached prefix matchable under SWA eviction pressure. A prefix
    # of length P only needs live SWA for its trailing window (generation reads
    # SWA positions [P-window, P); ``_match_prefix_helper`` accepts a match only
    # while >= sliding_window_size non-tombstoned tokens follow the last
    # tombstone). But ``evict``'s SWA branch tombstones internal nodes root-first
    # and DELETES leaves -- the trailing node dies last, and with it the whole
    # match (cached_tokens 89,728 -> 0, not -> 89,465). With this on, the SWA
    # evictor runs a protected pass first: internal nodes whose live run below
    # would drop under keep_target are skipped; leaves are TRIMMED to
    # keep_target = sliding_window_size + 2*page_size (tombstone the head, keep
    # the tail; the +2 pages cover the match key ending up to two pages before
    # the insert end) and, once short, are never deleted. If the protected pass
    # misses its reclaim target, an unprotected fallback pass restores the old
    # delete behavior so allocation cannot fail -- it firing means the tier is
    # genuinely undersized for even one window per cached prefix.
    # Known hole, accepted: the FULL branch of ``evict`` can still delete a
    # protected tail under full-tier pressure; irrelevant while the full tier is
    # orders of magnitude larger than the SWA tier.
    SGLANG_SWA_KEEP_PREFIX_TAIL = EnvBool(False)
    # Diagnostic: log what each SWA eviction pass actually did (internal
    # tombstones / leaf trims / leaf deletions / tail skips / fallback work)
    # rather than inferring it from aggregate cache-hit numbers. Two retention
    # models have fit the aggregates and then failed a prediction; this records
    # the primitive actions instead.
    SGLANG_SWA_EVICT_TRACE = EnvBool(False)
    # Max trace reports per server. The old hard-coded cap of 40 expired midway
    # through the cold pass of the run that mattered, hiding the evidence.
    SGLANG_SWA_EVICT_TRACE_CAP = EnvInt(400)
    # Oscar rotation + per-row clip for int2 KV cache. Learned per-layer
    # orthogonal matrices loaded from K/V rotation checkpoints.
    SGLANG_OSCAR_K_ROTATION_PATH = EnvStr("")
    SGLANG_OSCAR_V_ROTATION_PATH = EnvStr("")
    SGLANG_OSCAR_K_CLIP_RATIO = EnvFloat(0.0)
    SGLANG_OSCAR_V_CLIP_RATIO = EnvFloat(0.0)
    SGLANG_OSCAR_ABSORB_V_ROTATION = EnvBool(False)
    # Fuse oscar K-rotation (rows @ R_k) into the prefill clip+quantize+pack
    # kernel. Eliminates the separate bf16 GEMM staging and the intermediate
    # rotated-K tensor for the quant pack. Requires oscar mode, V-rotation
    # absorbed (so V skips rotation), per-row scale (single-scale int2), and
    # at least one of K/V clip ratios > 0. Off by default; safe to leave off.
    SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT = EnvBool(False)
    # Use Lloyd-Max MSE-optimal buckets for INT2 KV quantization instead of
    # the default uniform min-max. Applies only to single-scale pretransformed
    # clip kernels (num_groups == 1). Requires oscar rotation + clip enabled.
    SGLANG_LLOYD_MAX = EnvBool(False)
    # V-tier integer level count for the OSCAR min/max affine SQ path (levels =
    # MAX_Q + 1). Default 3 = INT2 (the official Nova/OSCAR serving setting).
    # 1 = the 1-bit variant of the same quantizer (2 levels {min, max}) used by
    # the Nova-KV-1b fair-bits rebuild: storage stays int2 crumbs physically
    # (values only use {0,1}), so this is a simulated bit-width in the same
    # sense as the NSN plain-pool port; nominal V accounting = 1.0 + scale/zero.
    SGLANG_V_INT_MAX_Q = EnvInt(3)
    # Physical bits per V coordinate in the quantized V arena: 2 (default, int2
    # crumbs -- 4 coords/byte) or 1 (int1 -- 8 coords/byte). This is STORAGE
    # width, orthogonal to SGLANG_V_INT_MAX_Q's level count, and setting it to 1
    # requires MAX_Q=1 (2 levels is all a 1-bit code can express). With it, the
    # 1-bit V arm's nominal accounting equals what is actually allocated:
    # 128 code bits + 2 x scale_dtype (scale, zero) per (token, head) = 1.25
    # bit/coord at bf16, against 2.25 when the same code sits in int2 crumbs.
    # The V arena also halves, so it is a real memory saving, not only honest
    # accounting. K is untouched: the vq2 arms store K as codebook indices, and
    # the non-VQ int2 arms keep 2-bit K.
    SGLANG_V_INT_BITS = EnvInt(2)
    # Escape hatch for the quantized-KV chunked-prefill guard (see
    # model_runner_kv_cache_mixin._assert_no_chunked_prefill_for_quant): chunked
    # prefill makes later prompt chunks attend to earlier ones through QUANT-TIER
    # reconstructions, so error can compound through the prompt forward pass.
    # Benchmarks must run with --chunked-prefill-size -1; set this
    # to 1 ONLY for deliberate chunked-serving experiments.
    SGLANG_ALLOW_CHUNKED_QUANT_PREFILL = EnvBool(False)
    # Path to a group-VQ codebook bundle (.pt). When set (with
    # --kv-cache-dtype int2 + mixed KV windows), the unified pool stores the
    # K quant tier as per-group VQ indices (uint8) decoded through a packed
    # fp8-e4m3 codebook instead of int2 affine. V stays on the int2 path.
    SGLANG_VQ_CODEBOOK_PATH = EnvStr("")
    # V-side analogue of SGLANG_VQ_CODEBOOK_PATH: when set, the V quant tier is
    # stored as per-group VQ indices in the R_v=U_S basis instead of int2 affine
    # (additive ablation; unset keeps OSCAR scalar-INT2 V, the default). The
    # flush-encode and stage-2 decode gather for VQ-V are not implemented yet.
    SGLANG_VQ_V_CODEBOOK_PATH = EnvStr("")
    # --- vq_optimized: per-optimization switches, applied one at a time so each
    # can be evaluated in isolation before the next is enabled. All default OFF
    # so plain vq2 numerics/perf are untouched.
    #   QMAP  : fused Triton per-head query map (removes vq_map_q's
    #           permute/reshape/contiguous chain; math unchanged).
    SGLANG_VQ_OPT_QMAP = EnvBool(False)
    #   KMAP  : the same fused kernel for the K map (sub/bmm/contiguous -> one
    #           kernel; math unchanged). Runs on the decode aging flush and on
    #           the prefill VQ write.
    SGLANG_VQ_OPT_KMAP = EnvBool(False)
    #   FLUSH : fused Triton nearest-centroid encode for the decode aging
    #           flush. The torch path materialises an [L, n, H, NG, K] fp32
    #           score tensor (~76 MB at n=64 for Qwen3-8B) and makes 3 passes
    #           over it to produce 0.5 MB of uint8 indices.
    SGLANG_VQ_OPT_FLUSH = EnvBool(False)
    #   PREFILL: route the prefill/extend VQ writes through the same fused
    #           nearest-centroid kernel FLUSH already uses. The unfused path
    #           materialises a [chunk, H, NG, K] fp32 score tensor -- 537 MB at a
    #           2048-token chunk -- which dominates TTFT.
    SGLANG_VQ_OPT_PREFILL = EnvBool(False)
    # Fix for the chunked-prefill prefix-cache cap in the mixed-KV pool.
    # common.py folds TWO different constraints into
    # ``mixed_kv_quant_slack_cutoff_len`` via a running min: (A) the HP-recent
    # tail bound on a non-final chunk, which is purely positional and grows with
    # seq_len, and (B) partial-quant-page ownership, which is backed by
    # accumulating ``mixed_kv_quant_slack_indices`` and legitimately persists.
    # Because A is min-accumulated it latches at chunk 1's value
    # (chunked_prefill_size - hp_recent_tokens) and never advances, so a prompt
    # longer than one chunk can only ever cache that many tokens. This flag gives
    # A its own per-step field; B is untouched.
    SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS = EnvBool(False)
    # Exact chunked prefill for the mixed-KV pool. With 2+ prefill chunks,
    # chunk N+1 attends to chunk N through DEQUANTIZED quant-tier rows, and
    # that error compounds through every layer of the prompt's own forward
    # pass (measured -23.7 NIAH points on gpt-oss vq2; bf16 unaffected). A
    # single-chunk prefill never reads its own quant rows, so quantization
    # only touches decode. This flag closes the gap: while a request is mid
    # chunked prefill, the pool keeps a bf16 "shadow" copy of its quant-tier
    # rows (stored space), later chunks of the SAME prompt read the shadow
    # instead of dequantizing, and the shadow is dropped when the final chunk
    # completes. Final cache contents are bit-identical either way -- codes
    # are still written at chunk time -- so decode behavior is unchanged.
    # Cost: the in-flight chunked prompt's K/V in bf16 (quant layers only)
    # for the duration of its prefill. OFF by default (user decision:
    # opt-in) -- export SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL=1 for
    # accuracy-critical serving with chunked prefill.
    SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL = EnvBool(False)
    # Ablation knob for the exact-chunked-prefill shadow: which side gets the
    # exact rows at read time ("kv" both, "k" keys only, "v" values only).
    # Capture always stores both; this only gates the swap, so a single boot
    # can attribute the chunked-prefill loss to K (VQ) vs V (int2) error.
    SGLANG_MIXED_KV_SHADOW_SIDE = EnvStr("kv")
    SGLANG_MIXED_KV_HP_MAX_SPLITS = EnvInt(8)
    HADAMARD_ORDER = EnvInt(16)
    # Naive-INT2 baseline: make the plain-int2 Hadamard an identity (no rotation) so int2
    # quantizes raw K/V. Only affects the plain (non-OSCAR, non-mixed) int2 path.
    SGLANG_INT2_NO_HADAMARD = EnvBool(False)
    # Simulated (fake) KV quantization on the BF16 write path: path to a bundle from
    # pipelines/oscar_e2e/build_simquant_turboquant.py. Lets us measure bit-widths this
    # stack has no pool for (there is no int4 path) on generation-dominated reasoning
    # tasks, where a prefill-only harness would compress ~0.3% of the KV. Stores the
    # DEQUANTIZED values as BF16 -- exact accuracy of a real pool of that width, no
    # memory saving. Unset => the write-path block is skipped entirely.
    SGLANG_SIMQUANT_PATH = EnvStr("")
    # Hybrid-SWA models (gpt-oss) hand each inner pool a POOL-LOCAL layer id, so the
    # SWA and full-attention pools both count 0..N-1 and would index the same bundle
    # rows. Default skips the SWA pool, matching the oscar_int2/vq2 arms, which also
    # only quantize the full-attention layers. Set 0 to quantize both (needs a bundle
    # whose layer axis covers them separately).
    SGLANG_SIMQUANT_SKIP_SWA = EnvBool(True)
    # Path to an NSNQuant bundle (see codebooks/build_nsn_bundle.py), applied the same way as
    # SGLANG_SIMQUANT_PATH -- round-trip encode+dequant in the plain BF16 write path, stores
    # dequantized values, no memory saving. Independent of and unrelated to SGLANG_VQ_CODEBOOK_PATH
    # (the CQ/TaSQ VQ path, served through UnifiedInt2HPKVPool) -- this hook only ever runs on
    # the plain MHATokenToKVPool, so it cannot interact with or disturb that other path. Unset =>
    # the write-path block is skipped entirely.
    SGLANG_NSN_PATH = EnvStr("")

    # Scheduler: memory leak test
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2 ** 31)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)

    # Scheduler: new token ratio hyperparameters
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)

    # Scheduler: recv interval
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)

    # PD Disaggregation (runtime)
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    # Extra slots in req_to_token_pool for decode workers (only effective when
    # max_num_reqs > 32). Increases pool capacity so more KV cache transfers
    # can overlap with decode execution without raising max_running_requests.
    SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS = EnvInt(0)

    # Scheduler: others:
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)  # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_DATA_PARALLEL_BUDGET_INTERVAL = EnvInt(1)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)

    # Test: pd-disaggregation
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)

    # Model Parallel
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)

    # Tool Calling
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)

    # Hi-Cache
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)
    # Staging buffer for heterogeneous TP KV transfer
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB = EnvInt(64)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    # TODO(yangminl): remove SGLANG_STAGING_USE_TORCH and the torch fallback in
    # staging_buffer.py once Triton kernels are fully validated in production.
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    # Mooncake KV Transfer
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)

    # Mooncake Store
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("tcp")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)

    # AMD & ROCm
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)

    # MPS (Apple Silicon)
    SGLANG_USE_MLX = EnvBool(False)

    # NPU
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)
    # Delay all-gather after qlora for better performance for Deepseek v3.2
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    SGLANG_NPU_FUSED_MOE_MODE = EnvInt(1)

    # Quantization
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2 = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")

    # Flashinfer
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
    # Skip-softmax threshold scale factor for TRT-LLM attention (prefill and decode separately).
    # None = standard attention. See https://arxiv.org/abs/2512.12087
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    # TODO(mmangkad): Remove this once the FlashInfer unified allreduce-fusion
    # transport issue on GB200/GB300 platforms is fixed and verified resolved.
    SGLANG_FLASHINFER_FORCE_POSIX_FD_TRANSPORT = EnvBool(None)

    # Triton
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)

    # Torch Compile
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)

    # EPLB
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC = EnvBool(False)

    # TBO
    SGLANG_TBO_DEBUG = EnvBool(False)

    # DeepGemm
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    SGLANG_DG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)

    # DeepSeek MHA Optimization
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)

    # DeepEP
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False)
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)

    # NIXL-EP
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)

    # NSA Backend
    SGLANG_NSA_FUSE_TOPK = EnvBool(True)
    SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA = EnvBool(True)
    SGLANG_USE_FUSED_METADATA_COPY = EnvBool(True)
    SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvInt(2048)

    # sgl-kernel
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # Flash Attention
    SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)

    # vLLM dependencies (TODO: they have been deprecated, we can remove them safely)
    USE_VLLM_CUTLASS_W8A8_FP8_KERNEL = EnvBool(False)

    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)
    SGLANG_CUTLASS_MOE = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)

    # Deterministic inference
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # RoPE cache configuration
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # Overlap Spec V2
    SGLANG_ENABLE_SPEC_V2 = EnvBool(False)
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)

    # Spec Config
    SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK = EnvBool(True)
    SGLANG_SPEC_NAN_DETECTION = EnvBool(False)
    SGLANG_SPEC_OOB_DETECTION = EnvBool(False)

    # VLM
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)


    # VLM Item CUDA IPC Transport
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(False)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(4 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # Mamba
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)

    # Release & Resume Memory
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)

    # Sparse Embeddings
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)

    # Logits processor
    SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK = EnvBool(False)
    SGLANG_LOGITS_PROCESSER_CHUNK_SIZE = EnvInt(2048)

    # Tool-Call behavior
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # Ngram
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # Warmup
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1) # in seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging. Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm

    # HTTP Server
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)

    # HTTP/2 Server
    SGLANG_GRANIAN_PARENT_PID = EnvInt(None)

    # Health Check
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)

    # Encoder gRPC
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    # Encoder receiver selection: http|grpc (used by EPD paths).
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")

    # External models
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # Numa
    SGLANG_NUMA_BIND_V2 = EnvBool(True)
    SGLANG_AUTO_NUMA_BIND = EnvBool(False)

    # Metrics
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)

    # Tokenizer
    SGLANG_PATCH_TOKENIZER = EnvBool(False)  # TODO enable by default

    # TokenizerManager
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)

    # Symmetric Memory
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)

    # Aiter
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)
    # fmt: on

    # EPD
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)

    # Elastic EP Backup Port
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)

    # Sglang Cache Dir
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))


envs = Envs()
EnvField._allow_set_name = False


def _print_deprecated_env(new_name: str, old_name: str):
    if old_name in os.environ:
        warnings.warn(
            f"Environment variable {old_name} will be deprecated, please use {new_name} instead"
        )
        os.environ[new_name] = os.environ[old_name]


def _warn_deprecated_env_to_cli_flag(env_name: str, suggestion: str):
    """Warn when a deprecated environment variable is used.

    This is for env vars that are deprecated in favor of CLI flags.
    """
    if env_name in os.environ:
        warnings.warn(f"Environment variable {env_name} is deprecated. {suggestion}")


def _convert_SGL_to_SGLANG():
    _print_deprecated_env("SGLANG_LOG_GC", "SGLANG_GC_LOG")
    _print_deprecated_env(
        "SGLANG_MOE_NVFP4_DISPATCH", "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH"
    )
    _print_deprecated_env(
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
    )
    _deprecated_ms_to_s = {
        "SGLANG_QUEUED_TIMEOUT_MS": "SGLANG_REQ_WAITING_TIMEOUT",
        "SGLANG_FORWARD_TIMEOUT_MS": "SGLANG_REQ_RUNNING_TIMEOUT",
    }
    for old_name, new_name in _deprecated_ms_to_s.items():
        if old_name in os.environ:
            ms_val = os.environ[old_name]
            warnings.warn(
                f"Environment variable {old_name} (in ms) is deprecated, "
                f"please use {new_name} (in seconds) instead"
            )
            os.environ[new_name] = str(float(ms_val) / 1000.0)

    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


_convert_SGL_to_SGLANG()
_warn_deprecated_env_to_cli_flag(
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE",
    "Please use '--enable-prefill-delayer' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES",
    "Please use '--prefill-delayer-max-delay-passes' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK",
    "Please use '--prefill-delayer-token-usage-low-watermark' instead.",
)

# Import cuda_coredump to trigger auto-injection of CUDA env vars
# when SGLANG_CUDA_COREDUMP=1. Best-effort; for strict guarantees,
# set CUDA_* env vars in the shell before launching Python.
import sglang.srt.debug_utils.cuda_coredump  # noqa: F401, E402


def example_with_exit_stack():
    # Use this style of context manager in unit test
    exit_stack = ExitStack()
    exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
    assert envs.SGLANG_TEST_RETRACT.get() is False
    exit_stack.close()
    assert envs.SGLANG_TEST_RETRACT.get() is None


def example_with_subprocess():
    command = ["python", "-c", "import os; print(os.getenv('SGLANG_TEST_RETRACT'))"]
    with envs.SGLANG_TEST_RETRACT.override(True):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.wait()
        output = process.stdout.read().decode("utf-8").strip()
        assert output == "True"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.stdout.read().decode("utf-8").strip()
    assert output == "None"


def example_with_implicit_bool_avoidance():
    @contextmanager
    def assert_throws(message_matcher: str):
        try:
            yield
        except Exception as e:
            assert message_matcher in str(e), f"{e=}"
            print(f"assert_throws find expected error: {e}")
            return
        raise AssertionError(f"assert_throws do not see exceptions")

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if (1 != 1) or envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT or (1 == 1):
            pass


def examples():
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT.clear()
    assert envs.SGLANG_TEST_RETRACT.get() is False

    envs.SGLANG_TEST_RETRACT.set(None)
    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    envs.SGLANG_TEST_RETRACT.clear()
    assert not envs.SGLANG_TEST_RETRACT.is_set()

    envs.SGLANG_TEST_RETRACT.set(True)
    assert envs.SGLANG_TEST_RETRACT.get() is True

    with envs.SGLANG_TEST_RETRACT.override(None):
        assert (
            envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None
        )

    assert envs.SGLANG_TEST_RETRACT.get() is True

    envs.SGLANG_TEST_RETRACT.set(None)
    with envs.SGLANG_TEST_RETRACT.override(True):
        assert envs.SGLANG_TEST_RETRACT.get() is True

    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    example_with_exit_stack()
    example_with_subprocess()
    example_with_implicit_bool_avoidance()


if __name__ == "__main__":
    examples()
