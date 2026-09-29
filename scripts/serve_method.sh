#!/usr/bin/env bash
# Serve one (model, method) arm with the vendored engine in ../python.
#
#   scripts/serve_method.sh MODEL METHOD ARTIFACT_DIR [extra sglang args...]
#
# METHOD:
#   bf16        stock BF16 KV cache (the reference arm)
#   vq          the VQ KV-cache engine every quantized arm runs on: mixed-KV
#               windows, calibrated rotations, INT2 V tier, group-VQ K tier
#   nsn         NSNQuant, simulated on the BF16 write path (ARTIFACT_DIR/nsn_bundle.pt;
#               no memory saving -- see python/sglang/srt/mem_cache/nsn_quant.py).
#               Uses the plain BF16 pool and never the mixed-KV path above.
#
# ARTIFACT_DIR must contain, for vq:
#   k_rotation_qqt_r_h_pbr.pt  v_rotation_sst_r_h_pbr.pt  [codebook.pt]
#
# Everything after ARTIFACT_DIR is passed to sglang.launch_server verbatim
# (--port, --context-length, --mem-fraction-static, --disable-radix-cache, ...).
#
# Env knobs (all optional):
#   PREFILL_BACKEND=fa3|triton   fa3 needs Hopper+; triton works everywhere
#   TP=1                         tensor parallel size
#   KV_SPLITS                    decode split-K (default: 48 for vq, 8 for bf16/nsn)
#   MAX_REQS=16                  --max-running-requests
#   MAX_TOKENS                   pin --max-total-tokens (default: derive from
#                                mem-fraction; pinning erases the capacity
#                                advantage the quantized pool exists to buy)
#   RADIX_CACHE=0|1              prefix cache (default off: eval rows are unique;
#                                prefix-reuse benchmarks set 1)
#   QUANT_GROUP_SIZE             min-max scale group (default: 128; hybrid-SWA
#                                models such as gpt-oss require per-head scales
#                                and default to 0 = omit the flag)
#   SGLANG_VQ2_CUDA=1            opt into the CUDA decode stage-1 kernel
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODEL METHOD ARTIFACT_DIR [sglang serve arguments...]" >&2
  exit 2
fi

model=$1
method=$2
artifact_dir=$3
shift 3

repo="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$repo/python:${PYTHONPATH:-}"

unset SGLANG_ENABLE_MIXED_KV_WINDOWS SGLANG_VQ_CODEBOOK_PATH SGLANG_SIMQUANT_PATH
unset SGLANG_OSCAR_K_ROTATION_PATH SGLANG_OSCAR_V_ROTATION_PATH
unset SGLANG_NSN_PATH

# Long-context evals exceed some models' config-derived limit (RoPE
# extrapolation, matching the HF-harness behaviour at 64K+).
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# expandable_segments lets the allocator return freed segments to the driver,
# which the capacity-sized quant pools rely on. Incompatible with the custom
# all-reduce's cudaIpc export under TP (vllm#42609), so TP>1 drops it.
if [[ "${TP:-1}" == "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

# Hybrid-SWA models (gpt-oss) need per-head scales (group flag omitted) and the
# runtime V-rotation path: the weight-folding helper assumes a dense layer
# range, but hybrid rotation bundles cover the full-attention layers only.
if [[ "$model" == *gpt-oss* ]]; then
  qgs_default=0
  absorb_v_default=0
else
  qgs_default=128
  absorb_v_default=1
fi

args=(
  --model-path "$model"
  --trust-remote-code
  --host 127.0.0.1
  --prefill-attention-backend "${PREFILL_BACKEND:-fa3}"
  --decode-attention-backend triton
  --sampling-backend pytorch
  --tp-size "${TP:-1}"
  --max-running-requests "${MAX_REQS:-16}"
)
[[ -n "${MAX_TOKENS:-}" ]] && args+=(--max-total-tokens "$MAX_TOKENS")
[[ "${RADIX_CACHE:-0}" == "1" ]] || args+=(--disable-radix-cache)

kv_splits_default=8

case "$method" in
  bf16)
    ;;
  vq)
    for f in k_rotation_qqt_r_h_pbr.pt v_rotation_sst_r_h_pbr.pt; do
      if [[ ! -f "$artifact_dir/$f" ]]; then
        echo "missing rotation bundle: $artifact_dir/$f" >&2
        exit 2
      fi
    done
    export SGLANG_ENABLE_MIXED_KV_WINDOWS=1
    export SGLANG_OSCAR_K_ROTATION_PATH="$artifact_dir/k_rotation_qqt_r_h_pbr.pt"
    export SGLANG_OSCAR_V_ROTATION_PATH="$artifact_dir/v_rotation_sst_r_h_pbr.pt"
    export SGLANG_OSCAR_K_CLIP_RATIO="${SGLANG_OSCAR_K_CLIP_RATIO:-0.96}"
    export SGLANG_OSCAR_V_CLIP_RATIO="${SGLANG_OSCAR_V_CLIP_RATIO:-0.92}"
    export SGLANG_OSCAR_ABSORB_V_ROTATION="${ABSORB_V_ROT:-$absorb_v_default}"
    export SGLANG_MIXED_KV_PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
    export SGLANG_MIXED_KV_RECENT_TOKENS="${RECENT_TOKENS:-256}"
    # V-tier integer levels for the OSCAR min/max SQ path: 3 = INT2 (official
    # default), 1 = the Nova-KV-1b fair-bits rebuild's 1-bit V (same quantizer,
    # 2 levels; storage stays int2 crumbs -- simulated width).
    export SGLANG_V_INT_MAX_Q="${V_INT_MAX_Q:-3}"
    # PHYSICAL width of the V arena: 2 = int2 crumbs (default, every historical arm),
    # 1 = int1. Set it through V_INT_BITS, not SGLANG_V_INT_BITS -- the line above shows
    # why: this script re-exports the SGLANG_* name from its own knob, so an
    # SGLANG_-prefixed value passed in from outside is silently replaced by the default
    # and the server comes up on the baseline path wearing the treatment's label.
    # V_INT_BITS=1 requires V_INT_MAX_Q=1 (the pool asserts it at boot).
    export SGLANG_V_INT_BITS="${V_INT_BITS:-2}"
    export SGLANG_MIXED_KV_HP_DTYPE=bfloat16
    export SGLANG_MIXED_KV_SCALE_DTYPE="${SCALE_DTYPE:-float16}"
    # Retaining cached prefixes at ~(window + ring) SWA tokens instead of full
    # length; hybrid-SWA only, inert on dense models. bf16 stays stock.
    export SGLANG_SWA_KEEP_PREFIX_TAIL="${SGLANG_SWA_KEEP_PREFIX_TAIL:-1}"
    args+=(--kv-cache-dtype int2)
    qgs="${QUANT_GROUP_SIZE:-$qgs_default}"
    [[ "$qgs" != "0" ]] && args+=(--kv-cache-quant-group-size "$qgs")
    if [[ "$method" == vq ]]; then
      if [[ ! -f "$artifact_dir/codebook.pt" ]]; then
        echo "missing VQ codebook: $artifact_dir/codebook.pt" >&2
        exit 2
      fi
      export SGLANG_VQ_CODEBOOK_PATH="$artifact_dir/codebook.pt"
      export SGLANG_VQ_FP8_FMT="${SGLANG_VQ_FP8_FMT:-e5m2}"
      export SGLANG_VQ_OPT_QMAP="${SGLANG_VQ_OPT_QMAP:-1}"
      export SGLANG_VQ_OPT_FLUSH="${SGLANG_VQ_OPT_FLUSH:-1}"
      export SGLANG_VQ_OPT_PREFILL="${SGLANG_VQ_OPT_PREFILL:-1}"
      # KMAP fused kernel: validated bit-identical to the einsum path (rel 0.0 on
      # decode/prefill shapes, real TaSQ bundle) and ~1.7x faster at the per-step
      # HP-write shape.
      export SGLANG_VQ_OPT_KMAP="${SGLANG_VQ_OPT_KMAP:-1}"
      # On-the-fly trig in the vqwide stage-1 (KVQuant deployment-kernel trick):
      # replaces per-token CosSin loads with in-kernel cos/sin from inv_freq.
      # Exactness-gated vs the table path (same tolerance vs torch reference);
      # measured -19% stage-1 time for PERM_ROPE (TaSQ), -3% for PRE_ROPE (CQ).
      export SGL_VQWIDE_TRIG_ONFLY="${SGL_VQWIDE_TRIG_ONFLY:-1}"
      kv_splits_default=48
    fi
    ;;
  nsn)
    bundle="$artifact_dir/nsn_bundle.pt"
    if [[ ! -f $bundle ]]; then
      echo "missing NSNQuant bundle: $bundle" >&2
      exit 2
    fi
    export SGLANG_NSN_PATH="$bundle"
    # Fused, CUDA-graph-compatible NSNQuant write path. Set to 0 for the reference implementation.
    export SGLANG_NSN_FUSED="${SGLANG_NSN_FUSED:-1}"
    # Same residual-policy parameters as the int2 methods above (vq/cq/tasq sweeps run with
    # RECENT_TOKENS=64): most recent N tokens stay raw, older windows are flushed to NSN VQ;
    # PREFIX = permanent raw attention-sink prefix (reference NSNQuant's KV_SINK analogue).
    export SGLANG_NSN_RECENT_TOKENS="${RECENT_TOKENS:-64}"
    export SGLANG_NSN_PREFIX_TOKENS="${PREFIX_TOKENS:-0}"
    ;;
  *)
    echo "unsupported method: $method" >&2
    exit 2
    ;;
esac

args+=(--triton-attention-num-kv-splits "${KV_SPLITS:-$kv_splits_default}")

exec "${NOVA_PYTHON:-python3}" -m sglang.launch_server "${args[@]}" "$@"
