#!/bin/bash
# Shared configuration for the build and sweep scripts. Sourced, not run.
#
# Paths are repository-relative and experiment settings are environment-overridable.

# ---- paths, all repo-relative ----
canonical_method() {
  # Accept the historical baseline ID, but write all new results as bf16.
  case "$1" in fp16) echo bf16 ;; *) echo "$1" ;; esac
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBOOKS="$REPO/codebooks"
CALIB="$REPO/calibration"
SCRIPTS="$REPO/scripts"
ARTIFACTS="${ARTIFACTS:-$REPO/artifacts}"
WORK="${WORK:-$REPO/work}"         # dumps, Fisher, pickles, logs, results
mkdir -p "$WORK"

# ---- vendored calibration code (third_party/) ----
# Everything the four arms need lives in the repo; no external checkout is required.
VENDOR="$REPO/third_party/kvquant"
# Optional: reuse an existing NSN codebook instead of generating one (see README).
NSN_CODEBOOK="${NSN_CODEBOOK:-}"

# lm_eval task dir holding the custom `math500` task (omit if you only run the other three).
CUSTOM_TASKS="${CUSTOM_TASKS:-}"

# ---- conda envs ----
# Two. Calibration and serving share one env; the lm_eval client needs its own because lm_eval
# 0.4.9 does not import under transformers 5.x (it reads AutoModelForVision2Seq, removed there),
# and the fork needs transformers 5.x. The client talks HTTP and never touches a GPU.
ENV_SERVE="${ENV_SERVE:-tasq-serve}"      # sglang, codebook fitting, converters
ENV_EVAL="${ENV_EVAL:-tasq-eval}"          # lm_eval client (transformers 4.53.x, CPU torch)
CONDA_BASE="$(conda info --base 2>/dev/null)"
EVAL_PY="${EVAL_PY:-$CONDA_BASE/envs/$ENV_EVAL/bin/python3}"

# ---- calibration (must match the published rows) ----
CALIB_DATASET="gpqa_code"   # KVQuant-side corpus for CQ/TaSQ
# The paper uses 64 windows, including 16 CodeParrot windows.
CALIB_NSAMPLES="${CALIB_NSAMPLES:-64}"
CALIB_MIX_CODE_N="${CALIB_MIX_CODE_N:-16}"  # gpqa_code: how many of CALIB_NSAMPLES windows are
                            # code; this is a count rather than a ratio.

# Include non-default calibration sizes in artifact names.
if [[ "$CALIB_NSAMPLES" == "16" && "$CALIB_MIX_CODE_N" == "4" ]]; then
  CALIB_TAG=""                 # suffix on the CQ/TaSQ pickles
  NOVA_CALIB_TAG="calib16"     # nova's artifact directory component
else
  CALIB_TAG="_calib${CALIB_NSAMPLES}c${CALIB_MIX_CODE_N}"
  NOVA_CALIB_TAG="calib${CALIB_NSAMPLES}c${CALIB_MIX_CODE_N}"
fi
CALIB_SEQLEN=2048
CALIB_SEED=0
COUPLED="${COUPLED:-8}"     # CQ group dim  (num_coupled)
ABITS="${ABITS:-10}"        # CQ code width (abits)
SPARSITY=0.99
HEAD_DIM=128
NSN_WINDOW=64

# ---- GPU topology (auto) ----
# Run one replica per visible GPU by default. Set GPUS to pin a subset.
detect_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "${CUDA_VISIBLE_DEVICES//,/ }"; return
  fi
  local n; n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
  [[ "${n:-0}" -gt 0 ]] || n=1
  seq -s' ' 0 $((n - 1))
}
GPUS="${GPUS:-$(detect_gpus)}"
read -r -a _GPU_ARR <<< "$GPUS"
# Replicas = visible devices / tensor-parallel degree.
TP="${TP:-1}"
N_REPLICAS=$(( ${#_GPU_ARR[@]} / TP ))
[[ "$N_REPLICAS" -ge 1 ]] || die "TP=$TP exceeds the ${#_GPU_ARR[@]} visible GPU(s) in GPUS='$GPUS'"

# ---- serving ----
# Serving kernel defaults shared by all benchmark modes.
export SGLANG_VQ_OPT_KMAP="${SGLANG_VQ_OPT_KMAP:-1}"
export SGL_VQWIDE_TRIG_ONFLY="${SGL_VQWIDE_TRIG_ONFLY:-1}"

# REASONING=1 switches the whole sweep to thinking-on mode: a different lm_eval driver
# (local-chat-completions carrying chat_template_kwargs.enable_thinking, which only survives as a
# nested dict), sampled decoding, a 32k generation budget, and the long-form task set.
REASONING="${REASONING:-0}"
if [[ "$REASONING" == "1" ]]; then
  TASKS=(${REASONING_TASKS:-lcb_v6})
  # 2 per replica: a 32k bf16 KV seq is 4.5 GB, so more in flight thrashes the pool.
  REASON_PER_REPLICA="${REASON_PER_REPLICA:-2}"
  NUM_CONCURRENT="${NUM_CONCURRENT:-$(( REASON_PER_REPLICA * N_REPLICAS ))}"
  REASON_TEMP="${REASON_TEMP:-0.6}"
  REASON_TOP_P="${REASON_TOP_P:-0.95}"
  REASON_MAX_GEN="${REASON_MAX_GEN:-32768}"
  # lm_eval applies this timeout to the complete evaluation request queue.
  REASON_TIMEOUT="${REASON_TIMEOUT:-172800}"

  # Use the repository task definitions and robust answer extractors.
  CUSTOM_TASKS="${CUSTOM_TASKS:-$REPO/tasks}"
else
  # SHORT_TASKS mirrors REASONING_TASKS above: any generate_until task the run_gsm8k_api.py
  # driver can serve (chat template + fewshot_as_multiturn + --no_think), e.g. bbh_cot_fewshot.
  TASKS=(${SHORT_TASKS:-gsm8k_cot_llama humaneval_instruct mbpp_instruct math500})
  CUSTOM_TASKS="${CUSTOM_TASKS:-$REPO/tasks}"
  SHORT_PER_REPLICA="${SHORT_PER_REPLICA:-16}"
  NUM_CONCURRENT="${NUM_CONCURRENT:-$(( SHORT_PER_REPLICA * N_REPLICAS ))}"
fi
# Server-side request cap per replica.
if [[ "$REASONING" == "1" ]]; then
  MAX_RUNNING="${MAX_RUNNING:-16}"
else
  MAX_RUNNING="${MAX_RUNNING:-$SHORT_PER_REPLICA}"
fi
# Full-precision residual policy used by the reported experiments.
if [[ "$REASONING" == "1" ]]; then
  PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
  RECENT_TOKENS="${RECENT_TOKENS:-256}"
else
  PREFIX_TOKENS="${PREFIX_TOKENS:-0}"
  RECENT_TOKENS="${RECENT_TOKENS:-64}"
fi
CHUNKED_PREFILL=-1          # no request splitting; see README
MAX_PREFILL_TOKENS=8192
MEM_FRACTION="${MEM_FRACTION:-0.80}"

# ---- helpers ----
have_conda() { source "$CONDA_BASE/etc/profile.d/conda.sh"; }
say() { echo "[exp] $* $(date -Is)"; }
die() { echo "[exp] FAILED: $*" >&2; exit 1; }
need() {  # need <var> <arm> <what>
  [[ -n "${!1:-}" ]] || die "$2 needs \$$1 -- $3"
  [[ -e "${!1}" ]]   || die "\$$1 does not exist: ${!1}"
}

# HF id -> short tag used for artifact dirs and result filenames
model_tag() {
  case "$1" in
    # More specific variants first -- a bare *Qwen3-4B* glob also matches the 2507 releases and
    # would silently share their artifacts and results directory with the base model.
    *Qwen3-4B-Thinking-2507*) echo qwen3_4b_think ;;
    *Qwen3-4B-Instruct-2507*) echo qwen3_4b_inst ;;
    *Qwen3-4B*) echo qwen3_4b ;;
    *Qwen3-8B*) echo qwen3_8b ;;
    *Llama-3.1-8B*|*Meta-Llama-3.1-8B*) echo llama31_8b ;;
    *Phi-4-reasoning-plus*) echo phi4_reason_plus ;;
    # Distill of R1 into a Llama-8B body. Deliberately NOT matched by the *Llama-3.1-8B*
    # glob above -- the name carries "Llama-8B", not "Llama-3.1-8B" -- but spelled out here
    # so the tag is stable rather than coming from the basename fallback.
    *DeepSeek-R1-Distill-Llama-8B*) echo dsr1_llama8b ;;
    # Qwen2 body (28 layers, 12 Q / 2 KV heads, head_dim 128), so it runs the same llama
    # path every other model here does. The size is in the pattern on purpose: NVIDIA also
    # ships 7B/14B/32B OpenReasoning-Nemotron models, and a bare glob would silently share
    # one artifact directory between them.
    *OpenReasoning-Nemotron-1.5B*) echo nemotron_1_5b ;;
    # Same family, Qwen2 body at 48 layers / 40 Q / 8 KV heads, head_dim 128. Spelled out
    # for the reason the 1.5B entry gives: the sizes must not share an artifact directory.
    *OpenReasoning-Nemotron-14B*) echo nemotron_14b ;;
    # Same family, 28 layers / 28 Q / 4 KV heads, head_dim 128 -- so kv_group_num is SEVEN,
    # which divides by no power of two. The pre-RoPE decode kernel therefore needs
    # SGL_PREROPE_BLOCK_H=8 to cover the group whole (nemotron_7b_env.sh); the packed NSN
    # kernel derives its own block_h and is fine unaided.
    *OpenReasoning-Nemotron-7B*) echo nemotron_7b ;;
    *) basename "$1" | tr 'A-Z.-' 'a-z__' ;;
  esac
}
