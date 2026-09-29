#!/bin/bash
# Run the 4-benchmark sweep for one model, one arm at a time.
#
#   ./scripts/run_sweep.sh Qwen/Qwen3-8B                    # bf16 tasq nsn cq nova1b, all node GPUs
#   GPUS="4 5 6 7" ./scripts/run_sweep.sh Qwen/Qwen3-8B nsn # keep off the calibration GPUs
#
# One single-GPU replica per device, so the replica count -- and with it the request concurrency
# and the per-replica server cap -- follows whatever the node has (see config.sh "GPU topology").
#
# Each arm = one single-GPU server per GPU behind a round-robin proxy; the benchmarks then run
# sequentially against the proxy, so every request spreads over all replicas. Arms without an
# artifact are skipped, finished tasks are skipped, so re-running resumes.
# Results: work/<tag>/results/<tag>_<arm>_0_<task>.json
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
have_conda
conda activate "$ENV_SERVE"
export HF_ALLOW_CODE_EVAL=1

MODEL="${1:?usage: run_sweep.sh <hf-model> [bf16 tasq nsn cq nova1b]}"; shift
ARMS=("$@"); [[ $# -eq 0 ]] && ARMS=(bf16 tasq nsn cq nova1b)
for i in "${!ARMS[@]}"; do ARMS[$i]=$(canonical_method "${ARMS[$i]}"); done
TAG="$(model_tag "$MODEL")"
W="$WORK/$TAG"; LOG="$W/logs"; RESULTS="$W/results"; mkdir -p "$LOG" "$RESULTS"
# config.sh detects the node's GPUs and derives NUM_CONCURRENT/MAX_RUNNING from the count; this
# just splits its answer. Set GPUS (or CUDA_VISIBLE_DEVICES) to pin a subset.
read -r -a GPU_LIST <<< "$GPUS"
BASE_PORT="${BASE_PORT:-30081}"
PROXY_PORT="${PROXY_PORT:-30080}"
NO_THINK=(); [[ "$TAG" == qwen3* ]] && NO_THINK=(--no_think)
INCLUDE=(); [[ -n "$CUSTOM_TASKS" ]] && INCLUDE=(--include_path "$CUSTOM_TASKS")
# LIMIT=<n> caps docs per task -- smoke tests only; a limited result file is indistinguishable
# from a full one by name, so clear it before the real run.
LIM=(); [[ -n "${LIMIT:-}" ]] && LIM=(--limit "$LIMIT")
# lm_eval applies this timeout to the complete task queue.
SHORT_TIMEOUT="${SHORT_TIMEOUT:-86400}"
# Served context for the NSN packed arm (window arena bound). Reasoning uses REASON_CTX; NIAH
# drivers export NSN_CTX to their sequence length; short tasks default to 8192.
if [[ "${REASONING:-0}" == "1" ]]; then NSN_CTX="${NSN_CTX:-${REASON_CTX:-40960}}"; else NSN_CTX="${NSN_CTX:-8192}"; fi
[[ -x "$EVAL_PY" ]] || die "eval python not found: $EVAL_PY (see README 'Environments')"
PROXY_PID=""
SERVER_PIDS=()   # pid per GPU slot, so teardown never reaches another sweep's servers

arm_spec() {   # arm -> "engine|artifact|extra_env"
  local d="$ARTIFACTS/$TAG"
  local vq="SGLANG_VQ_PRE_ROPE=1 SGLANG_OSCAR_K_CLIP_RATIO=1.0 SGLANG_OSCAR_V_CLIP_RATIO=1.0"
  case "$1" in
    # bf16 ignores the artifact contents; serve_method.sh only needs the positional argument, so
    # point it at the repo itself -- the baseline must be runnable before anything is calibrated.
    # The baseline uses the BF16 serving engine.
    bf16) echo "bf16|$REPO|" ;;
    # NSN on its packed store: fused write path, two-tier packed read, CUDA graphs on. The bf16
    # tier is a per-request ring holding only the raw tail, so the token budget is set by the
    # packed store, not by a bf16 pool. HP_PREASSIGN must cover the request pool (max_running + 1);
    # MAXWIN is the window arena per request = context / window_size.
    nsn)  echo "nsn|${d}_nsn_1bit|SGLANG_NSN_FUSED=1 SGLANG_NSN_PACKED=1 SGLANG_NSN_PACKED_KEEP_BF16=0 SGLANG_NSN_HP_RING=512 SGLANG_NSN_HP_PREASSIGN=$((MAX_RUNNING + 2)) SGLANG_NSN_PACKED_MAXWIN_PER_REQ=$(( (NSN_CTX + 63) / 64 ))" ;;
    cq)   echo "vq|${d}_cq_g8|SGLANG_VQ_V_CODEBOOK_PATH=${d}_cq_g8/vq_v_codebook.pt $vq" ;;
    # NovaKV at matched bits: its own K trainer at G=8/10 bits, scalar 1-bit V.
    nova1b) echo "vq|${d}_nova1375_${NOVA_CALIB_TAG}|V_INT_MAX_Q=1 V_INT_BITS=1 SCALE_DTYPE=bfloat16" ;;
    # Ours.
    tasq) echo "vq|${d}_tasq_g8|SGLANG_VQ_V_CODEBOOK_PATH=${d}_tasq_g8/vq_v_codebook.pt $vq SCALE_DTYPE=float16" ;;
    *) die "unknown arm: $1" ;;
  esac
}

start_servers() {
  local arm=$1 engine=$2 artifact=$3 extra_env=$4 ups=() extra=()
  # NSN runs on the plain BF16 pool: apply_nsn_quant does host-side bookkeeping per call, which
  # is illegal under CUDA-graph capture. It also peaks higher during prefill, so on 24 GB cards
  # it gets a smaller reservation and prefill batch (memory knobs only -- no score effect).
  local memfrac="$MEM_FRACTION" prefill="$MAX_PREFILL_TOKENS"
  # Optionally cap CUDA-graph capture batches for models with limited free memory.
  local gbs=(); [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]] && gbs=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
  local ctx=(); [[ "$REASONING" == "1" ]] && ctx=(--context-length "${REASON_CTX:-40960}")
  case "$arm" in
    nsn)
      # Packed NSN: decode graphs on, piecewise off as for the int2 arms. Its pool is the
      # packed store plus a fixed bf16 ring tier, sized by pool_configurator like the int2 arms',
      # so it takes the common reservation and prefill batch. The window arena is bounded by the
      # served context, so the non-reasoning protocol pins one (NSN_CTX, default 8192 -- every
      # short task is far below it); reasoning/NIAH already pass their own --context-length.
      extra=(--disable-piecewise-cuda-graph)
      [[ "$REASONING" == "1" ]] || ctx=(--context-length "$NSN_CTX") ;;
    nova1b|cq|tasq)
      # the int2 arms keep decode CUDA graphs but disable the piecewise ones, as the
      # reference chain does (piecewise warmup compiles the whole prefill graph and OOMs)
      extra=(--disable-piecewise-cuda-graph) ;;
  esac
  # Allocate one rendezvous port per replica below the ephemeral-port range.
  local NCCL_BASE=$(( 20000 + ($$ % 500) * 16 ))
  local _try
  for _try in $(seq 1 40); do
    local _busy=0 _i _p
    for _i in "${!GPU_LIST[@]}"; do
      _p=$((NCCL_BASE + _i))
      ss -lnt "sport = :$_p" 2>/dev/null | grep -q LISTEN && { _busy=1; break; }
    done
    [[ "$_busy" == "0" ]] && break
    NCCL_BASE=$((NCCL_BASE + 16))
    [[ "$NCCL_BASE" -gt 28000 ]] && NCCL_BASE=20000
  done
  say "$arm: nccl ports $NCCL_BASE-$((NCCL_BASE + ${#GPU_LIST[@]} - 1))"
  for i in "${!GPU_LIST[@]}"; do
    local port=$((BASE_PORT + i)); ups+=("http://localhost:$port")
    ( cd "$REPO" && env $extra_env CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" \
        PYTHONPATH="$REPO/python${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA}" \
        PREFILL_BACKEND=triton TP=1 PREFIX_TOKENS=$PREFIX_TOKENS RECENT_TOKENS=$RECENT_TOKENS \
        nohup bash scripts/serve_method.sh "$MODEL" "$engine" "$artifact" \
        --port "$port" --nccl-port $((NCCL_BASE + i)) --max-running-requests $MAX_RUNNING \
        --chunked-prefill-size $CHUNKED_PREFILL --max-prefill-tokens "$prefill" \
        --mem-fraction-static "$memfrac" "${gbs[@]}" "${ctx[@]}" "${extra[@]}" \
        > "$LOG/server_${arm}_gpu${GPU_LIST[$i]}.log" 2>&1 & )
    sleep 1
    SERVER_PIDS[$i]=$(pgrep -f "sglang.launch_serve[r].*--port $port( |$)" | head -1)
  done
  for i in "${!GPU_LIST[@]}"; do
    local port=$((BASE_PORT + i))
    for _ in $(seq 1 120); do curl -sf "http://localhost:$port/health" >/dev/null && break; sleep 10; done
    curl -sf "http://localhost:$port/health" >/dev/null || {
      tail -30 "$LOG/server_${arm}_gpu${GPU_LIST[$i]}.log" >&2
      echo "[exp] $arm: server :$port did not start -- skipping arm" >&2; return 1; }
  done
  ( cd "$REPO" && nohup python3 "$SCRIPTS/rr_proxy.py" "$PROXY_PORT" "${ups[@]}" \
      > "$LOG/proxy_$arm.log" 2>&1 & )
  sleep 3
  PROXY_PID=$(pgrep -f "rr_proxy.py $PROXY_PORT" | head -1)
  say "$arm: ${#GPU_LIST[@]} servers + proxy :$PROXY_PORT ready"
}

stop_servers() {
  [[ -n "$PROXY_PID" ]] && kill -9 "$PROXY_PID" 2>/dev/null
  PROXY_PID=""
  # Kill only servers started by this sweep.
  for i in "${!GPU_LIST[@]}"; do
    local port=$((BASE_PORT + i)) pid="${SERVER_PIDS[$i]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      pkill -9 -P "$pid" 2>/dev/null
      kill -9 "$pid" 2>/dev/null
    elif [[ -z "$pid" ]]; then
      pkill -9 -f "sglang.launch_serve[r].*$port" 2>/dev/null
    fi
  done
  SERVER_PIDS=()
  # Wait until GPU memory from the previous arm is released.
  local waited=0 busy=0 used g
  while [ "$waited" -lt 120 ]; do
    busy=0
    for g in "${GPU_LIST[@]}"; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      if [ -n "$used" ] && [ "$used" -gt 2000 ]; then busy=1; fi
    done
    if [ "$busy" -eq 0 ]; then break; fi
    sleep 5; waited=$((waited + 5))
  done
  if [ "$busy" -eq 1 ]; then say "WARNING: GPU memory still held after ${waited}s -- next arm may OOM"; fi
  sleep 3
}
trap stop_servers EXIT

for arm in "${ARMS[@]}"; do
  IFS='|' read -r engine artifact extra_env <<< "$(arm_spec "$arm")"
  [[ -e "$artifact" ]] || { say "SKIP $arm: no artifact ($artifact) -- run build_bundles.sh"; continue; }
  say "===== $arm ====="
  start_servers "$arm" "$engine" "$artifact" "$extra_env" || { stop_servers; continue; }
  for task in "${TASKS[@]}"; do
    if [[ "$REASONING" == "1" ]]; then
      # Run sampled reasoning seeds concurrently against the same servers.
      pids=()
      for S in ${REASON_SEEDS:-0}; do
        out="$RESULTS/${TAG}_${arm}_${S}_${task}.json"
        [[ -f "$out" ]] && { say "$arm/$task seed $S already done"; continue; }
        ( cd "$REPO" && "$EVAL_PY" "$SCRIPTS/run_reasoning_api.py" --task "$task" --seed "$S" \
            --save_postfix "${TAG}_${arm}" --model_name "$MODEL" \
            --base_url "http://localhost:$PROXY_PORT/v1/chat/completions" \
            --output_dir "$RESULTS" --num_concurrent $NUM_CONCURRENT --timeout "$REASON_TIMEOUT" \
            --temperature "$REASON_TEMP" --top_p "$REASON_TOP_P" --max_gen_toks "$REASON_MAX_GEN" \
            "${INCLUDE[@]}" "${LIM[@]}" > "$LOG/eval_${arm}_${task}_s${S}.log" 2>&1 ) &
        pids+=($!)
      done
      [[ ${#pids[@]} -gt 0 ]] && wait "${pids[@]}"
      for S in ${REASON_SEEDS:-0}; do
        o="$RESULTS/${TAG}_${arm}_${S}_${task}.json"
        say "$arm/$task s$S: $(tr -d '[:space:]' < "$o" 2>/dev/null || echo "NO RESULT -- see $LOG/eval_${arm}_${task}_s${S}.log")"
      done
      continue
    fi
    out="$RESULTS/${TAG}_${arm}_0_${task}.json"
    [[ -f "$out" ]] && { say "$arm/$task already done"; continue; }
    # short tasks stay single-seed: greedy few-shot, so a second seed would return the same
    # answers. REASON_SEEDS applies only to the sampled reasoning path above.
    ( cd "$REPO" && "$EVAL_PY" "$SCRIPTS/run_gsm8k_api.py" --task "$task" --seed 0 \
        --save_postfix "${TAG}_${arm}" --chat_via_completions --fewshot_as_multiturn "${NO_THINK[@]}" \
        --model_name "$MODEL" --base_url "http://localhost:$PROXY_PORT/v1/completions" \
        --output_dir "$RESULTS" --num_concurrent $NUM_CONCURRENT --timeout "$SHORT_TIMEOUT" \
        "${INCLUDE[@]}" "${LIM[@]}" > "$LOG/eval_${arm}_${task}.log" 2>&1 )
    say "$arm/$task: $(tr -d '[:space:]' < "$out" 2>/dev/null || echo "NO RESULT -- see $LOG/eval_${arm}_${task}.log")"
  done
  stop_servers
done

say "sweep done -- results in $RESULTS"
