#!/bin/bash
# Table 3, one model: every context length x every seed x every method.
#
#   ./run_model.sh Qwen/Qwen3-4B-Thinking-2507
#   CTX="8192 16384" SEEDS="0" ./run_model.sh Qwen/Qwen3-8B tasq
#
# NIAH does not go through lm_eval -- rows come from tasks/niah/build_niah_rows.py and scoring is
# scripts/run_cell.py's contains_answer over the row file. So this script owns its serving loop
# instead of calling scripts/run_sweep.sh, but every serving knob is read from config.sh so the
# two cannot drift.
#
# One server session per method covers all lengths and seeds -- restarting per cell would pay the
# model-load cost 12 times over. Cells already in outputs/ are reported and skipped; if a method
# has nothing left to do its server is never started.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../common.sh"
OUTDIR="$HERE/../outputs"; mkdir -p "$OUTDIR"

MODEL="${1:?usage: run_model.sh <hf-model> [methods...]}"; shift
# NovaKV arm = nova1b; see experiments/common.sh.
METHODS=("$@"); [[ $# -eq 0 ]] && METHODS=(bf16 nsn nova1b cq tasq)

# NIAH is the long-context table: it uses the project's long-context residual protocol
# (PREFIX=64/RECENT=256, the setting reports under), not the short-task one.
export PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
export RECENT_TOKENS="${RECENT_TOKENS:-256}"
source "$REPO/config.sh"
for i in "${!METHODS[@]}"; do METHODS[$i]=$(canonical_method "${METHODS[$i]}"); done
# PREFIX/RECENT are exported above; NUM_CONCURRENT is not one of this path's knobs (NIAH drives
# its own client with NIAH_WORKERS, not lm_eval's num_concurrent), so record the number that is
# actually in flight rather than leaving the provenance field null. MAX_RUNNING is the matching
# server-side cap and comes from config.sh as a plain variable.
export NUM_CONCURRENT="${NIAH_WORKERS:-32}"
export MAX_RUNNING
have_conda; conda activate "$ENV_SERVE"

TAG="$(model_tag "$MODEL")"
# Grid: 4k .. 64k, no 128k. At 131072 the prompt must be prefilled in ONE
# forward -- chunked prefill is refused for the quantized arms
# (_assert_no_chunked_prefill_for_quant) -- and the activations for a 131k forward do not fit
# beside a pool large enough to hold the sequence: the bf16 arm booted with a 184606-token pool
# and 9.2 GB free, then died on "Tried to allocate 3.48 GiB" for every 128k row while 8k-64k had
# all passed. 4096 replaces it, which also gives the curve a short-context anchor.
read -r -a CTX_LIST  <<< "${CTX:-4096 8192 16384 32768 65536}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2}"
read -r -a GPU_LIST  <<< "$GPUS"   # detected in config.sh unless pinned
BASE_PORT="${BASE_PORT:-30081}"; PROXY_PORT="${PROXY_PORT:-30080}"
W="$WORK/$TAG/niah"; LOG="$W/logs"; mkdir -p "$LOG"
NIAH_MAX_TOKENS="${NIAH_MAX_TOKENS:-128}"   # raise to 8192 for a thinking model
# Served context of this run, read by scripts/run_sweep.sh's arm_spec for the packed NSN arm's window
# arena; must equal the --context-length the servers below are given.
_nsn_maxctx=0; for _c in "${CTX_LIST[@]}"; do (( _c > _nsn_maxctx )) && _nsn_maxctx=$_c; done
export NSN_CTX=$(( _nsn_maxctx + NIAH_MAX_TOKENS + 1024 ))
NIAH_WORKERS="${NIAH_WORKERS:-32}"
# Tensor parallelism. TP=1 puts one replica on each GPU in $GPUS; TP=n groups them n at a time,
# so $GPUS must be a multiple of n and the replica count drops accordingly. The bf16 reference
# row needs TP>=2 at the long lengths on 24 GB cards -- KV alone is 16 GiB (Llama) / 18 GiB
# (Qwen3) at 128k, on top of 8-16 GiB of weights. The quantized arms fit at TP=1 throughout.
TP="${TP:-1}"
# TP>1 is a BF16-only escape hatch. The quantized arms have never been exercised at TP>1 in this
# fork: there is no per-rank sharding of the codebook or the TaSQ permutation, and the decode
# kernel asserts "TP size must divide the KV head count (no replication support)" -- which is a
# live path, not a theoretical one (it fired for a 10-KV-head model). Refuse rather
# than emit a number from it; the BF16 row is the only one that needs the memory anyway.
if (( TP > 1 )); then
  for _m in "${METHODS[@]}"; do
    # Only the baseline supports tensor parallelism here.
    [[ "$_m" == "bf16" ]] || pe_die "TP=$TP requested with method '$_m'. Only the bf16 baseline is supported at TP>1 (see README). Run: TP=$TP $0 <model> bf16"
  done
fi
PROXY_PID=""; SERVER_PIDS=()

# ---- row files (deterministic given model + length + seed) ----
rows_for() { echo "$W/niah_${1}_${TAG}_s${2}.jsonl"; }
ensure_rows() {
  local ctx=$1 seed=$2 f; f=$(rows_for "$ctx" "$seed")
  [[ -s "$f" ]] && return 0
  pe_say "building rows ctx=$ctx seed=$seed"
  ( cd "$REPO" && python3 tasks/niah/build_niah_rows.py --model "$MODEL" \
      --length "$ctx" --seed "$seed" --out "$f" ) > "$LOG/rows_${ctx}_s${seed}.log" 2>&1 \
    || pe_die "row build failed (ctx=$ctx seed=$seed) -- see $LOG/rows_${ctx}_s${seed}.log"
}

cell_file() { echo "$OUTDIR/${TAG}__niah@${1}__${2}__s${3}.json"; }

# ---- serving (mirrors scripts/run_sweep.sh; knobs from config.sh) ----
source_arm_spec() {   # arm -> "engine|artifact|extra_env", by reusing run_sweep's own table
  bash -c 'source "'"$REPO"'/config.sh"; '"$(sed -n '/^arm_spec()/,/^}/p' "$REPO/scripts/run_sweep.sh")"'; MODEL='"$MODEL"'; TAG='"$TAG"'; arm_spec '"$1"
}
start_servers() {
  local arm=$1 engine=$2 artifact=$3 extra_env=$4 ups=() extra=()
  local memfrac="$MEM_FRACTION" prefill="$MAX_PREFILL_TOKENS"
  local maxctx_probe=0; for c in "${CTX_LIST[@]}"; do (( c > maxctx_probe )) && maxctx_probe=$c; done
  case "$arm" in
  # Cap graph capture for the quantized arms. Their decode path has many more kernels per layer
  # than bf16's, so capture uses more memory. Batches above the cap run eager; model outputs are
  # unchanged.
    nsn)  extra=(--disable-piecewise-cuda-graph --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-8}")
          # Packed NSN: pool = packed store + fixed bf16 ring tier, sized by
          # pool_configurator like the int2 arms', so it takes the common reservation and
          # prefill batch. Rings cover this driver's --max-running-requests 8 (+2 margin).
          extra_env="$extra_env SGLANG_NSN_HP_PREASSIGN=10" ;;
    nova|novabf16|novasq|novasq2|nova1b|cq|tasq|tasq*) extra=(--disable-piecewise-cuda-graph --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-8}") ;;
  esac
  # NIAH_MAX_RUNNING caps how many requests the SERVER runs at once. 8 is the historical value
  # and it is a RACE at long contexts, not a safe bound: a 32k request occupies
  # (32768 - P - R)/N_Q ~= 4056 quant pages of the 25906 the int2 pool has, so only ~6.4 can be
  # resident. It normally survives because NIAH generations are short (128 tokens) and pages free
  # faster than prefills claim them -- 64k cells, which are worse still, have completed at 8. When
  # the race is lost the server raises "failed to allocate quant slots after eviction" and SIGQUITs
  # mid-cell, and the cell is not lost loudly: it lands as a NUMBER scored on whatever rows made it
  # (llama31_8b/niah@32768/s2 came back 100.00 on 23/200, then 100.00 on 3/200). _drop_short_cells
  # is what keeps that out of the table. Lower this for a cell that keeps losing the race.
  # Chunked prefill, and why it is arm-dependent. With CHUNKED_PREFILL=-1 a 131072-token prompt is
  # prefilled in ONE forward, and its activations do not fit next to the KV pool: the bf16 arm came
  # up with a 184606-token pool and 9.2 GB free, then died on "Tried to allocate 3.48 GiB" for
  # every 128k row while 8k-64k had passed. Chunking bounds that working set and is
  # mathematically the same attention, so the BF16 row gets it. The quantized arms cannot: the
  # serving stack refuses the combination outright
  # (model_runner_kv_cache_mixin._assert_no_chunked_prefill_for_quant), so for them the only lever
  # is leaving more room beside the pool -- see NIAH_QUANT_MEMFRAC below.
  local chunked="$CHUNKED_PREFILL"
  if (( maxctx_probe > 65536 )) && [[ "$arm" == "bf16" ]]; then
    chunked="${NIAH_BF16_CHUNKED:-${NIAH_FP16_CHUNKED:-8192}}"
  elif [[ -n "${NIAH_QUANT_MEMFRAC:-}" ]]; then
    # Applies whenever it is set, with no length condition. It used to be gated on
    # maxctx_probe > 65536, which after the grid dropped 128k could never be true -- so the one
    # lever the quantized arms have at their longest length was dead, and a probe sweeping
    # fractions would have reported every one of them as failing for a reason that was the
    # gate itself.
    # One server session serves every length, so the fraction has to satisfy the WORST case. The
    # pool still needs room for one full-length sequence (~16 GB of bf16 KV for an 8B at 128k),
    # and what is left over has to absorb the unchunked prefill. Set this only after measuring
    # the boot at the longest length -- guessing it low makes the pool too small to hold a single
    # 128k row, which fails differently and just as hard.
    memfrac="$NIAH_QUANT_MEMFRAC"
  fi
  echo "[pe] $arm: chunked-prefill=$chunked mem-fraction=$memfrac max-ctx=$maxctx_probe"
  local maxctx=$maxctx_probe
  local nrep=$(( ${#GPU_LIST[@]} / TP ))
  (( nrep >= 1 )) || pe_die "TP=$TP needs at least $TP GPUs, got ${#GPU_LIST[@]}"
  (( ${#GPU_LIST[@]} % TP == 0 )) || pe_die "GPUS count ${#GPU_LIST[@]} is not a multiple of TP=$TP"
  for (( i = 0; i < nrep; i++ )); do
    local port=$((BASE_PORT + i)); ups+=("http://localhost:$port")
    local devs; devs=$(IFS=,; echo "${GPU_LIST[*]:i*TP:TP}")
    ( cd "$REPO" && env $extra_env CUDA_VISIBLE_DEVICES="$devs" PYTHONPATH="$REPO/python" \
        PREFILL_BACKEND=triton TP=$TP PREFIX_TOKENS=$PREFIX_TOKENS RECENT_TOKENS=$RECENT_TOKENS \
        nohup bash scripts/serve_method.sh "$MODEL" "$engine" "$artifact" \
        --port "$port" --max-running-requests "${NIAH_MAX_RUNNING:-8}" \
        --chunked-prefill-size $chunked --max-prefill-tokens "$prefill" \
        --mem-fraction-static "$memfrac" \
        --context-length $((maxctx + NIAH_MAX_TOKENS + 1024)) "${extra[@]}" \
        > "$LOG/server_${arm}_rep${i}.log" 2>&1 & )
    sleep 1
    SERVER_PIDS[$i]=$(pgrep -f "sglang.launch_serve[r].*--port $port( |$)" | head -1)
  done
  for (( i = 0; i < nrep; i++ )); do
    local port=$((BASE_PORT + i))
    for _ in $(seq 1 120); do curl -sf "http://localhost:$port/health" >/dev/null && break; sleep 10; done
    curl -sf "http://localhost:$port/health" >/dev/null || { pe_say "server $port never came up"; return 1; }
  done
  ( cd "$REPO" && RR_PROXY_TIMEOUT=7200 nohup python3 scripts/rr_proxy.py "$PROXY_PORT" "${ups[@]}" \
      > "$LOG/proxy_${arm}.log" 2>&1 & )
  sleep 3; PROXY_PID=$(pgrep -f "rr_proxy.py $PROXY_PORT" | head -1)
}
stop_servers() {
  [[ -n "$PROXY_PID" ]] && kill -9 "$PROXY_PID" 2>/dev/null; PROXY_PID=""
  for pid in "${SERVER_PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null; done
  SERVER_PIDS=(); sleep 10
}
trap stop_servers EXIT

# ---- main ----
for m in "${METHODS[@]}"; do
  # The served scale dtype, for emit_niah.py's provenance. nova and novabf16 share a codebook,
  # a protocol and an artifact_md5 -- this is the only field that separates their rows. Keep in
  # step with scripts/run_sweep.sh's arm_spec and serve_method.sh's float16 VQ default.
  case "$m" in
    novabf16|novasq|novasq2|nova1b) export PE_SCALE_DTYPE=bfloat16 ;;
    cq|tasq|tasq*|tasq*) export PE_SCALE_DTYPE=float16 ;;
    *)        export PE_SCALE_DTYPE=float32 ;;
  esac
  todo=(); already=()
  for ctx in "${CTX_LIST[@]}"; do for s in "${SEED_LIST[@]}"; do
    if [[ "${FORCE:-0}" != "1" ]] && SEED="$s" REPLICATE="" pe_cell_exists "$OUTDIR" "$TAG" "niah@$ctx" "$m"; then already+=("${ctx}/s${s}")
    else todo+=("${ctx}:${s}"); fi
  done; done
  [[ ${#already[@]} -gt 0 ]] && pe_say "$m: ALREADY RUN, skipping ${#already[@]} cell(s): ${already[*]}"
  if [[ ${#todo[@]} -eq 0 ]]; then pe_say "$m: nothing to do (all cells in outputs/). FORCE=1 to re-run."; continue; fi

  for spec in "${todo[@]}"; do ensure_rows "${spec%%:*}" "${spec##*:}"; done

  IFS='|' read -r engine artifact extra_env <<< "$(source_arm_spec "$m")"
  [[ -e "$artifact" ]] || { pe_say "SKIP $m: no artifact ($artifact) -- run build/build_artifacts.sh"; continue; }
  pe_say "===== $m: ${#todo[@]} cell(s) ====="
  start_servers "$m" "$engine" "$artifact" "$extra_env" || { stop_servers; continue; }

  for spec in "${todo[@]}"; do
    ctx="${spec%%:*}"; s="${spec##*:}"
    raw="$W/niah${ctx}_${m}_s${s}.jsonl"; rm -f "$raw"
    ( cd "$REPO" && PYTHONPATH="$REPO" python3 scripts/run_cell.py \
        --task ruler_niah --model "$MODEL" --method "$m" --seed "$s" \
        --input "$(rows_for "$ctx" "$s")" --output "$raw" \
        --workers "$NIAH_WORKERS" --max-tokens "$NIAH_MAX_TOKENS" \
        --base-url "http://127.0.0.1:$PROXY_PORT" ) > "$LOG/${m}_${ctx}_s${s}.log" 2>&1
    if [[ -s "$raw" ]]; then
      SEED="$s" python3 "$HERE/emit_niah.py" "$raw" "$(cell_file "$ctx" "$m" "$s")" \
        "$TAG" "niah@${ctx}" "$m" "$s" "$ARTIFACTS/${TAG}_$(pe_artifact_suffix "$m")"
    else
      pe_say "NO RESULT $m/${ctx}/s${s} -- see $LOG/${m}_${ctx}_s${s}.log"
    fi
  done
  stop_servers
done
pe_say "done -- build the table with ../make_table.py"
