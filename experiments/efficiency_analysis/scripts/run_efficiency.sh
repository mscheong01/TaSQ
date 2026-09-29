#!/bin/bash
# Efficiency figure, one panel at a time. Three arms: bf16 / cq / tasq (the adopted setting).
#
#   ./run_efficiency.sh decode     # panel 1: bs=1 decode latency vs context length, 1k..128k
#   ./run_efficiency.sh thru       # panel 2: E2E throughput vs batch size, 2048 in / 32768 out
#   ./run_efficiency.sh prefill    # panel 3: prefill latency at 8k / 16k / 32k
#   ARMS="tasq" CTXS="65536" ./run_efficiency.sh decode      # subset re-run
#   OUTDIR=... EXTRA_ENV="SGLANG_OFOLD=1" EXTRA_PYPATH=/path/to/patch ./run_efficiency.sh thru
#   EFF_SCALE_DTYPE=float32 ./run_efficiency.sh thru       # controlled dtype ablation
#   EXTRA_ENV_CQ="SGLANG_PTNFREE_POOL=1" ./run_efficiency.sh thru   # per-arm env (cq-only flags)
#   THRU_CAP_FROM=tasq ARMS="cq" ./run_efficiency.sh thru   # end cq's curve at tasq's batch ceiling
#
# Measurements use the same server path as the accuracy experiments. Each point runs alone on one
# GPU with CUDA graphs enabled; shared serving settings are held fixed across methods.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EFF="$(dirname "$HERE")"
source "$HERE/../../common.sh"
# OUTDIR: where cell JSONs land. Override to measure a variant without touching the recorded
# baseline cells.
OUTDIR="${OUTDIR:-$EFF/outputs}"; mkdir -p "$OUTDIR"

PANEL="${1:?usage: run_efficiency.sh <decode|thru|prefill>}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"

# Figure 4 uses the reference efficiency policy: no attention sink and a 64-token raw tail.
# Export before config.sh so the reasoning model does not substitute the benchmark-only 64/256
# policy. This setting is part of the capacity measurement and is recorded in every result cell.
export PREFIX_TOKENS="${PREFIX_TOKENS:-0}"
export RECENT_TOKENS="${RECENT_TOKENS:-64}"
source "$REPO/config.sh"
have_conda; conda activate "$ENV_SERVE"

TAG="$(model_tag "$MODEL")"
# Native context window, read from the checkpoint. It determines two serving constraints:
#   * serving ABOVE it silently engages rope scaling, which changes the model rather than the
#     KV format -- Llama-3.1-8B stops at exactly 131072, so the +DEC_OUT+1024 margin the decode
#     panel adds would cross it while Qwen3-4B-Thinking (262144) has room to spare;
#   * serving exactly AT it is refused per REQUEST: the server rejects input+output == the
#     context length ("Requested token count exceeds the model's maximum context length"), which
#     is what the panel's longest point is built to be. Measured on qwen: 131008 + 64.
NATIVE_CTX="$(python3 - "$MODEL" <<'PYX'
import json, glob, os, sys
m = sys.argv[1]
# A local checkout, else the hub cache -- whose location is HF_HUB_CACHE, else $HF_HOME/hub,
# else the default. Hardcoding ~/.cache/huggingface made this step fail on any machine that
# sets either variable.
cand = [os.path.join(m, "config.json")]
hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
cand += sorted(glob.glob(os.path.join(hub, "models--" + m.replace("/", "--"),
                                      "snapshots", "*", "config.json")))
for p in cand:
    if os.path.exists(p):
        print(json.load(open(p)).get("max_position_embeddings", 0) or 0); break
else:
    print(0)
PYX
)"
[[ "${NATIVE_CTX:-0}" -gt 0 ]] || pe_die "could not read max_position_embeddings for $MODEL"
pe_say "native context window: $NATIVE_CTX"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-30091}"
PIECEWISE="${PIECEWISE:-0}"
# Match the effective FlashAttention prefill path across methods.
PREFILL_BACKEND="${PREFILL_BACKEND:-fa3}"
CHUNKED="${CHUNKED:--1}"
# Panel (a) times DECODE steps only, so how the context got loaded is irrelevant to what it
# measures -- the quantized arms' chunked-prefill refusal is an ACCURACY guard (later chunks
# attend to quantized reconstructions of earlier ones), and no accuracy is read here. So if
# a long unchunked prefill will not boot, panel (a) alone may set
#   DEC_CHUNKED=8192 SGLANG_ALLOW_CHUNKED_QUANT_PREFILL=1
# Panel (c) must NOT: prefill latency is the measurement there, and chunking changes it.
DEC_CHUNKED="${DEC_CHUNKED:-$CHUNKED}"
MEMFRAC="${MEMFRAC:-0.85}"
# Decode sweeps choose the smallest pool reservation that fits the longest context, leaving
# sufficient workspace for unchunked prefill.
DEC_MEMFRAC_LADDER="${DEC_MEMFRAC_LADDER:-0.60 0.65 0.70 0.75 0.80 0.85}"
# All arms use the same memory fraction by default. The override remains available for controlled
# capacity studies with a different residual policy.
THRU_QUANT_MEMFRAC="${THRU_QUANT_MEMFRAC:-$MEMFRAC}"
W="$WORK/$TAG/efficiency"; LOG="$W/logs"; mkdir -p "$LOG"
read -r -a ARM_LIST <<< "${ARMS:-bf16 cq tasq}"

# ---- panel grids ----
# decode: x = TOTAL context. input_len = ctx - output_len so the served context never exceeds the
# model's 131072 native window (no rope scaling anywhere in this figure).
DEC_OUT=64
read -r -a DEC_CTXS <<< "${CTXS:-1024 2048 4096 8192 16384 32768 65536 131072}"
# thru: 2048 in / 32768 out. bf16's grid is TRUNCATED AT ITS MEASURED CAPACITY, not at a guess --
# see cap_batches below, which reads the engine's own max_total_num_tokens.
THRU_IN="${THRU_IN:-2048}"; THRU_OUT="${THRU_OUT:-32768}"
# Non-default shapes include the shape in their cell key. The published 2048/32768 setting keeps
# the bare `bsN` key consumed by the default figure loader.
THRU_KEY=""; (( THRU_IN == 2048 && THRU_OUT == 32768 )) || THRU_KEY="${THRU_IN}x${THRU_OUT}_"
# The grid includes both the BF16 capacity (6) and the quantized capacity (84) reported in the
# paper; each arm is also measured once at its exact engine-reported capacity.
read -r -a THRU_BS <<< "${BSS:-1 2 4 8 16 32 64 128 256}"
# prefill: output_len 4 rather than 1 -- the reported prefill latency is last_ttft either way, but
# a 1-token generation makes (latency - ttft) ~ 0 and the throughput fields meaningless.
PRE_OUT=4
read -r -a PRE_CTXS <<< "${CTXS:-8192 16384 32768}"

source_arm_spec() {
  bash -c 'source "'"$REPO"'/config.sh"; '"$(sed -n '/^arm_spec()/,/^}/p' "$REPO/scripts/run_sweep.sh")"'; MODEL='"$MODEL"'; TAG='"$TAG"'; arm_spec '"$1"
}

gpu_guard() {
  nvidia-smi -L >/dev/null 2>&1 || pe_die "GPU guard: nvidia-smi -L failed"
  python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || pe_die "GPU guard: torch.cuda unavailable"
}

SERVER_PID=""
stop_server() {
  [[ -n "$SERVER_PID" ]] && kill -9 "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  for _ in $(seq 1 40); do
    pgrep -f "sglang.launch_serve[r].*--port $PORT( |$)" >/dev/null || break
    sleep 3
  done
  sleep 5
}
trap stop_server EXIT

# start_server <arm> <engine> <artifact> <extra_env> <ctx> <maxbs>
start_server() {
  local arm=$1 engine=$2 artifact=$3 extra_env=$4 ctx=$5 maxbs=$6
  # Use a distinct server log for each panel, arm, GPU, and throughput shape.
  local slog="$LOG/server_${PANEL}_${THRU_KEY:-}${arm}_gpu${GPU_ID}.log"
  local extra=()
  [[ "$PIECEWISE" == "1" ]] || extra+=(--disable-piecewise-cuda-graph)
  pe_say "$arm: booting gpu=$GPU_ID ctx=$ctx cuda-graph-max-bs=$maxbs mem-frac=$MEMFRAC chunked=$CHUNKED"
  # EXTRA_ENV / EXTRA_PYPATH inject an out-of-tree patch into the server; both empty by
  # default, so the baseline protocol is unchanged.
  ( cd "$REPO" && env $extra_env ${EXTRA_ENV:-} CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$REPO/python${EXTRA_PYPATH:+:$EXTRA_PYPATH}" \
      PREFILL_BACKEND="$_pb" TP=1 PREFIX_TOKENS="$PREFIX_TOKENS" RECENT_TOKENS="$RECENT_TOKENS" \
      nohup bash scripts/serve_method.sh "$MODEL" "$engine" "$artifact" \
      --port "$PORT" --max-running-requests "$maxbs" \
      --chunked-prefill-size "$CHUNKED" --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
      --mem-fraction-static "$MEMFRAC" --context-length "$ctx" \
      --cuda-graph-max-bs "$maxbs" "${extra[@]}" \
      > "$slog" 2>&1 & )
  sleep 2
  SERVER_PID=$(pgrep -f "sglang.launch_serve[r].*--port $PORT( |$)" | head -1)
  for _ in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/health" >/dev/null && break
    # fail fast instead of burning the whole health window on a dead server
    grep -qE "OutOfMemory|Tried to allocate|SIGQUIT received|Capture cuda graph failed" "$slog" 2>/dev/null \
      && { pe_say "$arm: server DIED during boot -- see $slog"; return 1; }
    sleep 10
  done
  curl -sf "http://localhost:$PORT/health" >/dev/null || { pe_say "$arm: server never came up -- see $slog"; return 1; }
  assert_graphs_on "$arm" "$slog" || return 1
  return 0
}

# The figure is only meaningful if every point ran under CUDA graphs. A boot that skipped or
# fell back to eager decode must not silently become a data point.
assert_graphs_on() {
  local arm=$1 slog=$2
  grep -q "Capture cuda graph end" "$slog" || {
    pe_say "$arm: REFUSING -- no 'Capture cuda graph end' in $slog (graphs did not capture)"; return 1; }
  grep -q "disable_cuda_graph=True" "$slog" && {
    pe_say "$arm: REFUSING -- server reports disable_cuda_graph=True"; return 1; }
  pe_say "$arm: cuda graphs captured ($(grep -c 'Capture cuda graph end' "$slog") capture block(s))"
  return 0
}

max_total_tokens() {
  curl -sf "http://localhost:$PORT/get_server_info" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("max_total_num_tokens",0))' 2>/dev/null \
    || echo 0
}

cell() { echo "$OUTDIR/${TAG}__${PANEL}__${1}__${2}.json"; }

# run_point <arm> <key> <bs> <inlen> <outlen> <artifact> <extra_env>
run_point() {
  local arm=$1 key=$2 bs=$3 inlen=$4 outlen=$5 artifact=$6 extra_env=$7
  local f; f="$(cell "$arm" "$key")"
  # Resume on SUCCESSFUL cells only. A cell whose JSON exists with "ok": false is what a killed or
  # crashed run leaves behind; counting it as done makes a re-run skip the point and the panel come
  # back missing it.
  if [[ -s "$f" && "${FORCE:-0}" != "1" ]]; then
    if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("ok") else 1)' "$f" 2>/dev/null; then
      pe_say "$arm/$key: already done, skipping"; return 0
    fi
    pe_say "$arm/$key: previous cell recorded a FAILURE -- re-measuring"
  fi
  # bench_point.py defaults to a 5400 s (90 min) subprocess timeout, which is shorter than the
  # biggest cells in this panel: at 2048 in / 32768 out, bs64 needs ~88 min of generation and the
  # bs84 capacity point ~2 h. That silently killed four cells -- bs64 on both arms,
  # which then made the peak check read bs32 as the last point and go on to spend another 3 h on
  # bs84 cells that could not finish either. Size the budget from the cell, not from a constant:
  # output_len * batch / a pessimistic 200 tok/s aggregate floor, plus 30 min for boot and warmup.
  # Measured aggregate output rates are 400-500 tok/s at the large batches, so that is
  # ~2.3x headroom on bs64 (12,285 s budget against a 5,280 s cell) while still catching a hang.
  local _to="${BENCH_TIMEOUT:-}"
  if [[ -z "$_to" ]]; then _to=$(( 1800 + outlen * bs / 200 )); fi
  python3 "$HERE/bench_point.py" --base-url "http://localhost:$PORT" --out "$f" --timeout "$_to" \
    --model "$MODEL" --tag "$TAG" --arm "$arm" --panel "$PANEL" --key "$key" \
    --batch-size "$bs" --input-len "$inlen" --output-len "$outlen" \
    --artifact "$artifact" --extra-env "$extra_env" 2>&1 | tee -a "$LOG/${PANEL}_${arm}.log"
}

pe_say "===== efficiency panel '$PANEL' -- model=$TAG arms=${ARM_LIST[*]} gpu=$GPU_ID ====="
gpu_guard

for arm in "${ARM_LIST[@]}"; do
  IFS='|' read -r engine artifact extra_env <<< "$(source_arm_spec "$arm")"
  [[ -e "$artifact" ]] || { pe_say "SKIP $arm: no artifact ($artifact)"; continue; }
  # Use one scale dtype across quantized methods unless explicitly overridden.
  EFF_SCALE_DTYPE="${EFF_SCALE_DTYPE-float16}"
  if [[ -n "$EFF_SCALE_DTYPE" && "$arm" != "bf16" && "$extra_env" != *"SCALE_DTYPE="* ]]; then
    extra_env="$extra_env SCALE_DTYPE=$EFF_SCALE_DTYPE"
    pe_say "$arm: SCALE_DTYPE not pinned by arm_spec -> forcing $EFF_SCALE_DTYPE (figure fairness)"
  fi
  # Quantized prefill requires the Triton wrapper but invokes the same effective FlashAttention
  # kernel used by the BF16 FA3 path. Record the selected backend in each cell's provenance.
  _pb="$PREFILL_BACKEND"
  [[ "$arm" == "bf16" ]] || _pb="${PREFILL_BACKEND_QUANT:-triton}"
  extra_env="$extra_env PREFILL_BACKEND=$_pb"
  _armvar="EXTRA_ENV_${arm^^}"
  if [[ -n "${!_armvar:-}" ]]; then
    extra_env="$extra_env ${!_armvar}"
    pe_say "$arm: per-arm env ${!_armvar}"
  fi

  case "$PANEL" in
  decode)
    maxctx=0; for c in "${DEC_CTXS[@]}"; do (( c > maxctx )) && maxctx=$c; done
    _srvctx=$(( maxctx + DEC_OUT + 1024 ))
    (( _srvctx > NATIVE_CTX )) && _srvctx=$NATIVE_CTX
    # +DEC_OUT+1024, not the bare maxctx: the server refuses a request whose input+output EQUALS
    # its --context-length, and this panel's longest point is built as input = ctx - DEC_OUT, so
    # a server sized at exactly maxctx rejects that point with "Requested token count exceeds the
    # model's maximum context length" -- a boundary error that reads exactly like the OOM this
    # point was expected to hit (measured: bf16 ctx131072, 131008 + 64 = 131072).
    # Same margin run_model.sh uses. Qwen3-4B-Thinking's native window is 262144, so the margin
    # costs no rope scaling.
    # MEM-FRACTION BY MEASUREMENT, not by a constant. At batch 1 the pool only ever holds ONE
    # sequence, so the panel wants the SMALLEST pool that still fits the longest point -- every
    # byte beyond that is taken from the unchunked prefill beside it, which is what OOM'd at 0.85
    # on qwen (2.38 GiB refused with 7.04 GB free). But the right value is model-specific:
    # qwen3-4b needs >=0.54 (7.67 GiB weights + 18.0 GiB KV) and llama31-8b >=0.68 (16.1 + 16.0),
    # so the 0.65 that worked for qwen leaves llama's pool unable to hold one 128k sequence at
    # all. Walk up from the low end and take the first fraction whose pool holds the longest
    # sequence with a 5% margin.
    _chosen=""
    for _mf in $DEC_MEMFRAC_LADDER; do
      CHUNKED="$DEC_CHUNKED" MEMFRAC="$_mf" \
        start_server "$arm" "$engine" "$artifact" "$extra_env" "$_srvctx" 8 || { stop_server; continue; }
      _mtt=$(max_total_tokens)
      if (( _mtt * 100 >= maxctx * 105 )); then
        pe_say "$arm: mem-fraction $_mf -> pool $_mtt tok, holds the $maxctx point (chosen)"
        _chosen="$_mf"; break
      fi
      pe_say "$arm: mem-fraction $_mf -> pool $_mtt tok, too small for $maxctx; stepping up"
      stop_server
    done
    [[ -n "$_chosen" ]] || { pe_say "SKIP $arm: no mem-fraction in '$DEC_MEMFRAC_LADDER' fits $maxctx"; stop_server; continue; }
    for ctx in "${DEC_CTXS[@]}"; do
      _in=$(( ctx - DEC_OUT ))
      # keep input+output strictly BELOW the served context (see NATIVE_CTX above)
      (( _in + DEC_OUT >= _srvctx )) && _in=$(( _srvctx - DEC_OUT - 1 ))
      run_point "$arm" "ctx${ctx}" 1 "$_in" "$DEC_OUT" "$artifact" "$extra_env"
    done
    stop_server ;;
  prefill)
    maxctx=0; for c in "${PRE_CTXS[@]}"; do (( c > maxctx )) && maxctx=$c; done
    start_server "$arm" "$engine" "$artifact" "$extra_env" $(( maxctx + PRE_OUT + 64 )) 8 \
      || { stop_server; continue; }
    for ctx in "${PRE_CTXS[@]}"; do
      run_point "$arm" "ctx${ctx}" 1 "$ctx" "$PRE_OUT" "$artifact" "$extra_env"
    done
    stop_server ;;
  thru)
    bsmax=0; for b in "${THRU_BS[@]}"; do (( b > bsmax )) && bsmax=$b; done
    # All arms use MEMFRAC by default. THRU_QUANT_MEMFRAC is retained as an explicit override, and
    # the selected value is recorded in each cell's provenance.
    _mf="$MEMFRAC"
    [[ "$arm" == "bf16" ]] || _mf="$THRU_QUANT_MEMFRAC"
    MEMFRAC="$_mf" start_server "$arm" "$engine" "$artifact" "$extra_env" \
      $(( THRU_IN + THRU_OUT + 64 )) "$bsmax" || { stop_server; continue; }
    # Capacity from the ENGINE's own pool, not from a guess: the largest batch whose
    # (input + output) KV fits max_total_num_tokens. This is also the bf16 capacity line.
    mtt=$(max_total_tokens)
    capb=$(( mtt / (THRU_IN + THRU_OUT) ))
    pe_say "$arm: max_total_num_tokens=$mtt -> capacity $capb requests at $((THRU_IN+THRU_OUT)) tok"
    # THRU_CAP_FROM=<arm>: hold this arm's batch ceiling to that arm's measured capacity instead
    # of its own (for cq vs tasq). The two quantized arms then end their
    # curves at the SAME batch, so the panel compares throughput at equal concurrency rather than
    # also crediting whichever arm's bundle happens to buy a few more slots. The arm's own
    # measured capacity is still recorded (capacity_batch_own) -- only the grid is capped.
    _capown=$capb; _capsrc="own"
    if [[ -n "${THRU_CAP_FROM:-}" && "$arm" != "${THRU_CAP_FROM}" ]]; then
      _capf="$OUTDIR/${TAG}__thru_capacity__${THRU_KEY}${THRU_CAP_FROM}.json"
      pe_say "$arm: waiting for ${THRU_CAP_FROM}'s capacity ($_capf)"
      for _ in $(seq 1 360); do [[ -s "$_capf" ]] && break; sleep 10; done
      if [[ -s "$_capf" ]]; then
        _capref=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["capacity_batch"])' "$_capf" 2>/dev/null || echo 0)
        if (( _capref > 0 && _capref < capb )); then
          pe_say "$arm: capacity $capb -> $_capref (pinned to ${THRU_CAP_FROM})"
          capb=$_capref; _capsrc="$THRU_CAP_FROM"
        else
          pe_say "$arm: own capacity $capb <= ${THRU_CAP_FROM}'s $_capref -- not raised, grid stays at $capb"
        fi
      else
        pe_die "$arm: THRU_CAP_FROM=${THRU_CAP_FROM} but $_capf never appeared"
      fi
    fi
    echo "{\"model\":\"$TAG\",\"arm\":\"$arm\",\"max_total_num_tokens\":$mtt,\"tokens_per_request\":$((THRU_IN+THRU_OUT)),\"capacity_batch\":$capb,\"capacity_batch_own\":$_capown,\"capacity_source\":\"$_capsrc\"}" \
      > "$OUTDIR/${TAG}__thru_capacity__${THRU_KEY}${arm}.json"
    # The power-of-two grid stops at the last point under capacity, which leaves the curve short
    # of the ceiling it exists to show (for the paper, bf16 cap 6 and quantized cap 84). So the
    # arm's OWN measured capacity is run as a final point, and each curve ends where
    # that arm actually runs out. A point AT capacity is the pool exactly full and the scheduler
    # may refuse it -- that is a real answer too, and the cell records it either way.
    _ran_max=0
    for bs in "${THRU_BS[@]}"; do
      if (( bs > capb )); then pe_say "$arm/bs${bs}: ABOVE CAPACITY ($capb) -- not run"; continue; fi
      run_point "$arm" "${THRU_KEY}bs${bs}" "$bs" "$THRU_IN" "$THRU_OUT" "$artifact" "$extra_env"
      _ran_max=$bs
    done
    if (( capb > _ran_max && capb <= bsmax )) && [[ "${THRU_SKIP_CAPACITY:-0}" == "1" ]]; then
      # THRU_SKIP_CAPACITY=1 measures the power-of-two grid only and leaves the capacity point for
      # a later pass. The capacity point is the most expensive cell in the panel by far (bs84 at
      # 2048/32768 is ~2 h against bs64's ~1.5 h), and it is only worth paying for if the curve is
      # still climbing at the last grid point -- so the decision is made from the drawn curve
      # rather than up front. Run the second pass with the SAME BSS so --cuda-graph-max-bs, and
      # therefore the capture budget and the pool it leaves, are identical across the two passes.
      pe_say "$arm: capacity point bs=$capb SKIPPED (THRU_SKIP_CAPACITY=1)"
    elif (( capb > _ran_max && capb <= bsmax )); then
      pe_say "$arm: capacity point bs=$capb (grid stopped at $_ran_max)"
      run_point "$arm" "${THRU_KEY}bs${capb}" "$capb" "$THRU_IN" "$THRU_OUT" "$artifact" "$extra_env"
    elif (( capb > bsmax )); then
      pe_say "$arm: capacity $capb exceeds --cuda-graph-max-bs $bsmax; not run (would serve eager)"
    fi
    stop_server ;;
  *) pe_die "unknown panel '$PANEL'" ;;
  esac
done
pe_say "===== panel '$PANEL' done -- cells in $OUTDIR ====="
