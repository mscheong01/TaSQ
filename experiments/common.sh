#!/bin/bash
# Shared plumbing for the experiment harness. Sourced, not run.
#
# This layer does NOT reimplement the build and sweep scripts -- it points their own config at
# isolated artifact and work directories and calls scripts/run_sweep.sh. Serving, protocol, and
# method settings therefore stay in one place (`config.sh`).
set -uo pipefail

PE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$PE")"

# Evaluation runs read and write their own codebooks instead of reusing development artifacts.
export ARTIFACTS="$PE/artifacts"
export WORK="$PE/work"

# The metric key per benchmark lives in ONE place: table_lib.METRICS, which raises rather than
# guesses when a benchmark is unregistered. Do not add a second copy here -- a shell-side mapping
# has no runner reading it, so it drifts silently while looking authoritative.

pe_say() { echo "[pe] $* $(date -Is)"; }
pe_die() { echo "[pe] FAILED: $*" >&2; exit 1; }

# emit_result <outdir> <model_tag> <bench> <method> <seed|-> <src_json>
#
# Copies a finished lm_eval result into the table's input directory under the naming contract,
# wrapped with the provenance the table builder checks. Recording the artifact hash and the
# protocol is what lets the builder refuse to average cells that were produced by different
# codebooks or a different residual policy -- both of which have silently happened before.
emit_result() {
  local outdir=$1 tag=$2 bench=$3 method=$4 seed=$5 src=$6
  [[ -s "$src" ]] || { echo "[pe] no result to emit: $src" >&2; return 1; }
  local name; name=$(SEED="${seed/#-/}" pe_cell_name "$tag" "$bench" "$method")
  local art="$ARTIFACTS/${tag}_$(pe_artifact_suffix "$method")"
  # bf16 has no artifact by design; every other arm has exactly one primary bundle, but it is
  # named differently per engine (nsn ships nsn_bundle.pt, the int2 arms codebook.pt). Hashing
  # only codebook.pt silently recorded "n/a" for nsn -- i.e. no artifact provenance at all on the
  # arm whose bundle we most recently rebuilt.
  local ahash="n/a" f
  for f in "$art/codebook.pt" "$art/nsn_bundle.pt"; do
    [[ -f "$f" ]] && { ahash=$(md5sum "$f" | cut -c1-12); break; }
  done
  if [[ "$method" != "bf16" && "$ahash" == "n/a" ]]; then
    echo "[pe] WARNING no bundle found under $art -- artifact provenance will be blank" >&2
  fi
  mkdir -p "$outdir"
  python3 - "$src" "$outdir/$name.json" "$tag" "$bench" "$method" "$seed" "$ahash" <<'PY'
import json, os, subprocess, sys, socket, datetime
src, dst, tag, bench, method, seed, ahash = sys.argv[1:8]
res = json.load(open(src))
def git(*a):
    # cwd MUST be pinned to the repo: this runs wherever the caller's driver happened to be, and
    # the parent of the repo is not a git checkout, so an unpinned call silently recorded
    # commit="unknown" while leaking git usage text into the log.
    try:
        return subprocess.check_output(["git", *a], text=True, cwd=os.environ["PE_REPO"],
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
json.dump({
    "model": tag, "bench": bench, "method": method,
    "seed": None if seed == "-" else int(seed),
    "results": res.get("results", res),
    "provenance": {
        "commit": git("rev-parse", "--short", "HEAD"),
        # TRACKED modifications only. Untracked files are other experiments' scratch sitting
        # in this shared working tree; they cannot change the code that produced this
        # cell, and counting them made every run report dirty and the flag useless.
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "dirty_files": git("diff", "--name-only", "HEAD") or None,
        "artifact_md5": ahash,
        # Recorded because `nova` and `novabf16` are the same codebook and the same protocol,
        # and differ ONLY here. Without it the two rows carry byte-identical provenance and
        # nothing in the artifact could tell them apart after the fact.
        "scale_dtype": os.environ.get("PE_SCALE_DTYPE"),
        "prefix_tokens": os.environ.get("PREFIX_TOKENS"),
        "recent_tokens": os.environ.get("RECENT_TOKENS"),
        "num_concurrent": os.environ.get("NUM_CONCURRENT"),
        # recorded alongside num_concurrent because the two together define the load: the client
        # rate and the per-replica server cap throttle each other, so one without the other does
        # not describe how the cell was actually served
        "max_running": os.environ.get("MAX_RUNNING"),
        "host": socket.gethostname(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    },
}, open(dst, "w"), indent=1)
print(f"[pe] wrote {dst}")
PY
}

# pe_cell_name <tag> <bench> <method> -> the outputs/ basename (no .json)
pe_cell_name() {
  local n="$1__$2__$3"
  [[ -n "${SEED:-}" ]]      && n="${n}__s${SEED}"
  [[ -n "${REPLICATE:-}" ]] && n="${n}__r${REPLICATE}"
  echo "$n"
}

# A completed legacy baseline also satisfies the skip check. New writes use bf16.
pe_cell_exists() {
  local outdir=$1 model=$2 bench=$3 method=$4
  [[ -s "$outdir/$(pe_cell_name "$model" "$bench" "$method").json" ]] && return 0
  [[ "$method" == "bf16" ]] &&
    [[ -s "$outdir/$(pe_cell_name "$model" "$bench" fp16).json" ]]
}

# pe_run_cell <outdir> <hf-model> <bench> <methods...>
#
# One table cell = one (model, bench) across the method axis. Any node can run any cell script;
# results land in outputs/ under the naming contract and the table builder merges them.
#
# A method whose output already exists is reported and skipped -- re-running a script is always
# safe and costs nothing. FORCE=1 re-runs regardless. If every requested method is already
# present the script exits without touching a GPU at all.
pe_run_cell() {
  local outdir=$1 model=$2 bench=$3; shift 3
  # Evaluation methods: bf16 reference, NSNQuant 1-bit, NovaKV, CQ g8, TaSQ g8.
  #
  # The NovaKV arm is `nova1b`. The two arms it replaces were both
  # wrong for this column in ways that had nothing to do with NovaKV's method:
  #   * their V was a symmetric SIGN quantiser, not OSCAR's asymmetric min/max -- a substitution
  #     forced by a decode-kernel limitation (no scalar-V branch under a wide-G K codebook), not
  #     chosen. The published NovaKV 2-bit build runs OSCAR's scalar V unchanged.
  #   * their 1-bit V code was stored in int2 crumbs, so the row's nominal 1.125 and its
  #     allocated 2.25 disagreed by a whole bit.
  # nova1b fixes both: OSCAR's own quantiser with the level count at 2, stored 1 bit wide
  # (nominal == allocated == 1.25 V), and K at 10 bits/group so the codebook capacity matches
  # CQ/TaSQ's 1024 centroids. NOTE it is therefore K 1.375 + V 1.250, NOT bit-matched to CQ's
  # 1.25/1.25 -- do not describe this axis as equal-BPA without saying which side differs.
  local methods=("$@"); [[ $# -eq 0 ]] && methods=(bf16 nsn nova1b cq tasq)
  source "$REPO/config.sh"
  local i
  for i in "${!methods[@]}"; do methods[$i]=$(canonical_method "${methods[$i]}"); done
  # config.sh sets these as plain shell variables; emit_result reads them from the ENVIRONMENT of
  # a python subprocess, so without exporting them the residual protocol -- the single most
  # important thing a cell records -- lands in the provenance as null.
  export PREFIX_TOKENS RECENT_TOKENS NUM_CONCURRENT MAX_RUNNING PE_REPO="$REPO"
  # Checked BEFORE anything is served: a cell whose protocol cannot be recorded is not worth the
  # GPU time, and failing afterwards would discard a finished evaluation.
  [[ -n "${PREFIX_TOKENS:-}" && -n "${RECENT_TOKENS:-}" ]] \
    || pe_die "residual protocol unset after sourcing config.sh -- refusing to run a cell whose provenance cannot be recorded"
  local tag; tag=$(model_tag "$model")
  # Seeds must be known BEFORE the done-check: a multi-seed cell is written as one file per seed
  # (..._tasq__s0.json), so a check built without the suffix looks for a name that is never
  # created and every reasoning cell re-runs from scratch -- the skip contract failing precisely
  # on the cells that cost hours. A cell counts as done only when EVERY seed file is present;
  # a partial cell (server died after seed 0) re-runs, and run_sweep then skips the seeds whose
  # raw results already exist.
  local seeds="0"
  [[ "${REASONING:-0}" == "1" ]] && seeds="${REASON_SEEDS:-0}"
  # Labelled by POLICY, not by how many seeds this invocation happens to run: the reasoning table
  # is multi-seed, so its cells always carry __sN. Deciding from the count meant re-running a
  # single lost seed wrote an unsuffixed file, which would then be double-counted the next time
  # all three seeds ran.
  local multi=0; [[ "${REASONING:-0}" == "1" ]] && multi=1

  local todo=() done_list=()
  for m in "${methods[@]}"; do
    local complete=1 s
    for s in $seeds; do
      if [[ "$multi" == 1 ]]; then
        SEED="$s" pe_cell_exists "$outdir" "$tag" "$bench" "$m" || complete=0
      else
        pe_cell_exists "$outdir" "$tag" "$bench" "$m" || complete=0
      fi
    done
    if [[ "$complete" == 1 && "${FORCE:-0}" != "1" ]]; then done_list+=("$m"); else todo+=("$m"); fi
  done
  [[ ${#done_list[@]} -gt 0 ]] && pe_say "ALREADY RUN, skipping: ${done_list[*]}  (in $outdir)"
  if [[ ${#todo[@]} -eq 0 ]]; then
    pe_say "nothing to do for $tag/$bench -- all ${#methods[@]} methods already in outputs/. Use FORCE=1 to re-run."
    return 0
  fi
  pe_say "running $tag/$bench: ${todo[*]}"

  local raw="$WORK/$tag/results"
  # $seeds was resolved above, before the done-check. The reasoning path is sampled and runs every
  # seed against one set of servers, writing <tag>_<arm>_<seed>_<task>.json; the short-task path is
  # greedy few-shot, where a second seed returns the same answers, so it stays single-seed at 0.
  for m in "${todo[@]}"; do
    # What the arm will actually serve the per-token scale as. Derived from the arm rather than
    # read back from the server, because emit_result runs after the servers are already gone.
    # Keep in step with scripts/run_sweep.sh's arm_spec and serve_method.sh: VQ arms use the
    # float16 default unless the method explicitly pins another dtype.
    case "$m" in
      # nova1b pins bfloat16 for the same reason novabf16 did -- it is the dtype the published
      # NovaKV accounting assumes -- and it now also sets the V side's cost: the (scale, zero)
      # pair is 0.25 bit/coord at bf16 against 0.5 at fp32, which is the difference between the
      # row costing 1.25 and 1.50 on V.
      nova1b)   export PE_SCALE_DTYPE=bfloat16 ;;
      # TaSQ is fit at fp16 (simquant post-norm-bits 16); CQ and TaSQ serve at the fp16 default.
      cq|tasq)  export PE_SCALE_DTYPE=float16 ;;
      *)        export PE_SCALE_DTYPE=float32 ;;
    esac
    # run_sweep.sh also skips an existing raw result; that is the wrong behaviour here because a
    # raw file may be a limited smoke run or a different vintage. outputs/ is the authority.
    for s in $seeds; do rm -f "$raw/${tag}_${m}_${s}_${bench}.json"; done
    if [[ "${REASONING:-0}" == "1" ]]; then
      REASONING_TASKS="$bench" REASON_SEEDS="$seeds" bash "$REPO/scripts/run_sweep.sh" "$model" "$m"
    else
      SHORT_TASKS="$bench" bash "$REPO/scripts/run_sweep.sh" "$model" "$m"
    fi
    for s in $seeds; do
      local tag_seed="-"
      # Only label the output file with a seed when there is genuinely more than one; a single
      # greedy run must keep the unsuffixed name the table builder already expects. Uses the same
      # $multi the done-check used, so the two can never disagree about a cell's filename.
      [[ "$multi" == 1 ]] && tag_seed="$s"
      emit_result "$outdir" "$tag" "$bench" "$m" "$tag_seed" "$raw/${tag}_${m}_${s}_${bench}.json" \
        || pe_say "NO RESULT for $m/$bench seed $s -- see $WORK/$tag/logs/eval_${m}_${bench}_s${s}.log"
    done
  done
}

# arm name -> artifact directory suffix, mirroring scripts/run_sweep.sh's arm_spec
pe_artifact_suffix() {
  case "$1" in
    bf16) echo "" ;;
    nsn)    echo "nsn_1bit" ;;
    # Derived, not hardcoded: config.sh tags the directory by the calibration it was built with,
    # so a fixed "calib16" here silently missed the 64c16 bundles the paper uses.
    nova1b) echo "nova1375_${NOVA_CALIB_TAG:-calib16}" ;;
    cq)     echo "cq_g8" ;;
    tasq)   echo "tasq_g8" ;;
    *)      echo "$1" ;;
  esac
}
