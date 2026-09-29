#!/bin/bash
# One Table-1 cell: (model, benchmark) x all five methods, thinking off.
#
#   ./run_cell.sh NousResearch/Meta-Llama-3.1-8B-Instruct gsm8k_cot_llama
#   ./run_cell.sh Qwen/Qwen3-8B math500 tasq nsn        # subset of methods
#   FORCE=1 ./run_cell.sh ... ; REPLICATE=2 ./run_cell.sh ...   # re-run / add a replicate
#
# Safe to re-run on any node: a method already present in outputs/ is reported and skipped, and
# if all of them are present the script exits without starting a server.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../common.sh"
export REASONING=0            # short-task chain: greedy, chat template, PREFIX=0/RECENT=64
pe_run_cell "$HERE/../outputs" "${1:?usage: run_cell.sh <hf-model> <bench> [methods...]}" \
            "${2:?usage: run_cell.sh <hf-model> <bench> [methods...]}" \
            "${@:3}" || exit 1
