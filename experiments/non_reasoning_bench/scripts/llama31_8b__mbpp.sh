#!/bin/bash
# Table 1 cell: llama31_8b x mbpp_instruct, all five methods.
# Re-runnable: methods already in ../outputs/ are reported and skipped.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/run_cell.sh" "NousResearch/Meta-Llama-3.1-8B-Instruct" "mbpp_instruct" "$@"
