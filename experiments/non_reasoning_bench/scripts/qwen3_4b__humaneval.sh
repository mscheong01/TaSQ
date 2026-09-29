#!/bin/bash
# Table 1 cell: qwen3_4b x humaneval_instruct, all five methods.
# Re-runnable: methods already in ../outputs/ are reported and skipped.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/run_cell.sh" "Qwen/Qwen3-4B" "humaneval_instruct" "$@"
