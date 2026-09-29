#!/bin/bash
# Table 2 cell: qwen3_4b_think x aime24, all five methods.
# Re-runnable: methods already in ../outputs/ are reported and skipped.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/run_cell.sh" "Qwen/Qwen3-4B-Thinking-2507" "aime24" "$@"
