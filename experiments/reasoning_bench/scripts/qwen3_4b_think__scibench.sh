#!/bin/bash
# Table 2 cell: qwen3_4b_think x scibench (vendored tasks/scibench, 692 problems).
# Thinking model -> the reasoning chain (sampled 0.6/0.95, 32k budget, PREFIX=64/RECENT=256).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/run_cell.sh" "Qwen/Qwen3-4B-Thinking-2507" "scibench" "$@"
