#!/bin/bash
# Table 2 cell: Phi-4-reasoning-plus x scibench (vendored tasks/scibench, 692 problems).
# Thinking model -> the reasoning chain (sampled 0.6/0.95, 32k budget, PREFIX=64/RECENT=256),
# same as this model's other three cells.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/phi4_env.sh"   # cuda-graph cap + memory fractions this model needs; see that file
exec "$HERE/run_cell.sh" "microsoft/Phi-4-reasoning-plus" "scibench" "$@"
