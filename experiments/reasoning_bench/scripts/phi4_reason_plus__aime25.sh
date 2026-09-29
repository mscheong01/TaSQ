#!/bin/bash
# Table 2 cell: Phi-4-reasoning-plus x aime25 (vendored tasks/aime25, 30 problems).
# Re-runnable: methods already in ../outputs/ are reported and skipped.
#
# tasks/aime25 is aime24 with only the dataset fields changed and the SAME scorer file, so
# this column is directly comparable to the AIME'24 one beside it. phi4_env.sh is sourced for the
# same reason the other three phi-4 cells source it: 28 GB of bf16 weights need the measured
# memory fractions and the native 32768 window, not the 40960 the Qwen cells use.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/phi4_env.sh"
exec "$HERE/run_cell.sh" "microsoft/Phi-4-reasoning-plus" "aime25" "$@"
