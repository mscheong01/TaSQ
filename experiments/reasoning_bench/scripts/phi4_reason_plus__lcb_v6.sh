#!/bin/bash
# Table 2 cell: Phi-4-reasoning-plus x lcb_v6, all five methods.
#
# The fused-qkv calibration shim
# lives in calibration/shims/phi4/, and all four bundles are built. It needs a >=40 GB card -- 28 GB of bf16
# weights do not fit a 24 GB one -- so run this on the RTX 6000 Ada node, not the 3090 node.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/phi4_env.sh"   # cuda-graph cap + memory fractions this model needs; see that file
exec "$HERE/run_cell.sh" "microsoft/Phi-4-reasoning-plus" "lcb_v6" "$@"
