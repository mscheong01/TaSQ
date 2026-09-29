#!/bin/bash
# Build evaluation codebooks into experiments/artifacts/.
#
#   ./build/build_artifacts.sh Qwen/Qwen3-8B              # all arms
#   ./build/build_artifacts.sh Qwen/Qwen3-8B nsn cq tasq   # subset
#
# Thin wrapper: codebooks/build_bundles.sh does the work with isolated ARTIFACTS/WORK directories.
# Idempotent -- an artifact that exists is skipped.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
MODEL="${1:?usage: build_artifacts.sh <hf-model> [arms...]}"; shift
# Methods included in the evaluation.
ARMS=("$@"); [[ $# -eq 0 ]] && ARMS=(nsn nova1b cq tasq)
pe_say "building into $ARTIFACTS (work: $WORK): ${ARMS[*]}"
bash "$REPO/codebooks/build_bundles.sh" "$MODEL" "${ARMS[@]}"
