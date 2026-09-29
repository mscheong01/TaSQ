#!/bin/bash
# Table 3, Qwen3-4B-Thinking-2507: 4k/8k/16k/32k/64k x 3 seeds x 5 methods.
#
# 262144 native context, so no rope scaling at any length here either.
#
# Thinking models use an 8,192-token response budget so retrieval is scored after the reasoning
# span closes.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NIAH_MAX_TOKENS="${NIAH_MAX_TOKENS:-8192}"
exec "$HERE/run_model.sh" "Qwen/Qwen3-4B-Thinking-2507" "$@"
