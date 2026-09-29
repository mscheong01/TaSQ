#!/bin/bash
# Table 2 cell: dsr1_llama8b x scibench (vendored tasks/scibench, 692 problems).
# A thinking model like Qwen3-Thinking: it opens a reasoning span before the answer, so it runs
# under the same reasoning chain (sampled decoding, 32k budget) rather than the short-task one.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/run_cell.sh" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "scibench" "$@"
