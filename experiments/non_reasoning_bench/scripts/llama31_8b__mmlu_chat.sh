#!/bin/bash
# Table 1 cell: llama31_8b x mmlu_chat (vendored tasks/mmlu_chat), all five methods.
# Generative CoT MMLU. The loglikelihood `mmlu` run could not tell the arms
# apart -- 4,572 prefill batches to 65 decode batches, so the quantized cache was never read.
# Not in make_table.py's BENCHES yet, by request.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/run_cell.sh" "NousResearch/Meta-Llama-3.1-8B-Instruct" "mmlu_chat" "$@"
