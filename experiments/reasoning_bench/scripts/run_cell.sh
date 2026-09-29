#!/bin/bash
# One Table-2 cell: (model, benchmark) x all five methods, thinking on.
#
#   ./run_cell.sh Qwen/Qwen3-4B-Thinking-2507 lcb_v6
#
# Same skip-if-present contract as Table 1. Note the protocol differs from Table 1 and is set by
# config.sh's REASONING branch, not here: sampled decoding (0.6/0.95), 32k budget,
# PREFIX=64/RECENT=256. Expect many hours per cell.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../common.sh"
export REASONING=1
# Sampled decoding -> the seed is a real seed, so reasoning cells are multi-seed by default
# (Table 1 stays single-run because greedy decoding makes a second seed return the same answers).
export REASON_SEEDS="${REASON_SEEDS:-0 1 2}"
# Cap graph capture for the quantized arms. Their decode path has many more kernels per layer
# than bf16's, so capture uses more memory. Batches above the cap run eager; model outputs are
# unchanged.
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
pe_run_cell "$HERE/../outputs" "${1:?usage: run_cell.sh <hf-model> <bench> [methods...]}" \
            "${2:?usage: run_cell.sh <hf-model> <bench> [methods...]}" \
            "${@:3}" || exit 1
