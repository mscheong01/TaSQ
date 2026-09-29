#!/bin/bash
# Table 3, Llama-3.1-8B-Instruct: 4k/8k/16k/32k/64k x 3 seeds x 5 methods.
#
# 131072 native context, so no rope scaling is involved at any length in this table -- but the
# grid stops at 64k anyway (see run_model.sh: a 131k prompt cannot be prefilled beside a pool
# large enough to hold it, and chunked prefill is refused for the quantized arms).
# Non-thinking model -> a 128-token answer budget is enough.
#
# The bf16 row does NOT fit at TP=1 on 24 GB cards past 32k (KV alone is 8 GiB at 64k and 16 GiB
# at 128k, on top of 16 GiB of weights), so run it with TP=2 there. The quantized arms are fine
# at TP=1 throughout -- which is the point of the table.
#
#   ./llama31_8b.sh                        # quantized arms + bf16 up to 32k
#   GPUS="0 1 2 3 4 5 6 7" TP=2 ./llama31_8b.sh bf16     # the long bf16 cells
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/run_model.sh" "NousResearch/Meta-Llama-3.1-8B-Instruct" "$@"
