#!/bin/bash
# Table 3, Phi-4-reasoning-plus: 8k/16k/32k x 3 seeds x 5 methods.
#
# THREE lengths, not five, and that is a property of the model rather than a choice: its
# max_position_embeddings is 32768. 64k and 128k cannot run without RoPE scaling (YaRN), and
# scaling moves the BF16 baseline itself, which would make this row incomparable with the other
# two in the same table. The same reasoning already keeps Qwen3-4B and Qwen3-8B (40960) out of
# this table entirely. Even 32768 is the edge: the prompt, question and answer all have to fit
# inside that window alongside the haystack.
#
# NIAH_MAX_TOKENS=8192, as for Qwen3-4B-Thinking: a reasoning model has to close its thinking span
# before it emits the needle, so the 128-token budget the rows carry would score it wrong no
# matter what it retrieved.
#
# phi4_env.sh carries the 14B serving knobs (cuda-graph cap, memory fractions) measured per arm on
# Without the graph cap every quantized arm dies in capture on this model.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../reasoning_bench/scripts/phi4_env.sh"
export NIAH_MAX_TOKENS="${NIAH_MAX_TOKENS:-8192}"

# Filter rather than trust: a stray CTX=131072 here would spend an hour building rows the model
# cannot be asked about, then fail at serve time with a context-length error.
NATIVE=32768
KEEP=""; DROP=""
# Default to the FULL table grid and let the filter below remove what this model cannot do.
# A hardcoded short list here silently drops lengths when the grid changes; the filter is the
# only thing that should be model-specific.
for c in ${CTX:-4096 8192 16384 32768 65536}; do
  if [ "$c" -le "$NATIVE" ]; then KEEP="$KEEP $c"; else DROP="$DROP $c"; fi
done
[ -n "$DROP" ] && echo "[phi4-niah] dropping length(s)$DROP -- above this model's native $NATIVE"
export CTX="${KEEP# }"
[ -n "$CTX" ] || { echo "[phi4-niah] no runnable lengths left"; exit 2; }
echo "[phi4-niah] lengths: $CTX   max_tokens=$NIAH_MAX_TOKENS"
exec "$HERE/run_model.sh" "microsoft/Phi-4-reasoning-plus" "$@"
