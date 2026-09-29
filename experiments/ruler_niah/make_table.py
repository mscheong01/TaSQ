#!/usr/bin/env python3
"""Table 3: RULER NIAH. Model x method rows, CONTEXT LENGTH columns.

Three seeds per cell (needle position/haystack sampling), folded to mean+-std. Unlike the other
two tables the column axis is a context length, so the bench name encodes it and the column
labels are lengths.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_lib as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
# Only models whose native context reaches 128k -- Qwen3-4B/8B stop at 40960 and would need
# YaRN, which changes the BF16 baseline itself.
#
# phi4_reason_plus is back in. It was dropped while its only NIAH cells were
# bf16 -- a row whose every quantized column renders "--" tells the reader nothing -- with the note
# to restore it once the grid was filled. It is now filled: 48 cells (cq, nsn, novasq10,
# tasq x 4 lengths x 3 seeds) produced 09-18..09-21, all 200 rows, none set aside short.
#
# ITS 64k COLUMN IS BLANK ON PURPOSE. phi-4's max_position_embeddings is 32768, so 65536 needs
# YaRN, and RoPE scaling moves the BF16 baseline itself -- which is the same reason Qwen3-4B/8B
# (40960) are absent from this table entirely. ruler_niah/scripts/phi4_reason_plus.sh filters the
# grid against that window and drops the length before a row is ever built, so the blank is a
# property of the model, not a missing run. Read the phi-4 row across 4k-32k only.
MODELS = ["llama31_8b", "qwen3_4b_think", "phi4_reason_plus"]
CTX = [4096, 8192, 16384, 32768, 65536]   # 128k dropped; see scripts/run_model.sh
NEEDLE = os.environ.get("NIAH_TASK", "niah_single_2")

# Column key = the bench field written by the runner: "<task>@<ctx>".
# The key the runner actually writes is "niah@<ctx>" (run_model.sh's cell_file / emit_niah.py).
# It carries no needle name: a run covers all eight RULER-NIAH subtasks and scores their mean.
# NEEDLE supplies the shared metric definition only. Using a needle-prefixed result key here made
# every finished cell invisible -- and a missing cell renders "--", so the empty table looked
# exactly like "not run yet".
BENCHES = [f"niah@{c}" for c in CTX]
for b, c in zip(BENCHES, CTX):
    T.METRICS[b] = T.METRICS[NEEDLE]
    T.PRETTY_BENCH[b] = f"{c // 1024}k"

print(T.build(
    OUT, MODELS, BENCHES,
    caption="RULER NIAH accuracy versus context length under low-bit KV cache quantization. "
            "Mean$\\pm$std over 3 seeds.",
    label="tab:niah",
    expect_seeds=3,
    show_std=True,
    png_note="3 seeds per cell (needle placement / haystack sampling). BF16 needs TP>=2 past 32k on 24 GB cards.",
))
