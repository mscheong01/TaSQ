#!/usr/bin/env python3
"""Table 2: reasoning (thinking-on) benchmarks. Model x method rows, benchmark columns.

Three seeds per cell, folded to mean+-std. Decoding here is sampled (temp 0.6 / top-p 0.95, 32k
budget), so the seed is a real seed -- unlike Table 1, whose greedy decoding makes repeats mere
replicates of batch scheduling. AIME24 is the binding case: 30 items means one problem is
3.33 pp against a per-arm std around 5 pp, so a single seed resolves nothing there.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_lib as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODELS = ["qwen3_4b_think", "dsr1_llama8b", "phi4_reason_plus"]
# SciBench: 692 open-ended college science problems (xw27/scibench), vendored
# as tasks/scibench.
#
# AIME'25 (tasks/aime25): the 2025 problem set under a harness byte-identical
# to aime24's, so the two AIME columns differ by the problems alone.
BENCHES = ["aime24", "aime25", "lcb_v6", "scibench"]

print(T.build(
    OUT, MODELS, BENCHES,
    caption="Reasoning benchmark accuracy under 1.25-bit KV quantization "
            "(thinking on, 32k generation budget, residual protocol prefix=64/recent=256).",
    label="tab:reasoning",
    expect_seeds=3,
    show_std=True,
    # Unweighted mean of the benchmark columns SHOWN, all-or-nothing per row (see table_lib
    # .row_average). A newly added column therefore blanks every average until that column has
    # run everywhere -- expected, and it fills back in on its own as the cells land. Its +- is the std of the average, propagated as sqrt(sum std_i^2)/k from
    # each benchmark's own across-seed std -- the benchmarks are separate runs, so their seeds
    # are independent, and this keeps "+-" meaning the same thing it means in every other cell.
    average=True,
    png_note="Mean$\\pm$std over 3 sampling seeds; T=0.6, top-p 0.95, 32k generation budget, "
             "residual protocol prefix=64/recent=256. Avg. is the unweighted mean of the "
             "columns shown (blank unless every one of them ran); its spread is "
             "propagated as sqrt(sum std^2)/k.",
))
