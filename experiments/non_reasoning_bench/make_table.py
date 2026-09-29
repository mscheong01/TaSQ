#!/usr/bin/env python3
"""Table 1: general benchmarks (thinking off). Model x method rows, benchmark columns.

Single run per cell. Greedy decoding is not bit-deterministic under continuous batching, but
two independent full runs of the identical bf16 config (llama31_8b) differ
by exactly one document per benchmark: GSM8K 83.78/83.85, HumanEval 70.12/70.12, MBPP
60.20/60.20, MATH500 42.00/41.80. That is far below the method gaps this table reports, so
replicates are optional (REPLICATE=n on any script) rather than required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_lib as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

MODELS = ["llama31_8b", "qwen3_4b"]
# BBH uses the vendored chat-safe variant (tasks/bbh_chat): upstream bbh_cot_fewshot
# stops on "\n\n"/"Q", which a chat-templated reply hits before its conclusion, and every
# doc scores [invalid]. The group yaml carries aggregate_metric_list (macro mean), so the
# 27 subtasks arrive already aggregated under the group key table_lib.METRICS expects.
BENCHES = ["gsm8k_cot_llama", "math500", "mbpp_instruct", "humaneval_instruct",
           "bbh_cot_fewshot_chat",
           # MMLU-CoT: the vendored generative variant (tasks/mmlu_chat),
           # 1,531 validation items over all 57 subjects. Rows whose NSN cell is still missing
           # print "--" here and their Avg. is suppressed, by the rule below.
           "mmlu_chat"]

print(T.build(
    OUT, MODELS, BENCHES,
    caption="General benchmark accuracy under 1.25-bit KV quantization. "
            "Best quantized method per column in bold.",
    label="tab:non_reasoning",
    show_std=True,
    # Unweighted mean of the six benchmark columns, and only when all six ran: a mean over
    # whichever benchmarks happen to be finished is a different quantity, indistinguishable
    # from the real one once printed. An incomplete row prints "--" and table_lib says which
    # column is missing.
    average=True,
    png_note="Single run per cell. Replicate spread measured on bf16/llama31_8b: "
             "GSM8K 0.07, HumanEval 0.00, MBPP 0.00, MATH500 0.20 points -- one document each. "
             "Avg. is the unweighted mean of the six columns; blank unless all six ran.",
))
