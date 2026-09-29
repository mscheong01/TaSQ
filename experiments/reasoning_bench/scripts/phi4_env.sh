# Serving knobs Phi-4-reasoning-plus needs. Sourced by the three phi4 cell scripts.
#
# Measured on the RTX 6000 Ada node, per arm:
#
#   CUDA_GRAPH_MAX_BS=8
#     The VQ decode path has many more kernels per layer than the bf16 one, and graph capture
#     costs 6.06 GB at bs<=8 against roughly twice that at bs<=16. A 14B leaves only 11-13 GB
#     after 27.3 GB of weights, so bf16 captures fine while ALL FOUR quantized arms die with
#     "Capture cuda graph failed: CUDA error: out of memory" -- and run_sweep then polls the dead
#     servers' /health for 15 minutes per arm before moving on. MAX_RUNNING is 16, so capping at 8
#     costs the top half of the batch range: batches above the cap run eager, which changes speed
#     and never output.
#
#   MEM_FRACTION = 0.78
#     With capture down to 6 GB there is room for a full pool again. Verified boots:
#     nsn max_total_num_tokens=48808, nova/cq/tasq=312368 (their KV really is ~2 bits), against
#     bf16's 53738. At 0.70 the nsn pool drops to 29088, below one 30k generation per replica,
#     which serialises the arm for no reason.
#
# These are per-model, not per-benchmark, hence one file rather than three copies.
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
export MEM_FRACTION="${MEM_FRACTION:-0.78}"
# Native max_position_embeddings is 32768, not the 40960 the Qwen cells use; asking for more than
# the model was trained for is a silent quality change. The budget leaves ~2k for the prompt.
export REASON_CTX="${REASON_CTX:-32768}"
export REASON_MAX_GEN="${REASON_MAX_GEN:-30720}"
