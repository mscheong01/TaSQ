# vendor/kvquant

Trimmed copy of the KVQuant pipeline — only what the `cq`/`tasq` arms actually execute:
Fisher gradients (`run_fisher.py` + `gradients/datautils.py`) and the coupled-VQ centroid fit
(`llama_simquant.py` + `kvquant/`, plus the `kmeans_tools` CUDA extension).

Left behind: the dbrx quantizer variant and `gradients/utils/` (36 HuggingFace repo-maintenance
scripts, none of them reachable from this path).

Build the extension once per environment:

    cd third_party/kvquant && pip install -e . --no-build-isolation

Upstream: https://github.com/SqueezeAILab/KVQuant
