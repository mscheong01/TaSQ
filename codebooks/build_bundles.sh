#!/bin/bash
# Build the serving artifacts for one model: nsn, nova1b, cq, tasq.
#
#   ./codebooks/build_bundles.sh Qwen/Qwen3-8B                 # all four
#   ./codebooks/build_bundles.sh Qwen/Qwen3-8B nsn nova        # only the ones with no KVQuant dep
#
# Idempotent: an artifact that already exists is skipped, so a crashed run resumes by re-running
# the same command. Intermediates (dump, Fisher, pickles, logs) go under work/<tag>/.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
have_conda

MODEL="${1:?usage: build_bundles.sh <hf-model> [nsn nova1b cq tasq]}"; shift
STEPS=("$@"); [[ $# -eq 0 ]] && STEPS=(nsn nova1b cq tasq)
TAG="$(model_tag "$MODEL")"
HF="${MODEL//\//_}"
W="$WORK/$TAG"; LOG="$W/logs"; mkdir -p "$LOG"
DUMP="$W/qkv_dump"
# Fisher statistics are calibration-corpus specific, so the path includes the dataset and sample
# configuration.
FISHER="$W/fisher_${CALIB_DATASET}${CALIB_TAG}"
PICKLE="$W/quantizers_${HF}_${COUPLED}c${ABITS}b_${CALIB_DATASET}${CALIB_TAG}.pickle"
# The method is not configurable. make_tasq.py carries exactly one: W from the causal diag(M)
# estimator, key transformed W^1/2 k, pooled-RMS normalised, then grouped on the covariance of
# THAT vector. The pkl name therefore needs no method tag.
TASQPKL="$W/tasq_c${COUPLED}_${HF}_${CALIB_DATASET}${CALIB_TAG}.pkl"
# TaSQ needs its OWN centroids: the codebook must be fit in the space the online encoder actually
# quantizes in -- after the channel weight, the per-token scale and the permutation -- so simquant
# is re-run with --tasq-file. Feeding CQ's centroids, fit on unweighted contiguous groups, would
# put codebook and data in different coordinate systems.
TASQPICKLE="$W/quantizers_${HF}_${COUPLED}c${ABITS}b_${CALIB_DATASET}${CALIB_TAG}_tasq.pickle"
TASQDIR="$ARTIFACTS/${TAG}_tasq_g8"

# KV-head count for cross-head normalization. Read from the model config and allow an explicit
# override; fall back to 8 if model metadata is unavailable.
HEAD_GROUP="${HEAD_GROUP:-$(python3 - "$MODEL" <<'PYCFG' 2>/dev/null || echo 8
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
n = getattr(c, "num_key_value_heads", None) or getattr(c, "num_attention_heads")
print(int(n))
PYCFG
)}"
[[ "$HEAD_GROUP" =~ ^[0-9]+$ ]] || HEAD_GROUP=8

# Fisher must SEE several GPUs (an 8B model's params+grads exceed one 24 GB card), so it gets the
# whole node by default -- config.sh detected the list. Pin CAL_GPUS to share the machine.
CAL_GPUS="${CAL_GPUS:-$(echo $GPUS | tr ' ' ',')}"
ONE_GPU="${ONE_GPU:-0}"

has() { [[ " ${STEPS[*]} " == *" $1 "* ]]; }

# ------------------------------------------------------------------ nsn (no calibration)
if has nsn && [[ ! -f "$ARTIFACTS/${TAG}_nsn_1bit/nsn_bundle.pt" ]]; then
  conda activate "$ENV_SERVE"
  # Use the released NSNQuant codebook vendored under assets/nsn/ for bitwise reproducibility.
  # NSN_CODEBOOK may point to another codebook for controlled ablations.
  if [[ -z "$NSN_CODEBOOK" ]]; then
    NSN_CODEBOOK="$REPO/assets/nsn/1bit_codebook.pt"
  fi
  [[ -f "$NSN_CODEBOOK" ]] || die "\$NSN_CODEBOOK does not exist: $NSN_CODEBOOK"
  say "nsn bundle"
  mkdir -p "$ARTIFACTS/${TAG}_nsn_1bit"
  python3 "$CODEBOOKS/build_nsn_bundle.py" \
    --codebook_path "$NSN_CODEBOOK" --model_name "$MODEL" --window_size "$NSN_WINDOW" \
    --out "$ARTIFACTS/${TAG}_nsn_1bit/nsn_bundle.pt" > "$LOG/nsn.log" 2>&1 \
    || die "build_nsn_bundle (see $LOG/nsn.log)"
fi

# NovaKV at CQ/TaSQ's code width: G=8 / 10 bits-per-group (1024 centroids), so every VQ arm gets
# the same codebook capacity. K costs 1.250 and the scalar V tier 0.125, for 1.375 total.
#
# NOVA_POOL_STRIDE (default 4) controls token subsampling for k-means. At stride 4 this dump gives
# 8192 samples per head, 8 per centroid at 1024 centroids -- thin. Set it to 1 for 32768 per head.
if has nova1b; then
  conda activate "$ENV_SERVE"
  ART10="$ARTIFACTS/${TAG}_nova1375_${NOVA_CALIB_TAG}${NOVA_BPG_SUFFIX:-}"
  DUMP16="$W/qkv_dump_${NOVA_CALIB_TAG}"
  mkdir -p "$ART10"

  # Self-contained: prompts, dump and rotations are produced here when absent.
  if [[ ! -f "$DUMP16.prompts.jsonl" ]]; then
    say "nova1b: calib prompts (gpqa_code windows)"
    PYTHONPATH="$VENDOR" python3 "$CALIB/make_calib16_prompts.py" --model "$MODEL" \
      --nsamples "$CALIB_NSAMPLES" --seqlen "$CALIB_SEQLEN" --seed "$CALIB_SEED" \
      --mix-code-n "$CALIB_MIX_CODE_N" \
      --out "$DUMP16.prompts.jsonl" > "$LOG/calib_prompts_nova1b.log" 2>&1 \
      || die "make_calib16_prompts (see $LOG/calib_prompts_nova1b.log)"
  fi
  if [[ ! -d "$DUMP16" ]]; then
    say "nova1b: QKV dump ($NOVA_CALIB_TAG)"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES="$ONE_GPU" PYTHONPATH="$REPO/python" \
      python3 "$CALIB/"dump_qkv.py --model "$MODEL" --prompts "$DUMP16.prompts.jsonl" \
        --num-prompts "$CALIB_NSAMPLES" --max-seq-len "$CALIB_SEQLEN" --no-chat-template \
        --out "$DUMP16" ) > "$LOG/dump_nova1b.log" 2>&1 || die "dump_qkv (see $LOG/dump_nova1b.log)"
  fi
  NL16=$(ls -d "$DUMP16"/layer_* 2>/dev/null | wc -l)
  if [[ ! -f "$ART10/k_rotation_qqt_r_h_pbr.pt" ]]; then
    say "nova1b: K/V rotations"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES="$ONE_GPU" PYTHONPATH="$REPO/python" \
      python3 "$CALIB/"compute_kv_rotation.py --dump-path "$DUMP16" \
        --output-dir "$ART10" --chunk-id all --head-dim "$HEAD_DIM" \
        --method qqt_sst --composition r_h_pbr --num-layers "$NL16" ) \
      > "$LOG/rotation_nova1b.log" 2>&1 || die "compute_kv_rotation (see $LOG/rotation_nova1b.log)"
  fi
  if [[ ! -f "$ART10/codebook.pt" ]]; then
    say "nova1b: VQ codebook (G=8 bits-per-group=10, ${NOVA_CALIB_TAG}, pool-stride=${NOVA_POOL_STRIDE:-4}) -> $ART10"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES="$ONE_GPU" PYTHONPATH="$REPO/python" \
      python3 "$CALIB/"fit_vq_codebook.py --dump "$DUMP16" \
        --out "$ART10/codebook.pt" --G 8 --bits-per-group 10 \
        --pool-stride "${NOVA_POOL_STRIDE:-4}" \
        --grouping stratified ) > "$LOG/fit_vq_nova1b.log" 2>&1 \
      || die "fit_vq_codebook nova1b (see $LOG/fit_vq_nova1b.log)"
  fi
  # No sign-V codebook here. This arm serves V on the scalar OSCAR tier
  # (SGLANG_V_INT_MAX_Q=1 SGLANG_V_INT_BITS=1), which needs no artifact.
fi

# ------------------------------------------------------------------ cq / tasq (needs KVQuant)
if has cq || has tasq; then
  conda activate "$ENV_SERVE"
  # The vendored extension must be built once per env: cd third_party/kvquant && pip install -e .
  # KMEANS_LOCAL lets a node whose torch ABI does not match the vendored .so point at its own
  # build without overwriting the shared file (see tools/run_with_local_kmeans.py). Unset =
  # unchanged behaviour.
  PY_KVQ=(python3)
  if [[ -n "${KMEANS_LOCAL:-}" ]]; then
    PY_KVQ=(python3 "$CODEBOOKS/run_with_local_kmeans.py")
    PYTHONPATH="$KMEANS_LOCAL${PYTHONPATH:+:$PYTHONPATH}" python3 -c "import torch, kmeans_tools" 2>/dev/null \
      || die "kmeans_tools not importable from \$KMEANS_LOCAL=$KMEANS_LOCAL"
  else
    python3 -c "import torch, kmeans_tools" 2>/dev/null \
      || die "kmeans_tools not built in env '$ENV_SERVE' -- run: (cd $VENDOR && pip install -e . --no-build-isolation)"
  fi

  if [[ ! -d "$FISHER" ]]; then
    say "Fisher gradients (GPUs $CAL_GPUS)"
    ( cd "$VENDOR" && CUDA_VISIBLE_DEVICES="$CAL_GPUS" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 run_fisher.py --model_name_or_path "$MODEL" --output_dir "$FISHER" \
        --dataset "$CALIB_DATASET" --seqlen "$CALIB_SEQLEN" --maxseqlen 32768 \
        --num_examples "$CALIB_NSAMPLES" --mix_code_n "$CALIB_MIX_CODE_N" ) > "$LOG/fisher.log" 2>&1 \
      || die "run-fisher-my (see $LOG/fisher.log)"
  fi

  # `has cq`, not just the file check: TaSQ fits its own centroids under --tasq-file and never
  # reads this pickle, so without it a tasq-only build sits through CQ's fit for nothing.
  if has cq && [[ ! -f "$PICKLE" ]]; then
    say "CQ centroids (simquant ${COUPLED}c${ABITS}b)"
    ( cd "$VENDOR" && CUDA_VISIBLE_DEVICES="$CAL_GPUS" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY_KVQ[@]}" llama_simquant.py "$MODEL" --num_coupled "$COUPLED" --abits "$ABITS" \
        --nsamples "$CALIB_NSAMPLES" --seqlen "$CALIB_SEQLEN" --dataset "$CALIB_DATASET" \
        --mix-c4-n "$CALIB_MIX_CODE_N" --nuq --fisher "$FISHER" --quantize --include_sparse \
        --sparsity-threshold "$SPARSITY" --quantizer-path "$PICKLE" ) \
      > "$LOG/simquant.log" 2>&1 || die "llama_simquant (see $LOG/simquant.log)"
  fi

  if has tasq && [[ ! -f "$TASQPKL" ]]; then
    say "TaSQ permutation + weights (causal diag(M), grouped on Cov(k^wn))"
    # make_tasq takes no method flags: it computes W from the causal diag(M), applies it,
    # pools the per-token RMS over all KV heads, and groups on the covariance of THAT vector.
    # Costs two forwards at calibration time; serving, storage and bundle format are unchanged.
    CUDA_VISIBLE_DEVICES="$CAL_GPUS" python3 "$CODEBOOKS/make_tasq.py" --model "$MODEL" \
      --nsamples "$CALIB_NSAMPLES" --seqlen "$CALIB_SEQLEN" --seed "$CALIB_SEED" \
      --coupled "$COUPLED" --dataset "$CALIB_DATASET" --mix-code-n "$CALIB_MIX_CODE_N" \
      --out "$TASQPKL" \
      > "$LOG/make_tasq.log" 2>&1 || die "make_tasq (see $LOG/make_tasq.log)"
  fi

  # Fit TaSQ centroids in the weighted, normalized, and permuted space. The Fisher statistic is
  # collected in native coordinates, so --fisher-native-coord applies the corresponding 1/w_i
  # factor before weighted k-means. TaSQ's V side remains plain CQ.
  FISHER_NATIVE_COORD="${FISHER_NATIVE_COORD:-1}"
  fnc=(); [[ "$FISHER_NATIVE_COORD" != "0" ]] && fnc=(--fisher-native-coord)
  if has tasq && [[ ! -f "$TASQPICKLE" ]]; then
    say "TaSQ centroids (simquant ${COUPLED}c${ABITS}b + P/W/N)"
    ( cd "$VENDOR" && CUDA_VISIBLE_DEVICES="$CAL_GPUS" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY_KVQ[@]}" llama_simquant.py "$MODEL" --num_coupled "$COUPLED" --abits "$ABITS" \
        --nsamples "$CALIB_NSAMPLES" --seqlen "$CALIB_SEQLEN" --dataset "$CALIB_DATASET" \
        --mix-c4-n "$CALIB_MIX_CODE_N" --nuq --fisher "$FISHER" --quantize --include_sparse \
        --sparsity-threshold "$SPARSITY" \
        --tasq-file "$TASQPKL" --per-token-norm --post-norm-bits "${POST_NORM_BITS:-16}" --head-group "$HEAD_GROUP" \
        --ptnorm-weight ${FISHER_KMEANS_INIT:+--fisher-kmeans-init} \
        ${KMEANS_ITERS:+--kmeans-iters $KMEANS_ITERS} \
        "${fnc[@]}" \
        ${FISHER_DIAG_METRIC:+--fisher-diag-metric} \
        --quantizer-path "$TASQPICKLE" ) \
      > "$LOG/simquant_tasq.log" 2>&1 || die "llama_simquant --tasq-file (see $LOG/simquant_tasq.log)"
  fi

  conda activate "$ENV_SERVE"
  if has cq && [[ ! -f "$ARTIFACTS/${TAG}_cq_g8/codebook.pt" ]]; then
    say "cq bundle"; mkdir -p "$ARTIFACTS/${TAG}_cq_g8"
    python3 "$CODEBOOKS/build_cq_bundle.py" --quantizer_path "$PICKLE" \
      --out_k "$ARTIFACTS/${TAG}_cq_g8/codebook.pt" \
      --out_v "$ARTIFACTS/${TAG}_cq_g8/vq_v_codebook.pt" \
      --head_dim "$HEAD_DIM" --group_dim "$COUPLED" > "$LOG/build_cq.log" 2>&1 \
      || die "build_cq_bundle (see $LOG/build_cq.log)"
  fi
  if has tasq && [[ ! -f "$TASQDIR/codebook.pt" ]]; then
    say "tasq bundle -> $TASQDIR"; mkdir -p "$TASQDIR"
    python3 "$CODEBOOKS/build_tasq_bundle.py" --quantizer_path "$TASQPICKLE" --tasq_path "$TASQPKL" \
      --out_k "$TASQDIR/codebook.pt" \
      --out_v "$TASQDIR/vq_v_codebook.pt" \
      --head_dim "$HEAD_DIM" --group_dim "$COUPLED" > "$LOG/build_tasq.log" 2>&1 \
      || die "build_tasq_bundle (see $LOG/build_tasq.log)"
  fi

  # CQ and TaSQ serve UNROTATED -- their rotation files are objective="identity" (verified
  # numerically against the shipped llama/qwen3-4b bundles). serve_method.sh still requires the
  # files to exist, so write identity ones. Substituting nova's real rotations would silently
  # turn these arms into a different method.
  for m in cq tasq; do
    has $m || continue
    d="$ARTIFACTS/${TAG}_${m}_g8"
    [[ -f "$d/k_rotation_qqt_r_h_pbr.pt" ]] && continue
    say "$m: identity rotations"
    python3 - "$d" "$HEAD_DIM" <<'PY' || die "identity rotations"
import sys, torch
out_dir, D = sys.argv[1], int(sys.argv[2])
import json, os
# layer count comes from the codebook we just wrote (forward is [L, H, D, D])
L = torch.load(os.path.join(out_dir, "codebook.pt"), map_location="cpu",
               weights_only=False)["forward"].shape[0]
for name in ("k_rotation_qqt_r_h_pbr.pt", "v_rotation_sst_r_h_pbr.pt"):
    # memory_pool.load_oscar_rotations expects the key "rotation".
    torch.save({"format_version": 1, "objective": "identity", "source_grouping": None,
                "layers": {i: {"layer_id": i, "rotation": torch.eye(D)} for i in range(L)}},
               os.path.join(out_dir, name))
print(f"identity rotations for {L} layers")
PY
  done
fi

say "artifacts for $TAG:"
ls -d "$ARTIFACTS/${TAG}"* 2>/dev/null || echo "  (none)"
