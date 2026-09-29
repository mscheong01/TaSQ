#!/usr/bin/env bash
# Create the two conda environments this repo needs, and install into them.
#
#   ./setup_envs.sh            # both
#   ./setup_envs.sh serve      # just the serving/calibration env
#   ./setup_envs.sh eval       # just the lm_eval client
#
# Names default to config.sh's ENV_SERVE / ENV_EVAL and can be overridden:
#   ENV_SERVE=my-serve ./setup_envs.sh serve
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVE="${ENV_SERVE:-tasq-serve}"
ENV_EVAL="${ENV_EVAL:-tasq-eval}"
WHICH=("$@"); [[ $# -eq 0 ]] && WHICH=(serve eval)
command -v conda >/dev/null || { echo "conda not on PATH"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

has() { printf '%s\n' "${WHICH[@]}" | grep -qx "$1"; }

if has serve; then
  echo "== $ENV_SERVE : serving, calibration, codebook fitting"
  conda env list | grep -q "^$ENV_SERVE " || conda create -n "$ENV_SERVE" python=3.10 -y
  conda activate "$ENV_SERVE"
  # The fork brings its own torch (2.9.1) and the rest of its dependencies. It installs as
  # tasq-sglang and imports as sglang, so it cannot be confused with an upstream SGLang.
  pip install -e "$REPO/python"
  # --no-build-isolation is required: without it the build picks up a different torch, and the
  # resulting kmeans_tools is importable only from the interpreter that built it.
  pip install -e "$REPO/third_party/kvquant" --no-build-isolation
  python -c "import torch, sglang; import kmeans_tools" \
    && echo "   ok: sglang and kmeans_tools import"
  conda deactivate
fi

if has eval; then
  echo "== $ENV_EVAL : the lm_eval client (no GPU)"
  # Separate env because lm_eval 0.4.9 does not import under transformers 5.x, which the fork
  # requires. 4.53.2 is also the floor: under 4.48.x mbpp_instruct silently scores 0.
  conda env list | grep -q "^$ENV_EVAL " || conda create -n "$ENV_EVAL" python=3.10 -y
  conda activate "$ENV_EVAL"
  pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu \
                               --extra-index-url https://pypi.org/simple
  pip install lm_eval==0.4.9 transformers==4.53.2 tenacity "numpy<2"
  python -c "import lm_eval" && echo "   ok: lm_eval imports"
  conda deactivate
fi
