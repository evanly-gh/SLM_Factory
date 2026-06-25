#!/bin/bash
# Build the GPU virtualenv for real (Unsloth) training. Run on a GPU node (it pulls a
# CUDA torch). Caches go to gscratch (home is only 10GB).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache}"

cd /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory

if [ ! -d .venv_gpu ]; then
  uv venv --python 3.11 .venv_gpu
fi
source .venv_gpu/bin/activate

# Unsloth pulls compatible torch/transformers/trl/peft/accelerate/bitsandbytes/datasets.
uv pip install --upgrade pip
uv pip install "unsloth"
uv pip install "trl" "transformers" "peft" "accelerate" "datasets" "bitsandbytes"
# Orchestration deps (same as the dev venv).
uv pip install "langgraph>=0.2.0" "anthropic>=0.40.0" "exa-py" "python-dotenv"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "GPU env ready: .venv_gpu"
