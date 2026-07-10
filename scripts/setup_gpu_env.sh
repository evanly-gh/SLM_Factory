#!/bin/bash
# Build the GPU virtualenv for real (Unsloth) training. Run on a GPU node (it pulls a
# CUDA torch). Caches go to gscratch (home is only 10GB).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache}"

cd /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory

# Rebuild if the venv is missing OR broken. A venv can become a zombie when its
# uv-managed interpreter under ~/.local/share/uv/python is garbage-collected/deleted:
# the directory still exists (so a `[ -d .venv_gpu ]` guard passes) but `python` is a
# dangling symlink and every command fails with exit 127. Validate the interpreter, not
# just the directory, and recreate from scratch when it can't run.
if ! .venv_gpu/bin/python -c '' >/dev/null 2>&1; then
  echo "[setup] .venv_gpu missing or broken interpreter — (re)creating"
  rm -rf .venv_gpu
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
