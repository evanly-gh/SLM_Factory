#!/bin/bash
# Build a SEPARATE venv for the vLLM synthesis server (B161). Kept apart from .venv_gpu
# because vLLM pins specific torch/transformers versions that can conflict with the Unsloth
# training stack. The task scripts launch the server from this venv on its own GPU and talk to
# it over HTTP, so the two environments never need to coexist in one interpreter.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache}"

cd /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory

if ! .venv_vllm/bin/python -c '' >/dev/null 2>&1; then
  echo "[setup] creating .venv_vllm"
  rm -rf .venv_vllm
  uv venv --python 3.11 .venv_vllm
fi
source .venv_vllm/bin/activate

# vLLM >= 0.19 is recommended for Qwen3.6 (per the model card). --torch-backend=auto lets uv
# resolve the CUDA torch that matches the serving GPU.
uv pip install --upgrade pip
uv pip install "vllm>=0.19.0" --torch-backend=auto
# ninja: flashinfer JIT-compiles CUDA sampler/attention kernels at runtime and shells out to
# `ninja` (+ nvcc from a CUDA module). Without it: "FileNotFoundError: ... 'ninja'".
uv pip install ninja

python -c "import vllm; print('vllm', vllm.__version__)"
echo "vLLM env ready: .venv_vllm"
