#!/bin/bash
# Build the GPU virtualenv for real (Unsloth) training. Run on a GPU node (it pulls a
# CUDA torch). Caches go to gscratch (home is only 10GB).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache}"

cd /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory

# logs/ is gitignored, so these cannot be committed into existence — and Slurm does not
# create an --output directory, it fails the job. Create both here so a fresh checkout
# is submit-ready after the standard setup step.
mkdir -p logs/slurm logs/gpu_setup

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
# timm: required to load the gemma-3n-e2b-it multimodal wrapper (TimmWrapperModel),
# otherwise its scaling-curve probe fails (see BUGS B114).
uv pip install "timm"
# Orchestration deps (same as the dev venv). langchain-anthropic is required by
# iterate_node's LLM per-iteration decision; without it every iteration silently
# falls back to score-band rules (see BUGS B112).
uv pip install "langgraph>=0.2.0" "anthropic>=0.40.0" "exa-py" "python-dotenv" "langchain-anthropic"
# matplotlib: run.py renders the end-of-run trajectory graphics. It is declared in
# requirements.txt but was never installed here, so every cluster run ended with
# "graphics: skipped (ModuleNotFoundError: No module named 'matplotlib')" (see BUGS B215).
uv pip install "matplotlib>=3.7.0"

# --- Quantized-accuracy eval deps (config.QUANT_ACCURACY_EVAL / SLM_QUANT_EVAL=1) ---
# gguf: needed by llama.cpp's convert_hf_to_gguf.py. llama-cpp-python: CPU-side GGUF
# inference used to score Q4_K_M/Q8_0 honestly. CPU-only build (no CUDA in the wheel).
uv pip install "gguf"
CMAKE_ARGS="-DGGML_CUDA=OFF" FORCE_CMAKE=1 uv pip install "llama-cpp-python"

# llama.cpp CLI tools (llama-quantize + convert_hf_to_gguf.py) are located via PATH by
# training/quantize.py (shutil.which). Point at the prebuilt checkout on gscratch and
# make the tools available to anything that sources this venv's activate script.
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/llama.cpp}"
if [ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
  LLAMA_PATH_LINE="export PATH=\"$LLAMA_CPP_DIR/build/bin:$LLAMA_CPP_DIR:\$PATH\"  # SLM_Factory llama.cpp tools"
  if ! grep -qF "SLM_Factory llama.cpp tools" .venv_gpu/bin/activate; then
    echo "$LLAMA_PATH_LINE" >> .venv_gpu/bin/activate
  fi
  # llama-quantize and the llama-cpp-python native lib are built with gcc/12.3.0 and need
  # its newer libstdc++ (GLIBCXX_3.4.29/30) at RUNTIME; the RHEL8 system libstdc++ is too
  # old, so quantization fails with "GLIBCXX_... not found" (B160). Put the gcc-12 runtime
  # lib on LD_LIBRARY_PATH for anything that sources this venv.
  GCC12_LIB="/sw/gcc/12.3.0/lib64"
  LLAMA_LDLIB_LINE="export LD_LIBRARY_PATH=\"$GCC12_LIB:\${LD_LIBRARY_PATH:-}\"  # SLM_Factory gcc12 libstdc++ (GLIBCXX for llama.cpp/llama-cpp-python)"
  if [ -d "$GCC12_LIB" ] && ! grep -qF "SLM_Factory gcc12 libstdc++" .venv_gpu/bin/activate; then
    echo "$LLAMA_LDLIB_LINE" >> .venv_gpu/bin/activate
  fi
else
  echo "[setup] NOTE: llama-quantize not found under $LLAMA_CPP_DIR/build/bin — build it with:"
  echo "        module load cmake/3.25.1 gcc/12.3.0"
  echo "        cmake -S \"$LLAMA_CPP_DIR\" -B \"$LLAMA_CPP_DIR/build\" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF"
  echo "        cmake --build \"$LLAMA_CPP_DIR/build\" -j --target llama-quantize"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "GPU env ready: .venv_gpu"
