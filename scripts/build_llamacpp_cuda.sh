#!/bin/bash
# Rebuild llama-cpp-python with CUDA so GGUF accuracy eval runs on the GPU (B161). The
# quantized GGUF's ACCURACY is identical CPU vs GPU (same weights); GPU is just ~10-50x
# faster, which makes the 800-example eval tractable inside the loop. Without this, GGUF eval
# still works but on CPU (slow) — training/slm_helpers.py falls back automatically.
#
# RUN ON A GPU NODE (needs nvcc matching the torch CUDA build). e.g.:
#   srun -A intelligentsystems -p ckpt-g2 --gres=gpu:l40:1 --time=1:00:00 --pty \
#        bash scripts/build_llamacpp_cuda.sh
# When submitted with sbatch, send the log to the GPU-infrastructure directory rather
# than logs/slurm/ (which is for pipeline run logs):
#   sbatch --output=logs/gpu_setup/llamacpp-cuda-build-%j.out scripts/build_llamacpp_cuda.sh
# NOTE: no `-u` (nounset) — the Lmod bash init references $LD_LIBRARY_PATH, which is unbound
# in a fresh SLURM shell and would abort the script before any build. Pre-init it and use -eo.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
mkdir -p logs/gpu_setup
export PATH="$HOME/.local/bin:$PATH"
source /etc/profile.d/modules.sh 2>/dev/null || true
# IMPORTANT: do NOT pipe `module load` (e.g. `module load X | tail`) — `module` is a shell
# function, and piping runs it in a SUBSHELL so its PATH/LD_LIBRARY_PATH changes are lost
# (that made nvcc "command not found"). Run module loads unpiped so the env persists.
module load cmake/3.25.1 gcc/12.3.0
# The venv's torch is cu128 (CUDA 12.8). Prefer the EXACT 12.8 toolkit module (safest — we
# know the node driver supports 12.8 since cu128 torch runs here); else newest 12.x; else
# 12.2.2. Avoid a toolkit NEWER than the driver supports (e.g. 12.9) which can fail at runtime.
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1)
[ -z "$CUDA_MOD" ] && CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.[0-9.]+" | sort -V | tail -1)
CUDA_MOD="${CUDA_MOD:-cuda/12.2.2}"
echo "[build] loading $CUDA_MOD (torch is cu128 / CUDA 12.8)"
module load "$CUDA_MOD"
source .venv_gpu/bin/activate

echo "=== nvcc ==="; nvcc --version | tail -2 || { echo "nvcc missing — load a cuda module"; exit 1; }
# Point CMake explicitly at this nvcc / CUDA toolkit (module doesn't set CUDA_HOME/CUDACXX).
export CUDACXX="$(command -v nvcc)"
export CUDA_HOME="$(dirname "$(dirname "$CUDACXX")")"
echo "[build] CUDACXX=$CUDACXX  CUDA_HOME=$CUDA_HOME"
# GGML_NATIVE=OFF: build a PORTABLE CPU backend instead of `-march=native`. The GGUF eval
# offloads all layers to CUDA (SLM_GGUF_GPU_LAYERS=-1), so CPU SIMD is irrelevant here, and
# `-march=native` makes gcc-12 emit AVX512-BF16 (`vdpbf16ps`) that the node's old system
# assembler rejects with "no such instruction", failing the whole build (B199).
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$CUDACXX -DGGML_NATIVE=OFF"
export FORCE_CMAKE=1
export TMPDIR=/mmfs1/gscratch/intelligentsystems/evanly/tmp
mkdir -p "$TMPDIR"

# Reinstall ONLY llama-cpp-python from source (CUDA compile). --no-deps keeps the venv's
# existing numpy/jinja2/etc. untouched (--no-binary :all: would rebuild+reinstall those too,
# risking a numpy change that torch/unsloth depend on); its deps are already installed.
uv pip install --python .venv_gpu/bin/python --force-reinstall --no-deps --no-binary llama-cpp-python llama-cpp-python

echo "=== verify CUDA offload ==="
.venv_gpu/bin/python - <<'PY'
import llama_cpp, inspect
print("llama_cpp", llama_cpp.__version__)
# supports_gpu_offload is exposed when built with a GPU backend.
try:
    print("gpu offload supported:", llama_cpp.llama_supports_gpu_offload())
except Exception as e:
    print("could not query gpu offload:", e)
PY
echo "Done. GGUF eval will now use the GPU (SLM_GGUF_GPU_LAYERS=-1 default)."
