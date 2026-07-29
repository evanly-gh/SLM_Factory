#!/bin/bash
#SBATCH --job-name=slm-synth-server
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/gpu_setup/slm-synth-server-%j.out
#
# Local synthesis server (B161): serve Qwen3.6-35B-A3B via vLLM's OpenAI-compatible API for
# hard-negative / rare-class synthesis, so the pipeline never calls Claude for synthetic data
# (no cost, contamination-safe, reproducible). One server serves ALL concurrent training runs.
#
# GPU fit: Qwen3.6-35B-A3B is 36B params, BF16 ≈ 72GB. Fits comfortably on ONE H200 (141GB)
# with room for KV cache (tp=1). On an A100-80GB use FP8 instead (add --quantization fp8 and
# swap --gres=gpu:a100:1 / --partition=ckpt). We serve TEXT-ONLY (--language-model-only) since
# synthesis is text, which frees the vision-encoder memory for more KV cache.
#
# On startup it writes its reachable endpoint to logs/synth_endpoint.txt; training runs read
# that into SLM_SYNTH_ENDPOINT (the strategy scripts do this automatically).
# no `-u`: lmod init references unbound LD_LIBRARY_PATH; pre-init it and use -eo.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1

# vLLM/flashinfer JIT-compile CUDA sampling kernels at RUNTIME → need nvcc on PATH +
# CUDA_HOME, else "nvcc: No such file or directory" crashes the engine (B161). Load the
# CUDA toolkit matching torch's cu128. Do NOT pipe `module load` (subshell loses env).
source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1)
CUDA_MOD="${CUDA_MOD:-cuda/12.8.1}"
module load "$CUDA_MOD"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
echo "[serve] CUDA_HOME=$CUDA_HOME  nvcc=$(command -v nvcc)"

nvidia-smi -L || true
.venv_vllm/bin/python -c 'import vllm' >/dev/null 2>&1 || bash scripts/setup_vllm_env.sh
source .venv_vllm/bin/activate

MODEL="${SLM_SYNTH_MODEL:-Qwen/Qwen3.6-35B-A3B}"
PORT="${SLM_SYNTH_PORT:-8000}"
HOSTIP=$(hostname -s)
ENDPOINT="http://${HOSTIP}:${PORT}/v1"

echo "$ENDPOINT" > "$PROJ/logs/synth_endpoint.txt"
echo "=== synthesis server: model=$MODEL endpoint=$ENDPOINT (written to logs/synth_endpoint.txt) ==="

# Text-only, modest context (synthesis prompts are short); tp=1 fits BF16 on one H200.
vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --served-model-name "$MODEL"
