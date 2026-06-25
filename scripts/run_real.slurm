#!/bin/bash
#SBATCH --job-name=slm-real
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-real-%j.out
# Real autonomous run on a GPU: builds the Unsloth env if needed, then runs the loop
# with REAL train/inference (no stubs), bounded for a first smoke test.
set -euo pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache

nvidia-smi -L || true
[ -d .venv_gpu ] || bash scripts/setup_gpu_env.sh
source .venv_gpu/bin/activate

# Bounded smoke: a known Unsloth-supported tiny base model, few steps.
export SLM_FORCE_MODEL="Qwen/Qwen3-0.6B"
export SLM_MAX_STEPS="${SLM_MAX_STEPS:-6}"
export SLM_TARGET_INCREASES="${SLM_TARGET_INCREASES:-2}"

TASK="${SLM_TASK:-Fine tune a small model to classify a research paper by general area of study, that should run efficiently on my moto g stylus 5g - 2023 with 6 Gb of ram and 256 gb of ROM}"
echo "=== REAL run: $TASK ==="
python run_autonomous.py --real "$TASK"
echo "=== done ==="
