#!/bin/bash
#SBATCH --job-name=slm-sms-spam
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-sms-spam-%j.out
# Classification pipeline test: SMS Spam binary classification
# Targeting Moto G Stylus 5G 2023 (6GB RAM, 256GB storage)
# Paper baseline: GLiNER2-base F1 0.159 → 0.997 (+83.8pp) in 10 iterations
# Our target: decoder model from Android pool, similar improvement trajectory
#
# Termination: reaches F1 >= stop_threshold (default 0.96, calibrated by planner)
# or exhausts the 1500-turn cold-start budget. No artificial step caps.
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

TASK="Fine tune a small model for SMS spam classification (binary: spam vs ham) using the UCI SMS Spam Collection dataset, targeting deployment on my Moto G Stylus 5G 2023 with 6GB RAM and 256GB storage"
echo "=== SMS Spam Classification pipeline test ==="
echo "Task: $TASK"
python run.py "$TASK"
echo "=== done ==="
