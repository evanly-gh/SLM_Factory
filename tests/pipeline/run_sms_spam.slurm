#!/bin/bash
#SBATCH --job-name=slm-sms-spam
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-sms-spam-%j.out
# TEST CASE: Classification — clean convergence baseline
#
# PURPOSE: Verify the full classification pipeline converges correctly on an
# achievable task. SMS spam is well-separated (high TF-IDF signal, short texts)
# so a tier-1 or tier-2 model should reach F1 >= stop_threshold within ~10
# iterations without needing escalation. This is the happy-path test.
#
# What to check in logs/runs/<timestamp>/scores.json:
#   - converged: true
#   - iterations: <= 12
#   - no "ESCALATE" lines in run.log
#   - task_type: "classification"
#
# Paper baseline: GLiNER2-base F1 0.159 → 0.997 (+83.8pp) in 10 iterations
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
