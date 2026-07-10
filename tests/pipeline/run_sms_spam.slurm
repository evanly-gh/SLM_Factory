#!/bin/bash
#SBATCH --job-name=slm-fin-sentiment
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-fin-sentiment-%j.out
# TEST CASE: Classification — clean convergence baseline + Exa HW fallback
#
# PURPOSE: Verify the full classification pipeline converges correctly on an
# achievable 3-class task. FinancialPhraseBank sentiment (positive/negative/neutral)
# is learnable for a tier-1/2 model within ~10-15 iterations without needing
# escalation. This is the happy-path test.
#
# The Moto G Stylus 5G 2023 is NOT in devices.csv (only the 2020/2021 models are),
# so hardware_research will fall back to Exa web search for specs — intentionally
# testing the Exa hardware fallback path. Check device_research.json after the run
# to confirm spec_source="exa" and that usable_ram_mb is reasonable (~3500-4500MB).
#
# What to check in logs/runs/<timestamp>/scores.json:
#   - converged: true
#   - iterations: <= 15
#   - no "ESCALATE" lines in run.log
#   - task_type: "classification"
# What to check in logs/runs/<timestamp>/device_research.json:
#   - spec_source: "exa" (confirms the fallback path was exercised)
set -euo pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1   # xet transport stalls on compute nodes (BUGS B113); use plain HTTPS
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache

nvidia-smi -L || true
.venv_gpu/bin/python -c '' >/dev/null 2>&1 || bash scripts/setup_gpu_env.sh
source .venv_gpu/bin/activate

# Cost-saving orchestrator: Haiku drives all planning/decision/generation calls.
# With no DEEPSEEK_API_KEY/OPENAI_API_KEY set, CoT teachers also fall back to this.
export SLM_ORCHESTRATOR_MODEL=claude-haiku-4-5

TASK="Fine tune a small model for financial news sentiment classification (positive, negative, neutral) using the FinancialPhraseBank dataset, targeting deployment on my Moto G Stylus 5G 2023 with 6GB RAM and 256GB storage"
echo "=== Financial Sentiment Classification pipeline test ==="
echo "Task: $TASK"
python tests/pipeline/run.py "$TASK"
echo "=== done ==="
