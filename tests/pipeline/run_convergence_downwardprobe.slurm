#!/bin/bash
#SBATCH --job-name=slm-convergence
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-convergence-%j.out
#
# TEST — convergence path + DOWNWARD_PROBE + classification (complements run_qwen_escalation).
# ============================================================================
# The escalation test pins the threshold ABOVE the pool's best score so nothing
# converges (to force full-tier escalation). This test does the OPPOSITE: a threshold
# a mid-tier model CAN clear, on a genuinely learnable CLASSIFICATION task, so the run
# CONVERGES — which is the only way to exercise the interventions the escalation run
# structurally cannot reach.
#
# Setup: FinancialPhraseBank (real 3-class sentiment via the datasets loader), device
# large enough that tiers 0 AND 1 are feasible, smallest_first, threshold 0.72.
# Expected: Qwen3-0.6B (tier 0) can't clear 0.72 → escalate to tier 1 (Qwen3-1.7B) →
# converges (≥0.72) → because the converged model is tier>0, DOWNWARD_PROBE fires
# (trains+evals a tier-0 model to see if a smaller one also clears) → TERMINATE.
#
# COVERS (that run_qwen_escalation misses):
#   - classification task_type + classification scorer (accuracy/F1, argmax labels)
#   - convergence (score >= threshold) and terminate-on-convergence
#   - downward_probe node (real train+eval of a smaller tier, then adopt-or-keep)
#   - surgical intervention IF a tier reaches >= 0.95 (possible on easy FPB, not guaranteed)
# STILL NOT covered here: NER/generation/code task types, non-smallest_first strategies,
#   multimodal models (see the other new tests).
# ============================================================================
set -euo pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache

nvidia-smi -L || true
.venv_gpu/bin/python -c '' >/dev/null 2>&1 || bash scripts/setup_gpu_env.sh
source .venv_gpu/bin/activate

export SLM_ORCHESTRATOR_MODEL=claude-haiku-4-5
export SLM_MODEL_SELECTION_STRATEGY=smallest_first
export SLM_STOP_THRESHOLD=0.72   # tier-0 can't reach; tier-1 can → converge → downward_probe

TASK="Fine tune a small model for financial news sentiment classification (positive, negative, neutral) using the FinancialPhraseBank dataset, targeting deployment on my Pixel 8 with 8GB RAM and 128GB storage"
echo "=== Convergence + downward_probe validation test (classification, threshold 0.72) ==="
echo "Task: $TASK"
python tests/pipeline/run.py "$TASK"
echo "=== done ==="
