#!/bin/bash
#SBATCH --job-name=slm-largest-first
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-largest-first-%j.out
#
# TEST — largest_first model-selection strategy (probe phase) at the graph level.
# ============================================================================
# run_qwen_escalation uses smallest_first. The 4 selection strategies are unit-tested
# (tests/cold_start/test_model_selection.py), but only largest_first has extra GRAPH
# behavior: it starts on the LARGEST feasible model as a feasibility probe, sets
# state["_largest_first_phase"]="probe", and iterate_node's probe hooks then either
# (a) switch to the smallest model on probe success, or (b) TERMINATE on probe stall
# (task infeasible). This test exercises that path end-to-end.
#
# Setup: FPB classification, largest_first, threshold 0.72 (reachable), device fits all tiers.
# Expected: probe starts on the largest feasible model; if it clears 0.72, the loop
# switches to the smallest model and escalates from there; if it stalls, it terminates.
# Look for: "[model_selection:largest_first] Probe phase — selected largest", then either
# "Probe succeeded ... Switching to smallest model" or "largest_first probe stagnated".
#
# COVERS (that run_qwen_escalation misses): largest_first strategy + its iterate probe hooks.
# (interpolation + orchestrator_choice are covered by unit tests; run those live only if the
#  extra Exa/LLM cost is warranted — swap SLM_MODEL_SELECTION_STRATEGY below.)
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
export SLM_MODEL_SELECTION_STRATEGY=largest_first
export SLM_STOP_THRESHOLD=0.72

TASK="Fine tune a small model for financial news sentiment classification (positive, negative, neutral) using the FinancialPhraseBank dataset, targeting deployment on my Pixel 8 with 8GB RAM and 128GB storage"
echo "=== largest_first strategy validation test ==="
echo "Task: $TASK"
python tests/pipeline/run.py "$TASK"
echo "=== done ==="
