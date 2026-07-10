#!/bin/bash
#SBATCH --job-name=slm-arc-challenge
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-arc-challenge-%j.out
# TEST CASE: Generation — guaranteed impossible (budget exhaustion without convergence)
#
# PURPOSE: Verify the pipeline handles the case where no model in the feasible
# pool can reach the stop_threshold. The device is deliberately bottlenecked to
# 2GB RAM (Redmi 9A — one of the most RAM-constrained Android devices shipped
# in volume), which restricts the planner to tier-0 models only (~500MB peak).
# Sub-500M parameter models top out at ~35-40% on ARC-Challenge even with CoT
# fine-tuning; the planner will calibrate stop_threshold against published SOTA
# (~72% for 3B models), which is structurally unreachable at this model size.
#
# Expected behavior:
#   - Planner selects a tier-0 model (only option within 2GB)
#   - iterate_node escalates when stagnation fires
#   - escalate_node terminates (no higher tier fits within 2GB)
#   - Run ends with converged: false in scores.json
#   - The iterate LLM may lower stop_threshold, but the floor prevents it from
#     dropping below the initial calibrated value
#
# What to check in logs/runs/<timestamp>/scores.json:
#   - converged: false
#   - "ESCALATE" followed by "terminate" (top tier already) in run.log
#   - task_type: "generation" or "math_reasoning"
#
# Paper baseline: Llama 3.2-3B Acc 5.3% → 72.6% (+67.3pp) in 11 iterations
# Uses A100 for faster generation-task training (longer sequences, CoT annotation)
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

# Cost-saving orchestrator: Haiku drives all planning/decision/generation calls.
# With no DEEPSEEK_API_KEY/OPENAI_API_KEY set, CoT teachers also fall back to this.
export SLM_ORCHESTRATOR_MODEL=claude-haiku-4-5

# Redmi 9A: 2GB RAM, 32GB storage (confirmed in devices.csv: 2048MB RAM).
# After OS overhead, usable RAM ~500MB — only the smallest tier-0 variants fit.
# Framed as open-ended reasoning (not multiple-choice) so the planner assigns
# task_type="generation" or "math_reasoning" rather than "classification",
# exercising the 256-token / LLM-as-judge / exact-match scorer paths.
# NOTE: the planner sets stop_threshold dynamically against SOTA. If it anchors
# too low (knowing sub-500M models score ~30%), the task may not be impossible.
# Monitor task_plan.json after the run — if stop_threshold < 0.50, the test
# is not a clean "impossible" case. The Redmi 9A chipset (Helio G25) is not in
# CHIP_SCALE_FACTORS and will fall back to snapdragon_778g for throughput estimation.
TASK="Fine tune a small model to solve ARC-Challenge science questions by producing step-by-step reasoning and a final answer in natural language (not multiple choice), targeting deployment on my Redmi 9A with 2GB RAM and 32GB storage"
echo "=== ARC-Challenge (impossible — guaranteed no convergence) pipeline test ==="
echo "Task: $TASK"
python tests/pipeline/run.py "$TASK"
echo "=== done ==="
