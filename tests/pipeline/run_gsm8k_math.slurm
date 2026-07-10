#!/bin/bash
#SBATCH --job-name=slm-gsm8k-math
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-gsm8k-math-%j.out
# TEST CASE: Math reasoning (generation) — escalation then convergence
#
# PURPOSE: Verify the generation pipeline (math_reasoning scorer: exact match
# after normalization) on a task that is hard for small models but achievable
# at tier-2/3, so the run exercises escalation AND eventual convergence.
# GSM8K grade-school math requires multi-step arithmetic reasoning; tier-0
# models (~500M params) score ~5-15% even with CoT fine-tuning. A 1.7B model
# (tier 1-2) with CoT distillation from DeepSeek-R1 can reach ~60-70%.
#
# The device (Pixel 6a, 6GB RAM) allows models up to tier-2/3, so the planner
# will start at a mid-tier model. If it stagnates, escalation climbs to tier-3.
# The DeepSeek-R1 teacher path (DEEPSEEK_API_KEY) is exercised if available.
#
# What to check in logs/runs/<timestamp>/:
#   - task_type: "math_reasoning" in scores.json
#   - max_new_tokens=256 used (verify in run.log infer_batch calls)
#   - converged: true in scores.json (target: ~65% exact match)
#   - At least one "ESCALATE" line if starting tier cannot reach threshold
#   - dag.json shows CoT-annotated dataset versions (intervention: data_rebuild
#     early, then hyperparameter/surgical as score climbs)
#
# Scorer: eval/scorers/generation.py — math_reasoning branch (exact match)
# Teacher: DeepSeek-R1 (math CoT annotation) if DEEPSEEK_API_KEY set,
#          else falls back to claude teacher
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

# Pixel 6a: 6GB RAM, 128GB storage — tier-1/2 models are feasible.
# GSM8K is hard enough to require several iterations and likely one escalation,
# but within reach for a tier-2 model with CoT fine-tuning.
TASK="Fine tune a small model for grade-school math word problem solving (GSM8K dataset), producing step-by-step reasoning and a final numeric answer, targeting deployment on my Pixel 6a with 6GB RAM and 128GB storage"
echo "=== GSM8K Math Reasoning (generation, escalation-convergent) pipeline test ==="
echo "Task: $TASK"
python run.py "$TASK"
echo "=== done ==="
