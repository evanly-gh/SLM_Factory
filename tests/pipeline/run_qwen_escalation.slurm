#!/bin/bash
#SBATCH --job-name=slm-qwen-escalation
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-qwen-escalation-%j.out
#
# VALIDATION TEST — "does the whole thing run well?" (qwen-only pool)
# ============================================================================
# GOAL (per request): a cheap, doable task with a good growth trajectory that
# (a) does NOT end too early, (b) escalates through EVERY model tier, and
# (c) exercises as many interventions as one autonomous run can.
#
# WHY THIS SETUP
# --------------
# Task  : GSM8K grade-school math (real benchmark data via the B119 loader:
#         openai/gsm8k, capped at 300 train / 80 test — cheap). Math is genuinely
#         hard for sub-3B models, so small models cannot shortcut to convergence —
#         this is what forces escalation instead of an early stop.
# Device: 16 GB RAM phone → ~14 GB usable, so ALL 18 pool variants (tiers 0-3)
#         are hardware-feasible and escalation has somewhere to climb to.
# Strategy: smallest_first — start at the smallest model (tier 0, Qwen3-0.6B) and let
#         the stagnation/stall logic escalate upward tier by tier.
# Threshold: pinned to 0.90 via SLM_STOP_THRESHOLD. The pool's BEST GSM8K score is
#         Qwen3-4B-Instruct-2507 at ~0.88, so NO tier can clear 0.90 — every tier
#         stalls and escalates, walking tier 0 → 1 → 2 → 3 before terminating at
#         the top. (Any threshold ≤ ~0.88 would converge at the tier where the 4B
#         model first appears and never climb to the top.)
#
# EXPECTED TRAJECTORY (verify in logs/runs/<ts>/ and the .out file)
# -----------------------------------------------------------------
#   model selection strategy: smallest_first   (run.log header)
#   Selected unsloth/Qwen3-0.6B (tier=0 ...)    (model_selection node)
#   Growth curve climbs roughly 0.30 → 0.50 → 0.65 → ~0.87 across tiers
#   [iterate] → CURATE (data_rebuild)           at low scores (<0.80 band)
#   [iterate] → TRAIN  (hyperparameter ...)     once a tier reaches the 0.80-0.95 band
#   [rollback] REGRESSION detected ...          on any noisy regression (then → iterate)
#   [iterate] → ESCALATE (stagnation|N stalled evals ...)   ~3 times (tier 0→1→2→3)
#   [escalate] PROMOTING: tier k → tier k+1 | <model> → <model>
#   [escalate] No feasible models above tier 3 — TERMINATING   (final)
#   scores.json: converged=false (intended — this run measures the LOOP, not accuracy)
#
# INTERVENTION COVERAGE (honest)
# ------------------------------
#   data_rebuild   ✓  (early low-score iterations)
#   hyperparameter ✓  (mid-range 0.80-0.95 band on the stronger tiers)
#   rollback       ✓  (training variance → at least one regression)
#   escalate       ✓  (×3, tier 0→1→2→3 — the headline of this test)
#   terminate      ✓  (no tier above 3)
#   downward_probe ✗  (only fires on convergence, i.e. score ≥ threshold; mutually
#                      exclusive with "escalate through all tiers" — see COMPLEMENTARY)
#   surgical       ✗  (fallback only picks it at score ≥ 0.95, which no pool model
#                      reaches on GSM8K; needs an easy task — see COMPLEMENTARY)
#
# COMPLEMENTARY RUN (to cover downward_probe; run separately if desired)
# ---------------------------------------------------------------------
#   Re-run with SLM_STOP_THRESHOLD=0.60 (below DeepSeek's 0.87). The run then
#   escalates tier 0 → tier 1, converges on DeepSeek-R1-Distill (tier 1), and fires
#   a DOWNWARD_PROBE (tier>0) before TERMINATE. surgical realistically needs an easy
#   classification task where a fine-tuned model exceeds 0.95 F1 (GSM8K cannot).
# ============================================================================
set -euo pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1   # xet transport stalls on compute nodes (BUGS B113)
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache

nvidia-smi -L || true
.venv_gpu/bin/python -c '' >/dev/null 2>&1 || bash scripts/setup_gpu_env.sh
source .venv_gpu/bin/activate

# --- Validation knobs -------------------------------------------------------
export SLM_ORCHESTRATOR_MODEL=claude-haiku-4-5      # cheap orchestrator
export SLM_MODEL_SELECTION_STRATEGY=smallest_first  # start tier 0, escalate upward
export SLM_STOP_THRESHOLD=0.90                       # > pool-best (0.87) → forces full escalation
# ---------------------------------------------------------------------------

# 16 GB device → all four tiers feasible, so escalation can climb tier 0 → 3.
TASK="Fine tune a small model for grade-school math word problem solving (GSM8K dataset), producing step-by-step reasoning and a final numeric answer, targeting deployment on my OnePlus 12 with 16GB RAM and 256GB storage"
echo "=== Qwen-only ESCALATION validation test (smallest_first, threshold 0.90) ==="
echo "Task: $TASK"
python tests/pipeline/run.py "$TASK"
echo "=== done ==="
