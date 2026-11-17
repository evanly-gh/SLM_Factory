#!/bin/bash
#SBATCH --job-name=slm-policy-ablation-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=logs/slurm/slm-policy-ablation-l40s-%j.out
# ============================================================================================
# ABLATION — WHO CHOOSES THE INTERVENTION: the orchestrator LLM, or a seeded random policy?
# Added 2026-11-16 for the reviewer concern that the search gains mix the orchestrator's choices
# with deterministic rollback/escalation rules (and most interventions are rolled back).
#
# One template for both arms and every task. Parameters (environment; sbatch forwards them):
#   TASK       registry task name, e.g. calendar_json (NOT a description: the description, the
#              teacher/verifier settings and SLM_BENCHMARK_TASK are taken from the task's own
#              main-table launcher, see BASE_LAUNCHER)
#   POLICY     random | llm                                         (default random)
#   SEED       random-policy seed; also names the run dir for llm   (default 1)
#   VARIANT    exact pool selector, base model + quant, e.g. Qwen/Qwen3.5-0.8B@Q4_K_M  (required)
#   MAX_ITERS  search turns after the first fine-tune               (default 20)
#   GOAL_NO_STOP  1 = meeting the goal does not end the run, so both arms spend exactly MAX_ITERS
#              (default 1; set 0 to let a met goal terminate as in the main runs)
#   BASE_LAUNCHER  launcher whose task block is reused (default run_<TASK>_l40s.slurm; for
#              clinc150 / sms_spam pass run_<TASK>_calibrated_l40s.slurm to match the default-goal
#              Table 3 runs 42806317 / 42806318)
#
# Usage (from the repo root; submit.sh supplies account/partition/QOS from site.env):
#   TASK=calendar_json POLICY=random SEED=1 VARIANT='Qwen/Qwen3.5-0.8B@Q4_K_M' MAX_ITERS=20 \
#     scripts/submit.sh -J slm-policy-calendar-json-random-s1 \
#       -o 'logs/slurm/slm-policy-calendar-json-random-s1-%j.out' \
#       tests/pipeline/run_policy_ablation_l40s.slurm
#
# WHAT IS HELD IDENTICAL BETWEEN THE TWO ARMS (everything except SLM_ABLATION_POLICY):
#   * the variant: `single_model` + SLM_FORCE_MODEL pins VARIANT for the whole run, with no LLM
#     model-choice call; the ladder (escalation and the downward probe) is off;
#   * SLM_ABLATION_NO_ESCALATION=1: stagnation and the 30-eval cap neither escalate nor terminate;
#   * SLM_ABLATION_MAX_ITERS: the same explicit search budget;
#   * SLM_THRESHOLD_RAISE=0: no stretch-goal raise (and no stretch-goal LLM call);
#   * starting data, teacher, verifier settings and task spec (the main launcher's task block),
#     training seed SLM_SEED, rollback, run-health watchdog, and curate's synthetic-row allocation
#     (25-600 rows per failure category in proportion to failure count) — none of these read the
#     policy switch.
# The random policy's semantics (option set, hyperparameter neighbours, data-rebuild defaults) are
# documented in agent/random_policy.py.
#
# NOTE: these runs start VARIANT from the seed curriculum. In the original smallest_first runs the
# winning variant inherited a curriculum grown on the tiers below it, so the first-FT score here can
# differ from that run's first FT on the same variant; compare the two arms with each other.
#
# WALL CLOCK: 24h covers 21 train+eval cycles at up to ~60 min each plus ~45 min setup; small
# variants need far less (see the task/budget notes in the PR). The USR1 checkpoint-and-requeue
# contract fires two hours before the wall, so an overrun resumes rather than dying.
# ============================================================================================
set -eo pipefail
source "${SLM_ROOT:-$SLURM_SUBMIT_DIR}/scripts/site_env.sh"

ABL_TASK="${TASK:-}"
ABL_POLICY="${POLICY:-random}"
ABL_SEED="${SEED:-1}"
ABL_VARIANT="${VARIANT:-}"
ABL_MAX_ITERS="${MAX_ITERS:-20}"
ABL_GOAL_NO_STOP="${GOAL_NO_STOP:-1}"
BASE_LAUNCHER="${BASE_LAUNCHER:-run_${ABL_TASK}_l40s.slurm}"

if ! [[ "$ABL_TASK" =~ ^[a-z0-9_]+$ ]]; then
    echo "ERROR: TASK must be a registry task name such as calendar_json, got '${ABL_TASK}'" >&2
    exit 2
fi
case "$ABL_POLICY" in
    llm|random) ;;
    *) echo "ERROR: POLICY must be llm or random, got '${ABL_POLICY}'" >&2; exit 2 ;;
esac
if ! [[ "$ABL_SEED" =~ ^[0-9]+$ ]] || ! [[ "$ABL_MAX_ITERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SEED must be a non-negative integer and MAX_ITERS a positive integer" >&2
    exit 2
fi
if [ -z "$ABL_VARIANT" ] || [[ "$ABL_VARIANT" != *@* ]]; then
    echo "ERROR: VARIANT must be an exact selector <model_id>@<quant>, e.g. Qwen/Qwen3.5-0.8B@Q4_K_M" >&2
    exit 2
fi

# The task block of the main-table launcher: every line above its body-source line (exports of
# SLM_BENCHMARK_TASK, the task description, teacher/verifier settings). Reused rather than copied so
# this ablation cannot drift from the runs it is compared against. SLM_SYNTH_API_MODE is unset first,
# because sbatch forwards the submitting shell's environment and some task blocks branch on it.
unset SLM_SYNTH_API_MODE
BASE_PATH="$SLM_ROOT/tests/pipeline/$BASE_LAUNCHER"
BODY_LINE='source "${SLM_ROOT:-$SLURM_SUBMIT_DIR}/tests/pipeline/_l40s_task_body.sh"'
if [ ! -f "$BASE_PATH" ] || ! grep -qxF "$BODY_LINE" "$BASE_PATH"; then
    echo "ERROR: $BASE_LAUNCHER is missing or does not source the shared L40S body" >&2
    exit 2
fi
eval "$(awk -v stop="$BODY_LINE" '$0 == stop {exit} {print}' "$BASE_PATH")"
if [ "${SLM_BENCHMARK_TASK:-}" != "$ABL_TASK" ]; then
    echo "ERROR: $BASE_LAUNCHER sets SLM_BENCHMARK_TASK='${SLM_BENCHMARK_TASK:-}', expected '$ABL_TASK'" >&2
    exit 2
fi

# --- The ablation switches (agent/ablations.py) ---
export SLM_MODEL_SELECTION_STRATEGY=single_model
export SLM_FORCE_MODEL="$ABL_VARIANT"
export SLM_ABLATION_POLICY="$ABL_POLICY"
export SLM_ABLATION_POLICY_SEED="$ABL_SEED"
export SLM_ABLATION_NO_ESCALATION=1
export SLM_ABLATION_MAX_ITERS="$ABL_MAX_ITERS"
export SLM_THRESHOLD_RAISE=0
if [ "$ABL_GOAL_NO_STOP" = "1" ]; then
    export SLM_GOAL_NO_STOP=1
else
    unset SLM_GOAL_NO_STOP
fi

# --- Run directory: the policy_ablation purpose folder, plus the flat-name symlink logs/README.md
# describes (scripts look runs up by name in logs/runs/). Keyed by job id, so stable across requeue.
_ABL_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual-$$}}"
_ABL_NAME="slm-policy-${ABL_TASK//_/-}-${ABL_POLICY}-s${ABL_SEED}-${_ABL_KEY}"
_ABL_DEFAULT_DIR="$SLM_ROOT/logs/runs/policy_ablation/$_ABL_NAME"
export SLM_RUN_DIR="${SLM_RUN_DIR:-$_ABL_DEFAULT_DIR}"
# Only the purpose folder is created here; run.py creates (or, on requeue, resumes) the run dir.
mkdir -p "$(dirname "$SLM_RUN_DIR")"
_ABL_LINK="$SLM_ROOT/logs/runs/$_ABL_NAME"
if [ "$SLM_RUN_DIR" = "$_ABL_DEFAULT_DIR" ] && [ ! -e "$_ABL_LINK" ] && [ ! -L "$_ABL_LINK" ]; then
    ln -s "policy_ablation/$_ABL_NAME" "$_ABL_LINK"
fi
echo "=== policy ablation: task=$ABL_TASK policy=$ABL_POLICY seed=$ABL_SEED variant=$ABL_VARIANT max_iters=$ABL_MAX_ITERS goal_no_stop=$ABL_GOAL_NO_STOP base=$BASE_LAUNCHER ==="

source "${SLM_ROOT:-$SLURM_SUBMIT_DIR}/tests/pipeline/_l40s_task_body.sh"
