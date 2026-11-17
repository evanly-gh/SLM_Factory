#!/bin/bash
#SBATCH --job-name=slm-rescore-unseen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-11%4
#SBATCH --output=logs/slurm/rescore-unseen-%A_%a.out
# Rescore each Table-3 model on held-out rows its run never selected on (scripts/rescore_unseen.py).
# Inference only: one L40S per task, no teacher, no API calls. Paper context: §5.1 "selection set".
# Second pass (difficulty control): scripts/submit.sh --array=1-6,8,10 --export=ALL,RESCORE_TAG=_control,RESCORE_FLAGS="--no-replay --control-base" <this file>
# Third pass (noise): RESCORE_TAG=_chunks RESCORE_FLAGS="--no-replay --control-base --chunk-rows 1000 --max-chunks 5"
# First-FT pass (Table 5 "First FT" column): the same chunks/fresh sets, scored with the run's first fine-tune
# of its chosen variant; replay is kept, so the log must reproduce Table 3's First FT. 5 and 7 are skipped:
# their best model IS the first fine-tune. <trash> = the 2026-11-11 cleanup's trash/artifacts (intermediate GGUFs).
#   --array=1-4,6,8,10 RESCORE_TAG=_first_ft_chunks RESCORE_FLAGS="--first-ft --reuse-from <trash> --chunk-rows 1000 --max-chunks 5"
#   --array=0 RESCORE_TAG=_first_ft_fresh_unused RESCORE_FLAGS="--first-ft --reuse-from <trash> --external-rows data/fresh_sets/calendar_unused.jsonl"
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${SLM_ROOT:-$SLURM_SUBMIT_DIR}/scripts/site_env.sh"
PROJ="$SLM_ROOT"
cd "$PROJ"

RUNS=(
  "calendar_json slm-calendar-json-l40s-42138096"
  "ner_bc5cdr slm-ner-bc5cdr-l40s-42138097"
  "topv2 slm-topv2-l40s-39969102"
  "multiconer slm-multiconer-l40s-42138121"
  "xlam_bfcl slm-xlam-bfcl-l40s-41959026"
  "goemotions slm-goemotions-ckpt-39708679"
  "gec_bea19 slm-gec-bea19-l40s-39881529"
  "dialogsum slm-dialogsum-l40s-39969104"
  "clinc150 slm-clinc150-ablation-nostop-l40s-40105479"
  "sms_spam slm-sms-spam-ablation-nostop-l40s-40105478"
  "clinc150 slm-clinc150-calibrated-l40s-42806317"
  "sms_spam slm-sms-spam-calibrated-l40s-42806318"
)
read -r TASK RUN <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_XET=1
export ERRANT_VENV="$PROJ/.venv_errant"
export METRICS_VENV="$PROJ/.venv_metrics"
source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1); CUDA_MOD="${CUDA_MOD:-cuda/12.8.1}"
module load "$CUDA_MOD"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
source .venv_gpu/bin/activate
# config/config.py reads ANTHROPIC_API_KEY and EXA_API_KEY at import; this job makes no API calls.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-unused-by-rescore-no-api-calls}"
export EXA_API_KEY="${EXA_API_KEY:-unused-by-rescore-no-api-calls}"

export SLM_BENCHMARK_TASK="$TASK"
export SLM_CUDA_ISOLATION=1
export SLM_QUANT_EVAL=1
export SLM_MAX_SEQ_LENGTH=4096
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SLM_SEED=3407
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "task=$TASK run=$RUN host=$(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-?}"
EXTRA=""
# multiconer selects on the 871-row dev split; its loader's uncapped eval split is a 20k test slice.
[ "$TASK" = "multiconer" ] && EXTRA="--allow-disjoint-split"
python scripts/rescore_unseen.py --task "$TASK" --run-dir "logs/runs/$RUN" \
    --out "logs/reports/unseen/${TASK}__${RUN}${RESCORE_TAG:-}.json" --max-rows 2000 $EXTRA ${RESCORE_FLAGS:-}
