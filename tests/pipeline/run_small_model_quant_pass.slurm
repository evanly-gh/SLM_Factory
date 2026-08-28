#!/bin/bash
#SBATCH --job-name=slm-small-quant-cse
#SBATCH --account=gpu-l40s-cse
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-small-quant-cse-%j.out
#
# QUANTIZATION PASS over the checkpoints an earlier probe already trained.
#
# WHY THIS EXISTS. Probe 38765131 lost its `quantized` column to an import bug of mine —
# `merge_for_quantization` lives in `training.lora_trainer`, not `training.quantize`. The bug is
# fixed, but the running job held the old bytecode, so it skipped quantization on every cell while
# still writing good zero-shot and fine-tuned numbers. Retraining six models to recover one column
# would be absurd when the checkpoints are sitting on disk, so the probe learned to resume:
#
#   * a cell whose workdir holds `training/final_checkpoint` reuses it (unless --retrain)
#   * --skip-zero-shot drops the baseline pass, which is the single most expensive step — an
#     untrained model rarely emits a stop token, so every row runs the full output reserve
#     (38 minutes for Gemma on 300 BC5CDR rows, against 3 for the same model fine-tuned)
#
# So this does merge → GGUF → load-validate → score, plus a cheap re-run of the fine-tuned eval as
# the bf16-vs-Q4 comparison point. PROBE_WORKDIR must name the earlier run's work directory.
#
# Submit chained to the probe so it starts the moment the GPU frees:
#   sbatch --dependency=afterany:<probe_job_id> tests/pipeline/run_small_model_quant_pass.slurm
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache

# llama.cpp's convert/quantize CLIs must be on PATH or the whole point of this job reports a
# tooling error instead of a model result (training/quantize.py:336).
export PATH="/mmfs1/gscratch/intelligentsystems/evanly/llama.cpp:/mmfs1/gscratch/intelligentsystems/evanly/llama.cpp/build/bin:$PATH"

source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1); CUDA_MOD="${CUDA_MOD:-cuda/12.8.1}"
module load "$CUDA_MOD"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

nvidia-smi -L || true
source .venv_gpu/bin/activate
export SLM_CUDA_ISOLATION=0
export SLM_MAX_SEQ_LENGTH="${SLM_MAX_SEQ_LENGTH:-4096}"

WORKDIR="${PROBE_WORKDIR:-$PROJ/logs/probes/work-38765131}"
if [ ! -d "$WORKDIR" ]; then
    echo "ERROR: PROBE_WORKDIR $WORKDIR does not exist; nothing to quantize"
    exit 2
fi
OUT="$PROJ/logs/probes/small-models-quant-${SLURM_JOB_ID:-manual}.json"
mkdir -p "$(dirname "$OUT")"

echo "=== quantization pass over $WORKDIR → $OUT ==="
echo "=== checkpoints found: ==="
find "$WORKDIR" -maxdepth 3 -name final_checkpoint -type d | sed 's/^/    /'

python scripts/probe_small_models.py \
    --models google/gemma-3-270m-it HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-135M-Instruct \
    --tasks ner_bc5cdr xlam_bfcl \
    --eval-rows "${PROBE_EVAL_ROWS:-300}" \
    --train-rows "${PROBE_TRAIN_ROWS:-3000}" \
    --skip-zero-shot \
    --workdir "$WORKDIR" \
    --out "$OUT"
echo "=== done ==="
