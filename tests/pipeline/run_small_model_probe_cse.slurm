#!/bin/bash
#SBATCH --job-name=slm-small-probe-cse
#SBATCH --account=gpu-l40s-cse
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-small-probe-cse-%j.out
#
# MEASUREMENT JOB, not a pipeline run. Answers one question: do google/gemma-3-270m,
# SmolLM2-360M and SmolLM2-135M train, score and quantize in this stack, and what do they get on
# ner_bc5cdr and xlam_bfcl.
#
# ONE GPU, NO TEACHER. The agent loop needs a second GPU for the vLLM synthesis server; this
# probe does no synthesis, no mining and no orchestration, so it needs neither. That also makes it
# schedulable in gaps a two-GPU job cannot fill.
#
# NOT REQUEUED and time-boxed to 8h, for the same reason the verification runs are not: a probe
# that dies should stay dead so the failure gets read. Results are written to the JSON after every
# (model, task) cell, so a job killed partway still reports the cells that finished.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache

# llama.cpp's convert/quantize CLIs must be on PATH or the quantization half reports a
# tooling error instead of a model result (training/quantize.py:336).
export PATH="/mmfs1/gscratch/intelligentsystems/evanly/llama.cpp:/mmfs1/gscratch/intelligentsystems/evanly/llama.cpp/build/bin:$PATH"

source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1); CUDA_MOD="${CUDA_MOD:-cuda/12.8.1}"
module load "$CUDA_MOD"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

nvidia-smi -L || true
source .venv_gpu/bin/activate

# The probe drives train + eval directly rather than through the agent, so the disposable-worker
# isolation the loop relies on is unnecessary here and only adds process churn.
export SLM_CUDA_ISOLATION=0
export SLM_MAX_SEQ_LENGTH="${SLM_MAX_SEQ_LENGTH:-4096}"

OUT="$PROJ/logs/probes/small-models-${SLURM_JOB_ID:-manual}.json"
mkdir -p "$(dirname "$OUT")"

# INSTRUCT by default, and the naming is a genuine trap worth spelling out. Qwen uses the PLAIN
# name for the post-trained model and suffixes the base one `-Base`, so the pool's
# `Qwen/Qwen3-0.6B` is instruct. Google and HuggingFaceTB do the opposite: plain is the base
# checkpoint and `-it` / `-Instruct` is post-trained. Comparing a bare `gemma-3-270m` against
# `Qwen3-0.6B` therefore compares a base model against an instruct one, which makes the zero-shot
# column meaningless and understates the candidate. Override PROBE_MODELS to measure the bases.
PROBE_MODELS="${PROBE_MODELS:-google/gemma-3-270m-it HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-135M-Instruct}"

echo "=== small-model probe → $OUT ==="
echo "=== models: $PROBE_MODELS ==="
python scripts/probe_small_models.py \
    --models $PROBE_MODELS \
    --tasks ner_bc5cdr xlam_bfcl \
    --eval-rows "${PROBE_EVAL_ROWS:-300}" \
    --train-rows "${PROBE_TRAIN_ROWS:-3000}" \
    --workdir "$PROJ/logs/probes/work-${SLURM_JOB_ID:-manual}" \
    --out "$OUT"
echo "=== done ==="
