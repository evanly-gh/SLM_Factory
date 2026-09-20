#!/bin/bash
#SBATCH --job-name=slm-suite-harness-probe
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-suite-harness-probe-%j.out
#
# MEASUREMENT JOB, not a pipeline run, and it exists because the pipeline currently CANNOT run.
#
# WHY THIS EXISTS (2026-09-08). All five on-device SFT suite runs died at their first orchestrator
# decision with `FatalLLMError: [escalate] Claude API call failed — 'Your credit balance is too low
# to access the Anthropic API'`. That is a billing state, not a defect, and the pipeline is right
# to stop rather than continue on substituted heuristics while still reporting an orchestrator run.
#
# But the QUESTIONS that still need answering before those runs are worth restarting do not need an
# orchestrator at all:
#
#   1. Is each task's baseline eval scored correctly, with no scorer false positives or negatives?
#   2. Is any baseline depressed by TOKEN TRUNCATION rather than by model quality?
#   3. Does each bespoke eval harness — prompt, extractor, scorer — work end to end on a real model?
#   4. Are any of the accuracy numbers implausibly low?
#
# `scripts/probe_small_models.py` answers all four: it drives train and eval DIRECTLY, makes zero
# Claude calls (verified: no anthropic import, no orchestrator reference), and reports zero-shot,
# fine-tuned and quantized scores per (model, task) with format_valid alongside. So the five tasks
# can be de-risked now and restarted clean the moment credits exist, instead of rediscovering
# harness bugs over days of seven-day runs.
#
# ONE GPU, NO TEACHER. The agent loop needs a second GPU for the vLLM synthesis server; this probe
# does no synthesis, no mining and no orchestration, so it needs neither — and that makes it
# schedulable in gaps a two-GPU job cannot fill.
#
# NOT REQUEUED and time-boxed, for the same reason the verification runs are not: a probe that dies
# should stay dead so the failure gets read. Results are written to the JSON after every
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

# The report metrics live outside the training venv: ERRANT needs spacy<4 and BERTScore predates
# transformers 5.x. `gec_bea19` cannot be scored at all without the first one, so a probe that
# omitted these would report a harness failure that is really a missing PATH.
export ERRANT_VENV="${ERRANT_VENV:-$PROJ/.venv_errant}"
export METRICS_VENV="${METRICS_VENV:-$PROJ/.venv_metrics}"

# llama.cpp's convert/quantize CLIs must be on PATH or the quantization half reports a tooling
# error instead of a model result (training/quantize.py:336).
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

OUT="$PROJ/logs/probes/suite-harness-${SLURM_JOB_ID:-manual}.json"
mkdir -p "$(dirname "$OUT")"

# TIER 1 AND A MID TIER. `SmolLM2-360M-Instruct` is where the agent loop actually starts under
# `smallest_first`, so it is the model whose baseline the runs will report first and the one whose
# oddities matter most; `Qwen3-1.7B` is a real pool member two tiers up and catches anything that
# is an artifact of the smallest model rather than of the harness.
#
# INSTRUCT deliberately. Qwen uses the plain name for the post-trained model and suffixes the base
# `-Base`; HuggingFaceTB does the opposite. Mixing the two conventions would compare a base model
# against an instruct one and make the zero-shot column meaningless.
PROBE_MODELS="${PROBE_MODELS:-HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen3-1.7B}"

# Quantization is SKIPPED by default. It answers "does the deploy artifact survive", which is a
# real question and not this one — and it roughly doubles the runtime per cell. The zero-shot pass
# is explicitly KEPT even though it is the slowest step, because it is the measurement being
# audited: an untrained model rarely emits a stop token, so every row runs the full output reserve,
# which is exactly the condition under which a truncation bug would show.
echo "=== suite harness probe → $OUT ==="
echo "=== models: $PROBE_MODELS ==="
python scripts/probe_small_models.py \
    --models $PROBE_MODELS \
    --tasks topv2 multiconer gec_bea19 goemotions dialogsum \
    --eval-rows "${PROBE_EVAL_ROWS:-300}" \
    --train-rows "${PROBE_TRAIN_ROWS:-3000}" \
    --skip-quant \
    --workdir "$PROJ/logs/probes/work-suite-${SLURM_JOB_ID:-manual}" \
    --out "$OUT"
echo "=== done ==="
