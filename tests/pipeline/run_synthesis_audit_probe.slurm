#!/bin/bash
#SBATCH --job-name=slm-synthesis-audit-probe
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-synthesis-audit-probe-%j.out
#
# MEASUREMENT JOB, not a pipeline run, and the second of the two audits standing in for runs that
# cannot start. See run_suite_harness_probe.slurm for the first.
#
# WHY: two of the six failure modes the suite must be checked against live entirely inside the
# synthesis path — is the generated data legitimate, and do the verifier passes behave. Both are
# normally only observable partway through a multi-day agent run, and as of 2026-09-08 those runs
# die at their first orchestrator decision on an Anthropic credit-balance error.
#
# Synthesis needs no orchestrator: `data.curriculum.synthesize_examples` takes a `generate_fn` and
# the teacher behind it is the local vLLM server. So `scripts/probe_synthesis.py` generates rows
# per task, runs the EXACT verifier in enforce mode exactly as the pipeline does, then runs the
# TEACHER pass in SHADOW mode — recorded, never acted on — and reports both rates with reasons.
#
# `goemotions` already has this data from run 39708679, where the teacher's JSON-wrapper false
# positives first showed up (38-72% would-reject, all formatting complaints about correctly
# formatted rows). The other four tasks have none, and that is what this fills in.
#
# TWO GPUs because the vLLM teacher needs one of its own, same as a pipeline run. NOT requeued and
# time-boxed: a probe that dies should stay dead so the failure gets read.
#
# It does not train, score a model, or touch `stop_threshold`. Nothing here can change a run's goal.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache
export ERRANT_VENV="${ERRANT_VENV:-$PROJ/.venv_errant}"
export METRICS_VENV="${METRICS_VENV:-$PROJ/.venv_metrics}"

source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1); CUDA_MOD="${CUDA_MOD:-cuda/12.8.1}"
module load "$CUDA_MOD"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
nvidia-smi -L || true

# --- The same teacher the pipeline serves, launched the same way ---
export PATH="$PROJ/.venv_vllm/bin:$PATH"
.venv_vllm/bin/python -c 'import vllm' >/dev/null 2>&1 || bash scripts/setup_vllm_env.sh
SYNTH_MODEL="${SLM_SYNTH_MODEL:-Qwen/Qwen3.6-35B-A3B}"
SYNTH_LOG="$PROJ/logs/gpu_setup/synth-audit-${SLURM_JOB_ID:-manual}.out"
mkdir -p "$(dirname "$SYNTH_LOG")"
SYNTH_PORT=$(( 20000 + (${SLURM_JOB_ID:-0} % 20000) ))
# 16384 rather than the body's 8192 default: topv2's prompt enumerates all 166 intent and slot
# names, so five demonstrations plus the question do not fit a smaller context (run 39719881
# fell back to a zero-shot measurement for exactly this reason). max-num-seqs halves to pay for
# the KV cache, as the body's own guidance says.
export SLM_SYNTH_MAX_MODEL_LEN="${SLM_SYNTH_MAX_MODEL_LEN:-16384}"
CUDA_VISIBLE_DEVICES=0 .venv_vllm/bin/vllm serve "$SYNTH_MODEL" \
    --host 127.0.0.1 --port "$SYNTH_PORT" \
    --tensor-parallel-size 1 --quantization fp8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$SLM_SYNTH_MAX_MODEL_LEN" \
    --max-num-seqs 32 --gdn-prefill-backend triton \
    --language-model-only --reasoning-parser qwen3 --served-model-name "$SYNTH_MODEL" \
    > "$SYNTH_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "stopping vLLM ($VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true' EXIT
export SLM_SYNTH_ENDPOINT="http://127.0.0.1:${SYNTH_PORT}/v1"
export SLM_SYNTH_CONCURRENCY="${SLM_SYNTH_CONCURRENCY:-32}"
export SLM_SYNTH_WAIT_S="${SLM_SYNTH_WAIT_S:-2400}"
echo "=== vLLM synth ($SYNTH_MODEL) → $SLM_SYNTH_ENDPOINT → $SYNTH_LOG ==="

source .venv_gpu/bin/activate
export CUDA_VISIBLE_DEVICES=1
export SLM_CUDA_ISOLATION=0
export SLM_MAX_SEQ_LENGTH="${SLM_MAX_SEQ_LENGTH:-4096}"
# SHADOW, process-wide, exactly as the five task launchers set it. This is the whole point of the
# audit: the teacher judges every generated row and the verdict is recorded without being acted
# on. Setting it only around a later call — which an earlier version of the probe did — leaves the
# teacher pass INSIDE `synthesize_examples` running in enforce mode, so it silently drops rows and
# the measurement describes a different pipeline than the one that will run.
export SLM_VERIFY_SYNTH=shadow

OUT="$PROJ/logs/probes/synthesis-audit-${SLURM_JOB_ID:-manual}.json"
mkdir -p "$(dirname "$OUT")"
python - <<'PY'
import os, time
from data.synth_client import wait_until_available
print("waiting for the teacher to come up...", flush=True)
wait_until_available(log=print)
PY
python scripts/probe_synthesis.py \
    --tasks ${AUDIT_TASKS:-topv2 multiconer gec_bea19 goemotions dialogsum} \
    --n "${PROBE_SYNTH_N:-60}" \
    --out "$OUT"
echo "=== done ==="
