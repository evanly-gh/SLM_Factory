#!/bin/bash
#SBATCH --job-name=slm-emo-orch-l40s
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-emo-orch-l40s-%j.out
#
# Same task as run_emotion_orch_full.slurm, but on the group-OWNED gpu-l40s partition
# (dedicated, non-preemptible, long walltime) with MULTIPLE GPUs, split so hard-negative
# data curation is fast:
#   GPUs 0,1  → vLLM synth server, Qwen3.6-35B-A3B FP8, tensor-parallel across 2 L40S (96GB
#               → large KV, --max-num-seqs 64). No training shares these, so util can be high.
#   GPU 2     → the LoRA training / eval pipeline (pinned via CUDA_VISIBLE_DEVICES).
# Combined with concurrent synthesis (SLM_SYNTH_CONCURRENCY), the ~485 sequential per-rebuild
# generations (~15 min on the single-A100 run) collapse to ~n/W batched round-trips (~1-2 min).
#
# L40S is 48GB, so the 35B does NOT fit alongside training on ONE card — hence the 2-GPU
# tensor-parallel server + a dedicated 3rd GPU for training (cross-node localhost is firewalled
# on this cluster, so server + pipeline MUST co-locate on one node; gpu-l40s nodes have 8).
# no `-u`: lmod init references unbound LD_LIBRARY_PATH.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache
_SLM_RUN_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual-$$}}"
export SLM_RUN_DIR="${SLM_RUN_DIR:-$PROJ/logs/runs/${SLURM_JOB_NAME:-slm-emotion}-${_SLM_RUN_KEY}}"
export SLM_CURATION_LOG_PATH="${SLM_CURATION_LOG_PATH:-$SLM_RUN_DIR/data-curation.md}"
export SLM_CUDA_ISOLATION=1
SLM_TERM_GRACE_S="${SLM_TERM_GRACE_S:-120}"
if ! [[ "$SLM_TERM_GRACE_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SLM_TERM_GRACE_S must be a positive integer"
    exit 2
fi

nvidia-smi -L || true

# --- Launch the co-located vLLM synthesis server (localhost), tensor-parallel on GPUs 0,1 ---
source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1); CUDA_MOD="${CUDA_MOD:-cuda/12.8.1}"
module load "$CUDA_MOD"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
# flashinfer JITs kernels at runtime → needs nvcc (CUDA module) AND ninja (vllm venv bin).
export PATH="$PROJ/.venv_vllm/bin:$PATH"
.venv_vllm/bin/python -c 'import vllm' >/dev/null 2>&1 || bash scripts/setup_vllm_env.sh
.venv_vllm/bin/python -c 'import ninja' >/dev/null 2>&1 || .venv_vllm/bin/python -m pip install ninja 2>/dev/null || true

SYNTH_MODEL="${SLM_SYNTH_MODEL:-Qwen/Qwen3.6-35B-A3B}"
SYNTH_LOG="$PROJ/logs/slurm/synth-l40s-${SLURM_JOB_ID}.out"
# JOB-UNIQUE port (shared localhost namespace on multi-tenant nodes).
SYNTH_PORT=$(( 20000 + (${SLURM_JOB_ID:-0} % 20000) ))
echo "=== launching vLLM synth ($SYNTH_MODEL) TP=2 on GPUs 0,1 → localhost:$SYNTH_PORT → $SYNTH_LOG ==="
# TP=2 over 2×L40S (96GB): fits the 35B FP8 with a large KV budget, so --gpu-memory-utilization
# 0.90 and --max-num-seqs 64 (these GPUs are dedicated to synthesis). triton GDN prefill +
# enforce-eager per the model card (skip the fragile flashinfer GDN JIT / CUDA-graph capture).
CUDA_VISIBLE_DEVICES=0,1 .venv_vllm/bin/vllm serve "$SYNTH_MODEL" --host 127.0.0.1 --port "$SYNTH_PORT" \
    --tensor-parallel-size 2 --quantization fp8 --gpu-memory-utilization 0.90 --max-model-len 8192 \
    --max-num-seqs 64 --gdn-prefill-backend triton --enforce-eager \
    --language-model-only --reasoning-parser qwen3 --served-model-name "$SYNTH_MODEL" \
    > "$SYNTH_LOG" 2>&1 &
VLLM_PID=$!
PIPELINE_PID=""
REQUEUE_REQUESTED=0
TERM_REQUESTED=0
_cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }
_wait_for_pipeline_with_grace() {
    local grace_s="$1"
    local watchdog_pid
    local status
    (
        sleep "$grace_s"
        if kill -0 "$PIPELINE_PID" 2>/dev/null; then
            echo "ERROR: TERM grace expired; force-killing pipeline $PIPELINE_PID"
            kill -KILL "$PIPELINE_PID" 2>/dev/null || true
        fi
    ) &
    watchdog_pid=$!
    wait "$PIPELINE_PID"
    status=$?
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    return "$status"
}
_checkpoint_requeue() {
    if [ "$TERM_REQUESTED" -eq 1 ]; then
        return
    fi
    REQUEUE_REQUESTED=1
    if [ -n "$PIPELINE_PID" ]; then
        kill -USR1 "$PIPELINE_PID" 2>/dev/null || true
    fi
}
_checkpoint_term() {
    if [ "$TERM_REQUESTED" -eq 0 ]; then
        TERM_REQUESTED=1
        if [ -n "$PIPELINE_PID" ]; then
            kill -TERM "$PIPELINE_PID" 2>/dev/null || true
        fi
    fi
}
trap _cleanup EXIT
trap _checkpoint_requeue USR1
trap _checkpoint_term TERM
export SLM_SYNTH_ENDPOINT="http://127.0.0.1:${SYNTH_PORT}/v1"

# --- Run the pipeline (its own venv), pinned to GPU 2 so it never contends with the server ---
source .venv_gpu/bin/activate
export CUDA_VISIBLE_DEVICES=2
export SLM_ORCHESTRATOR_MODEL=claude-haiku-4-5
export SLM_QUANT_EVAL=1
export SLM_MODEL_SELECTION_STRATEGY=orchestrator_choice
export SLM_MAX_SEQ_LENGTH=4096
export SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION=50
export SLM_SYNTH_WAIT_S=2400        # preflight waits up to 40 min for the co-located server
export SLM_SYNTH_CONCURRENCY=32     # concurrent hard-neg generations → feeds vLLM's 64 slots
# Slurm USR1 drives the durable checkpoint/requeue rollover. Disable aggregate wall-clock auto-termination
# so cumulative resumed time cannot silently end before that signal.
export SLM_MAX_WALLCLOCK_S=0

TASK="Fine tune a small model to classify the emotion expressed in a short English text message into exactly one of six categories — sadness, joy, love, anger, fear, or surprise — targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
_durable_resume_ready() {
    [ -s "$SLM_RUN_DIR/run-manifest.json" ] &&
    [ -s "$SLM_RUN_DIR/checkpoint.json" ] &&
    "$PROJ/.venv_gpu/bin/python" - "$SLM_RUN_DIR" <<'PY'
import sys
from agent.checkpoint import durable_resume_available
raise SystemExit(0 if durable_resume_available(sys.argv[1]) else 1)
PY
}
RESUME_ARGS=()
if [ -n "${SLM_RESUME:-}" ]; then
    RESUME_ARGS=(--resume "$SLM_RESUME")
elif _durable_resume_ready; then
    export SLM_RESUME="$SLM_RUN_DIR/checkpoint.json"
    RESUME_ARGS=(--resume "$SLM_RESUME")
fi
echo "=== Emotion orchestrator_choice run — OWNED gpu-l40s, TP=2 synth + concurrent curation ==="
echo "Task: $TASK"
python tests/pipeline/run.py "${RESUME_ARGS[@]}" "$TASK" &
PIPELINE_PID=$!
set +e
wait "$PIPELINE_PID"
PIPELINE_STATUS=$?
if [ "$REQUEUE_REQUESTED" -eq 1 ] && [ "$TERM_REQUESTED" -eq 0 ]; then
    wait "$PIPELINE_PID" 2>/dev/null
    PIPELINE_STATUS=$?
fi
if [ "$TERM_REQUESTED" -eq 1 ]; then
    trap '' TERM USR1
    _wait_for_pipeline_with_grace "$SLM_TERM_GRACE_S"
    FINAL_STATUS=$?
    if [ "$FINAL_STATUS" -ne 127 ]; then
        PIPELINE_STATUS=$FINAL_STATUS
    fi
fi
set -e
if [ "$TERM_REQUESTED" -eq 1 ]; then
    exit "$PIPELINE_STATUS"
fi
if [ "$REQUEUE_REQUESTED" -eq 1 ]; then
    if _durable_resume_ready; then
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    echo "ERROR: durable manifest/checkpoint missing; refusing unsafe requeue"
    exit 1
fi
if [ "$PIPELINE_STATUS" -ne 0 ]; then
    exit "$PIPELINE_STATUS"
fi
echo "=== done ==="
