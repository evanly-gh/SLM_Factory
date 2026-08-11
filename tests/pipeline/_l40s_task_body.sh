#!/bin/bash
# Shared run body for L40S task-type pipeline tests (sourced by run_*_l40s.slurm).
# The caller sets `TASK` (and any optional SLM_* overrides) BEFORE sourcing this file; the
# #SBATCH directives live in the caller (they are only read from the submitted script).
#
# GPU LAYOUT — one node (cross-node localhost is firewalled here):
#   2 GPUs: synth GPU 0 at TP=1; pipeline GPU 1 (exclusive).
#   3 GPUs: synth GPUs 0-1 at TP=2; pipeline GPU 2 (exclusive).
#   4 GPUs: compatibility profile: synth GPUs 0-3 at TP=4 and pipeline GPU 0 shared.
#   5+ GPUs: synth GPUs 0-3 at TP=4; pipeline GPU 4 (exclusive).
# IDs are logical indices within the allocation. SLM_GPU_COUNT and the exported profile
# settings below are operator-overridable, with validation before model setup.
# ORCHESTRATOR: claude-sonnet-5 — $2/$10 per MTok introductory through 2026-08-31 (then $3/$15,
# i.e. sonnet-4-6 parity), 1M context at standard pricing. NON-cheap (full curriculum
# synthesis + CoT annotation + Sonnet decisions). SLM_ORCHESTRATOR_1M is kept but is a no-op on
# 4.6-and-later ids: the 1M window has been GA at standard rates since 2026-03-13 and the beta
# header is ignored.
# MODEL SELECTION: smallest_first — start at the smallest feasible model and escalate only on
# failure, so the cheapest model that clears the bar wins.
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

if [ -z "${TASK:-}" ]; then echo "ERROR: TASK not set by the caller script"; exit 2; fi

_detect_allocated_gpu_count() {
    local count="${SLM_GPU_COUNT:-}"
    local visible
    local -a devices
    if [ -z "$count" ] && [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        count="$SLURM_GPUS_ON_NODE"
    fi
    if [ -z "$count" ]; then
        visible="${CUDA_VISIBLE_DEVICES:-}"
        if [ -n "$visible" ] && [ "$visible" != "-1" ] && [ "$visible" != "NoDevFiles" ]; then
            IFS=',' read -r -a devices <<< "$visible"
            count="${#devices[@]}"
        fi
    fi
    if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: cannot detect allocated GPU count; set SLM_GPU_COUNT, SLURM_GPUS_ON_NODE, or CUDA_VISIBLE_DEVICES" >&2
        return 2
    fi
    printf '%s\n' "$count"
}

_validate_gpu_id() {
    local gpu_id="$1"
    local gpu_count="$2"
    local setting="$3"
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $setting contains invalid GPU ID '$gpu_id'" >&2
        return 2
    fi
    if [ "$gpu_id" -ge "$gpu_count" ]; then
        echo "ERROR: $setting GPU ID $gpu_id is outside the ${gpu_count}-GPU allocation" >&2
        return 2
    fi
}

_configure_gpu_profile() {
    local gpu_count
    local default_profile
    local default_synth_ids
    local default_tp
    local default_utilization
    local default_max_num_seqs
    local default_concurrency
    local default_pipeline_id
    local gpu_id
    local seen=","
    local -a synth_ids

    gpu_count="$(_detect_allocated_gpu_count)" || return $?
    if [ "$gpu_count" -lt 2 ]; then
        echo "ERROR: automatic L40S profiles require at least 2 allocated GPUs" >&2
        return 2
    fi
    case "$gpu_count" in
        2)
            default_profile="auto-2gpu"
            default_synth_ids="0"
            default_tp=1
            # The synth GPU is EXCLUSIVE in this profile (the pipeline trains on GPU 1), so the
            # only claim on its memory is vLLM itself. At 0.82 the 35B fp8 weights left just
            # 2.43 GiB of KV cache — 89,367 tokens, i.e. ~10.9x concurrency at full context —
            # which capped the useful batch far below what the GPU could decode. 0.90 roughly
            # triples the KV budget and is what makes the higher max-num-seqs below reachable.
            default_utilization=0.90
            default_max_num_seqs=64
            default_concurrency=48
            default_pipeline_id=1
            ;;
        3)
            default_profile="auto-3gpu"
            default_synth_ids="0,1"
            default_tp=2
            default_utilization=0.82
            default_max_num_seqs=48
            default_concurrency=32
            default_pipeline_id=2
            ;;
        4)
            default_profile="compat-4gpu-shared"
            default_synth_ids="0,1,2,3"
            default_tp=4
            default_utilization=0.45
            default_max_num_seqs=96
            default_concurrency=64
            default_pipeline_id=0
            ;;
        *)
            default_profile="auto-5plus-gpu"
            default_synth_ids="0,1,2,3"
            default_tp=4
            default_utilization=0.82
            default_max_num_seqs=96
            default_concurrency=64
            default_pipeline_id=4
            ;;
    esac

    export SLM_GPU_COUNT="$gpu_count"
    export SLM_GPU_PROFILE="${SLM_GPU_PROFILE:-$default_profile}"
    export SLM_SYNTH_GPU_IDS="${SLM_SYNTH_GPU_IDS:-$default_synth_ids}"
    export SLM_SYNTH_TP="${SLM_SYNTH_TP:-$default_tp}"
    export SLM_SYNTH_GPU_UTILIZATION="${SLM_SYNTH_GPU_UTILIZATION:-$default_utilization}"
    export SLM_SYNTH_MAX_NUM_SEQS="${SLM_SYNTH_MAX_NUM_SEQS:-$default_max_num_seqs}"
    export SLM_SYNTH_CONCURRENCY="${SLM_SYNTH_CONCURRENCY:-$default_concurrency}"
    export SLM_PIPELINE_GPU_ID="${SLM_PIPELINE_GPU_ID:-$default_pipeline_id}"

    case "$SLM_SYNTH_TP" in
        1|2|4) ;;
        *)
            echo "ERROR: SLM_SYNTH_TP must be one of 1, 2, or 4" >&2
            return 2
            ;;
    esac
    IFS=',' read -r -a synth_ids <<< "$SLM_SYNTH_GPU_IDS"
    if [ "${#synth_ids[@]}" -ne "$SLM_SYNTH_TP" ]; then
        echo "ERROR: SLM_SYNTH_GPU_IDS must contain exactly SLM_SYNTH_TP=$SLM_SYNTH_TP IDs" >&2
        return 2
    fi
    for gpu_id in "${synth_ids[@]}"; do
        _validate_gpu_id "$gpu_id" "$gpu_count" "SLM_SYNTH_GPU_IDS" || return $?
        if [[ "$seen" == *",$gpu_id,"* ]]; then
            echo "ERROR: SLM_SYNTH_GPU_IDS contains duplicate GPU ID $gpu_id" >&2
            return 2
        fi
        seen="${seen}${gpu_id},"
    done
    _validate_gpu_id "$SLM_PIPELINE_GPU_ID" "$gpu_count" "SLM_PIPELINE_GPU_ID" || return $?
    if [ "$gpu_count" -ne 4 ] && [[ "$seen" == *",$SLM_PIPELINE_GPU_ID,"* ]]; then
        echo "ERROR: synth and pipeline GPUs must not overlap outside the 4-GPU compatibility profile" >&2
        return 2
    fi
    if ! [[ "$SLM_SYNTH_MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]] ||
       ! [[ "$SLM_SYNTH_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: SLM_SYNTH_MAX_NUM_SEQS and SLM_SYNTH_CONCURRENCY must be positive integers" >&2
        return 2
    fi
    if ! [[ "$SLM_SYNTH_GPU_UTILIZATION" =~ ^0\.[0-9]*[1-9][0-9]*$|^1(\.0+)?$ ]]; then
        echo "ERROR: SLM_SYNTH_GPU_UTILIZATION must be greater than 0 and at most 1" >&2
        return 2
    fi
}

_configure_gpu_profile
echo "=== GPU profile $SLM_GPU_PROFILE: allocation=$SLM_GPU_COUNT synth=$SLM_SYNTH_GPU_IDS TP=$SLM_SYNTH_TP util=$SLM_SYNTH_GPU_UTILIZATION max-seqs=$SLM_SYNTH_MAX_NUM_SEQS concurrency=$SLM_SYNTH_CONCURRENCY pipeline=$SLM_PIPELINE_GPU_ID ==="

# Stable across Slurm requeues (same job id), but operator-overridable for a
# deliberate continuation or a separately managed run directory.
_SLM_RUN_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual-$$}}"
export SLM_RUN_DIR="${SLM_RUN_DIR:-$PROJ/logs/runs/${SLURM_JOB_NAME:-slm-task}-${_SLM_RUN_KEY}}"
export SLM_CURATION_LOG_PATH="${SLM_CURATION_LOG_PATH:-$SLM_RUN_DIR/data-curation.md}"
export SLM_CUDA_ISOLATION=1
SLM_TERM_GRACE_S="${SLM_TERM_GRACE_S:-120}"
if ! [[ "$SLM_TERM_GRACE_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SLM_TERM_GRACE_S must be a positive integer"
    exit 2
fi
nvidia-smi -L || true

# --- Launch the co-located vLLM synth server (localhost) on the selected profile GPUs ---
source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1); CUDA_MOD="${CUDA_MOD:-cuda/12.8.1}"
module load "$CUDA_MOD"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
# flashinfer JITs kernels at runtime → needs nvcc (CUDA module) AND ninja (vllm venv bin).
export PATH="$PROJ/.venv_vllm/bin:$PATH"
.venv_vllm/bin/python -c 'import vllm' >/dev/null 2>&1 || bash scripts/setup_vllm_env.sh
.venv_vllm/bin/python -c 'import ninja' >/dev/null 2>&1 || .venv_vllm/bin/python -m pip install ninja 2>/dev/null || true

SYNTH_MODEL="${SLM_SYNTH_MODEL:-Qwen/Qwen3.6-35B-A3B}"
# GPU infrastructure logs live apart from pipeline run logs (logs/slurm/) so the run
# directory stays readable. The redirect below fails outright if the dir is absent.
SYNTH_LOG="$PROJ/logs/gpu_setup/synth-l40s-${SLM_GPU_PROFILE}-${SLURM_JOB_ID:-manual}.out"
mkdir -p "$(dirname "$SYNTH_LOG")"
# JOB-UNIQUE port (shared localhost namespace on multi-tenant nodes).
SYNTH_PORT=$(( 20000 + (${SLURM_JOB_ID:-0} % 20000) ))
# CUDA graphs are ON by default. Qwen3.6-35B-A3B activates only ~3B parameters per token, so
# decode is dominated by per-layer kernel-launch overhead rather than by arithmetic, and
# --enforce-eager (which disables torch.compile AND CUDA graphs) is exactly the wrong trade for
# that shape: run 38245785 sustained 76 tok/s across 8 concurrent requests, ~9.5 tok/s per
# sequence, on hardware that should do far better. Eager mode remains one variable away in case
# graph capture misbehaves on the GDN hybrid layers.
SYNTH_EAGER_FLAG=""
if [ "${SLM_SYNTH_ENFORCE_EAGER:-0}" = "1" ]; then
    SYNTH_EAGER_FLAG="--enforce-eager"
fi
echo "=== vLLM synth ($SYNTH_MODEL) TP=$SLM_SYNTH_TP on GPUs $SLM_SYNTH_GPU_IDS → localhost:$SYNTH_PORT → $SYNTH_LOG ==="
echo "=== vLLM synth: max-num-seqs=$SLM_SYNTH_MAX_NUM_SEQS util=$SLM_SYNTH_GPU_UTILIZATION cuda-graphs=$([ -n "$SYNTH_EAGER_FLAG" ] && echo off || echo on) ==="
CUDA_VISIBLE_DEVICES="$SLM_SYNTH_GPU_IDS" .venv_vllm/bin/vllm serve "$SYNTH_MODEL" \
    --host 127.0.0.1 --port "$SYNTH_PORT" \
    --tensor-parallel-size "$SLM_SYNTH_TP" --quantization fp8 \
    --gpu-memory-utilization "$SLM_SYNTH_GPU_UTILIZATION" --max-model-len 8192 \
    --max-num-seqs "$SLM_SYNTH_MAX_NUM_SEQS" --gdn-prefill-backend triton $SYNTH_EAGER_FLAG \
    --language-model-only --reasoning-parser qwen3 --served-model-name "$SYNTH_MODEL" \
    > "$SYNTH_LOG" 2>&1 &
VLLM_PID=$!
PIPELINE_PID=""
REQUEUE_REQUESTED=0
TERM_REQUESTED=0

_cleanup_task_run() {
    echo "stopping vLLM ($VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
}

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

_checkpoint_and_requeue() {
    if [ "$TERM_REQUESTED" -eq 1 ]; then
        return
    fi
    echo "received Slurm USR1 notice; checkpointing pipeline before requeue"
    REQUEUE_REQUESTED=1
    if [ -n "$PIPELINE_PID" ]; then
        kill -USR1 "$PIPELINE_PID" 2>/dev/null || true
    fi
}

_forward_term() {
    if [ "$TERM_REQUESTED" -eq 0 ]; then
        echo "received TERM; requesting pipeline checkpoint and finalization"
        TERM_REQUESTED=1
        if [ -n "$PIPELINE_PID" ]; then
            kill -TERM "$PIPELINE_PID" 2>/dev/null || true
        fi
    fi
}

trap _cleanup_task_run EXIT
trap _checkpoint_and_requeue USR1
trap _forward_term TERM
export SLM_SYNTH_ENDPOINT="http://127.0.0.1:${SYNTH_PORT}/v1"

# --- Run the pipeline (its own venv), pinned to the profile GPU; Sonnet-1M; NON-cheap ---
source .venv_gpu/bin/activate
export CUDA_VISIBLE_DEVICES="$SLM_PIPELINE_GPU_ID"
export SLM_ORCHESTRATOR_MODEL="${SLM_ORCHESTRATOR_MODEL:-claude-sonnet-5}"   # Sonnet 5
export SLM_ORCHESTRATOR_1M=1                     # enable Sonnet's 1M-token context (beta)
unset SLM_CHEAP                                  # NON-cheap: full synthesis + CoT + Sonnet
export SLM_QUANT_EVAL=1
export SLM_MODEL_SELECTION_STRATEGY="${SLM_MODEL_SELECTION_STRATEGY:-smallest_first}"
export SLM_SYNTH_WAIT_S=2400                     # preflight waits up to 40 min for the server
export SLM_SYNTH_CONCURRENCY
export SLM_MAX_SEQ_LENGTH="${SLM_MAX_SEQ_LENGTH:-4096}"
export SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION="${SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION:-50}"
export SLM_EVAL_MAX_NEW_TOKENS_NER="${SLM_EVAL_MAX_NEW_TOKENS_NER:-512}"
export SLM_EVAL_MAX_NEW_TOKENS_MATH="${SLM_EVAL_MAX_NEW_TOKENS_MATH:-512}"
export SLM_EVAL_MAX_NEW_TOKENS_GENERATION="${SLM_EVAL_MAX_NEW_TOKENS_GENERATION:-512}"
export SLM_EVAL_MAX_NEW_TOKENS_APPS="${SLM_EVAL_MAX_NEW_TOKENS_APPS:-1024}"
export SLM_APPS_PROBLEM_TIMEOUT_S="${SLM_APPS_PROBLEM_TIMEOUT_S:-6}"
# These jobs checkpoint and requeue from Slurm's USR1 notice. The aggregate wall-clock auto-termination is disabled
# so cumulative resumed time cannot end a healthy run before the scheduler signal; a
# nonzero guard remains available for non-requeue executions.
export SLM_MAX_WALLCLOCK_S="${SLM_MAX_WALLCLOCK_S:-0}"
# Readiness runs intentionally exercise Exa+Sonnet dataset discovery first. Every paid call
# is captured by cost-observability; checksum-verified data/local bundles remain the local fallback.
export SLM_AGENT_FIRST_DATASET_DISCOVERY=1

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

echo "=== SLM Factory task run — $SLM_GPU_PROFILE, NON-cheap, Sonnet-1M orchestrator ==="
echo "Task: $TASK"
echo "Run dir: $SLM_RUN_DIR"
if [ "${#RESUME_ARGS[@]}" -gt 0 ]; then
    echo "Resume checkpoint: ${RESUME_ARGS[1]}"
fi

python tests/pipeline/run.py "${RESUME_ARGS[@]}" "$TASK" &
PIPELINE_PID=$!
set +e
wait "$PIPELINE_PID"
PIPELINE_STATUS=$?
if [ "$REQUEUE_REQUESTED" -eq 1 ] && [ "$TERM_REQUESTED" -eq 0 ]; then
    # Reap after the forwarded signal so the runner has time to finish its
    # atomic JSON + SQLite checkpoint publication.
    wait "$PIPELINE_PID" 2>/dev/null
    PIPELINE_STATUS=$?
fi
if [ "$TERM_REQUESTED" -eq 1 ]; then
    # Ignore repeated scheduler signals while the child owns final checkpoint,
    # SQLite, summary, and observability publication.
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
        echo "checkpoint complete; requeueing Slurm job ${SLURM_JOB_ID}"
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
