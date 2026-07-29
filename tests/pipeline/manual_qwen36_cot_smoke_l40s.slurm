#!/bin/bash
#SBATCH --job-name=manual-qwen36-cot
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/manual-qwen36-cot-%j.out
#
# MANUAL / UNSUBMITTED readiness script. Do not auto-submit this file.
# It intentionally downloads/loads Qwen3.6 only when a human submits it.
#
# PASS CRITERIA:
#   1. A co-located FP8 vLLM server becomes reachable on the single allocated L40S.
#   2. Exactly one bounded OpenAI-compatible CoT generation request returns non-empty text.
#   3. Non-thinking mode emits no <think> transport preamble.
#   4. Readiness fails within 15 minutes, first-call JIT/request within 4 minutes, and server within 35 minutes.
# No paid/cloud endpoint is configured or called.
# OBSERVABILITY: this standalone smoke does not create cost.json or timings.json.
# Inspect the SLURM output for the PASS line and the vLLM server log named below for
# startup/request diagnostics.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
test -x .venv_vllm/bin/vllm
test -x .venv_gpu/bin/python

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
export MAX_GENERATION_CALLS=1
# Required only because config.py validates global API settings at import time.
# The smoke points exclusively at localhost and performs no cloud call.
export ANTHROPIC_API_KEY=manual-readiness-no-call
export EXA_API_KEY=manual-readiness-no-call
MODEL=Qwen/Qwen3.6-35B-A3B
PORT=$((24000 + (${SLURM_JOB_ID:-0} % 16000)))
LOG="$PROJ/logs/slurm/manual-qwen36-vllm-${SLURM_JOB_ID:-manual}.out"

source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1) || true
if [ -n "$CUDA_MOD" ] && module load "$CUDA_MOD"; then
    :
elif module load cuda/12.8.1; then
    CUDA_MOD=cuda/12.8.1
else
    echo "FAIL: no usable CUDA 12.8 module found (discovery and cuda/12.8.1 fallback failed)"
    exit 2
fi
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$PROJ/.venv_vllm/bin:$PATH"

timeout --signal=TERM 35m .venv_vllm/bin/vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --gpu-memory-utilization 0.82 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 2048 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --served-model-name "$MODEL" \
    --enforce-eager \
    > "$LOG" 2>&1 &
VLLM_PID=$!
cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

export SLM_SYNTH_ENDPOINT="http://127.0.0.1:${PORT}/v1"
export SLM_SYNTH_MODEL="$MODEL"
export SLM_SYNTH_API_KEY=EMPTY

READY=0
READINESS_TIMEOUT_S=900
READINESS_DEADLINE=$((SECONDS + READINESS_TIMEOUT_S))
while [ "$SECONDS" -lt "$READINESS_DEADLINE" ]; do
    REMAINING_S=$((READINESS_DEADLINE - SECONDS))
    if [ "$REMAINING_S" -le 0 ]; then
        break
    fi
    CURL_TIMEOUT_S=3
    if [ "$REMAINING_S" -lt "$CURL_TIMEOUT_S" ]; then
        CURL_TIMEOUT_S="$REMAINING_S"
    fi
    if curl -fsS --noproxy '*' --max-time "$CURL_TIMEOUT_S" "$SLM_SYNTH_ENDPOINT/models" >/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "FAIL: vLLM exited before readiness; inspect $LOG"
        exit 1
    fi
    REMAINING_S=$((READINESS_DEADLINE - SECONDS))
    if [ "$REMAINING_S" -le 0 ]; then
        break
    fi
    SLEEP_S=5
    if [ "$REMAINING_S" -lt "$SLEEP_S" ]; then
        SLEEP_S="$REMAINING_S"
    fi
    sleep "$SLEEP_S"
done
if [ "$READY" -ne 1 ]; then
    echo "FAIL: vLLM did not become ready within 15 minutes; inspect $LOG"
    exit 1
fi

timeout --signal=TERM 240s .venv_gpu/bin/python - <<'PY'
import os

from data.synth_client import get_generate_fn


max_calls = int(os.environ["MAX_GENERATION_CALLS"])
assert max_calls == 1
generate = get_generate_fn(log=print, request_timeout=210.0)
assert generate is not None
calls = 0
calls += 1
reasoning = generate(
    "Explain in three concise steps why 12 divided by 3 equals 4. "
    "Return only the reasoning steps, not a separate final-answer line.",
    temperature=0.3,
    max_tokens=96,
)
assert calls == max_calls
assert reasoning.strip()
assert "<think>" not in reasoning.lower()
print("PASS qwen36 local CoT smoke:", reasoning)
PY
