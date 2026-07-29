#!/bin/bash
#SBATCH --job-name=manual-qwen35-ready
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/manual-qwen35-ready-%j.out
#
# MANUAL / UNSUBMITTED readiness script. Do not auto-submit this file.
# It intentionally downloads/loads Qwen3.5 only when a human submits it.
#
# PASS CRITERIA:
#   1. One-epoch text-only LoRA training finishes in a disposable CUDA worker.
#   2. Separate disposable workers pass single inference AND real infer_batch over
#      four non-empty, variable-length prompts in one short batch (>1 prompt).
#   3. FastVisionModel merge produces a non-empty Q4_K_M GGUF through llama.cpp.
#   4. Deployment-format llama-cpp eval completes with finite 0 <= F1 <= 1.
# TIMING CRITERIA:
#   train <=15m; single infer <=6m; batched infer <=6m; Q4 build <=15m;
#   deployment eval <=8m; the complete Python smoke is hard-bounded to <=35m.
# VRAM CRITERIA:
#   sampled peak GPU use <=44000 MiB and post-worker use <= baseline +512 MiB.
#   Any CUDA OOM, missing sample, or exceeded threshold fails the script.
set -euo pipefail

PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
LLAMA_CPP_DIR=/mmfs1/gscratch/intelligentsystems/evanly/llama.cpp
export PATH="$LLAMA_CPP_DIR:$LLAMA_CPP_DIR/build/bin:$PATH"

test -x .venv_gpu/bin/python
export CUDA_VISIBLE_DEVICES=0
export SLM_CUDA_ISOLATION=1
export SLM_EARLY_STOPPING=0
export SLM_QUANT_EVAL=1
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
# Required import-time placeholders only; this smoke has no cloud/API call path.
export ANTHROPIC_API_KEY=manual-readiness-no-call
export EXA_API_KEY=manual-readiness-no-call

CONVERT_TOOL=""
for candidate in convert_hf_to_gguf convert_hf_to_gguf.py convert-hf-to-gguf.py; do
    if command -v "$candidate" >/dev/null 2>&1; then
        CONVERT_TOOL=$(command -v "$candidate")
        break
    fi
done
if [ -z "$CONVERT_TOOL" ]; then
    echo "FAIL: convert_hf_to_gguf not found. Build llama.cpp and add its conversion script to PATH."
    exit 2
fi
if ! command -v llama-quantize >/dev/null 2>&1; then
    echo "FAIL: llama-quantize not found. Build llama.cpp and add its binaries to PATH."
    exit 2
fi
if ! .venv_gpu/bin/python -c 'import llama_cpp' >/dev/null 2>&1; then
    echo "FAIL: llama-cpp-python is unavailable in .venv_gpu; install the CUDA-enabled build."
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FAIL: nvidia-smi is required for the VRAM criteria."
    exit 2
fi

RUN_DIR="$PROJ/artifacts/manual_qwen35_readiness_${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR"
export RUN_DIR
GPU_SNAPSHOT_LOG="$RUN_DIR/gpu-memory-snapshots.csv"
MAX_PEAK_USED_MIB=44000
MAX_POST_WORKER_DELTA_MIB=512
IFS=',' read -r BASELINE_USED_MIB GPU_TOTAL_MIB <<< "$(
    nvidia-smi --id=0 \
        --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits
)"
BASELINE_USED_MIB=${BASELINE_USED_MIB// /}
GPU_TOTAL_MIB=${GPU_TOTAL_MIB// /}
printf 'timestamp_epoch_s,used_mib,total_mib\n' > "$GPU_SNAPSHOT_LOG"

record_gpu_snapshot() {
    local now
    now=$(date +%s)
    nvidia-smi --id=0 \
        --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits |
        awk -F',' -v now="$now" '{
            gsub(/ /, "", $1); gsub(/ /, "", $2);
            print now "," $1 "," $2
        }' >> "$GPU_SNAPSHOT_LOG"
}

monitor_gpu_memory() {
    while true; do
        record_gpu_snapshot
        sleep 1
    done
}

record_gpu_snapshot
monitor_gpu_memory &
GPU_MONITOR_PID=$!
stop_gpu_monitor() {
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap stop_gpu_monitor EXIT

cat > "$RUN_DIR/train.jsonl" <<'JSONL'
{"text":"I am delighted today.","label":"joy"}
{"text":"This result made me smile.","label":"joy"}
{"text":"What a wonderful surprise.","label":"joy"}
{"text":"I feel cheerful and hopeful.","label":"joy"}
{"text":"I am disappointed and gloomy.","label":"sadness"}
{"text":"This news made me feel low.","label":"sadness"}
{"text":"I cannot stop feeling sorrow.","label":"sadness"}
{"text":"The day has been bleak.","label":"sadness"}
JSONL

timeout --signal=TERM 35m .venv_gpu/bin/python - <<'PY'
import math
import os
import time
from pathlib import Path

from agent.nodes.evaluate import _build_gguf_for_eval
from data.eval_set import EvalSet
from eval.harness import run_eval
from training.slm_helpers import infer, infer_batch, train


model_id = "Qwen/Qwen3.5-0.8B"
run_dir = Path(os.environ["RUN_DIR"])
phase_limits_s = {
    "train": 15 * 60,
    "single_infer": 6 * 60,
    "batch_infer": 6 * 60,
    "q4_build": 15 * 60,
    "deployment_eval": 8 * 60,
}
phase_times_s = {}
smoke_started = time.perf_counter()


def measure(name, operation):
    started = time.perf_counter()
    value = operation()
    elapsed = time.perf_counter() - started
    phase_times_s[name] = elapsed
    assert elapsed <= phase_limits_s[name], (
        name,
        elapsed,
        phase_limits_s[name],
    )
    return value


trained = measure(
    "train",
    lambda: train(
        dataset_path=str(run_dir / "train.jsonl"),
        base_model=model_id,
        nr_epochs=1,
        learning_rate=2e-4,
        batch_size=1,
        lora_rank=4,
        output_dir=str(run_dir / "train_output"),
        task_type="classification",
    ),
)
assert Path(trained.weights_ref).is_dir(), trained
assert any(
    child.is_file() and child.stat().st_size > 0
    for child in Path(trained.weights_ref).rglob("*")
), trained.weights_ref

prediction = measure(
    "single_infer",
    lambda: infer(
        "Classify the emotion: I feel wonderful.",
        trained.weights_ref,
        model_id,
        max_new_tokens=8,
    ),
)
assert isinstance(prediction, str) and prediction.strip(), type(prediction)

batch_prompts = [
    "Emotion: joyful.",
    "Classify the emotion in this sentence: I feel deeply sad today.",
    "Choose joy or sadness: The unexpectedly good news made everyone smile.",
    "Label this much longer account as joy or sadness: After a difficult week, "
    "the successful result left me relieved, cheerful, and hopeful about tomorrow.",
]
batched_predictions = measure(
    "batch_infer",
    lambda: infer_batch(
        batch_prompts,
        trained.weights_ref,
        model_id,
        max_new_tokens=8,
        task_type="classification",
    ),
)
assert len(batched_predictions) == len(batch_prompts) > 1, batched_predictions
assert all(
    isinstance(item, str) and item.strip()
    for item in batched_predictions
), batched_predictions

try:
    gguf_path = measure(
        "q4_build",
        lambda: _build_gguf_for_eval(
            trained.weights_ref,
            model_id,
            "Q4_K_M",
            f"{model_id} [Q4_K_M]",
        ),
    )
except Exception as exc:
    raise RuntimeError(
        "Qwen3.5 merge→GGUF Q4_K_M failed. Verify FastVisionModel merge support, "
        "convert_hf_to_gguf, llama-quantize, and llama-cpp-python."
    ) from exc
assert Path(gguf_path).is_file() and Path(gguf_path).stat().st_size > 0, gguf_path

eval_set = EvalSet(
    pos=[{"text": "I am very happy.", "label": "joy"}],
    neg=[{"text": "I feel deeply sad.", "label": "sadness"}],
    boundary=[{"text": "I am smiling despite a hard day.", "label": "joy"}],
    task_type="classification",
)
result = measure(
    "deployment_eval",
    lambda: run_eval(
        eval_set,
        trained.weights_ref,
        model_id,
        task_type="classification",
        quant="Q4_K_M",
        gguf_path=gguf_path,
    ),
)
assert math.isfinite(result.f1) and 0.0 <= result.f1 <= 1.0, result
total_s = time.perf_counter() - smoke_started
assert total_s <= 35 * 60, total_s
print(
    "PASS qwen35 manual readiness:",
    {
        "weights_ref": trained.weights_ref,
        "prediction": prediction,
        "batch_size": len(batch_prompts),
        "prompt_lengths": [len(prompt) for prompt in batch_prompts],
        "batched_predictions": batched_predictions,
        "gguf_path": gguf_path,
        "deployment_f1": result.f1,
        "phase_times_s": {
            name: round(elapsed, 3)
            for name, elapsed in phase_times_s.items()
        },
        "total_s": round(total_s, 3),
    },
)
PY

record_gpu_snapshot
stop_gpu_monitor
trap - EXIT
PEAK_USED_MIB=$(
    awk -F',' '
        NR > 1 {
            samples += 1
            if (($2 + 0) > peak) peak = $2 + 0
        }
        END {
            if (samples < 1) exit 1
            print peak
        }
    ' "$GPU_SNAPSHOT_LOG"
)
IFS=',' read -r POST_USED_MIB POST_TOTAL_MIB <<< "$(
    nvidia-smi --id=0 \
        --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits
)"
POST_USED_MIB=${POST_USED_MIB// /}
POST_TOTAL_MIB=${POST_TOTAL_MIB// /}
POST_DELTA_MIB=$((POST_USED_MIB - BASELINE_USED_MIB))
if [ "$POST_DELTA_MIB" -lt 0 ]; then
    POST_DELTA_MIB=0
fi
if [ "$PEAK_USED_MIB" -gt "$MAX_PEAK_USED_MIB" ]; then
    echo "FAIL: peak GPU use ${PEAK_USED_MIB} MiB exceeded ${MAX_PEAK_USED_MIB} MiB"
    exit 1
fi
if [ "$POST_DELTA_MIB" -gt "$MAX_POST_WORKER_DELTA_MIB" ]; then
    echo "FAIL: disposable workers left ${POST_DELTA_MIB} MiB above baseline"
    exit 1
fi
echo "PASS qwen35 VRAM criteria: peak_gpu_used_mib=$PEAK_USED_MIB/$GPU_TOTAL_MIB post_worker_delta_mib=$POST_DELTA_MIB baseline_mib=$BASELINE_USED_MIB snapshots=$GPU_SNAPSHOT_LOG"
