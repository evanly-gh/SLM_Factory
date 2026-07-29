#!/bin/bash
#SBATCH --job-name=manual-qwen36-tp4-coloc
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:30:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/manual-qwen36-tp4-coloc-%j.out
#
# MANUAL / UNSUBMITTED co-location smoke. A human must inspect and submit it.
# No cloud keys or calls: only an exact localhost Qwen3.6 judge is contacted.
# Inert import-time placeholders below are not credentials and are never used.
#
# PASS CRITERIA:
#   1. Qwen/Qwen3.6-35B-A3B FP8 starts at TP=4 and long-run-safe 0.45 GPU util.
#   2. Exact served-model preflight and one strict local judge request pass.
#   3. Shared GPU 0 completes Qwen3.5-4B text-only LoRA train and real batched infer.
#   4. A generation eval worker keeps its model resident while the localhost judge runs.
#   5. One batch exercises APPS 4096/1024 batch pressure beside the TP4 server.
#   6. Disposable workers merge a non-empty Q4_K_M GGUF and deployment-evaluate it.
#   7. Every worker exits without CUDA OOM; adapter/GGUF artifacts are non-empty.
#   8. nvidia-smi snapshots preserve per-GPU peaks; Slurm bounds total time to 90m.
# TIMING: server readiness <=20m and the post-readiness pipeline <=60m (<=80m total,
# leaving >=10m of the 90m allocation for module startup and cleanup).
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
test -x .venv_vllm/bin/vllm
test -x .venv_gpu/bin/python
command -v nvidia-smi >/dev/null 2>&1

LLAMA_CPP_DIR=/mmfs1/gscratch/intelligentsystems/evanly/llama.cpp
export PATH="$LLAMA_CPP_DIR:$LLAMA_CPP_DIR/build/bin:$PATH"
for required in llama-quantize; do
    if ! command -v "$required" >/dev/null 2>&1; then
        echo "FAIL: $required is required for Q4 deployment eval"
        exit 2
    fi
done
CONVERT_TOOL=""
for candidate in convert_hf_to_gguf convert_hf_to_gguf.py convert-hf-to-gguf.py; do
    if command -v "$candidate" >/dev/null 2>&1; then
        CONVERT_TOOL=$(command -v "$candidate")
        break
    fi
done
if [ -z "$CONVERT_TOOL" ]; then
    echo "FAIL: convert_hf_to_gguf is required for Q4 deployment eval"
    exit 2
fi
if ! .venv_gpu/bin/python -c 'import llama_cpp' >/dev/null 2>&1; then
    echo "FAIL: CUDA-enabled llama-cpp-python is required"
    exit 2
fi

MODEL=Qwen/Qwen3.6-35B-A3B
TRAIN_MODEL=Qwen/Qwen3.5-4B
PORT=$((26000 + (${SLURM_JOB_ID:-0} % 12000)))
RUN_DIR="$PROJ/artifacts/manual_qwen36_tp4_colocation_${SLURM_JOB_ID:-manual}"
if [ -e "$RUN_DIR" ]; then
    echo "FAIL: refusing to reuse smoke artifacts at $RUN_DIR"
    exit 2
fi
mkdir -p "$RUN_DIR"
export RUN_DIR MODEL TRAIN_MODEL
VLLM_LOG="$RUN_DIR/vllm.log"
GPU_SNAPSHOT_LOG="$RUN_DIR/gpu-memory-snapshots.csv"
export SLM_RUN_DIR="$RUN_DIR"
export SLM_COST_EVENT_PATH="$RUN_DIR/cost-events.jsonl"
export SLM_TIMING_EVENT_PATH="$RUN_DIR/timing-events.jsonl"
export SLM_OBSERVABILITY_REQUIRED=1
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
export SLM_CUDA_ISOLATION=1
export SLM_EARLY_STOPPING=0
export SLM_QUANT_EVAL=1
export SLM_MAX_SEQ_LENGTH=4096
export SLM_EVAL_BATCH_SIZE=4
export SLM_GGUF_GPU_LAYERS=-1
export ANTHROPIC_API_KEY=manual-local-only-no-call
export EXA_API_KEY=manual-local-only-no-call
unset OPENAI_API_KEY DEEPSEEK_API_KEY

printf 'timestamp_epoch_s,gpu_index,gpu_uuid,used_mib,total_mib\n' > "$GPU_SNAPSHOT_LOG"
record_gpu_snapshot() {
    local now
    now=$(date +%s)
    nvidia-smi \
        --query-gpu=index,uuid,memory.used,memory.total \
        --format=csv,noheader,nounits |
        awk -F',' -v now="$now" '{
            for (field = 1; field <= 4; field += 1) {
                gsub(/^ +| +$/, "", $field)
            }
            print now "," $1 "," $2 "," $3 "," $4
        }' >> "$GPU_SNAPSHOT_LOG"
}
monitor_gpu_memory() {
    while true; do
        record_gpu_snapshot
        sleep 2
    done
}
record_gpu_snapshot
monitor_gpu_memory &
GPU_MONITOR_PID=$!
VLLM_PID=""
cleanup() {
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    if [ -n "$VLLM_PID" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1) || true
if [ -n "$CUDA_MOD" ] && module load "$CUDA_MOD"; then
    :
elif module load cuda/12.8.1; then
    CUDA_MOD=cuda/12.8.1
else
    echo "FAIL: no usable CUDA 12.8 module found"
    exit 2
fi
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$PROJ/.venv_vllm/bin:$PATH"

timeout --signal=TERM 85m env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    .venv_vllm/bin/vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --quantization fp8 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --gdn-prefill-backend triton \
    --language-model-only \
    --reasoning-parser qwen3 \
    --served-model-name "$MODEL" \
    --enforce-eager \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

export SLM_SYNTH_ENDPOINT="http://127.0.0.1:${PORT}/v1"
export SLM_SYNTH_MODEL="$MODEL"
export SLM_SYNTH_API_KEY=EMPTY
export SLM_JUDGE_ENDPOINT="$SLM_SYNTH_ENDPOINT"
export SLM_JUDGE_MODEL="$MODEL"
export SLM_JUDGE_API_KEY=EMPTY
export SLM_JUDGE_CACHE_PATH="$RUN_DIR/judge-cache.jsonl"

READY=0
READINESS_DEADLINE=$((SECONDS + 20 * 60))
while [ "$SECONDS" -lt "$READINESS_DEADLINE" ]; do
    if curl -fsS --noproxy '*' --max-time 3 \
        "$SLM_SYNTH_ENDPOINT/models" > "$RUN_DIR/models.json"; then
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "FAIL: TP4 vLLM exited before readiness; inspect $VLLM_LOG"
        exit 1
    fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "FAIL: TP4 vLLM did not become ready within 20 minutes"
    exit 1
fi

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

# Every model-owning operation below inherits GPU 0 and SLM_CUDA_ISOLATION=1.
export CUDA_VISIBLE_DEVICES=0
timeout --signal=TERM 60m .venv_gpu/bin/python - <<'PY'
import json
import math
import os
import time
from pathlib import Path

from agent.nodes.evaluate import _build_gguf_for_eval
from data.eval_set import EvalSet
from eval.harness import run_eval
from eval.judge_client import LocalJudgeClient
from eval.scorers.generation import build_code_prompt
from training.slm_helpers import infer_batch, train


run_dir = Path(os.environ["RUN_DIR"])
judge_model = os.environ["MODEL"]
train_model = os.environ["TRAIN_MODEL"]
endpoint = os.environ["SLM_JUDGE_ENDPOINT"]
phase_times_s = {}
smoke_started = time.perf_counter()


def measure(name, operation):
    started = time.perf_counter()
    value = operation()
    phase_times_s[name] = time.perf_counter() - started
    return value


def run_pipeline():
    judge = LocalJudgeClient(
        endpoint=endpoint,
        model=judge_model,
        api_key="EMPTY",
        concurrency=1,
        request_timeout=180.0,
        cache_path=run_dir / "judge-cache.jsonl",
        cost_event_path=run_dir / "cost-events.jsonl",
        timing_event_path=run_dir / "timing-events.jsonl",
    )
    measure("judge_preflight", lambda: judge.preflight())
    judge_scores = measure(
        "judge_request",
        lambda: judge.score_many(
            [("What is two plus two?", "4", "4")]
        ),
    )
    assert (
        len(judge_scores) == 1
        and math.isfinite(judge_scores[0])
        and 0.0 <= judge_scores[0] <= 1.0
    ), judge_scores

    trained = measure(
        "train",
        lambda: train(
            dataset_path=str(run_dir / "train.jsonl"),
            base_model=train_model,
            nr_epochs=1,
            learning_rate=2e-4,
            batch_size=1,
            lora_rank=4,
            output_dir=str(run_dir / "train_output"),
            task_type="classification",
        ),
    )
    weights = Path(trained.weights_ref)
    assert weights.is_dir(), trained
    assert any(
        child.is_file() and child.stat().st_size > 0
        for child in weights.rglob("*")
    ), weights

    batch_prompts = [
        "Emotion: joyful.",
        "Choose joy or sadness: I feel very sad today.",
        "Classify this longer message: The good result made me smile and feel hopeful.",
        "Label the emotion after reading this deliberately longer account of a "
        "difficult day that ended with wonderful, relieving news.",
    ]
    predictions = measure(
        "batch_infer",
        lambda: infer_batch(
            batch_prompts,
            trained.weights_ref,
            train_model,
            max_new_tokens=8,
            task_type="classification",
        ),
    )
    assert len(predictions) == len(batch_prompts) > 1, predictions
    assert all(
        isinstance(prediction, str) and prediction.strip()
        for prediction in predictions
    ), predictions

    # APPS 4096/1024 batch pressure: long call-based prompts use the production
    # context/output reserve and all four rows enter one initial eval batch.
    apps_prompts = [
        build_code_prompt(
            {
                "text": (
                    f"Implement solve_variant_{variant}. "
                    + (
                        "Use integer arithmetic and preserve input order. "
                        * 320
                    )
                ),
                "starter_code": (
                    f"def solve_variant_{variant}(values):\n"
                    "    pass"
                ),
                "entry_point": f"solve_variant_{variant}",
                "execution_mode": "call_based",
                "input_output": {
                    "fn_name": f"solve_variant_{variant}",
                    "inputs": ["[[1, 2, 3]]"],
                    "outputs": ["6"],
                },
            }
        )
        for variant in range(4)
    ]
    apps_predictions = measure(
        "apps_4096_1024_batch_pressure",
        lambda: infer_batch(
            apps_prompts,
            trained.weights_ref,
            train_model,
            max_new_tokens=1024,
            task_type="code_generation",
        ),
    )
    assert len(apps_predictions) == len(apps_prompts) == 4
    assert all(isinstance(prediction, str) for prediction in apps_predictions)

    # The generation eval worker keeps its model resident while the strict local
    # judge runs against the co-located TP4 server in the same run_eval call.
    generation_eval_set = EvalSet(
        pos=[{
            "text": "Summarize: Alex confirmed the release is ready.",
            "answer": "Alex confirmed that the release is ready.",
            "label": "generation",
        }],
        neg=[],
        boundary=[],
        task_type="generation",
    )
    generation_result = measure(
        "generation_eval_model_held_during_judge",
        lambda: run_eval(
            generation_eval_set,
            trained.weights_ref,
            train_model,
            task_type="generation",
        ),
    )
    assert (
        math.isfinite(generation_result.f1)
        and 0.0 <= generation_result.f1 <= 1.0
    ), generation_result

    gguf_path = measure(
        "merge_q4",
        lambda: _build_gguf_for_eval(
            trained.weights_ref,
            train_model,
            "Q4_K_M",
            f"{train_model} [Q4_K_M]",
        ),
    )
    gguf = Path(gguf_path)
    assert gguf.is_file() and gguf.stat().st_size > 0, gguf

    eval_set = EvalSet(
        pos=[{"text": "I am very happy.", "label": "joy"}],
        neg=[{"text": "I feel deeply sad.", "label": "sadness"}],
        boundary=[
            {
                "text": "I am smiling despite a hard day.",
                "label": "joy",
            }
        ],
        task_type="classification",
    )
    result = measure(
        "q4_deployment_eval",
        lambda: run_eval(
            eval_set,
            trained.weights_ref,
            train_model,
            task_type="classification",
            quant="Q4_K_M",
            gguf_path=gguf_path,
        ),
    )
    assert math.isfinite(result.f1) and 0.0 <= result.f1 <= 1.0, result
    total_s = time.perf_counter() - smoke_started
    assert total_s <= 60 * 60, total_s
    return {
        "judge_model": judge_model,
        "judge_score": judge_scores[0],
        "train_model": train_model,
        "weights_ref": trained.weights_ref,
        "batch_size": len(batch_prompts),
        "predictions": predictions,
        "apps_batch_size": len(apps_prompts),
        "apps_prediction_lengths": [
            len(prediction) for prediction in apps_predictions
        ],
        "generation_judge_score": generation_result.f1,
        "gguf_path": gguf_path,
        "deployment_f1": result.f1,
        "phase_times_s": {
            name: round(value, 3)
            for name, value in phase_times_s.items()
        },
        "total_s": round(total_s, 3),
    }


try:
    summary = run_pipeline()
except BaseException as exc:
    detail = f"{type(exc).__name__}: {exc}"
    if "OutOfMemoryError" in detail or "CUDA out of memory" in detail:
        raise RuntimeError(f"FAIL: co-location CUDA OOM: {detail}") from exc
    raise
print("PASS qwen36 TP4 co-location smoke:", json.dumps(summary, sort_keys=True))
PY

if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "FAIL: TP4 vLLM exited during the shared-GPU pipeline"
    exit 1
fi
if grep -Eiq 'OutOfMemoryError|CUDA out of memory|NCCL.*out of memory' "$VLLM_LOG"; then
    echo "FAIL: vLLM log contains an OOM marker"
    exit 1
fi
record_gpu_snapshot
kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true
PEAK_SUMMARY=$(
    awk -F',' '
        NR > 1 {
            gpu = $2
            if (($4 + 0) > peak[gpu]) peak[gpu] = $4 + 0
            total[gpu] = $5 + 0
            samples += 1
        }
        END {
            if (samples < 1) exit 1
            for (gpu in peak) {
                printf "gpu%s=%dMiB/%dMiB ", gpu, peak[gpu], total[gpu]
            }
        }
    ' "$GPU_SNAPSHOT_LOG"
)
echo "PASS nvidia-smi peak snapshots: $PEAK_SUMMARY log=$GPU_SNAPSHOT_LOG"
