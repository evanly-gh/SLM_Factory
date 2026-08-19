#!/bin/bash
#SBATCH --job-name=slm-teacher-fewshot-probe
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-teacher-fewshot-probe-%j.out
#
# Measures the Qwen3.6-35B teacher ZERO-SHOT vs FEW-SHOT on a task's own frozen eval set.
#
# WHY: every claim about the teacher's fitness to generate synthetic data currently rests on its
# zero-shot score — 0.0999 on BC5CDR NER. Exact-span NER is dominated by output-FORMAT compliance,
# so that number mostly measures whether the teacher guessed our JSON contract and span
# conventions, not whether it can find entities. Few-shot supplies the convention explicitly. If the
# gap is large, "the teacher scores 0.0999 so it cannot generate NER data" was never a valid
# inference, and the synthesis gate needs a different input.
#
# Scores through the SAME scorer the pipeline uses, on the SAME frozen eval set, so numbers are
# directly comparable to the run logs. Demonstrations are drawn from TRAIN only — never from the
# eval set — so few-shot cannot leak the answer.
#
# This is a measurement job, not a pipeline run: it starts the vLLM server, runs the probe, exits.
#
# NOTE: deliberately NO `set -u`. lmod's init script references LD_LIBRARY_PATH unguarded, so
# nounset makes `module load cuda` fail silently — after which `command -v nvcc` finds nothing,
# CUDA_HOME is wrong, and torch-inductor dies 11 minutes into model loading with a confusing
# `PermissionError: [Errno 13] Permission denied: 'nvcc'`. That is exactly what killed job 38550928.
# `_l40s_task_body.sh` does not set -u either, for the same reason.
set -o pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"

export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export TOKENIZERS_PARALLELISM=false
[ -f .env ] && set -a && . ./.env && set +a

# --- vLLM synth server on GPU 0 (same launch contract as _l40s_task_body.sh) ---
source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1); CUDA_MOD="${CUDA_MOD:-cuda/12.8.1}"
module load "$CUDA_MOD"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$PROJ/.venv_vllm/bin:$PATH"
# Fail LOUD and EARLY if the CUDA module did not load. Without this the run gets 11 minutes into
# weight loading before torch-inductor reports a misleading nvcc permission error (job 38550928).
if ! command -v nvcc >/dev/null 2>&1; then
    echo "FATAL: nvcc not on PATH after 'module load $CUDA_MOD' — torch-inductor will fail to"
    echo "       JIT kernels. CUDA_HOME='$CUDA_HOME'. Check that lmod initialised correctly."
    exit 1
fi
echo "=== CUDA ok: $(command -v nvcc), CUDA_HOME=$CUDA_HOME ==="
.venv_vllm/bin/python -c 'import vllm' >/dev/null 2>&1 || bash scripts/setup_vllm_env.sh
.venv_vllm/bin/python -c 'import ninja' >/dev/null 2>&1 || .venv_vllm/bin/python -m pip install ninja 2>/dev/null || true

SYNTH_MODEL="${SLM_SYNTH_MODEL:-Qwen/Qwen3.6-35B-A3B}"
SYNTH_PORT=$(( 20000 + (${SLURM_JOB_ID:-0} % 20000) ))
SYNTH_LOG="$PROJ/logs/gpu_setup/synth-probe-${SLURM_JOB_ID:-manual}.out"
mkdir -p "$(dirname "$SYNTH_LOG")" "$PROJ/logs/probes"

echo "=== vLLM synth ($SYNTH_MODEL) TP=2 → localhost:$SYNTH_PORT → $SYNTH_LOG ==="
CUDA_VISIBLE_DEVICES=0,1 .venv_vllm/bin/vllm serve "$SYNTH_MODEL" \
    --host 127.0.0.1 --port "$SYNTH_PORT" \
    --tensor-parallel-size 2 --quantization fp8 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --max-num-seqs 32 --gdn-prefill-backend triton \
    --language-model-only --reasoning-parser qwen3 --served-model-name "$SYNTH_MODEL" \
    > "$SYNTH_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "stopping vLLM ($VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true' EXIT

export SLM_SYNTH_ENDPOINT="http://127.0.0.1:${SYNTH_PORT}/v1"
export SLM_SYNTH_MODEL="$SYNTH_MODEL"

echo "=== waiting for the endpoint (up to 40 min) ==="
for i in $(seq 1 480); do
    if curl -s -m 5 "http://127.0.0.1:${SYNTH_PORT}/v1/models" | grep -q "$SYNTH_MODEL"; then
        echo "=== endpoint up after $((i*5))s ==="; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "FATAL: vLLM died during startup; tail of $SYNTH_LOG:"; tail -40 "$SYNTH_LOG"; exit 1
    fi
    sleep 5
done

# 200 rows per condition keeps the whole sweep well inside the wall clock while still separating
# scores that differ by more than a couple of points, which is the scale the question is about.
TASKS="${SLM_PROBE_TASKS:-ner_bc5cdr}"
for TASK in $TASKS; do
    echo ""
    echo "############ probing $TASK ############"
    .venv/bin/python scripts/probe_teacher_fewshot.py \
        --task "$TASK" --shots 0 1 3 5 --corrupt-shots 5 --n "${SLM_PROBE_N:-200}"
done

echo "=== probe complete ==="
