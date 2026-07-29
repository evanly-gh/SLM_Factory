#!/bin/bash
#SBATCH --job-name=quant-math-base
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/gpu_setup/quant-math-base-%j.out
#
# Precision sweep on the Qwen3.5-4B **BASE** model (no adapter, no fine-tuning) against
# the math run's held-out GSM8K eval set.
#
# WHY THIS RUN EXISTS
#   The NER sweep (job 37810980) found quantization nearly free: bf16 0.8636 vs Q4_K_M
#   0.8628, a 0.09% relative loss. But NER is the friendliest possible case — short,
#   highly-constrained span extraction where a confident argmax rarely flips.
#
#   Multi-step math reasoning is the opposite regime: long chain-of-thought where a single
#   perturbed token derails the whole derivation, and only the final numeric answer is
#   scored. If quantization ever costs real accuracy, it should show up HERE.
#
#   Using the BASE model (not the fine-tuned winner) isolates the effect of precision from
#   the effect of LoRA sharpening. A fine-tune makes predictions peaked, which is part of
#   why quantization looked free on NER; the base model has no such help.
#
# WHAT TO LOOK FOR
#   - A materially larger bf16 → Q4_K_M gap than NER's 0.0008. That would prove the quant
#     choice must be task-dependent rather than always-Q4_K_M.
#   - If Q4_K_M again ties bf16, quantization really is cheap across both task families.
set -euo pipefail

PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
mkdir -p logs/gpu_setup

# Sourcing the venv supplies the llama.cpp tool PATH *and* /sw/gcc/12.3.0/lib64, which
# libllama.so needs for GLIBCXX_3.4.29 (B160). Direct .venv_gpu/bin/python misses both.
source .venv_gpu/bin/activate

# libllama.so also links libcudart.so.12; the venv's nvidia-* wheels are version-matched
# to torch and always present, unlike Lmod on an arbitrary compute node.
NV_LIBS=$(find "$PROJ/.venv_gpu/lib/python3.11/site-packages/nvidia" \
    -maxdepth 3 -name 'lib' -type d 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES=0
export SLM_CUDA_ISOLATION=1
export SLM_QUANT_EVAL=1
# Converting the 8.7GB BASE snapshot straight out of the shared HF cache on Lustre
# exceeds the 600s default (job 37816978 timed out mid-convert). The merged-checkpoint
# path in the NER sweep was local and fast; this one is not.
export SLM_QUANT_TIMEOUT_S=3600
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
# Import-time placeholders only — no orchestrator/API call on this path.
export ANTHROPIC_API_KEY=quant-compare-no-call
export EXA_API_KEY=quant-compare-no-call

RUN=slm-math-l40s-37576194
EVAL_SET="$PROJ/logs/runs/$RUN/artifacts/eval_set.json"
OUT="$PROJ/artifacts/quant_math_base_${SLURM_JOB_ID:-manual}"
REPORT="$PROJ/logs/quant_eval/qwen35-4b-base-gsm8k"

test -f "$EVAL_SET" || { echo "FAIL: eval set missing: $EVAL_SET"; exit 2; }
command -v llama-quantize >/dev/null || { echo "FAIL: llama-quantize not on PATH"; exit 2; }
python -c 'import llama_cpp; print("llama_cpp", llama_cpp.__version__)' \
    || { echo "FAIL: llama_cpp import failed (CUDA runtime not resolvable)"; exit 2; }

nvidia-smi -L || true
echo "=== model:    Qwen/Qwen3.5-4B (BASE, no adapter)"
echo "=== eval set: $EVAL_SET (800 held-out GSM8K examples)"
echo "=== report:   $REPORT"
echo "=== NER reference: bf16 0.8636 / Q8_0 0.8627 / Q4_K_M 0.8628 (fine-tuned, span-F1)"

python hardware_eval/quant_accuracy_eval.py \
    --checkpoint Qwen/Qwen3.5-4B \
    --eval-set "$EVAL_SET" \
    --quants Q8_0,Q4_K_M \
    --include-bf16 \
    --out "$OUT" \
    --report-dir "$REPORT" \
    --label "Qwen3.5-4B BASE (no adapter) — GSM8K math reasoning, 800 examples"

echo "=== done. Chart + results.json in $REPORT"
