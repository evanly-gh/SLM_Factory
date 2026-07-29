#!/bin/bash
#SBATCH --job-name=quant-acc-cmp
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/gpu_setup/quant-acc-cmp-%j.out
#
# Validate the quantization path end-to-end: take the NER run's WINNING checkpoint
# (Qwen/Qwen3.5-4B + iter46-d4 LoRA adapter, the model the pipeline actually chose) and
# score it at bf16 / Q8_0 / Q4_K_M against the SAME 900-example held-out eval set the run
# used.
#
# WHAT THIS CONFIRMS
#   1. The conversion+quantization toolchain produces loadable, scoreable GGUFs.
#   2. Accuracy degrades monotonically with precision: bf16 >= Q8_0 >= Q4_K_M.
#      If Q4_K_M ties bf16 exactly, the eval is probably NOT actually running the
#      quantized weights (the exact bug the (weights_ref, quant) cache key was added to
#      prevent — see agent/nodes/evaluate.py:_build_or_reuse_gguf).
#   3. The run's reported 0.8628 Q4_K_M figure is reproducible.
#
# Q4_K_M here is rebuilt from scratch into a scratch dir rather than reusing
# artifacts/gguf/Qwen_Qwen3.5-4B/485ca7619192/ — a fresh build is what actually tests the
# converter. Compare the printed Q4_K_M F1 against the run's 0.8628 to confirm.
set -euo pipefail

PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
mkdir -p logs/gpu_setup

# Source the venv rather than calling .venv_gpu/bin/python directly: setup_gpu_env.sh
# appends two RUNTIME-critical lines to its activate script — the llama.cpp tool PATH,
# and /sw/gcc/12.3.0/lib64 on LD_LIBRARY_PATH. llama-cpp-python's libllama.so is built
# with gcc-12 and needs its newer libstdc++ (GLIBCXX_3.4.29); the RHEL8 system one is too
# old and the import dies (B160). This is the same env the pipeline runs under.
source .venv_gpu/bin/activate

# libllama.so also links libcudart.so.12. The nvidia-* wheels vendored in the venv are
# version-matched to torch and present regardless of node config, unlike Lmod — an
# unguarded `module avail | grep` under `set -o pipefail` aborts the script with NO
# output when `module` is undefined on a compute node.
NV_LIBS=$(find "$PROJ/.venv_gpu/lib/python3.11/site-packages/nvidia" \
    -maxdepth 3 -name 'lib' -type d 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES=0
export SLM_CUDA_ISOLATION=1
export SLM_QUANT_EVAL=1
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
# Import-time placeholders only — this script makes no orchestrator/API call.
export ANTHROPIC_API_KEY=quant-compare-no-call
export EXA_API_KEY=quant-compare-no-call

RUN=slm-ner-l40s-37531245
ADAPTER="$PROJ/logs/runs/$RUN/artifacts/training/Qwen_Qwen3.5-4B__Q4_K_M/iter46-d4/final_checkpoint"
EVAL_SET="$PROJ/logs/runs/$RUN/artifacts/eval_set.json"
OUT="$PROJ/artifacts/quant_compare_${SLURM_JOB_ID:-manual}"

test -d "$ADAPTER" || { echo "FAIL: adapter missing: $ADAPTER"; exit 2; }
test -f "$EVAL_SET" || { echo "FAIL: eval set missing: $EVAL_SET"; exit 2; }
command -v llama-quantize >/dev/null || { echo "FAIL: llama-quantize not on PATH"; exit 2; }
.venv_gpu/bin/python -c 'import llama_cpp; print("llama_cpp", llama_cpp.__version__)' \
    || { echo "FAIL: llama_cpp import failed (CUDA runtime not resolvable)"; exit 2; }

nvidia-smi -L || true
echo "=== adapter:  $ADAPTER"
echo "=== eval set: $EVAL_SET (900 held-out BC5CDR examples)"
echo "=== out:      $OUT"
echo "=== reference: the run reported F1=0.8628 for this checkpoint at Q4_K_M"

.venv_gpu/bin/python hardware_eval/quant_accuracy_eval.py \
    --adapter "$ADAPTER" \
    --base Qwen/Qwen3.5-4B \
    --eval-set "$EVAL_SET" \
    --quants Q8_0,Q4_K_M \
    --include-bf16 \
    --out "$OUT"

echo "=== done. Expect bf16 >= Q8_0 >= Q4_K_M; Q4_K_M should land near 0.8628."
