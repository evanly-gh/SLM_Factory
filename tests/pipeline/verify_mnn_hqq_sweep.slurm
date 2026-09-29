#!/bin/bash
#SBATCH --job-name=slm-mnn-hqq-sweep
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/%x-%j.out
#
# CAN MNN'S 4-BIT EXPORT MATCH llama.cpp's Q4_K_M? (2026-09-28)
#
# The backend matrix (40305686) showed MNN's runtime is sound — its FP16 and 8-bit exports score
# where llama.cpp's Q4_K_M does — and that its default 4-bit export (min/max rounding, 64-weight
# blocks) loses 0.12-0.25 macro_f1 on the Qwen models zero-shot. This measures the two recipe
# changes MNN's own docs point at, against the same weights and the same 1,000 rows:
#
#   RECIPE=llama_cpp     Q4_K_M through llama.cpp — the target
#   RECIPE=mnn_default   MNN 4-bit, block 64                     (what the loop ships today)
#   RECIPE=mnn_hqq       MNN 4-bit, block 64, --hqq              (MNN's recommended setting)
#   RECIPE=mnn_hqq_b32   MNN 4-bit, block 32, --hqq              (finer groups, ~11% larger)
#
# One job per recipe, submitted as
#   sbatch --job-name=slm-mnn-hqq-sweep-<recipe> --export=ALL,RECIPE=<recipe> verify_mnn_hqq_sweep.slurm
#
# WEIGHTS, 8 per recipe: four pool models, each untrained (zero-shot, where the gap is largest) and
# with the best clinc150 adapter from MNN loop run 40298521 (what the loop actually ships).
# ROWS: that run's frozen 1,000-row eval set, identical across recipes.
# ONE MNN CUDA LOAD PER PROCESS: MNN's CUDA runtime does not survive a second model load, so every
# (weights, recipe) pair is its own quant_accuracy_eval invocation and scoring runs in a worker.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache

source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1)
CUDA_MOD="${CUDA_MOD:-cuda/12.8.1}"
module load "$CUDA_MOD"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

source .venv_gpu/bin/activate
export SLM_QUANT_EVAL=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export SLM_SEED="${SLM_SEED:-3407}"
# clinc150's prompt carries all 151 intent names; every clinc150 run scored at 4096.
export SLM_MAX_SEQ_LENGTH=4096
# HQQ runs inside MNNConvert and its cost on a 4B export is unmeasured; the default 30-minute
# floor would turn a slow export into a missing cell.
export SLM_MNN_EXPORT_TIMEOUT_FLOOR_S=5400

RECIPE="${RECIPE:?set RECIPE to llama_cpp, mnn_default, mnn_hqq or mnn_hqq_b32}"
# MNN arms score in a disposable CUDA worker: `validate_and_record` has already loaded the artifact
# on CUDA in this process, and `slm_helpers._note_cuda_load` refuses a second load (job 40827153).
export SLM_CUDA_ISOLATION=1
case "$RECIPE" in
    llama_cpp)   BACKEND=llama_cpp; export SLM_CUDA_ISOLATION=0 ;;
    mnn_default) BACKEND=mnn; export SLM_MNN_QUANT_BLOCK=64; export SLM_MNN_HQQ=0 ;;
    mnn_hqq)     BACKEND=mnn; export SLM_MNN_QUANT_BLOCK=64; export SLM_MNN_HQQ=1 ;;
    mnn_hqq_b32) BACKEND=mnn; export SLM_MNN_QUANT_BLOCK=32; export SLM_MNN_HQQ=1 ;;
    *) echo "ERROR: unknown RECIPE=$RECIPE"; exit 2 ;;
esac

RUN=logs/runs/slm-clinc150-mnn-verify-40298521
EVAL_SET="$RUN/artifacts/eval_set.json"
TRAINED="$RUN/artifacts/training"
STAMP="${SLURM_JOB_ID:-manual-$$}"
OUT=/mmfs1/gscratch/intelligentsystems/evanly/tmp/mnn-hqq-sweep-$STAMP
REPORTS=logs/quant_eval/mnn-hqq-sweep-$STAMP-$RECIPE

# base model | adapter (best fine-tuned iteration of that tier in 40298521)
CELLS=(
  "HuggingFaceTB/SmolLM2-360M-Instruct|$TRAINED/HuggingFaceTB_SmolLM2-360M-Instruct__Q4_K_M/iter2-d2/final_checkpoint"
  "Qwen/Qwen3.5-0.8B|$TRAINED/Qwen_Qwen3.5-0.8B__Q4_K_M/iter2-d3/final_checkpoint"
  "Qwen/Qwen3-1.7B|$TRAINED/Qwen_Qwen3-1.7B__Q4_K_M/iter2-d4/final_checkpoint"
  "Qwen/Qwen3-4B-Instruct-2507|$TRAINED/Qwen_Qwen3-4B-Instruct-2507__Q4_K_M/iter1-d4/final_checkpoint"
)

test -f "$EVAL_SET" || { echo "ERROR: eval set not found: $EVAL_SET"; exit 2; }
for cell in "${CELLS[@]}"; do
  test -d "${cell#*|}" || { echo "ERROR: adapter not found: ${cell#*|}"; exit 2; }
done
mkdir -p "$OUT" "$REPORTS"
echo "=== MNN HQQ sweep: RECIPE=$RECIPE backend=$BACKEND block=${SLM_MNN_QUANT_BLOCK:-n/a} hqq=${SLM_MNN_HQQ:-n/a} ==="
nvidia-smi -L || true

# Optional narrowing for re-running a subset: ONLY_MODEL is a substring of the base id, KINDS a
# space-separated subset of "zeroshot finetuned".
ONLY_MODEL="${ONLY_MODEL:-}"
KINDS="${KINDS:-zeroshot finetuned}"

FAILED=0
for cell in "${CELLS[@]}"; do
  BASE="${cell%%|*}"
  ADAPTER="${cell#*|}"
  SAFE="${BASE//\//_}"
  if [ -n "$ONLY_MODEL" ] && [[ "$BASE" != *"$ONLY_MODEL"* ]]; then continue; fi
  for KIND in $KINDS; do
    echo ""
    echo "============================================================"
    echo "  $RECIPE  $BASE  $KIND"
    echo "============================================================"
    if [ "$KIND" = zeroshot ]; then
      SOURCE=(--checkpoint "$BASE")
    else
      SOURCE=(--adapter "$ADAPTER" --base "$BASE")
    fi
    python hardware_eval/quant_accuracy_eval.py \
      --backend "$BACKEND" "${SOURCE[@]}" \
      --eval-set "$EVAL_SET" \
      --quants Q4_K_M \
      --out "$OUT/$SAFE-$KIND" \
      --report-dir "$REPORTS/$SAFE-$KIND" \
      --label "$RECIPE $BASE $KIND" || FAILED=$((FAILED + 1))
  done
done

echo ""
echo "=== $RECIPE complete ($FAILED invocation(s) failed); reports in $REPORTS ==="
REPORTS="$REPORTS" python - <<'PY'
import glob, json, os
print(f"{'cell':<48}{'MB':>8}{'macro_f1':>10}{'n':>6}")
for path in sorted(glob.glob(os.path.join(os.environ["REPORTS"], "*", "results.json"))):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for entry in payload["results"]:
        mb, f1 = entry["on_disk_mb"], entry["f1"]
        print(f"{os.path.basename(os.path.dirname(path)):<48}"
              f"{(f'{mb:.1f}' if isinstance(mb, (int, float)) else 'n/a'):>8}"
              f"{(f'{f1:.4f}' if isinstance(f1, (int, float)) else 'FAILED'):>10}"
              f"{payload['n_examples']:>6}")
PY
