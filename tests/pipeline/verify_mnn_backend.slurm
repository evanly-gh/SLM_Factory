#!/bin/bash
#SBATCH --job-name=slm-mnn-backend-verify
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-mnn-backend-verify-%j.out
#
# BACKEND A/B: the same fine-tuned adapter, quantized to 4 bits by BOTH toolchains and scored on
# the SAME eval rows. This is the acceptance test for the MNN quantization backend, and it is an
# A/B rather than a single MNN number on purpose: "the MNN artifact scores 0.79" means nothing on
# its own, because a 360M model fine-tuned on 151 intents could plausibly score anywhere from 0.3
# to 0.85. Against llama.cpp's number for the same weights it means everything — a match says the
# export, the quantization, the tokenizer and the prompt rendering are all correct, and a collapse
# localises the fault to whatever MNN does differently.
#
# WHAT IS HELD FIXED
#   * the adapter: iteration 30 of clinc150 run 40175895, the best SmolLM2-360M checkpoint that run
#     produced (macro_f1 0.8047 through llama.cpp/GGUF, measured on all 1,000 eval rows).
#   * the eval rows: that run's own frozen eval_set.json, truncated to the same first N for both
#     arms by `--max-rows`.
#   * the prompt rendering: `_serving_prompt_prefix`, shared by both backends by construction
#     (MNN is configured with `use_template: false` and handed the rendered string).
# WHAT VARIES: only the quantization toolchain and the decoding engine.
#
# WHY A GPU IS REQUESTED FOR A CPU MEASUREMENT: the LoRA merge runs through Unsloth, which wants
# CUDA, and the GGUF arm offloads its layers to the GPU for speed (identical predictions, ~10-50x
# faster — see training/slm_helpers.infer_batch_gguf). The MNN arm decodes on CPU, which is what
# the phone does anyway.
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
# In-process rather than through a disposable CUDA worker: this script is a measurement, not a
# multi-hour run, and a linear log is what makes the two arms comparable by eye.
export SLM_CUDA_ISOLATION=0
export SLM_QUANT_EVAL=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export SLM_SEED="${SLM_SEED:-3407}"
# THE CONTEXT THE REFERENCE NUMBER WAS MEASURED AT, and it has to be set explicitly.
# `task_max_seq_length` gives this env var precedence over the task spec, `_l40s_task_body.sh`
# exports 4096, and every clinc150 run — including 40175895, whose 0.8047 this is compared against
# — therefore ran at 4096 rather than the spec's 1024. Without this line the first attempt of this
# job died at `Rendered GGUF prompt index 0 contains 1410 tokens, exceeding input budget 974`: a
# CLINC150 prompt carries all 151 intent names, so 1024 does not fit one, let alone one with the
# output reserve subtracted. Both arms must see the same number or the comparison is meaningless.
export SLM_MAX_SEQ_LENGTH=4096

RUN=logs/runs/slm-clinc150-ablation-scarce-synth-l40s-40175895
ADAPTER="$RUN/artifacts/training/HuggingFaceTB_SmolLM2-360M-Instruct__Q4_K_M/iter30-d4/final_checkpoint"
EVAL_SET="$RUN/artifacts/eval_set.json"
BASE=HuggingFaceTB/SmolLM2-360M-Instruct
ROWS="${MNN_VERIFY_ROWS:-150}"
STAMP="${SLURM_JOB_ID:-manual-$$}"
OUT=/mmfs1/gscratch/intelligentsystems/evanly/tmp/mnn-verify-$STAMP
REPORTS=logs/quant_eval

test -d "$ADAPTER" || { echo "ERROR: adapter not found: $ADAPTER"; exit 2; }
test -f "$EVAL_SET" || { echo "ERROR: eval set not found: $EVAL_SET"; exit 2; }
mkdir -p "$OUT" "$REPORTS"

echo "=== MNN backend verification ==="
echo "  adapter   $ADAPTER"
echo "  base      $BASE"
echo "  eval set  $EVAL_SET (first $ROWS rows)"
echo "  reference macro_f1 = 0.8047 (llama.cpp/GGUF Q4_K_M, all 1000 rows, run 40175895 iter 30)"
nvidia-smi -L || true

for BACKEND in llama_cpp mnn; do
  echo ""
  echo "============================================================"
  echo "  BACKEND: $BACKEND"
  echo "============================================================"
  python hardware_eval/quant_accuracy_eval.py \
    --backend "$BACKEND" \
    --adapter "$ADAPTER" \
    --base "$BASE" \
    --eval-set "$EVAL_SET" \
    --quants Q4_K_M \
    --max-rows "$ROWS" \
    --out "$OUT/$BACKEND" \
    --report-dir "$REPORTS/mnn-verify-$STAMP-$BACKEND" \
    --label "clinc150 SmolLM2-360M iter30 — $BACKEND"
done

echo ""
echo "=== both arms complete; reports in $REPORTS/mnn-verify-$STAMP-* ==="
# Scoped to THIS job's stamp, not `mnn-verify-*`: the first attempt of this job left a failed
# report behind, and an unscoped glob printed it in the summary as a third row belonging to a run
# that no longer existed.
SUMMARY_GLOB="logs/quant_eval/mnn-verify-$STAMP-*/results.json" python - <<'PY'
import glob, json, os
rows = []
for path in sorted(glob.glob(os.environ["SUMMARY_GLOB"])):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for entry in payload["results"]:
        rows.append((payload.get("backend", "?"), entry["variant"], entry["on_disk_mb"],
                     entry["f1"], payload["n_examples"]))
print(f"{'backend':<12}{'variant':<10}{'MB':>8}{'metric':>10}{'n':>6}")
for backend, variant, mb, f1, n in rows:
    mb_text = f"{mb:.1f}" if isinstance(mb, (int, float)) else "n/a"
    f1_text = f"{f1:.4f}" if isinstance(f1, (int, float)) else "FAILED"
    print(f"{backend:<12}{variant:<10}{mb_text:>8}{f1_text:>10}{n:>6}")
PY
