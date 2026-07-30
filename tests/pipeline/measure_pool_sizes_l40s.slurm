#!/bin/bash
#SBATCH --job-name=measure-pool
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/gpu_setup/measure-pool-%j.out
#
# Fill out config/measured_metrics.json for every pool model family that still has none.
#
# WHY THIS RUN EXISTS
#   The pool carries NO modelled runtime metrics — that was removed deliberately. What is
#   left is on-disk weight size, which for an unmeasured variant is bytes-per-parameter
#   ARITHMETIC, not a measurement. On the one family that has been measured (Qwen3.5-4B),
#   that arithmetic was 20% LOW at Q4_K_M (predicted 2200 MB, actual 2654.5 MB) — enough to
#   move the variant from tier 2 to tier 3 and change which candidates a hardware budget
#   admits. Every other family is currently gated on the same unverified arithmetic.
#
# WHAT IT DOES
#   For each model x {Q4_K_M, Q8_0}: convert the HF snapshot to GGUF, quantize, and record
#   os.path.getsize of the real file. Also records the bf16 size from the real safetensors.
#   Then, on the host CPU only, measures peak RSS + decode tok/s via llama.cpp's own timing.
#
#   The chip key for the runtime numbers is "host_cpu" — an L40S node's CPU. That is an
#   honest key: it is NOT a phone, and measured_metrics_for() never interpolates across
#   chips, so a phone lookup still correctly reports "unmeasured". The SIZES, by contrast,
#   are chip-independent and are the real payload of this job.
#
# NOTE ON Qwen3.5-2B
#   This model is B113: its snapshot has repeatedly stalled mid-download on compute nodes.
#   It is deliberately sequenced LAST so a stall does not block the other four families,
#   and the loop continues past a failure rather than aborting the job.
set -euo pipefail

PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
mkdir -p logs/gpu_setup

# Sourcing the venv supplies the llama.cpp tool PATH *and* /sw/gcc/12.3.0/lib64, which
# libllama.so needs for GLIBCXX_3.4.29 (B160). Direct .venv_gpu/bin/python misses both.
source .venv_gpu/bin/activate

NV_LIBS=$(find "$PROJ/.venv_gpu/lib/python3.11/site-packages/nvidia" \
    -maxdepth 3 -name 'lib' -type d 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES=0
export SLM_CUDA_ISOLATION=1
export SLM_QUANT_TIMEOUT_S=3600
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
# Import-time placeholders only — no orchestrator/API call on this path.
export ANTHROPIC_API_KEY=measure-pool-no-call
export EXA_API_KEY=measure-pool-no-call

OUT="$PROJ/artifacts/measure_pool_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

command -v llama-quantize >/dev/null || { echo "FAIL: llama-quantize not on PATH"; exit 2; }
python -c 'import llama_cpp; print("llama_cpp", llama_cpp.__version__)' \
    || { echo "FAIL: llama_cpp import failed (CUDA runtime not resolvable)"; exit 2; }

nvidia-smi -L || true

# The driver iterates every pool family, skips ones already recorded, and — importantly —
# continues past a family that fails to download or convert, leaving it UNMEASURED rather
# than recording a guess. Qwen3.5-2B (B113) is the known-risky one.
STATUS=0
python hardware_eval/measure_pool.py --out "$OUT" || STATUS=$?

echo ""
echo "=== exit status $STATUS (1 means at least one variant was left unmeasured)"
echo "=== scratch GGUFs under $OUT (removed per-variant unless --keep-gguf)"
du -sh "$OUT" 2>/dev/null || true
exit 0
