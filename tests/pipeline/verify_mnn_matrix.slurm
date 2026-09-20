#!/bin/bash
#SBATCH --job-name=slm-mnn-matrix
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=20:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-mnn-matrix-%j.out
#
# DOES THE MNN BACKEND WORK FOR EVERY TIER, AT EVERY PRECISION, ON THE GPU?
#
# 9 pool models (tiers 1-5) x {Q4_K_M, Q8_0, FP16} x {cuda, cpu} = 27 exports, each load-validated
# and scored on the same clinc150 rows on both devices. One cell is one independent test of MNN's
# architecture support, because `llmexport.py` has per-model branches and a Qwen3.5 is not a
# SmolLM2 is not a Gemma — and because the loop WALKS this ladder, so a backend that works at tier
# 1 and fails at tier 3 is a backend that dies three hours into a run.
#
# WHY BOTH DEVICES. The GPU is here for speed; the CPU is what a phone runs. Scoring both on the
# same rows is what turns "the GPU is faster" and "the GPU computes the same thing" into
# measurements. It is also why the per-artifact build does NOT pay for a GPU-vs-CPU probe: this
# job answers that question once, across everything.
#
# READING THE SPEED COLUMN. These are BASE models, untrained, so many of them ramble to the
# 50-token cap and their eval time is decode-bound — which is the one place the GPU LOSES (measured
# ~36 tok/s decode against the CPU's ~141 for a 360M, because a per-token kernel launch is pure
# overhead at this size). The fine-tuned case is the opposite and is where the 6x lives: a trained
# model answers with one label, so the ~1,385-token prefill is the whole cost and the GPU does that
# 10x faster. `tests/pipeline/verify_mnn_backend.slurm` measures that case against llama.cpp.
#
# ONE GPU, and no requeue: this is a measurement, not a results run.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1

source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1)
module load "${CUDA_MOD:-cuda/12.8.1}"
source .venv_gpu/bin/activate

# The context a real CLINC150 prompt needs (~1,385 tokens with all 151 intents). The task spec
# declares 1,024 and `task_max_seq_length` gives this variable precedence, which is what every
# pipeline launcher relies on; without it the eval refuses the prompt outright.
export SLM_MAX_SEQ_LENGTH=4096
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export SLM_SEED="${SLM_SEED:-3407}"

STAMP="${SLURM_JOB_ID:-manual-$$}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --discard deletes each artifact once scored: the full matrix is ~40GB of MNN weights otherwise,
# and every cell is reproducible from its export line.
python hardware_eval/mnn_backend_matrix.py \
  --rows "${MNN_MATRIX_ROWS:-100}" \
  --devices cuda,cpu \
  --out "/mmfs1/gscratch/intelligentsystems/evanly/tmp/mnn-matrix-$STAMP" \
  --discard \
  --report "logs/probes/mnn-backend-matrix-$STAMP.json"
