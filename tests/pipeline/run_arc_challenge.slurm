#!/bin/bash
#SBATCH --job-name=slm-arc-challenge
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-arc-challenge-%j.out
# Generation pipeline test: ARC-Challenge science reasoning
# Targeting Moto G Stylus 5G 2023 (6GB RAM, 256GB storage)
# Paper baseline: Llama 3.2-3B Acc 5.3% → 72.6% (+67.3pp) in 11 iterations
# Our target: smaller model from Android pool (<=2B), with hardware constraints
# Uses A100 for faster training of generation tasks (longer sequences, CoT annotation)
#
# Termination: reaches accuracy >= stop_threshold (calibrated by planner vs published SOTA)
# or exhausts the 1500-turn cold-start budget. No artificial step caps.
set -euo pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache

nvidia-smi -L || true
[ -d .venv_gpu ] || bash scripts/setup_gpu_env.sh
source .venv_gpu/bin/activate


TASK="Fine tune a small model for science reasoning on ARC-Challenge (multiple choice science questions from grades 3-9 requiring multi-step reasoning), targeting deployment on my Moto G Stylus 5G 2023 with 6GB RAM and 256GB storage"
echo "=== ARC-Challenge pipeline test ==="
echo "Task: $TASK"
python run.py "$TASK"
echo "=== done ==="
