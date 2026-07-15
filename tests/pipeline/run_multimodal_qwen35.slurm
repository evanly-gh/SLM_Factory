#!/bin/bash
#SBATCH --job-name=slm-multimodal
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-multimodal-%j.out
#
# TEST — multimodal Qwen3.5 text-only LoRA via FastVisionModel (B136, UNVERIFIED path).
# ============================================================================
# run_qwen_escalation started on Qwen3-0.6B (text) and escalated through the text Qwen3
# models, so it NEVER exercised the multimodal Qwen3.5 loading/training/inference path.
# This test forces a multimodal model to validate that path specifically.
#
# SLM_FORCE_MODEL pins Qwen/Qwen3.5-0.8B (multimodal). is_multimodal_model() should route
# training through FastVisionModel with finetune_vision_layers=False (language-only LoRA)
# and inference through FastVisionModel too. A cheap classification task keeps it short.
#
# THIS IS THE CRITICAL UNVERIFIED PATH — watch for:
#   - successful load via FastVisionModel (NOT "Incorrect image source ... <|im_start|>")
#   - training completes (loss decreases) and eval produces non-garbage predictions
#   - if it crashes in the vision processor, that confirms the FastVisionModel recipe needs
#     the UnslothVisionDataCollator / a text-only collator (documented follow-up in B136).
#
# COVERS (that run_qwen_escalation misses): multimodal Qwen3.5 train + infer path.
# ============================================================================
set -euo pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR=/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache

nvidia-smi -L || true
.venv_gpu/bin/python -c '' >/dev/null 2>&1 || bash scripts/setup_gpu_env.sh
source .venv_gpu/bin/activate

export SLM_ORCHESTRATOR_MODEL=claude-haiku-4-5
export SLM_MODEL_SELECTION_STRATEGY=smallest_first
export SLM_FORCE_MODEL="Qwen/Qwen3.5-0.8B"   # pin the multimodal model (all strategies honor this)
export SLM_STOP_THRESHOLD=0.72

TASK="Fine tune a small model for financial news sentiment classification (positive, negative, neutral) using the FinancialPhraseBank dataset, targeting deployment on my Pixel 8 with 8GB RAM and 128GB storage"
echo "=== Multimodal Qwen3.5 (FastVisionModel text-only LoRA) validation test ==="
echo "Task: $TASK  (forced model: $SLM_FORCE_MODEL)"
python tests/pipeline/run.py "$TASK"
echo "=== done ==="
