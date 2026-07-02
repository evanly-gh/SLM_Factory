#!/bin/bash
#SBATCH --job-name=slm-conll-ner
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-conll-ner-%j.out
# NER pipeline test: CoNLL-2003 style named entity recognition
# Targeting Moto G Stylus 5G 2023 (6GB RAM, 256GB storage)
# Paper baseline: GLiNER2-base Entity F1 0.345 → 0.810 (+0.465) in 6 iterations
# Our target: decoder model from Android pool, entity extraction for PER/ORG/LOC/MISC
#
# Termination: reaches entity F1 >= stop_threshold (calibrated by planner vs published SOTA)
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

TASK="Fine tune a small model for named entity recognition (NER) on English news text, extracting PER (person), ORG (organization), LOC (location), and MISC (miscellaneous) entity types in CoNLL-2003 style, targeting deployment on my Moto G Stylus 5G 2023 with 6GB RAM and 256GB storage"
echo "=== CoNLL NER pipeline test ==="
echo "Task: $TASK"
python run.py "$TASK"
echo "=== done ==="
