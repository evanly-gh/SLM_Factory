#!/bin/bash
#SBATCH --job-name=slm-conll-ner
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-conll-ner-%j.out
# TEST CASE: NER — stagnation-triggered escalation
#
# PURPOSE: Verify the escalation path fires for a NER task. The device is
# deliberately constrained to 3GB RAM (Samsung Galaxy A14 5G) so the planner
# selects a tier-0 or tier-1 model to start. Nested NER with fine-grained
# biomedical entity types (Gene, Disease, Chemical, Variant) is significantly
# harder than CoNLL-2003 news NER for sub-1B models — expect the tier-0 model
# to plateau below stop_threshold after 3-4 iterations, triggering the
# stagnation detector (STAGNATION_WINDOW=3, STAGNATION_MIN_DELTA=0.02) and
# routing to escalate_node to try a larger tier.
#
# What to check in logs/runs/<timestamp>/run.log:
#   - "ESCALATE (stagnation overrides LLM decision)" appears at least once
#   - selected_model.tier increases across escalation events in dag.json
#   - task_type: "NER"
#   - The run eventually converges or exhausts budget at a higher tier
#
# Dataset: BC5CDR (PubMed abstracts, Chemical + Disease NER, ~1500 train docs)
# Stop threshold: calibrated by planner vs published SOTA (~0.87 entity F1 for 1B models)
# Paper baseline: GLiNER2-base Entity F1 0.345 → 0.810 (+0.465) in 6 iterations
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

# Samsung Galaxy A14 5G: 3GB RAM, 64GB storage — forces planner to pick a small
# tier-0/1 model that will stagnate on biomedical NER, triggering escalation.
TASK="Fine tune a small model for biomedical named entity recognition (NER) on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy A14 5G with 3GB RAM and 64GB storage"
echo "=== Biomedical NER (escalation path) pipeline test ==="
echo "Task: $TASK"
python run.py "$TASK"
echo "=== done ==="
