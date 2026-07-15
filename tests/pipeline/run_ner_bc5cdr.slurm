#!/bin/bash
#SBATCH --job-name=slm-ner
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-ner-%j.out
#
# TEST — NER task type end-to-end (entity-span extraction + entity-F1 scorer).
# ============================================================================
# run_qwen_escalation only exercised math_reasoning. This test covers the NER path:
# the BC5CDR real benchmark loader (B131: token+BIO tags → {text, entities} spans), the
# NER training prompt format, the NER scorer (entity-F1 via Counter multiset), and the
# NER-aware curate report (entity-type distribution, provenance). If the HF dataset is
# unreachable it falls back to the bounded/diversified Exa acquisition + Claude entity
# annotation ladder (B140), which is itself worth exercising.
#
# Setup: BC5CDR biomedical NER, smallest_first, moderate threshold. Uses 256-token eval
# (raised for NER in B132 so entity JSON isn't truncated).
#
# COVERS (that run_qwen_escalation misses): NER task_type, NER scorer/metrics, NER curate
#   accounting, and (on fallback) the real-benchmark→Exa→annotation acquisition ladder.
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
export SLM_STOP_THRESHOLD=0.70

TASK="Fine tune a small model for biomedical named entity recognition (NER) on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset), targeting deployment on my Samsung Galaxy S24 with 8GB RAM and 128GB storage"
echo "=== NER (BC5CDR) validation test ==="
echo "Task: $TASK"
python tests/pipeline/run.py "$TASK"
echo "=== done ==="
