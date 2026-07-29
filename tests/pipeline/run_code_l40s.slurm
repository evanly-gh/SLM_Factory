#!/bin/bash
#SBATCH --job-name=slm-code-l40s
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=224G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-code-l40s-%j.out
#
# TASK TYPE: code_generation, benchmark: APPS introductory exactly.
# OFFLINE DATA: checksum-verified offline APPS introductory bundle with official train/test
# splits, gold/starter code, and preserved call-based and stdin/stdout execution cases.
# Pass@1 executes every preserved case with a per-case timeout in isolated candidate workers.
# RUN PROFILE: Samsung Galaxy S24 Ultra target; one node with 4 L40S GPUs; Sonnet-1M
# orchestrator; Qwen3.6-primary CoT annotation with specialist fallbacks; honest quantized
# accuracy (SLM_QUANT_EVAL=1); disposable model workers (SLM_CUDA_ISOLATION=1).
# Curation remains gold-only for code: no broken-code hard negatives.
export SLM_MAX_SEQ_LENGTH=4096
export SLM_CUDA_ISOLATION=1
export TASK="Fine tune a small model on the APPS introductory benchmark for Python code generation, producing correct executable programs for both call-based and stdin/stdout problems that pass the preserved APPS tests, targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
