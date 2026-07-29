#!/bin/bash
#SBATCH --job-name=slm-generation-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-generation-l40s-%j.out
#
# TASK TYPE: generation (strict local Qwen3.6 LLM-as-judge 0..1 scorer). Abstractive
# summarization exercises free-form generation, required local judge evaluation, and the
# generation curation path (no label balancing; length-filtered gold).
export SLM_CUDA_ISOLATION=1
export TASK="Fine tune a small model for abstractive dialogue summarization (SAMSum dataset style), producing a concise natural-language summary of a short chat conversation, targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
