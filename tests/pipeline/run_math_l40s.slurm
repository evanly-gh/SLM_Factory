#!/bin/bash
#SBATCH --job-name=slm-math-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-math-l40s-%j.out
#
# TASK TYPE: math_reasoning (exact-match-after-normalization scorer). Exercises Qwen3.6-primary CoT
# annotation with task-aware specialist fallbacks (DeepSeek then OpenAI), the 512-token
# generation eval, and the "no wrong-answer hard negatives" curation rule for math.
export SLM_CUDA_ISOLATION=1
export TASK="Fine tune a small model for grade-school math word problem solving (GSM8K dataset), producing step-by-step reasoning and a final numeric answer, targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
