#!/bin/bash
#SBATCH --job-name=slm-gsm8k-l40s
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-gsm8k-l40s-%j.out
#
# CURATED BENCHMARK: gsm8k (in-distribution control, metric exact_match). Promoted from the
# autonomous discovery path on 2026-08-18 — it was the last task whose rows were built by an inline
# converter inside web_acquire rather than by its own loader. Exercises CoT annotation (the loader
# ships gold chains-of-thought, which are preserved rather than regenerated) and the new-correct
# synthesis path.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=gsm8k
export TASK="Fine tune a small model for grade-school math word problem solving (GSM8K dataset), producing step-by-step reasoning and a final numeric answer, targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
