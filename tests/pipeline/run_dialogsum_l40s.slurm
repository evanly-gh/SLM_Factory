#!/bin/bash
#SBATCH --job-name=slm-dialogsum-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-dialogsum-l40s-%j.out
#
# CURATED BENCHMARK: dialogsum (in-distribution, metric judge_mean_0_1).
# SLM_BENCHMARK_TASK pins the deterministic DialogSum + SAMSum summarization loader instead of
# autonomous web_acquire — run.py starts the run NON-autonomous against this task's registry spec,
# which is where its loader, scorer, metric and token budget are declared. The datasets are pulled
# live from HuggingFace on the compute node (public, no token). This is the ONLY task scored by the
# LLM judge, served by the co-located local Qwen judge (no cloud judge calls).
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=dialogsum
export TASK="Fine tune a small model for dialogue summarization (DialogSum and SAMSum chat/meeting transcripts), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
