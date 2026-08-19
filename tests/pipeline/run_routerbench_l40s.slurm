#!/bin/bash
#SBATCH --job-name=slm-routerbench-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-routerbench-l40s-%j.out
#
# CURATED BENCHMARK: routerbench (out-of-distribution, metric minority_f1 — the headline number
# has always been the minority class's F1, and it was reported under a `macro_f1` label until the
# task registry made each task name its own metric). SLM_BENCHMARK_TASK pins the deterministic
# RouterBench loader (reframed to a binary local-vs-route decision) instead of autonomous
# web_acquire — run.py starts the run NON-autonomous against this task's registry spec, which is
# where its loader, scorer, metric and token budget are declared. The dataset is pulled live from
# HuggingFace on the compute node (public, no token).
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=routerbench
export TASK="Fine tune a small model to decide whether a user query should be answered locally on-device or routed to a larger model (RouterBench), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
