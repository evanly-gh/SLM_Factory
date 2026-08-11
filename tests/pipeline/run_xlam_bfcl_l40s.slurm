#!/bin/bash
#SBATCH --job-name=slm-xlam-bfcl-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-xlam-bfcl-l40s-%j.out
#
# CURATED BENCHMARK: xlam_bfcl (task_type=function_call, metric ast_arg_match). SLM_BENCHMARK_TASK
# pins the deterministic loader (train on Salesforce xLAM-60k, evaluate on the Berkeley
# Function-Calling Leaderboard) instead of autonomous web_acquire — run.py starts the run
# NON-autonomous with the loader's task_type. The datasets are pulled live from HuggingFace on the
# compute node (public; BFCL ships a loader script, trust_remote_code). Scoring is judge-free
# executable AST argument match.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=xlam_bfcl
export TASK="Fine tune a small model for function/tool calling (train on xLAM-60k, evaluate on the Berkeley Function-Calling Leaderboard), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
