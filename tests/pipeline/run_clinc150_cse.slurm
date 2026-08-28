#!/bin/bash
#SBATCH --job-name=slm-clinc150-cse
#SBATCH --account=gpu-l40s-cse
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-clinc150-cse-%j.out
#
# CSE-quota twin of run_clinc150_l40s.slurm, added 2026-08-23. clinc150 was the only format-bound
# or classification task in the suite that could run on one quota, which meant it queued behind
# whatever the intelligentsystems account was doing while three of its peers could use either.
# Identical in every behavioural respect — same task, same shared body, same model-selection and
# escalation policy — differing only in --account and the 24h wall the CSE quota allows.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=clinc150
export TASK="Fine tune a small model for user-intent classification (CLINC150: 150 intents plus out-of-scope detection), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
