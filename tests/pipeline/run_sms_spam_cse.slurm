#!/bin/bash
#SBATCH --job-name=slm-sms-spam-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-sms-spam-cse-%j.out
#
# CSE-quota twin of run_sms_spam_l40s.slurm. Identical in every behavioural respect — same task,
# same shared body, same model-selection and escalation policy — differing only in --account and
# the 24h wall the CSE quota allows. See the l40s script for what the task is and why its metric
# is minority_f1.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=sms_spam
export TASK="Fine tune a small model to classify SMS text messages as unsolicited spam or legitimate (ham), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
