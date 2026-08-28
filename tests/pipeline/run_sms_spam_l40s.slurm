#!/bin/bash
#SBATCH --job-name=slm-sms-spam-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-sms-spam-l40s-%j.out
#
# CURATED BENCHMARK: sms_spam (in-distribution, metric minority_f1). SLM_BENCHMARK_TASK pins the
# deterministic loader (the checksummed data/local/sms_spam bundle, HuggingFace ucirvine/sms_spam
# as the fallback) instead of autonomous web_acquire — run.py starts the run NON-autonomous
# against this task's registry spec, which is where its loader, scorer, metric and token budget
# are declared.
#
# WHY THE METRIC IS minority_f1 AND NOT accuracy. The corpus is ~87% ham. Answering `ham` for
# every message scores 0.87 accuracy and ~0.47 macro-F1, both of which look like a working model.
# Minority-class F1 scores it 0.0. This task is the suite's cheapest collapse detector and the
# metric is the only reason that is true — see tasks/sms_spam.py.
#
# HISTORY WORTH KNOWING: an sms_spam loader existed before 2026-08-18 and was deleted with the
# `task_type` channel. It is not restored, it is rewritten: the old split was unshuffled (B44), the
# corpus was never deduplicated so copies of the same message sat on both sides of the firewall,
# and the data came over plain HTTP with no checksum. All three are fixed in the new loader.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=sms_spam
export TASK="Fine tune a small model to classify SMS text messages as unsolicited spam or legitimate (ham), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
