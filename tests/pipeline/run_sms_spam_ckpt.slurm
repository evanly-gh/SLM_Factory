#!/bin/bash
#SBATCH --job-name=slm-sms-spam-ckpt
#SBATCH --account=ckpt-intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --qos=ckpt-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-sms-spam-ckpt-%j.out
#
# PREEMPTIBLE VARIANT. Identical science to the `_l40s` twin — every SLM_ export below is copied
# from it verbatim and held there by `test_a_task_is_configured_identically_on_all_accounts` — and a
# different SCHEDULING bargain.
#
# WHY: on 2026-08-27 the `gpu-l40s` queue was unusable for this account. Priority on this cluster is
# almost entirely fair-share (`PriorityWeightFairShare=5000`, `PriorityWeightAge=0`, so a job NEVER
# improves by waiting), the account was running at ~2x its entitlement, and a fresh submission was
# projected to start 30+ hours out. `PriorityWeightQOS` is 100000 — twenty times the fair-share
# weight — and `ckpt-gpu` carries the highest QOS priority defined on the cluster (300), so the same
# job here scores ~100000 against ~3000 and starts in minutes rather than days.
#
# WHAT IT COSTS: this job can be preempted whenever an owning account wants the node back.
# `PreemptMode=REQUEUE`, so preemption puts the job back in the queue rather than killing it
# outright, and `--requeue` below is what lets that happen. But `GraceTime=0` and `KillWait=10s` mean
# there are about ten seconds of warning, which is not enough to checkpoint a training step. So a
# preempted run resumes from its last COMMITTED checkpoint and loses whatever was in flight.
#
# That trade is right for these three tasks and wrong for toolbench: sms_spam, clinc150 and
# ner_bc5cdr train in minutes, so a lost in-flight step costs minutes. Toolbench trains for ~93
# minutes a step and belongs on a non-preemptible queue.
#
# The job ID survives a requeue, so SLM_RUN_DIR and the artifacts under it survive with it.
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
