#!/bin/bash
#SBATCH --job-name=slm-sms-spam-ablation-nostop-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-sms-spam-ablation-nostop-l40s-%j.out
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

# ─────────────────────────────────────────────────────────────────────────────────────────────
# NO-CONVERGENCE VARIANT of run_sms_spam_l40s.slurm. Same task, same data, same everything except
# the accuracy goal — this run is not allowed to stop when it gets good.
#
# WHY IT EXISTS. Both sms_spam and its sibling converged in single digits: sms_spam cleared its
# 0.8662 teacher-derived goal on iteration 2, was raised to 0.89 and then 0.95, peaked at 0.944 and
# stopped after 7 iterations on a 360M model; clinc150 stopped after 4 on a 0.6B. Neither ever
# escalated a tier, because escalation only fires for a model stuck BELOW the goal. So the suite
# has no answer for what these two tasks reach when pushed the way the five never-converging tasks
# were pushed — up the whole ladder until stagnation or the eval cap.
#
# THE PIN. SLM_STOP_THRESHOLD=0.99 is THRESHOLD_CEILING (agent/threshold.py); 1.0 is deliberately
# disallowed because label noise makes it unreachable. Pinned here the goal is never met, so the
# run plateaus, escalates, and exits only on tier exhaustion or MAX_EVALS_BEFORE_ESCALATION=30 per
# model — which is exactly the requested stopping rule, and is the pipeline's existing default
# behaviour rather than anything new.
#
# IT ALSO CLOSES A HOLE. The pin sets `initial_stop_threshold` to 0.99 as well, and the floor is
# what the orchestrator's threshold_adjustment is clamped to. On the ORIGINAL sms_spam run the
# orchestrator raised the goal to 0.95, failed four attempts, then LOWERED it to 0.92 — below the
# 0.944 already achieved — and declared convergence. Legal, because 0.92 cleared the 0.8662 floor.
# With floor == goal == 0.99 that move is arithmetically impossible here.
#
# Shadow verification and the synthesis bypass are set to match the five suite runs so the numbers
# are comparable. Note both teachers actually CLEAR the 0.80 fitness gate on these two tasks
# (sms_spam 0.8662, clinc150 0.9064), so the bypass changes nothing and is set only for parity.
export SLM_STOP_THRESHOLD=0.99
export SLM_VERIFY_SYNTH=shadow
export SLM_TEACHER_SYNTH_BYPASS=1

export TASK="Fine tune a small model to classify SMS text messages as unsolicited spam or legitimate (ham), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
