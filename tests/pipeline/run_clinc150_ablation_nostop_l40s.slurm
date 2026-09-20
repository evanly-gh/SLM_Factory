#!/bin/bash
#SBATCH --job-name=slm-clinc150-ablation-nostop-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-clinc150-ablation-nostop-l40s-%j.out
#
# CURATED BENCHMARK: clinc150 (out-of-distribution, metric macro_f1). SLM_BENCHMARK_TASK
# pins the deterministic CLINC150 loader (150 intents + out-of-scope) instead of autonomous
# web_acquire — run.py starts the run NON-autonomous against this task's registry spec, which is
# where its loader, scorer, metric and token budget are declared. The dataset is pulled live from
# HuggingFace on the compute node (public, no token).
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=clinc150

# ─────────────────────────────────────────────────────────────────────────────────────────────
# NO-CONVERGENCE VARIANT of run_clinc150_l40s.slurm. Same task, same data, same everything except
# the accuracy goal — this run is not allowed to stop when it gets good.
#
# WHY IT EXISTS. Both clinc150 and its sibling converged in single digits: sms_spam cleared its
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

export TASK="Fine tune a small model for user-intent classification (CLINC150: 150 intents plus out-of-scope detection), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
