#!/bin/bash
#SBATCH --job-name=slm-xlam-verify-cse
#SBATCH --account=gpu-l40s-cse
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-xlam-verify-cse-%j.out
#
# VERIFICATION RUN, not a results run. Its job is to prove that the 2026-08-19 interventions actually
# add data before a full-length run spends money finding out they do not — which is what happened on
# 38566712, where eight consecutive rebuilds added zero rows over 7h42m and nothing stopped it.
#
# What it must demonstrate, in order:
#   1. the xlam_bfcl loader returns a few thousand gold train rows and ~1,000 BFCL eval rows;
#   2. the orchestrator authors a task brief from real rows and it is logged in full;
#   3. the teacher is scored FIVE-SHOT on the eval set and the synthetic-data gate is decided;
#   4. every data_rebuild ADDS rows — a few hundred at a time — by re-reading xLAM past the initial
#      slice, by a web-discovered dataset, or by targeted synthesis;
#   5. quality control removes a plausible number of rows rather than eating the curriculum.
#
# `agent/run_health.py` stops the run itself if any of those stops being true: two consecutive empty
# rebuilds, a QC drop over 1,000 rows or 25%, two verification wipeouts, three mining shutouts, or
# two load failures. So the guard against a wasted run is in the code, not in the operator watching.
#
# Deliberately NOT requeued and time-boxed to 8h: a verification run that dies should stay dead so
# the failure is examined rather than silently restarted.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=xlam_bfcl

# single_model: the orchestrator picks one model and the run keeps it. No escalation on failure, no
# downward regression on success — so every score change is attributable to a data or hyperparameter
# intervention rather than to a model swap, which is the whole point of a verification run.
export SLM_MODEL_SELECTION_STRATEGY=single_model

# Bounded two ways, because a verification run should end on its own rather than be remembered.
#
# The eval cap is the real bound: under `single_model` there is nothing to escalate to, so hitting it
# TERMINATES. This was 10 while the interventions were being verified — enough to watch the loader,
# both rebuild sub-strategies and the curriculum growth, cheaply. That verification is done, and 10
# turned out to be the binding constraint rather than a safety net: run 38708719 ended at 0.8600
# against a 0.8660 goal, still improving, purely because it ran out of evals. Back to the project
# default so the run can actually converge.
export SLM_MAX_EVALS_BEFORE_ESCALATION=30

# Wall clock, well inside the 8h allocation so the pipeline writes its own report rather than being
# killed mid-write by Slurm.
export SLM_MAX_WALLCLOCK_S=21600

export TASK="Fine tune a small model for function/tool calling (train on xLAM-60k, evaluate on the Berkeley Function-Calling Leaderboard), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
