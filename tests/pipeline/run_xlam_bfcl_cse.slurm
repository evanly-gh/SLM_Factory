#!/bin/bash
#SBATCH --job-name=slm-xlam-bfcl-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-xlam-bfcl-cse-%j.out
#
# CURATED BENCHMARK: xlam_bfcl (format-bound, metric ast_arg_match). SLM_BENCHMARK_TASK
# pins the deterministic loader (train on Salesforce xLAM-60k, evaluate on the Berkeley
# Function-Calling Leaderboard) instead of autonomous web_acquire — run.py starts the run
# NON-autonomous against this task's registry spec, which is where its loader, scorer, metric and
# token budget are declared. Scoring is judge-free AST argument match.
#
# DATA CONTRACT (validated live 2026-08-13, B253):
#   - xLAM is a GATED repo. It resolves via HF_TOKEN, which run.py loads from .env with
#     override=True before any loader import. Without it the run dies at eval_setup.
#   - BFCL is NOT load_dataset-able: 52 category-named JSON files, prompts and gold in separate
#     files joined on `id`, gold args are lists of acceptable values. data/loaders/xlam_bfcl.py
#     names the six AST-gradable categories explicitly and carries the acceptable-value map on
#     `_accept`. Verified: 2,309 scoreable rows, self-consistency exactly 1.0.
#
# CSE quota: 2x L40S (synth server on GPU0, pipeline on GPU1 — the auto-2gpu profile in
# _l40s_task_body.sh). gpu-l40s-cse caps wall-clock at 24h, so --signal=B:USR1@7200 fires the
# checkpoint 2h before the limit and the body calls `scontrol requeue`. A requeued job keeps its
# job ID, so SLM_RUN_DIR is stable and the run resumes from checkpoint.json across the boundary.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=xlam_bfcl
# TEACHER VERIFICATION OF GENERATED ROWS: OFF for this task.
#
# This task has an EXACT programmatic verifier (`TaskSpec.synth_verifier`) that already checks every
# decidable property of a generated row. The teacher pass runs after it and, measured on 2026-08-25,
# does net harm on exactly these tasks: `verify_ner_row` passed 25/25 and the teacher then rejected
# 25/25 for "format" twice running, which stopped that run; calendar lost 1,965 of 8,704 rows (22.6%),
# including "7pm start plus 60 mins is 8pm, not 20:00" -- 8pm IS 20:00.
#
# `verifier_notes` was added to tell the teacher the conventions it was missing, but whether an LLM
# then complies is not something a unit test can establish, and a false rejection is expensive: it
# removes a good row AND can stop the run. Until the notes are shown to lower the rejection rate on a
# live run, the exact verifier is trusted on its own. Set SLM_VERIFY_SYNTH=1 to measure the pass again.
export SLM_VERIFY_SYNTH=0

export TASK="Fine tune a small model for function/tool calling (train on xLAM-60k, evaluate on the Berkeley Function-Calling Leaderboard), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
