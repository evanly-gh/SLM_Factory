#!/bin/bash
#SBATCH --job-name=slm-calendar-json-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-calendar-json-cse-%j.out
#
# CURATED BENCHMARK: calendar_json (format-bound, metric ast_arg_match). NEW 2026-08-13.
#
# The task: turn "Schedule a meeting at 5pm tomorrow" into a Google Calendar API v3
# events.insert request body with a RESOLVED ISO-8601 timestamp, ready for the host app to post.
# Modelled as function_call so it reuses the existing scorer, including the format_valid vs
# content_correct split — the distinction that matters when the output feeds an API.
#
# DATA CONTRACT (validated live 2026-08-13):
#   train — TOPv2 `reminder` CREATE_REMINDER (WillHeld/top_v2, parquet). 4,242 usable rows.
#   eval  — SGD Calendar_1 AddEvent frames, raw JSON from the google-research-datasets GitHub
#           repo (the HF mirror is script-based and dead under datasets>=4). 478 usable rows.
#   Two independently built corpora, so eval is a transfer test, not a held-out slice.
#   Each row pins its own reference datetime (hashed from the row key, so it is reproducible but
#   varies across rows) and states it in the prompt; rows whose temporal expression does not
#   resolve under the strict grammar in data/loaders/calendar_json.py are DROPPED rather than
#   guessed at. Self-consistency verified at exactly 1.0 on both sides.
#
# CSE quota: 2x L40S (auto-2gpu profile). 24h wall-clock; --signal=B:USR1@7200 checkpoints 2h
# early and the body requeues, keeping the job ID and therefore SLM_RUN_DIR stable.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=calendar_json
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
# SYNTHESIS GATE: kept identical to this task's l40s launcher, where the reasoning is written out.
# The teacher measurement, the real 0.80 gate and `bypassed` are all still recorded on the verdict;
# BYPASS changes only the decision. Divergence here is what made two runs of the same task different.
export SLM_TEACHER_SYNTH_BYPASS=1

export TASK="Fine tune a small model to convert natural-language scheduling requests into Google Calendar events.insert JSON with resolved ISO-8601 timestamps, targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
