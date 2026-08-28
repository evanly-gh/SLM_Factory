#!/bin/bash
#SBATCH --job-name=slm-calendar-json-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-calendar-json-l40s-%j.out
#
# CURATED BENCHMARK: calendar_json (format-bound, metric ast_arg_match). SLM_BENCHMARK_TASK pins the
# deterministic loader — TOPv2 reminder utterances plus the frozen SGD Calendar_1 bundle — instead of
# autonomous web_acquire, so run.py starts NON-autonomous against this task's registry spec, where its
# loader, scorer, metric and token budget are declared.
#
# What makes this task different from xlam_bfcl, and why it is worth running alongside it: the answer
# is not merely a well-formed call, it is a RESOLVED one. Every gold label required turning a relative
# expression ("tomorrow", "tonight", "Friday at 4") into an absolute ISO-8601 instant against a
# per-row reference. That makes it the only task in the registry whose synthetic rows can be checked
# for CONTENT rather than form, by re-resolving the request with the loader's own grammar
# (`data/synth_verifiers.verify_calendar_row`) — no teacher call, no judgement.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 — the
# auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is the safety net
# against node failure and maintenance across a 7-day reservation.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=calendar_json
# TEACHER VERIFICATION OF GENERATED ROWS: OFF for this task.
#
# This task has an EXACT programmatic verifier (`TaskSpec.synth_verifier`) that already checks every
# decidable property of a generated row. The teacher pass runs after it and, measured on 2026-08-25,
# does net harm on exactly these tasks:
#   ner_bc5cdr    verify_ner_row passed 25/25; the teacher then rejected 25/25 for "format", twice
#                 running, which is what stopped the run.
#   calendar_json 1,965 of 8,704 rows rejected (22.6%), including "7pm start plus 60 mins is 8pm,
#                 not 20:00" -- 8pm IS 20:00 -- and six rejections of the year-roll the gold uses.
#
# `verifier_notes` was added to tell the teacher the conventions it was missing, but whether an LLM
# then complies is not something a unit test can establish, and a false rejection is expensive: it
# removes a good row AND can stop the run. Until the notes are shown to lower the rejection rate on a
# live run, the exact verifier is trusted on its own. Set SLM_VERIFY_SYNTH=1 to measure the pass again.
export SLM_VERIFY_SYNTH=0

export TASK="Fine tune a small model to convert natural-language scheduling requests into Google Calendar events.insert JSON with resolved ISO-8601 timestamps, targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
