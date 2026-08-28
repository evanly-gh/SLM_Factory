#!/bin/bash
#SBATCH --job-name=slm-ner-bc5cdr-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-ner-bc5cdr-l40s-%j.out
#
# CURATED BENCHMARK: ner_bc5cdr (format-bound, metric span_f1). Promoted 2026-08-13 from the
# autonomous path, where it reached BC5CDR only because the orchestrator happened to emit a plan
# web_acquire could resolve — not reproducible. SLM_BENCHMARK_TASK pins the loader.
#
# WHAT CHANGED SINCE slm-ner-l40s-37531245 (44.8h, best span-F1 0.8628 vs a 0.88 threshold,
# died with `bounded data_rebuild plan space is exhausted`):
#   1. Data source is now the checksummed local bundle data/local/bc5cdr — 5,096 train rows
#      against the 3,403 the live loader produced. The previous run ran out of gold to resample,
#      which is what exhausted the plan space, so this addresses the actual cause.
#   2. tner/bc5cdr is script-based and DEAD under datasets 4.3.0; spyysalo/bc5cdr is gone from
#      the Hub. The local bundle needs no network at all; T-NER raw JSON is the fallback.
#   3. Train/serve prompt skew FIXED. training/lora_trainer.py::_training_turn now imports
#      eval.scorers.ner.NER_PROMPT instead of hand-copying a version that omitted the
#      "Reply with [] if there are no entities" sentence. Verified byte-identical.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (auto-2gpu profile), seven-day wall.
# --requeue + USR1 remain a safety net against node failure, not a wall-clock mechanism.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=ner_bc5cdr
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

export TASK="Fine tune a small model for biomedical named entity recognition on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
