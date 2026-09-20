#!/bin/bash
#SBATCH --job-name=slm-ner-ablation-reset-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-ner-ablation-reset-cse-%j.out
#
# ============================================================================================
# ABLATION 1 of 3 — DOES REUSING THE CURRICULUM ACROSS MODEL TIERS BUY ANYTHING?
#
# BASELINE: slm-ner-bc5cdr-l40s-39311801 (8h46m, 3 tiers, span_f1 0.8532 against a goal that
# started at 0.8000 and was raised to 0.8500). This script is that one file-for-file, with a
# single line added, so the two runs differ in exactly one variable.
#
# THE VARIABLE: `escalate_node` normally carries the dataset forward, which means tier 3 trained
# on a curriculum that tiers 1 and 2 had already spent teacher tokens growing — 2,603 seed rows
# had become 2,660 by the time Qwen3.5-2B first saw them, and every synthesized row in that
# delta was paid for at a lower tier. SLM_ABLATION_RESET_DATA_ON_ESCALATION=1 rewinds to
# `artifacts/dataset_v1.jsonl` on every promotion instead, so each tier starts from the same
# seed the first tier started from: no mined rows, no surgical rows, no inherited mining
# position and no inherited failure-category history (see agent/ablations.py for the full list
# of what is reverted, and why run health's counters are among them).
#
# WHAT EITHER RESULT MEANS. Synthetic data is the expensive part of this pipeline, and it is
# generated per tier. If the reset run reaches roughly 0.8532 anyway, the lower tiers' synthesis
# spend bought nothing that survived promotion and the reuse is incidental. If it lands clearly
# short, then carrying data forward IS the intervention — the cheapest one in the suite, since a
# row generated at tier 1 gets used three times.
#
# The reset is NOT free of confounds and the writeup should say so: each tier also gets fewer
# total rows, so a deficit could be curriculum SIZE rather than curriculum LINEAGE. Tier-1
# numbers are the control for that — nothing has been reset yet when tier 1 runs, so tier 1 must
# reproduce the baseline's 26-iteration trajectory closely or something other than the ablation
# has changed.
# ============================================================================================
export SLM_ABLATION_RESET_DATA_ON_ESCALATION=1

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
# SYNTHESIS GATE: bypassed deliberately, and the measurement is still taken honestly.
#
# The local teacher measures span_f1=0.6140 5-shot against the 0.80 gate, so `synthesis_allowed`
# refuses surgical_synthesis for the whole run and the orchestrator is left with mining and
# hyperparameters. Runs 38832588 and 39294409 only ever generated a row because this flag happened
# to be exported in the SUBMITTING SHELL; it was never in the script, so a clean shell silently
# produced a different experiment. It lives here now so the run is reproducible from the file.
#
# The 0.6140 also understates the teacher: `eval/scorers/ner.NER_PROMPT` never names the
# Chemical/Disease vocabulary that `_pairs` then compares exactly, so the teacher's zero-shot pass
# returned the RIGHT spans under its own label names (MEDICAL, SUBSTANCE, DISEASE) and scored 0.1011.
# BYPASS changes only the DECISION — the score, the real gate and `bypassed=True` are all still
# recorded on the verdict, so the end-of-run report says the data would not have cleared the bar.
export SLM_TEACHER_SYNTH_BYPASS=1

export TASK="Fine tune a small model for biomedical named entity recognition on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
