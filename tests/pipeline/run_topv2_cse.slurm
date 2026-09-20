#!/bin/bash
#SBATCH --job-name=slm-topv2-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-topv2-cse-%j.out
#
# CURATED BENCHMARK: topv2 (format-bound, metric exact_match). Added 2026-09-06 with the on-device
# SFT suite. SLM_BENCHMARK_TASK pins the loader.
#
# WHAT THIS TASK MEASURES
#   An assistant command in, a nested intent/slot parse tree out, under TOPv2's published
#   low-resource protocol: train on the six source domains, adapt to `reminder` and `weather` with
#   a 25-SPIS sample of each (501 and 177 rows as reconstructed here). The comparison scalar is the
#   MEAN of the two domains' exact match, and both components are always reported — `weather` is
#   flat and will saturate while `reminder` is 21.5% compositional and will not, so an unlabelled
#   average would hide which half moved.
#
# TWO THINGS TO KNOW BEFORE CITING A NUMBER FROM THIS RUN
#   1. THE SPIS SPLITS ARE RECONSTRUCTED. The official low-resource files ship with the gated
#      TOPv2 release, are not in the `WillHeld/top_v2` mirror, and have no mirror on the Hub
#      (checked 2026-09-06). `data/loaders/topv2.py` reimplements the sampling rule from its
#      definition at a published seed; it lands within 1.6% of the released 25-SPIS sizes (501 vs
#      493 on reminder, 177 vs 176 on weather), which is the evidence the rule and the data are
#      both right. It is not the released file.
#   2. EM DEPENDS ON THE PARSE SERIALIZATION, and ours is the mirror's `semantic_parse` string
#      verbatim. Together with (1) that makes this number a controlled comparison against our own
#      baseline, not a leaderboard entry against RINE or shift-reduce.
#
# SEED VARIANCE IS WORST HERE, WHICH IS PART OF WHY THE TASK IS IN THE SUITE. A production-sized
# TOPv2 baseline churns on ~21% of examples across retrains with identical data and hyperparameters
# (EM 83.74 / EM@10 73.18 / AGR 78.47, arXiv:2204.04735), and this trains on 501 and 177 rows.
# `scripts/report_eval.py` therefore defaults to FIVE seeds for this task and three elsewhere. The
# published number comes from that script on the full 11,449-row test split, never from this run's
# in-loop eval, which is capped at 1,000 rows for checkpoint selection only.
#
# CSE quota: 2x L40S (auto-2gpu profile), 24h wall-clock with checkpoint-and-requeue. This exists
# as a TWIN of the `_l40s` launcher because the dedicated quota is 10 GPUs for the whole group and
# is routinely saturated; whichever account frees first is the one that runs. Every `export SLM_*`
# line below is byte-identical to the twin's and is held there by
# `test_a_task_is_configured_identically_on_all_accounts` — a task measured under different
# settings depending on which account had a free GPU is a lottery on the science, and it has
# happened before (toolbench carried SLM_VERIFY_SYNTH=0 and SLM_EVAL_SIZE_CAP=500 on one account
# and neither on the other).
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=topv2
# TEACHER VERIFICATION OF GENERATED ROWS: SHADOW MODE — it runs, and it drops nothing.
#
# The pass is measured, not trusted. `enforce` has been measured doing net harm on this pipeline:
# on run 38832588 the teacher rejected 25 of 25 `ner_bc5cdr` rows that `verify_ner_row` had just
# accepted, twice running, and that STOPPED the run; on `calendar_json` it rejected 1,965 of 8,704
# (22.6%), including "7pm start plus 60 mins is 8pm, not 20:00" -- 8pm IS 20:00. Both tasks ship
# with it off, and their launchers say it should be re-measured once `verifier_notes` exists to
# tell the teacher the conventions it was missing.
#
# Those notes now exist for this task. Shadow mode is how we find out whether they worked without
# betting a multi-day run on the answer: the teacher is called on every generated row, the verdict
# is recorded, and EVERY ROW IS KEPT REGARDLESS. Grep `[verify:shadow]` for the rate and the
# reason histogram.
#
# It is also non-interfering by construction. Shadow mode deliberately does NOT feed rejection
# reasons into the next generation prompt, because that would change what the teacher generates
# and the measured rate would then describe a trajectory that only exists because we measured it.
# TEACHER CONTEXT: raised for this task, and the second knob pays for the first.
#
# This task's prompt ENUMERATES all 166 intent and slot names (~666 of its ~829 tokens), because
# exact match compares label names character by character and no model can guess
# `IN:GET_REMINDER_DATE_TIME`. That makes each demonstration a full ~850-token prompt, so five of
# them plus the question do not fit the default 8,192-token teacher context. Run 39719881 reported
# exactly that and fell back to a ZERO-shot measurement:
#
#   "a 17113-char demonstration block does not fit beside the longest 3463-char prompt inside a
#    7680-token input budget (room for 15737 chars of demonstrations), so the k-shot measurement
#    is SKIPPED and the gate uses the zero-shot number ... RAISE SLM_SYNTH_MAX_MODEL_LEN for this
#    task (and lower SLM_SYNTH_MAX_NUM_SEQS to pay for the KV cache) until five demonstrations
#    fit."
#
# 16384 leaves room for the 17,113 chars of demonstrations plus the question. max-num-seqs halves
# to 32 because the two trade directly against each other in the KV cache on a single L40S, and a
# server that cannot allocate its cache does not start at all.
#
# NOT by lowering SLM_SYNTH_SHOTS, which the same message warns against: on toolbench that was
# tried twice and the programmatic verifier kept 0 rows of 1,019 attempts at 0 shots and 0 of 133
# at the 1-2 shots that did fit (runs 38832586, 38985393). A format-bound task cannot be
# synthesized from a prompt that does not show the format enough times (B276).
export SLM_SYNTH_MAX_MODEL_LEN=16384
export SLM_SYNTH_MAX_NUM_SEQS=32
export SLM_VERIFY_SYNTH=shadow
# SYNTHESIS GATE: bypassed deliberately, and the measurement is still taken honestly.
#
# The gate refuses `surgical_synthesis` when the teacher's own few-shot score on the task falls
# below threshold. On a task nobody has measured yet that is a near-certain refusal, which would
# leave the orchestrator with mining and hyperparameters only and tell us nothing about whether
# synthesis helps here.
#
# BYPASS changes only the DECISION. The teacher's measured score, the real gate and
# `bypassed=True` are all still recorded on the verdict, so the end-of-run report still says
# whether the data would have cleared the bar. It lives in the file rather than the submitting
# shell because that is how runs 38832588, 39294409 and 39361648 all became configurations nobody
# had written down.
export SLM_TEACHER_SYNTH_BYPASS=1

export TASK="Fine tune a small model to parse spoken assistant commands into nested intent and slot trees for reminders and weather queries (TOPv2 dataset style, low-resource 25-SPIS adaptation), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
