#!/bin/bash
#SBATCH --job-name=slm-multiconer-ckpt
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-multiconer-ckpt-%j.out
#
# CURATED BENCHMARK: multiconer (format-bound, SELECTION metric micro_f1, REPORTED metric
# macro_f1). Added 2026-09-06 with the on-device SFT suite. SLM_BENCHMARK_TASK pins the loader.
#
# THE TWO METRICS ARE NOT THE SAME NUMBER, AND NEITHER IS OPTIONAL
#   This run selects checkpoints on entity-level MICRO-F1 over the official 871-row dev split. The
#   published headline is entity-level MACRO-F1 over a fixed stratified 20,000-row slice of the
#   249,980-row test split, produced once by `scripts/report_eval.py`.
#
#   Both halves are forced by the data. A 33-class macro average over 871 rows is noise: measured
#   on that split, eight or more classes have single-digit support, so macro-F1 there is a fact
#   about a handful of entities and a loop selecting on it would chase annotation accident. And the
#   test split cannot be scored every iteration — it is 62 MB of CoNLL. Micro pools every mention,
#   so it is stable at dev size and moves with real improvement; macro weights the 33 classes
#   equally, which is what stops a model scoring well by learning the four common ones.
#
#   The report slice publishes its seed (20260906) and a content hash, because a macro-F1 "on the
#   20k slice" means nothing if two runs drew two different slices. Measured: the slice covers all
#   33 types with the rarest at 73 mentions, against 8+ classes below 20 in dev.
#
# WHAT THIS RUN DOES NOT REPORT: the clean-versus-corrupted robustness gap. The typo-corrupted test
# portion is the task's headline feature and the partition is NOT in the release — all 249,980 test
# sentence headers carry only `# id <uuid>`, with none of the `domain=en` attribute train and dev
# have. Secondary sources disagree about it too (74,960 + 210,267 does not reconcile with 249,980),
# which is consistent with it having been organizer-side metadata. Reconstructing it heuristically
# would mean reporting a gap measured against a partition we invented.
#
# THE PROMPT ENUMERATES ALL 33 TYPE NAMES, and that is not padding. `eval.scorers.ner.NER_PROMPT`
# omits its vocabulary and the scorer compares types exactly; on the BC5CDR teacher probe that cost
# a measured 0.1011 against a real 0.6140, because the teacher returned the right spans under its
# own label names. With two guessable types that was a 6x understatement. With 33, including
# `OtherPROD` and `Medication/Vaccine`, an unenumerated prompt measures vocabulary telepathy.
#
# PREEMPTIBLE VARIANT — A SMOKE-TEST LAUNCHER, NOT WHERE THIS TASK'S RESULTS COME FROM.
#
# Identical science to the `_l40s` twin: every `export SLM_*` below is copied from it verbatim and
# held there by `test_a_task_is_configured_identically_on_all_accounts`. What differs is the
# SCHEDULING bargain, and why anyone would take it.
#
# WHY THIS FAMILY GREW ON 2026-09-06. It was previously limited to tasks that train in minutes, on
# the reasoning that a preemption then costs minutes. That rule is about RESULTS runs. These
# launchers exist because the on-device SFT suite's first executions had to be shaken out, and the
# `gpu-l40s` queue could not start a seven-day job until after the Sept 8-9 maintenance window —
# so a first run there would have discovered its first bug two days later. The first two smoke
# tests found two real scorer bugs (39707195: ERRANT tokenization mismatch, teacher 0.1569 -> 0.5111
# once fixed; 39707196: prose harvested as a 20-label prediction), which is the whole argument.
#
# For a smoke test, preemption is close to irrelevant: the question being asked is "does this task
# run end to end and are the numbers coherent", and that is answered in the first iteration.
#
# WHAT IT COSTS: preemption at any time. `PreemptMode=REQUEUE` plus `--requeue` returns the job to
# the queue rather than killing it, but `GraceTime=0` and `KillWait=10s` leave ~10 seconds — not
# enough to checkpoint a training step — so a preempted run resumes from its last COMMITTED
# checkpoint and loses what was in flight.
#
# DO NOT POOL NUMBERS FROM THIS FAMILY WITH THE DEDICATED RUNS. The `-ckpt` job name is what keeps
# them apart in logs/slurm/, and it is asserted.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=multiconer
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

export TASK="Fine tune a small model for fine-grained named entity recognition on short search queries and Wikipedia sentences across 33 entity classes including people, creative works, products and medical terms (MultiCoNER II dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
