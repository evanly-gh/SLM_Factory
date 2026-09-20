#!/bin/bash
#SBATCH --job-name=slm-goemotions-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-goemotions-cse-%j.out
#
# CURATED BENCHMARK: goemotions (in-distribution, SELECTION metric ekman_macro_f1, REPORTED metric
# macro_auprc). Added 2026-09-06 with the on-device SFT suite. SLM_BENCHMARK_TASK pins the loader.
#
# THE MOST TRAINING-DATA-SENSITIVE TASK IN THE SUITE, which is why it is here for a curation
# method: the BERT baseline sits at 0.46 macro-F1, data augmentation moves it to 0.52 and a clipped
# asymmetric loss to 0.54. Nothing in that ladder is scale.
#
# THE HEADLINE IS THRESHOLD-FREE AUPRC, AND THE LOOP CANNOT COMPUTE IT
#   Macro-F1 over 28 labels is not a property of the model. Practitioners threshold at a fixed 0.5,
#   a fixed 0.3, or a dev-tuned sweep from 0.05 to 0.95, and the same model moves several points
#   between them — which reads as a model improvement and is not one. Macro average precision asks
#   only whether the model RANKS the correct labels highly, so no cutoff can move it.
#
#   But AUPRC needs a ranking and generation returns a hard decision. `scripts/report_eval.py`
#   therefore scores all 28 labels per row through `training.slm_helpers.infer_label_scores_batch`
#   — the one new inference capability this suite required — and that runs ONCE, on the full
#   5,427-row test split. This run selects on Ekman-7 macro-F1 from ordinary generation, whose
#   seven classes all have real support.
#
# THE TAIL IS STATISTICALLY EMPTY AND MUST NEVER CARRY A HEADLINE. Measured test support: grief 6,
# relief 11, pride 16, nervousness 23, embarrassment 37, against neutral 1,787. A published result
# moving `grief` from 0.00 to 0.57 F1 is +2 macro-F1 points earned on six examples. The scorer
# reports per-label F1 in frequency BANDS — head >=300, mid 50-300, tail <50 — banded by gold
# support in the split actually scored, so a smaller draw re-bands honestly rather than claiming
# head-class support it does not have.
#
# WATCH THE DEGENERATE SOLUTION. `neutral` is 1,787 of the test split's mentions, so always
# answering it scores far above chance while being worth nothing. The scorer names that failure
# `neutral_collapse` rather than folding it into a generic mismatch, so the orchestrator can see it.
#
# CONTAMINATION: HIGH. 2020, canonical on the Hub, so the zero-shot baseline is inflated — which
# compresses the apparent delta rather than exaggerating it.
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
export SLM_BENCHMARK_TASK=goemotions
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

export TASK="Fine tune a small model to label the emotions expressed in short social media comments across 27 emotions plus neutral, allowing multiple labels per comment (GoEmotions dataset style), targeting on-device tone detection for reply suggestions on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
