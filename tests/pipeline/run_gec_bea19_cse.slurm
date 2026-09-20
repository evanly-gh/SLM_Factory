#!/bin/bash
#SBATCH --job-name=slm-gec-bea19-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-gec-bea19-cse-%j.out
#
# CURATED BENCHMARK: gec_bea19 (in-distribution, metric errant_f05). Added 2026-09-06 with the
# on-device SFT suite. SLM_BENCHMARK_TASK pins the loader.
#
# WHY THIS TASK IS WORTH A WEEK OF L40S
#   The largest SFT delta in the suite, and it points the right way for this project: zero-shot
#   GPT-4 scores 43.0 ERRANT F0.5 on BEA-19 dev, fine-tuned small models reach 62-75, and
#   fine-tuning GPT-4o gained +22.07 over its own zero-shot. A local model beating a frontier cloud
#   model outright is the claim, and it is measurable here. 34,308 training sentences IS the
#   official BEA-2019 low-resource track, so prior work exists at exactly this data scale.
#
#   It also has the cheapest rows in the suite at ~18.6 tokens, and the most favourable variance
#   (a recent system reports 71.24 +/- 0.28). The AGR/EM@N churn machinery TOPv2 needs is
#   deliberately NOT imported here.
#
# THE SCORER RUNS IN ITS OWN INTERPRETER, AND THE BODY POINTS AT IT
#   ERRANT requires spacy<4 and a pinned en_core_web_sm; `.venv_gpu` is unsloth + transformers
#   5.5.0 + numpy 2.4.6. `_l40s_task_body.sh` exports `ERRANT_VENV` (defaulted, override-able), and
#   `eval/scorers/gec.py` reads it and holds no absolute path — which is what keeps this launcher
#   and its `_cse` twin identical in every executable line. Build it once with
#   `bash scripts/setup_metric_envs.sh`. A MISSING scorer raises `ErrantUnavailable` rather than
#   scoring zero, because a zero would look exactly like a catastrophic regression and send the
#   loop into hours of data rebuilding to chase it.
#
# THE NUMBER'S PROVENANCE, WHICH HAS TO TRAVEL WITH IT
#   1. SINGLE-REFERENCE. W&I+LOCNESS train and dev have one annotator, which caps measurable
#      recall. F0.5 is strongly reference-count dependent: CoNLL-14 at two references puts top
#      systems near 68, and the same systems re-scored against a 10-annotator extension reach 80-81
#      against a human 72.58. Our number is comparable to other single-reference BEA-19 dev numbers
#      and to nothing else. The scorer records `references_per_sentence`.
#   2. THE ORACLE CEILING IS ~0.89, NOT 1.00. Feeding the gold correction back in as the hypothesis
#      scores F0.5 0.8934 (measured on 120 dev sentences), because the reference edits are the
#      annotator's own segmentation while the hypothesis edits are derived by ERRANT's alignment
#      rules. A system at 0.75 is at ~84% of achievable, not 75%.
#   3. PRECISION AND RECALL ARE REPORTED SEPARATELY, ALWAYS. Fine-tuned small models under-correct
#      and large untuned ones over-correct, and both land in the same middling F0.5.
#   4. CONTAMINATION: HIGH. BEA and CoNLL are old and public, so the zero-shot baseline is
#      inflated — which compresses the apparent delta rather than exaggerating it.
#
# LICENCE: research/educational only. Not commercially releasable.
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
export SLM_BENCHMARK_TASK=gec_bea19
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

export TASK="Fine tune a small model to correct grammatical errors in learner English sentences, changing as little as possible (W&I+LOCNESS BEA-2019 dataset style), targeting on-device keyboard correction on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
