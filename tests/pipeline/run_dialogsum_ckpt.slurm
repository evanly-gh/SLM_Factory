#!/bin/bash
#SBATCH --job-name=slm-dialogsum-ckpt
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-dialogsum-ckpt-%j.out
#
# CURATED BENCHMARK: dialogsum (in-distribution, SELECTION metric rouge_l, REPORTED metric
# rouge_1_2_l_bertscore). SLM_BENCHMARK_TASK pins the loader.
#
# REWRITTEN 2026-09-06. This launcher previously described a task that no longer exists: it said
# "metric judge_mean_0_1", "the deterministic DialogSum + SAMSum summarization loader", and
# "LLM judge, served by the co-located local Qwen judge". All three are now wrong. What changed,
# and why each change is about the data rather than a preference:
#
#   1. SAMSUM IS GONE. Same task, single reference, easier, and it was diluting the training set
#      with a second summary style while contributing nothing the metric could use.
#
#   2. THE SOURCE CHANGED, because the old one cannot express the metric. DialogSum's test split
#      ships THREE independent human summaries per dialogue, and `knkarthick/dialogsum` — the
#      mirror the old loader read — FLATTENS them: its test.csv is 1,500 rows with a single
#      `summary` column, the same dialogue repeated once per annotator. Scoring that would count
#      every dialogue three times against one arbitrary reference. The loader now reads the
#      original release and REFUSES a split whose rows do not carry exactly three references.
#
#   3. THE JUDGE IS GONE AS THE HEADLINE. A judge score is comparable to no published number, is
#      non-deterministic, and costs money on every eval in a loop that evaluates dozens of times.
#      `needs_judge` is now False and `toolbench` is the only judged task left — which is right,
#      because ToolEval's pass rate is DEFINED as a majority vote of judged assessments.
#
#      NOTE: the local Qwen server this job still starts is the SYNTHESIS teacher, not a judge.
#      Nothing in this task's scoring path calls a model any more.
#
#   4. THE CEILING IS RECORDED. One annotator scored against the other two reaches ROUGE-1 53.35 /
#      ROUGE-2 26.72 / ROUGE-L 50.84; out-of-the-box models sit near 36 and the best fine-tuned
#      near 47. So ~11 points is what fine-tuning buys and ~6 remain, and a 47 is about 88% of
#      human rather than 47% of perfect. The scorer carries all three numbers in `per_class` so a
#      report cannot lose them.
#
# WATCH-OUTS FOR THIS RUN
#   500 test dialogues is the smallest eval in the suite. Workable BECAUSE ROUGE is continuous per
#   example rather than 0/1 — the CI lands near +/-1.3 points — but it is reported, not glossed.
#   And DialogSum appears in instruction-tuning collections, so an instruct base will have an
#   inflated zero-shot; that compresses the apparent delta rather than exaggerating it.
#
#   BERTScore is a REPORT-only metric and never runs in the loop: roberta-large on CPU, ~10 minutes
#   for the full split. `_l40s_task_body.sh` points `METRICS_VENV` at its venv.
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
export SLM_BENCHMARK_TASK=dialogsum
# TEACHER VERIFICATION OF GENERATED ROWS: SHADOW MODE — it runs, and it drops nothing.
#
# The pass is measured, not trusted. `enforce` has been measured doing net harm on this pipeline:
# on run 38832588 the teacher rejected 25 of 25 `ner_bc5cdr` rows that `verify_ner_row` had just
# accepted, twice running, and that STOPPED the run; on `calendar_json` it rejected 1,965 of 8,704
# (22.6%), including "7pm start plus 60 mins is 8pm, not 20:00" -- 8pm IS 20:00.
#
# This task has NO exact verifier at all (`synth_verifier=None`) — whether a summary is accurate
# and complete is a judgement, not a computation — so in principle the teacher pass is the only
# gate it could have. Shadow mode is still the right first move: it tells us the rejection rate
# and the reasons before we let those verdicts bind, and on a task with no programmatic backstop a
# false-rejection cascade would be unopposed. Grep `[verify:shadow]`.
#
# Non-interfering by construction: shadow mode deliberately does NOT feed rejection reasons into
# the next generation prompt, because that would change what the teacher generates and the
# measured rate would then describe a trajectory that only exists because we measured it.
export SLM_VERIFY_SYNTH=shadow
# SYNTHESIS GATE: bypassed deliberately, and the measurement is still taken honestly.
#
# The gate refuses `surgical_synthesis` when the teacher's own few-shot score on the task falls
# below threshold. This task's metric changed, so any prior measurement is void and a refusal
# would leave the orchestrator with mining and hyperparameters only.
#
# BYPASS changes only the DECISION. The teacher's measured score, the real gate and
# `bypassed=True` are all still recorded on the verdict, so the end-of-run report still says
# whether the data would have cleared the bar. It lives in the file rather than the submitting
# shell because that is how runs 38832588, 39294409 and 39361648 all became configurations nobody
# had written down.
export SLM_TEACHER_SYNTH_BYPASS=1

export TASK="Fine tune a small model to summarize spoken conversations in one to three sentences (DialogSum chat and meeting transcripts, scored against three human reference summaries), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
