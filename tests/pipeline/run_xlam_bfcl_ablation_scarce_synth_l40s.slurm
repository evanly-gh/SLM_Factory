#!/bin/bash
#SBATCH --job-name=slm-xlam-bfcl-ablation-scarce-synth-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-xlam-bfcl-ablation-scarce-synth-l40s-%j.out
#
# CURATED BENCHMARK: xlam_bfcl (train on xLAM-60k, evaluate on BFCL, metric ast_arg_match).
# SLM_BENCHMARK_TASK pins the deterministic loader instead of autonomous web_acquire.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=xlam_bfcl

# ─────────────────────────────────────────────────────────────────────────────────────────────
# ABLATION 5 ON XLAM_BFCL, arm B: DOES SYNTHETIC DATA EARN ITS KEEP WHEN REAL DATA IS SCARCE?
#
# THE THIRD PAIR, AND THE ONE THAT TESTS THE HYPOTHESIS MOST DIRECTLY. sms_spam showed synthesis
# worth +0.2981 on a 360M model and nothing measurable above it; clinc150 showed it worth +0.013 to
# +0.059 at every tier. 09-17-scarcity-followups.md reads those as "synthetic data substitutes for
# one to two tiers of model scale at the small end". xlam_bfcl is where that claim is cheapest to
# falsify: its tier-1 gemma-3-270m reached 0.0460 with FORMAT VALIDITY 0.3730 — two thirds of its
# outputs were not parseable at all — while its 4B tiers already score ~0.85 zero-shot. If
# synthesis cannot rescue a model that is failing on the output contract itself, the claim is
# narrower than stated.
#
# WHAT SCARCITY MEANS HERE IS DIFFERENT, AND IT MAKES THE PAIR CLEANER. clinc150 needed one row per
# intent because its label space is closed. xlam has no such constraint: tools are supplied in the
# prompt, and even the FULL 4,954-row anchor shares just 23 of the eval set's 823 tools (2.8%). The
# model was never memorising APIs, it was learning the call format, so 100 rows is scarce in the
# same way at any cap — and 100 matches the sms_spam pair exactly, which makes the three pairs
# comparable at one fixed budget.
export SLM_ABLATION_TRAIN_CAP=100
export SLM_ABLATION_DISALLOW_MINING=1
# THE TEACHER IS THE LOCAL QWEN, AND THE `unset` IS LOAD-BEARING. sbatch exports the submitting
# shell by default, and that is exactly how run 39361648 silently became a DeepSeek run. Qwen3.6
# measures 0.8660 zero-shot on this task against DeepSeek's 0.8700, so it is not the weaker teacher
# here — and it is the only one of the two whose generated rows are reproducible, because vLLM
# honours a per-request seed (training/determinism.py) and the DeepSeek API has no seed parameter.
# That matters more for this arm than for arm A: the curriculum IS the experiment here.
unset SLM_SYNTH_API_MODE
# WHY THE GOAL IS PINNED AT THE CEILING. xlam is the one suite task that CONVERGED on its own terms
# (0.8740 against a 0.8700 teacher-derived bar), so left alone it would stop at tier 5 after a
# handful of iterations and the arms could differ by run length rather than by synthesis. 0.99 is
# THRESHOLD_CEILING, unreachable for both, so each climbs the full ladder. It also sidesteps the
# usual teacher-bar problem: with the goal pinned, it no longer matters that Qwen and DeepSeek
# would set different bars.
export SLM_STOP_THRESHOLD=0.99
# The 500-row curriculum floor has to move for a run that is 100 rows by construction; 50 leaves
# room for quality control to drop a few without tripping it.
export SLM_MIN_CURRICULUM_ROWS=50
# Early stopping is left ON, unlike the clinc150 pair. There it had to go because one row per intent
# meant a random 12% validation split would strip 18 intents of their only example. xlam has no
# class structure to strip, so the default stands and this pair matches the sms_spam one.
#
# ARM B SPECIFIC: synthesis allowed and guaranteed to run. The bypass is inherited from the baseline
# launcher and kept, because the gate is re-measured every run and a measurement landing below 0.80
# would silently turn this arm into a duplicate of arm A.
#
# SHADOW VERIFICATION, WHERE THE BASELINE RUNS WITH VERIFICATION OFF. The baseline disables the
# teacher-as-verifier pass because it false-rejects function-call rows (calendar_json lost 22.6% of
# good rows that way, including "7pm plus 60 mins is 8pm, not 20:00"). Shadow mode has neither
# problem: it RECORDS the teacher's opinion of every generated row and drops nothing, which is what
# made the null results in the other two pairs interpretable — on sms_spam it came back N/N across
# 40 rebuilds, so that arm's flat tiers could be blamed on the models rather than the rows. The
# exact `verify_function_call_row` check runs either way.
export SLM_TEACHER_SYNTH_BYPASS=1
export SLM_VERIFY_SYNTH=shadow

if [ "${SLM_SYNTH_API_MODE:-0}" != "1" ]; then
    export SLM_SYNTH_MAX_MODEL_LEN=16384
    export SLM_SYNTH_MAX_NUM_SEQS=32
    export SLM_SYNTH_CONCURRENCY=24
fi
export TASK="Fine tune a small model for function/tool calling (train on xLAM-60k, evaluate on the Berkeley Function-Calling Leaderboard), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
