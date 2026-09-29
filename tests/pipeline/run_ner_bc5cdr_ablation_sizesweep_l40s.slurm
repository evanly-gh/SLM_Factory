#!/bin/bash
#SBATCH --job-name=slm-ner-bc5cdr-ablation-sizesweep-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-ner-bc5cdr-ablation-sizesweep-l40s-%j.out
#
# ============================================================================================
# ABLATION — WHAT DOES THE TIERING SYSTEM SAVE? (naive exhaustive sweep, added 2026-09-28)
#
# BASELINE: slm-ner-bc5cdr-l40s-39311801 (8h46m, 55 iterations, 3 models, span_f1 0.8532,
# converged on tier 3 Qwen3.5-2B@Q4_K_M). Same task, same data, same teacher, same per-model
# budget — but no tiers and no orchestrator choice of model. The run starts on the smallest
# variant in the pool (SmolLM2-135M@Q4_K_M, 101MB) and every escalation moves to the NEXT-LARGER
# VARIANT by on-disk size, through all 27 (9 base models x Q4_K_M / Q8_0 / bf16), ending on
# Qwen3.5-4B@bf16 (8.9GB). `agent/nodes/cold_start/model_selection/size_sweep.py`.
#
# TWO NUMBERS TO READ OFF IT, against the baseline:
#   * wall clock — the baseline trained 3 of 27 variants; this trains all of them;
#   * best span_f1 across the sweep — the ceiling the ladder could have found had it not skipped
#     the other 24. Expected slightly above 0.8532 at far greater cost.
# Also read from the per-iteration `Score:` lines: the elapsed time at which the sweep FIRST
# crosses the baseline's 0.80 goal and its 0.8532 endpoint, which is the like-for-like
# time-to-goal comparison.
#
# THE PIN. SLM_STOP_THRESHOLD=0.99 (THRESHOLD_CEILING) makes convergence unreachable, so every
# variant runs to the same terminal rule the baseline's non-final tiers ran to — stagnation over
# SLM_STAGNATION_WINDOW=15 evals or MAX_EVALS_BEFORE_ESCALATION=30 — and the run ends only when
# the sweep is exhausted. Left calibrated, the sweep would stop at the first variant to touch 0.80
# and never visit the rest. The cost, as on the two no-convergence reruns: the orchestrator's
# intervention decisions see a goal of 0.99 where the baseline's saw 0.80 → 0.85.
#
# EXPECT ~3-4 DAYS. ~20 evals per variant x 27 variants at 7-20 min an iteration (the 4B bf16
# variants dominate). The seven-day wall plus the USR1 checkpoint-and-requeue contract covers an
# overrun. Claude spend is roughly one intervention decision per iteration, ~$15-25 in total.
#
# CONFOUND, stated rather than removed: the curriculum carries forward across all 27 models, as it
# does across the baseline's 3, so later variants train on data grown by every variant before them.
#
# `unset SLM_SYNTH_API_MODE` keeps the local Qwen teacher the baseline used: sbatch forwards the
# submitting shell's environment, which is how run 39361648 silently became a DeepSeek run.
# ============================================================================================
#
# CURATED BENCHMARK: ner_bc5cdr (format-bound, metric span_f1). Promoted 2026-08-13 from the
# autonomous path, where it reached BC5CDR only because the orchestrator happened to emit a plan
# web_acquire could resolve — not reproducible. SLM_BENCHMARK_TASK pins the loader.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (auto-2gpu profile), seven-day wall.
# --requeue + USR1 remain a safety net against node failure, not a wall-clock mechanism.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=ner_bc5cdr
# TEACHER VERIFICATION OF GENERATED ROWS: OFF for this task — see run_ner_bc5cdr_l40s.slurm.
export SLM_VERIFY_SYNTH=0
# SYNTHESIS GATE: bypassed deliberately, as in the baseline — see run_ner_bc5cdr_l40s.slurm.
export SLM_TEACHER_SYNTH_BYPASS=1

export SLM_MODEL_SELECTION_STRATEGY=size_sweep
export SLM_STOP_THRESHOLD=0.99
unset SLM_SYNTH_API_MODE

export TASK="Fine tune a small model for biomedical named entity recognition on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
