#!/bin/bash
#SBATCH --job-name=slm-ner-ablation-deepseek-cse
#SBATCH --account=gpu-l40s-cse
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-ner-ablation-deepseek-cse-%j.out
#
# ============================================================================================
# ABLATION 2b of 3 — DOES THE TEACHER MODEL CHANGE THE QUALITY OF THE SYNTHETIC DATA?
#
# BASELINE: slm-ner-bc5cdr-l40s-39311801 (8h46m, 3 tiers, span_f1 0.8532), whose teacher was the
# local Qwen3.6-35B-A3B. This run swaps in deepseek-v4-flash and changes nothing else.
#
# NER IS THE HIGH-SIGNAL HALF of this ablation: it is where synthetic data bought the most across
# the whole benchmark suite. Every one of the baseline's three tiers opened on a
# surgical_synthesis step, and of the 27 rebuilds the kept ones were mostly synthesis — so if
# teacher identity matters anywhere, it matters here. Its partner is
# run_xlam_bfcl_ablation_qwen_teacher, which swaps the same two teachers the other way round.
#
# ONE GPU, NOT TWO. In API mode no vLLM server is launched, so the card the local teacher
# occupied is not needed and _l40s_task_body.sh's api-teacher profile hands the whole allocation
# to training. Training was already pinned to one exclusive L40S in the baseline, so the student
# sees identical hardware.
export SLM_SYNTH_API_MODE=1
export SLM_SYNTH_API_MODEL=deepseek-v4-flash
#
# WHY THE GOAL IS PINNED, even though it probably would not move. The goal is the teacher's own
# five-shot score floored at 0.80, and Qwen3.6 measured 0.3348 on this eval set — far below the
# floor, so the FLOOR set the baseline's goal at 0.8000 (the orchestrator later raised it to
# 0.8500 of its own accord). DeepSeek is expected to land under 0.80 as well, in which case the
# floor wins again and this pin is inert. It is set anyway because "expected" is not "measured":
# if DeepSeek came in at, say, 0.86, it would set its own 0.86 bar and this run would be chasing
# a target the baseline never had. Pinning costs nothing when the floor wins and saves the
# comparison when it does not.
export SLM_STOP_THRESHOLD=0.80
#
# COST. Budget on the order of $15-30 of DeepSeek tokens. The xLAM API run spent $38.05 over
# 18,167 calls; this task made a comparable 17,831 teacher calls but its rows are PubMed
# sentences rather than xLAM's dense tool-schema JSON, so the per-call token count is far lower.
#
# SCORES REMAIN COMPARABLE TO THE BASELINE. API mode also switches the eval JUDGE to DeepSeek,
# and judge scores are cached per model — but ner_bc5cdr is scored by `span_f1`, which is exact
# and judge-free, so nothing about the measurement changes. (This would NOT hold for dialogsum or
# toolbench, which are judged.) Generated rows are labelled `synth:deepseek` rather than
# `synth:vllm`, which is exactly the provenance distinction this ablation wants recorded.
# ============================================================================================
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
# Dedicated gpu-l40s-intelligentsystems quota: 1x L40S (api-teacher-1gpu profile — see the
# ablation header above for why this is one card and not the baseline's two), seven-day wall.
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
#
# KEPT FOR THIS ABLATION, and it is doing real work rather than riding along. The paragraphs above
# describe Qwen3.6; the teacher here is DeepSeek, which is also expected to miss the 0.80 gate on
# span_f1. The baseline generated rows ONLY because of this override, so an ablation that dropped
# it would compare a run with synthetic data against a run with none — which is ablation 3's
# question (run_ner_bc5cdr_ablation_nosynth), not this one's.
export SLM_TEACHER_SYNTH_BYPASS=1

export TASK="Fine tune a small model for biomedical named entity recognition on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
