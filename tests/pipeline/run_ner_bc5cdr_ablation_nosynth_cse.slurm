#!/bin/bash
#SBATCH --job-name=slm-ner-ablation-nosynth-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-ner-ablation-nosynth-cse-%j.out
#
# ============================================================================================
# ABLATION 3 of 3 — DOES SYNTHETIC DATA ACTUALLY HELP, OR IS IT JUST EXPENSIVE?
#
# BASELINE: slm-ner-bc5cdr-l40s-39311801 (8h46m, 3 tiers, span_f1 0.8532, converged). Same task,
# same configs, same teacher — but the orchestrator may not generate a single row. Mining real
# rows and hyperparameter tuning are its whole action space.
#
# WHY THIS TASK. NER is where synthetic data appeared to earn its keep: all three of the
# baseline's tiers opened on a surgical_synthesis step, and the run's reported gain traces back
# to one. If mining and hyperparameters alone reach ~0.8532, that gain was attribution rather
# than causation and the pipeline's most expensive stage is not paying for itself.
export SLM_SYNTH_DISALLOW=1
#
# HOW IT IS ENFORCED, and why not simply by dropping the bypass. Dropping SLM_TEACHER_SYNTH_BYPASS
# would work TODAY — the teacher measured 0.3348 against a 0.80 gate, so the gate refuses
# synthesis on its own — but it makes the experiment depend on the teacher continuing to score
# badly, and it would leave the transcript claiming the teacher's output "would be labelled
# noise" as the reason. SLM_SYNTH_DISALLOW forces `synthesis_allowed` false ahead of both the
# verdict and the bypass, and stamps `disallowed_by_operator` on the verdict so the run reports an
# operator refusal instead of a failed gate. The bypass is removed below as well, so the two
# cannot appear to contradict each other in the log.
#
# THE TEACHER SERVER STILL RUNS, and still needs its own card — hence 2x L40S, unchanged from the
# baseline. Its five-shot measurement is the SOLE source of the run's accuracy goal
# (agent/nodes/cold_start/eval_setup.py::_calibrate_qwen_goal_if_pending raises
# QwenBaselineUnavailableError without it), so a run with no teacher would have no bar to clear.
# It is measured, reported and then not asked to generate anything. No goal pin is needed: the
# same model on the same eval set lands under the 0.80 floor, exactly as in the baseline.
#
# NO CoT EITHER, so this really is gold-plus-mined-real and nothing else: tasks/ner_bc5cdr.py
# declares cot_annotation=False, so the teacher never writes reasoning onto real rows on this
# task. There is no back door through which generated text reaches the curriculum.
#
# EXPECT A SHORTER RUN THAN 8h46m, AND EXPECT MINING TO DIE EARLY. The checksummed local bundle
# holds 5,096 train rows and eval_setup already took 5,000, so `_reread_known_sources` can add at
# most ~96 before the source is exhausted and mining falls through to Exa web discovery. Once
# that spends its allowance, `data_rebuild_available` goes false and the orchestrator is confined
# to hyperparameters until it stagnates, escalates, and eventually runs out of tiers. That
# trajectory is the finding, not a malfunction: it is what this pipeline looks like with the
# synthesis stage removed.
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
# SYNTHESIS GATE: the bypass the baseline sets is DELIBERATELY ABSENT here, and is unset rather
# than merely omitted.
#
# The baseline exports SLM_TEACHER_SYNTH_BYPASS=1 because the local teacher measures well under
# the 0.80 gate on this task (0.3348 on run 39311801) and would otherwise never generate a row.
# That override is the thing this ablation is removing, so leaving it set would leave the log
# saying synthesis was both refused by the operator and allowed by an override.
#
# `unset`, because sbatch exports the submitting shell's environment by default and this flag has
# a documented history of living there rather than in a script: runs 38832588 and 39294409 only
# ever generated a row because it happened to be exported in the shell that launched them. An
# ablation that inherits it silently becomes a different experiment.
#
# SLM_SYNTH_DISALLOW=1 above is what actually enforces the refusal, so this line is belt-and-
# braces — the disallow is checked ahead of the bypass either way.
unset SLM_TEACHER_SYNTH_BYPASS

export TASK="Fine tune a small model for biomedical named entity recognition on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
