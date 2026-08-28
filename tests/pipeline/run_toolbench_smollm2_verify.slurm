#!/bin/bash
#SBATCH --job-name=slm-toolbench-smollm2-verify
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=20:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-toolbench-smollm2-verify-%j.out
#
# VERIFICATION RUN for the `toolbench` task (added 2026-08-24), pinned to SmolLM2-360M-Instruct.
#
# Its job is to answer whether the NEW task works end to end inside the agent loop, not to produce a
# headline accuracy. Specifically, in order:
#   1. the loader returns complete ToolBench paths and all 765 ToolEval eval rows across six subsets;
#   2. the ToolEval judge answers, the 3-round majority vote resolves, and per-subset pass rates and
#      the query-count-weighted overall appear in the report;
#   3. `format_valid` and `undeclared_api_rate` are populated, so a low score can be attributed to
#      shape, to fabrication, or to genuine capability;
#   4. every data_rebuild ADDS rows — mining re-reads the ToolBench prefix deeper for free;
#   5. quality control removes a plausible number of rows rather than eating the curriculum.
#
# `agent/run_health.py` stops the run itself if any of those stops being true.
#
# WHY THIS MODEL: SmolLM2-360M-Instruct is the strongest sub-billion entry in the pool and the only
# one measured lossless under Q4_K_M on both existing structured-output tasks (`xlam_bfcl`
# ast_arg_match 0.4500). It is the pool's best chance of a NON-ZERO ToolBench pass rate, which is
# what makes it the useful half of this pair — see the gemma twin for the other half.
#
# EXPECTATION, recorded before the run so it can be checked against: ToolBench is materially harder
# than xlam_bfcl. A single call against a handful of declared tools becomes a whole multi-step path
# against 2-16 APIs drawn from ~16,000, terminated correctly, with a final answer a judge finds
# responsive. A low pass rate here is the expected outcome and is NOT by itself a bug. What would be
# a bug: format_valid at 0.0000 (the model never emits Action/Action Input, i.e. a prompt or
# chat-template problem), an empty eval set, or a judge that never returns a verdict.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=toolbench

# single_model: no escalation on failure, no downward regression on success. Every score change is
# attributable to a data or hyperparameter intervention rather than to a model swap, which is the
# whole point of a verification run — and here it also pins the comparison between the two models.
export SLM_MODEL_SELECTION_STRATEGY=single_model
export SLM_FORCE_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct@Q4_K_M"

# Stop on the FIRST goal rather than chasing a raised one. A stretch goal on a bring-up run turns a
# clean termination into more iterations of something already demonstrated.
export SLM_THRESHOLD_RAISE=0

# The three termination conditions, stated explicitly rather than inherited, because they are the
# contract this run is being judged against: the accuracy goal, OR 15 evals whose best gain is under
# 2%, OR 30 evals. Under single_model there is nothing to escalate to, so both bounds TERMINATE.
export SLM_STAGNATION_WINDOW=15
export SLM_STAGNATION_MIN_DELTA=0.02
export SLM_MAX_EVALS_BEFORE_ESCALATION=30

# Teacher-fitness override. ToolBench's metric is itself a JUDGED pass rate, so the teacher's fitness
# score is a judged number measured against a 0.80 gate calibrated on exact-match tasks — a bar no
# model is likely to clear on a multi-step agentic benchmark. Gating a first bring-up run on it would
# silently remove `surgical_synthesis` from the intervention menu, which is one of the two things
# this run exists to exercise. The measurement still happens and is still logged in full against the
# real gate; only the decision is overridden, and the verdict is stamped `bypassed`.
export SLM_TEACHER_SYNTH_BYPASS=1

# Wall clock inside the 20h allocation so the pipeline writes its own report rather than being killed
# mid-write by Slurm. ToolBench's eval is the suite's most expensive (765 generations plus ~2,295
# judge calls per pass), so this leaves ~2h of margin rather than the usual 1h.
export SLM_MAX_WALLCLOCK_S=64800

# Deliberately NOT requeued: a verification run that dies should stay dead so the failure is examined
# rather than silently restarted.
# CONTEXT. `_l40s_task_body.sh` defaults SLM_MAX_SEQ_LENGTH to 4096, and `task_max_seq_length`
# gives that env var precedence OVER the task spec — so without this line the spec's measured 8192 is
# silently clamped and the run dies. It did: job 38812203 aborted at eval with "4020 tokens, exceeding
# input budget 2560 ... inside configured max sequence length 4096" and then at train with "row 43
# contains 4110 tokens, exceeding configured context 4096".
#
# 8192 is the measured requirement, not a guess: real eval prompts run to a median 1,248 tokens and a
# p99 of 3,973, and whole-path targets to a p99 of 1,410. It is also the ceiling, because
# SmolLM2-360M's position limit is exactly 8192. The handful of rows that do not fit even that are
# dropped by the loader rather than truncated.
export SLM_MAX_SEQ_LENGTH=8192

# DEMONSTRATIONS: five-shot, as everywhere else. This used to be 0 for this task because five
# RANDOM full-prompt demonstrations are ~11,500 tokens against a teacher served at 8,192 and every
# request 400'd. Zero-shot was the wrong cure: synthesis then generated 636, 326 and 57 candidate
# rows across three rebuilds and the programmatic verifier kept ZERO of them each time — a
# format-bound task cannot be synthesized without showing the format (B276).
#
# `agent.teacher_fitness.fit_demonstrations` now picks the SHORTEST demonstrations that fit instead
# of sampling at random, and toolbench's shortest complete paths are ~809 tokens, so five of them
# fit inside 8,192 with room for the question. Five-shot is therefore back on by default and this
# task needs no override.

export TASK="Fine tune a small model for agentic multi-step tool calling on ToolBench (train on ToolLLM's DFSDT solution paths, evaluate with ToolEval pass rate over the six G1/G2/G3 test subsets), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
