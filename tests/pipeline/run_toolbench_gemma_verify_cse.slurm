#!/bin/bash
#SBATCH --job-name=slm-toolbench-gemma-verify-cse
#SBATCH --account=gpu-l40s-cse
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=20:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-toolbench-gemma-verify-cse-%j.out
#
# VERIFICATION RUN for the `toolbench` task (added 2026-08-24), pinned to gemma-3-270m-it — the
# smallest-tier Gemma entry and the twin of run_toolbench_smollm2_verify.slurm. Same bounds, same
# overrides, one model changed, so the pair isolates model capability from everything else.
#
# See the SmolLM2 twin for what the run must demonstrate; `agent/run_health.py` enforces it.
#
# WHY THIS MODEL, AND WHAT TO EXPECT: gemma-3-270m-it is Tier 1 (241 MB at Q4_K_M) and is measured
# near the FLOOR on function calling — `xlam_bfcl` ast_arg_match 0.1000 [bf16 0.0567] against
# SmolLM2-360M's 0.4500. Its shape explains the reading: 168M of its 270M parameters are a 262k-token
# embedding table, leaving only ~102M of transformer, and `config/model_capabilities.md` records that
# thin transformers appear to hurt on compositional tool calls.
#
# So a ToolBench pass rate at or near ZERO is the PREDICTED result for this model, not a defect, and
# this run's value is largely as the negative control: it says whether a floor score on the new task
# looks like a floor score (paths emitted in the right shape, judged unsolved) or like a broken
# harness (nothing parses, no eval rows, no judge verdicts). Read `format_valid` first, then
# `undeclared_api_rate`, then the pass rate — in that order.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=toolbench

# single_model: no escalation on failure, no downward regression on success. Without it a model
# measured near the floor would escalate out of Tier 1 on its first stagnation and this would stop
# being a run about gemma-3-270m-it.
export SLM_MODEL_SELECTION_STRATEGY=single_model
export SLM_FORCE_MODEL="google/gemma-3-270m-it@Q4_K_M"

export SLM_THRESHOLD_RAISE=0

# The accuracy goal, OR 15 evals whose best gain is under 2%, OR 30 evals. Under single_model both
# bounds TERMINATE rather than escalate. On a model expected to sit at the floor, the 15-eval
# stagnation window is the one that should fire.
export SLM_STAGNATION_WINDOW=15
export SLM_STAGNATION_MIN_DELTA=0.02
export SLM_MAX_EVALS_BEFORE_ESCALATION=30

# See the SmolLM2 twin for the full reasoning: ToolBench's teacher-fitness score is itself a judged
# number against a gate calibrated on exact-match tasks, and gating a bring-up run on it would remove
# `surgical_synthesis` from the intervention menu. Measured and logged in full; decision overridden.
export SLM_TEACHER_SYNTH_BYPASS=1

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
