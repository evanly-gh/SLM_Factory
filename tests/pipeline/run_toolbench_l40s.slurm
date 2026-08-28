#!/bin/bash
#SBATCH --job-name=slm-toolbench-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-toolbench-l40s-%j.out
#
# CURATED BENCHMARK: toolbench (format-bound, metric tooleval_pass_rate). Reimplements the
# evaluation of arXiv:2512.15943 ("Small Language Models for Efficient Agentic Tool Calling"):
# train on ToolLLM's ToolBench DFSDT corpus, score with ToolEval pass rate over its six named test
# subsets. SLM_BENCHMARK_TASK pins the deterministic loader instead of autonomous web_acquire.
#
# DATA CONTRACT (validated live 2026-08-24):
#   - TRAIN: Yhyu13/ToolBench_toolllama_G123_dfs / toolllama_G123_dfs_train.json. Ungated. It is a
#     single 2,002,419,658-byte JSON array (187,542 rows — the paper's training size exactly), and
#     the canonical ToolBench/ToolBench repo 404s. The loader reads a growing byte PREFIX over HTTP
#     range (verified HTTP 206; ~51 MiB for 5,000 rows) and caches it under
#     $SLM_TOOLBENCH_PREFIX_CACHE, defaulting inside HF_HOME. The full 2 GB is never downloaded.
#   - EVAL: StableToolBench's solvable_queries/test_instruction from raw.githubusercontent.com —
#     the six ToolEval subsets with api_list inline, filtered for solvability (765 queries:
#     163/153/158/106/124/61). Solvability filtering is load-bearing for a pass-rate metric: an
#     unsolvable query caps the ceiling below 1.0 and reports a benchmark defect as a model failure.
#   - TOOL ENV: stabletoolbench/ToolEnv2404 (11.7 MB) for each tool's prose description, which the
#     test files omit and ToolBench's system prompt includes. Without it every eval prompt would
#     differ from every train prompt in the largest block of the prompt.
#   Two live network dependencies beyond the Hub (github raw). Both are checked at eval_setup and
#   the run refuses to start on an empty held-out set rather than proceeding.
#
# JUDGE: this is the suite's SECOND judged task (dialogsum is the other) and the only judged
# structured-output one. ToolEval pass rate is a majority vote over 3 independent assessments per
# query by the run's own Qwen3.6 teacher, so budget ~3x765 judge calls per eval — cached on disk by
# (query, answer, round), so a re-eval of unchanged predictions is free. needs_judge=True, so a
# judge outage stops the run instead of scoring every query unsolved.
#
# KNOWN CAVEAT, stated here because it bounds what the number means: there is no API server, so the
# model plans a whole path in one generation with no observations fed back, and pass rate therefore
# partly measures writing a plausible final answer. Watch per_class.undeclared_api_rate — it is the
# direct measure of fabrication. See docs/Evan's Notes/08-24-toolbench-tooleval-harness.md.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth/judge server on GPU0, pipeline on
# GPU1 — the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a
# safety net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=toolbench
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

# EVAL SET: 500 rows instead of the task's 1,000 cap.
#
# This task's eval is the suite's most expensive: measured 41 min per pass, 8 passes = 5.5 h of a
# 20.8 h run. The cost is `rows x max_new_tokens` sequential decode steps, so it can be cut from
# either side — and cutting ROWS is the safe side. Lowering max_new_tokens would clip the tail of a
# long solution path, which is exactly the part being graded, so a cheaper run would come out of the
# result. Halving the rows costs only variance.
#
# The loader interleaves the six ToolEval subsets ROUND-ROBIN before this cap applies, so 500 rows
# keeps all six present in their original proportions rather than dropping G3 off the end.
export SLM_EVAL_SIZE_CAP=500

# SYNTHESIS GATE: overridden, deliberately, and this run's numbers are not clean because of it.
#
# The gate asks whether the teacher can do the task before letting it label data. Toolbench's teacher
# measured tooleval_pass_rate=0.0050 against a 0.80 gate, so synthesis is refused on the merits. It is
# forced on here because the QUESTION this run exists to answer is whether `fit_demonstrations` made
# toolbench synthesis produce anything at all — the previous run kept 0 rows from 1,019 attempts — and
# that cannot be observed with synthesis switched off.
#
# Set explicitly rather than inherited from the submitting shell. Run 38832586 had this flag on from
# the environment, which is why its log carries the override warning while neither launcher mentioned
# it: the configuration lived in a terminal instead of in the repo, and was therefore unreproducible.
#
# Every generated row is still checked by the exact verifier. Do not report this run's accuracy as a
# clean result.
export SLM_TEACHER_SYNTH_BYPASS=1

# TEACHER CONTEXT: 16384, not the body's 8192. This is what makes synthesis possible on this task.
#
# MEASURED on run 38985393's own curriculum artifact (4,995 rows). A demonstration is a COMPLETE
# prompt, and on toolbench that prompt carries the whole callable API list, which IS most of it:
#
#   cheapest demonstration  3,383 chars     median  8,582 chars
#   five cheapest           17,814 chars = 7,126 tokens
#   demo budget at 8192     10,240 chars = 4,096 tokens   -> only 2 of 5 demonstrations fit
#   five demos + a question + the 1,536-token output reserve = ~12,100 tokens
#
# So `fit_demonstrations` did its job and still could not reach five, because shortest-first cannot
# shrink an API list. Run 38985393 sent 1-2 shot prompts and the exact verifier kept 0 of 28, 0 of 25
# and 0 of 80 rows — the same shutout as the 0-shot run before it (B276): a format-bound task cannot
# be synthesized from a prompt that does not show the format enough times.
#
# 16384 leaves the five cheapest demonstrations (7,126 tok) beside a median question (3,433 tok) and
# the output reserve with room to spare. It is not a model limit: Qwen3.6-35B-A3B declares
# max_position_embeddings=262144.
export SLM_SYNTH_MAX_MODEL_LEN=16384

# Concurrency, lowered to pay for the context above. This is division, not preference.
#
# vLLM reports its own budget on startup, and on this hardware (one L40S, 35B fp8 weights, util 0.90)
# it is:
#     GPU KV cache size: 194,373 tokens
#     Maximum concurrency for 8,192 tokens per request: 23.73x
#
# Concurrency IS that budget divided by max-model-len, so raising the context spends concurrency
# one-for-one:  8,192 -> 23.7x   16,384 -> 11.9x   32,768 -> 5.9x   131,072 -> 1.5x
#
# 11, therefore, not the profile default of 64 and not the 16 this line said first: 11 x 16,384 =
# 180,224 tokens, which is the largest whole number of full-length sequences that fits 194,373. At 16
# the worst case is 262,144 and vLLM would admit batches it cannot hold, then preempt and recompute
# them — slower than the smaller batch it should have scheduled.
export SLM_SYNTH_MAX_NUM_SEQS=11

export TASK="Fine tune a small model for agentic multi-step tool calling on ToolBench (train on ToolLLM's DFSDT solution paths, evaluate with ToolEval pass rate over the six G1/G2/G3 test subsets), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
