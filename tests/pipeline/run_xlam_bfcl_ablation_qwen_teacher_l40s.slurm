#!/bin/bash
#SBATCH --job-name=slm-xlam-ablation-qwen
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-xlam-ablation-qwen-%j.out
#
# ============================================================================================
# ABLATION 2a of 3 — DOES THE TEACHER MODEL CHANGE THE QUALITY OF THE SYNTHETIC DATA?
#
# BASELINE: slm-xlam-bfcl-l40s-39361648 (1d13h, 5 tiers, ast_arg_match 0.8740 against a 0.8700
# goal; $38.05 of DeepSeek tokens over 18,167 calls). That run's teacher was deepseek-v4-flash.
# This one is the same task, same configs, with the LOCAL Qwen3.6-35B-A3B teacher instead.
#
# xLAM is the cheap half of this ablation and NER (run_ner_bc5cdr_ablation_deepseek_teacher) is
# the expensive half; together they swap the teacher in both directions, so a difference that
# shows up in only one of them is a task interaction rather than a teacher difference.
#
# NOTE THAT THIS FILE IS BARELY MODIFIED, because the committed xlam launcher was ALREADY the
# local-Qwen configuration. Run 39361648 became a DeepSeek run because SLM_SYNTH_API_MODE=1 was
# exported in the submitting shell and never written down — the same class of accident the
# SLM_TEACHER_SYNTH_BYPASS comment below documents. The `unset` is therefore load-bearing: sbatch
# exports the submitting environment by default, so without it this job would silently inherit
# API mode from whatever shell launched it and reproduce the baseline instead of ablating it.
unset SLM_SYNTH_API_MODE
#
# WHY THE GOAL IS PINNED. The accuracy goal is normally the teacher's OWN five-shot score on the
# frozen eval set, floored at 0.80 (agent/threshold.py::threshold_from_endpoint_baseline). That is
# the right default and it is fatal here: DeepSeek measured 0.8700 and set 39361648's bar, while
# Qwen3.6 has measured 0.8250-0.8650 on this task across runs 38708719, 38723935, 38732019,
# 38734202 and 38734723. Left alone, the weaker teacher would also set a LOWER bar, and this run
# could converge sooner while looking like the better result — a comparison of two different
# questions. Pinning both runs to 0.8700 makes synthetic-data quality the only thing that moves.
#
# The pin routes through agent/nodes/cold_start/task_analysis.py and skips CALIBRATION only. The
# five-shot teacher measurement still runs, still gates synthesis and is still reported, so the
# number this ablation is actually about is not lost.
export SLM_STOP_THRESHOLD=0.87
#
# GPU COUNT DIFFERS FROM THE BASELINE, UNAVOIDABLY: 2x L40S here against the baseline's 1x. A
# local teacher needs a card of its own (the auto-2gpu profile puts vLLM on GPU 0 and training on
# GPU 1), where an API teacher needs none. Training is pinned to a single exclusive L40S in both
# cases, so the student's throughput is unchanged — the second card serves the teacher and does
# not accelerate anything the baseline did on one.
# ============================================================================================
#
# CURATED BENCHMARK: xlam_bfcl (format-bound, metric ast_arg_match). SLM_BENCHMARK_TASK
# pins the deterministic loader (train on Salesforce xLAM-60k, evaluate on the Berkeley
# Function-Calling Leaderboard) instead of autonomous web_acquire — run.py starts the run
# NON-autonomous against this task's registry spec, which is where its loader, scorer, metric and
# token budget are declared. The datasets are pulled live from HuggingFace on the compute node
# (public; BFCL ships a loader script, trust_remote_code). Scoring is judge-free executable AST
# argument match.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=xlam_bfcl
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
# SYNTHESIS GATE: set for parity with the other two curated benchmarks, and expected to be INERT here.
#
# xlam_bfcl is the one task in this group whose teacher clears the bar on its own — measured
# ast_arg_match 0.8250-0.8650 5-shot across runs 38708719, 38723935, 38732019, 38734202 and 38734723,
# against a 0.80 gate. BYPASS only changes the decision when the gate was NOT cleared, so on a normal
# xlam run this flag does nothing and the verdict records a genuine pass with bypassed unset.
#
# It is set anyway so that a teacher that drifts below 0.80 does not silently turn synthesis off
# mid-experiment, which is exactly how calendar_json and ner_bc5cdr came to run without it.
export SLM_TEACHER_SYNTH_BYPASS=1
# TEACHER SERVED CONTEXT: 16384, not the 8192 default, because THIS task's rows do not fit 8192.
#
# This is the served context (a vLLM `--max-model-len` launch flag), not an output cap. The nine
# hardcoded `max_tokens` constants are already gone; `config/token_budget.output_budget` now sizes
# every reply and CLAMPS it to whatever the server was launched with. At 8192 that clamp was the
# binding constraint on run 39311800:
#
#   BadRequestError: 400 - maximum context length is 8192 tokens. However, you requested 4736
#   output tokens and your prompt contains at least 3457 input tokens, for a total of at least 8193.
#
# Over by ONE token: `output_budget` estimates the prompt at 4 chars/token and keeps a 64-token
# margin, and xlam's dense tool-schema JSON tokenizes worse than that, so the estimate ran short and
# the margin absorbed the error. Two of five synthesis batches were lost that way (0 of 25 and 4 of
# 126 kept, 368 of 515 rows overall) and the 5-shot fitness measurement was SKIPPED entirely — an
# 8126-char demonstration block does not fit beside a 12940-char prompt in a 7936-token input budget
# — so the synthesis gate fell back to zero-shot.
#
# 8192 is NOT a model limit: Qwen3.6-35B-A3B declares max_position_embeddings=262144. It is a memory
# trade: context costs KV cache on one L40S, so max-num-seqs is halved to pay for it, keeping the
# in-flight KV footprint (seqs x context) where the auto-2gpu profile already sized it.
#
# ALL THREE ARE LOCAL-vLLM SETTINGS AND ARE SKIPPED IN API MODE.
#     They describe a 35B model sharing one 48 GB L40S: weights take ~35 GB, leaving ~7 GB of KV
#     cache, and KV demand is context x concurrent sequences — so a wider context has to be paid for
#     by halving max-num-seqs. None of that is true of a hosted endpoint, where we rent tokens
#     rather than GPU memory.
#
#     Applying them anyway would actively hurt: `config.config` serves DeepSeek at 131072 by
#     default, and exporting 16384 would cap it to an eighth of that. The demonstration budget is
#     `SYNTH_MAX_MODEL_LEN * 1.25` characters, so the cap also shrinks the room for demonstrations
#     from ~164k chars to ~20k — and xlam is the task where too few demonstrations means ZERO kept
#     rows, not merely worse ones (0 of 1,019 at zero-shot, 0 of 133 at the 1-2 shots that fit an
#     8,192 context; B276). The local cap fixes a local problem and creates a remote one.
if [ "${SLM_SYNTH_API_MODE:-0}" != "1" ]; then
    export SLM_SYNTH_MAX_MODEL_LEN=16384
    export SLM_SYNTH_MAX_NUM_SEQS=32
    export SLM_SYNTH_CONCURRENCY=24
fi

export TASK="Fine tune a small model for function/tool calling (train on xLAM-60k, evaluate on the Berkeley Function-Calling Leaderboard), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
