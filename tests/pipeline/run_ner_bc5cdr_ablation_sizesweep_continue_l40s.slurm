#!/bin/bash
#SBATCH --job-name=slm-ner-bc5cdr-ablation-sizesweep-continue-l40s
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-ner-bc5cdr-ablation-sizesweep-continue-l40s-%j.out
#
# ============================================================================================
# CONTINUATION OF THE NAIVE SWEEP — the 9 variants job 40827150 never reached (added 2026-10-07)
#
# THIS IS NOT A NEW RUN. It resumes run 40827150's own thread, in its own run directory, with its
# curriculum, DAG, eval set, score history and lifetime best intact. `SLM_RUN_DIR` is pinned below,
# which is what makes `_durable_resume_ready` in `_l40s_task_body.sh` hand run.py `--resume`.
#
# WHY A REVIVAL WAS NEEDED FIRST. 40827150 did not crash and did not exhaust its pool: it trained
# 18 of 27 variants and then `_must_terminate` fired at `_graph_steps >= MAX_TURNS_MAIN` (1500), so
# `_route_after_evaluate` returned "terminate" and the graph routed to END. That left the LangGraph
# thread TERMINAL (`next_nodes == ()`), and LangGraph resumes an existing thread by passing `None`,
# which on a finished thread does nothing. Raising the budget alone would have produced a job that
# started, found a completed graph and exited.
#
# `scripts/revive_sweep.py` re-anchored the thread at its last real node (`evaluate`) with the
# budget already raised, so LangGraph recomputed that node's conditional edge and scheduled the
# successor the run would have taken had the budget not run out. It scheduled `rollback`, which is
# correct: the final eval scored 0.8448 against a 0.8586 best, so `should_rollback` is true. Nothing
# was rewritten — `update_state` appends.
#
# WHAT IT WILL DO. Resume at `rollback` on Qwen3.5-2B@Q8_0 (sweep position 18/27, 20 evals used of
# the 30-eval cap), finish that variant, then escalate through the nine the first job never reached,
# in on-disk size order:
#   19 Qwen3-1.7B@Q8_0        20 Qwen3-4B-Instruct-2507@Q4_K_M   21 Qwen3.5-4B@Q4_K_M
#   22 Qwen3-1.7B@bf16        23 Qwen3-4B-Instruct-2507@Q8_0     24 Qwen3.5-2B@bf16
#   25 Qwen3.5-4B@Q8_0        26 Qwen3-4B-Instruct-2507@bf16     27 Qwen3.5-4B@bf16
#
# THE BUDGET RAISE IS RECORDED, NOT HIDDEN. `MAX_TURNS_MAIN` is part of `runtime_config_snapshot`,
# so both `run-manifest.json` and `checkpoint.json` carried a `config_fingerprint` computed under
# 1500 and a resume under 6000 is refused by design — that guard exists so "budget exhausted"
# cannot be made unfalsifiable after the fact. Both files were re-stamped, the originals kept as
# `*.pre-revive.bak`, and the change written to `budget_raise_history` on the manifest. The new
# fingerprint was DERIVED from the stored effective config with only `MAX_TURNS_MAIN` changed
# (verified by reproducing the stored fingerprint from the stored config first), so the delta is
# provably one key wide; run.py re-checks it live against this launcher's environment at startup.
#
# 6000 is sized, not arbitrary: the first job averaged 4.27 graph steps per train->eval iteration,
# and 9 variants at the 30-eval cap is ~1,160 steps on top of the 1,500 already spent.
#
# WALL CLOCK: 5 days, not 7. The cluster holds an `October13_Maintenance` reservation over every
# g31xx L40S node from 2026-10-13 09:00, and SLURM will not start a job whose wall clock runs into
# it — a 7-day request is unschedulable today. Nine large variants at the first job's measured
# ~20 min/iteration (more for the 4B tiers) is an estimated 3-5 days, so 5 days is the honest ask.
# If it overruns, USR1 fires two hours early, the segment checkpoints and requeues, and the run
# continues after maintenance exactly as this resume does.
#
# EVERY OTHER SETTING IS 40827150'S, UNCHANGED, because run.py compares this launcher's live
# `runtime_config_snapshot` against the manifest and refuses any drift beyond the budget.
# ============================================================================================
export SLM_RUN_DIR=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/runs/slm-ner-bc5cdr-ablation-sizesweep-l40s-40827150

export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=ner_bc5cdr
export SLM_VERIFY_SYNTH=0
export SLM_TEACHER_SYNTH_BYPASS=1

export SLM_MODEL_SELECTION_STRATEGY=size_sweep
export SLM_STOP_THRESHOLD=0.99
export SLM_MAX_TURNS_MAIN=6000
unset SLM_SYNTH_API_MODE
# PINNED TO THE VALUES 40827150 RECORDED, not changed. The first submission of this launcher (job
# 41517188) died at startup on `effective config drift: SLM_MNN_HQQ: stored='0', expected='';
# SLM_MNN_QUANT_BLOCK: stored='64', expected=''` — the run started while the working tree defaulted
# these to 0/64, and commit 7f0a9a5 later moved the defaults to '' (HQQ+32 on 4-bit). The run
# quantizes through llama.cpp (SLM_QUANT_BACKEND=llama_cpp), so neither knob touches its weights;
# exporting the stored values just makes the live snapshot equal the manifest again.
export SLM_MNN_HQQ=0
export SLM_MNN_QUANT_BLOCK=64

export TASK="Fine tune a small model for biomedical named entity recognition on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
