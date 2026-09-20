#!/bin/bash
#SBATCH --job-name=slm-clinc150-mnn-verify
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-clinc150-mnn-verify-%j.out
#
# VERIFICATION RUN for the MNN quantization backend inside the FULL agent loop (added 2026-09-17).
#
# `verify_mnn_backend.slurm` already answers "does an MNN artifact score what the GGUF scores?" on
# fixed weights and fixed rows. What it cannot answer is whether the loop ROUTES to MNN everywhere
# it needs to, because it never runs the loop. That is what this is for, and the things it has to
# demonstrate are all plumbing rather than accuracy:
#   1. the zero-shot baseline is measured on an MNN artifact, not a GGUF and not BF16;
#   2. each iteration's trained adapter is merged, exported, bit-width-verified and scored on MNN;
#   3. the artifact cache hits per (weights_ref, quant) and the reaper deletes MNN DIRECTORIES
#      rather than failing on a path that is not a file;
#   4. an ESCALATION rebuilds at the new tier's quant — the path that silently reused the wrong
#      file under llama.cpp (B161) and has to be right for a second artifact shape too;
#   5. the DAG records `quant_backend` on every node, so the scores cannot later be pooled with
#      llama.cpp numbers from the same task.
#
# BOUNDED ON PURPOSE. This is a plumbing proof, not a results run, and it must not spend a results
# run's Anthropic budget to produce one. Three bounds, in the order they will fire:
#   * SLM_MAX_EVALS_BEFORE_ESCALATION=3 — escalate quickly, so (4) is exercised early instead of
#     after a day of hyperparameter search at tier 1.
#   * SLM_MAX_WALLCLOCK_S=9000 — iterate_node forces a CLEAN terminal state at 2.5h, which writes
#     the summary, the DAG and the report rather than being killed mid-write.
#   * the 6h allocation, as the backstop if the pipeline somehow outlives its own guard.
# Expected spend at Sonnet rates: a few iterations' worth of intervention decisions, well under $1.
#
# WHAT A PASS LOOKS LIKE: every `── STEP 1/2: QUANTIZE → ... via MNN ──` line followed by a
# `MNN artifact built and load-validated` line, scores in the same range the llama.cpp baseline for
# this task reached at the same tiers (SmolLM2-360M @ Q4_K_M reached 0.80 macro_f1 in run 40175895),
# and at least one `Reaped non-best MNN artifact` plus one tier change. A FAIL is a
# QuantizationInfrastructureError, a score of 0.0000 on a tier where llama.cpp scored 0.3+, or any
# `model-q4_k_m.gguf` in the log — the last would mean the flag did not route.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=clinc150

# THE POINT OF THE RUN. Read by `config.config`, by `training/quant_backend.py` (which resolves it
# from the environment so disposable CUDA workers inherit it), and re-exported by `run.py` after
# argparse validates it.
export SLM_QUANT_BACKEND=mnn

# Escalate after two evals rather than thirty. Under the ladder this is what forces requirement
# (4) to happen inside the time box; the thing being tested is the artifact rebuild at the new
# tier, not how long tier 1 can be tuned.
export SLM_MAX_EVALS_BEFORE_ESCALATION=2
export SLM_STAGNATION_WINDOW=2
export SLM_STAGNATION_MIN_DELTA=0.02

# 600 gold rows instead of the task's 5,000. Measured on the first attempt (job 40259455): the
# full curriculum trains at ~50 min/epoch for three epochs, and with mid-training validation every
# 20 steps that is ~2.5h for ONE iteration — the whole time box spent before the first MNN artifact
# is even built. A plumbing proof does not need a good model, it needs several iterations and a
# tier change, and 600 rows still trains a real adapter (the loss was already under 0.2 at 0.4
# epochs on the full set). The EVAL set is untouched at its frozen 1,000 rows, so the scores here
# remain comparable to this task's other runs; only the curriculum is small.
#
# 600 rather than a rounder 400 because `curate_node::_enforce_curriculum_floor` refuses a
# curriculum under 500 rows — correctly, since for a real run that means the loader or the quality
# controls ate the data. Staying above the floor keeps this run honest against that check instead
# of overriding it with SLM_MIN_CURRICULUM_ROWS.
export SLM_ABLATION_TRAIN_CAP=600

# Stop at the first goal rather than chasing a raised one: a stretch goal on a plumbing run buys
# more iterations of something already demonstrated.
export SLM_THRESHOLD_RAISE=0

# Clean self-termination at 4h, inside the 6h allocation. The margin is large because BOTH MNN
# stages are slower than their llama.cpp counterparts, and job 40258512 measured how much:
#   * export: llmexport traces the model to ONNX and MNNConvert rewrites the graph — 105s for a
#     merged 360M checkpoint against 24s for convert_hf_to_gguf + llama-quantize, and every
#     iteration pays it once.
#   * eval: MNN decodes on CPU where the GGUF path offloads to the GPU — 0.46 rows/s against
#     150 rows in 9s, i.e. ~36 minutes for a 1,000-row clinc150 pass at the 360M tier and ~15 at
#     the 135M tier. Prefill dominates (2,016ms for a 1,376-token prompt against 11ms of decode
#     for a two-token label), because a CLINC150 prompt carries all 151 intent names.
# Four and a half hours therefore buys roughly: teacher warmup, two tier-1 iterations (each a
# baseline-or-trained MNN build plus a 1,000-row CPU eval), the escalation, and one iteration at
# the new tier — which is exactly the list this run has to demonstrate.
export SLM_MAX_WALLCLOCK_S=16200

# 8, NOT the 14 this run originally used. MNN's thread count is a correctness setting: at 14 the
# escalated `Qwen/Qwen3.5-0.8B@Q4_K_M` artifact decoded `%+!!!!!!!!!!` on all 1,000 eval rows and
# scored 0.0000, while the SAME artifact at 1/2/4/8/10/12/13/16 threads answered them correctly.
# The build now cross-checks the configured count against a known-good one and refuses the
# artifact on a mismatch (`quantize_mnn._assert_threaded_compute_is_sound`), so this would fail
# fast rather than silently — but a verification run should not be the thing that trips it.
export SLM_MNN_THREADS=8

# Deliberately NOT requeued: a verification run that dies should stay dead so the failure is
# examined rather than silently restarted.
export TASK="Fine tune a small model for user-intent classification (CLINC150: 150 intents plus out-of-scope detection), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
