#!/bin/bash
#SBATCH --job-name=slm-routerbench-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-routerbench-cse-%j.out
#
# CURATED BENCHMARK: routerbench (out-of-distribution, metric minority_f1).
#
# The task: given a prompt, decide whether an on-device model can answer it (`local`) or it must
# be escalated to a larger cloud model (`route`).
#
# DATA CONTRACT (validated live 2026-08-13, B254):
#   - RouterBench ships ONLY routerbench_{0shot,5shot,raw}.pkl. `datasets` has no pickle reader,
#     so load_dataset raises DataFilesNotFoundError. Read via hf_hub_download + pandas.read_pickle.
#   - There is no `small_model_correct` column; RouterBench has one correctness column per
#     candidate model, named after the model. The old loader assumed the former and would have
#     produced an empty dataset even with a working reader.
#   - Routing boundary = `mistralai/mistral-7b-chat`, the SMALLEST model in the benchmark and so
#     the closest available analogue to something that runs on an S24 Ultra. It answers 29.9% of
#     prompts correctly, making `local` the minority class the metric scores.
#   - The `prompt` column is a STRING CONTAINING A PYTHON LIST LITERAL of the message turns;
#     it is literal_eval'd and joined, or every eval prompt would carry brackets and quote marks.
#   - No official split: a deterministic ~20% holdout is hashed from `sample_id`.
#     29,230 train / 7,267 test, zero text overlap, self-consistency 1.0.
#
# CSE quota: 2x L40S (auto-2gpu profile), 24h wall-clock with checkpoint-and-requeue.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=routerbench
export TASK="Fine tune a small model to decide whether a user prompt can be answered on-device or must be escalated to a larger cloud model, targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
