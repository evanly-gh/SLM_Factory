#!/bin/bash
#SBATCH --job-name=slm-proactive-listening-cse
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-proactive-listening-cse-%j.out
#
# CURATED BENCHMARK: proactive_listening (out-of-distribution, metric minority_f1 — F1 on the
# minority `interrupt` class).
#
# THE TASK — LlamaPIE's "when to respond" decision (arXiv:2505.04066 §2.3.1). An in-ear assistant
# listens to a conversation; at each pause it must decide whether to whisper a 1-3 word hint to its
# wearer (`interrupt`) or stay silent (`wait`). This is LlamaPIE's SMALL model, the on-device
# component that runs continuously — the one worth fine-tuning. The separate "what to say" large
# model is out of scope here.
#
# CATEGORY: out-of-distribution. The label is not a property of the input's surface form, so a base
# model has no pattern to match and zero-shot performance should be poor; fine-tuning is what
# supplies the decision boundary. Unlike RouterBench, though, the label IS a property of THIS
# conversation (hesitation, a trailing clause, an about-to-be-needed detail from the user's
# memory) rather than of some other model's behaviour — so the signal is genuinely present.
#
# DATA CONTRACT (validated live 2026-08-15):
#   - The dataset is NOT on HuggingFace. It ships as a Google Drive tarball linked from
#     https://github.com/chentuochao/LlamaPIE (Main_dataset.tar, id 1TEquHZR8E53WLR-v09F1Do5UMQH5tjYZ),
#     so it is VENDORED at data/local/proactive_listening/ with a manifest + sha256 checksums.
#     Nothing is downloaded at run time. Override with SLM_PROACTIVE_DIR.
#   - 7,065 train / 878 test dialogues from the `synthetic`, `perl` and `soda` sub-corpora.
#   - A decision point is ANY `|MARKER >` token, not only `|SILENCE >`: the authors mask on the
#     ` >` token, and the excluded `synthetic0` corpus attaches whispers to emotion markers
#     (`|ANGRY >`, `|NEUTRAL >`). Matching silence alone found ZERO positives there.
#   - `synthetic0` is EXCLUDED by default: it is half the training bundle but carries emotion
#     markers that the held-out split has none of, which would be a train/serve mismatch.
#     Opt in with SLM_PROACTIVE_SOURCES=synthetic0,synthetic,perl,soda.
#   - `interrupt` is the minority class (33.4% after balanced decision-point sampling, ~12% in the
#     raw corpus), so the harness reports F1 on `interrupt` — exactly the "hard precision/recall"
#     LlamaPIE reports in its Table 2.
#   - PUBLISHED REFERENCE: their fine-tuned Llama-3.2-1B reaches hard precision 0.728-0.759 and
#     hard recall 0.719-0.777, i.e. F1 ~= 0.74, for a model one size step above our tier-0 0.6B.
#   - Verified before launch: train/test positive rate matched at 33.4%, zero text overlap between
#     splits, zero whisper-marker leakage into the model input, gold-vs-gold self-consistency 1.0,
#     and an all-`wait` degenerate prediction scores exactly 0.0.
#
# CSE quota: 2x L40S (auto-2gpu profile), 24h wall-clock with checkpoint-and-requeue. The
# --requeue + USR1 checkpoint contract is a safety net against the 24h cap and node failure.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=proactive_listening
export TASK="Fine tune a small model to act as a proactive in-ear listening assistant that decides, at each pause in a conversation, whether to whisper a short hint to its wearer or stay silent, targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
