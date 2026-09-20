#!/bin/bash
#SBATCH --job-name=slm-sms-spam-ablation-scarce-nosynth-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-sms-spam-ablation-scarce-nosynth-l40s-%j.out
#
# CURATED BENCHMARK: sms_spam (in-distribution, metric minority_f1). SLM_BENCHMARK_TASK pins the
# deterministic loader (the checksummed data/local/sms_spam bundle, HuggingFace ucirvine/sms_spam
# as the fallback) instead of autonomous web_acquire — run.py starts the run NON-autonomous
# against this task's registry spec, which is where its loader, scorer, metric and token budget
# are declared.
#
# WHY THE METRIC IS minority_f1 AND NOT accuracy. The corpus is ~87% ham. Answering `ham` for
# every message scores 0.87 accuracy and ~0.47 macro-F1, both of which look like a working model.
# Minority-class F1 scores it 0.0. This task is the suite's cheapest collapse detector and the
# metric is the only reason that is true — see tasks/sms_spam.py.
#
# HISTORY WORTH KNOWING: an sms_spam loader existed before 2026-08-18 and was deleted with the
# `task_type` channel. It is not restored, it is rewritten: the old split was unshuffled (B44), the
# corpus was never deduplicated so copies of the same message sat on both sides of the firewall,
# and the data came over plain HTTP with no checksum. All three are fixed in the new loader.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=sms_spam

# ─────────────────────────────────────────────────────────────────────────────────────────────
# ABLATION 4, arm A: DOES SYNTHETIC DATA EARN ITS KEEP WHEN REAL DATA IS SCARCE?
#
# THE QUESTION. Across the 09-2026 suite, surgical_synthesis contributed +0.0083 (gec_bea19),
# +0.0413 (multiconer) and +0.0495 (dialogsum) — small next to hyperparameter tuning's +0.075 to
# +0.174. One explanation is that it never had room to matter: those curricula were 5% to 31%
# synthetic, so thousands of real rows dominated whatever the teacher produced. This pair tests
# that directly by starving the run to 100 real rows and changing nothing else.
#
# THE PAIR. Arm A refuses synthesis; arm B allows it. Every other setting is identical, so the
# difference between their final scores IS the value of synthetic data at this data volume.
#   arm A  run_sms_spam_ablation_scarce_nosynth_l40s.slurm   100 real rows, no mining, NO synth
#   arm B  run_sms_spam_ablation_scarce_synth_l40s.slurm     100 real rows, no mining, synth ON
#
# WHY sms_spam, of the ten tasks run. The control that matters is whether the teacher can actually
# do the task — synthesis from a teacher that cannot is a test of bad rows, not of scarcity. Only
# three tasks clear the 0.80 fitness gate: clinc150 (0.9064), xlam_bfcl (0.8700), sms_spam
# (0.8662). xlam_bfcl is out because fine-tuning bought it just +0.026 — a 4B model already calls
# functions at 0.848, so there is no headroom for synthesis to occupy. clinc150 would show a larger
# gap but is confounded: 100 rows over 151 intents is 0.66 rows per class, so roughly a third of
# classes get zero examples in BOTH arms and part of any difference would be that artifact rather
# than the effect. sms_spam is binary, so 100 rows is ~50 per class — genuinely scarce without
# being degenerate — its full-data fine-tuning gain was +0.7086 so there is room to move, and it is
# the fastest task in the suite (119-token prompts).
#
# WHY MINING MUST BE OFF. `mine_new_real` reads deeper into the SAME corpus, so a run capped at 100
# rows would mine straight back to thousands and measure nothing. The cap without the refusal is
# not a scarcity experiment.
#
# WHY THE GOAL IS PINNED AT THE CEILING. The original sms_spam run converged in 7 iterations and
# never escalated a tier, because escalation only fires below the goal. If one arm converged and
# the other did not, their final scores would differ by run length rather than by synthesis.
# SLM_STOP_THRESHOLD=0.99 (THRESHOLD_CEILING) makes convergence unreachable for both, so each arm
# climbs the full ladder and the comparison is between ceilings.
export SLM_ABLATION_TRAIN_CAP=100
export SLM_ABLATION_DISALLOW_MINING=1
export SLM_STOP_THRESHOLD=0.99
# THE CURRICULUM FLOOR HAS TO MOVE, and only because this arm is the case it was written to catch.
# `MIN_CURRICULUM_ROWS` defaults to 500 and aborts the run below it, on the reasoning that a
# curriculum that small normally means the loader broke rather than that the data is genuinely
# absent. Here it is genuinely absent by construction, which is the whole experiment — the first
# attempt (jobs 40107707/40107711) died on exactly this guard. Set to 50 rather than 100 so the
# quality-control passes can still drop a few rows without tripping it.
export SLM_MIN_CURRICULUM_ROWS=50
# ARM A SPECIFIC: synthesis refused outright. The bypass is deliberately absent — with synthesis
# disallowed, setting it would make the log claim the operator both refused and forced the same
# thing. Hyperparameter tuning is this arm's entire action space.
export SLM_SYNTH_DISALLOW=1

export TASK="Fine tune a small model to classify SMS text messages as unsolicited spam or legitimate (ham), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
