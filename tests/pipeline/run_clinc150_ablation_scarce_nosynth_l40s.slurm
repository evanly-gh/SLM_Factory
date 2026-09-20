#!/bin/bash
#SBATCH --job-name=slm-clinc150-ablation-scarce-nosynth-l40s
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
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-clinc150-ablation-scarce-nosynth-l40s-%j.out
#
# CURATED BENCHMARK: clinc150 (out-of-distribution, metric macro_f1). SLM_BENCHMARK_TASK pins the
# deterministic CLINC150 loader (150 intents + out-of-scope) instead of autonomous web_acquire —
# run.py starts the run NON-autonomous against this task's registry spec, which is where its
# loader, scorer, metric and token budget are declared.
#
# Dedicated gpu-l40s-intelligentsystems quota: 2x L40S (synth server on GPU0, pipeline on GPU1 —
# the auto-2gpu profile in _l40s_task_body.sh). --requeue + USR1 checkpoint contract is a safety
# net against node failure/maintenance.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=clinc150

# ─────────────────────────────────────────────────────────────────────────────────────────────
# ABLATION 5 ON CLINC150, arm A: DOES SYNTHETIC DATA EARN ITS KEEP WHEN REAL DATA IS SCARCE?
#
# WHY REPEAT THIS PAIR. The sms_spam pair (jobs 40128751 / 40126479) answered the question once and
# the answer depended on where you looked: +0.0189 between the two runs' final scores, but +0.2981
# between the two arms on the SAME 360M model. The gap is the model ladder — arm A bought the task's
# ceiling by escalating to a 4B model instead of by adding data — and sms_spam could not separate
# them further because it saturates near 0.96. clinc150 is the follow-up that does not have that
# problem: 151 classes under macro_f1, a published-SOTA-adjacent ceiling the full-data run only
# reached at 0.8952, and a 15,250-row corpus to be starved of.
#
# ONE ROW PER INTENT. SLM_ABLATION_TRAIN_CAP=151 is not an arbitrary number — clinc150 has exactly
# 151 labels (150 intents + oos) and `stratified_by_label` in data/loaders/clinc150.py draws the
# cap round-robin across labels, so 151 lands exactly one example on every class. That is the most
# data-scarce configuration the task admits while still being learnable in principle, and it is the
# regime where synthesis has the most room: the full-data anchor carries 32-34 rows per intent, so
# this is 3% of that density.
#
# WHY MINING MUST BE OFF. `mine_new_real` reads deeper into the SAME corpus, so a run capped at 151
# rows would mine straight back to thousands and measure nothing. The cap without the refusal is
# not a scarcity experiment.
#
# WHY THE GOAL IS PINNED AT THE CEILING. clinc150's original run converged in 4 iterations on a
# 0.6B model and never escalated, because escalation only fires below the goal. If one arm converged
# and the other did not, their final scores would differ by run length rather than by synthesis.
# SLM_STOP_THRESHOLD=0.99 (THRESHOLD_CEILING) makes convergence unreachable for both, so each arm
# climbs the full ladder and the comparison is between ceilings. It also matches the no-convergence
# clinc150 run (job 40105479), which therefore serves as the FULL-DATA control for this pair: that
# run differs from arm B only in the cap and the mining refusal.
export SLM_ABLATION_TRAIN_CAP=151
export SLM_ABLATION_DISALLOW_MINING=1
export SLM_STOP_THRESHOLD=0.99
# THE CURRICULUM FLOOR HAS TO MOVE, and only because this arm is the case it was written to catch.
# `MIN_CURRICULUM_ROWS` defaults to 500 and aborts the run below it, on the reasoning that a
# curriculum that small normally means the loader broke rather than that the data is genuinely
# absent. Here it is absent by construction. Set to 120 rather than 151 so the quality-control
# passes can still drop a few rows without tripping it.
export SLM_MIN_CURRICULUM_ROWS=120
# EARLY STOPPING HAS TO GO, and this is specific to one row per class. The trainer carves a random,
# UNSTRATIFIED 12% validation split out of the curriculum (training/lora_trainer.py) — at 151 rows
# that is 18 rows, and because each class owns exactly one row, 18 intents would have their only
# example moved into validation and receive zero training examples. Arm A could never recover them
# and arm B could synthesise them back, so the pair would credit synthesis for repairing a holdout
# artifact. Independently, the sms_spam pair showed the selector has no signal at this scale: two
# tier-1 configs 0.40 minority_f1 apart finished at validation loss 0.1600 and 0.1581. Rollback on
# regression still protects both arms, and it is set in BOTH so they stay symmetric.
export SLM_EARLY_STOPPING=0
# ARM A SPECIFIC: synthesis refused outright. The bypass is deliberately absent — with synthesis
# disallowed, setting it would make the log claim the operator both refused and forced the same
# thing. Hyperparameter tuning is this arm's entire action space.
export SLM_SYNTH_DISALLOW=1

export TASK="Fine tune a small model for user-intent classification (CLINC150: 150 intents plus out-of-scope detection), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
