#!/bin/bash
#SBATCH --job-name=slm-ner-bc5cdr-ablation-singlemodel-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=logs/slurm/slm-ner-bc5cdr-ablation-singlemodel-l40s-%j.out
#
# ============================================================================================
# ABLATION — ONE SHOT AT THE WHOLE POOL (orchestrator picks once, no ladder; added 2026-09-28)
#
# BASELINE: slm-ner-bc5cdr-l40s-39311801 (8h46m, 55 iterations, span_f1 0.8532, converged on
# tier 3 Qwen3.5-2B@Q4_K_M after starting at tier 1 and escalating twice). Same task, data, teacher
# and goal — but instead of starting in the smallest tier, the orchestrator sees ALL 27 feasible
# variants at once, picks one, and the run stays on it: no escalation on failure, no downward
# probe on success. It ends when that model meets the goal, stagnates, or hits the 30-eval cap.
#
# The pick is `orchestrator_choice_node` verbatim (`single_model` delegates to it): candidates
# listed smallest first with their tier, size and published metrics, and an instruction to take the
# most resource-efficient variant that can plausibly reach the goal after LoRA fine-tuning. The
# chosen selector and the orchestrator's one-sentence reason are printed at the top of the log.
#
# WHAT IT ANSWERS: does one informed choice get to the baseline's answer faster than climbing?
# If the orchestrator picks Qwen3.5-2B@Q4_K_M it skips the baseline's 46 tier-1 and tier-2
# iterations; if it picks smaller it may never reach 0.8532; if it picks a 4B it pays in size for
# accuracy the task did not need. All three outcomes are the finding.
#
# THE GOAL IS LEFT CALIBRATED, deliberately. It appears in the selection prompt, so pinning it at
# 0.99 to force a run-to-stagnation would push the orchestrator toward the largest model and turn
# this into a different experiment. The orchestrator may raise it, as the baseline's did (0.80 →
# 0.85), so both runs stop under the same rule.
#
# `unset SLM_SYNTH_API_MODE` keeps the local Qwen teacher the baseline used: sbatch forwards the
# submitting shell's environment, which is how run 39361648 silently became a DeepSeek run.
#
# WALL CLOCK CUT 7d -> 12h on 2026-10-06, after job 40831542 had already run (it took 1h59m, so
# this changes nothing about that result — the limit was never approached). The seven-day request
# was inherited from the baseline and is wrong for this arm in a way that has a measured cost: the
# nine arms generated for the other tasks were all unschedulable with it, sitting
# `ReqNodeNotAvail, Reserved_for_maintenance` against the cluster's October13_Maintenance
# reservation, and a weeklong request gets no backfill on a 10-GPU account shared by a dozen users.
# A `single_model` run is pinned to one variant and bounded by the 30-eval cap, so unlike a
# ladder-climbing baseline it has a real ceiling and can ask for it honestly.
# ============================================================================================
#
# CURATED BENCHMARK: ner_bc5cdr (format-bound, metric span_f1). Promoted 2026-08-13 from the
# autonomous path, where it reached BC5CDR only because the orchestrator happened to emit a plan
# web_acquire could resolve — not reproducible. SLM_BENCHMARK_TASK pins the loader.
#
# Dedicated allocation: 2x L40S (auto-2gpu profile), 12h wall (above).
# --requeue + USR1 remain a safety net against node failure, not a wall-clock mechanism.
export SLM_CUDA_ISOLATION=1
export SLM_BENCHMARK_TASK=ner_bc5cdr
# TEACHER VERIFICATION OF GENERATED ROWS: OFF for this task — see run_ner_bc5cdr_l40s.slurm.
export SLM_VERIFY_SYNTH=0
# SYNTHESIS GATE: bypassed deliberately, as in the baseline — see run_ner_bc5cdr_l40s.slurm.
export SLM_TEACHER_SYNTH_BYPASS=1

export SLM_MODEL_SELECTION_STRATEGY=single_model
unset SLM_SYNTH_API_MODE

export TASK="Fine tune a small model for biomedical named entity recognition on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source "${SLM_ROOT:-$SLURM_SUBMIT_DIR}/tests/pipeline/_l40s_task_body.sh"
