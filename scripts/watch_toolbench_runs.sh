#!/bin/bash
# Health summary for the toolbench bring-up runs (2026-08-24).
#
# Pulls the handful of signals that decide whether a run is WORKING, in the order they should be
# read. The order is the point: on this task a low pass rate is the expected result for a
# sub-billion model, so the score is the LAST thing to look at. What separates "the model cannot do
# ToolBench" from "the harness is broken" is upstream of the score — did the loader return both
# splits, did the judge answer, is the output the right shape.
#
# Usage: scripts/watch_toolbench_runs.sh [logfile ...]
#        with no arguments, picks up every slm-toolbench-*-verify* log in logs/slurm.
set -uo pipefail
PROJ="/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory"
cd "$PROJ" || exit 1

logs=("$@")
if [ ${#logs[@]} -eq 0 ]; then
    mapfile -t logs < <(ls -t logs/slurm/slm-toolbench-*verify*.out 2>/dev/null)
fi
if [ ${#logs[@]} -eq 0 ]; then
    echo "no toolbench verify logs yet"
    squeue -u evanly -o "%.10i %.34j %.8T %.12M %.24R"
    exit 0
fi

for log in "${logs[@]}"; do
    echo "================================================================"
    echo "LOG  $log"
    echo "     $(stat -c '%y  %s bytes' "$log" 2>/dev/null)"
    jobid="${log##*-}"; jobid="${jobid%.out}"
    squeue -j "$jobid" -h -o "     STATE %T on %R since %M" 2>/dev/null \
        || echo "     not in queue (finished or never started)"

    echo "--- 1. did it start, and on what ------------------------------"
    grep -E "GPU PROFILE|SLM_GPU_PROFILE|synth server|vLLM|serving model|SLM_RUN_DIR" "$log" \
        | tail -6
    echo "--- 2. model actually selected --------------------------------"
    grep -iE "FORCE_MODEL|selected_model|MODEL SELECTION|pinned|single_model|escalat" "$log" \
        | tail -8
    echo "--- 3. data: loader, both splits, curriculum ------------------"
    grep -E "\[toolbench\]|CURRICULUM|eval set|EVAL SET|train / .* eval" "$log" | tail -14
    echo "--- 4. quality control ----------------------------------------"
    grep -E "\[qc\]" "$log" | tail -8
    echo "--- 5. teacher fitness + synthesis gate -----------------------"
    grep -E "\[teacher\]|BYPASS|synthetic data" "$log" | tail -10
    echo "--- 6. THE HARNESS: format first, then fabrication, then score-"
    grep -E "format_valid|undeclared_api|judge_unsure|judged_rows|tooleval_pass_rate|G1_|G2_|G3_" \
        "$log" | tail -20
    echo "--- 7. scores over time ---------------------------------------"
    grep -E "Baseline|BASELINE|F1 =|f1=|score=|ITERATION|iteration [0-9]" "$log" | tail -16
    echo "--- 8. sample predictions (is the output even the right shape?)"
    grep -A4 "sample predictions" "$log" | tail -24
    echo "--- 9. orchestrator decisions ---------------------------------"
    grep -E "DECISION|next_action|INTERVENTION|rebuild|hypothesis|mine_new_real|surgical_synthesis" \
        "$log" | tail -14
    echo "--- 10. run health / stalls -----------------------------------"
    grep -E "RUN HEALTH|run_health|STOPPING|STAGNAN|TERMINAT|exhausted|added 0 new rows|⚠|WARN" \
        "$log" | tail -14
    echo "--- 11. errors ------------------------------------------------"
    grep -nE "Traceback|Error|ERROR|FAILED|refus|abort|CUDA out of memory" "$log" | tail -14
done
echo "================================================================"
squeue -u evanly -o "%.10i %.34j %.8T %.12M %.24R"
