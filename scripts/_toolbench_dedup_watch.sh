#!/bin/bash
# Wait for the toolbench verify runs to start; cancel the duplicate SmolLM2 job when one does.
#
# WHY A GUARD RATHER THAN JUST PICKING ONE QUEUE: both the dedicated (10 GPU) and CSE (14 GPU)
# accounts were fully consumed by other users at submit time, and their release times differ by
# ~7 hours. Queueing the SmolLM2 run on BOTH takes whichever frees first, which is the same reason
# the repo keeps two launchers per benchmark. The cost of that is a real risk of two identical runs
# starting minutes apart and burning four GPUs to produce one result, so the duplicate is cancelled
# the moment either twin writes its first log line.
set -uo pipefail
PROJ="/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory"
cd "$PROJ" || exit 1

SMOL_INTSYS="${1:?intsys smollm2 jobid}"
SMOL_CSE="${2:?cse smollm2 jobid}"
resolved=0

for i in $(seq 1 1400); do   # ~11.5h at 30s
    for f in logs/slurm/slm-toolbench-*verify*.out; do
        [ -s "$f" ] || continue
        echo "TOOLBENCH_STARTED:$f"
    done

    if [ "$resolved" -eq 0 ]; then
        intsys_state=$(squeue -h -j "$SMOL_INTSYS" -o "%T" 2>/dev/null)
        cse_state=$(squeue -h -j "$SMOL_CSE" -o "%T" 2>/dev/null)
        if [ "$intsys_state" = "RUNNING" ]; then
            echo "TOOLBENCH_DEDUP: intsys $SMOL_INTSYS is RUNNING; cancelling cse twin $SMOL_CSE"
            scancel "$SMOL_CSE" 2>&1
            resolved=1
        elif [ "$cse_state" = "RUNNING" ]; then
            echo "TOOLBENCH_DEDUP: cse $SMOL_CSE is RUNNING; cancelling intsys twin $SMOL_INTSYS"
            scancel "$SMOL_INTSYS" 2>&1
            resolved=1
        fi
    fi

    if [ $((i % 20)) -eq 0 ]; then
        echo "waiting t=$((i / 2))min :: $(squeue -u evanly -h -o '%i=%T' | tr '\n' ' ')"
    fi
    sleep 30
done
echo "TOOLBENCH_WATCH_TIMEOUT"
