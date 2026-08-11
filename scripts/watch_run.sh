#!/bin/bash
# Poll a submitted SLM_Factory Slurm job and emit stable sentinel lines on state changes so an
# operator (or agent) can react without tailing the whole log. Usage: watch_run.sh <jobid>
set -uo pipefail
JOB="${1:?usage: watch_run.sh <jobid>}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
# Ask Slurm where the job actually writes, so this works for any partition variant instead of
# assuming one --output naming scheme.
LOG=$(scontrol show job "$JOB" 2>/dev/null | grep -oP 'StdOut=\K\S+')
LOG="${LOG:-$PROJ/logs/slurm/slm-clinc150-l40s-${JOB}.out}"
echo "WATCH_SENTINEL INIT job=$JOB log=$LOG"
last_state=""

while true; do
    state=$(squeue -h -j "$JOB" -O "State:20" 2>/dev/null | tr -d ' ')

    if [ -z "$state" ]; then
        # Job left the queue: settle, then read the accounting record.
        sleep 20
        acct=$(sacct -n -j "${JOB}.batch" -o State,ExitCode 2>/dev/null | head -1 | tr -s ' ')
        [ -z "$acct" ] && acct=$(sacct -n -j "$JOB" -o State,ExitCode 2>/dev/null | head -1 | tr -s ' ')
        if [ -s "$LOG" ] && grep -q "RUN COMPLETE\|=== done ===" "$LOG" 2>/dev/null; then
            echo "WATCH_SENTINEL RUN_CLEAN job=$JOB acct=$acct"
        elif printf '%s' "$acct" | grep -q "CANCELLED"; then
            # Losing side of the partition race — not a pipeline failure.
            echo "WATCH_SENTINEL RUN_CANCELLED job=$JOB acct=$acct"
        else
            echo "WATCH_SENTINEL RUN_FAILED job=$JOB acct=$acct"
            [ -s "$LOG" ] && grep -nE "RUN FAILED|exception in|Error:|Traceback" "$LOG" | tail -5
        fi
        exit 0
    fi

    if [ "$state" != "$last_state" ]; then
        echo "WATCH_SENTINEL STATE job=$JOB state=$state $(date '+%H:%M:%S')"
        last_state="$state"
    fi

    # Catch an in-flight crash while the batch shell is still finalizing.
    if [ -s "$LOG" ] && grep -q "RUN FAILED" "$LOG" 2>/dev/null; then
        echo "WATCH_SENTINEL RUN_FAILED job=$JOB (detected in log)"
        grep -nE "RUN FAILED|exception in|Error:" "$LOG" | tail -5
        exit 0
    fi

    sleep 60
done
