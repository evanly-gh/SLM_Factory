#!/bin/bash
# Overnight campaign watchdog (2026-08-13).
#
# Polls the four CSE runs and prints a single ALERT line per detected problem, so a monitoring
# agent can grep for "ALERT" rather than reading megabytes of pipeline log. Detects the failure
# modes that actually cost time on previous runs:
#
#   DIED        — job left the queue without the pipeline printing its completion banner.
#   NODATA      — eval set empty, or a loader raised at cold start.
#   ITERATE     — repeated orchestrator decision failures (the B-series reask loop).
#   FLATLINE    — the last N eval scores are byte-identical, i.e. training is not moving.
#   SYNTHDOWN   — the vLLM synth server never became reachable.
#   NOPROGRESS  — the log has not grown between two polls of a RUNNING job.
#
# Read-only. It never cancels or resubmits; that decision stays with the operator.
set -uo pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
STATE_DIR="$PROJ/logs/watchdog"
mkdir -p "$STATE_DIR"

FLATLINE_N="${FLATLINE_N:-8}"      # identical consecutive scores before calling it stuck
INTERVAL="${INTERVAL:-600}"        # seconds between polls

# The campaign's job ids. Scoped explicitly: logs/slurm/ holds months of older *-cse-* runs and
# every one of them "died without a completion banner", so globbing would bury tonight's signal
# under historical alerts. Override with CAMPAIGN_JOBS to follow a resubmitted job.
CAMPAIGN_JOBS="${CAMPAIGN_JOBS:-38454799 38455147 38455148 38455150 38455213 38455214}"

# grep -c exits 1 when it counts zero, so `$(grep -c ... || echo 0)` yields the two-line string
# "0\n0" and every later [ -gt ] comparison dies with "integer expression expected".
count_matches() {
    local n
    n="$(grep -cE "$1" "$2" 2>/dev/null)" || n=0
    printf '%s' "${n:-0}"
}

poll_once() {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
    squeue -u "$USER" -o "%.10i %.26j %.9T %.8M %.11L %R" -h | sed 's/^/  /'

    for jobid in $CAMPAIGN_JOBS; do
        log="$(ls -t "$PROJ"/logs/slurm/*-"$jobid".out 2>/dev/null | head -1)"
        if [ -z "$log" ]; then
            state="$(squeue -j "$jobid" -h -o '%T' 2>/dev/null)"
            printf '  [job %s] state=%s (no log yet)\n' "$jobid" "${state:-GONE}"
            [ -z "$state" ] && echo "  ALERT DIED $jobid — no log and not in the queue"
            continue
        fi
        job="$(basename "$log" .out)"
        state="$(squeue -j "$jobid" -h -o '%T' 2>/dev/null)"
        [ -z "$state" ] && state="GONE"
        size="$(stat -c %s "$log" 2>/dev/null || echo 0)"
        prev_file="$STATE_DIR/$job.size"
        prev="$(cat "$prev_file" 2>/dev/null || echo -1)"
        echo "$size" > "$prev_file"

        printf '  [%s] state=%s bytes=%s\n' "$job" "$state" "$size"

        # --- terminal states -------------------------------------------------
        if [ "$state" = "GONE" ] && ! grep -q "=== done ===" "$log" 2>/dev/null; then
            echo "  ALERT DIED $job — left the queue with no completion banner"
            tail -25 "$log" 2>/dev/null | sed 's/^/      | /'
        fi

        # --- data / loader problems -----------------------------------------
        if grep -qE "refusing to proceed with an empty|produced zero eval rows|DataFilesNotFound|DatasetNotFoundError|Dataset scripts are no longer supported" "$log" 2>/dev/null; then
            echo "  ALERT NODATA $job"
            grep -nE "refusing to proceed with an empty|produced zero eval rows|DataFilesNotFound|DatasetNotFoundError|Dataset scripts are no longer supported" "$log" | tail -3 | sed 's/^/      | /'
        fi

        # --- orchestrator decision loop --------------------------------------
        reasks="$(count_matches "Decision failed validation" "$log")"
        if [ "$reasks" -gt 12 ]; then
            echo "  ALERT ITERATE $job — $reasks orchestrator decision validation failures"
            grep "Decision failed validation" "$log" | tail -2 | sed 's/^/      | /'
        fi

        # --- synth server --------------------------------------------------
        if grep -q "SLM_SYNTH_WAIT_S" "$log" 2>/dev/null && grep -qE "synth (endpoint|server) (unreachable|never became)" "$log" 2>/dev/null; then
            echo "  ALERT SYNTHDOWN $job"
        fi

        # --- flatlined score -------------------------------------------------
        scores="$(grep -oE "F1=[0-9]+\.[0-9]+" "$log" 2>/dev/null | tail -"$FLATLINE_N")"
        count="$(echo "$scores" | grep -c . || echo 0)"
        distinct="$(echo "$scores" | sort -u | grep -c . || echo 0)"
        if [ "${count:-0}" -ge "$FLATLINE_N" ] && [ "${distinct:-0}" -eq 1 ]; then
            echo "  ALERT FLATLINE $job — last $FLATLINE_N evals all $(echo "$scores" | tail -1)"
        fi

        # --- stalled log ------------------------------------------------------
        if [ "$state" = "RUNNING" ] && [ "$prev" -ge 0 ] && [ "$size" -eq "$prev" ]; then
            echo "  ALERT NOPROGRESS $job — log unchanged since last poll ($size bytes)"
            tail -5 "$log" 2>/dev/null | sed 's/^/      | /'
        fi

        # --- useful signal, not an alert -------------------------------------
        best="$(grep -oE "F1=[0-9]+\.[0-9]+" "$log" 2>/dev/null | sort -t= -k2 -g | tail -1)"
        [ -n "$best" ] && echo "      best so far: $best  (evals: $(grep -c 'F1=' "$log" 2>/dev/null))"
    done
}

if [ "${1:-}" = "once" ]; then
    poll_once
    exit 0
fi

while true; do
    poll_once
    sleep "$INTERVAL"
done
