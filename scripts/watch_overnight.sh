#!/bin/bash
# Watch tonight's runs and print a sentinel line the moment something needs a human.
#
# WHY A SEPARATE WATCHER FROM scripts/monitor_runs.py
#     `monitor_runs.py` answers "how is this run doing" — it parses the whole log and prints a verdict
#     block per run, which is what you read when you sit down. This answers a narrower question:
#     "should someone be woken up". It prints almost nothing on a healthy run, and prints exactly one
#     grep-able line per new problem, so it can be left running unattended.
#
# THE CONTRACT
#     Every abnormal finding is a single line beginning `ANOMALY`. Nothing else in the output starts
#     with that word. Each finding fires ONCE per run (a marker file per job/reason), because a
#     traceback that stays in the log would otherwise re-alert on every poll and the noise would bury
#     the next real finding.
#
# WHAT IT LOOKS FOR, and why each one earned a place
#     traceback / RunHealthError / OOM   the run is over or about to be; nothing else matters
#     LOAD FAILURE                       "data not loading" — the loader could not produce rows
#     format_valid=0.0000                the model emitted nothing readable: a prompt or chat-template
#                                        problem, not a data problem, and it makes the score
#                                        meaningless rather than low (B290)
#     score 1.0000                       too good. On a fresh harness the likely causes are gold
#                                        leaking into the prompt or a scorer comparing a value to
#                                        itself, both of which look like success
#     eval=0 / train=0                   the split loaded empty and every downstream number is noise
#     vanished from the queue            died without writing a completion line, e.g. the node fell
#                                        over or Slurm killed it — invisible to any log-only check
#
# USAGE
#     scripts/watch_overnight.sh 38985393 38985395 38985396
#     scripts/watch_overnight.sh                       # every live slm-* job
set -uo pipefail
cd "$(dirname "$0")/.."

POLL_S="${POLL_S:-180}"
STATE_DIR="${STATE_DIR:-logs/slurm/.watch-state}"
mkdir -p "$STATE_DIR"

JOBS=("$@")
if [ "${#JOBS[@]}" -eq 0 ]; then
    mapfile -t JOBS < <(squeue -u "$USER" -h -o "%i" -n "$(squeue -u "$USER" -h -o '%j' | paste -sd, -)" 2>/dev/null || squeue -u "$USER" -h -o "%i")
fi

# Fire a finding once. The marker is keyed on job AND reason so a run can report several distinct
# problems, but never the same one twice.
flag() {
    local job="$1" reason="$2" detail="$3"
    local marker="$STATE_DIR/${job}.${reason}"
    [ -e "$marker" ] && return 0
    touch "$marker"
    echo "ANOMALY job=$job reason=$reason :: $detail"
}

log_for() {
    ls -t logs/slurm/*"$1".out 2>/dev/null | head -1
}

echo "watching: ${JOBS[*]}  (poll ${POLL_S}s, state $STATE_DIR)"

while :; do
    live=0
    for job in "${JOBS[@]}"; do
        [ -z "$job" ] && continue
        state="$(squeue -j "$job" -h -o '%T' 2>/dev/null | head -1)"
        log="$(log_for "$job")"

        if [ -z "$state" ]; then
            # Gone from the queue. Only interesting if the log never reached a terminal line: a run
            # that finished or was deliberately cancelled says so, and re-flagging it is noise.
            if [ -n "$log" ] && ! grep -qE "FINAL|run complete|Best score|scancel|CANCELLED" "$log" 2>/dev/null; then
                flag "$job" "vanished" "left the queue with no completion line in $(basename "$log")"
            fi
            continue
        fi
        live=$((live + 1))
        [ "$state" != "RUNNING" ] && continue
        [ -z "$log" ] && continue

        # --- hard failures -------------------------------------------------------------
        if grep -q "Traceback (most recent call last)" "$log" 2>/dev/null; then
            flag "$job" "traceback" "$(grep -A3 'Traceback (most recent call last)' "$log" | tail -2 | tr '\n' ' ')"
        fi
        if grep -q "RunHealthError" "$log" 2>/dev/null; then
            flag "$job" "runhealth" "$(grep -m1 -A2 'RunHealthError' "$log" | tr '\n' ' ')"
        fi
        if grep -qiE "CUDA out of memory|OutOfMemoryError|GGML_ASSERT" "$log" 2>/dev/null; then
            flag "$job" "oom" "$(grep -m1 -iE 'CUDA out of memory|OutOfMemoryError|GGML_ASSERT' "$log")"
        fi
        if grep -q "LOAD FAILURE" "$log" 2>/dev/null; then
            flag "$job" "load" "$(grep -m1 'LOAD FAILURE' "$log")"
        fi

        # --- numbers that mean the measurement is broken rather than bad ---------------
        if grep -qE "format_valid=0\.0000" "$log" 2>/dev/null; then
            flag "$job" "format_zero" "$(grep -m1 -E 'format_valid=0\.0000' "$log" | sed 's/^ *//')"
        fi
        if grep -qE "(f1|accuracy|pass_rate|macro_f1|minority_f1)=1\.0000" "$log" 2>/dev/null; then
            flag "$job" "perfect_score" "$(grep -m1 -E '=1\.0000' "$log" | sed 's/^ *//')"
        fi
        if grep -qE "train=0 |eval=0 |eval set is empty|0 usable" "$log" 2>/dev/null; then
            flag "$job" "empty_split" "$(grep -m1 -E 'train=0 |eval=0 |eval set is empty|0 usable' "$log" | sed 's/^ *//')"
        fi
    done
    [ "$live" -eq 0 ] && { echo "all watched jobs have left the queue; watcher exiting"; break; }
    echo "heartbeat $(date +%H:%M) live=$live"
    sleep "$POLL_S"
done
