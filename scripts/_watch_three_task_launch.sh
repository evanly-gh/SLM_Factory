#!/bin/bash
# Watch the queued/running L40S task jobs and print one ALERT line per event worth reacting to.
#
# Two classes of event, deliberately on the same channel so a supervising process only has to
# follow one stream:
#   STATECHANGE  a job entered or left the scheduler
#   ALERT        a run printed one of the signatures that ended a previous run
#
# The ALERT patterns are the failure modes this task family actually hit, not a generic error grep:
# a verification wipeout, a synthesis batch that kept nothing, a generation truncated against the
# output-token budget, and the teacher gate refusing synthetic data.
LOGDIR=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm
JOBS="39361648"
declare -A LAST SEEN
for j in $JOBS; do LAST[$j]="?"; SEEN[$j]=0; done

# `hit the output token limit` was dropped on 2026-08-30. It fires once per synthesis batch on
# xlam under a reasoning teacher, and it is not by itself a failure: generation over-requests, so a
# truncated reply costs spend rather than rows. Five identical alerts in ten minutes buried the
# events worth reacting to. The case where truncation DOES cost rows is still caught, by the
# `kept 0 of` / `-> kept 0` patterns that describe the outcome rather than the mechanism.
ALERT_RE='VERIFICATION WIPEOUT|kept 0 of|-> kept 0|rejected EVERY generated row|synthetic data is REFUSED|Traceback \(most recent call last\)|slurmstepd: error|PROMOTING: tier|CONVERGED'

while true; do
    alive=0
    for j in $JOBS; do
        state=$(squeue -h -j "$j" -o "%T" 2>/dev/null | tr -d ' ')
        if [ -z "$state" ]; then
            state=$(sacct -n -X -j "$j" -o State 2>/dev/null | head -1 | tr -d ' ' | cut -d' ' -f1)
            [ -z "$state" ] && state="GONE"
        else
            alive=1
        fi
        if [ "$state" != "${LAST[$j]}" ]; then
            echo "STATECHANGE job=$j ${LAST[$j]} -> $state at $(date '+%F %T')"
            LAST[$j]="$state"
        fi

        # Scan only the lines added since the last pass, so one old hit is not re-reported forever.
        log=$(ls -1 "$LOGDIR"/*-"$j".out 2>/dev/null | head -1)
        [ -z "$log" ] && continue
        total=$(wc -l < "$log" 2>/dev/null || echo 0)
        prev=${SEEN[$j]}
        if [ "$total" -gt "$prev" ]; then
            tail -n +$((prev + 1)) "$log" 2>/dev/null \
                | grep -E "$ALERT_RE" \
                | head -5 \
                | sed "s|^|ALERT job=$j |"
            SEEN[$j]=$total
        fi
    done
    [ "$alive" -eq 0 ] && { echo "STATECHANGE every watched job has left the scheduler"; break; }
    sleep 120
done
