#!/bin/bash
# De-duplicate the SAME pipeline run queued on both a dedicated (non-preemptible) partition and a
# preemptible ckpt partition, so two copies never burn GPU-days and paid orchestrator calls at once.
#
# ASYMMETRIC ON PURPOSE. An earlier symmetric version cancelled whichever job reached RUNNING
# second, and that was wrong: on ckpt, RUNNING is not a durable win. On 2026-08-04 the ckpt-g2 copy
# started, this script cancelled the safe dedicated copy, and ckpt was then preempted after 1m57s
# and again after 3m10s — shorter than the ~4.5 min the vLLM synth server needs to boot, so it made
# no progress at all and the reliable slot was gone.
#
# Two policies, because the right answer depends on whether either side can be preempted:
#
#   (default)     protected/opportunistic. Use when one job is on a preemptible partition. The
#                 first argument is NEVER cancelled; the second is cancelled only once the first
#                 is actually RUNNING.
#   --first-wins  symmetric. Use ONLY when both jobs are non-preemptible (e.g. the same gpu-l40s
#                 partition billed to two different accounts). Reaching RUNNING is then a durable
#                 win, so whichever starts first keeps the hardware and the other is cancelled.
#
# Usage: race_arbiter.sh [--first-wins] <jobA> <jobB>
set -uo pipefail
POLICY="protected"
if [ "${1:-}" = "--first-wins" ]; then
    POLICY="first-wins"
    shift
fi
DED="${1:?usage: race_arbiter.sh [--first-wins] <jobA> <jobB>}"
CKPT="${2:?usage: race_arbiter.sh [--first-wins] <jobA> <jobB>}"

state_of() { squeue -h -j "$1" -O "State:20" 2>/dev/null | tr -d ' '; }

echo "ARBITER policy=$POLICY A=$DED B=$CKPT"
while true; do
    sd=$(state_of "$DED")
    sc=$(state_of "$CKPT")

    if [ -z "$sd" ] && [ -z "$sc" ]; then
        echo "ARBITER both jobs left the queue; exiting"
        exit 0
    fi

    # Symmetric policy: either side reaching RUNNING resolves the race.
    if [ "$POLICY" = "first-wins" ] && [ "$sc" = "RUNNING" ]; then
        echo "ARBITER WINNER=$CKPT (RUNNING first) — cancelling $DED"
        scancel "$DED" 2>/dev/null
        echo "ARBITER done"
        exit 0
    fi

    if [ "$sd" = "RUNNING" ]; then
        if [ -n "$sc" ]; then
            echo "ARBITER WINNER=$DED (RUNNING) — cancelling $CKPT"
            scancel "$CKPT" 2>/dev/null
        else
            echo "ARBITER WINNER=$DED (RUNNING); other job already gone"
        fi
        echo "ARBITER done"
        exit 0
    fi

    # A disappeared first job leaves the second as the only remaining copy.
    if [ -z "$sd" ]; then
        echo "ARBITER $DED left the queue; leaving $CKPT to run"
        exit 0
    fi

    sleep 5
done
