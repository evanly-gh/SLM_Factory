#!/bin/bash
# Verify each ablation arm is doing the thing it was launched to do — not that it is merely alive.
#
# A run that produces plausible numbers while its ablation silently did nothing is the failure mode
# this suite is most exposed to: three of the four arms differ from their baseline by a single
# exported variable, and if that variable is misspelled, unset by an inherited environment, or
# never read, the job completes normally and reproduces the control. Nothing in the score would say
# so. So this checks the CONSEQUENCE of each flag in the run's own log:
#
#   reset      the per-escalation rewind actually fired, and the curriculum shrank when it did
#   deepseek   the teacher is DeepSeek, the goal was pinned, and rows are labelled synth:deepseek
#   nosynth    synthesis was refused BY THE OPERATOR, and no synthesis round ever ran
#   xlam-qwen  the teacher is the LOCAL Qwen3.6 (not DeepSeek), and the goal was pinned to 0.87
#
# Usage: scripts/check_ablations.sh [job_id ...]   (defaults to the four arms submitted 2026-09-04)
set -uo pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
SLURM_LOGS="$PROJ/logs/slurm"

JOBS=("$@")
if [ "${#JOBS[@]}" -eq 0 ]; then
    JOBS=(39562027 39562028 39562029 39562030)
fi

_hit() { # pattern, label, logfile
    local n
    n=$(grep -c -- "$1" "$3" 2>/dev/null || true)
    n=${n:-0}
    if [ "$n" -gt 0 ]; then
        printf '    PASS  %-46s (%s hit(s))\n' "$2" "$n"
    else
        printf '    ....  %-46s (not yet)\n' "$2"
    fi
}

_must_not() { # pattern, label, logfile
    local n
    n=$(grep -c -- "$1" "$3" 2>/dev/null || true)
    n=${n:-0}
    if [ "$n" -gt 0 ]; then
        printf '    FAIL  %-46s (%s hit(s) — should be ZERO)\n' "$2" "$n"
    else
        printf '    ok    %-46s (still zero)\n' "$2"
    fi
}

for job in "${JOBS[@]}"; do
    line=$(squeue -h -j "$job" -o "%T|%M|%R|%j" 2>/dev/null)
    if [ -n "$line" ]; then
        state=${line%%|*}; rest=${line#*|}
        elapsed=${rest%%|*}; rest=${rest#*|}
        reason=${rest%%|*}; name=${rest##*|}
    else
        read -r name state elapsed <<<"$(sacct -n -j "$job" --format=JobName%40,State%20,Elapsed -P 2>/dev/null | head -1 | tr '|' ' ')"
        reason="-"
    fi
    log=$(ls -1t "$SLURM_LOGS"/*-"$job".out 2>/dev/null | head -1)
    printf '\n== %s  %s  state=%s  elapsed=%s  %s\n' "$job" "${name:-?}" "${state:-?}" "${elapsed:-?}" \
        "$([ "${reason:-}" != "-" ] && [ -n "${reason:-}" ] && echo "reason=$reason")"
    if [ -z "$log" ]; then
        echo "    (no log yet)"
        continue
    fi
    printf '    log: %s (%s lines)\n' "$log" "$(wc -l < "$log")"

    case "$name" in
      *reset*)
        _hit "seed snapshot captured" "seed snapshot taken at initial_gold" "$log"
        _hit "ABLATION: SLM_ABLATION_RESET_DATA_ON_ESCALATION=1" "ablation banner declared at startup" "$log"
        # The consequence. Absent this line an escalation carried the dataset forward, i.e. the
        # arm silently ran the baseline.
        _hit "Dataset RESET to seed (ablation)" "reset FIRED on a tier promotion" "$log"
        _must_not "Dataset carried forward" "no tier carried its dataset forward" "$log"
        grep -h "Dataset RESET to seed (ablation)" "$log" 2>/dev/null | sed 's/^/      > /'
        ;;
      *deepseek*)
        _hit "API TEACHER MODE ON (deepseek" "teacher is DeepSeek, no vLLM server" "$log"
        _hit "SLM_STOP_THRESHOLD=0.8000 pinned" "goal pinned to the baseline's 0.80" "$log"
        _hit "measuring deepseek-v4-flash" "DeepSeek measured on this eval set" "$log"
        _must_not "Qwen/Qwen3.6-35B-A3B 5-shot" "local Qwen was NOT used as teacher" "$log"
        grep -hE "\[teacher\] ner_bc5cdr:|\[threshold\] (teacher baseline|SLM_STOP)" "$log" 2>/dev/null | tail -3 | sed 's/^/      > /'
        ;;
      *nosynth*)
        _hit "ABLATION: SLM_SYNTH_DISALLOW=1" "ablation banner declared at startup" "$log"
        _hit "SLM_SYNTH_DISALLOW=1 — synthetic data is REFUSED" "refusal is BY OPERATOR, not the gate" "$log"
        # The consequence: not one generated row may enter the curriculum.
        _must_not "surgical_synthesis      +" "no synthesis round added rows" "$log"
        _must_not "SLM_TEACHER_SYNTH_BYPASS=1" "bypass did NOT leak in from the shell" "$log"
        grep -hE "surgical_synthesis is UNAVAILABLE|synthetic data is REFUSED" "$log" 2>/dev/null | tail -2 | sed 's/^/      > /'
        ;;
      *xlam*)
        _hit "vLLM synth (Qwen/Qwen3.6-35B-A3B)" "teacher is the LOCAL Qwen3.6" "$log"
        _hit "SLM_STOP_THRESHOLD=0.8700 pinned" "goal pinned to DeepSeek run's 0.87" "$log"
        _must_not "API TEACHER MODE ON" "API mode did NOT leak in from the shell" "$log"
        grep -hE "\[teacher\] xlam_bfcl:|\[threshold\] SLM_STOP" "$log" 2>/dev/null | tail -2 | sed 's/^/      > /'
        ;;
    esac
    # Anything that would end the run early, in any arm.
    grep -hE "Traceback|ERROR:|RunHealthError|AblationStateError|plan space is exhausted" "$log" 2>/dev/null \
        | tail -3 | sed 's/^/    !! /'
done
echo
