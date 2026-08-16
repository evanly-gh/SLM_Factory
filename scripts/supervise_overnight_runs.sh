#!/bin/bash
# Overnight campaign supervisor (2026-08-13).
#
# Each of the four tasks is queued on up to two accounts, because they are blocked for DIFFERENT
# reasons and neither is reliably first:
#
#   gpu-l40s-cse                 11 of 14 account GPUs free, but 30th in the fairshare queue.
#                                Reason=Priority. Slurm projected start ~35h.
#   gpu-l40s-intelligentsystems  10 of 10 account GPUs in use. Reason=AssocGrpGRES. Starts the
#                                moment a co-tenant job ends; Slurm projected start ~8h.
#
# Policy, as set by the operator:
#   * NO ckpt/preemptible partitions. Ever. The two sanctioned accounts only.
#   * AUTO-CANCEL: the instant one copy of a task starts, cancel its twins so we stop holding
#     queue slots other people are waiting on.
#   * Submit an int-sys copy only when int-sys actually has room.
#
# Emits one ALERT line per problem so it can be grepped rather than read. Read-only with respect
# to running jobs: it cancels PENDING twins and nothing else.
set -uo pipefail
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
LOG="$PROJ/logs/watchdog/supervisor.log"
STATE_DIR="$PROJ/logs/watchdog"
mkdir -p "$STATE_DIR"

TASKS="xlam_bfcl calendar_json ner_bc5cdr routerbench"
INT_SYS_ACCOUNT="gpu-l40s-intelligentsystems"
INT_SYS_CAP=10
GPUS_PER_JOB=2
INTERVAL="${INTERVAL:-180}"
FLATLINE_N="${FLATLINE_N:-8}"

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# squeue's %b prints N/A for some jobs that DO hold GPUs, so counting it undercounts the account
# and makes a full account look half free. AllocTRES is authoritative.
account_gpus_in_use() {
    local account="$1" total=0 job gpus
    for job in $(squeue -A "$account" -h -t RUNNING -o "%i" 2>/dev/null); do
        gpus="$(scontrol show job "$job" 2>/dev/null \
            | grep -oE 'AllocTRES=[^ ]*' | grep -oE 'gres/gpu:l40s=[0-9]+' | cut -d= -f2)"
        total=$((total + ${gpus:-0}))
    done
    printf '%s' "$total"
}

# All of MY jobs for one task, on either account. The job name is the join key:
# slm-<task-with-dashes>-{cse,l40s}
task_jobs() {
    local task="$1" pattern
    pattern="slm-${task//_/-}-"
    squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null | awk -v p="$pattern" '$2 ~ "^" p'
}

count_matches() {
    local n
    n="$(grep -cE "$1" "$2" 2>/dev/null)" || n=0
    printf '%s' "${n:-0}"
}

check_health() {
    local jobid="$1" name="$2" log
    log="$(ls -t "$PROJ"/logs/slurm/*-"$jobid".out 2>/dev/null | head -1)"
    [ -z "$log" ] && return 0

    if grep -qE "refusing to proceed with an empty|produced zero eval rows|DataFilesNotFound|DatasetNotFoundError|Dataset scripts are no longer supported" "$log" 2>/dev/null; then
        say "ALERT NODATA $name ($jobid)"
        grep -nE "refusing to proceed|produced zero eval rows|DataFilesNotFound|DatasetNotFoundError|no longer supported" "$log" | tail -3 | sed 's/^/    | /' | tee -a "$LOG"
    fi

    local reasks; reasks="$(count_matches "Decision failed validation" "$log")"
    if [ "$reasks" -gt 12 ]; then
        say "ALERT ITERATE $name ($jobid) — $reasks orchestrator decision validation failures"
    fi

    local scores count distinct
    scores="$(grep -oE "F1=[0-9]+\.[0-9]+" "$log" 2>/dev/null | tail -"$FLATLINE_N")"
    count="$(printf '%s\n' "$scores" | grep -c . || true)"
    distinct="$(printf '%s\n' "$scores" | sort -u | grep -c . || true)"
    if [ "${count:-0}" -ge "$FLATLINE_N" ] && [ "${distinct:-0}" -eq 1 ]; then
        say "ALERT FLATLINE $name ($jobid) — last $FLATLINE_N evals all $(printf '%s\n' "$scores" | tail -1)"
    fi

    local size prev pf
    size="$(stat -c %s "$log" 2>/dev/null || echo 0)"
    pf="$STATE_DIR/$jobid.size"; prev="$(cat "$pf" 2>/dev/null || echo -1)"
    echo "$size" > "$pf"
    if [ "$prev" -ge 0 ] && [ "$size" -eq "$prev" ]; then
        say "ALERT NOPROGRESS $name ($jobid) — log unchanged since last poll ($size bytes)"
        tail -4 "$log" 2>/dev/null | sed 's/^/    | /' | tee -a "$LOG"
    fi

    local best n_evals
    best="$(grep -oE "F1=[0-9]+\.[0-9]+" "$log" 2>/dev/null | sort -t= -k2 -g | tail -1)"
    n_evals="$(count_matches "F1=" "$log")"
    [ -n "$best" ] && say "    $name best=$best evals=$n_evals"
}

poll_once() {
    local int_used int_free
    int_used="$(account_gpus_in_use "$INT_SYS_ACCOUNT")"
    int_free=$((INT_SYS_CAP - int_used))
    say "--- poll | int-sys ${int_used}/${INT_SYS_CAP} used (${int_free} free) | cse $(account_gpus_in_use gpu-l40s-cse)/14 used"

    for task in $TASKS; do
        local rows running_id running_name pending_ids have_intsys
        rows="$(task_jobs "$task")"
        running_id=""; running_name=""; pending_ids=""; have_intsys=0
        while read -r jid jname jstate; do
            [ -z "${jid:-}" ] && continue
            : "${jname:=}" "${jstate:=}"
            case "$jname" in *-l40s) have_intsys=1 ;; esac
            if [ "$jstate" = "RUNNING" ]; then
                running_id="$jid"; running_name="$jname"
            elif [ "$jstate" = "PENDING" ]; then
                pending_ids="$pending_ids $jid"
            fi
        done <<< "$rows"

        if [ -n "$running_id" ]; then
            say "    $task: RUNNING as $running_name ($running_id)"
            # AUTO-CANCEL the twins. One task must never burn two allocations, and holding a
            # pending slot we will never use is a tax on everyone else in the account.
            #
            # EXCEPT a twin that is PENDING because it REQUEUED ITSELF across the 24h wall clock.
            # That job is not a redundant duplicate — it is a run with hours of progress banked in
            # a checkpoint, sitting in the queue waiting to resume. On 2026-08-14 this exact case
            # destroyed slm-routerbench-cse-38455150: 22 hours, all four model tiers, best 0.7584,
            # checkpointed cleanly at 06:49, cancelled as a "twin" at 06:54 in favour of a
            # from-scratch job three minutes old. A pending job holding a durable checkpoint is
            # worth strictly more than a fresh one, so it is never the thing to kill.
            for twin in $pending_ids; do
                local twin_name twin_dir
                twin_name="$(squeue -j "$twin" -h -o '%j' 2>/dev/null)"
                twin_dir="$PROJ/logs/runs/${twin_name}-${twin}"
                if [ -s "$twin_dir/checkpoint.json" ]; then
                    say "    KEEP twin $twin of $task — holds a durable checkpoint; it is a requeued run, not a duplicate"
                    say "ALERT DUPLICATE-PROGRESS $task — checkpointed $twin is queued while from-scratch $running_id runs; decide which to keep"
                    continue
                fi
                say "    CANCEL twin $twin of $task (superseded by running $running_id)"
                scancel "$twin" 2>/dev/null || say "    WARN scancel $twin failed"
            done
            check_health "$running_id" "$running_name"
            continue
        fi

        # A task with NO jobs at all has either never been submitted or — the case that actually
        # bit us — had its twins auto-cancelled when a copy started, and then that copy FAILED.
        # xlam_bfcl and calendar_json sat dead for 28h that way: the cancel was correct, but
        # nothing noticed the winner had died. Report it loudly; do not silently resubmit, because
        # a deterministic crash will just reproduce and burn another allocation.
        if [ -z "$pending_ids" ]; then
            local last_log last_state
            last_log="$(ls -t "$PROJ"/logs/slurm/slm-${task//_/-}-*.out 2>/dev/null | head -1)"
            if [ -n "$last_log" ] && ! grep -q "=== done ===" "$last_log" 2>/dev/null; then
                say "ALERT ORPHANED $task — no jobs queued and last run did not complete: $(basename "$last_log")"
                grep -E "^(ValueError|RuntimeError|TypeError|KeyError|Error)|outcome  " "$last_log" 2>/dev/null | tail -3 | sed 's/^/    | /' | tee -a "$LOG"
            elif [ -n "$last_log" ]; then
                say "    $task: finished, no jobs queued ($(basename "$last_log"))"
            else
                say "    $task: no jobs queued, never run"
            fi
            continue
        fi

        say "    $task: pending$pending_ids"
        # Add an int-sys copy only when int-sys genuinely has room. int-sys clears via co-tenant
        # jobs ending (AssocGrpGRES), not via fairshare, so capacity is the only gate that matters.
        if [ "$have_intsys" -eq 0 ] && [ "$int_free" -ge "$GPUS_PER_JOB" ]; then
            local script="$PROJ/tests/pipeline/run_${task}_l40s.slurm"
            if [ -f "$script" ]; then
                say "    SUBMIT int-sys copy of $task (${int_free} GPUs free)"
                (cd "$PROJ" && sbatch "$script" 2>&1 | tee -a "$LOG")
                int_free=$((int_free - GPUS_PER_JOB))
            else
                say "    WARN no int-sys script for $task at $script"
            fi
        fi
    done
}

if [ "${1:-}" = "once" ]; then poll_once; exit 0; fi
say "===== supervisor start (interval ${INTERVAL}s, ckpt partitions DISABLED by policy) ====="
while true; do poll_once; sleep "$INTERVAL"; done
