"""The cross-iteration guard that stops a run which cannot make progress.

WHY THIS FILE EXISTS
    xlam run 38566712 ran for 7h42m and $1.34 while every one of its eight data rebuilds added zero
    rows. Nothing stopped it, because every individual symptom looked survivable — a `0 novel` line,
    a quality-control drop, a mining round that found nothing. `agent/run_health.py` watches across
    iterations instead, and the two halves of its contract pull in opposite directions:

      it must fire      on the pattern that means no further iteration can learn anything, with the
                        diagnosis attached, in minutes rather than hours;
      it must NOT fire  on one bad iteration. A guard that trips on noise gets switched off, and a
                        switched-off guard is worse than none.

    So every trip condition below is tested twice: once at the threshold, and once JUST SHORT of it
    — one iteration short for the repeat counters, one row short for the magnitude ones. The second
    assertion is the one that keeps the guard usable.

    Interleaving matters and is deliberate in these fixtures. A rebuild that added rows resets the
    empty-rebuild counter without resetting the mining or verification counters, which is how a run
    reaches three mining shutouts or two verification wipeouts without having already been stopped
    for adding nothing — and it is what these tests have to reproduce to exercise each check on its
    own rather than through the empty-rebuild check firing first.
"""
from __future__ import annotations

import math

import pytest

from agent.run_health import (
    CURRICULUM_SHRINK_FRACTION_ALARM,
    CURRICULUM_SHRINK_ROW_ALARM,
    MAX_CONSECUTIVE_EMPTY_REBUILDS,
    MAX_CONSECUTIVE_EMPTY_SYNTHESIS,
    MAX_CONSECUTIVE_TOTAL_VERIFY_REJECTIONS,
    MAX_EMPTY_MINING_ROUNDS,
    MAX_LOAD_FAILURES,
    MAX_MINING_ATTEMPTS_WITHOUT_ACCEPTANCE,
    MINING_RETIRED_KEY,
    IterationRecord,
    RunHealthError,
    format_health_summary,
    observe_iteration,
    record_load_failure,
)


def _record(iteration=1, strategy="mine_new_real", rows_added=0, **overrides) -> IterationRecord:
    fields = {
        "iteration": iteration,
        "strategy": strategy,
        "rows_added": rows_added,
        "curriculum_before": 3000,
        "curriculum_after": 3000 + max(0, rows_added),
        "qc_removed": 0,
        "firewall_removed": 0,
    }
    fields.update(overrides)
    return IterationRecord(**fields)


def _productive_mining(iteration: int) -> IterationRecord:
    """A mining round that added rows and ran no verification.

    Clears the empty-rebuild counter without touching the verification counter, which is how a run
    reaches a second wipeout without having already been stopped for adding nothing.
    """
    return _record(iteration=iteration, strategy="mine_new_real", rows_added=200)


def _productive_synthesis(iteration: int) -> IterationRecord:
    """A synthesis round that added rows. Clears the empty-rebuild counter; the mining counter is
    only touched by `mine_new_real` records, so this leaves it standing."""
    return _record(iteration=iteration, strategy="surgical_synthesis", rows_added=200,
                   verify_attempted=250, verify_kept=200)


def _observe(state, record) -> None:
    observe_iteration(state, record, log=lambda *_: None)


# --------------------------------------------------------------------------
# 1. Rebuilds that add nothing
# --------------------------------------------------------------------------


def test_one_empty_mining_round_is_survivable_and_does_not_retire_the_route():
    """A single fruitless mining round is a normal outcome — a source is exhausted, a search found
    nothing — and the loop is allowed to try it once more or switch sub-strategy next turn."""
    state: dict = {}
    _observe(state, _record(iteration=1))
    assert state["run_health"]["empty_mining"] == 1
    assert not state.get(MINING_RETIRED_KEY)


def test_empty_mining_rounds_retire_the_route_but_never_stop_the_run():
    """Mining finding nothing is an ANSWER about the available corpora, not a malfunction.

    Re-asking cannot change it, so the route closes — but synthesis and hyperparameter tuning are
    untouched and can still make progress, which is why this must not raise. Killing the run here is
    what ended runs 38735780 and 38832588 with hours of budget left."""
    state: dict = {}
    for iteration in range(1, MAX_EMPTY_MINING_ROUNDS + 1):
        _observe(state, _record(iteration=iteration))
    assert state["run_health"]["empty_mining"] == MAX_EMPTY_MINING_ROUNDS
    assert "added 0 rows" in str(state[MINING_RETIRED_KEY])
    # And it stays closed and quiet for the rest of the run.
    for iteration in range(MAX_EMPTY_MINING_ROUNDS + 1, MAX_EMPTY_MINING_ROUNDS + 6):
        _observe(state, _record(iteration=iteration))


def test_a_retired_mining_route_is_no_longer_offered_to_the_planner():
    """The retirement has to reach the PLANNER, not just the log. Otherwise the orchestrator is
    still shown `mine_new_real`, picks it, and the run spends turns on a route already known dead."""
    from agent.data_rebuild import mining_available_for_state

    state: dict = {"source_progress": {"some_corpus": {"consumed": 10}}}
    assert mining_available_for_state(state), "unexhausted source: available before retirement"
    for iteration in range(1, MAX_EMPTY_MINING_ROUNDS + 1):
        _observe(state, _record(iteration=iteration))
    # Overrides the source bookkeeping: evidence from actually running mining beats
    # `source_progress`'s assumption that a source of unknown length still has rows.
    assert not mining_available_for_state(state)


def test_consecutive_empty_synthesis_stops_the_run_at_the_budget():
    """Synthesis is the one route whose emptiness IS fatal: the teacher was reachable and a category
    was targeted, so nothing surviving means the generator and verifier disagree about the contract,
    and they will disagree identically next turn."""
    state: dict = {}
    empty = {"strategy": "surgical_synthesis"}
    for iteration in range(1, MAX_CONSECUTIVE_EMPTY_SYNTHESIS):
        _observe(state, _record(iteration=iteration, **empty))
        assert state["run_health"]["empty_synthesis"] == iteration
    with pytest.raises(RunHealthError) as excinfo:
        _observe(state, _record(iteration=MAX_CONSECUTIVE_EMPTY_SYNTHESIS, **empty))
    message = str(excinfo.value)
    assert f"ZERO rows on {MAX_CONSECUTIVE_EMPTY_SYNTHESIS} consecutive attempts" in message
    # The diagnosis, not just the symptom: nothing was generated at all, which is a different fault
    # from rows generated and then rejected, and calls for a different fix.
    assert "Synthesis generated nothing at all" in message


def test_empty_mining_rounds_do_not_count_toward_the_synthesis_budget():
    """The whole point of splitting the counter: a run may not be stopped for mining outcomes."""
    state: dict = {}
    for iteration in range(1, 8):
        _observe(state, _record(iteration=iteration))
    assert state["run_health"]["empty_synthesis"] == 0


def test_a_synthesis_round_that_added_rows_clears_the_synthesis_count():
    """Otherwise the guard accumulates across a whole run and eventually fires on a healthy one."""
    state: dict = {}
    _observe(state, _record(iteration=1, strategy="surgical_synthesis"))
    assert state["run_health"]["empty_synthesis"] == 1
    _observe(state, _productive_synthesis(2))
    assert state["run_health"]["empty_synthesis"] == 0


def test_an_unknown_rebuild_strategy_still_has_a_guard():
    """No strategy other than the two exists today, but a future one must not be unguarded."""
    state: dict = {}
    for iteration in range(1, MAX_CONSECUTIVE_EMPTY_REBUILDS):
        _observe(state, _record(iteration=iteration, strategy="some_future_strategy"))
    with pytest.raises(RunHealthError):
        _observe(state, _record(iteration=MAX_CONSECUTIVE_EMPTY_REBUILDS,
                                strategy="some_future_strategy"))


def test_the_initial_gold_build_is_not_a_rebuild():
    """Cold start builds the curriculum rather than growing it, so `rows_added` is not the measure
    of whether it worked and it must not consume one of the two allowed empty rebuilds."""
    state: dict = {}
    for iteration in range(1, 4):
        _observe(state, _record(iteration=iteration, strategy="initial_gold"))
    assert state["run_health"]["empty_rebuilds"] == 0


def test_the_empty_rebuild_diagnosis_names_synthesis_when_synthesis_is_what_failed():
    state: dict = {}
    wipeout = {"strategy": "surgical_synthesis", "verify_attempted": 40, "verify_kept": 0}
    _observe(state, _record(iteration=1, **wipeout))
    with pytest.raises(RunHealthError) as excinfo:
        _observe(state, _record(iteration=2, **wipeout))
    assert "verification rejected" in str(excinfo.value)


# --------------------------------------------------------------------------
# 2. The curriculum shrinking
# --------------------------------------------------------------------------
#
# What this check measures is the curriculum ending an iteration SMALLER than it started it — not
# "quality control removed a lot of rows", and the distinction is the entire point of the check.
# Rows that were ALREADY in the curriculum trained a model successfully on a previous iteration,
# and the curriculum is designed to be cumulative, so a step now rejecting them is rejecting the
# task's own data: a QC key that does not match the task's row schema, or a label filter running
# against the wrong vocabulary. That cannot be recovered by iterating.
#
# Measuring the removal count instead false-fired on a healthy run. See
# `test_a_duplicate_heavy_mining_round_that_qc_dedups_away_is_not_a_shrink` below: a mining round
# returning 400 near-duplicates of rows already present, all 400 dropped by dedup, is 99% of what QC
# examined and no reason to stop anything. That shape is a mining-NOVELTY problem, and it is already
# covered by `mining_shutouts` and the empty-rebuild counter — it is explicitly not this alarm.
#
# The guard as a whole exists because nothing watched across iterations at all (B308).


def _shrunk(before: int, shrank: int, *, rows_added: int = 500, **overrides) -> IterationRecord:
    """A rebuild that left the curriculum `shrank` rows SMALLER than it found it.

    `rows_added` is deliberately non-zero so these fixtures exercise the shrink check on its own: a
    rebuild that added nothing would also increment the empty-rebuild counter. The removal count is
    the arithmetic that produces the shrink — everything the rebuild added, plus the shrink itself —
    and `qc_examined` is what QC actually looked at, the previous curriculum plus the new rows.
    """
    fields = {
        "curriculum_before": before,
        "curriculum_after": before - shrank,
        "qc_removed": rows_added + shrank,
        "qc_examined": before + rows_added,
    }
    fields.update(overrides)
    return _record(iteration=1, strategy="mine_new_real", rows_added=rows_added, **fields)


def _shrink_error(record: IterationRecord) -> str:
    """Observe `record` on a fresh state, require it to stop the run, and return the message.

    Asserts the message points at the [qc] and [firewall] lines, which is what identifies THIS
    check as the one that fired rather than another check tripping on the same record.
    """
    with pytest.raises(RunHealthError) as excinfo:
        _observe({}, record)
    message = str(excinfo.value)
    assert "[qc]" in message and "[firewall]" in message
    return message


def test_a_shrink_at_the_absolute_row_alarm_stops_the_run():
    """The threshold is stated as an absolute count as well as a fraction because a fraction alone
    misses a large loss out of a large curriculum. 1,000 rows out of 250,000 is 0.4%, nowhere near
    the fraction alarm, so this exercises the absolute limb by itself."""
    message = _shrink_error(_shrunk(before=250_000, shrank=CURRICULUM_SHRINK_ROW_ALARM))
    assert str(CURRICULUM_SHRINK_ROW_ALARM) in message
    assert "250000" in message


def test_a_shrink_one_row_under_the_absolute_row_alarm_is_survivable():
    """The assertion that keeps the guard switched on. Curricula lose rows for ordinary reasons —
    a re-dedup, a slightly stricter length bound — and a guard that stops the run for those gets
    disabled, which is worse than not having it."""
    _observe({}, _shrunk(before=250_000, shrank=CURRICULUM_SHRINK_ROW_ALARM - 1))


def test_a_shrink_at_the_fraction_alarm_of_a_small_curriculum_stops_the_run():
    """And as a fraction as well, because an absolute count alone misses a catastrophic loss out of
    a small curriculum: a quarter of 400 rows is 100, which the row alarm would let through, and a
    task that just lost a quarter of its cumulative data is in exactly the state this guard is for.
    """
    before = 400
    shrank = math.ceil(before * CURRICULUM_SHRINK_FRACTION_ALARM)
    assert shrank < CURRICULUM_SHRINK_ROW_ALARM, "fixture must isolate the fraction limb"
    message = _shrink_error(_shrunk(before=before, shrank=shrank))
    assert str(shrank) in message


def test_a_shrink_one_row_under_the_fraction_alarm_is_survivable():
    before = 400
    shrank = math.ceil(before * CURRICULUM_SHRINK_FRACTION_ALARM) - 1
    _observe({}, _shrunk(before=before, shrank=shrank))


def test_a_duplicate_heavy_mining_round_that_qc_dedups_away_is_not_a_shrink():
    """The regression test for measuring the curriculum instead of the removal count.

    This is the exact shape that false-fired the previous version of the check: a 4-row curriculum,
    a mining round returning 400 rows that are near-duplicates of what is already there, and dedup
    dropping all 400. That is 400 of the 404 rows QC examined, and nothing whatsoever is wrong — the
    curriculum still grew, and the real complaint (mining is not finding novel data) belongs to
    `mining_shutouts` and the empty-rebuild counter, which already report it. The removal counts stay
    in the ledger, they are just no longer what the alarm is measured against.
    """
    state: dict = {}
    _observe(state, _record(iteration=1, strategy="mine_new_real", rows_added=400,
                            curriculum_before=4, curriculum_after=5,
                            qc_removed=400, qc_examined=404))
    assert state["run_health"]["history"][-1]["qc_removed"] == 400


@pytest.mark.parametrize("net", [0, 1, 5000], ids=["flat", "grew_by_one", "grew_a_lot"])
@pytest.mark.parametrize("qc_removed", [0, 400, CURRICULUM_SHRINK_ROW_ALARM * 10])
def test_a_curriculum_that_did_not_get_smaller_never_trips_the_alarm(net, qc_removed):
    """However many rows QC threw away, if the curriculum came out no smaller than it went in then
    none of the rows that already trained a model were lost, which is the only thing this alarm is
    about. The removal count on its own carries no information about that."""
    before = 3000
    _observe({}, _record(iteration=1, strategy="mine_new_real", rows_added=qc_removed + net,
                         curriculum_before=before, curriculum_after=before + net,
                         qc_removed=qc_removed, qc_examined=before + qc_removed + net))


def test_the_shrink_message_names_both_removal_counts_and_where_to_read_them():
    """Two different components delete rows and they fail for different reasons — a QC key that does
    not match the task's row schema versus an eval-overlap filter running against the wrong eval
    set. A reader given only the net shrink cannot tell which one to go and look at, so the message
    has to attribute the loss and name the log lines that explain it."""
    message = _shrink_error(_shrunk(before=250_000, shrank=1200,
                                    qc_removed=903, firewall_removed=797))
    assert "903" in message
    assert "797" in message


# --------------------------------------------------------------------------
# 3. Verification rejecting everything
# --------------------------------------------------------------------------


def _wipeout(iteration: int) -> IterationRecord:
    return _record(iteration=iteration, strategy="surgical_synthesis", rows_added=0,
                   verify_attempted=60, verify_kept=0)


def test_one_verification_wipeout_is_survivable():
    state: dict = {}
    _observe(state, _wipeout(1))
    assert state["run_health"]["verify_wipeouts"] == 1


def test_two_consecutive_verification_wipeouts_stop_the_run():
    """The productive rebuild in between is what isolates this check: it clears the empty-rebuild
    counter, so the error raised on the second wipeout is about verification alone."""
    state: dict = {}
    _observe(state, _wipeout(1))
    _observe(state, _productive_mining(2))
    with pytest.raises(RunHealthError) as excinfo:
        _observe(state, _wipeout(3))
    message = str(excinfo.value)
    assert (
        f"verification rejected EVERY generated row on "
        f"{MAX_CONSECUTIVE_TOTAL_VERIFY_REJECTIONS} consecutive iterations"
    ) in message
    assert "consecutive data rebuilds added ZERO rows" not in message


def test_a_batch_with_survivors_clears_the_wipeout_count():
    state: dict = {}
    _observe(state, _wipeout(1))
    _observe(state, _record(iteration=2, strategy="surgical_synthesis", rows_added=30,
                            verify_attempted=60, verify_kept=30))
    assert state["run_health"]["verify_wipeouts"] == 0


def test_a_batch_that_generated_nothing_is_not_a_wipeout():
    """`0 kept of 0 attempted` means synthesis never ran, which is a different failure from the
    verifier and the generator disagreeing — and counting it here would hide the real one."""
    state: dict = {}
    _observe(state, _record(iteration=1, strategy="surgical_synthesis",
                            verify_attempted=0, verify_kept=0))
    assert state["run_health"]["verify_wipeouts"] == 0


# --------------------------------------------------------------------------
# 4. Mining that sees candidates and accepts none
# --------------------------------------------------------------------------


def _shutout(iteration: int) -> IterationRecord:
    return _record(iteration=iteration, strategy="mine_new_real", rows_added=0,
                   candidates_seen=400)


def test_mining_shutouts_are_counted_separately_from_empty_searches():
    """The counter earns its keep as a DIAGNOSIS even though it no longer stops the run: a filter
    refusing every candidate needs a different fix from an exhausted corpus, and the log line has to
    say which one happened."""
    state: dict = {}
    _observe(state, _shutout(1))
    _observe(state, _productive_synthesis(2))
    assert state["run_health"]["mining_shutouts"] == 1


def test_repeated_mining_shutouts_retire_the_route_rather_than_stopping_the_run():
    """A filter rejecting everything, not an empty search — the shape the pre-2026-08-19
    whole-source overlap check produced on every candidate (B293).

    At default thresholds the empty-mining counter reaches its budget first, so retirement is already
    in hand by the second shutout; this test pins the OUTCOME (route closed, run alive) so it holds
    whichever of the two counters gets there first."""
    state: dict = {}
    for iteration in range(1, MAX_MINING_ATTEMPTS_WITHOUT_ACCEPTANCE + 1):
        _observe(state, _shutout(iteration))
    assert state.get(MINING_RETIRED_KEY)
    assert "candidate" in str(state[MINING_RETIRED_KEY])


def test_mining_that_saw_no_candidates_is_not_a_shutout():
    """An honest empty search. Counting it here would attribute an exhausted corpus to a broken
    filter and send the reader looking for a gate that is not there."""
    state: dict = {}
    _observe(state, _record(iteration=1, strategy="mine_new_real", candidates_seen=0))
    assert state["run_health"]["mining_shutouts"] == 0


def test_a_mining_round_that_accepted_rows_clears_the_count():
    state: dict = {}
    _observe(state, _shutout(1))
    _observe(state, _record(iteration=2, strategy="mine_new_real", rows_added=90,
                            candidates_seen=400))
    assert state["run_health"]["mining_shutouts"] == 0


# --------------------------------------------------------------------------
# 5. Load failures
# --------------------------------------------------------------------------


def test_one_load_failure_is_survivable():
    state: dict = {}
    record_load_failure(state, what="xlam train split", error=OSError("connection reset"),
                        log=lambda *_: None)
    assert state["run_health"]["load_failures"] == 1


def test_a_repeat_load_failure_stops_the_run():
    """A loader reads a cached corpus deterministically, so a repeat failure is an environment or
    data problem that will not resolve by retrying for another six hours."""
    state: dict = {}
    for _ in range(MAX_LOAD_FAILURES - 1):
        record_load_failure(state, what="xlam train split", error=OSError("connection reset"),
                            log=lambda *_: None)
    with pytest.raises(RunHealthError) as excinfo:
        record_load_failure(state, what="xlam train split", error=OSError("connection reset"),
                            log=lambda *_: None)
    message = str(excinfo.value)
    assert f"data loading failed {MAX_LOAD_FAILURES} times" in message
    # Names what failed, so the reader does not have to scroll for it.
    assert "xlam train split" in message


# --------------------------------------------------------------------------
# 6. What this module deliberately does NOT count: refused data plans (B312/B316)
# --------------------------------------------------------------------------
#
# There was a `rejected_data_plans` counter here, tripping on the SECOND plan the validator refused.
# It could never fire. It only made sense while `iterate` absorbed a refused decision and substituted
# a fallback intervention, which is how run 38658213 spent five iterations on plans nobody could run;
# now the FIRST refusal raises `OrchestratorDecisionError` and the run stops there, so a counter
# waiting for a repeat is unreachable code with a test proving it works.
#
# The contract that replaced it is pinned in `tests/nodes/test_iterate_rejected_data_plan.py` and
# `tests/nodes/test_iterate_no_fallback.py` instead.


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


def test_the_health_block_is_replaced_rather_than_mutated_in_place():
    """LangGraph propagates a channel only when the value it holds is a NEW object. Incrementing
    the existing dict would leave every counter at its checkpointed value on the next node, so the
    second empty rebuild would never be seen as the second one (B122)."""
    state: dict = {}
    _observe(state, _record(iteration=1))
    first = state["run_health"]
    # A second observation, still inside the budget, must hand back a DIFFERENT dict and leave the
    # first one untouched. Asserting this without raising keeps the property under test independent
    # of what the budget happens to be.
    _observe(state, _record(iteration=2))
    assert state["run_health"] is not first
    assert first["empty_mining"] == 1
    assert state["run_health"]["empty_mining"] == 2


def test_the_summary_is_empty_before_anything_has_been_observed():
    """An empty table under a heading reads as "nothing happened"; nothing at all is honest."""
    assert format_health_summary({}) == []
    assert format_health_summary({"run_health": {"history": []}}) == []


def test_the_summary_renders_one_row_per_iteration_and_counts_the_wasted_ones():
    state: dict = {}
    _observe(state, _record(iteration=1, strategy="initial_gold", rows_added=3000,
                            curriculum_before=0, curriculum_after=3000))
    _observe(state, _record(iteration=2, strategy="mine_new_real", rows_added=250,
                            curriculum_before=3000, curriculum_after=3250, qc_removed=12))
    _observe(state, _record(iteration=3, strategy="surgical_synthesis", rows_added=0,
                            curriculum_before=3250, curriculum_after=3250,
                            verify_attempted=300, verify_kept=0))

    lines = format_health_summary(state)
    body = "\n".join(lines)
    assert "Curriculum growth per iteration:" in lines[0]
    # One row per observed iteration, plus the header line and the header row.
    assert len(lines) == 2 + 3 + 1
    assert "initial_gold" in body and "mine_new_real" in body and "surgical_synthesis" in body
    assert "3000 → 3250" in body
    # Kept/attempted, so a rebuild that generated 300 rows and kept none cannot be read as a
    # rebuild that generated nothing.
    assert "0/300" in body
    # The count that answers the question the table exists for: did the interventions add data.
    assert "1 of 2 rebuild(s) added no rows" in body
    assert "wasted train+eval cycle" in body


def test_the_summary_does_not_flag_wasted_cycles_when_there_were_none():
    state: dict = {}
    _observe(state, _record(iteration=1, strategy="mine_new_real", rows_added=250))
    body = "\n".join(format_health_summary(state))
    assert "0 of 1 rebuild(s) added no rows" in body
    assert "wasted train+eval cycle" not in body


def test_the_history_is_bounded_so_it_does_not_grow_across_the_checkpoint_boundary():
    """The ledger is for the final report, not an audit log, and it is written to the LangGraph
    channel on every iteration."""
    state: dict = {}
    for iteration in range(1, 60):
        _observe(state, _record(iteration=iteration, strategy="mine_new_real", rows_added=10))
    assert len(state["run_health"]["history"]) == 40


def test_the_health_state_survives_a_round_trip_through_a_plain_dict():
    """It lives on the run state so it survives a resume, which means it has to be reconstructible
    from JSON rather than from the dataclass that wrote it."""
    state: dict = {}
    _observe(state, _shutout(1))
    resumed = {"run_health": dict(state["run_health"])}
    _observe(resumed, _productive_synthesis(2))
    _observe(resumed, _shutout(3))
    assert resumed["run_health"]["mining_shutouts"] == 2
