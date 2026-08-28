"""The run report's answer to "what did each intervention actually buy?"

WHY THIS FILE EXISTS
    `agent/run_health.py` already answered "did the interventions add DATA" (see
    `tests/test_run_health.py`). It could not answer the question that follows immediately —
    "and was that data worth anything" — because the two records needed to answer it live apart:
    the DAG knows the SCORE each attempt produced and whether it survived, and
    `run_health.history` knows what the attempt did to the curriculum. `_dag_rows` joins them on
    ITERATION, `_attribute` turns the joined scores into per-strategy credit, and
    `format_health_summary` renders both.

    Three things about that join and that arithmetic are load-bearing, and each has its own
    section below.

    THE JOIN IS OFF BY ONE IF NOBODY PINS IT
        `curate` writes the curriculum that the NEXT iteration trains on, so it records
        `iteration = state["iteration"] + 1`. Recording the CURRENT value made every ledger row
        join to the previous iteration's score — a synthesis round's rows were reported against a
        hyperparameter step's result, which is worse than reporting nothing because it reads as
        evidence. `test_a_ledger_entry_lands_on_the_iteration_it_will_train` is the regression
        test, and the self-check tests around it pin the behaviour that makes a future drift
        show up as a BLANK row rather than as another iteration's numbers.

    THE DELTA IS MEASURED AGAINST THE LAST KEPT SCORE
        A pruned attempt is rolled back, so the next attempt starts from the last KEPT
        checkpoint. Measuring against the previous ITERATION instead credits an attempt with the
        difference between two discarded experiments.

    THE TOTALS HAVE TO ADD UP
        `final == start + sum(gains)` exactly. That identity is the whole reason the table can be
        audited: a strategy's number is the accuracy the run RETAINED from it, not the accuracy it
        transiently reached. Without it the block is a set of plausible-looking numbers nobody can
        check.
"""
from __future__ import annotations

import re

import pytest

from agent.run_health import _attribute, _dag_rows, format_health_summary

DEFAULT_SELECTOR = "Qwen/Qwen3-1.7B@Q4_K_M"


def _node(
    iteration: int,
    *,
    score: float | None,
    strategy: str | None = None,
    intervention: str = "data_rebuild",
    pruned: bool = False,
    selector: str = DEFAULT_SELECTOR,
    tier: int | None = 1,
    total_examples: int | None = None,
) -> dict:
    """One DAG node in the shape `evaluate_node` appends.

    `strategy` becomes `pi.D.plan.strategy` — the plan the orchestrator actually authorised, which
    is the only trustworthy statement of what this iteration did to the data. A hyperparameter step
    carries no plan at all, which is why `strategy=None` is the default.
    """
    return {
        "iteration": iteration,
        "selector": selector,
        "tier": tier,
        "score": score,
        "pruned": pruned,
        "intervention": intervention,
        "pi": {
            "D": {
                "plan": {"strategy": strategy, "rows": 400} if strategy else None,
                "composition": ({"total_examples": total_examples}
                                if total_examples is not None else None),
            },
        },
    }


def _ledger(
    iteration: int,
    strategy: str,
    *,
    rows_added: int = 0,
    before: int = 3000,
    after: int | None = None,
    qc_removed: int = 0,
    verify: tuple[int, int] = (0, 0),
) -> dict:
    """One `run_health.history` entry in the shape `observe_iteration` writes.

    `iteration` is the iteration this curriculum will be TRAINED on, i.e. curate's own iteration
    plus one. Fixtures below state it explicitly rather than deriving it, because the +1 convention
    is the thing under test and a fixture that computed it would agree with a broken production
    module by construction.
    """
    return {
        "iteration": iteration,
        "strategy": strategy,
        "rows_added": rows_added,
        "curriculum": [before, before + rows_added if after is None else after],
        "qc_removed": qc_removed,
        "firewall_removed": 0,
        "verify": list(verify),
    }


def _state(dag: list[dict], history: list[dict]) -> dict:
    return {"dag": dag, "run_health": {"history": history}}


# --------------------------------------------------------------------------
# Reading the rendered table
# --------------------------------------------------------------------------


def _table_rows(lines: list[str]) -> dict[int, str]:
    """The per-iteration rows of the table, keyed by iteration.

    Identified by the kept/rolled-back column rather than by position, so a heading, the
    wasted-rebuild count, or the contribution block cannot be mistaken for a row — the
    wasted-rebuild line also begins with a digit.
    """
    rows: dict[int, str] = {}
    for line in lines:
        if "✓ kept" in line or "✗ rolled back" in line:
            rows[int(line.split()[0])] = line
    return rows


def _headings(lines: list[str]) -> list[str]:
    return [line for line in lines if "iteration(s)) ──" in line]


def _contribution_line(lines: list[str], strategy: str) -> str:
    """The per-strategy credit line for `strategy` from the contribution block."""
    matches = [line for line in lines
               if line.strip().startswith(strategy)
               and ("attempt(s) kept" in line or "starting point" in line)]
    assert len(matches) == 1, f"expected exactly one contribution line for {strategy}: {matches}"
    return matches[0]


def _final_line(lines: list[str]) -> str:
    matches = [line for line in lines if line.strip().startswith("FINAL")]
    assert len(matches) == 1, f"expected exactly one FINAL line: {matches}"
    return matches[0]


def _floats(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"[-+]?\d+\.\d+", text)]


def _by_iteration(rows: list[dict]) -> dict[int, dict]:
    """Attributed rows keyed by iteration rather than by list position.

    The whole point of these tests is which iteration a fact belongs to, so a fixture that
    identified rows positionally would be asserting the thing it is supposed to be checking.
    """
    return {row["iteration"]: row for row in rows}


# --------------------------------------------------------------------------
# 1. The totals add up
# --------------------------------------------------------------------------


def _multi_strategy_run() -> dict:
    """A run that escalated through both sub-strategies and had two attempts rolled back.

    Deliberately mixed: two kept mining rounds, two pruned synthesis rounds, and a kept
    hyperparameter step between them. The pruned attempts are what make the identity worth
    asserting — they move the score without being allowed to contribute to it.
    """
    dag = [
        _node(1, score=0.70),
        _node(2, score=0.76, strategy="mine_new_real"),
        _node(3, score=0.71, strategy="surgical_synthesis", pruned=True),
        _node(4, score=0.78, intervention="hyperparameter"),
        _node(5, score=0.74, strategy="surgical_synthesis", pruned=True),
        _node(6, score=0.81, strategy="mine_new_real"),
    ]
    history = [
        _ledger(1, "initial_gold", rows_added=3000, before=0),
        _ledger(2, "mine_new_real", rows_added=982, before=3000),
        _ledger(3, "surgical_synthesis", rows_added=120, before=3982, verify=(120, 400)),
        _ledger(5, "surgical_synthesis", rows_added=0, before=4102, verify=(0, 380)),
        _ledger(6, "mine_new_real", rows_added=310, before=4102),
    ]
    return _state(dag, history)


def test_the_kept_gains_sum_exactly_to_the_final_score():
    """`final == start + sum(gains)`, which is what makes the block auditable.

    If the identity does not hold, every number in the contribution block is unfalsifiable: a
    reader cannot tell a mis-attributed gain from a real one, and the table's only purpose is to
    let them.
    """
    rows, contributions = _attribute(_dag_rows(_multi_strategy_run()))

    start = next(slot["start"] for slot in contributions.values() if "start" in slot)
    final = next(float(row["score"]) for row in reversed(rows)
                 if row["score"] is not None and not row["pruned"])
    gained = sum(slot["gain"] for slot in contributions.values())

    assert start == pytest.approx(0.70)
    assert final == pytest.approx(0.81)
    assert final == pytest.approx(start + gained)


def test_the_rendered_final_line_states_the_same_identity():
    """The identity has to survive rendering, not just hold internally — the line is the only form
    of it a human reading the run report ever sees."""
    lines = format_health_summary(_multi_strategy_run())
    final, start, gained = _floats(_final_line(lines))
    assert final == pytest.approx(0.81)
    assert final == pytest.approx(start + gained)


def test_only_kept_steps_accumulate_into_a_strategy_total():
    """Two mining rounds were kept (+0.06 and +0.03) and two synthesis rounds were rolled back.

    Mining's credit is the sum of what it retained; synthesis's is zero, because a rolled-back
    attempt left the run exactly where it found it however good its transient score looked.
    """
    _rows, contributions = _attribute(_dag_rows(_multi_strategy_run()))
    assert contributions["mine_new_real"]["gain"] == pytest.approx(0.09)
    assert contributions["surgical_synthesis"]["gain"] == pytest.approx(0.0)
    assert contributions["hyperparameter"]["gain"] == pytest.approx(0.02)


# --------------------------------------------------------------------------
# 2. The delta is measured against the last KEPT score
# --------------------------------------------------------------------------


def test_the_delta_is_measured_against_the_last_kept_score_not_the_previous_iteration():
    """kept(0.80) → pruned(0.70) → kept(0.82): the third attempt gained +0.02, not +0.12.

    The pruned attempt was rolled back, so the third attempt trained from the 0.80 checkpoint and
    never saw 0.70. Measuring against the previous ITERATION would credit it with recovering
    ground that was never actually lost — a strategy that follows a bad attempt would look good
    for doing nothing, which is the most misleading possible reading of this table.
    """
    state = _state(
        [
            _node(1, score=0.80),
            _node(2, score=0.70, strategy="surgical_synthesis", pruned=True),
            _node(3, score=0.82, strategy="mine_new_real"),
        ],
        [
            _ledger(1, "initial_gold", rows_added=3000, before=0),
            _ledger(2, "surgical_synthesis", rows_added=200, before=3000),
            _ledger(3, "mine_new_real", rows_added=150, before=3200),
        ],
    )
    rows, contributions = _attribute(_dag_rows(state))
    attributed = _by_iteration(rows)

    assert attributed[2]["delta"] == pytest.approx(-0.10)
    assert attributed[3]["delta"] == pytest.approx(0.02)
    assert attributed[3]["delta"] != pytest.approx(0.12)
    assert contributions["mine_new_real"]["gain"] == pytest.approx(0.02)


def test_a_pruned_attempt_contributes_nothing_but_still_counts_as_an_attempt():
    """Cost and benefit are separate columns. A rolled-back attempt bought no accuracy, and it
    still spent a train+eval cycle — a strategy shown as `0/3 attempt(s) kept` is being reported
    as expensive AND useless, which is different from not having been tried."""
    state = _state(
        [
            _node(1, score=0.80),
            _node(2, score=0.70, strategy="surgical_synthesis", pruned=True),
            _node(3, score=0.75, strategy="surgical_synthesis", pruned=True),
        ],
        [
            _ledger(1, "initial_gold", rows_added=3000, before=0),
            _ledger(2, "surgical_synthesis", rows_added=200, before=3000, verify=(200, 480)),
            _ledger(3, "surgical_synthesis", rows_added=180, before=3200, verify=(180, 460)),
        ],
    )
    _rows, contributions = _attribute(_dag_rows(state))
    assert contributions["surgical_synthesis"]["gain"] == pytest.approx(0.0)
    assert contributions["surgical_synthesis"]["attempts"] == 2
    assert contributions["surgical_synthesis"]["kept"] == 0

    line = _contribution_line(format_health_summary(state), "surgical_synthesis")
    assert "0/2 attempt(s) kept" in line
    assert _floats(line) == [pytest.approx(0.0)]


# --------------------------------------------------------------------------
# 3. Grouped by model, because a run that escalates is several experiments
# --------------------------------------------------------------------------


def test_rows_from_two_models_are_grouped_under_separate_headings_naming_the_tier():
    """Pooling deltas across models averages over different ceilings: +0.02 on a 400MB variant and
    +0.02 on a 2.2GB one are not the same result, and a single table implies they are. The tier is
    on the heading because it is what says which of those two a reader is looking at."""
    state = _state(
        [
            _node(1, score=0.62, selector="Qwen/Qwen3-0.6B@Q4_K_M", tier=0),
            _node(2, score=0.66, strategy="mine_new_real",
                  selector="Qwen/Qwen3-0.6B@Q4_K_M", tier=0),
            _node(3, score=0.74, selector="Qwen/Qwen3-4B-Instruct-2507@Q4_K_M", tier=3),
            _node(4, score=0.79, strategy="surgical_synthesis",
                  selector="Qwen/Qwen3-4B-Instruct-2507@Q4_K_M", tier=3),
        ],
        [
            _ledger(1, "initial_gold", rows_added=3000, before=0),
            _ledger(2, "mine_new_real", rows_added=400, before=3000),
            _ledger(4, "surgical_synthesis", rows_added=250, before=3400, verify=(250, 300)),
        ],
    )
    lines = format_health_summary(state)
    headings = _headings(lines)

    assert len(headings) == 2
    small, large = headings
    assert "Qwen/Qwen3-0.6B@Q4_K_M" in small and "Tier 0" in small
    assert "Qwen/Qwen3-4B-Instruct-2507@Q4_K_M" in large and "Tier 3" in large

    # Each model's iterations sit under ITS heading. A correct pair of headings above a single
    # pooled block would satisfy the assertions above and report the wrong thing.
    body = "\n".join(lines)
    first, second = body.index(small), body.index(large)
    for iteration, line in _table_rows(lines).items():
        position = body.index(line)
        assert (first < position < second) == (iteration <= 2), (
            f"iteration {iteration} is grouped under the wrong model"
        )


def test_a_model_with_no_recorded_tier_still_gets_its_own_heading():
    """`tier` is absent from a DAG node written before it was recorded, and a resumed run reports
    on those iterations too. Grouping must degrade to "no tier shown", not to "no heading"."""
    state = _state(
        [_node(1, score=0.62, selector="Qwen/Qwen3-0.6B@Q4_K_M", tier=None)],
        [_ledger(1, "initial_gold", rows_added=3000, before=0)],
    )
    headings = _headings(format_health_summary(state))
    assert len(headings) == 1
    assert "Qwen/Qwen3-0.6B@Q4_K_M" in headings[0]


# --------------------------------------------------------------------------
# 4. The join: on iteration, and self-checking
# --------------------------------------------------------------------------


def test_a_ledger_entry_lands_on_the_iteration_it_will_train():
    """The regression test for the off-by-one join.

    `curate` runs BEFORE the iteration counter advances, so the curriculum it writes is trained on
    the NEXT iteration and it records `state["iteration"] + 1`. Here iteration 2 is a
    hyperparameter step (curate did not run for it) and the curate DURING iteration 2 planned the
    surgical synthesis that iteration 3 trains on.

    Recording the current value instead attributed those 400 generated rows and their 120 keeps to
    the hyperparameter step, and left the synthesis round — the one that actually spent the teacher
    budget — showing nothing. The reader is then told a hyperparameter change generated training
    data, which is not a smaller error than a blank row; it is a confident wrong answer.
    """
    state = _state(
        [
            _node(1, score=0.7000),
            _node(2, score=0.7200, intervention="hyperparameter"),
            _node(3, score=0.6900, strategy="surgical_synthesis", pruned=True),
        ],
        [
            _ledger(1, "initial_gold", rows_added=3000, before=0),
            _ledger(3, "surgical_synthesis", rows_added=120, before=3000, qc_removed=7,
                    verify=(120, 400)),
        ],
    )
    rows = _table_rows(format_health_summary(state))

    assert "120/400" in rows[3]
    assert "+120" in rows[3]
    assert "surgical_synthesis" in rows[3]

    assert "120/400" not in rows[2]
    assert "+120" not in rows[2]
    assert "hyperparameter" in rows[2]


def test_a_ledger_entry_whose_strategy_contradicts_the_plan_is_dropped():
    """The join cross-checks itself instead of trusting the iteration numbers.

    The two records are written by different nodes at different points in the turn. If either
    side's notion of "iteration" drifts again, the mismatch has to surface as a BLANK row — a
    reader who sees `—` goes and looks, while a reader who sees another iteration's 982 rows
    attributed to this strategy draws a conclusion from it.
    """
    state = _state(
        [
            _node(1, score=0.70),
            _node(2, score=0.76, strategy="surgical_synthesis"),
        ],
        [
            _ledger(1, "initial_gold", rows_added=3000, before=0),
            _ledger(2, "mine_new_real", rows_added=982, before=3000, qc_removed=41,
                    verify=(0, 0)),
        ],
    )
    row = _table_rows(format_health_summary(state))[2]

    # The plan is what says what this iteration did; the contradicting ledger contributes nothing.
    assert "surgical_synthesis" in row
    assert "mine_new_real" not in row
    assert "982" not in row
    assert "41" not in row


def test_a_matching_ledger_entry_survives_the_self_check():
    """The other half of the contract. A cross-check that dropped agreeing entries too would blank
    the whole table and look exactly like a run that recorded nothing."""
    state = _state(
        [
            _node(1, score=0.70),
            _node(2, score=0.76, strategy="mine_new_real"),
        ],
        [
            _ledger(1, "initial_gold", rows_added=3000, before=0),
            _ledger(2, "mine_new_real", rows_added=982, before=3000, qc_removed=41),
        ],
    )
    row = _table_rows(format_health_summary(state))[2]
    assert "+982" in row
    assert "3000 → 3982" in row


def test_a_hyperparameter_step_is_never_credited_with_the_previous_rebuilds_strategy():
    """A hyperparameter step carries no plan, and the ledger's `strategy` is whatever the last
    curate wrote — for a hyperparameter turn, the PREVIOUS rebuild's. Reading that as this
    iteration's cause is how a run reports mining rounds it never ran on iterations that only
    changed the learning rate."""
    state = _state(
        [
            _node(1, score=0.70),
            _node(2, score=0.74, intervention="hyperparameter"),
        ],
        [
            _ledger(1, "initial_gold", rows_added=3000, before=0),
            _ledger(2, "mine_new_real", rows_added=982, before=3000),
        ],
    )
    rows, contributions = _attribute(_dag_rows(state))

    assert _by_iteration(rows)[2]["strategy"] == "hyperparameter"
    assert "mine_new_real" not in contributions
    assert "mine_new_real" not in _table_rows(format_health_summary(state))[2]


def test_an_iteration_with_no_ledger_entry_renders_blanks_rather_than_borrowing_one():
    """A resumed run's history is bounded to the last 40 entries, so early iterations legitimately
    have a score and no ledger. Those rows must be visibly empty."""
    state = _state(
        [_node(4, score=0.70, strategy="mine_new_real")],
        [_ledger(9, "mine_new_real", rows_added=982, before=3000)],
    )
    row = _table_rows(format_health_summary(state))[4]
    assert "982" not in row
    assert "—" in row


# --------------------------------------------------------------------------
# 5. No DAG at all
# --------------------------------------------------------------------------


def test_a_run_with_no_dag_falls_back_to_the_data_only_ledger():
    """A run that died in cold start never scored anything, so there is nothing to attribute — but
    it still built a curriculum, and that is the evidence for why it died. The fallback heading
    says outright that no attribution is available, so an absent Δ column is not read as zero."""
    state = _state([], [_ledger(1, "initial_gold", rows_added=3000, before=0),
                        _ledger(2, "mine_new_real", rows_added=0, before=3000)])
    lines = format_health_summary(state)

    assert lines[0].startswith("  Curriculum growth per iteration:")
    body = "\n".join(lines)
    assert "no score attribution" in body
    assert "1 of 1 rebuild(s) added no rows" in body


def test_a_dag_present_but_scoreless_still_attributes_nothing_rather_than_zero():
    """An iteration whose eval never completed has `score=None`. Rendering that as 0.0000 would
    put a catastrophic regression in the table and in the strategy's total."""
    state = _state(
        [_node(1, score=0.70), _node(2, score=None, strategy="mine_new_real")],
        [_ledger(1, "initial_gold", rows_added=3000, before=0),
         _ledger(2, "mine_new_real", rows_added=982, before=3000)],
    )
    rows, contributions = _attribute(_dag_rows(state))

    assert _by_iteration(rows)[2]["delta"] is None
    assert contributions["mine_new_real"]["gain"] == pytest.approx(0.0)
    assert contributions["mine_new_real"]["attempts"] == 1
    final, start, gained = _floats(_final_line(format_health_summary(state)))
    assert final == pytest.approx(0.70)
    assert final == pytest.approx(start + gained)
