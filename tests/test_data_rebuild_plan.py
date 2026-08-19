"""The `data_rebuild` plan contract: two strategies, and nothing the orchestrator asked for is ignored.

WHY THIS FILE EXISTS
    The plan is the orchestrator's only lever on the curriculum, and the failure mode is silence.
    Several fields were removed on 2026-08-19 because they could not add information:

      * `resample` re-drew rows from the pool the curriculum was already built from — one traced
        rebuild resampled 3,308 rows of which 122 were novel;
      * the universal gold FILL re-selected the identical ~3,235 rows and honestly reported
        `0 novel` on eight consecutive rebuilds of one run;
      * `target_rows` presumed a target the cumulative curriculum no longer has.

    A prompt is a long-lived artifact and an orchestrator writing against the old contract is the
    expected case, not an unlikely one. So a removed field must be REJECTED rather than dropped:
    silently ignoring it lets the orchestrator believe it asked for something, and the run then
    behaves in a way its own recorded reasoning does not explain.
"""
from __future__ import annotations

import pytest

from agent.data_rebuild import (
    DATA_REBUILD_SCHEMA_VERSION,
    DATA_REBUILD_STRATEGIES,
    MAX_REBUILD_ROWS,
    MIN_REBUILD_ROWS,
    MINE_NEW_REAL,
    SURGICAL_SYNTHESIS,
    fallback_data_rebuild_plan,
    normalize_data_rebuild_plan,
)

# Every field the removed strategies used to carry. Each one is a plan written against a contract
# that no longer exists.
REMOVED_FIELDS = {
    "target_rows": 1200,
    "synth_rows": 300,
    "new_real_rows": 300,
    "resample_fraction": 0.5,
    "max_acquire_rounds": 3,
    "difficulty_buckets": {"easy": 0.5, "hard": 0.5},
    "confusion_pairs": [{"gold": "a", "predicted": "b"}],
}


def _plan(**overrides):
    plan = {
        "schema_version": DATA_REBUILD_SCHEMA_VERSION,
        "strategy": SURGICAL_SYNTHESIS,
        "rows": 300,
        "target_categories": [],
        "pattern_hint": "",
    }
    plan.update(overrides)
    return plan


def _normalize(**overrides):
    return normalize_data_rebuild_plan(_plan(**overrides), task="xlam_bfcl")


# --------------------------------------------------------------------------
# Exactly two strategies
# --------------------------------------------------------------------------


def test_there_are_exactly_two_strategies():
    assert DATA_REBUILD_STRATEGIES == (MINE_NEW_REAL, SURGICAL_SYNTHESIS)


@pytest.mark.parametrize("strategy", [MINE_NEW_REAL, SURGICAL_SYNTHESIS])
def test_both_strategies_are_accepted(strategy):
    assert _normalize(strategy=strategy)["strategy"] == strategy


@pytest.mark.parametrize("strategy", ["resample", "synthesize", "acquire", "", None, 3])
def test_a_strategy_that_no_longer_exists_is_rejected(strategy):
    """`resample` and the balanced `synthesize` fill were both removed. Accepting either would run
    a mechanism that cannot add information while reporting that it did."""
    with pytest.raises(ValueError, match="must be one of"):
        _normalize(strategy=strategy)


def test_the_rejection_names_the_strategies_that_do_exist():
    """The orchestrator is re-asked with this message, so it has to be actionable."""
    with pytest.raises(ValueError) as excinfo:
        _normalize(strategy="resample")
    message = str(excinfo.value)
    assert MINE_NEW_REAL in message and SURGICAL_SYNTHESIS in message


# --------------------------------------------------------------------------
# Removed fields are rejected, not ignored
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(REMOVED_FIELDS))
def test_every_removed_field_is_rejected(field):
    with pytest.raises(ValueError, match=r"unknown field\(s\)"):
        _normalize(**{field: REMOVED_FIELDS[field]})


def test_the_rejection_names_the_offending_field_and_the_allowed_set():
    with pytest.raises(ValueError) as excinfo:
        _normalize(target_rows=1200)
    message = str(excinfo.value)
    assert "target_rows" in message
    assert "strategy" in message and "rows" in message


def test_an_unrecognised_field_is_rejected_even_alongside_a_valid_plan():
    """The dangerous case: the plan is otherwise perfect, so dropping the extra field would look
    like success."""
    with pytest.raises(ValueError, match=r"unknown field\(s\)"):
        _normalize(strategy=MINE_NEW_REAL, rows=500, seed=1234)


def test_a_plan_that_is_not_an_object_is_rejected():
    with pytest.raises(ValueError, match="must be an object"):
        normalize_data_rebuild_plan(["surgical_synthesis"], task="xlam_bfcl")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Schema version
# --------------------------------------------------------------------------


def test_the_schema_version_is_three():
    assert DATA_REBUILD_SCHEMA_VERSION == 3
    assert _normalize()["schema_version"] == 3


@pytest.mark.parametrize("version", [1, 2, 4])
def test_a_plan_at_another_schema_version_is_rejected(version):
    """A version bump is how the removals are announced; accepting v2 would accept the contract
    those removals were made from."""
    with pytest.raises(ValueError, match="schema_version must be 3"):
        _normalize(schema_version=version)


def test_an_absent_schema_version_defaults_to_the_current_one():
    plan = _plan()
    del plan["schema_version"]
    assert normalize_data_rebuild_plan(plan, task="xlam_bfcl")["schema_version"] == 3


# --------------------------------------------------------------------------
# `rows` is clamped
# --------------------------------------------------------------------------


def test_rows_is_clamped_to_the_floor():
    """The floor stops the orchestrator spending a whole train+eval cycle on a handful of rows."""
    assert _normalize(rows=1)["rows"] == MIN_REBUILD_ROWS
    assert _normalize(rows=-500)["rows"] == MIN_REBUILD_ROWS


def test_rows_is_clamped_to_the_ceiling():
    """The ceiling stops one turn dominating the run."""
    assert _normalize(rows=10**9)["rows"] == MAX_REBUILD_ROWS


def test_the_bounds_are_fifty_and_two_thousand():
    assert (MIN_REBUILD_ROWS, MAX_REBUILD_ROWS) == (50, 2000)


def test_a_row_count_inside_the_bounds_is_preserved():
    assert _normalize(rows=750)["rows"] == 750


def test_a_fractional_row_count_is_rounded_rather_than_truncated():
    assert _normalize(rows=300.6)["rows"] == 301


def test_an_absent_row_count_gets_a_bounded_default():
    plan = _plan()
    del plan["rows"]
    rows = normalize_data_rebuild_plan(plan, task="xlam_bfcl")["rows"]
    assert MIN_REBUILD_ROWS <= rows <= MAX_REBUILD_ROWS


@pytest.mark.parametrize("value", ["300", True, [300]])
def test_a_non_numeric_row_count_is_rejected(value):
    """`True` included deliberately: it is an `int` in Python, so without the explicit bool check it
    would silently clamp to the floor and the rebuild would ask for 50 rows."""
    with pytest.raises(ValueError, match="must be a number"):
        _normalize(rows=value)


# --------------------------------------------------------------------------
# Target categories
# --------------------------------------------------------------------------


def test_target_categories_are_carried_with_their_counts():
    """The categories come from the task's own failure taxonomy via the test report, so they name
    something the scorer measured rather than a class the orchestrator invented (B296)."""
    plan = _normalize(target_categories=[
        {"category": "wrong_arguments", "count": 147},
        {"category": "unparseable_output", "count": 12},
    ])
    assert plan["target_categories"] == [
        {"category": "wrong_arguments", "count": 147},
        {"category": "unparseable_output", "count": 12},
    ]


def test_target_categories_are_bounded():
    from agent.data_rebuild import MAX_TARGET_CATEGORIES

    plan = _normalize(target_categories=[
        {"category": f"cat_{i}", "count": i} for i in range(50)
    ])
    assert len(plan["target_categories"]) == MAX_TARGET_CATEGORIES


def test_a_target_category_with_no_name_is_dropped():
    plan = _normalize(target_categories=[
        {"category": "", "count": 5}, {"category": "wrong_function", "count": 3},
    ])
    assert [entry["category"] for entry in plan["target_categories"]] == ["wrong_function"]


def test_target_categories_that_are_not_a_list_are_rejected():
    with pytest.raises(ValueError, match="must be a list"):
        _normalize(target_categories={"category": "x"})


def test_a_target_category_that_is_not_an_object_is_rejected():
    with pytest.raises(ValueError, match="must be an object"):
        _normalize(target_categories=["wrong_arguments"])


def test_absent_target_categories_normalize_to_an_empty_list():
    assert _normalize(target_categories=None)["target_categories"] == []


# --------------------------------------------------------------------------
# Free text is bounded
# --------------------------------------------------------------------------


def test_the_pattern_hint_is_carried_and_bounded():
    from agent.data_rebuild import PATTERN_HINT_MAX_CHARS

    assert _normalize(pattern_hint="  favour multi-call rows  ")["pattern_hint"] == (
        "favour multi-call rows"
    )
    with pytest.raises(ValueError, match="at most"):
        _normalize(pattern_hint="x" * (PATTERN_HINT_MAX_CHARS + 1))


def test_the_hypothesis_is_carried_in_full_up_to_its_ceiling():
    """The hypothesis is the orchestrator's causal reasoning and the single most information-dense
    field in the run; it was being cut in five separate places (B238)."""
    from agent.data_rebuild import HYPOTHESIS_MAX_CHARS

    reasoning = "because " * 100
    plan = normalize_data_rebuild_plan(_plan(), task="xlam_bfcl", hypothesis=reasoning)
    assert plan["hypothesis"] == reasoning.strip()

    with pytest.raises(ValueError, match="at most"):
        normalize_data_rebuild_plan(
            _plan(), task="xlam_bfcl", hypothesis="x" * (HYPOTHESIS_MAX_CHARS + 1),
        )


def test_the_plan_records_the_task_it_belongs_to():
    assert normalize_data_rebuild_plan(_plan(), task="ner_bc5cdr")["task"] == "ner_bc5cdr"


def test_the_normalized_plan_has_exactly_the_expected_keys():
    """A plan is written to the artifact and read back on resume, so an extra or missing key is a
    format change nothing announced."""
    assert set(_normalize()) == {
        "schema_version", "strategy", "rows", "target_categories",
        "pattern_hint", "hypothesis", "task",
    }


def test_the_plan_carries_no_seed_and_no_dedup_identity():
    """Curation is deliberately non-deterministic and the orchestrator freely re-picks a strategy
    each turn; escalation-on-no-improvement is the sole stuck-run backstop."""
    plan = _normalize()
    for gone in ("seed", "plan_identity", "dedup_identity"):
        assert gone not in plan


# --------------------------------------------------------------------------
# Mining that cannot work is rewritten rather than run
# --------------------------------------------------------------------------


def test_mine_new_real_is_rewritten_when_mining_cannot_add_a_row():
    """Running it anyway spends a full train+eval cycle on an intervention that provably cannot add
    a row, and the iteration then repeats the previous one exactly."""
    plan = normalize_data_rebuild_plan(
        _plan(strategy=MINE_NEW_REAL), task="xlam_bfcl", mining_available=False,
    )
    assert plan["strategy"] == SURGICAL_SYNTHESIS


def test_mine_new_real_survives_while_mining_is_available():
    plan = normalize_data_rebuild_plan(
        _plan(strategy=MINE_NEW_REAL), task="xlam_bfcl", mining_available=True,
    )
    assert plan["strategy"] == MINE_NEW_REAL


def test_surgical_synthesis_is_never_rewritten():
    """It is the terminal strategy: there is nothing to fall back to."""
    for available in (True, False):
        plan = normalize_data_rebuild_plan(
            _plan(strategy=SURGICAL_SYNTHESIS), task="xlam_bfcl",
            mining_available=available,
        )
        assert plan["strategy"] == SURGICAL_SYNTHESIS


def test_the_row_count_and_targets_survive_the_rewrite():
    """The rewrite changes the MECHANISM, not the size or aim of the rebuild."""
    plan = normalize_data_rebuild_plan(
        _plan(strategy=MINE_NEW_REAL, rows=800,
              target_categories=[{"category": "wrong_arguments", "count": 9}]),
        task="xlam_bfcl", mining_available=False,
    )
    assert plan["rows"] == 800
    assert plan["target_categories"] == [{"category": "wrong_arguments", "count": 9}]


# --------------------------------------------------------------------------
# The deterministic fallback
# --------------------------------------------------------------------------


def test_the_fallback_prefers_real_rows_while_a_source_has_them():
    """Real data is free of teacher error, and on this project a gold-only curriculum produced the
    best result anyone has measured (BC5CDR, 0.8098)."""
    plan = fallback_data_rebuild_plan({
        "task": "xlam_bfcl",
        "source_progress": {"Salesforce/xlam-function-calling-60k": {"consumed": 3250}},
    })
    assert plan["strategy"] == MINE_NEW_REAL


def test_the_fallback_switches_to_synthesis_once_mining_is_retired():
    plan = fallback_data_rebuild_plan({
        "task": "xlam_bfcl",
        "source_progress": {"Salesforce/xlam-function-calling-60k": {"exhausted": True}},
        "failed_discovery_rounds": 2,
    })
    assert plan["strategy"] == SURGICAL_SYNTHESIS


def test_the_fallback_aims_at_whatever_the_last_report_says_is_failing():
    plan = fallback_data_rebuild_plan({
        "task": "xlam_bfcl",
        "source_progress": {"src": {"exhausted": True}},
        "failed_discovery_rounds": 2,
        "test_report": {"confusion_pairs": [
            {"gold": "wrong_arguments", "count": 40},
            {"gold": "unparseable_output", "count": 8},
        ]},
    })
    assert [entry["category"] for entry in plan["target_categories"]] == [
        "wrong_arguments", "unparseable_output",
    ]


def test_the_fallback_says_in_its_hypothesis_that_it_is_a_fallback():
    """Otherwise a run's recorded reasoning attributes a deterministic choice to the orchestrator."""
    plan = fallback_data_rebuild_plan({"task": "xlam_bfcl"})
    assert "fallback" in plan["hypothesis"].lower()


def test_iterate_calls_the_fallback_with_arguments_it_accepts():
    """`iterate` reaches the fallback precisely when the orchestrator call failed or returned
    unusable JSON, which is the one moment nothing else can rescue the run. A call site that passes
    keywords the signature does not accept turns that rescue into a TypeError — the documented
    orchestrator-failure route in docs/interventions.md section 2 then kills the run it exists to
    save, and only on the rare turn that reaches it.

    Checked as an AGREEMENT between the call site and the signature rather than by invoking either
    one, because either side can legitimately change: the callee may grow a parameter, or the call
    may drop one. A test that invoked only the callee would keep passing after the call site drifted.
    """
    import ast
    import inspect

    from agent.nodes import iterate

    accepted = set(inspect.signature(fallback_data_rebuild_plan).parameters)
    tree = ast.parse(inspect.getsource(iterate))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "fallback_data_rebuild_plan"
    ]
    assert calls, "iterate no longer has a deterministic rebuild-plan fallback at all"
    for call in calls:
        passed = {keyword.arg for keyword in call.keywords if keyword.arg}
        assert passed <= accepted, (
            f"iterate.py:{call.lineno} passes {sorted(passed - accepted)} to "
            f"fallback_data_rebuild_plan, which accepts {sorted(accepted)}"
        )


def test_the_fallback_produces_a_plan_the_normalizer_accepts():
    """It goes through `normalize_data_rebuild_plan` itself, so this is the round trip: a fallback
    that the validator would reject would fail the run at the point it was trying to rescue it."""
    plan = fallback_data_rebuild_plan({"task": "clinc150"})
    fields = {k: v for k, v in plan.items() if k not in ("hypothesis", "task")}
    assert normalize_data_rebuild_plan(fields, task="clinc150")["strategy"] == plan["strategy"]
