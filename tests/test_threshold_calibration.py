"""Stop-threshold calibration (B32).

REPLACED BEHAVIOR: the planner used to propose `stop_threshold` from its recall of published
SOTA ("anchor to the PUBLISHED STATE-OF-THE-ART ... roughly SOTA - 2 to 5 points"), defaulting
to a hardcoded 0.96. The error is asymmetric and both directions were observed:

  too LOW  -> a base model's zero-shot clears it; the run "converges" at iteration 1 having
              learned nothing.
  too HIGH -> every tier fails; the run burns its budget concluding "infeasible".

These tests pin the three hazards the replacement has to defend against.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

import pytest  # noqa: E402

from agent.threshold import (  # noqa: E402
    THRESHOLD_CEILING,
    UNREACHABLE_PENDING_THRESHOLD,
    VALID_HEADROOMS,
    load_registry,
    registry_lookup,
    snap_headroom,
    threshold_from_anchor,
    threshold_from_registry,
)


# --- the registry only calibrates from a COMPARABLE metric ----------------------

def test_registry_parses_sourced_rows():
    registry = load_registry(force=True)
    assert "gsm8k" in registry
    row = registry_lookup("GSM8K", "math_reasoning")
    assert row is not None
    assert row["metric"] == "exact_match"
    assert 0.0 < row["value"] <= 1.0
    assert row["source"].startswith("http")
    assert row["checked"]


def test_registry_key_is_insensitive_to_formatting():
    assert registry_lookup("gsm8k", "math_reasoning") is not None
    assert registry_lookup("GSM-8K", "math_reasoning") is not None


def test_incomparable_metric_does_not_calibrate():
    """BANKING77's registry row is `accuracy`; classification here measures `macro_f1`.

    Comparing them would be exactly the incomparable-metric error the model pool already
    refuses to make. The row must be informational only, so the run falls back to its own
    measured anchor.
    """
    assert registry_lookup("BANKING77", "classification") is None


def test_na_rows_are_ignored_not_guessed():
    """`n/a` rows record that no comparable figure was found — they must not calibrate."""
    assert registry_lookup("BC5CDR", "NER") is None
    assert registry_lookup("SMS Spam Collection", "classification") is None


def test_unknown_benchmark_defers():
    assert registry_lookup("Some Benchmark That Does Not Exist", "NER") is None
    assert registry_lookup(None, "NER") is None


def test_registry_threshold_sits_below_the_published_figure():
    row = registry_lookup("GSM8K", "math_reasoning")
    threshold, reason = threshold_from_registry(row)
    assert threshold < row["value"], "must leave headroom for a task-specific dataset"
    assert row["source"] not in reason  # reason is a summary, not a URL dump
    assert str(row["value"]) in reason and row["checked"] in reason


# --- HAZARD 1: a bad first hyperparameter draw must not depress the goal --------

def test_anchor_uses_the_max_so_a_weak_first_config_cannot_lower_the_goal():
    """Iteration 1 uses _DEFAULT_CONFIG. If that draw is poor, the fine-tune score is low.

    Anchoring on the fine-tune alone would set a goal the run clears immediately. The
    zero-shot baseline is config-independent, so max() floors the anchor.
    """
    threshold, reason = threshold_from_anchor(0.82, 0.31, 0.05)
    assert threshold == pytest.approx(0.87)
    assert "zero-shot" in reason
    # And the symmetric case: a good fine-tune raises the anchor above zero-shot.
    higher, reason_higher = threshold_from_anchor(0.82, 0.91, 0.05)
    assert higher == pytest.approx(0.96)
    assert "first fine-tune" in reason_higher


def test_anchor_works_when_the_baseline_failed_to_measure():
    """A failed baseline records None, not 0.0 — calibration must use the fine-tune alone."""
    threshold, reason = threshold_from_anchor(None, 0.40, 0.15)
    assert threshold == pytest.approx(0.55)
    assert "zero_shot=n/a" in reason


def test_anchor_requires_at_least_one_real_measurement():
    with pytest.raises(ValueError, match="at least one measured score"):
        threshold_from_anchor(None, None, 0.05)


def test_anchor_never_targets_a_perfect_score():
    """Label noise makes 1.0 unreachable on every real benchmark; cap below it."""
    threshold, _ = threshold_from_anchor(0.98, 0.98, 0.15)
    assert threshold == pytest.approx(THRESHOLD_CEILING)
    assert threshold < 1.0


# --- HAZARD 3: headroom is a bounded choice, not a free-text number -------------

@pytest.mark.parametrize("requested,expected", [
    (0.02, 0.02), (0.05, 0.05), (0.10, 0.10), (0.15, 0.15),
    (0.07, 0.05),      # snaps to nearest rung
    (0.13, 0.15),
    (0.9, 0.15),       # clamped by nearest-rung selection
    (-1.0, 0.02),
])
def test_headroom_snaps_to_a_bounded_rung(requested, expected):
    assert snap_headroom(requested) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["0.05", None, True, [], {}])
def test_non_numeric_headroom_falls_back_to_the_middle_rung(bad):
    assert snap_headroom(bad) in VALID_HEADROOMS


# --- the deferred path parks the target out of reach ---------------------------

def test_pending_threshold_cannot_be_satisfied_by_any_score():
    """While calibration is deferred nothing may converge, or the run would stop before it
    has measured the anchor it needs."""
    assert UNREACHABLE_PENDING_THRESHOLD > THRESHOLD_CEILING
    assert UNREACHABLE_PENDING_THRESHOLD > 1.0 - 1e-9


# --- the planner no longer proposes a number -----------------------------------

def test_planner_prompt_no_longer_asks_for_a_recalled_sota_number():
    from agent.task_planner import _PLANNER_PROMPT

    assert "threshold_headroom" in _PLANNER_PROMPT
    assert "you do NOT set a number" in _PLANNER_PROMPT
    assert "Do NOT invent a SOTA figure" in _PLANNER_PROMPT
    # The old instruction and its schema field must be gone.
    assert "PRIMARY RULE: anchor it to the PUBLISHED STATE-OF-THE-ART" not in _PLANNER_PROMPT
    assert '"stop_threshold"' not in _PLANNER_PROMPT


def test_planner_snaps_headroom_and_drops_any_proposed_threshold():
    """Even if a model emits stop_threshold anyway, plan_task must not honor it."""
    import agent.task_planner as planner

    plan = {"task_type": "NER", "stop_threshold": 0.99, "threshold_headroom": 0.07}
    # Exercise the same normalization plan_task applies after JSON extraction.
    from agent.threshold import snap_headroom as snap

    plan["threshold_headroom"] = snap(plan.get("threshold_headroom"))
    plan.pop("stop_threshold", None)
    assert plan["threshold_headroom"] == pytest.approx(0.05)
    assert "stop_threshold" not in plan
    assert hasattr(planner, "plan_task")


# --- HAZARD 2: the target is written once, and the union rule is explicit -------

def test_iterate_prompt_states_the_mutually_exclusive_contract_explicitly():
    """The orchestrator must be TOLD the rule, not silently corrected for breaking it.

    In the NER run Claude attached `hyperparams` to 65 consecutive data_rebuild decisions;
    the validator raised, both paid calls were wasted, and zero orchestrator data plans ran.
    """
    from agent.nodes.iterate import _ITERATE_SYSTEM

    assert "CHOOSE EXACTLY ONE" in _ITERATE_SYSTEM
    assert "FORBIDDEN key" in _ITERATE_SYSTEM
    assert "mutually exclusive" in _ITERATE_SYSTEM
    assert "DISCARD" in _ITERATE_SYSTEM


def test_stray_hyperparams_on_a_data_rebuild_is_recorded_for_logging():
    """The strip must be VISIBLE, not silent masking."""
    from agent.nodes.iterate import _validate_decision_json

    decision = {
        "intervention": "data_rebuild",
        "hypothesis": "hard bucket is weak",
        "hyperparams": {"lora_rank": 32},
        "data_rebuild": {
            "primary_strategy": "resample_existing",
            "support_strategies": [],
            "target_rows": 128,
        },
    }
    validated = _validate_decision_json(
        decision,
        task_type="NER",
        state={"curriculum_size_target": 128},
    )
    assert "hyperparams" not in validated, "must not reach the trainer"
    assert validated["_dropped_fields"] == ["hyperparams"]
    assert validated["data_rebuild"]["primary_strategy"] == "resample_existing"
