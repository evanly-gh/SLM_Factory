"""Stretch goals: raising the accuracy target when a model converges quickly.

The accuracy goal is the Qwen-3.6 teacher's zero-shot score floored at 0.80. When the teacher
scores below the floor the floor sets the bar, so the goal reflects what we refused to go below
rather than what the task permits — BC5CDR cleared 0.8000 in five iterations with hours of budget
left. On meeting the goal the orchestrator is now asked whether to raise it.

The two properties that matter for safety are covered here: raising TERMINATES (it is a ratchet
bounded by the ceiling), and raising can never turn an already-successful run into a reported
failure (the cleared goal is banked first).
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from agent.nodes.iterate import (
    _THRESHOLD_RAISE_MIN_STEP,
    _bank_convergence,
    _maybe_raise_threshold,
    _validate_threshold_raise,
    iterate_node,
)
from agent.threshold import THRESHOLD_CEILING, describe_threshold_provenance


def _model(tier=0):
    m = MagicMock()
    m.model_id = "test/Model-1B"
    m.selector = "test/Model-1B@bf16"
    m.quant = None
    m.tier = tier
    return m


def _state(score=0.8098, threshold=0.8000, iteration=5, **over):
    state = {
        "selected_model": _model(),
        "scores": [score],
        "eval_history": [score],
        "best_score": score,
        "iteration": iteration,
        "turn_budget": 1000,
        "stop_threshold": threshold,
        "initial_stop_threshold": threshold,
        "max_stop_threshold": 0.0,
        "convergence_banked": None,
        "threshold_raises": [],
        "task": "ner_bc5cdr",
        "last_eval": None,
        "hw_gating_enabled": False,
        "hardware_constraints": MagicMock(),
        "threshold_calibration": {
            "source": "qwen_baseline",
            "threshold": threshold,
            "floor": 0.8,
            "floored": True,
            "measured_qwen": 0.0999,
            "measured_metric": "span_f1",
        },
    }
    state.update(over)
    return state


@pytest.fixture
def raising_enabled(monkeypatch):
    monkeypatch.setenv("SLM_THRESHOLD_RAISE", "1")
    monkeypatch.delenv("SLM_CHEAP", raising=False)


def _raise_to(value, reason="cleared the goal quickly with budget to spare"):
    return {"raise_goal": True, "new_threshold": value, "reason": reason}


def _intervention_decision():
    """An orchestrator intervention decision the validator accepts.

    Needed by the routing case where a raise sends the run back below its (new) goal and the turn
    continues to the intervention call.
    """
    return {
        "intervention": "hyperparameter",
        "hypothesis": "span recall is the binding limit, so the adapter needs more capacity",
        "hyperparams": {
            "lora_rank": 16,
            "alpha_ratio": 2,
            "weight_decay": 0.01,
            "learning_rate": 0.0001,
            "nr_epochs": 4,
        },
    }


# --------------------------------------------------------------------------
# Banking: a cleared goal is recorded before the bar can move
# --------------------------------------------------------------------------

def test_convergence_is_banked_before_any_raise():
    state = _state()
    _bank_convergence(state, 0.8098, "m")
    assert state["convergence_banked"]["threshold"] == 0.8
    assert state["convergence_banked"]["score"] == 0.8098
    assert state["convergence_banked"]["iteration"] == 5


def test_banking_keeps_the_highest_goal_cleared():
    state = _state()
    _bank_convergence(state, 0.8098, "m")
    state["stop_threshold"] = 0.87
    _bank_convergence(state, 0.8750, "m")
    assert state["convergence_banked"]["threshold"] == 0.87
    # A later, LOWER goal must not overwrite the better banked result.
    state["stop_threshold"] = 0.82
    _bank_convergence(state, 0.8300, "m")
    assert state["convergence_banked"]["threshold"] == 0.87


def test_raise_banks_convergence_even_when_the_orchestrator_declines(raising_enabled):
    state = _state()
    with patch("agent.nodes.iterate._llm_threshold_raise", return_value=None):
        assert _maybe_raise_threshold(state, 0.8098, "m") is False
    assert state["convergence_banked"] is not None
    assert state["stop_threshold"] == 0.8


# --------------------------------------------------------------------------
# The ratchet: raises are monotonic, bounded, and terminate
# --------------------------------------------------------------------------

def test_raise_applies_and_records_history(raising_enabled):
    state = _state()
    with patch("agent.nodes.iterate._llm_threshold_raise", return_value=_raise_to(0.87)):
        assert _maybe_raise_threshold(state, 0.8098, "m") is True
    assert state["stop_threshold"] == 0.87
    assert state["max_stop_threshold"] == 0.87
    assert state["threshold_raises"] == [{
        "from": 0.8,
        "to": 0.87,
        "score_at_raise": 0.8098,
        "iteration": 5,
        "reason": "cleared the goal quickly with budget to spare",
    }]


def test_raise_below_the_minimum_step_is_declined(raising_enabled):
    state = _state()
    tiny = 0.8 + _THRESHOLD_RAISE_MIN_STEP / 2
    with patch("agent.nodes.iterate._llm_threshold_raise", return_value=_raise_to(tiny)):
        assert _maybe_raise_threshold(state, 0.8098, "m") is False
    assert state["stop_threshold"] == 0.8


def test_raise_is_capped_at_the_ceiling(raising_enabled):
    state = _state()
    with patch("agent.nodes.iterate._llm_threshold_raise", return_value=_raise_to(1.5)):
        assert _maybe_raise_threshold(state, 0.8098, "m") is True
    assert state["stop_threshold"] == THRESHOLD_CEILING


def test_raise_cannot_reuse_a_band_below_the_high_water_mark(raising_enabled):
    """A goal LOWERED after a raise must not let the same band be re-raised into.

    Without the high-water clamp, lower-then-raise is an unbounded cycle: the orchestrator could
    lower to 0.80, clear it, raise to 0.87, lower again, and repeat forever.
    """
    state = _state(threshold=0.87, max_stop_threshold=0.87)
    state["stop_threshold"] = 0.80  # as if lowered mid-run
    with patch("agent.nodes.iterate._llm_threshold_raise", return_value=_raise_to(0.85)):
        assert _maybe_raise_threshold(state, 0.8098, "m") is False
    assert state["stop_threshold"] == 0.80


def test_repeated_raising_terminates():
    """The ratchet bounds how many raises a run can perform.

    Each raise must exceed the high-water mark by at least the minimum step and is capped at the
    ceiling, so the sequence is strictly increasing with a fixed upper bound. This asserts the
    bound directly rather than trusting the prose.
    """
    os.environ["SLM_THRESHOLD_RAISE"] = "1"
    try:
        state = _state()
        raises = 0
        # Always ask for the ceiling: the greediest possible orchestrator.
        with patch(
            "agent.nodes.iterate._llm_threshold_raise",
            return_value=_raise_to(THRESHOLD_CEILING),
        ):
            while raises < 100:
                state["iteration"] += 1  # a raise is asked at most once per iteration
                if not _maybe_raise_threshold(state, 1.0, "m"):
                    break
                raises += 1
        assert raises == 1, "reaching the ceiling must exhaust the ratchet immediately"
        assert state["stop_threshold"] == THRESHOLD_CEILING
    finally:
        os.environ["SLM_THRESHOLD_RAISE"] = "0"


def test_raise_asked_at_most_once_per_iteration(raising_enabled):
    state = _state()
    with patch(
        "agent.nodes.iterate._llm_threshold_raise", return_value=None
    ) as ask:
        _maybe_raise_threshold(state, 0.8098, "m")
        _maybe_raise_threshold(state, 0.8098, "m")
    assert ask.call_count == 1


def test_at_the_ceiling_no_call_is_made(raising_enabled):
    state = _state(threshold=THRESHOLD_CEILING)
    with patch("agent.nodes.iterate.tracked_chat_anthropic_invoke") as invoke:
        assert _maybe_raise_threshold(state, 0.999, "m") is False
    invoke.assert_not_called()


# --------------------------------------------------------------------------
# Disabled paths spend nothing
# --------------------------------------------------------------------------

def test_disabled_by_env_makes_no_call(monkeypatch):
    monkeypatch.setenv("SLM_THRESHOLD_RAISE", "0")
    state = _state()
    with patch("agent.nodes.iterate._llm_threshold_raise") as ask:
        assert _maybe_raise_threshold(state, 0.8098, "m") is False
    ask.assert_not_called()
    # Banking still happens: the run did meet its goal.
    assert state["convergence_banked"] is not None


def test_cheap_mode_makes_no_call(monkeypatch):
    monkeypatch.setenv("SLM_THRESHOLD_RAISE", "1")
    monkeypatch.setenv("SLM_CHEAP", "1")
    with patch("agent.nodes.iterate._llm_threshold_raise") as ask:
        assert _maybe_raise_threshold(_state(), 0.8098, "m") is False
    ask.assert_not_called()


# --------------------------------------------------------------------------
# Decision validation
# --------------------------------------------------------------------------

def test_a_decline_keeps_the_orchestrators_reason():
    """A decline used to parse to a bare None, throwing the reason away.

    The prompt asks for a `reason` on BOTH branches, so the model explained itself every time and
    the code discarded it. That is the decision that ENDS a converged run: on the 2026-09-05
    ablation arms the logs read "asking the orchestrator whether to raise it" followed immediately
    by "TERMINATE", and why two runs stopped at 0.8035/0.8041 while the baseline pushed to 0.85 had
    to be inferred from score margins after the fact.
    """
    decision = _validate_threshold_raise(
        {"raise_goal": False, "reason": "barely cleared"}
    )
    assert decision == {"raise_goal": False, "reason": "barely cleared"}


def test_a_decline_without_a_reason_is_still_a_decline():
    """A raise REQUIRES a reason; a decline must not be rejected for lacking one.

    Declining is always the safe outcome, so a model that omits the field must not turn into a
    ValueError that the caller then logs as an invalid decision.
    """
    assert _validate_threshold_raise({"raise_goal": False}) == {
        "raise_goal": False,
        "reason": "no reason given",
    }


def test_the_decline_reason_reaches_the_log(raising_enabled, capsys):
    """End to end: the reason the run stopped is printed, with the margin that motivated it."""
    state = _state(score=0.8041, threshold=0.8000, iteration=1)
    with patch(
        "agent.nodes.iterate._llm_threshold_raise",
        return_value={"raise_goal": False, "reason": "only +0.004 over the goal on one iteration"},
    ):
        assert _maybe_raise_threshold(state, 0.8041, "m") is False
    out = capsys.readouterr().out
    assert "Stretch goal DECLINED by the orchestrator" in out
    assert "only +0.004 over the goal on one iteration" in out
    # The margin is in the line too, since it is what the policy turns on.
    assert "+0.0041" in out
    assert state["stop_threshold"] == 0.8000, "a decline must not move the goal"


def test_raise_requires_a_reason():
    with pytest.raises(ValueError, match="non-empty reason"):
        _validate_threshold_raise({"raise_goal": True, "new_threshold": 0.9})


def test_raise_requires_a_finite_number():
    for bad in (None, "0.9", True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite numeric"):
            _validate_threshold_raise(
                {"raise_goal": True, "new_threshold": bad, "reason": "why"}
            )


def test_non_object_decision_raises():
    with pytest.raises(ValueError, match="JSON object"):
        _validate_threshold_raise(["nope"])


# --------------------------------------------------------------------------
# Routing: a raise keeps the run going instead of terminating
# --------------------------------------------------------------------------

@patch("agent.nodes.iterate._llm_iterate", return_value=_intervention_decision())
def test_raise_prevents_termination(llm, raising_enabled):
    """A raised goal makes the score below-threshold again, so the turn carries on as an ordinary
    intervention turn — which means the orchestrator IS consulted and needs a usable answer. Stubbing
    a failure here would raise `OrchestratorDecisionError` instead (B316) and prove nothing about
    whether the raise prevented termination."""
    state = _state(iteration=2)
    state["feasible_models"] = [_model()]
    with patch("agent.nodes.iterate._llm_threshold_raise", return_value=_raise_to(0.87)):
        out = iterate_node(state)
    assert out["next_action"] != "terminate"
    assert out["stop_threshold"] == 0.87
    llm.assert_called_once()


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
def test_decline_still_terminates(_llm, raising_enabled):
    state = _state(iteration=2)
    state["feasible_models"] = [_model()]
    with patch("agent.nodes.iterate._llm_threshold_raise", return_value=None):
        out = iterate_node(state)
    assert out["next_action"] == "terminate"
    assert out["stop_threshold"] == 0.8


# --------------------------------------------------------------------------
# Goal provenance (the reporting half of the change)
# --------------------------------------------------------------------------

def test_provenance_names_the_floor_when_the_floor_won():
    text = describe_threshold_provenance({
        "source": "qwen_baseline", "floor": 0.8, "floored": True,
        "measured_qwen": 0.0999, "measured_metric": "span_f1",
    })
    assert "floor 0.80 OVERRODE" in text
    assert "0.0999" in text and "span_f1" in text


def test_provenance_credits_the_teacher_when_it_set_the_goal():
    text = describe_threshold_provenance({
        "source": "qwen_baseline", "floor": 0.8, "floored": False,
        "measured_qwen": 0.87, "measured_metric": "ast_arg_match",
    })
    assert "measured teacher" in text
    assert "0.8700" in text and "ast_arg_match" in text
    assert "OVERRODE" not in text


def test_provenance_names_the_model_shots_and_format_validity_when_recorded():
    """The goal is only interpretable if the reader knows WHICH teacher set it and HOW.

    0.87 from a local Qwen3.6 and 0.87 from deepseek-v4-flash are different claims, and 0.87 at
    format_valid 0.62 is a different claim again — the teacher was bounded by a broken output
    contract rather than by competence, so the goal it set is an artefact of that. All three used
    to be unreportable: the line hardcoded "Qwen-3.6 zero-shot" no matter what ran.
    """
    text = describe_threshold_provenance({
        "source": "qwen_baseline", "floor": 0.8, "floored": False,
        "measured_qwen": 0.87, "measured_metric": "ast_arg_match",
        "measured_model": "deepseek-v4-flash", "measured_shots": 5,
        "measured_format_valid": 0.9930, "measured_n": 1000,
    })
    assert "deepseek-v4-flash" in text
    assert "5-shot" in text
    assert "format_valid=0.9930" in text
    assert "1000 eval row(s)" in text
    assert "zero-shot" not in text


def test_provenance_stays_silent_about_fields_an_old_checkpoint_never_stored():
    """A resumed pre-revamp checkpoint has no model, shot count or format_valid recorded.

    Degrading to silence is the only safe option: rendering a default would invent the very
    provenance this line exists to report, and "5-shot" printed for a measurement that was in fact
    zero-shot is worse than saying nothing.
    """
    text = describe_threshold_provenance({
        "source": "qwen_baseline", "floor": 0.8, "floored": False,
        "measured_qwen": 0.87, "measured_metric": "ast_arg_match",
    })
    assert "-shot" not in text
    assert "format_valid" not in text
    assert "eval row(s)" not in text


def test_provenance_handles_missing_calibration():
    assert describe_threshold_provenance(None) == "provenance unrecorded"
    assert describe_threshold_provenance({}) == "provenance unrecorded"


def test_provenance_reports_a_pending_measurement():
    assert "pending" in describe_threshold_provenance(
        {"source": "pending_qwen_baseline", "floor": 0.8}
    )
