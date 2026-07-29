"""Anti-infinite-loop guarantees in iterate_node (B124 stall backstop + largest_first)."""
import pytest
from unittest.mock import patch, MagicMock
from agent.nodes.iterate import (
    MAX_STALL_EVALS,
    STAGNATION_WINDOW,
    _is_stagnant,
    iterate_node,
)


def _model(tier=0):
    m = MagicMock()
    m.model_id = "unsloth/Qwen3-0.6B"
    m.quant = None
    m.tier = tier
    return m


def _state(cni, score=0.30, threshold=0.90, phase=None, scores=None):
    return {
        "selected_model": _model(),
        "scores": scores if scores is not None else [score],
        "best_score": score,
        "iteration": 3,
        "turn_budget": 1000,
        "stop_threshold": threshold,
        "initial_stop_threshold": threshold,
        "task_type": "math_reasoning",
        "last_eval": None,
        "hw_gating_enabled": False,
        "consecutive_no_improvement": cni,
        "_largest_first_phase": phase,
    }


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_stall_backstop_escalates(_mock):
    # Below threshold, not enough scores to be "stagnant", but cni has hit the cap →
    # must ESCALATE (not keep churning data_rebuild forever).
    out = iterate_node(_state(cni=MAX_STALL_EVALS))
    assert out["next_action"] == "escalate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_below_cap_does_not_escalate_on_count_alone(_mock):
    out = iterate_node(_state(cni=MAX_STALL_EVALS - 1))
    assert out["next_action"] != "escalate"  # still trying interventions (curate/train)


@patch(
    "agent.nodes.iterate._llm_iterate",
    side_effect=RuntimeError("authentication error: invalid API key"),
)
def test_fatal_auth_still_fails_fast_before_convergence(_mock):
    from agent.llm_errors import FatalLLMError

    with pytest.raises(FatalLLMError, match="Claude API call failed"):
        iterate_node(_state(cni=0, score=0.30, threshold=0.90))


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_largest_first_probe_stall_terminates(_mock):
    # In the largest_first probe phase a stall means the task is infeasible → TERMINATE,
    # never escalate (there is nothing larger than the largest).
    out = iterate_node(_state(cni=MAX_STALL_EVALS, phase="probe"))
    assert out["next_action"] == "terminate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_stagnation_window_also_escalates(_mock):
    # A full flat window (delta < 0.02 over STAGNATION_WINDOW evals) → stagnation escalates
    # even with low cni. Uses the configured window length (raised to 50 in B161).
    out = iterate_node(_state(cni=0, scores=[0.30] * STAGNATION_WINDOW))
    assert out["next_action"] == "escalate"


def test_stagnation_requires_full_50_score_window():
    assert STAGNATION_WINDOW == 50
    assert _is_stagnant([0.30] * 49) is False


@pytest.mark.parametrize(
    ("best_score", "expected"),
    (
        (0.119, True),
        # 0.12 - 0.10 is slightly below 0.02 in binary floating point. The
        # mathematical boundary must still be treated as non-stagnant.
        (0.120, False),
        (0.121, False),
    ),
)
def test_stagnation_uses_robust_chronological_gain_boundary(
    best_score,
    expected,
):
    assert STAGNATION_WINDOW == 50
    scores = [0.10] * (STAGNATION_WINDOW - 1) + [best_score]

    assert _is_stagnant(scores) is expected


def test_decline_and_below_origin_oscillation_are_stagnant():
    declining = [0.50 - (i * 0.001) for i in range(STAGNATION_WINDOW)]
    oscillating_below_origin = [0.50] + [0.40, 0.49] * 24 + [0.45]

    assert _is_stagnant(declining) is True
    assert _is_stagnant(oscillating_below_origin) is True


def test_oscillation_with_sufficient_gain_is_not_stagnant():
    scores = [0.50] + [0.45] * 24 + [0.521] + [0.46] * 24

    assert len(scores) == STAGNATION_WINDOW
    assert _is_stagnant(scores) is False


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "smallest_first"})
@patch(
    "agent.nodes.iterate._llm_iterate",
    side_effect=RuntimeError("authentication error: invalid API key"),
)
def test_flat_window_at_stop_threshold_routes_convergence_not_escalation(_mock):
    out = iterate_node(
        _state(
            cni=MAX_STALL_EVALS,
            score=0.90,
            threshold=0.90,
            scores=[0.90] * STAGNATION_WINDOW,
        )
    )

    assert out["next_action"] == "terminate"
    _mock.assert_not_called()


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "orchestrator_choice"})
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_flat_window_above_stop_threshold_routes_downward_not_escalation(_mock):
    state = _state(
        cni=MAX_STALL_EVALS,
        score=0.95,
        threshold=0.90,
        scores=[0.95] * STAGNATION_WINDOW,
    )
    current = _model(tier=2)
    lower = _model(tier=1)
    state.update(
        {
            "selected_model": current,
            "feasible_models": [lower, current],
            "downward_tiers_tried": [],
            "downward_probe_done": False,
        }
    )

    out = iterate_node(state)

    assert out["next_action"] == "downward_probe"
    _mock.assert_not_called()
