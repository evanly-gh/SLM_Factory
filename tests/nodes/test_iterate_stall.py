"""Anti-infinite-loop guarantees in iterate_node (B124 stall backstop + largest_first)."""
import pytest
from unittest.mock import patch, MagicMock
from agent.nodes.iterate import (
    MAX_EVALS_BEFORE_ESCALATION,
    STAGNATION_WINDOW,
    _is_stagnant,
    iterate_node,
)

# Every "does NOT escalate" case below runs with `_llm_iterate` failing, so it carries on into the
# deterministic `data_rebuild` fallback. That makes each of them a live check that the rescue path
# out of an unusable orchestrator reply still works, on top of the window boundary it is written
# for: a call-site/signature disagreement there once turned the non-escalating branch into a
# TypeError, i.e. the rescue killed the run it existed to save.


def _model(tier=0):
    m = MagicMock()
    m.model_id = "unsloth/Qwen3-0.6B"
    m.quant = None
    m.tier = tier
    return m


def _state(cni, score=0.30, threshold=0.90, phase=None, scores=None, eval_history=None):
    return {
        "selected_model": _model(),
        "scores": scores if scores is not None else [score],
        # Stagnation reads the append-only eval history (rollback pops "scores").
        "eval_history": eval_history if eval_history is not None else (
            scores if scores is not None else [score]
        ),
        "best_score": score,
        "iteration": 3,
        "turn_budget": 1000,
        "stop_threshold": threshold,
        "initial_stop_threshold": threshold,
        "task": "gsm8k",
        "last_eval": None,
        "hw_gating_enabled": False,
        "consecutive_no_improvement": cni,
        "_largest_first_phase": phase,
    }


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_flat_eval_history_escalates_even_though_rollback_emptied_scores(_mock):
    # The realistic shape: rollback popped every regression so "scores" is tiny, but the
    # append-only eval history shows a full flat window → must ESCALATE.
    out = iterate_node(
        _state(cni=0, scores=[0.30], eval_history=[0.30] * STAGNATION_WINDOW)
    )
    assert out["next_action"] == "escalate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_one_eval_short_of_the_window_does_not_escalate(_mock):
    out = iterate_node(
        _state(cni=0, scores=[0.30], eval_history=[0.30] * (STAGNATION_WINDOW - 1))
    )
    assert out["next_action"] != "escalate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_improvements_inside_the_window_do_not_reset_it(_mock):
    """An improvement, 9 regressions, another improvement, 4 regressions = 15 evals.

    Total gain across the window is under 2%, so it must still escalate — improvements do not
    reset the counter, only genuine cumulative progress avoids escalation.
    """
    history = [0.300] + [0.28] * 9 + [0.310] + [0.29] * 4
    assert len(history) == 15
    out = iterate_node(_state(cni=0, scores=[0.310], eval_history=history))
    assert out["next_action"] == "escalate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_real_cumulative_gain_across_the_window_does_not_escalate(_mock):
    history = [0.300] + [0.28] * 9 + [0.350] + [0.29] * 4   # +0.05 over the window
    out = iterate_node(_state(cni=0, scores=[0.350], eval_history=history))
    assert out["next_action"] != "escalate"


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
    out = iterate_node(_state(cni=0, phase="probe", eval_history=[0.30] * STAGNATION_WINDOW))
    assert out["next_action"] == "terminate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_stagnation_window_also_escalates(_mock):
    # A full flat window (delta < 0.02 over STAGNATION_WINDOW evals) → stagnation escalates
    # even with low cni. Uses the configured window length (15 as of 2026-08-04).
    out = iterate_node(_state(cni=0, scores=[0.30] * STAGNATION_WINDOW))
    assert out["next_action"] == "escalate"


def test_stagnation_requires_a_full_score_window():
    # Policy 2026-08-04: escalate after 15 evals that gained <2%.
    assert STAGNATION_WINDOW == 15
    assert _is_stagnant([0.30] * (STAGNATION_WINDOW - 1)) is False


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
    assert STAGNATION_WINDOW == 15
    scores = [0.10] * (STAGNATION_WINDOW - 1) + [best_score]

    assert _is_stagnant(scores) is expected


def test_decline_and_below_origin_oscillation_are_stagnant():
    declining = [0.50 - (i * 0.001) for i in range(STAGNATION_WINDOW)]
    # Exactly one full window that never rises above its first entry.
    oscillating_below_origin = (
        [0.50] + ([0.40, 0.49] * STAGNATION_WINDOW)[: STAGNATION_WINDOW - 1]
    )
    assert len(oscillating_below_origin) == STAGNATION_WINDOW

    assert _is_stagnant(declining) is True
    assert _is_stagnant(oscillating_below_origin) is True


def test_oscillation_with_sufficient_gain_is_not_stagnant():
    # A window whose best beats its FIRST entry by >= STAGNATION_MIN_DELTA is real progress,
    # however noisy the path between. Built from the configured window so the policy value can
    # change without silently invalidating the case.
    tail = STAGNATION_WINDOW - 2
    scores = [0.50] + [0.45] * (tail // 2) + [0.521] + [0.46] * (tail - tail // 2)

    assert len(scores) == STAGNATION_WINDOW
    assert _is_stagnant(scores) is False


def test_eval_cap_escalates_even_when_rollback_hides_the_score_history():
    """The 30-eval ceiling must fire on iteration count, not on surviving scores.

    Rollback pops the regressing score, so a run that mostly regresses keeps `scores` tiny and
    can never trip the stagnation window. The cap counts evals actually performed instead.
    """
    from agent.nodes.iterate import MAX_EVALS_BEFORE_ESCALATION

    state = _state(cni=0, scores=[0.30, 0.42])  # only 2 surviving scores, gain 0.12 = "healthy"
    state["iteration"] = MAX_EVALS_BEFORE_ESCALATION

    with patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network")):
        out = iterate_node(state)

    assert out["next_action"] == "escalate"


def test_eval_cap_does_not_fire_one_eval_early():
    from agent.nodes.iterate import MAX_EVALS_BEFORE_ESCALATION

    state = _state(cni=0, scores=[0.30, 0.42])
    state["iteration"] = MAX_EVALS_BEFORE_ESCALATION - 1

    with patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network")):
        out = iterate_node(state)

    assert out["next_action"] != "escalate"


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "smallest_first"})
@patch(
    "agent.nodes.iterate._llm_iterate",
    side_effect=RuntimeError("authentication error: invalid API key"),
)
def test_flat_window_at_stop_threshold_routes_convergence_not_escalation(_mock):
    out = iterate_node(
        _state(
            cni=0,
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
        cni=0,
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
