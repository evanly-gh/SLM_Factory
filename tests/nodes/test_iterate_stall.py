"""Anti-infinite-loop guarantees in iterate_node (B124 stall backstop + largest_first)."""
from unittest.mock import patch, MagicMock
from agent.nodes.iterate import iterate_node, MAX_STALL_EVALS


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


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_largest_first_probe_stall_terminates(_mock):
    # In the largest_first probe phase a stall means the task is infeasible → TERMINATE,
    # never escalate (there is nothing larger than the largest).
    out = iterate_node(_state(cni=MAX_STALL_EVALS, phase="probe"))
    assert out["next_action"] == "terminate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
def test_stagnation_window_also_escalates(_mock):
    # Three flat scores (delta < 0.02) → stagnation escalates even with low cni.
    out = iterate_node(_state(cni=0, scores=[0.30, 0.30, 0.30]))
    assert out["next_action"] == "escalate"
