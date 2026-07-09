from unittest.mock import patch
from agent.nodes.iterate import iterate_node


def _state(iteration, turn_budget, score=0.5, threshold=0.96):
    return {
        "selected_model": None,
        "scores": [score],
        "best_score": score,
        "iteration": iteration,
        "turn_budget": turn_budget,
        "stop_threshold": threshold,
        "initial_stop_threshold": threshold,
        "task_type": "classification",
        "last_eval": None,
        "hw_gating_enabled": False,
    }


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network in test"))
def test_terminates_when_turn_budget_exhausted(_mock):
    state = _state(iteration=500, turn_budget=1000)  # turns_used=1000 >= 1000
    out = iterate_node(state)
    assert out["next_action"] == "terminate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network in test"))
def test_does_not_terminate_below_budget(_mock):
    state = _state(iteration=3, turn_budget=1000)
    out = iterate_node(state)
    assert out["next_action"] != "terminate"
