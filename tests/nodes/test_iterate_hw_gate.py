from unittest.mock import patch, MagicMock
from agent.nodes.iterate import iterate_node


def _model():
    m = MagicMock(); m.model_id = "test/Model-1B"; m.quant = None; m.tier = 0
    return m


def _state(hw_gating):
    return {
        "selected_model": _model(),
        "scores": [0.99],
        "best_score": 0.99,
        "iteration": 2,
        "turn_budget": 1000,
        "stop_threshold": 0.96,
        "initial_stop_threshold": 0.96,
        "task": "clinc150",
        "last_eval": None,
        "hw_gating_enabled": hw_gating,
        "hardware_constraints": MagicMock(),
    }


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
@patch("agent.nodes.iterate.all_constraints_pass", return_value=False)
@patch("agent.nodes.iterate.check_hardware_constraints", return_value={})
def test_hw_fail_blocks_termination_when_gating_on(_c, _a, _l):
    out = iterate_node(_state(hw_gating=True))
    assert out["next_action"] != "terminate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
@patch("agent.nodes.iterate.all_constraints_pass", return_value=False)
@patch("agent.nodes.iterate.check_hardware_constraints", return_value={})
def test_hw_fail_ignored_when_gating_off(_c, _a, _l):
    out = iterate_node(_state(hw_gating=False))
    assert out["next_action"] == "terminate"
