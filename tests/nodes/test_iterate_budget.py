from unittest.mock import patch
from agent.nodes.iterate import _wallclock_elapsed_s, iterate_node


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


def test_wallclock_elapsed_includes_completed_resume_segments(monkeypatch):
    monkeypatch.setenv("SLM_RUN_ELAPSED_S", "7200.5")
    monkeypatch.setenv("SLM_RUN_START_TS", "1000")
    monkeypatch.setattr("agent.nodes.iterate._time.time", lambda: 1060.25)

    assert _wallclock_elapsed_s() == 7260.75


@patch(
    "agent.nodes.iterate._llm_iterate",
    return_value={
        "intervention": "data_rebuild",
        "hypothesis": "malformed threshold should not escape",
        "data_rebuild": {
            "strategy": "synthesize",
        },
        "threshold_adjustment": {
            "new_threshold": "not-a-number",
            "reason": "bad payload",
        },
    },
)
def test_malformed_llm_decision_uses_safe_fallback_and_logs(_mock, capsys):
    state = _state(iteration=3, turn_budget=1000)

    out = iterate_node(state)

    assert out["last_intervention"] == "data_rebuild"
    assert out["llm_iterate_decision"] is None
    assert out["next_action"] == "curate"
    rendered = capsys.readouterr().out
    assert "LLM call failed" in rendered
    assert "finite numeric" in rendered
