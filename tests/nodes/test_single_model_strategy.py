"""`single_model` — the naive baseline that never changes model.

The ablation control for "what does the model ladder buy?". Everything else in the loop still runs;
only escalation and downward regression are off. There are THREE gates that can reintroduce the
ladder (stagnation escalation, eval-cap escalation, post-convergence downward probe) and all three
are tested here, because a missed one is silent — the run would just quietly climb a tier.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.nodes.cold_start.model_selection import STRATEGIES, get_model_selection_node
from agent.nodes.iterate import iterate_node
from config.config import model_ladder_enabled


def _model(tier=2):
    m = MagicMock()
    m.model_id = "Qwen/Qwen3.5-2B"
    m.selector = "Qwen/Qwen3.5-2B@Q8_0"
    m.quant = "Q8_0"
    m.tier = tier
    return m


def _lower(tier=0):
    m = MagicMock()
    m.model_id = "Qwen/Qwen3-0.6B"
    m.selector = "Qwen/Qwen3-0.6B@Q4_K_M"
    m.quant = "Q4_K_M"
    m.tier = tier
    return m


def _state(scores, threshold=0.80, iteration=None):
    current = _model()
    return {
        "selected_model": current,
        "feasible_models": [_lower(), current],
        "scores": list(scores),
        "eval_history": list(scores),
        "best_score": max(scores),
        "iteration": iteration if iteration is not None else len(scores),
        "turn_budget": 1000,
        "stop_threshold": threshold,
        "initial_stop_threshold": threshold,
        "max_stop_threshold": 0.0,
        "convergence_banked": None,
        "threshold_raises": [],
        "task": "clinc150",
        "last_eval": None,
        "hw_gating_enabled": False,
        "hardware_constraints": MagicMock(),
        "threshold_calibration": None,
        "downward_tiers_tried": [],
    }


# --------------------------------------------------------------------------
# Registration and the flag
# --------------------------------------------------------------------------

def test_strategy_is_registered_and_resolvable():
    assert "single_model" in STRATEGIES
    assert get_model_selection_node("single_model") is STRATEGIES["single_model"]


def test_ladder_flag_is_off_only_for_single_model():
    assert model_ladder_enabled("single_model") is False
    for other in ("smallest_first", "largest_first", "interpolation", "orchestrator_choice"):
        assert model_ladder_enabled(other) is True


def test_selection_delegates_to_orchestrator_choice():
    """The user asked for orchestrator choice specifically — it must be the same code path, not a
    reimplementation that could drift from it."""
    from agent.nodes.cold_start.model_selection import single_model

    state = {"feasible_models": [_lower(), _model()]}
    with patch.object(
        single_model, "orchestrator_choice_node", return_value={**state, "selected_model": _model()}
    ) as chooser:
        out = single_model.single_model_node(state)
    chooser.assert_called_once()
    assert out["selected_model"].selector == "Qwen/Qwen3.5-2B@Q8_0"


# --------------------------------------------------------------------------
# Gate 1 + 2: failing the goal must NOT escalate
# --------------------------------------------------------------------------

# A full STAGNATION_WINDOW of flat evals below the goal: the best score in the window improves on
# its first entry by less than STAGNATION_MIN_DELTA, which is the stagnation signature.
FLAT_WINDOW = [0.60, 0.601, 0.599, 0.6005] + [0.60] * 11


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "single_model"})
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("must not be called"))
def test_stagnation_terminates_instead_of_escalating(_llm):
    out = iterate_node(_state(FLAT_WINDOW))
    assert out["next_action"] == "terminate"
    assert "single_model" in out["last_hypothesis"]


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "smallest_first"})
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip"))
def test_stagnation_still_escalates_under_the_default_strategy(_llm):
    """Guard the guard: the change must not disable escalation for everyone."""
    out = iterate_node(_state(FLAT_WINDOW))
    assert out["next_action"] == "escalate"


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "single_model"})
@patch("agent.nodes.iterate.MAX_EVALS_BEFORE_ESCALATION", 3)
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("must not be called"))
def test_eval_cap_terminates_instead_of_escalating(_llm):
    """The eval cap is the OTHER escalation trigger, and it fires on rising scores where the
    stagnation check does not — so it needs its own gate and its own test.

    The constant is patched directly rather than through the environment: it is read at import
    time, so setting the env var would either do nothing or leak into every later test in the file.
    """
    out = iterate_node(_state([0.40, 0.50, 0.60], iteration=5))
    assert out["next_action"] == "terminate"
    assert "single_model" in out["last_hypothesis"]


# --------------------------------------------------------------------------
# Gate 3: meeting the goal must NOT regress to a smaller model
# --------------------------------------------------------------------------

@patch.dict(
    "os.environ",
    {"SLM_MODEL_SELECTION_STRATEGY": "single_model", "SLM_THRESHOLD_RAISE": "0"},
)
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("must not be called"))
def test_convergence_terminates_without_a_downward_probe(_llm):
    out = iterate_node(_state([0.85], threshold=0.80))
    assert out["next_action"] == "terminate"


@patch.dict(
    "os.environ",
    {"SLM_MODEL_SELECTION_STRATEGY": "smallest_first", "SLM_THRESHOLD_RAISE": "0"},
)
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip"))
def test_convergence_still_probes_downward_under_the_default_strategy(_llm):
    out = iterate_node(_state([0.85], threshold=0.80))
    assert out["next_action"] == "downward_probe"


# --------------------------------------------------------------------------
# What must still work
# --------------------------------------------------------------------------

@patch.dict(
    "os.environ",
    {"SLM_MODEL_SELECTION_STRATEGY": "single_model", "SLM_THRESHOLD_RAISE": "0"},
)
def test_the_rest_of_the_loop_still_runs():
    """Only the ladder is off. A below-threshold, non-stagnant score must still consult the
    orchestrator and route to a normal intervention — otherwise this is not a loop ablation, it is
    just a shorter run."""
    decision = {
        "intervention": "hyperparameter",
        "hypothesis": "try a higher rank",
        "hyperparams": {
            "lora_rank": 32, "alpha_ratio": 2, "weight_decay": 0.01,
            "learning_rate": 2e-4, "nr_epochs": 3,
        },
    }
    with patch("agent.nodes.iterate._llm_iterate", return_value=decision):
        out = iterate_node(_state([0.40, 0.52, 0.61]))
    assert out["next_action"] == "train"
    assert out["last_intervention"] == "hyperparameter"
