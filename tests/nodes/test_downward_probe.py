from unittest.mock import patch, MagicMock
from config.android_pool import ModelSpec, HardwareConstraints, ANDROID_POOL
from eval.harness import EvalResult


def _model(tier, model_id="test/M", quant=None):
    return ModelSpec(
        model_id=model_id, size_mb=1000, tier=tier,
        tok_s_snapdragon_660=8.0, tok_s_snapdragon_778g=14.0, tok_s_snapdragon_8gen3=38.0,
        peak_memory_mb=1400, gsm8k=0.6, mmlu=0.5, quant=quant,
    )


def _state(current):
    return {
        "selected_model": current,
        "task_type": "classification",
        "task_plan": {"task_type": "classification"},
        "stop_threshold": 0.90,
        "best_score": 0.95,
        "best_weights_ref": "/big/ckpt",
        "current_dataset_path": "/data.jsonl",
        "eval_set": MagicMock(),
        "hardware_constraints": HardwareConstraints(
            storage_mb=10000, memory_mb=10000, latency_ttft_ms=5000,
        ),
    }


def test_tier0_model_skips_probe():
    from agent.nodes.downward_probe import downward_probe_node
    out = downward_probe_node(_state(_model(tier=0)))
    assert out["next_action"] == "terminate"
    assert out["selected_model"].tier == 0  # unchanged
    assert out["downward_probe_done"] is True


@patch("agent.nodes.downward_probe._train_and_eval")
@patch("agent.nodes.downward_probe._llm_choose_model")
@patch("agent.nodes.downward_probe.filter_pool")
def test_adopts_smaller_model_when_it_clears_threshold(mock_fp, mock_choose, mock_te):
    from agent.nodes.downward_probe import downward_probe_node
    smaller = _model(tier=1, model_id="test/Small")
    mock_fp.return_value = [smaller]
    mock_choose.return_value = smaller
    mock_te.return_value = ("/small/ckpt", EvalResult(
        f1=0.92, per_class={}, pos_score=0.9, neg_score=0.9, boundary_score=0.9, failures=[]))
    out = downward_probe_node(_state(_model(tier=2)))
    assert out["selected_model"].model_id == "test/Small"
    assert out["best_weights_ref"] == "/small/ckpt"
    assert out["best_score"] == 0.92
    assert out["next_action"] == "terminate"


@patch("agent.nodes.downward_probe._train_and_eval")
@patch("agent.nodes.downward_probe._llm_choose_model")
@patch("agent.nodes.downward_probe.filter_pool")
def test_keeps_current_when_smaller_fails_threshold(mock_fp, mock_choose, mock_te):
    from agent.nodes.downward_probe import downward_probe_node
    smaller = _model(tier=1, model_id="test/Small")
    mock_fp.return_value = [smaller]
    mock_choose.return_value = smaller
    mock_te.return_value = ("/small/ckpt", EvalResult(
        f1=0.70, per_class={}, pos_score=0.7, neg_score=0.7, boundary_score=0.7, failures=[]))
    current = _model(tier=2, model_id="test/Big")
    out = downward_probe_node(_state(current))
    assert out["selected_model"].model_id == "test/Big"  # unchanged
    assert out["best_weights_ref"] == "/big/ckpt"        # unchanged
    assert out["next_action"] == "terminate"


@patch("agent.nodes.downward_probe.filter_pool")
def test_no_lower_tier_candidates_skips(mock_fp):
    from agent.nodes.downward_probe import downward_probe_node
    mock_fp.return_value = []  # nothing feasible
    out = downward_probe_node(_state(_model(tier=2)))
    assert out["selected_model"].tier == 2  # unchanged
    assert out["next_action"] == "terminate"


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "interpolation"})
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
def test_iterate_routes_to_downward_probe_on_success(_mock):
    # downward_probe only fires for strategies that don't start at the bottom of the
    # feasible set (interpolation / orchestrator_choice) — Q3/B?. With smallest_first it
    # is correctly skipped (already smallest) and the run terminates instead.
    from agent.nodes.iterate import iterate_node
    m = _model(tier=2)
    state = {
        "selected_model": m, "scores": [0.95], "best_score": 0.95, "iteration": 3,
        "turn_budget": 1000, "stop_threshold": 0.90, "initial_stop_threshold": 0.90,
        "task_type": "classification", "last_eval": None, "hw_gating_enabled": False,
        "downward_probe_done": False,
    }
    out = iterate_node(state)
    assert out["next_action"] == "downward_probe"


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "smallest_first"})
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
def test_iterate_skips_downward_probe_for_smallest_first(_mock):
    # smallest_first already ends at the smallest model → no downward probe → terminate.
    from agent.nodes.iterate import iterate_node
    m = _model(tier=2)
    state = {
        "selected_model": m, "scores": [0.95], "best_score": 0.95, "iteration": 3,
        "turn_budget": 1000, "stop_threshold": 0.90, "initial_stop_threshold": 0.90,
        "task_type": "classification", "last_eval": None, "hw_gating_enabled": False,
        "downward_probe_done": False,
    }
    out = iterate_node(state)
    assert out["next_action"] == "terminate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
def test_iterate_terminates_after_probe_done(_mock):
    from agent.nodes.iterate import iterate_node
    m = _model(tier=2)
    state = {
        "selected_model": m, "scores": [0.95], "best_score": 0.95, "iteration": 3,
        "turn_budget": 1000, "stop_threshold": 0.90, "initial_stop_threshold": 0.90,
        "task_type": "classification", "last_eval": None, "hw_gating_enabled": False,
        "downward_probe_done": True,  # already probed
    }
    out = iterate_node(state)
    assert out["next_action"] == "terminate"
