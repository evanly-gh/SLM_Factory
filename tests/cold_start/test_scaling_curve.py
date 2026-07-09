import json
import math
import os
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from config.android_pool import ModelSpec, HardwareConstraints
from agent.nodes.cold_start.scaling_curve import (
    _pick_candidates,
    scaling_curve_node,
)


def _make_model(model_id, int4_size_mb, tier=1, quant=None):
    return ModelSpec(
        model_id=model_id,
        int4_size_mb=int4_size_mb,
        tier=tier,
        tok_s_snapdragon_660=5.0,
        tok_s_snapdragon_778g=10.0,
        tok_s_snapdragon_8gen3=25.0,
        peak_memory_mb=int4_size_mb + 400,
        gsm8k=0.6,
        mmlu=0.5,
        quant=quant,
    )


def test_pick_candidates_three():
    models = [_make_model(f"m{i}", 1000 - i * 100) for i in range(6)]
    candidates = _pick_candidates(models)
    assert len(candidates) == 3
    assert candidates[0] is models[0]    # largest
    assert candidates[-1] is models[-1]  # smallest


def test_pick_candidates_one():
    models = [_make_model("only", 500)]
    assert _pick_candidates(models) == models


def test_pick_candidates_two():
    models = [_make_model("big", 900), _make_model("small", 300)]
    candidates = _pick_candidates(models)
    assert len(candidates) == 2


def _make_state(feasible):
    from data.eval_set import EvalSet
    eval_set = MagicMock(spec=EvalSet)
    return {
        "feasible_models": feasible,
        "stop_threshold": 0.80,
        "task_type": "classification",
        "hardware_constraints": HardwareConstraints(
            storage_mb=5000, memory_mb=5000, latency_ttft_ms=3000,
        ),
        "eval_set": eval_set,
        "current_dataset_path": "/fake/dataset.jsonl",
    }


@patch("agent.nodes.cold_start.scaling_curve._probe_model")
def test_selects_smallest_above_threshold(mock_probe):
    # Models sorted largest→smallest: 2000MB, 1000MB, 500MB
    models = [
        _make_model("large", 2000),
        _make_model("medium", 1000),
        _make_model("small", 500),
    ]
    # Probe returns f1 values that make a clean line
    # large→0.90, medium→0.85, small→0.75
    mock_probe.side_effect = [0.90, 0.85, 0.75]

    state = _make_state(models)
    result = scaling_curve_node(state)

    # Fit: f1 = a*log(size)+b. Smallest above 0.80 threshold should be medium (predicted ~0.85)
    assert result["selected_model"] is not None
    # medium or large should be selected (small is predicted below 0.80)
    assert result["selected_model"].model_id == "medium"


@patch("agent.nodes.cold_start.scaling_curve._probe_model")
def test_falls_back_to_largest_when_none_meet_threshold(mock_probe):
    models = [_make_model("large", 2000), _make_model("small", 300)]
    mock_probe.side_effect = [0.50, 0.30]  # neither meets 0.80

    state = _make_state(models)
    result = scaling_curve_node(state)
    assert result["selected_model"].model_id == "large"


@patch("agent.nodes.cold_start.scaling_curve._probe_model")
def test_single_model_selected_directly(mock_probe):
    models = [_make_model("only", 700)]
    mock_probe.return_value = 0.88

    state = _make_state(models)
    result = scaling_curve_node(state)
    assert result["selected_model"].model_id == "only"


def test_empty_feasible_raises():
    state = _make_state([])
    with pytest.raises(RuntimeError, match="feasible_models is empty"):
        scaling_curve_node(state)


def test_probe_uses_eval_set_when_no_dataset(tmp_path):
    """When current_dataset_path is None, probe should use eval_set examples."""
    from agent.nodes.cold_start.scaling_curve import _probe_model

    model = ModelSpec(
        model_id="test/Model", int4_size_mb=500, tier=1,
        tok_s_snapdragon_660=8.0, tok_s_snapdragon_778g=14.0, tok_s_snapdragon_8gen3=38.0,
        peak_memory_mb=900, gsm8k=0.6, mmlu=0.5, quant=None,
    )
    eval_set = MagicMock()
    eval_set.pos = [{"text": "hello", "label": "ham"}]
    eval_set.neg = [{"text": "win prize", "label": "spam"}]
    eval_set.boundary = []
    eval_set.task_type = "classification"

    state = {
        "task_type": "classification",
        "eval_set": eval_set,
        "current_dataset_path": None,   # <- the bug trigger
        "hardware_constraints": MagicMock(),
    }

    with patch("agent.nodes.cold_start.scaling_curve.run_lora_training") as mock_train, \
         patch("agent.nodes.cold_start.scaling_curve.run_eval") as mock_eval:
        from training.lora_trainer import TrainingOutput
        mock_train.return_value = TrainingOutput(weights_ref="/ckpt", gguf_path=None)
        mock_eval.return_value = MagicMock(f1=0.75)
        result = _probe_model(model, state, str(tmp_path))

    # Training was called (not skipped with 0.0)
    mock_train.assert_called_once()
    assert result == 0.75
