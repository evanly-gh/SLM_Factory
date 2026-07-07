# tests/nodes/test_evaluate_node.py
import pytest
from unittest.mock import patch, MagicMock
from config.android_pool import ModelSpec, HardwareConstraints
from training.lora_trainer import TrainingOutput
from eval.harness import EvalResult


def _make_model(quant=None):
    return ModelSpec(
        model_id="test/Model-1B",
        int4_size_mb=700,
        tier=1,
        tok_s_snapdragon_660=8.0,
        tok_s_snapdragon_778g=14.0,
        tok_s_snapdragon_8gen3=38.0,
        peak_memory_mb=1100,
        gsm8k=0.6,
        mmlu=0.5,
        quant=quant,
    )


def _make_state(quant=None):
    from data.eval_set import EvalSet
    return {
        "task_type": "classification",
        "selected_model": _make_model(quant=quant),
        "eval_set": MagicMock(spec=EvalSet),
        "iteration": 2,
        "best_score": 0.0,
        "scores": [],
        "dag": [],
        "dataset_version": 1,
        "current_dataset_path": "/data.jsonl",
        "hardware_constraints": HardwareConstraints(
            storage_mb=5000, memory_mb=5000, latency_ttft_ms=3000,
        ),
        "model_baselines": [],
        "last_curation": {},
        "last_hypothesis": "",
        "_pending_weights_refs": {"main": "/ckpt"},
        "_pending_training_outputs": {"main": TrainingOutput(weights_ref="/ckpt", gguf_path=None)},
        "_pending_configs": {"main": {"label": "main", "lora_rank": 8, "learning_rate": 2e-4, "nr_epochs": 3, "batch_size": 8}},
        "consecutive_no_improvement": 0,
    }


def _mock_eval_result(f1=0.85):
    return EvalResult(f1=f1, per_class={}, pos_score=0.9, neg_score=0.8, boundary_score=0.85, failures=[])


@patch("agent.nodes.evaluate.apply_iteration_policy", return_value={"band": "good", "intervention": "hyperparameter"})
@patch("agent.nodes.evaluate.run_eval")
@patch("agent.nodes.evaluate.CurationLog")
@patch("agent.nodes.evaluate.theoretical_hardware_profile", return_value={"int4_size_mb": 700, "tier": 1})
@patch("config.android_pool.check_hardware_constraints", return_value={"storage": {"pass": True}, "memory": {"pass": True}, "latency": {"pass": True}, "power": {"pass": True}})
def test_evaluate_node_uses_bf16_path_when_quant_none(mock_hw, mock_profile, mock_log, mock_eval, mock_policy):
    mock_eval.return_value = _mock_eval_result()
    mock_log.return_value.write_iteration = MagicMock()
    from agent.nodes.evaluate import evaluate_node
    state = _make_state(quant=None)
    evaluate_node(state)
    # run_eval called without gguf_path (or None)
    mock_eval.assert_called_once()
    call_kwargs = mock_eval.call_args[1]
    assert call_kwargs.get("gguf_path") is None
    assert call_kwargs.get("quant") is None


@patch("agent.nodes.evaluate.apply_iteration_policy", return_value={"band": "good", "intervention": "hyperparameter"})
@patch("agent.nodes.evaluate.quantize_from_model_spec", return_value="/gguf/model.gguf")
@patch("agent.nodes.evaluate.merge_for_quantization", return_value="/merged/checkpoint")
@patch("agent.nodes.evaluate.run_eval")
@patch("agent.nodes.evaluate.CurationLog")
@patch("agent.nodes.evaluate.theoretical_hardware_profile", return_value={"int4_size_mb": 700, "tier": 1})
@patch("config.android_pool.check_hardware_constraints", return_value={"storage": {"pass": True}, "memory": {"pass": True}, "latency": {"pass": True}, "power": {"pass": True}})
def test_evaluate_node_quantizes_and_uses_gguf_path_when_quant_set(
    mock_hw, mock_profile, mock_log, mock_eval, mock_merge, mock_quantize, mock_policy
):
    mock_eval.return_value = _mock_eval_result()
    mock_log.return_value.write_iteration = MagicMock()
    from agent.nodes.evaluate import evaluate_node
    state = _make_state(quant="Q4_K_M")
    evaluate_node(state)
    # merge was called with weights_ref
    mock_merge.assert_called_once()
    assert mock_merge.call_args[0][0] == "/ckpt"
    merge_call_args = mock_merge.call_args[0]
    assert "artifacts/merged/test_Model-1B/main/iter2" in merge_call_args[1].replace("\\", "/")
    # quantize was called with correct quant string
    mock_quantize.assert_called_once()
    assert mock_quantize.call_args[0][2] == "Q4_K_M"
    # run_eval called with gguf_path
    mock_eval.assert_called_once()
    call_kwargs = mock_eval.call_args[1]
    assert call_kwargs.get("gguf_path") == "/gguf/model.gguf"
    assert call_kwargs.get("quant") == "Q4_K_M"
