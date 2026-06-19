# tests/test_nodes.py
from unittest.mock import patch, MagicMock
from android_pool import HardwareConstraints, ModelSpec
from agent.state import AgentState
from agent.nodes.iterate import apply_iteration_policy
from agent.nodes.rollback import should_rollback

def _base_state() -> dict:
    return {
        "description": "fine-tune on SMS Spam binary classification",
        "target_metric": "F1",
        "hardware_constraints": HardwareConstraints(
            storage_mb=800, memory_mb=1200,
            latency_ttft_ms=2000, power_watts=5.0
        ),
        "task_type": "classification",
        "selected_model": None,
        "stop_threshold": 0.96,
        "train_examples": [],
        "eval_set": None,
        "current_dataset_path": None,
        "dataset_version": 0,
        "best_weights_ref": None,
        "best_score": 0.0,
        "iteration": 0,
        "scores": [],
        "dag": [],
        "consecutive_no_improvement": 0,
        "last_eval": None,
        "last_intervention": "",
        "next_action": "train",
        "messages": [],
    }

def test_iteration_policy_data_rebuild():
    decision = apply_iteration_policy(score=0.75)
    assert decision["band"] == "<0.80"
    assert decision["intervention"] == "data_rebuild"

def test_iteration_policy_hyperparameter():
    decision = apply_iteration_policy(score=0.87)
    assert decision["band"] == "0.80-0.95"
    assert decision["intervention"] == "hyperparameter"

def test_iteration_policy_surgical():
    decision = apply_iteration_policy(score=0.97)
    assert decision["band"] == ">=0.95"
    assert decision["intervention"] == "surgical"

def test_rollback_triggers_on_decrease():
    state = _base_state()
    state["scores"] = [0.85, 0.82]
    assert should_rollback(state) is True

def test_rollback_does_not_trigger_on_improvement():
    state = _base_state()
    state["scores"] = [0.82, 0.87]
    assert should_rollback(state) is False

def test_rollback_does_not_trigger_on_first_iteration():
    state = _base_state()
    state["scores"] = [0.75]
    assert should_rollback(state) is False
