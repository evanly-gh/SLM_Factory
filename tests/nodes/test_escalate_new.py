import pytest
from unittest.mock import patch, MagicMock
from config.android_pool import HardwareConstraints, ANDROID_POOL, ModelSpec


def _make_state(current_model):
    return {
        "selected_model": current_model,
        "scores": [0.70, 0.71, 0.71],
        "best_score": 0.71,
        "hardware_constraints": HardwareConstraints(
            storage_mb=10000, memory_mb=10000, latency_ttft_ms=5000,
        ),
        "hw_gating_enabled": False,
        "task_type": "classification",
        "task_plan": {"task_type": "classification", "task_name": "test"},
        "model_baselines": [],
        "current_dataset_path": "/data.jsonl",
        "dataset_version": 1,
        "dag": [{"iteration": 1, "score": 0.71, "model_id": "openbmb/MiniCPM4-0.5B", "pruned": False}],
        "iteration": 3,
        "consecutive_no_improvement": 0,
        "last_eval": None,
        "last_hypothesis": "",
        "llm_iterate_decision": None,
    }


def test_escalate_advances_to_next_tier():
    # Start with a tier-0 model; should escalate to tier 1
    tier0_model = next(m for m in ANDROID_POOL if m.tier == 0 and m.quant is None)
    state = _make_state(tier0_model)
    with patch("agent.nodes.escalate._llm_choose_model") as mock_llm:
        # LLM picks the first tier-1 model
        tier1_candidates = [m for m in ANDROID_POOL if m.tier == 1 and m.quant is None]
        mock_llm.return_value = tier1_candidates[0]
        from agent.nodes.escalate import escalate_node
        out = escalate_node(state)
    assert out["selected_model"].tier == 1
    assert out["scores"] == []
    assert out["dag"] == []


def test_escalate_terminates_at_top_tier():
    tier3_models = [m for m in ANDROID_POOL if m.tier == 3 and m.quant is None]
    if not tier3_models:
        pytest.skip("No tier-3 models in pool")
    state = _make_state(tier3_models[-1])  # largest tier-3 model
    with patch("agent.nodes.escalate._llm_choose_model"):
        from agent.nodes.escalate import escalate_node
        out = escalate_node(state)
    assert out["next_action"] == "terminate"
