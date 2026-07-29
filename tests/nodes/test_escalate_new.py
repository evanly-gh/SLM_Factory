from unittest.mock import patch

from config.android_pool import ANDROID_POOL, HardwareConstraints


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
        "last_curation": {
            "data_rebuild_plan_identity": "prior-plan",
            "total_examples": 20,
        },
        "data_rebuild_plan": {"primary_strategy": "resample_existing"},
        "data_rebuild_plan_identity": "prior-plan",
        "dag": [{"iteration": 1, "score": 0.71, "model_id": "openbmb/MiniCPM4-0.5B", "pruned": False}],
        "iteration": 3,
        "consecutive_no_improvement": 0,
        "last_eval": None,
        "last_hypothesis": "",
        "llm_iterate_decision": None,
    }


def test_escalate_advances_to_higher_tier():
    # Start with the lowest-tier feasible variant; should escalate to a higher RAM tier.
    lowest_tier = min(m.tier for m in ANDROID_POOL)
    start_model = next(m for m in ANDROID_POOL if m.tier == lowest_tier)
    state = _make_state(start_model)
    state["downward_probe_done"] = True
    state["downward_tiers_tried"] = [0]
    state["converged_model_ref"] = {
        "selector": start_model.selector,
        "tier": start_model.tier,
        "score": state["best_score"],
    }
    state["downward_probe_history"] = {
        "origin": {"selector": start_model.selector},
        "attempts": [{"selector": "test/lower@Q4_K_M"}],
    }
    # Determine the nearest higher non-empty tier (matches escalate's own logic).
    higher_tiers = sorted({m.tier for m in ANDROID_POOL if m.tier > lowest_tier})
    next_tier = higher_tiers[0]
    with patch("agent.nodes.escalate._llm_choose_model") as mock_llm:
        mock_llm.return_value = next(m for m in ANDROID_POOL if m.tier == next_tier)
        from agent.nodes.escalate import escalate_node
        out = escalate_node(state)
    assert out["selected_model"].tier == next_tier
    assert out["scores"] == []
    assert out["dag"] == []
    assert out["escalation_history"][0]["selector"] == start_model.selector
    assert mock_llm.call_args.kwargs["direction"] == "up"
    assert out["downward_probe_done"] is False
    assert out["downward_tiers_tried"] == []
    assert out["converged_model_ref"] is None
    assert out["downward_probe_history"] == {
        "origin": None,
        "attempts": [],
    }
    assert out["current_dataset_path"] == "/data.jsonl"
    assert out["dataset_version"] == 1
    assert out["last_curation"]["total_examples"] == 20
    assert out["data_rebuild_plan"] is None
    assert out["data_rebuild_plan_identity"] is None


def test_escalate_terminates_at_top_tier():
    top_tier = max(m.tier for m in ANDROID_POOL)
    # A variant already at the top RAM tier has nothing higher to escalate to.
    top_model = next(m for m in ANDROID_POOL if m.tier == top_tier)
    state = _make_state(top_model)
    with patch("agent.nodes.escalate._llm_choose_model"):
        from agent.nodes.escalate import escalate_node
        out = escalate_node(state)
    assert out["next_action"] == "terminate"
