"""`size_sweep` — the naive exhaustive baseline that trains every variant, smallest first.

The ablation for "what does the tiering system save?". The whole point is that the run visits
variants the ladder would skip, so the tests pin the two ways it could silently turn back into the
ladder: escalating a whole tier at a time (skipping same-tier variants), or asking the orchestrator
which model to try next.
"""
from unittest.mock import patch

import pytest

from agent.nodes.cold_start.model_selection import STRATEGIES, get_model_selection_node
from agent.nodes.cold_start.model_selection.size_sweep import (
    next_in_sweep,
    size_sweep_node,
    sweep_order,
)
from config.android_pool import HardwareConstraints, filter_pool
from config.config import model_ladder_enabled, size_sweep_enabled

S24_ULTRA = HardwareConstraints(storage_mb=256000, memory_mb=12288, latency_ttft_ms=5000)


@pytest.fixture
def feasible():
    return filter_pool(S24_ULTRA)


def _escalation_state(current):
    return {
        "selected_model": current,
        "scores": [0.70, 0.71, 0.71],
        "best_score": 0.71,
        "hardware_constraints": S24_ULTRA,
        "hw_gating_enabled": False,
        "task": "ner_bc5cdr",
        "task_plan": {"task": "ner_bc5cdr", "task_name": "test"},
        "model_baselines": [],
        "current_dataset_path": "/data.jsonl",
        "dataset_version": 1,
        "dag": [],
        "iteration": 3,
        "consecutive_no_improvement": 0,
        "last_eval": None,
        "last_hypothesis": "",
        "llm_iterate_decision": None,
    }


def test_strategy_is_registered_and_keeps_the_ladder_on():
    assert get_model_selection_node("size_sweep") is STRATEGIES["size_sweep"]
    assert model_ladder_enabled("size_sweep") is True
    assert size_sweep_enabled("size_sweep") is True
    for other in ("smallest_first", "orchestrator_choice", "single_model"):
        assert size_sweep_enabled(other) is False


def test_the_sweep_covers_every_feasible_variant_in_size_order(feasible):
    order = sweep_order(feasible)
    assert len(order) == len(feasible) == 27
    assert [m.size_mb for m in order] == sorted(m.size_mb for m in feasible)
    assert order[0].selector == "HuggingFaceTB/SmolLM2-135M-Instruct@Q4_K_M"
    assert order[-1].selector == "Qwen/Qwen3.5-4B@bf16"


def test_walking_next_in_sweep_visits_each_variant_once_then_stops(feasible):
    order = sweep_order(feasible)
    visited = [order[0]]
    while (following := next_in_sweep(feasible, visited[-1])) is not None:
        visited.append(following)
    assert [m.selector for m in visited] == [m.selector for m in order]


def test_initial_selection_is_the_smallest_variant_without_an_orchestrator_call(feasible):
    state = {"feasible_models": sorted(feasible, key=lambda m: m.size_mb, reverse=True)}
    with patch("agent.nodes.escalate._llm_choose_model", side_effect=AssertionError("no LLM")):
        out = size_sweep_node(state)
    assert out["selected_model"].selector == "HuggingFaceTB/SmolLM2-135M-Instruct@Q4_K_M"


def test_escalation_moves_to_the_next_variant_even_inside_the_same_tier(feasible):
    """SmolLM2-135M@Q4_K_M and @Q8_0 are both tier 1. The ladder would jump to tier 2."""
    from agent.nodes.escalate import escalate_node

    start, following = sweep_order(feasible)[:2]
    assert start.tier == following.tier
    with patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "size_sweep"}), patch(
        "agent.nodes.escalate._llm_choose_model", side_effect=AssertionError("no LLM")
    ):
        out = escalate_node(_escalation_state(start))
    assert out["selected_model"].selector == following.selector
    assert out["escalation_history"][0]["selector"] == start.selector
    assert out["scores"] == []
    assert out["next_action"] == "curate"


def test_escalation_terminates_after_the_largest_variant(feasible):
    from agent.nodes.escalate import escalate_node

    last = sweep_order(feasible)[-1]
    with patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "size_sweep"}), patch(
        "agent.nodes.escalate._llm_choose_model", side_effect=AssertionError("no LLM")
    ):
        out = escalate_node(_escalation_state(last))
    assert out["next_action"] == "terminate"


def test_the_ladder_still_escalates_by_tier_when_the_sweep_is_off(feasible):
    from agent.nodes.escalate import escalate_node

    start = sweep_order(feasible)[0]
    tier_two = next(m for m in feasible if m.tier == start.tier + 1)
    with patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "smallest_first"}), patch(
        "agent.nodes.escalate._llm_choose_model", return_value=tier_two
    ) as mock_llm:
        out = escalate_node(_escalation_state(start))
    assert mock_llm.called
    assert out["selected_model"].tier == start.tier + 1
