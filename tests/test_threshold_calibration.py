"""Stop-threshold calibration.

CURRENT BEHAVIOR: there is ONE automatic accuracy target — the separately-hosted Qwen-3.6
reference model's own zero-shot score on the run's frozen eval set E, floored at 0.8 and
measured in eval_setup (see test_qwen_baseline_goal.py and agent/threshold.py). The old
published-SOTA registry / measured-anchor / headroom machinery has been removed, and so has the
autonomous task planner (agent/task_planner.py, 2026-11-11) whose prompt tests used to live here.

These tests pin that the deferred goal parks out of reach, and the still-live data_rebuild
mutual-exclusivity contract in the iterate prompt.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

import pytest  # noqa: E402

from agent.threshold import (  # noqa: E402
    THRESHOLD_CEILING,
    UNREACHABLE_PENDING_THRESHOLD,
)


# --- the deferred path parks the target out of reach ---------------------------

def test_pending_threshold_cannot_be_satisfied_by_any_score():
    """While calibration is deferred nothing may converge, or the run would stop before it
    has measured the Qwen baseline it needs."""
    assert UNREACHABLE_PENDING_THRESHOLD > THRESHOLD_CEILING
    assert UNREACHABLE_PENDING_THRESHOLD > 1.0 - 1e-9


# --- the target is written once, and the union rule is explicit ----------------

def test_iterate_prompt_states_the_mutually_exclusive_contract_explicitly():
    """The orchestrator must be TOLD the rule, not silently corrected for breaking it.

    In the NER run Claude attached `hyperparams` to 65 consecutive data_rebuild decisions;
    the validator raised, both paid calls were wasted, and zero orchestrator data plans ran.
    """
    from agent.nodes.iterate import _ITERATE_SYSTEM

    assert "CHOOSE EXACTLY ONE" in _ITERATE_SYSTEM
    assert "FORBIDDEN key" in _ITERATE_SYSTEM
    assert "mutually exclusive" in _ITERATE_SYSTEM
    assert "DISCARD" in _ITERATE_SYSTEM


def test_stray_hyperparams_on_a_data_rebuild_is_recorded_for_logging():
    """The strip must be VISIBLE, not silent masking.

    The state carries a passing teacher-fitness verdict so the plan reaches this assertion with the
    strategy the orchestrator asked for. Without one, `synthesis_allowed` is False — absent IS a
    refusal, deliberately — and the validator rewrites `surgical_synthesis` to `mine_new_real`,
    which is correct behaviour that has nothing to do with what this test is about.
    """
    from agent.nodes.iterate import _validate_decision_json

    decision = {
        "intervention": "data_rebuild",
        "hypothesis": "hard bucket is weak",
        "hyperparams": {"lora_rank": 32},
        "data_rebuild": {
            "strategy": "surgical_synthesis",
        },
    }
    validated = _validate_decision_json(
        decision,
        task="ner_bc5cdr",
        state={"teacher_fitness": {"status": "measured", "score": 0.91,
                                   "synthesis_allowed": True}},
    )
    assert "hyperparams" not in validated, "must not reach the trainer"
    assert validated["_dropped_fields"] == ["hyperparams"]
    assert validated["data_rebuild"]["strategy"] == "surgical_synthesis"
