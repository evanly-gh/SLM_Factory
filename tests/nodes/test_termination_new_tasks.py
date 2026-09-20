"""Termination routing for the five on-device SFT suite tasks.

WHY THESE FIVE NEED THEIR OWN CASES

    The stall/convergence/escalation routing is well covered, but every existing case pins
    `task: "gsm8k"`. That was harmless while the suite's tasks all scored on comparable scales.
    These five do not: their selection metrics are `exact_match`, `micro_f1`, `errant_f05`,
    `ekman_macro_f1` and `rouge_l`, and the accuracy goal is
    `min(0.99, max(qwen_measured, 0.80))` — a floor that, as of 2026-09-09, sits ABOVE the best
    published result recorded in `config/benchmark_baselines.md` for all four of them that have
    one (dialogsum's cited SOTA is 0.3945 `rouge_l` against a 0.80 goal).

    That was raised and the floor was deliberately kept. So plateauing below goal is not an
    anomaly for these tasks, it is the EXPECTED path: each run climbs the model ladder, stagnates
    at every tier, and ends on tier exhaustion or the turn/wall-clock budget. Which makes the
    below-goal routing load-bearing in a way it never was for gsm8k, and worth pinning per task —
    a run that terminated early here would silently report a tier's plateau as the final answer,
    and one that failed to terminate would burn a multi-day allocation.

    These assert ROUTING ONLY and never the goal itself. The 0.80 floor is a user decision; a test
    that encoded a lower one would be lowering the goal by another name.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.nodes.iterate import STAGNATION_WINDOW, iterate_node

SUITE_TASKS = ("topv2", "multiconer", "gec_bea19", "goemotions", "dialogsum")

# Roughly what a fine-tuned small model actually reaches on each, from the 2026-09-09 harness probe
# (Qwen3-1.7B, fine-tuned). Every one is far below the 0.80 goal, which is the situation under test.
PLATEAU_SCORE = {
    "topv2": 0.597,
    "multiconer": 0.664,
    "gec_bea19": 0.496,
    "goemotions": 0.585,
    "dialogsum": 0.457,
}


def _model(tier=0):
    model = MagicMock()
    model.model_id = "unsloth/Qwen3-0.6B"
    model.quant = None
    model.tier = tier
    return model


def _state(task, score, threshold=0.80, scores=None):
    series = scores if scores is not None else [score]
    return {
        "selected_model": _model(),
        "scores": list(series),
        "eval_history": list(series),
        "best_score": max(series),
        "iteration": 3,
        "turn_budget": 1000,
        "stop_threshold": threshold,
        "initial_stop_threshold": threshold,
        "task": task,
        "last_eval": None,
        "hw_gating_enabled": False,
        "consecutive_no_improvement": 0,
        "_largest_first_phase": None,
        "dag": [],
    }


# A raising stub as a sentinel: escalation is rule-based and must NOT spend an orchestrator call.
# If one of these consults the LLM, the stub makes it loud rather than letting a silent API call
# through — and with credits exhausted it would be a hard failure mid-run.
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
@pytest.mark.parametrize("task", SUITE_TASKS)
def test_a_plateau_below_the_goal_escalates_rather_than_reporting_the_plateau(mock_llm, task):
    """The expected path for all five: stagnate under an unreachable goal, climb the ladder.

    Terminating here instead would report one tier's ceiling as the run's answer.
    """
    plateau = PLATEAU_SCORE[task]
    out = iterate_node(_state(task, plateau, scores=[plateau] * STAGNATION_WINDOW))

    assert out["next_action"] == "escalate", (
        f"{task}: a flat window at {plateau} under a 0.80 goal must escalate, got "
        f"{out['next_action']!r}"
    )
    mock_llm.assert_not_called()


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network"))
@pytest.mark.parametrize("task", SUITE_TASKS)
def test_reaching_the_goal_terminates_as_converged(mock_llm, task):
    """The goal is still honoured if a task does reach it — the ladder is not a one-way street.

    Pinned per task because convergence and escalation are the same branch reading opposite sides
    of one comparison, and a task whose metric never approaches 0.80 would otherwise have its
    convergence path untested entirely.
    """
    out = iterate_node(_state(task, 0.80, scores=[0.80] * STAGNATION_WINDOW))

    assert out["next_action"] == "terminate", f"{task}: meeting the goal must converge"
    mock_llm.assert_not_called()


@pytest.mark.parametrize("task", SUITE_TASKS)
def test_the_turn_budget_still_terminates_a_run_that_can_never_converge(task):
    """The backstop that guarantees a final report exists for these five.

    Because the goal is above published SOTA, none of these runs is expected to converge, so the
    budget guard is what ends them. If it did not fire, a run would iterate until SLURM hard-killed
    it and the summary/DAG would never be written — the deliverable is the report, so this is the
    difference between five results and five dead jobs.
    """
    plateau = PLATEAU_SCORE[task]
    state = _state(task, plateau, scores=[plateau] * STAGNATION_WINDOW)
    # `turns_used` is `(iteration + 1) * 2`, so this iteration is already over budget.
    state["iteration"] = 20
    state["turn_budget"] = 10

    out = iterate_node(state)
    assert out["next_action"] == "terminate", (
        f"{task}: an exhausted turn budget must terminate, got {out['next_action']!r}"
    )


@pytest.mark.parametrize("task", SUITE_TASKS)
def test_escalation_terminates_gracefully_once_no_larger_model_remains(task):
    """Tier exhaustion is where all five runs are expected to END.

    `escalate_node` steps past empty RAM buckets and terminates only when nothing higher is
    feasible. Asserted with an empty feasible pool, which is that condition regardless of what the
    real ladder holds.
    """
    from agent.nodes.escalate import escalate_node

    state = _state(task, PLATEAU_SCORE[task])
    state["hardware_constraints"] = {}
    state["model_baselines"] = []
    state["best_weights_ref"] = None

    with patch("agent.nodes.escalate.filter_pool", return_value=[]):
        out = escalate_node(state)

    assert out["next_action"] == "terminate", (
        f"{task}: with no feasible higher tier the run must terminate, not stall"
    )
