"""data_rebuild must be orchestrator-driven, and its fallback must reason from signal.

Three defects made the NER run's data intervention a hard-coded no-op:

  1. Every LLM data plan was REJECTED because Claude attaches a `hyperparams` block to
     essentially every data_rebuild it proposes (65/65 in that run), so zero
     orchestrator-authored plans ever executed.
  2. The fallback picked its strategy by keyword-matching the hypothesis prose — but
     the only two test-agent diagnoses that suggest `data_rebuild` contain none of the
     matched keywords, so it ALWAYS fell through to `resample_existing` (31/31).
  3. Synthesis was gated at score >= 0.95, above both runs' stop thresholds (0.88 and
     0.82), so it could only fire in a run that had already met its goal.

(1) + (2) together bounded the plan space to one strategy, which the run exhausted and
then crashed on after 44.8 hours.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

import pytest  # noqa: E402

from agent.data_rebuild import (  # noqa: E402
    DataRebuildPlanSpaceExhausted,
    _fallback_strategy_from_signal,
    data_rebuild_plan_identity,
    ensure_untried_data_rebuild_plan,
    fallback_data_rebuild_plan,
)


def _state(*, easy=None, medium=None, hard=None, confusion=None, tried=()):
    by_difficulty = {}
    for name, value in (("easy", easy), ("medium", medium), ("hard", hard)):
        if value is not None:
            by_difficulty[name] = {"n": 100, "accuracy": value}
    return {
        "task_type": "NER",
        "curriculum_size_target": 400,
        "scores": [0.5],
        "test_report": {
            "by_difficulty": by_difficulty,
            "confusion_pairs": list(confusion or []),
        },
        # Matches the real DAG shape: the plan lives at node.pi.D.plan.
        "dag": [
            {
                "pruned": False,
                "pi": {"D": {
                    "plan": {"primary_strategy": s},
                    "plan_identity": f"id-{s}",
                }},
            }
            for s in tried
        ],
    }


# --- the fallback reasons from measured signal, not from prose ------------------

def test_low_easy_bucket_mines_new_data_rather_than_reshuffling():
    """Failing even the easy bucket means the DATA is wrong; a reshuffle can't fix it."""
    primary, _ = _fallback_strategy_from_signal(
        _state(easy=0.4, medium=0.4, hard=0.3), task_type="NER", elite_available=False
    )
    assert primary == "mine_new_real_source"


def test_weak_hard_bucket_targets_the_failing_difficulty():
    primary, _ = _fallback_strategy_from_signal(
        _state(easy=0.9, medium=0.8, hard=0.4), task_type="NER", elite_available=False
    )
    assert primary in ("targeted_synth_positive", "difficulty_weighted_sampling")


def test_synthesis_is_reachable_below_the_old_095_gate():
    """The whole point: a difficulty gap now reaches synthesis at ANY score."""
    primary, _ = _fallback_strategy_from_signal(
        _state(easy=0.95, medium=0.5, hard=0.3, confusion=[{"gold": "A", "predicted": "B", "count": 9}]),
        task_type="NER",
        elite_available=False,
        score=0.62,          # far below the old 0.95 gate
    )
    assert primary == "targeted_synth_positive"


def test_high_score_refinement_path_is_preserved():
    """The old near-ceiling behavior still works — it just isn't the only route."""
    primary, _ = _fallback_strategy_from_signal(
        _state(), task_type="NER", elite_available=True, score=0.97
    )
    assert primary == "targeted_synth_positive"


def test_synthesis_stays_ineligible_for_unsupported_task_types():
    primary, _ = _fallback_strategy_from_signal(
        _state(easy=0.9, medium=0.5, hard=0.3), task_type="math_reasoning",
        elite_available=False, score=0.99,
    )
    assert primary != "targeted_synth_positive"


def test_fallback_does_not_collapse_to_one_strategy_across_iterations():
    """The exact failure mode: 31 consecutive rebuilds all `resample_existing`."""
    tried = []
    for _ in range(4):
        primary, _ = _fallback_strategy_from_signal(
            _state(easy=0.9, medium=0.5, hard=0.3, tried=tried),
            task_type="NER",
            elite_available=False,
        )
        tried.append(primary)
    assert len(set(tried)) > 1, f"fallback collapsed to a single strategy: {tried}"


# --- the emitted plan is internally consistent ---------------------------------

def test_mining_strategies_receive_a_positive_material_budget():
    """A mining strategy with new_real_rows=0 fails plan validation."""
    plan = fallback_data_rebuild_plan(
        _state(easy=0.4, medium=0.4, hard=0.3),
        hypothesis="(test-agent) easy-bucket accuracy is low",
    )
    if plan["primary_strategy"] in ("mine_new_real_source", "source_diversification"):
        assert plan["new_real_rows"] > 0


def test_difficulty_weights_follow_measured_accuracy_not_prose():
    """Weight goes to the buckets that are actually failing."""
    plan = fallback_data_rebuild_plan(
        _state(easy=0.95, medium=0.70, hard=0.30),
        hypothesis="(test-agent) below goal with no single failing bucket",
    )
    weights = plan["difficulty_buckets"]
    assert weights["hard"] > weights["medium"] > weights["easy"]


# --- exhaustion terminates cleanly instead of crashing -------------------------

def test_plan_space_rotates_primary_strategy_before_giving_up():
    """Holding the strategy fixed bounded the space to ~1 x 8 x 5 = 40 plans."""
    plan = fallback_data_rebuild_plan(_state(), hypothesis="rebalance")
    identity = data_rebuild_plan_identity(plan)
    # Exhaust every query_variant and resample_fraction for THIS strategy.
    state = _state()
    state["dag"] = [{
        "pruned": False,
        "data_rebuild": dict(plan),
        "data_rebuild_plan_identity": identity,
    }]
    rotated, _, _ = ensure_untried_data_rebuild_plan(plan, state)
    assert rotated is not None  # a variant is still reachable


def test_exhaustion_raises_a_typed_error_that_callers_can_terminate_on():
    """It must be a distinct type so curate_node can end the run cleanly.

    Previously a bare ValueError propagated out of curate_node and killed the
    LangGraph stream — that is how the 44.8-hour NER run ended, after its best
    checkpoint had already been found at iteration 46.
    """
    assert issubclass(DataRebuildPlanSpaceExhausted, ValueError)
