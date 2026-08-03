"""function_call / diff task types must be registered at EVERY validation touch-point.

Adding a task type to only some of these lists is the classic partial-registration bug: the
scorer exists but the eval set / token reserve / integrity check / planner rejects it at
runtime. This test enumerates the touch-points so a future removal fails loudly.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

import pytest  # noqa: E402

NEW_TYPES = ("function_call", "diff")


@pytest.mark.parametrize("task_type", NEW_TYPES)
def test_registered_in_eval_set_task_types(task_type):
    from data.eval_set import TASK_TYPES

    assert task_type in TASK_TYPES


@pytest.mark.parametrize("task_type", NEW_TYPES)
def test_eval_set_accepts_new_type(task_type):
    from data.eval_set import EvalSet

    es = EvalSet(all=[{"text": "x", "answer": "y"}], task_type=task_type)
    assert es.task_type == task_type
    assert len(es.all) == 1


@pytest.mark.parametrize("task_type", NEW_TYPES)
def test_registered_in_task_analysis(task_type):
    from agent.nodes.cold_start.task_analysis import _VALID_TASK_TYPES

    assert task_type in _VALID_TASK_TYPES


@pytest.mark.parametrize("task_type,metric", [
    ("function_call", "ast_arg_match"),
    ("diff", "apply_match"),
])
def test_registered_in_harness_metric_names(task_type, metric):
    from eval.harness import TASK_METRIC_NAMES

    assert TASK_METRIC_NAMES[task_type] == metric


@pytest.mark.parametrize("task_type", NEW_TYPES)
def test_token_reserve_is_defined_and_positive(task_type):
    from eval.harness import eval_output_token_reserve

    reserve = eval_output_token_reserve(task_type, max_seq_length=4096)
    assert reserve > 0


@pytest.mark.parametrize("task_type", NEW_TYPES)
def test_registered_in_dataset_integrity(task_type):
    from data.loaders.dataset_integrity import required_fields_for_task

    assert required_fields_for_task(task_type) == ("text", "answer")


@pytest.mark.parametrize("task_type", NEW_TYPES)
def test_registered_in_planner(task_type):
    from agent.task_planner import _VALID, _PLANNER_PROMPT

    assert task_type in _VALID
    assert task_type in _PLANNER_PROMPT


@pytest.mark.parametrize("task_type", NEW_TYPES)
def test_harness_dispatch_resolves_a_scorer(task_type):
    """The harness must import a scorer module for each new type (no ValueError)."""
    import importlib

    scorer = importlib.import_module(f"eval.scorers.{task_type}")
    for fn in ("build_prompts", "extract_predictions", "score"):
        assert hasattr(scorer, fn)
