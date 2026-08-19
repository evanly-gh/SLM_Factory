# tests/eval/test_harness_tokens.py
from unittest.mock import patch

import pytest

from data.eval_set import EvalSet
from eval.harness import eval_output_token_reserve, run_eval
from tasks import TASKS, get_task


def _eval_set(task):
    """A real eval set for `task`. The harness reads its budget off the task's spec, so the spec
    is what the assertions below are really about."""
    return EvalSet(all=[{"text": "p1", "label": "local", "answer": "42"}], task=task)


@patch("eval.harness.infer_batch")
def test_a_short_answer_task_asks_for_few_tokens(mock_infer, monkeypatch):
    monkeypatch.delenv("SLM_CUDA_ISOLATION", raising=False)
    mock_infer.return_value = ["local"]
    run_eval(_eval_set("routerbench"), "/w", "m")
    assert mock_infer.call_args.kwargs["max_new_tokens"] == 50


@patch("eval.harness.infer_batch")
def test_a_chain_of_thought_task_asks_for_room_to_reason(mock_infer, monkeypatch):
    monkeypatch.delenv("SLM_CUDA_ISOLATION", raising=False)
    mock_infer.return_value = ["42"]
    run_eval(_eval_set("gsm8k"), "/w", "m")
    assert mock_infer.call_args.kwargs["max_new_tokens"] == 512


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_reserve_leaves_a_nonzero_prompt_budget(monkeypatch, task):
    """The reserve and the context window are BOTH the task's own, and the spec validated their
    relationship at import time. The reserve used to be a dict keyed by task_type read against a
    `max_seq_length` the caller passed in separately, so the two could disagree."""
    monkeypatch.delenv("SLM_EVAL_MAX_NEW_TOKENS", raising=False)

    reserve = eval_output_token_reserve(task)

    spec = get_task(task)
    assert reserve == spec.max_new_tokens
    assert spec.max_seq_length - reserve > 0


def test_zero_prompt_budget_fails_before_generation_with_actionable_context(monkeypatch):
    monkeypatch.setenv("SLM_EVAL_MAX_NEW_TOKENS", "4096")

    with pytest.raises(
        ValueError,
        match=(
            r"task=ner_bc5cdr reserves 4096 output tokens.*no prompt budget"
            r".*max_seq_length"
        ),
    ):
        eval_output_token_reserve("ner_bc5cdr")


def test_nonpositive_task_output_reserve_is_rejected(monkeypatch):
    monkeypatch.setenv("SLM_EVAL_MAX_NEW_TOKENS", "0")

    with pytest.raises(
        ValueError,
        match=r"SLM_EVAL_MAX_NEW_TOKENS.*positive integer",
    ):
        eval_output_token_reserve("xlam_bfcl")
