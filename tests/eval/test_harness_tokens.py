# tests/eval/test_harness_tokens.py
from unittest.mock import patch, MagicMock

import pytest

from eval.harness import eval_output_token_reserve, run_eval
import eval.scorers.classification  # ensure attribute exists on package for patching
import eval.scorers.generation  # ensure attribute exists on package for patching


def _mock_scorer():
    s = MagicMock()
    s.build_prompts.return_value = ["p1"]
    s.extract_predictions.return_value = ["ans"]
    s.score.return_value = {
        "f1": 0.8, "per_class": {},
        "failures": [],
    }
    return s


def _eval_set():
    return MagicMock(task_type="classification")


@patch("eval.harness.infer_batch")
def test_classification_uses_50_tokens(mock_infer):
    mock_infer.return_value = ["spam"]
    with patch("eval.scorers.classification", _mock_scorer()):
        run_eval(_eval_set(), "/w", "m", "classification")
    _, kwargs = mock_infer.call_args
    assert kwargs.get("max_new_tokens", mock_infer.call_args[0][3] if len(mock_infer.call_args[0]) > 3 else None) == 50


@patch("eval.harness.infer_batch")
def test_math_uses_long_token_budget(mock_infer):
    mock_infer.return_value = ["42"]
    with patch("eval.scorers.generation", _mock_scorer()):
        run_eval(_eval_set(), "/w", "m", "math_reasoning")
    _, kwargs = mock_infer.call_args
    max_tok = kwargs.get("max_new_tokens")
    assert max_tok == 512, f"Expected 512 for math_reasoning (CoT room), got {max_tok}"


@patch("eval.harness.infer_batch")
def test_code_generation_uses_1024_token_output_budget(mock_infer):
    mock_infer.return_value = ["print(1)"]
    with patch("eval.scorers.generation", _mock_scorer()):
        run_eval(_eval_set(), "/w", "m", "code_generation")

    _, kwargs = mock_infer.call_args
    assert kwargs["max_new_tokens"] == 1024


@pytest.mark.parametrize(
    ("task_type", "expected_reserve"),
    [
        ("NER", 512),
        ("math_reasoning", 512),
        ("generation", 512),
        ("code_generation", 1024),
    ],
)
def test_default_long_task_reserves_leave_nonzero_4096_prompt_budget(
    monkeypatch,
    task_type,
    expected_reserve,
):
    setting = {
        "NER": "SLM_EVAL_MAX_NEW_TOKENS_NER",
        "math_reasoning": "SLM_EVAL_MAX_NEW_TOKENS_MATH",
        "generation": "SLM_EVAL_MAX_NEW_TOKENS_GENERATION",
        "code_generation": "SLM_EVAL_MAX_NEW_TOKENS_APPS",
    }[task_type]
    monkeypatch.delenv(setting, raising=False)

    reserve = eval_output_token_reserve(
        task_type,
        max_seq_length=4096,
    )

    assert reserve == expected_reserve
    assert 4096 - reserve > 0


def test_zero_prompt_budget_fails_before_generation_with_actionable_context():
    with pytest.raises(
        ValueError,
        match=(
            r"task_type=NER.*512 output tokens.*no prompt budget"
            r".*SLM_MAX_SEQ_LENGTH"
        ),
    ):
        eval_output_token_reserve("NER", max_seq_length=512)


def test_nonpositive_task_output_reserve_is_rejected(monkeypatch):
    monkeypatch.setenv("SLM_EVAL_MAX_NEW_TOKENS_APPS", "0")

    with pytest.raises(
        ValueError,
        match=r"SLM_EVAL_MAX_NEW_TOKENS_APPS.*positive integer",
    ):
        eval_output_token_reserve(
            "code_generation",
            max_seq_length=4096,
        )
