# tests/eval/test_harness_tokens.py
from unittest.mock import patch, MagicMock
from eval.harness import run_eval
import eval.scorers.classification  # ensure attribute exists on package for patching
import eval.scorers.generation  # ensure attribute exists on package for patching


def _mock_scorer():
    s = MagicMock()
    s.build_prompts.return_value = ["p1"]
    s.extract_predictions.return_value = ["ans"]
    s.score.return_value = {
        "f1": 0.8, "per_class": {}, "slices": {"pos": 1.0, "neg": 0.8, "boundary": 0.9},
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
def test_math_uses_256_tokens(mock_infer):
    mock_infer.return_value = ["42"]
    with patch("eval.scorers.generation", _mock_scorer()):
        run_eval(_eval_set(), "/w", "m", "math_reasoning")
    _, kwargs = mock_infer.call_args
    max_tok = kwargs.get("max_new_tokens")
    assert max_tok == 256, f"Expected 256 for math_reasoning, got {max_tok}"
