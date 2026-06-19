# tests/test_harness.py
from unittest.mock import patch
from eval.harness import run_eval, EvalResult
from data.eval_set import EvalSet

FAKE_CLASSIFICATION_SET = EvalSet(
    pos=[{"text": "WIN FREE PRIZE", "label": "spam"}],
    neg=[{"text": "Hey see you soon", "label": "ham"}],
    boundary=[{"text": "Exclusive offer click here", "label": "ham"}],
    task_type="classification",
)

def test_run_eval_classification_returns_eval_result():
    with patch("eval.harness.infer_batch", return_value=["spam", "ham", "ham"]):
        result = run_eval(
            FAKE_CLASSIFICATION_SET, "fake_ref", "Qwen/Qwen3-0.6B", task_type="classification"
        )
    assert isinstance(result, EvalResult)
    assert 0.0 <= result.f1 <= 1.0
    assert isinstance(result.failures, list)

def test_run_eval_identifies_failures():
    with patch("eval.harness.infer_batch", return_value=["ham", "ham", "ham"]):
        result = run_eval(
            FAKE_CLASSIFICATION_SET, "fake_ref", "Qwen/Qwen3-0.6B", task_type="classification"
        )
    assert any(f["label"] == "spam" for f in result.failures)

def test_run_eval_unknown_task_type_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown task_type"):
        run_eval(FAKE_CLASSIFICATION_SET, "r", "m", task_type="regression")
