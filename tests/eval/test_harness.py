# tests/eval/test_harness.py
import sys
import pytest
from unittest.mock import patch, MagicMock
from eval.harness import run_eval
from data.eval_set import EvalSet


def _make_eval_set():
    es = MagicMock(spec=EvalSet)
    es.task_type = "classification"
    es.multi_label = False
    return es


def _make_scorer_mock():
    scorer_mock = MagicMock()
    scorer_mock.build_prompts.return_value = ["p1", "p2"]
    scorer_mock.extract_predictions.return_value = ["spam", "ham"]
    return scorer_mock


@patch("eval.harness.infer_batch")
def test_run_eval_uses_infer_batch_when_no_gguf(mock_infer):
    mock_infer.return_value = ["spam", "ham"]
    scorer_mock = _make_scorer_mock()
    scorer_mock.score.return_value = {
        "f1": 0.9, "per_class": {}, "slices": {"pos": 1.0, "neg": 0.8, "boundary": 0.9}, "failures": []
    }
    with patch.dict(sys.modules, {"eval.scorers.classification": scorer_mock}):
        result = run_eval(_make_eval_set(), "/weights", "model-id", "classification")
    mock_infer.assert_called_once()
    assert result.f1 == 0.9


@patch("eval.harness.infer_batch_gguf")
def test_run_eval_uses_infer_batch_gguf_when_gguf_path_set(mock_gguf):
    mock_gguf.return_value = ["spam", "ham"]
    scorer_mock = _make_scorer_mock()
    scorer_mock.score.return_value = {
        "f1": 0.85, "per_class": {}, "slices": {"pos": 0.9, "neg": 0.8, "boundary": 0.85}, "failures": []
    }
    with patch.dict(sys.modules, {"eval.scorers.classification": scorer_mock}):
        result = run_eval(
            _make_eval_set(), "/weights", "model-id", "classification",
            quant="Q4_K_M", gguf_path="/model.gguf"
        )
    mock_gguf.assert_called_once_with(["p1", "p2"], "/model.gguf", max_new_tokens=50)
    assert result.f1 == 0.85


def test_run_eval_raises_on_unknown_task_type():
    with pytest.raises(ValueError, match="Unknown task_type"):
        run_eval(_make_eval_set(), "/w", "m", "unknown_task")
