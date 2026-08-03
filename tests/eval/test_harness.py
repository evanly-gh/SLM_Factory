# tests/eval/test_harness.py
import pytest
from unittest.mock import patch, MagicMock
from eval.harness import run_eval
from data.eval_set import EvalSet
import eval.scorers.classification  # noqa: F401  # ensure patch target exists


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
        "f1": 0.9, "per_class": {}, "failures": []
    }
    with patch("eval.scorers.classification", scorer_mock):
        result = run_eval(_make_eval_set(), "/weights", "model-id", "classification")
    mock_infer.assert_called_once_with(
        ["p1", "p2"],
        "/weights",
        "model-id",
        max_workers=20,
        max_new_tokens=50,
        task_type="classification",
    )
    assert result.f1 == 0.9


@patch("eval.harness.infer_batch_gguf")
def test_run_eval_uses_infer_batch_gguf_when_gguf_path_set(mock_gguf):
    mock_gguf.return_value = ["spam", "ham"]
    scorer_mock = _make_scorer_mock()
    scorer_mock.score.return_value = {
        "f1": 0.85, "per_class": {}, "failures": []
    }
    with patch("eval.scorers.classification", scorer_mock):
        result = run_eval(
            _make_eval_set(), "/weights", "model-id", "classification",
            quant="Q4_K_M", gguf_path="/model.gguf"
        )
    mock_gguf.assert_called_once_with(
        ["p1", "p2"],
        "/model.gguf",
        max_new_tokens=50,
        base_model="model-id",
    )
    assert result.f1 == 0.85


def test_run_eval_raises_on_unknown_task_type():
    with pytest.raises(ValueError, match="Unknown task_type"):
        run_eval(_make_eval_set(), "/w", "m", "unknown_task")


@patch("eval.harness.infer_batch")
def test_code_eval_preserves_execution_case_diagnostics(mock_infer, monkeypatch):
    import eval.scorers.generation  # noqa: F401

    monkeypatch.delenv("SLM_CUDA_ISOLATION", raising=False)
    mock_infer.return_value = ["print(1)"]
    scorer = MagicMock()
    scorer.build_prompts.return_value = ["prompt"]
    scorer.extract_predictions.return_value = ["print(1)"]
    scorer.score.return_value = {
        "f1": 1.0,
        "per_class": {},
        "failures": [],
        "execution_diagnostics": [
            {"tests_used": 10, "tests_total": 23, "score": 1.0}
        ],
    }
    eval_set = MagicMock(spec=EvalSet)
    eval_set.task_type = "code_generation"

    with patch("eval.scorers.generation", scorer):
        result = run_eval(
            eval_set,
            "/weights",
            "model-id",
            "code_generation",
        )

    assert result.execution_diagnostics == [
        {"tests_used": 10, "tests_total": 23, "score": 1.0}
    ]


def test_run_eval_delegates_to_disposable_worker_when_enabled(monkeypatch):
    from eval.harness import EvalResult

    eval_set = _make_eval_set()
    expected = EvalResult(0.7, {}, [])
    monkeypatch.setenv("SLM_CUDA_ISOLATION", "1")
    monkeypatch.delenv("SLM_CUDA_WORKER", raising=False)
    with patch("training.cuda_isolation.run_isolated", return_value=expected) as worker, \
         patch("training.slm_helpers.clear_inference_cache") as clear:
        result = run_eval(
            eval_set, "/weights", "model-id", "classification",
            quant=None, gguf_path=None,
        )

    assert result == expected
    assert clear.call_count == 2
    worker.assert_called_once_with(
        "eval",
        {
            "eval_set": eval_set,
            "weights_ref": "/weights",
            "base_model": "model-id",
            "task_type": "classification",
            "quant": None,
            "gguf_path": None,
        },
    )
