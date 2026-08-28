# tests/eval/test_harness.py
import pytest
from unittest.mock import patch
from eval.harness import run_eval
from data.eval_set import EvalSet


def _make_eval_set(task="routerbench"):
    """A REAL eval set, not a mock.

    The harness reads every decision off `tasks.get_task(eval_set.task)`, and a spec holds direct
    function references captured at import time — so patching `eval.scorers.classification` (which
    is what these tests used to do) no longer intercepts anything. Driving the real scorer is also
    closer to what the harness actually does.
    """
    return EvalSet(
        all=[{"text": "a", "label": "local"}, {"text": "b", "label": "route"}],
        task=task,
    )


@patch("eval.harness.infer_batch")
def test_run_eval_uses_infer_batch_when_no_gguf(mock_infer, monkeypatch):
    monkeypatch.delenv("SLM_CUDA_ISOLATION", raising=False)
    mock_infer.return_value = ["local", "route"]
    eval_set = _make_eval_set()

    result = run_eval(eval_set, "/weights", "model-id")

    prompts = eval_set.spec.build_prompts(eval_set)
    mock_infer.assert_called_once_with(
        prompts,
        "/weights",
        "model-id",
        max_workers=20,
        max_new_tokens=50,
        task="routerbench",
    )
    assert result.f1 == 1.0
    # The metric is the TASK's, not a channel default: RouterBench is binary and reports a
    # minority-class F1, which used to be returned under the label `macro_f1`.
    assert result.metric == "minority_f1"


@patch("eval.harness.infer_batch_gguf")
def test_run_eval_uses_infer_batch_gguf_when_gguf_path_set(mock_gguf, monkeypatch):
    monkeypatch.delenv("SLM_CUDA_ISOLATION", raising=False)
    mock_gguf.return_value = ["local", "local", "route"]
    # Three rows with `route` in the majority, so the minority class the scorer picks is
    # unambiguous and the expected number is deterministic.
    eval_set = EvalSet(
        all=[
            {"text": "a", "label": "local"},
            {"text": "b", "label": "route"},
            {"text": "c", "label": "route"},
        ],
        task="routerbench",
    )

    result = run_eval(
        eval_set, "/weights", "model-id", quant="Q4_K_M", gguf_path="/model.gguf"
    )

    mock_gguf.assert_called_once_with(
        eval_set.spec.build_prompts(eval_set),
        "/model.gguf",
        max_new_tokens=50,
        base_model="model-id",
        # The task must reach the GGUF path, or it sizes its scoring concurrency from an unresolvable
        # spec and silently degrades to one context — the old sequential behaviour, for the one caller
        # that matters. The bf16 branch always passed it; this branch did not.
        task="routerbench",
    )
    assert result.f1 == pytest.approx(2 / 3)


def test_an_eval_set_cannot_name_a_task_the_registry_does_not_know():
    """The task is resolved when the eval set is CONSTRUCTED, so no downstream consumer has to
    handle an unknown task. It used to be a free-form `task_type` string checked at scoring time,
    three hours into a run."""
    with pytest.raises(ValueError, match="unknown task"):
        EvalSet(all=[{"text": "x", "label": "y"}], task="unknown_task")


def test_run_eval_delegates_to_disposable_worker_when_enabled(monkeypatch):
    from eval.harness import EvalResult

    eval_set = _make_eval_set()
    expected = EvalResult(0.7, {}, [])
    monkeypatch.setenv("SLM_CUDA_ISOLATION", "1")
    monkeypatch.delenv("SLM_CUDA_WORKER", raising=False)
    with patch("training.cuda_isolation.run_isolated", return_value=expected) as worker, \
         patch("training.slm_helpers.clear_inference_cache") as clear:
        result = run_eval(eval_set, "/weights", "model-id", quant=None, gguf_path=None)

    assert result == expected
    assert clear.call_count == 2
    # The task is no longer part of the payload: it travels inside the eval set, so the two can
    # never be passed inconsistently.
    worker.assert_called_once_with(
        "eval",
        {
            "eval_set": eval_set,
            "weights_ref": "/weights",
            "base_model": "model-id",
            "quant": None,
            "gguf_path": None,
        },
    )


def test_a_reference_free_task_shows_the_query_instead_of_a_blank_gold():
    """ToolEval's pass rate has no reference, so `answer` is legitimately empty on every eval row.

    Printing a bare blank there reproduces B263 for a different reason: the reader cannot tell
    "no reference exists for this metric" from "the gold went missing". Toolbench's sample block
    was doubly unreadable because the `input` clip shows only the AutoGPT boilerplate that is
    identical on every row, so neither the query nor the gold was visible.
    """
    from eval.harness import _gold_for_display

    shown = _gold_for_display({
        "text": "You are AutoGPT, ...",
        "query": "what is the weather in Paris?",
        "answer": "",
    })
    assert "no reference" in shown
    assert "what is the weather in Paris?" in shown


def test_a_task_with_a_real_gold_is_unaffected_by_the_query_fallback():
    """The fallback must be last: a row that HAS a gold must still show the gold."""
    from eval.harness import _gold_for_display

    assert _gold_for_display({"answer": "#### 18", "query": "ignored"}) == "#### 18"
    assert _gold_for_display({"label": "spam", "query": "ignored"}) == "spam"
