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


def test_scoring_mode_selects_between_the_two_scorers_a_task_declares():
    """`resolve_scorer` is the only place the select/report choice is made.

    Asserted directly because the alternative — reading `spec.score` at the call site — is what
    let a single overloaded eval serve two purposes in the first place. An unknown mode must raise
    rather than fall back to selection: a typo'd mode that silently scored the SELECTION metric and
    labelled the output a report is the exact failure this whole split exists to prevent.
    """
    from eval.harness import SCORING_MODES, resolve_scorer
    from tasks import get_task

    spec = get_task("routerbench")
    assert SCORING_MODES == ("select", "report")
    assert resolve_scorer(spec, "select") == (spec.score, spec.metric_name)
    assert resolve_scorer(spec, "report") == (spec.report_score, spec.report_metric_name)
    with pytest.raises(ValueError, match="scoring must be one of"):
        resolve_scorer(spec, "reporting")


@patch("eval.harness.infer_batch")
def test_report_scoring_grades_with_the_report_scorer_and_names_its_metric(mock_infer, monkeypatch):
    """A report pass must carry the report metric's NAME, or the JSON misattributes the number.

    Driven through a stub task rather than a real one because every task in the registry today
    either reports the metric it selects on or is graded by a scorer needing real inference. The
    thing under test is the plumbing: that `scoring="report"` reaches `report_score`, and that the
    returned `EvalResult.metric` is the report name and not `spec.metric_name`.
    """
    import dataclasses

    from tasks import get_task

    monkeypatch.delenv("SLM_CUDA_ISOLATION", raising=False)
    mock_infer.return_value = ["local", "route"]
    eval_set = _make_eval_set()

    def report_score(_eval_set, _predictions) -> dict:
        return {
            "f1": 0.25,
            "metric": "pretend_auprc",
            "per_class": {"pretend_auprc": 0.25, "format_valid": 1.0},
            "failures": [],
            "format_valid": 1.0,
        }

    stub = dataclasses.replace(
        get_task("routerbench"), report_score=report_score,
        report_metric_name="pretend_auprc",
    )
    with patch("tasks.get_task", return_value=stub):
        selected = run_eval(eval_set, "/weights", "model-id")
        reported = run_eval(eval_set, "/weights", "model-id", scoring="report")

    assert (selected.f1, selected.metric) == (1.0, "minority_f1")
    assert (reported.f1, reported.metric) == (0.25, "pretend_auprc")


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
        eval_set, "/weights", "model-id", quant="Q4_K_M", quant_artifact="/model.gguf"
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


@patch("eval.harness.infer_batch_mnn")
@patch("eval.harness.infer_batch_gguf")
def test_run_eval_routes_an_mnn_artifact_to_the_mnn_engine(mock_gguf, mock_mnn, monkeypatch):
    """The backend decides the engine, NOT the path.

    Both artifacts are just paths, and the failure this guards against is silent: scoring an MNN
    directory through llama-cpp-python does not produce a wrong number, it produces a load error
    whose message blames the artifact. Worse in the other direction — a GGUF scored through the
    wrong engine under an MNN label would report a real accuracy for the wrong runtime.
    """
    monkeypatch.delenv("SLM_CUDA_ISOLATION", raising=False)
    mock_mnn.return_value = ["local", "local", "route"]
    eval_set = EvalSet(
        all=[
            {"text": "a", "label": "local"},
            {"text": "b", "label": "route"},
            {"text": "c", "label": "route"},
        ],
        task="routerbench",
    )

    result = run_eval(
        eval_set, "/weights", "model-id",
        quant="Q4_K_M",
        quant_artifact="/artifacts/mnn/model-mnn-q4",
        quant_backend="mnn",
    )

    mock_mnn.assert_called_once_with(
        eval_set.spec.build_prompts(eval_set),
        "/artifacts/mnn/model-mnn-q4",
        max_new_tokens=50,
        base_model="model-id",
        task="routerbench",
    )
    mock_gguf.assert_not_called()
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
        result = run_eval(eval_set, "/weights", "model-id", quant=None, quant_artifact=None)

    assert result == expected
    assert clear.call_count == 2
    # The task is no longer part of the payload: it travels inside the eval set, so the two can
    # never be passed inconsistently.
    #
    # `scoring` IS in the payload, and must be: the boundary is pickle, so the mode crosses as a
    # string and is re-resolved against the registry inside the worker. Passing the scorer itself
    # would either fail to pickle or pickle by qualified name and resolve to a different object in
    # the child — a parent and child silently disagreeing about which metric was computed. It
    # defaults to "select" here because that is what the loop always wants; a default of "report"
    # would feed a report metric straight into `best_score`.
    worker.assert_called_once_with(
        "eval",
        {
            "eval_set": eval_set,
            "weights_ref": "/weights",
            "base_model": "model-id",
            "quant": None,
            "quant_artifact": None,
            # The backend travels WITH the artifact, for the same reason `scoring` travels as a
            # string: the worker is a separate process, and a path alone cannot say which engine
            # should load it. Both are `None` here because this eval scores unquantized weights.
            "quant_backend": None,
            "scoring": "select",
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
