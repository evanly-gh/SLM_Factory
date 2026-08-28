"""Unit tests for the difficulty-stratified test-data agent (B161)."""
import dataclasses
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("EXA_API_KEY", "x")

from agent.nodes.test_agent import (
    build_test_report,
    score_by_difficulty, diagnose, _length_heuristic_buckets, label_difficulty,
    _outcome_breakdown,
)
from eval.harness import EvalResult
from tasks import get_task


class _FakeEvalSet:
    def __init__(self, texts):
        self.all = [{"text": t, "label": "x"} for t in texts]
        self.task = "clinc150"


def test_length_heuristic_buckets_partition():
    texts = [f"{'a'*i}" for i in range(1, 10)]  # increasing length
    b = _length_heuristic_buckets(texts, log=lambda *_: None)
    # Every text assigned exactly once; hard bucket holds the longest.
    assert sorted(b["easy"] + b["medium"] + b["hard"]) == sorted(texts)
    assert max(b["hard"], key=len) == texts[-1]


def test_score_by_difficulty_accuracy():
    es = _FakeEvalSet(["e1", "e2", "m1", "h1"])
    difficulty = {"easy": ["e1", "e2"], "medium": ["m1"], "hard": ["h1"]}
    correctness = {"e1": True, "e2": True, "m1": False, "h1": False}
    out = score_by_difficulty(es, correctness, difficulty)
    assert out["easy"]["accuracy"] == 1.0 and out["easy"]["n"] == 2
    assert out["medium"]["accuracy"] == 0.0
    assert out["hard"]["accuracy"] == 0.0


def test_report_exposes_aggregate_confusion_counts_without_eval_text():
    secret_a = "secret raw eval alpha"
    secret_b = "secret raw eval beta"
    eval_set = type("Eval", (), {
        "all": [
            {"text": secret_a, "label": "a"},
            {"text": secret_b, "label": "b"},
        ],
    })()
    result = EvalResult(
        f1=0.5,
        per_class={},
        failures=[
            {"text": secret_a, "label": "a", "predicted": "b"},
            {"text": secret_b, "label": "b", "predicted": "a"},
        ],
    )

    report = build_test_report(
        eval_set,
        result,
        {"easy": [secret_a], "medium": [], "hard": [secret_b]},
        0.9,
        "clinc150",
    )

    assert report["confusion_pairs"] == [
        {"gold": "a", "predicted": "b", "count": 1},
        {"gold": "b", "predicted": "a", "count": 1},
    ]
    assert secret_a not in repr(report)
    assert secret_b not in repr(report)


# --------------------------------------------------------------------------
# outcome_breakdown: WHICH rows were got wrong, not how hard they were
# --------------------------------------------------------------------------
#
# Difficulty answers "how hard were the failures". This answers "which ones", which is the question
# an intervention is actually chosen against — `surgical_synthesis` targets exactly these buckets,
# so the breakdown is the record of whether targeting worked, and `label_performance.png` is drawn
# straight from it. Two bucketings, because the informative unit differs by task.


def _eval_set(rows):
    return type("Eval", (), {"all": list(rows)})()


def test_outcome_breakdown_splits_a_classification_task_by_gold_class():
    """Counts, not rates, and both halves kept: a class at 50% on four rows and one at 50% on four
    hundred are the same F1 and completely different problems."""
    rows = [
        {"text": "t1", "label": "transfer"},
        {"text": "t2", "label": "transfer"},
        {"text": "t3", "label": "transfer"},
        {"text": "b1", "label": "balance"},
        {"text": "b2", "label": "balance"},
    ]
    result = EvalResult(f1=0.6, per_class={}, failures=[
        {"text": "t1", "label": "transfer", "predicted": "balance"},
        {"text": "t2", "label": "transfer", "predicted": "balance"},
    ])

    breakdown = _outcome_breakdown(_eval_set(rows), result, get_task("clinc150"))

    assert breakdown == [
        {"bucket": "transfer", "correct": 1, "failed": 2},
        {"bucket": "balance", "correct": 2, "failed": 0},
    ]


def test_outcome_breakdown_splits_an_open_ended_task_by_failure_category():
    """No class to break down by, so what varies is the KIND of error. The correct rows go in one
    bucket rather than being dropped, or the chart shows only the failures and a reader cannot see
    how large a share of the eval set they are."""
    rows = [{"text": f"request {i}", "answer": "[]"} for i in range(10)]
    result = EvalResult(f1=0.6, per_class={}, failures=[
        {"text": "request 0", "error_type": "wrong_arguments"},
        {"text": "request 1", "error_type": "wrong_arguments"},
        {"text": "request 2", "error_type": "unparseable_output"},
        {"text": "request 3", "error_type": "wrong_function"},
    ])

    breakdown = _outcome_breakdown(_eval_set(rows), result, get_task("xlam_bfcl"))

    assert breakdown == [
        {"bucket": "wrong_arguments", "correct": 0, "failed": 2},
        {"bucket": "unparseable_output", "correct": 0, "failed": 1},
        {"bucket": "wrong_function", "correct": 0, "failed": 1},
        {"bucket": "correct", "correct": 6, "failed": 0},
    ]


def test_outcome_breakdown_is_sorted_failures_first():
    """The chart shows the worst 25 buckets of a possibly-151-class task, so truncation has to keep
    the informative end. Sorting here rather than in the plot means the artifact and the chart agree
    about which buckets those are."""
    rows = (
        [{"text": f"a{i}", "label": "rare_but_broken"} for i in range(4)]
        + [{"text": f"b{i}", "label": "big_and_fine"} for i in range(40)]
        + [{"text": f"c{i}", "label": "flawless"} for i in range(6)]
    )
    result = EvalResult(f1=0.8, per_class={}, failures=(
        [{"text": f"a{i}", "label": "rare_but_broken"} for i in range(3)]
        + [{"text": f"b{i}", "label": "big_and_fine"} for i in range(5)]
    ))

    breakdown = _outcome_breakdown(_eval_set(rows), result, get_task("clinc150"))

    assert [entry["bucket"] for entry in breakdown] == [
        "big_and_fine", "rare_but_broken", "flawless",
    ]
    assert [entry["failed"] for entry in breakdown] == [5, 3, 0]
    # The flawless class is still listed with its size, so "not in the failing set" is visible
    # rather than absent.
    assert breakdown[-1] == {"bucket": "flawless", "correct": 6, "failed": 0}


def test_outcome_breakdown_reaches_the_report_the_orchestrator_and_the_chart_read():
    """`label_performance.png` reads `test_report["outcome_breakdown"]`, so a breakdown computed
    but not published would draw the placeholder on every run."""
    rows = [{"text": "t1", "label": "a"}, {"text": "t2", "label": "b"}]
    result = EvalResult(f1=0.5, per_class={}, failures=[
        {"text": "t1", "label": "a", "predicted": "b"},
    ])

    report = build_test_report(_eval_set(rows), result, {"easy": ["t1"], "medium": [],
                                                         "hard": ["t2"]}, 0.9, "clinc150")

    assert report["outcome_breakdown"] == [
        {"bucket": "a", "correct": 0, "failed": 1},
        {"bucket": "b", "correct": 1, "failed": 0},
    ]


def test_a_failure_taxonomy_that_raises_does_not_break_the_report():
    """A diagnostic must never be able to fail the run that produced it."""
    def _explode(_failure):
        raise KeyError("this failure record predates the taxonomy")

    spec = dataclasses.replace(get_task("xlam_bfcl"), failure_category=_explode)
    rows = [{"text": "request 0"}, {"text": "request 1"}]
    result = EvalResult(f1=0.5, per_class={}, failures=[{"text": "request 0"}])

    assert _outcome_breakdown(_eval_set(rows), result, spec) == [
        {"bucket": "uncategorised", "correct": 0, "failed": 1},
        {"bucket": "correct", "correct": 1, "failed": 0},
    ]


def test_diagnose_converged():
    d = diagnose({"easy": {"accuracy": 0.9, "n": 10}}, 0.9, 0.85, "clinc150")
    assert d["suggested_intervention"] == "none" and d["band"] == "converged"


def test_diagnose_easy_failing_is_data_problem():
    bd = {"easy": {"accuracy": 0.3, "n": 10}, "medium": {"accuracy": 0.2, "n": 10},
          "hard": {"accuracy": 0.1, "n": 10}}
    d = diagnose(bd, 0.2, 0.85, "clinc150")
    assert d["suggested_intervention"] == "data_rebuild" and d["band"] == "data"


def test_diagnose_hard_failing_is_optimization():
    bd = {"easy": {"accuracy": 0.9, "n": 10}, "medium": {"accuracy": 0.8, "n": 10},
          "hard": {"accuracy": 0.3, "n": 10}}
    d = diagnose(bd, 0.7, 0.85, "clinc150")
    assert d["suggested_intervention"] == "hyperparameter" and d["band"] == "optimization"


def test_label_difficulty_zeroshot_gradient():
    # Injected correctness_fn → deterministic; no models loaded.
    es = _FakeEvalSet(["both", "onlybig", "neither"])
    models = [type("M", (), {"model_id": "big"})(), type("M", (), {"model_id": "small"})()]

    def cf(model_id):
        if model_id == "small":
            return {"both": True, "onlybig": False, "neither": False}
        return {"both": True, "onlybig": True, "neither": False}  # big

    os.environ["SLM_DIFFICULTY"] = "zeroshot"
    b = label_difficulty(es, models, "clinc150", log=lambda *_: None, correctness_fn=cf)
    assert b["easy"] == ["both"]
    assert b["medium"] == ["onlybig"]
    assert b["hard"] == ["neither"]


def test_an_unaffordable_difficulty_probe_falls_back_to_the_heuristic():
    """The zero-shot gradient is two FULL generative evals before the run selects a model.

    Measured 2026-08-24 on toolbench jobs 38818333/38818334: 760 rows x 1,536 output tokens is
    1.17M generated tokens per probe model, and the largest feasible model is Qwen3.5-4B at BF16.
    The smallest model's pass alone took 70 minutes and the 4B pass would not have finished inside
    the 20-hour allocation, so both runs spent their whole budget on a REPORTING aid and never
    reached a training step.

    `correctness_fn` is injected here and must NOT be called: the point is that the expensive path is
    not entered at all, not that it returns quickly.
    """
    from agent.nodes.test_agent import MAX_DIFFICULTY_PROBE_TOKENS

    es = _FakeEvalSet([f"row {i}" for i in range(760)])
    models = [type("M", (), {"model_id": "big"})(), type("M", (), {"model_id": "small"})()]
    called: list[str] = []

    def cf(model_id):
        called.append(model_id)
        return {}

    logs: list[str] = []
    os.environ["SLM_DIFFICULTY"] = "zeroshot"
    buckets = label_difficulty(es, models, "toolbench", log=logs.append, correctness_fn=cf)

    assert called == [], "the expensive gradient ran on a task it should have been skipped for"
    assert set(buckets) == {"easy", "medium", "hard"}
    assert sum(len(v) for v in buckets.values()) == 760, "every row must still get a bucket"
    joined = " ".join(logs)
    assert "SKIPPING" in joined
    # The log must quantify the cost, or the next reader cannot tell a skip from a bug.
    assert f"{MAX_DIFFICULTY_PROBE_TOKENS:,}" in joined
    assert "SLM_DIFFICULTY=zeroshot-force" in joined


def test_an_affordable_task_still_runs_the_zeroshot_gradient():
    """The guard must be narrow. Every task that existed before it must be unaffected."""
    es = _FakeEvalSet(["both", "onlybig", "neither"])
    models = [type("M", (), {"model_id": "big"})(), type("M", (), {"model_id": "small"})()]
    called: list[str] = []

    def cf(model_id):
        called.append(model_id)
        return {"both": True, "onlybig": model_id == "big", "neither": False}

    os.environ["SLM_DIFFICULTY"] = "zeroshot"
    buckets = label_difficulty(es, models, "xlam_bfcl", log=lambda *_: None, correctness_fn=cf)
    assert sorted(called) == ["big", "small"]
    assert buckets["easy"] == ["both"]


def test_zeroshot_force_overrides_the_affordability_guard():
    """An escape hatch that does not work is worse than none, so it is asserted."""
    es = _FakeEvalSet([f"row {i}" for i in range(760)])
    models = [type("M", (), {"model_id": "big"})(), type("M", (), {"model_id": "small"})()]
    called: list[str] = []

    def cf(model_id):
        called.append(model_id)
        return {f"row {i}": True for i in range(760)}

    os.environ["SLM_DIFFICULTY"] = "zeroshot-force"
    try:
        label_difficulty(es, models, "toolbench", log=lambda *_: None, correctness_fn=cf)
    finally:
        os.environ["SLM_DIFFICULTY"] = "zeroshot"
    assert sorted(called) == ["big", "small"]


def test_label_difficulty_uses_unique_base_model_capacity_endpoints():
    es = _FakeEvalSet(["both", "onlybig", "neither"])

    class Model:
        def __init__(self, model_id, params, quant):
            self.model_id = model_id
            self.quant = quant
            self._params = params

        def est_params_b(self):
            return self._params

    # Quant siblings deliberately occupy both list ends; endpoint selection must
    # deduplicate model_id and use BF16 base-model capacity identities.
    models = [
        Model("test/large", 4.0, None),
        Model("test/small", 0.6, "Q8_0"),
        Model("test/small", 0.6, "Q4_K_M"),
        Model("test/large", 4.0, "Q4_K_M"),
    ]
    calls = []

    def cf(model_id):
        calls.append(model_id)
        if model_id == "test/small":
            return {"both": True, "onlybig": False, "neither": False}
        return {"both": True, "onlybig": True, "neither": False}

    os.environ["SLM_DIFFICULTY"] = "zeroshot"
    buckets = label_difficulty(
        es,
        models,
        "clinc150",
        log=lambda *_: None,
        correctness_fn=cf,
    )

    assert calls == ["test/small", "test/large"]
    assert buckets["medium"] == ["onlybig"]
