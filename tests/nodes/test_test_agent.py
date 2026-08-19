"""Unit tests for the difficulty-stratified test-data agent (B161)."""
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("EXA_API_KEY", "x")

from agent.nodes.test_agent import (
    build_test_report,
    score_by_difficulty, diagnose, _length_heuristic_buckets, label_difficulty,
)
from eval.harness import EvalResult


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
