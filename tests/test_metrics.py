# tests/test_metrics.py
from eval.metrics import binary_f1, per_slice_scores
from data.eval_set import EvalSet

def test_binary_f1_perfect():
    preds = ["spam", "spam", "ham", "ham"]
    labels = ["spam", "spam", "ham", "ham"]
    assert binary_f1(preds, labels, pos_label="spam") == 1.0

def test_binary_f1_all_wrong():
    preds = ["ham", "ham", "spam", "spam"]
    labels = ["spam", "spam", "ham", "ham"]
    assert binary_f1(preds, labels, pos_label="spam") == 0.0

def test_binary_f1_partial():
    preds  = ["spam", "ham", "spam", "ham"]
    labels = ["spam", "spam", "ham", "ham"]
    assert abs(binary_f1(preds, labels, pos_label="spam") - 0.5) < 1e-6

def test_per_slice_scores():
    es = EvalSet(
        pos=[{"text": "spam1", "label": "spam"}, {"text": "spam2", "label": "spam"}],
        neg=[{"text": "ham1", "label": "ham"}],
        boundary=[{"text": "promo", "label": "ham"}],
        task_type="classification",
    )
    preds = ["spam", "spam", "ham", "ham"]
    scores = per_slice_scores(es, preds)
    assert scores["pos"] == 1.0
    assert scores["neg"] == 1.0
    assert scores["boundary"] == 1.0
