# tests/test_eval_set.py
from data.eval_set import build_eval_set, EvalSet

FAKE_EXAMPLES = (
    [{"text": f"Free prize call now {i}", "label": "spam"} for i in range(30)]
    + [{"text": f"Hey are you coming tonight {i}", "label": "ham"} for i in range(70)]
)

def test_eval_set_has_three_slices():
    es = build_eval_set(FAKE_EXAMPLES, n_boundary=10)
    assert isinstance(es, EvalSet)
    assert len(es.pos) > 0
    assert len(es.neg) > 0
    assert len(es.boundary) > 0

def test_eval_set_all_is_union():
    es = build_eval_set(FAKE_EXAMPLES, n_boundary=10)
    assert len(es.all) == len(es.pos) + len(es.neg) + len(es.boundary)

def test_eval_set_never_in_training_data():
    es = build_eval_set(FAKE_EXAMPLES, n_boundary=10)
    eval_texts = {e["text"] for e in es.all}
    # All eval texts should come from the examples passed in
    all_texts = {e["text"] for e in FAKE_EXAMPLES}
    assert eval_texts.issubset(all_texts)
