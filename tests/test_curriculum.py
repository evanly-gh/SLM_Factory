# tests/test_curriculum.py
from data.curriculum import build_initial_curriculum, apply_quality_controls
from data.eval_set import EvalSet

FAKE_TRAIN = (
    [{"text": f"FREE PRIZE CALL NOW {i}!!", "label": "spam"} for i in range(200)]
    + [{"text": f"Hey how are you doing {i}", "label": "ham"} for i in range(800)]
)
FAKE_EVAL = EvalSet(
    pos=[{"text": "WIN FREE PRIZE", "label": "spam"}],
    neg=[{"text": "Hey see you", "label": "ham"}],
    boundary=[{"text": "Exclusive offer", "label": "ham"}],
    task_type="classification",
)


def test_initial_curriculum_size():
    dataset = build_initial_curriculum(FAKE_TRAIN, FAKE_EVAL, n_total=150)
    assert 100 <= len(dataset) <= 200


def test_initial_curriculum_excludes_eval():
    dataset = build_initial_curriculum(FAKE_TRAIN, FAKE_EVAL, n_total=150)
    eval_texts = {e["text"] for e in FAKE_EVAL.all}
    assert not any(e["text"] in eval_texts for e in dataset)


def test_initial_curriculum_label_balance():
    dataset = build_initial_curriculum(FAKE_TRAIN, FAKE_EVAL, n_total=150)
    spam = [e for e in dataset if e["label"] == "spam"]
    ham = [e for e in dataset if e["label"] == "ham"]
    # No label exceeds 3x the other
    assert len(ham) <= 3 * len(spam)
    assert len(spam) <= 3 * len(ham)


def test_quality_controls_label_balance():
    # 10 spam, 100 ham — ham exceeds 3x spam
    imbalanced = (
        [{"text": f"spam {i}", "label": "spam"} for i in range(10)]
        + [{"text": f"ham {i}", "label": "ham"} for i in range(100)]
    )
    result = apply_quality_controls(imbalanced)
    spam = [e for e in result if e["label"] == "spam"]
    ham = [e for e in result if e["label"] == "ham"]
    assert len(ham) <= 3 * len(spam)
