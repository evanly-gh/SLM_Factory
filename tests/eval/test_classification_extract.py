from unittest.mock import MagicMock
from eval.scorers.classification import extract_predictions


def _eval_set(labels):
    es = MagicMock()
    es.all = [{"label": l} for l in labels]
    return es


def test_exact_match_preferred_over_substring():
    es = _eval_set(["positive", "very_positive"])
    # Output contains "very_positive" — should match that, not "positive"
    preds = extract_predictions(["very_positive"], es)
    assert preds[0] == "very_positive", f"Expected 'very_positive', got {preds[0]}"


def test_word_boundary_match():
    es = _eval_set(["spam", "ham"])
    # "it's spam." — boundary match on "spam"
    preds = extract_predictions(["it's spam."], es)
    assert preds[0] == "spam"


def test_falls_back_to_unknown_when_no_match():
    es = _eval_set(["spam", "ham"])
    preds = extract_predictions(["something completely different"], es)
    assert preds[0] == "__EXTRACTION_FAILED__"
