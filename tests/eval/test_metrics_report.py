"""The report metrics, checked against hand-computed values and against the conventions they claim.

These are written out in `eval/metrics.py` rather than taken from `seqeval` / `scikit-learn`, so
the tests carry the burden a library's own test suite would otherwise carry. Each one pins a
CONVENTION as well as an arithmetic result, because the conventions are where the reported number
actually comes from: whether average precision interpolates, whether an unsupported class is a
zero or an exclusion, whether a repeated mention counts twice.
"""
from __future__ import annotations

import pytest

from eval.metrics import (
    average_precision,
    entity_macro_f1,
    entity_micro_f1,
    entity_prf_by_type,
    macro_average_precision,
    multi_reference_rouge,
)


# --------------------------------------------------------------------------
# Entity-level micro vs macro
# --------------------------------------------------------------------------


def test_macro_and_micro_diverge_exactly_where_the_label_space_is_skewed():
    """The reason `multiconer` selects on micro and publishes macro, in miniature.

    Nine gold mentions of a common type, all found. One gold mention of a rare type, missed. Micro
    sees 9/10 and calls it 0.947; macro sees one perfect class and one zeroed class and calls it
    0.5. Both are correct, and they answer different questions — which is why a task that reports
    one of them must say which.
    """
    gold = [[{"text": f"c{i}", "type": "Common"} for i in range(9)] + [{"text": "r", "type": "Rare"}]]
    pred = [[{"text": f"c{i}", "type": "Common"} for i in range(9)]]

    assert entity_micro_f1(pred, gold) == pytest.approx(2 * 9 / (9 + 10))
    assert entity_macro_f1(pred, gold) == pytest.approx(0.5)


def test_a_rare_class_can_be_excluded_from_the_macro_average_by_support():
    """`min_support` is what stops six rows carrying a headline.

    With the rare class excluded, macro-F1 is the common class alone. This is the mechanism the
    GoEmotions watch-out asks for — a published +0.57 F1 on `grief` is two macro points earned on
    six test examples — expressed as a threshold the caller has to state rather than a silent one.
    """
    gold = [[{"text": f"c{i}", "type": "Common"} for i in range(9)] + [{"text": "r", "type": "Rare"}]]
    pred = [[{"text": f"c{i}", "type": "Common"} for i in range(9)]]

    assert entity_macro_f1(pred, gold, min_support=5) == pytest.approx(1.0)
    # And the excluded class is still visible, so the exclusion is reportable rather than hidden.
    assert entity_prf_by_type(pred, gold)["Rare"]["support"] == 1


def test_repeated_mentions_are_counted_with_multiplicity():
    """Set arithmetic would score this 1.0. The same drug named three times is three mentions."""
    gold = [[{"text": "aspirin", "type": "Chemical"}] * 3]
    pred = [[{"text": "aspirin", "type": "Chemical"}]]

    stats = entity_prf_by_type(pred, gold)["Chemical"]
    assert (stats["support"], stats["recall"]) == (3, pytest.approx(1 / 3))


def test_a_type_the_model_invented_appears_as_pure_false_positives():
    """A hallucinated class must show in the breakdown, not vanish from it.

    If the per-type table were keyed on gold types alone, a model emitting a label nobody asked
    for would be invisible in the diagnostics and merely depress the micro score for unclear
    reasons.
    """
    gold = [[{"text": "x", "type": "Real"}]]
    pred = [[{"text": "x", "type": "Real"}, {"text": "y", "type": "Invented"}]]

    per_type = entity_prf_by_type(pred, gold)
    assert per_type["Invented"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0,
    }


def test_an_unparseable_row_scores_zero_rather_than_raising():
    """Scorers pass `[]` for output they could not read, and a malformed span must not crash."""
    gold = [[{"text": "x", "type": "Real"}], [{"text": "y", "type": "Real"}]]
    pred = [[], [{"no_text_key": 1}]]

    assert entity_micro_f1(pred, gold) == 0.0
    assert entity_macro_f1(pred, gold) == 0.0


# --------------------------------------------------------------------------
# Average precision
# --------------------------------------------------------------------------


def test_average_precision_matches_the_hand_computed_step_sum():
    """Ranking [pos, neg, pos, neg]: AP = 1*(1/2) + (2/3)*(1/2) = 0.8333...

    Pinned numerically because the alternative convention — interpolating the precision-recall
    curve trapezoidally — gives a DIFFERENT and optimistically biased answer on the same ranking.
    Published GoEmotions AUPRC uses the step sum, so this is the number that is comparable.
    """
    assert average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) == pytest.approx(0.8333333, abs=1e-6)


def test_a_perfect_ranking_scores_one_and_ordering_is_all_that_matters():
    """AP is invariant to the scores' scale — only their order. That is what threshold-free means."""
    assert average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert average_precision([100.0, 3.0, -5.0, -900.0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_tied_scores_do_not_let_input_order_decide_the_result():
    """One positive and one negative at the SAME score must score the same either way round.

    A model asked for 28 label scores off one prompt can genuinely tie. Without tie handling, AP
    would silently reward whichever order the label vocabulary happened to be in.
    """
    forward = average_precision([0.5, 0.5], [1, 0])
    backward = average_precision([0.5, 0.5], [0, 1])
    assert forward == pytest.approx(backward) == pytest.approx(0.5)


def test_a_label_with_no_positives_is_excluded_from_the_macro_not_scored_zero():
    """Scoring an absent class 0.0 would penalize the model for a class the split does not contain.

    GoEmotions' test split is brutally skewed, so this is the difference between a macro AUPRC
    that measures the model and one that measures the split.
    """
    scores = {"present": [0.9, 0.1], "absent": [0.9, 0.1]}
    gold = {"present": [1, 0], "absent": [0, 0]}

    macro, per_label = macro_average_precision(scores, gold)
    assert set(per_label) == {"present"}
    assert macro == pytest.approx(1.0)


def test_macro_average_precision_can_require_more_support_than_one_positive():
    scores = {"a": [0.9, 0.8, 0.1], "b": [0.9, 0.8, 0.1]}
    gold = {"a": [1, 1, 0], "b": [1, 0, 0]}

    _macro, per_label = macro_average_precision(scores, gold, min_support=2)
    assert set(per_label) == {"a"}


# --------------------------------------------------------------------------
# Multi-reference ROUGE
# --------------------------------------------------------------------------


def test_multi_reference_rouge_takes_the_best_matching_reference():
    """The DialogSum property SAMSum lacks, and the reason this task uses three references.

    The prediction is an exact copy of the SECOND reference. Scored against only the first it
    would look mediocre; scored max-over-references it is perfect. That difference is exactly the
    "lottery about whose phrasing you matched" the multi-reference protocol removes.
    """
    prediction = "Ms. Dawson takes a dictation about instant messaging."
    references = [["Something else entirely about trains.", prediction]]

    scores = multi_reference_rouge([prediction], references)
    assert scores["rouge1"] == pytest.approx(1.0)
    assert scores["rougeL"] == pytest.approx(1.0)


def test_an_empty_prediction_scores_zero_without_shrinking_the_denominator():
    """A model that declines to answer must not be rewarded for it.

    One good example and one empty prediction averages to half the good score, not to the good
    score — which is what dropping the empty row would have produced.
    """
    good = "the cat sat on the mat"
    scores = multi_reference_rouge([good, ""], [[good], [good]])
    assert scores["rouge1"] == pytest.approx(0.5)


def test_rouge_over_no_examples_is_zero_not_a_division_by_zero():
    assert multi_reference_rouge([], []) == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
