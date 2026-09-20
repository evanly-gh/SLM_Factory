"""GoEmotions: label conversion, multi-label extraction, Ekman selection, AUPRC reporting.

Network-free: rows are built here. The live parquet pull is exercised by
`scripts/preflight_tasks.py`.
"""
from __future__ import annotations

import pytest

from data.eval_set import build_eval_set
from data.loaders.goemotions import (
    EKMAN_GROUPS,
    EKMAN_OF,
    EMOTIONS,
    band_of,
    convert_goemotions_rows,
)
from eval.scorers import multilabel_emotion as scorer


def _row(text: str, labels: list[str]) -> dict:
    return {"text": text, "labels": sorted(labels), "label": ", ".join(sorted(labels))}


def _eval_set(rows: list[dict]):
    return build_eval_set(rows, task="goemotions", target=len(rows))


# --------------------------------------------------------------------------
# The label space
# --------------------------------------------------------------------------


def test_the_taxonomy_is_twenty_eight_labels_mapping_onto_seven_ekman_groups():
    """Both counts are load-bearing: 28 is what the prompt pins and AUPRC averages over, and the
    7-way grouping is the selection metric because all seven have real support."""
    assert len(EMOTIONS) == 28
    assert EMOTIONS[27] == "neutral", "neutral must stay at index 27; groupings depend on it"
    assert len(EKMAN_GROUPS) == 7
    # Every fine label belongs to exactly one group, or the selection metric silently drops labels.
    assert set(EKMAN_OF) == set(EMOTIONS)


def test_integer_label_ids_become_sorted_names():
    """A generative model emits words, so the ids have to be resolved at load."""
    rows = convert_goemotions_rows([{"text": "hi", "labels": [17, 2]}])
    assert rows[0]["labels"] == ["anger", "joy"]
    assert rows[0]["label"] == "anger, joy"


def test_a_row_with_no_usable_label_is_dropped():
    assert convert_goemotions_rows([{"text": "hi", "labels": []}]) == []
    assert convert_goemotions_rows([{"text": "", "labels": [1]}]) == []


def test_frequency_bands_follow_the_documented_thresholds():
    assert band_of(300) == "head"
    assert band_of(299) == "mid"
    assert band_of(50) == "mid"
    assert band_of(49) == "tail"
    assert band_of(6) == "tail"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_a_comma_separated_reply_becomes_a_sorted_label_list():
    eval_set = _eval_set([_row("a", ["joy"])])
    assert scorer.extract_predictions(["joy, gratitude"], eval_set) == [["gratitude", "joy"]]


def test_prose_that_echoes_the_prompts_label_list_is_a_format_failure():
    """THE BUG THIS EXISTS FOR — B271 in multi-label form, observed live on run 39707196.

    The base model replied with the PROMPT'S OWN LABEL LIST as prose: "The comment expresses a mix
    of emotions, including admiration, amusement, anger, annoyance, approval, caring, ...".
    Splitting on commas and keeping every chunk that happened to be a label turned that into a
    confident 20-label prediction — and because it found in-vocabulary labels it counted as
    `format_valid`, so a model that ignored the task scored as one that had attempted it.

    The RouterBench version of this mistake produced zero-shot baselines of
    0.4615 / 0.1685 / 0.1701 / 0.5443 across model tiers, an ordering unrelated to model size.
    """
    eval_set = _eval_set([_row("a", ["admiration"])])
    prose = (
        "The comment expresses a mix of emotions, including admiration, amusement, anger, "
        "annoyance, approval, caring, confusion."
    )
    assert scorer.extract_predictions([prose], eval_set) == [None]


def test_the_same_prose_did_not_extract_at_all_in_the_other_direction():
    """Over-harvesting and under-extraction were ONE root cause.

    "...including sadness, which is neutral." yielded nothing under the old rule, because no
    comma-separated chunk equalled a label exactly — while the sentence above yielded twenty. Both
    are now the same answer: the reply is not a label list.
    """
    eval_set = _eval_set([_row("a", ["sadness"])])
    prose = "The comment expresses a mix of emotions, including sadness, which is neutral."
    assert scorer.extract_predictions([prose], eval_set) == [None]


def test_an_out_of_vocabulary_word_fails_the_reply_rather_than_being_dropped():
    """"joy, happiness" is not the requested format. Silently reducing it to `joy` was how a wrong
    answer earned partial credit, and it is the same leniency that let prose through."""
    eval_set = _eval_set([_row("a", ["joy"])])
    assert scorer.extract_predictions(["joy, happiness"], eval_set) == [None]


def test_a_mildly_chatty_reply_with_the_list_after_a_colon_is_still_accepted():
    """The compliant-enough shape. Enforcing the contract must not mean rejecting an answer that
    plainly gave one — the same ladder `eval/scorers/classification.py` uses."""
    eval_set = _eval_set([_row("a", ["joy"])])
    assert scorer.extract_predictions(["Answer: gratitude, joy"], eval_set) == [
        ["gratitude", "joy"]
    ]


def test_a_reply_naming_no_valid_label_is_unreadable_rather_than_an_empty_prediction():
    """`[]` cannot legitimately occur: every gold row has at least one label and `neutral` is the
    explicit escape hatch. So naming nothing in-vocabulary is unreadable output, and it must not
    be scored as a considered prediction of "no emotion"."""
    eval_set = _eval_set([_row("a", ["joy"])])
    assert scorer.extract_predictions(["asdf qwer"], eval_set) == [None]


def test_overlapping_label_names_are_not_matched_as_substrings():
    """THE TRAP THIS AVOIDS. `approval` is a substring of `disapproval`.

    A naive `if label in reply` test would credit BOTH labels for a reply of "disapproval",
    inventing a false positive on every such row.
    """
    eval_set = _eval_set([_row("a", ["disapproval"])])
    assert scorer.extract_predictions(["disapproval"], eval_set) == [["disapproval"]]


def test_extraction_tolerates_casing_and_trailing_punctuation():
    eval_set = _eval_set([_row("a", ["joy"])])
    assert scorer.extract_predictions(['"Joy".'], eval_set) == [["joy"]]


# --------------------------------------------------------------------------
# Selection scoring
# --------------------------------------------------------------------------


def test_selection_scores_the_ekman_grouping_and_says_so():
    eval_set = _eval_set([_row("a", ["annoyance"]), _row("b", ["gratitude"])])
    raw = [row["label"] for row in eval_set.all]
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))

    assert result["metric"] == "ekman_macro_f1"
    assert result["f1"] == pytest.approx(result["per_class"]["ekman_macro_f1"])
    assert result["f1"] == pytest.approx(1.0)


def test_confusing_two_labels_inside_one_ekman_group_is_a_fine_miss_and_a_coarse_hit():
    """Which is exactly why the coarse grouping is the stable selection signal.

    `annoyance` and `disapproval` are both Ekman `anger`, and human raters genuinely disagree
    about them. A selection metric that treats the confusion as a total miss would rank
    checkpoints on annotation noise.
    """
    eval_set = _eval_set([_row("a", ["annoyance"])])
    result = scorer.score(eval_set, scorer.extract_predictions(["disapproval"], eval_set))

    assert result["per_class"]["ekman_macro_f1"] == pytest.approx(1.0)
    assert result["per_class"]["macro_f1_28"] == pytest.approx(0.0)


def test_a_label_absent_from_the_split_is_excluded_from_the_macro_not_scored_zero():
    """THE BUG THIS EXISTS FOR.

    Measured: a PERFECT prediction over a 300-row draw scored macro_f1_28 = 0.9286 rather than
    1.0, purely because two of the 28 labels happened not to appear in the draw and were averaged
    in as zeros. That penalizes the model for the split's composition. It also has to match how
    `macro_average_precision` treats an unsupported class, since the two sit side by side in one
    report.
    """
    eval_set = _eval_set([_row("a", ["joy"]), _row("b", ["anger"])])
    raw = [row["label"] for row in eval_set.all]
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))

    assert result["per_class"]["macro_f1_28"] == pytest.approx(1.0)
    # The excluded labels are still visible, with zero support.
    assert result["per_class"]["support_grief"] == 0


def test_per_label_scores_are_banded_by_support_in_the_split_actually_scored():
    """Banding is what stops six rows carrying a headline, and it must be computed from the
    scored split rather than a hardcoded list — otherwise a small draw claims head-class support
    it does not have."""
    rows = [_row(f"c{i}", ["joy"]) for i in range(60)] + [_row("rare", ["grief"])]
    eval_set = _eval_set(rows)
    raw = [row["label"] for row in eval_set.all]
    per_class = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))["per_class"]

    assert per_class["n_labels_mid"] == 1     # joy, 60 mentions
    assert per_class["n_labels_tail"] == 1    # grief, 1 mention
    assert per_class["n_labels_head"] == 0


def test_the_degenerate_neutral_answer_has_its_own_failure_category():
    """`neutral` is 1,787 of the test split's mentions, so always answering it scores far above
    chance on accuracy while being worth nothing. It is the specific collapse this label
    distribution invites, so it is named rather than folded into a generic mismatch."""
    eval_set = _eval_set([_row("a", ["joy"])])
    result = scorer.score(eval_set, scorer.extract_predictions(["neutral"], eval_set))
    assert result["failures"][0]["error_type"] == "neutral_collapse"


def test_over_and_under_prediction_are_distinguished():
    """Opposite errors on a multi-label task, wanting opposite interventions."""
    assert scorer.failure_category_of({
        "labels": ["joy", "gratitude"], "predicted": ["joy"],
    }) == "under_predicted"
    assert scorer.failure_category_of({
        "labels": ["joy"], "predicted": ["joy", "gratitude"],
    }) == "over_predicted"
    assert scorer.failure_category_of({
        "labels": ["joy"], "predicted": ["anger"],
    }) == "no_overlap"
    assert scorer.failure_category_of({
        "labels": ["joy"], "predicted": None,
    }) == "unparseable_output"


# --------------------------------------------------------------------------
# Report scoring: threshold-free AUPRC
# --------------------------------------------------------------------------


def _with_oracle_rankings(eval_set):
    for row in eval_set.all:
        row["label_scores"] = {
            emotion: (1.0 if emotion in row["labels"] else -1.0) for emotion in EMOTIONS
        }
    return eval_set


def test_the_report_pass_computes_auprc_from_per_label_rankings():
    eval_set = _with_oracle_rankings(
        _eval_set([_row("a", ["joy"]), _row("b", ["anger"]), _row("c", ["joy"])])
    )
    raw = [row["label"] for row in eval_set.all]
    result = scorer.score_report(eval_set, scorer.extract_predictions(raw, eval_set))

    assert result["metric"] == "macro_auprc"
    assert result["f1"] == pytest.approx(1.0)
    assert result["per_class"]["auprc_rows_scored"] == 3


def test_auprc_is_unmoved_by_the_scale_of_the_scores():
    """The whole point of a threshold-free metric: only the ranking may matter.

    If rescaling the scores moved the number, the metric would be a thresholding artifact in
    disguise — the exact problem it was chosen to escape.
    """
    first = _with_oracle_rankings(_eval_set([_row("a", ["joy"]), _row("b", ["anger"])]))
    raw = [row["label"] for row in first.all]
    baseline = scorer.score_report(first, scorer.extract_predictions(raw, first))["f1"]

    second = _eval_set([_row("a", ["joy"]), _row("b", ["anger"])])
    for row in second.all:
        row["label_scores"] = {
            emotion: (900.0 if emotion in row["labels"] else -0.001) for emotion in EMOTIONS
        }
    rescaled = scorer.score_report(
        second, scorer.extract_predictions([r["label"] for r in second.all], second)
    )["f1"]
    assert rescaled == pytest.approx(baseline)


def test_a_reversed_ranking_scores_far_below_a_correct_one():
    """Guards against an AUPRC that is high regardless of the model, which would be worse than
    no metric at all."""
    eval_set = _eval_set([_row(f"r{i}", ["joy"]) for i in range(10)]
                         + [_row(f"s{i}", ["anger"]) for i in range(10)])
    for row in eval_set.all:
        row["label_scores"] = {
            emotion: (-1.0 if emotion in row["labels"] else 1.0) for emotion in EMOTIONS
        }
    raw = [row["label"] for row in eval_set.all]
    result = scorer.score_report(eval_set, scorer.extract_predictions(raw, eval_set))
    assert result["f1"] < 0.6


def test_missing_rankings_degrade_loudly_instead_of_reporting_a_zero():
    """A missing input is not a bad model.

    An AUPRC of 0.0 is indistinguishable from catastrophic failure and would send the writeup, or
    the loop, chasing a regression that never happened. So the absence is reported as `None` with
    an explanation, and the metric name reverts to the selection metric so nothing downstream can
    label the number a headline it is not.
    """
    eval_set = _eval_set([_row("a", ["joy"])])
    raw = [row["label"] for row in eval_set.all]
    result = scorer.score_report(eval_set, scorer.extract_predictions(raw, eval_set))

    assert result["per_class"]["macro_auprc"] is None
    assert result["metric"] == "ekman_macro_f1"
    assert "infer_label_scores_batch" in result["per_class"]["auprc_unavailable"]


def test_an_empty_eval_set_scores_zero_without_dividing_by_zero():
    empty = build_eval_set([], task="goemotions", target=0)
    for score_fn in (scorer.score, scorer.score_report):
        result = score_fn(empty, [])
        assert result["f1"] == 0.0
        assert result["format_valid"] == 0.0
