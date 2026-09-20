"""MultiCoNER II: CoNLL parsing, BIO conversion, the report slice, and micro-vs-macro scoring.

Network-free: every test builds its own sentences. The live 62 MB test-split pull is exercised by
`scripts/preflight_tasks.py`.
"""
from __future__ import annotations

import json

import pytest

from data.eval_set import build_eval_set
from data.loaders.multiconer import (
    COARSE_GROUPS,
    ENTITY_TYPES,
    bio_to_entities,
    parse_conll,
    stratified_slice,
    to_rows,
)
from eval.scorers import fine_ner as scorer


# --------------------------------------------------------------------------
# The taxonomy
# --------------------------------------------------------------------------


def test_the_taxonomy_is_the_documented_thirty_three_classes_in_six_groups():
    """Both numbers are load-bearing: 33 is what the prompt enumerates and the headline averages
    over, and 6 is the low-variance diagnostic. A typo in the table would silently drop a class
    from the prompt, and the scorer compares types exactly."""
    assert len(ENTITY_TYPES) == 33
    assert len(set(COARSE_GROUPS.values())) == 6
    assert set(COARSE_GROUPS.values()) == {"LOC", "CW", "GRP", "PER", "PROD", "MED"}


# --------------------------------------------------------------------------
# CoNLL parsing and BIO conversion
# --------------------------------------------------------------------------


def test_parse_conll_splits_on_id_comments_and_blank_lines():
    text = (
        "# id abc\tdomain=en\n"
        "robert _ _ B-OtherPER\n"
        "gottschalk _ _ I-OtherPER\n"
        "\n"
        "# id def\tdomain=en\n"
        "panavision _ _ B-ORG\n"
    )
    sentences = parse_conll(text)
    assert [s["tokens"] for s in sentences] == [["robert", "gottschalk"], ["panavision"]]
    assert sentences[0]["tags"] == ["B-OtherPER", "I-OtherPER"]


def test_a_malformed_line_is_skipped_rather_than_absorbed_into_the_previous_tag():
    """Without the field-count check, a broken line becomes part of a real token's tag and
    silently corrupts one gold entity."""
    text = "# id abc\ngood _ _ B-ORG\nthis line has no separator\nalso _ _ I-ORG\n"
    sentences = parse_conll(text)
    assert sentences[0]["tokens"] == ["good", "also"]
    assert sentences[0]["tags"] == ["B-ORG", "I-ORG"]


def test_bio_conversion_joins_a_multi_token_span():
    assert bio_to_entities(
        ["robert", "gottschalk", "won"], ["B-OtherPER", "I-OtherPER", "O"]
    ) == [{"text": "robert gottschalk", "type": "OtherPER"}]


def test_a_dangling_continuation_tag_opens_a_span_instead_of_being_dropped():
    """The documented contamination source in this family of corpora, decided explicitly here.

    An `I-X` with no `B-X` before it is malformed, and the two readings differ: dropping it loses
    a real entity, understating gold support and inflating precision. Keeping it is the lenient,
    recall-preserving choice, and it is why this module does not delegate to `seqeval` — the
    decision should be readable where it is made, not inherited.
    """
    assert bio_to_entities(["acme", "corp"], ["I-ORG", "I-ORG"]) == [
        {"text": "acme corp", "type": "ORG"}
    ]


def test_two_adjacent_spans_of_different_types_do_not_merge():
    assert bio_to_entities(
        ["paris", "france"], ["B-HumanSettlement", "I-OtherLOC"]
    ) == [
        {"text": "paris", "type": "HumanSettlement"},
        {"text": "france", "type": "OtherLOC"},
    ]


def test_two_adjacent_spans_of_the_same_type_are_split_by_the_b_tag():
    """`B-` after `B-` is a new entity, not a continuation — otherwise two people standing next to
    each other in a sentence become one person."""
    assert bio_to_entities(
        ["alice", "bob"], ["B-OtherPER", "B-OtherPER"]
    ) == [
        {"text": "alice", "type": "OtherPER"},
        {"text": "bob", "type": "OtherPER"},
    ]


def test_rows_carry_space_joined_text_and_typed_spans():
    rows = to_rows(parse_conll("# id a\nacme _ _ B-ORG\nships _ _ O\n"))
    assert rows[0]["text"] == "acme ships"
    assert rows[0]["entities"] == [{"text": "acme", "type": "ORG"}]


# --------------------------------------------------------------------------
# The report slice
# --------------------------------------------------------------------------


def _row(types: list[str], marker: str) -> dict:
    return {
        "text": f"sentence {marker}",
        "entities": [{"text": f"e{i}", "type": t} for i, t in enumerate(types)],
    }


def test_the_slice_guarantees_rare_classes_support_a_uniform_draw_would_miss():
    """THE REASON THE SLICE IS STRATIFIED.

    9,000 rows of a common class against 30 of a rare one. A uniform 100-row draw of 9,030 would
    be expected to contain about 0.3 rows of the rare class, so the macro-F1 that class is in the
    headline to measure would rest on nothing. The stratifier takes all 30.
    """
    rows = [_row(["ORG"], f"c{i}") for i in range(9000)] + [
        _row(["PrivateCorp"], f"r{i}") for i in range(30)
    ]
    chosen = stratified_slice(rows, size=100, min_mentions=50)
    picked = [rows[i] for i in chosen]
    rare = sum(1 for r in picked for e in r["entities"] if e["type"] == "PrivateCorp")
    assert rare == 30, f"stratifier took only {rare} of 30 available rare rows"
    assert len(picked) == 100


def test_the_quota_is_clamped_to_what_the_split_can_actually_supply():
    """A class the corpus cannot support is a fact about the corpus, not a sampling failure.

    Without the clamp the greedy pass keeps scanning for a 50th mention that does not exist and
    drags in unrelated rows to find it.
    """
    rows = [_row(["ORG"], f"c{i}") for i in range(500)] + [_row(["Symptom"], "only")]
    chosen = stratified_slice(rows, size=20, min_mentions=50)
    assert len(chosen) == 20
    assert any("Symptom" in [e["type"] for e in rows[i]["entities"]] for i in chosen)


def test_the_slice_is_deterministic_and_returns_ascending_indices():
    """Two runs must produce the same slice, or a published macro-F1 is not reproducible."""
    rows = [_row([ENTITY_TYPES[i % 33]], f"s{i}") for i in range(2000)]
    first = stratified_slice(rows, size=400)
    assert first == stratified_slice(rows, size=400)
    assert first == sorted(first)
    assert first != stratified_slice(rows, size=400, seed=1)


def test_the_slice_fills_to_the_requested_size_after_the_quotas_are_met():
    """The bulk of the slice must still look like the test distribution, or the headline is
    computed on a set built entirely out of rare classes."""
    rows = [_row(["ORG"], f"c{i}") for i in range(1000)]
    assert len(stratified_slice(rows, size=250, min_mentions=5)) == 250


def test_a_nonsense_slice_size_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        stratified_slice([_row(["ORG"], "a")], size=0)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _eval_set(rows: list[dict]):
    return build_eval_set(rows, task="multiconer", target=len(rows))


def test_the_prompt_enumerates_every_type_name():
    """WHY THIS IS ASSERTED. `eval.scorers.ner.NER_PROMPT` omits its vocabulary and the scorer
    compares types exactly; on the BC5CDR teacher probe that cost a measured 0.1011 against a real
    0.6140, because the teacher returned the right spans under its own label names. With two types
    that is a 6x understatement. With 33, including `OtherPROD` and `Medication/Vaccine`, an
    unenumerated prompt measures vocabulary telepathy."""
    eval_set = _eval_set([_row(["ORG"], "a")])
    prompt = scorer.build_prompts(eval_set)[0]
    for name in ENTITY_TYPES:
        assert name in prompt, f"{name} is missing from the prompt"
    assert "sentence a" in prompt


def test_selection_reports_micro_and_the_report_pass_reports_macro():
    """The two scorers must differ in the SCALAR they return, not merely in a label.

    One big class fully found, one smaller class missed entirely: micro stays high because
    mentions are pooled, macro drops to 0.5 because the two classes weigh the same. If both
    scorers returned the same number the whole select/report split would be decoration.

    The missed class is given 20 mentions rather than one so that it clears the headline's
    support floor. Below the floor it is excluded from the reported macro by design, and the
    reported number would then be 1.0 — correct behaviour, but it would not exercise the
    micro-versus-macro difference this test is about.
    """
    rows = [_row(["ORG"] * 60, "common")] + [
        _row(["Symptom"], f"rare{i}") for i in range(20)
    ]
    eval_set = _eval_set(rows)
    raw = [
        json.dumps([e for e in row["entities"] if e["type"] == "ORG"])
        for row in eval_set.all
    ]
    predictions = scorer.extract_predictions(raw, eval_set)

    selected = scorer.score(eval_set, predictions)
    reported = scorer.score_report(eval_set, predictions)

    assert selected["metric"] == "micro_f1"
    assert reported["metric"] == "macro_f1"
    assert selected["f1"] == pytest.approx(selected["per_class"]["micro_f1"])
    assert reported["f1"] == pytest.approx(reported["per_class"]["macro_f1"])
    assert reported["f1"] == pytest.approx(0.5)
    assert selected["f1"] > reported["f1"]


def test_the_headline_macro_excludes_thin_classes_and_names_them():
    """The task's own rule: flag any class under 20 support and keep it out of the headline.

    Both numbers are reported, because the literature reports the unfiltered one and the gap says
    how much of the macro rests on near-empty classes.
    """
    rows = [_row(["ORG"] * 25, "common"), _row(["Symptom"], "thin")]
    eval_set = _eval_set(rows)
    raw = [json.dumps(row["entities"]) for row in eval_set.all]
    result = scorer.score_report(eval_set, scorer.extract_predictions(raw, eval_set))

    assert "Symptom" in result["per_class"]["classes_below_support"]
    assert result["per_class"]["headline_min_support"] == 20
    assert result["per_class"]["support_ORG"] == 25


def test_the_coarse_six_grouping_is_reported_and_is_more_forgiving_than_the_fine_one():
    """The low-variance diagnostic. Confusing two PER subtypes is a fine-grained miss and a coarse
    hit, which is the distinction the coarse number exists to expose."""
    rows = [_row(["Politician"], "a")]
    eval_set = _eval_set(rows)
    raw = [json.dumps([{"text": "e0", "type": "Scientist"}])]
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))

    assert result["per_class"]["micro_f1"] == pytest.approx(0.0)
    assert result["per_class"]["coarse6_macro_f1"] == pytest.approx(1.0)


def test_a_type_outside_the_taxonomy_is_its_own_failure_category():
    """It is a label-space problem — the prompt listed the types and the model used another — and
    it wants a different fix from a boundary error."""
    eval_set = _eval_set([_row(["OtherPER"], "a")])
    raw = [json.dumps([{"text": "e0", "type": "PERSON"}])]
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))
    assert result["failures"][0]["error_type"] == "type_outside_taxonomy"


def test_unreadable_output_is_a_format_failure_and_not_a_recall_failure():
    """THE BUG THIS EXISTS FOR.

    The failure record must carry the RAW prediction, with `None` preserved, and not the `[]` the
    metric scored it as. Handing the coalesced value to the categorizer makes its
    `unparseable_output` branch unreachable and files every format failure as
    `no_entities_predicted` — so a broken chat template presents as a recall problem and sends the
    orchestrator looking for more positive examples.
    """
    eval_set = _eval_set([_row(["ORG"], "a")])
    unreadable = scorer.score(eval_set, scorer.extract_predictions(["prose, no json"], eval_set))
    empty = scorer.score(eval_set, scorer.extract_predictions(["[]"], eval_set))

    assert unreadable["format_valid"] == pytest.approx(0.0)
    assert unreadable["failures"][0]["error_type"] == "unparseable_output"
    # A legitimate "no entities" is a real prediction, so its format score is perfect.
    assert empty["format_valid"] == pytest.approx(1.0)
    assert empty["failures"][0]["error_type"] == "no_entities_predicted"


def test_an_empty_eval_set_scores_zero_without_dividing_by_zero():
    empty = build_eval_set([], task="multiconer", target=0)
    for score_fn in (scorer.score, scorer.score_report):
        result = score_fn(empty, [])
        assert result["f1"] == 0.0
        assert result["format_valid"] == 0.0
