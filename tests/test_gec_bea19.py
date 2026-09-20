"""GEC: M2 parsing, edit application, the one-line contract, and real ERRANT scoring.

The M2 parsing and extraction tests are pure and fast. The scoring tests shell out to the real
ERRANT CLI and are skipped when `.venv_errant` has not been built, because a stubbed ERRANT would
test our mock rather than the scorer — and the scorer's entire justification is that it is the
real, published implementation.
"""
from __future__ import annotations

import os

import pytest

from data.eval_set import build_eval_set
from data.loaders.gec_bea19 import (
    _interleave_by_level,
    apply_m2_edits,
    parse_m2,
    to_rows,
)
from eval.scorers import gec as scorer


def _errant_available() -> bool:
    try:
        scorer.errant_bin("errant_parallel")
        return True
    except scorer.ErrantUnavailable:
        return False


needs_errant = pytest.mark.skipif(
    not _errant_available(),
    reason="ERRANT venv not built; run bash scripts/setup_metric_envs.sh",
)


# --------------------------------------------------------------------------
# M2 parsing and edit application
# --------------------------------------------------------------------------


M2_SAMPLE = """S I has went to the store yesterday .
A 1 2|||U:VERB:TENSE||||||REQUIRED|||-NONE-|||0

S This sentence is already correct .
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0
"""


def test_parse_m2_applies_the_edits_and_keeps_the_raw_block():
    blocks = parse_m2(M2_SAMPLE)
    assert [b["source"] for b in blocks] == [
        "I has went to the store yesterday .",
        "This sentence is already correct .",
    ]
    assert blocks[0]["target"] == "I went to the store yesterday ."
    # The raw block is what the scorer rebuilds the reference file from, so it must survive intact.
    assert blocks[0]["m2"].startswith("S I has went")
    assert "U:VERB:TENSE" in blocks[0]["m2"]


def test_a_noop_annotation_leaves_the_sentence_alone():
    """A sentence needing no correction is a real training row, not an empty one.

    About 36% of the dev split needs no correction, and a model that "fixes" correct text is the
    precision failure F0.5 weights double. Dropping these rows would remove the only examples that
    teach restraint.
    """
    blocks = parse_m2(M2_SAMPLE)
    assert blocks[1]["target"] == blocks[1]["source"]


def test_edits_are_applied_highest_start_first_so_indices_never_shift():
    """Two edits in one sentence, the second after the first, both must land.

    Applying left to right without an offset would put the second edit at the wrong index once the
    first changed the token count. Here the first edit deletes a token, so a naive left-to-right
    application would corrupt the second.
    """
    source = "I has went to the shop yesterday ."
    edits = [(1, 2, ""), (5, 6, "store")]
    assert apply_m2_edits(source, edits) == "I went to the store yesterday ."


def test_an_insertion_and_a_multi_token_replacement_both_work():
    assert apply_m2_edits("It difficult answer", [(1, 1, "is"), (2, 3, "to answer")]) == (
        "It is difficult to answer"
    )


def test_an_out_of_range_edit_is_skipped_rather_than_crashing():
    assert apply_m2_edits("short sentence", [(99, 100, "x")]) == "short sentence"


def test_rows_carry_the_level_and_fall_back_to_the_source_when_the_target_is_empty():
    rows = to_rows(parse_m2(M2_SAMPLE), "B")
    assert all(row["cefr"] == "B" for row in rows)
    assert all(row["answer"] for row in rows)


def test_interleaving_keeps_every_proficiency_level_in_any_prefix():
    """The per-level M2 files are concatenated in order, so a prefix would otherwise be all
    level A — the same truncation trap TOPv2's domain-ordered test split had. The levels are the
    episode axis, and an eval set missing N (native) cannot measure the zero-shot slice at all."""
    rows = [
        {"text": f"{level}{i}", "answer": "x", "cefr": level, "m2": "S x"}
        for level in ("A", "B", "C", "N")
        for i in range(20)
    ]
    interleaved = _interleave_by_level(rows)
    for cut in (4, 8, 40, 80):
        assert len({row["cefr"] for row in interleaved[:cut]}) == 4, cut


# --------------------------------------------------------------------------
# The one-line-per-sentence output contract
# --------------------------------------------------------------------------


def _eval_set(rows: list[dict]):
    return build_eval_set(rows, task="gec_bea19", target=len(rows))


def _row(source: str, target: str, m2: str, cefr: str = "A") -> dict:
    return {"text": source, "answer": target, "cefr": cefr, "m2": m2}


def test_a_single_line_reply_is_taken_as_the_correction():
    eval_set = _eval_set([_row("She are happy .", "She is happy .", "S She are happy .")])
    assert scorer.extract_predictions(["She is happy ."], eval_set) == ["She is happy ."]


def test_a_fenced_or_padded_reply_is_still_one_sentence():
    """Blank lines and code fences are formatting, not extra content."""
    eval_set = _eval_set([_row("a", "b", "S a")])
    assert scorer.extract_predictions(["\n```\nShe is happy .\n```\n"], eval_set) == [
        "She is happy ."
    ]


def test_a_genuinely_multi_line_reply_is_unusable_not_truncated():
    """THE FAILURE THIS CONTRACT EXISTS FOR.

    ERRANT aligns hypothesis lines to source lines POSITIONALLY. A reply carrying two lines of
    substance would shift every subsequent line in the file and score the whole corpus near zero
    for reasons unrelated to grammar. Silently keeping the first line would hide that the model
    is not obeying its output contract, so it is recorded as a format failure instead — which is
    the alternative this repo chose over constrained decoding (B286).
    """
    eval_set = _eval_set([_row("a", "b", "S a")])
    assert scorer.extract_predictions(["She is happy .\nHope that helps!"], eval_set) == [None]


def test_an_empty_reply_is_unusable():
    eval_set = _eval_set([_row("a", "b", "S a")])
    assert scorer.extract_predictions(["   \n  "], eval_set) == [None]


# --------------------------------------------------------------------------
# Failure taxonomy
# --------------------------------------------------------------------------


def test_the_taxonomy_separates_under_from_over_correction():
    """The two opposite failure modes, which a single F0.5 reports as the same middling number.

    `missed_correction` is the fine-tuned small model's signature and wants more training signal;
    `overcorrected_correct_sentence` is the untuned large model's and wants restraint. F0.5
    penalizes the second twice as heavily, so telling them apart is the point.
    """
    assert scorer.failure_category_of({
        "text": "She are happy .", "answer": "She is happy .", "predicted": "She are happy .",
    }) == "missed_correction"
    assert scorer.failure_category_of({
        "text": "She is happy .", "answer": "She is happy .", "predicted": "She seems happy .",
    }) == "overcorrected_correct_sentence"
    assert scorer.failure_category_of({
        "text": "She are happy .", "answer": "She is happy .", "predicted": "She was happy .",
    }) == "wrong_correction"
    assert scorer.failure_category_of({
        "text": "a", "answer": "b", "predicted": None,
    }) == "unusable_output"


# --------------------------------------------------------------------------
# Real ERRANT scoring
# --------------------------------------------------------------------------


# The fixture M2 is ERRANT-GENERATED, not hand-annotated, and that is deliberate. Feeding the real
# corpus's gold correction back in as the hypothesis scores F0.5 0.8934, not 1.0, because the
# reference edits are a human's segmentation and the hypothesis edits are derived by ERRANT's
# alignment rules — a real and structural gap documented in the scorer. These tests are about the
# plumbing, so their reference is machine-generated and the oracle really does reach 1.0.
_FIXTURE = [
    _row(
        "I has went to the store yesterday .",
        "I went to the store yesterday .",
        "S I has went to the store yesterday .\n"
        "A 1 2|||U:VERB:TENSE||||||REQUIRED|||-NONE-|||0",
    ),
    _row(
        "She are very happy today .",
        "She is very happy today .",
        "S She are very happy today .\n"
        "A 1 2|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0",
        cefr="B",
    ),
    _row(
        "This sentence is already correct .",
        "This sentence is already correct .",
        "S This sentence is already correct .\n"
        "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0",
        cefr="C",
    ),
]


@needs_errant
def test_the_oracle_reaches_a_perfect_score_through_the_real_scorer():
    """End-to-end: prompts, extraction, errant_parallel, errant_compare, parsing."""
    eval_set = _eval_set(_FIXTURE)
    raw = [row["answer"] for row in eval_set.all]
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))

    assert result["f1"] == pytest.approx(1.0)
    assert result["metric"] == "errant_f05"
    assert result["per_class"]["precision"] == pytest.approx(1.0)
    assert result["per_class"]["recall"] == pytest.approx(1.0)
    assert result["format_valid"] == pytest.approx(1.0)
    assert result["failures"] == []


@needs_errant
def test_the_score_is_invariant_to_the_model_writing_natural_spacing():
    """THE BUG THIS EXISTS FOR, AND IT COST A RUN'S TEACHER MEASUREMENT.

    W&I+LOCNESS is distributed WORD-TOKENIZED — a space before every period, `do n't` split in
    two — and ERRANT derives its edits by aligning token sequences. A generative model emits
    natural text, so the SAME correction written with natural spacing produced completely
    different edit spans from the gold's.

    Measured on 300 dev sentences before the fix, scoring the gold answer against its own
    reference: corpus tokenization F0.5 0.8726 with 66 false positives, natural spacing F0.5
    0.3102 with 760. A perfect answer lost 56 points on spacing alone.

    That is what produced the 0.1569 the local Qwen teacher first measured on this task in run
    39707195 — a number which then refused synthesis for the whole run and pinned the accuracy
    goal to the floor. It was never a measurement of correction quality.

    The concrete case from that run: source `... on the road .`, gold `... on the road ?`, model
    answered `... on the road?` — the right correction — and ERRANT compared "replace `.` with
    `?`" against "rewrite `road` as `road?` and delete `.`", scoring a false positive AND a false
    negative on a correct answer.
    """
    # Keyed on the source, because `gec_bea19` uses `eval_sampling="shuffled"` — a hand-ordered
    # list positioned to match `_FIXTURE` silently misaligns with `eval_set.all`, which makes the
    # test compare each hypothesis against somebody else's reference.
    natural_by_source = {
        "I has went to the store yesterday .": "I went to the store yesterday.",
        "She are very happy today .": "She is very happy today.",
        "This sentence is already correct .": "This sentence is already correct.",
    }
    eval_set = _eval_set(_FIXTURE)
    natural = [natural_by_source[row["text"]] for row in eval_set.all]
    tokenized = [row["answer"] for row in eval_set.all]

    natural_score = scorer.score(eval_set, scorer.extract_predictions(natural, eval_set))
    tokenized_score = scorer.score(eval_set, scorer.extract_predictions(tokenized, eval_set))

    assert natural_score["f1"] == pytest.approx(tokenized_score["f1"])
    assert natural_score["f1"] == pytest.approx(1.0)


@needs_errant
def test_retokenization_is_idempotent_so_a_fine_tuned_student_is_not_penalized():
    """The student learns the corpus convention from its training targets, so it arrives ALREADY
    tokenized — and a naive re-tokenization made that case worse, not better.

    The corpus writes `I 'm`; spaCy sees a bare apostrophe followed by `m` and splits it again
    into `I ' m`. Measured on 200 dev sentences that was 5 of 8 disagreements and cost 1.5 F0.5
    points against a perfect answer — a penalty aimed squarely at the one system that had done
    the right thing. Canonicalizing clitics before tokenizing removes it.
    """
    already = ["I have n't written to you for ages .", "I 'm eighteen years old ."]
    assert scorer.retokenize(already) == already
    # And natural text converges on the same thing, which is what makes the two cases score alike.
    assert scorer.retokenize(["I haven't written to you for ages .", "I'm eighteen years old ."]) == (
        already
    )


@needs_errant
def test_a_do_nothing_model_does_not_accumulate_false_positives_from_spacing():
    """Before the fix, a model that corrected NOTHING but wrote naturally racked up hundreds of
    false positives — so the metric punished it for its spacing rather than for its silence, and
    an under-correcting model could look like an over-correcting one."""
    eval_set = _eval_set(_FIXTURE)
    # The source itself, written naturally: no correction attempted, only the spacing differs.
    natural_copy = [
        row["text"].replace(" .", ".").replace(" ,", ",") for row in eval_set.all
    ]
    result = scorer.score(eval_set, scorer.extract_predictions(natural_copy, eval_set))

    assert result["per_class"]["fp"] == 0
    assert result["per_class"]["recall"] == pytest.approx(0.0)


@needs_errant
def test_copying_the_source_scores_zero_recall_and_is_reported_as_such():
    """The under-correction extreme. Recall must be 0 and it must be VISIBLE.

    This is why precision and recall are never collapsed into F0.5 alone: a model that changes
    nothing has perfect precision by ERRANT's convention, so precision on its own would look
    excellent.
    """
    eval_set = _eval_set(_FIXTURE)
    raw = [row["text"] for row in eval_set.all]
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))

    assert result["f1"] == pytest.approx(0.0)
    assert result["per_class"]["recall"] == pytest.approx(0.0)
    assert result["per_class"]["tp"] == 0
    # Readable output: the model answered, it just did not correct anything.
    assert result["format_valid"] == pytest.approx(1.0)


@needs_errant
def test_an_unusable_reply_costs_recall_without_earning_precision():
    """The substitution matters. An unusable reply becomes the SOURCE sentence — "corrected
    nothing" — which yields no true and no false positives.

    The alternatives are both wrong: dropping the row shortens the file and misaligns every line
    after it, and writing a blank line makes ERRANT read the sentence as one enormous deletion, a
    false positive the model never proposed.
    """
    eval_set = _eval_set(_FIXTURE)
    result = scorer.score(
        eval_set, scorer.extract_predictions(["one\ntwo"] * len(_FIXTURE), eval_set)
    )
    assert result["format_valid"] == pytest.approx(0.0)
    assert result["per_class"]["fp"] == 0
    assert result["per_class"]["recall"] == pytest.approx(0.0)


@needs_errant
def test_the_score_records_the_scorer_versions_and_the_reference_count():
    """Both are properties of the NUMBER, not of the run, and both change it.

    ERRANT's error typing runs off en_core_web_sm's POS tags, so a model bump moves F0.5 with no
    change to the system. And F0.5 is strongly reference-count dependent — the same systems move
    about twelve points between a 2-reference and a 10-reference CoNLL-14 — so a single-reference
    number compared against a multi-reference one is simply a wrong comparison.
    """
    eval_set = _eval_set(_FIXTURE)
    raw = [row["answer"] for row in eval_set.all]
    per_class = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))["per_class"]

    assert per_class["scorer_errant"] == "3.0.2"
    assert per_class["scorer_en_core_web_sm"] == "3.8.0"
    assert per_class["references_per_sentence"] == 1


@needs_errant
def test_per_error_type_scores_come_back_from_the_scorer():
    """Free from ERRANT's `-cat 2`, and the task's second episode axis."""
    eval_set = _eval_set(_FIXTURE)
    raw = [row["answer"] for row in eval_set.all]
    per_class = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))["per_class"]

    by_type = {k: v for k, v in per_class.items() if k.startswith("f0_5_")}
    assert by_type, "no per-error-type breakdown was parsed"
    assert "f0_5_VERB:SVA" in by_type or "f0_5_VERB:TENSE" in by_type


def test_an_empty_eval_set_scores_zero_without_invoking_the_scorer():
    """No subprocess, no temp files, no crash — and importantly this path must not need the venv."""
    empty = build_eval_set([], task="gec_bea19", target=0)
    result = scorer.score(empty, [])
    assert result["f1"] == 0.0
    assert result["format_valid"] == 0.0


def test_a_missing_errant_venv_raises_rather_than_scoring_zero(monkeypatch, tmp_path):
    """A missing scorer is not a bad model.

    Scoring zero would look exactly like a catastrophic regression and would send the loop into
    hours of data rebuilding to chase it — the same reasoning behind `needs_judge` failing loudly
    on a judge outage.
    """
    monkeypatch.setenv(scorer.ERRANT_VENV_ENV, str(tmp_path / "nonexistent"))
    monkeypatch.setattr(scorer.shutil, "which", lambda _name: None)
    with pytest.raises(scorer.ErrantUnavailable, match="setup_metric_envs"):
        scorer.errant_bin("errant_parallel")


def test_the_scorer_is_located_through_an_env_var_not_a_hardcoded_path():
    """`$ERRANT_VENV` indirection is what keeps this task's `_l40s` and `_cse` launchers
    byte-identical apart from their SBATCH headers, which the SLURM tests enforce."""
    assert scorer.ERRANT_VENV_ENV == "ERRANT_VENV"
    source = open("eval/scorers/gec.py", encoding="utf-8").read()
    assert "/mmfs1/" not in source, "an absolute cluster path leaked into the scorer"
