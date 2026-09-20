"""DialogSum after the 2026-09-06 rework: three references, ROUGE, and no judge.

Network-free: rows are built here. The live fetch is exercised by `scripts/preflight_tasks.py`.
The BERTScore path is not tested here — it needs a separate venv and roberta-large on CPU, is a
secondary metric by design, and degrades to a recorded absence rather than an error.
"""
from __future__ import annotations

import pytest

from data.eval_set import build_eval_set
from data.loaders.dialogsum import convert_dialogsum_rows
from eval.scorers import summarization as scorer


def _row(dialogue: str, references: list[str]) -> dict:
    return {
        "text": dialogue,
        "answer": references[0],
        "references": references,
        "topic": "t",
        "_instruction": "Summarize the following conversation in one to three sentences. "
                        "Write only the summary — do not continue the conversation or reply to it.",
    }


def _eval_set(rows: list[dict]):
    return build_eval_set(rows, task="dialogsum", target=len(rows))


# --------------------------------------------------------------------------
# The rework's structural claims
# --------------------------------------------------------------------------


def test_the_task_no_longer_uses_the_llm_judge():
    """The judge is comparable to no published number, non-deterministic, and costs money every
    eval. `toolbench` remains the only judged task, which is correct: ToolEval's pass rate is
    DEFINED as a judged vote."""
    from tasks import TASKS, get_task

    spec = get_task("dialogsum")
    assert spec.needs_judge is False
    assert spec.judge_overlap is False
    assert {name for name, s in TASKS.items() if s.needs_judge} == {"toolbench"}


def test_samsum_is_not_reachable_from_this_task_any_more():
    """It is single-reference, so rows mined from it could not be scored under this metric, and
    mixing them in would quietly restore the two-corpus blend the rework removed. Paid discovery
    is off so it cannot return by the side door either."""
    from tasks import get_task

    spec = get_task("dialogsum")
    assert all("samsum" not in source.hf_id.lower() for source in spec.mining_sources)
    assert spec.allow_paid_discovery is False


def test_the_selection_and_report_metrics_are_both_declared_and_differ():
    from tasks import get_task

    spec = get_task("dialogsum")
    assert spec.metric_name == "rouge_l"
    assert spec.report_metric_name == "rouge_1_2_l_bertscore"
    assert spec.report_score is not spec.score


# --------------------------------------------------------------------------
# Row conversion
# --------------------------------------------------------------------------


def test_three_test_summaries_become_three_references():
    rows = convert_dialogsum_rows([{
        "dialogue": "#Person1#: hi", "summary1": "a", "summary2": "b", "summary3": "c",
        "topic1": "greeting",
    }])
    assert rows[0]["references"] == ["a", "b", "c"]
    # `answer` is references[0] so a one-reference row and a three-reference row have one shape,
    # and `dataset_integrity.validate_rows` (which type-checks `answer`) still applies.
    assert rows[0]["answer"] == "a"


def test_a_single_summary_train_row_becomes_one_reference():
    rows = convert_dialogsum_rows([{"dialogue": "#Person1#: hi", "summary": "only one"}])
    assert rows[0]["references"] == ["only one"]
    assert rows[0]["answer"] == "only one"


def test_a_row_with_no_summary_or_no_dialogue_is_dropped():
    assert convert_dialogsum_rows([{"dialogue": "x", "summary": "  "}]) == []
    assert convert_dialogsum_rows([{"dialogue": "", "summary": "y"}]) == []


# --------------------------------------------------------------------------
# Multi-reference scoring
# --------------------------------------------------------------------------


def test_matching_any_one_of_the_three_references_scores_perfectly():
    """THE PROPERTY THE REWORK EXISTS FOR.

    A summary has many correct forms, so scoring against one annotator's wording makes the number
    a lottery about whose phrasing the model matched. Here the prediction copies the THIRD
    reference; against reference 1 alone it would look poor.
    """
    third = "Ms. Dawson takes a dictation about instant messaging in the office."
    rows = [_row("#Person1#: memo please", ["Something about trains.", "A memo is written.", third])]
    eval_set = _eval_set(rows)
    result = scorer.score(eval_set, scorer.extract_predictions([third], eval_set))

    assert result["f1"] == pytest.approx(1.0)
    assert result["per_class"]["references_per_row"] == pytest.approx(3.0)


def test_the_published_human_ceiling_travels_with_the_score():
    """So a 47 is read against 53, not against 100.

    Out of the box is ~36 ROUGE-1, fine-tuned ~47, human 53.35. Both the ~11 points fine-tuning
    buys and the ~6 still on the table are real, and neither is legible without the ceiling.
    """
    eval_set = _eval_set([_row("#Person1#: hi", ["a summary"])])
    per_class = scorer.score(eval_set, scorer.extract_predictions(["a summary"], eval_set))["per_class"]

    assert per_class["human_ceiling_rouge_1"] == pytest.approx(0.5335)
    assert per_class["human_ceiling_rouge_2"] == pytest.approx(0.2672)
    assert per_class["human_ceiling_rouge_l"] == pytest.approx(0.5084)


def test_the_report_pass_reports_rouge_1_and_2_alongside_l(monkeypatch, tmp_path):
    # BERTScore is deliberately disabled here: roberta-large on CPU takes minutes, and this test
    # is about the ROUGE decomposition. Its own absence-handling is covered separately below.
    monkeypatch.setenv(scorer.METRICS_VENV_ENV, str(tmp_path / "nonexistent"))
    eval_set = _eval_set([_row("#Person1#: hi", ["the cat sat on the mat"])])
    result = scorer.score_report(
        eval_set, scorer.extract_predictions(["a cat was sitting on the mat"], eval_set)
    )
    assert result["metric"] == "rouge_1_2_l_bertscore"
    for key in ("rouge_1", "rouge_2", "rouge_l"):
        assert 0.0 < result["per_class"][key] < 1.0


def test_a_missing_bertscore_venv_is_recorded_and_does_not_break_the_headline(monkeypatch, tmp_path):
    """BERTScore is SECONDARY. A missing secondary metric must not take down a report whose
    headline — ROUGE, which needs nothing outside the training venv — is perfectly valid."""
    monkeypatch.setenv(scorer.METRICS_VENV_ENV, str(tmp_path / "nonexistent"))
    eval_set = _eval_set([_row("#Person1#: hi", ["a summary"])])
    result = scorer.score_report(eval_set, scorer.extract_predictions(["a summary"], eval_set))

    assert result["f1"] == pytest.approx(1.0)
    assert "bertscore_unavailable" in result["per_class"]


# --------------------------------------------------------------------------
# The B250 failure mode
# --------------------------------------------------------------------------


def test_a_summary_may_mention_the_speakers_and_is_not_a_continuation():
    """THE BUG THIS EXISTS FOR.

    DialogSum's gold summaries REFER to the participants by tag — "Ms. Dawson helps #Person1# to
    write a memo" — and 78% of the 1,500 test references do so. Matching on a bare `#Person1#`
    scored the ORACLE at format_valid 0.15, condemning the reference summaries themselves as
    continuations of the conversation.
    """
    summary = "Ms. Dawson helps #Person1# write a memo about instant messaging."
    eval_set = _eval_set([_row("#Person1#: memo please", [summary])])
    result = scorer.score(eval_set, scorer.extract_predictions([summary], eval_set))

    assert result["format_valid"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_a_reply_carrying_a_turn_label_is_a_continuation_and_a_format_failure():
    """The colon is the test: `#Person1#:` is how the TRANSCRIPT marks who is speaking, and 0 of
    1,500 gold summaries contain it. This is the B250 failure — with no row-level instruction the
    model treated the transcript as something to reply to — and it must present as a prompt
    problem rather than as generic low overlap."""
    eval_set = _eval_set([_row("#Person1#: hi", ["They greet each other."])])
    result = scorer.score(
        eval_set, scorer.extract_predictions(["#Person1#: How about you?"], eval_set)
    )
    assert result["format_valid"] == pytest.approx(0.0)
    assert result["failures"][0]["error_type"] == "continued_the_conversation"


def test_copying_the_transcript_is_its_own_failure_category():
    """The characteristic failure on a corpus whose conversations are long and summaries short:
    the model extracts instead of abstracting. It wants different data from a wrong summary."""
    long_output = " ".join(["the speakers discuss many things at great length"] * 8)
    eval_set = _eval_set([_row("#Person1#: hi", ["They greet each other."])])
    result = scorer.score(eval_set, scorer.extract_predictions([long_output], eval_set))
    assert result["failures"][0]["error_type"] == "copied_the_transcript"


def test_an_empty_reply_is_its_own_category():
    assert scorer.failure_category_of(
        {"references": ["a"], "predicted": "   "}
    ) == "empty_output"


def test_the_reasoning_block_is_not_scored_as_part_of_the_summary():
    """A CoT-trained model emits `<reasoning>` before its answer, and handing that to the scorer
    would penalize output that contains a perfectly good summary (B251)."""
    eval_set = _eval_set([_row("#Person1#: hi", ["They greet each other."])])
    raw = ["<reasoning>\nThe two people say hello.\n</reasoning>\n\nThey greet each other."]
    assert scorer.extract_predictions(raw, eval_set) == ["They greet each other."]


def test_the_instruction_is_resolved_once_for_the_dataset_not_per_row():
    """Synthetic rows carry no `_instruction`, so a per-row lookup would silently give real and
    generated rows different prompts inside one training set."""
    rows = [_row("#Person1#: hi", ["a"]), {"text": "#Person1#: yo", "answer": "b", "references": ["b"]}]
    assert scorer.resolve_instruction(rows).startswith("Summarize the following conversation")
    # Even with no row carrying one, there is a real instruction and never the
    # question-answering default that broke this task before (B250).
    assert "do not continue the conversation" in scorer.resolve_instruction([{"text": "x"}])


def test_an_empty_eval_set_scores_zero_without_dividing_by_zero():
    empty = build_eval_set([], task="dialogsum", target=0)
    for score_fn in (scorer.score, scorer.score_report):
        result = score_fn(empty, [])
        assert result["f1"] == 0.0
        assert result["format_valid"] == 0.0
