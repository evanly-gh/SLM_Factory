"""address_bench — loader, prompt parity, and the F2 metric's honesty properties.

The tests that matter most here are the metric ones. F2 was chosen over `minority_f1` because the
gate must not miss addressed turns, and that choice buys a 0.625 floor for a model that answers
`addressed` to everything. These tests pin both halves of that trade: that F2 really does punish a
miss harder than a false yes, and that the diagnostics which expose the floor are present and
correct. A future change that swaps the metric back to something symmetric should fail here
loudly, not silently produce a run that looks fine.
"""
from __future__ import annotations

import pytest

from data.loaders.address_bench import (
    CONTEXT_LINES,
    convert_address_rows,
    load_address_bench,
    render_text,
)
from eval.scorers import address_gate as scorer
from eval.scorers.address_gate import ADDRESSED, NOT_ADDRESSED
from tasks import get_task


class _EvalSet:
    """The surface `build_prompts` / `score` use: an object with `.all`."""

    def __init__(self, rows):
        self.all = rows


def _rows(spec):
    """(label, bucket) pairs -> scoreable eval rows."""
    return _EvalSet([
        {"text": f"row {i}", "label": label, "bucket": bucket}
        for i, (label, bucket) in enumerate(spec)
    ])


# ---------------------------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------------------------

class TestRendering:
    def test_renders_context_speaker_and_quoted_latest_line(self):
        text = render_text({
            "context": "User: hey\nSpeaker2: hi",
            "speaker": "User",
            "current_turn": "Glasses, set a timer.",
        })
        assert "Conversation so far:" in text
        assert "User: hey" in text
        assert 'Latest line, User: "Glasses, set a timer."' in text

    def test_empty_context_uses_the_upstream_sentinel(self):
        text = render_text({"context": "", "speaker": "User", "current_turn": "Set a timer."})
        assert "(nothing yet)" in text

    def test_context_is_truncated_from_the_oldest_line(self):
        context = "\n".join(f"User: line {i}" for i in range(20))
        text = render_text({"context": context, "speaker": "User", "current_turn": "x"},
                           n_context=CONTEXT_LINES)
        assert "line 19" in text, "the most recent line must survive truncation"
        assert "line 11" not in text, f"only the last {CONTEXT_LINES} lines should be kept"

    def test_pending_requests_note_is_rendered_and_pluralized(self):
        one = render_text({"context": "", "current_turn": "x", "pending_requests": 1})
        many = render_text({"context": "", "current_turn": "x", "pending_requests": 3})
        assert "1 earlier request.)" in one
        assert "3 earlier requests.)" in many

    def test_no_note_when_nothing_is_pending(self):
        assert "owes the User" not in render_text({"context": "", "current_turn": "x"})


class TestConversion:
    def test_maps_addressed_boolean_to_the_label_vocabulary(self):
        rows = convert_address_rows([
            {"current_turn": "a", "addressed": True, "bucket": "P1"},
            {"current_turn": "b", "addressed": False, "bucket": "N1"},
        ])
        assert [r["label"] for r in rows] == [ADDRESSED, NOT_ADDRESSED]

    def test_carries_the_bucket_through_for_the_failure_taxonomy(self):
        rows = convert_address_rows([{"current_turn": "a", "addressed": True, "bucket": "P7"}])
        assert rows[0]["bucket"] == "P7"

    @pytest.mark.parametrize("bad", [
        {"addressed": True},                       # no current_turn
        {"current_turn": "a"},                     # no label
        {"current_turn": "   ", "addressed": True},  # blank turn
    ])
    def test_drops_rows_that_would_require_inventing_ground_truth(self, bad):
        assert convert_address_rows([bad]) == []


class TestSplit:
    """The vendored bundle is required; these are the real-corpus invariants."""

    def test_split_is_deterministic_across_calls(self):
        a_train, a_eval = load_address_bench(500, 200, log=lambda *_: None)
        b_train, b_eval = load_address_bench(500, 200, log=lambda *_: None)
        assert [r["text"] for r in a_eval] == [r["text"] for r in b_eval]
        assert [r["text"] for r in a_train] == [r["text"] for r in b_train]

    def test_no_row_straddles_the_contamination_firewall(self):
        train, test = load_address_bench(3000, 1000, log=lambda *_: None)
        normalize = lambda s: " ".join(s.lower().split())  # noqa: E731
        assert not ({normalize(r["text"]) for r in train} & {normalize(r["text"]) for r in test})

    def test_eval_split_keeps_every_bucket(self):
        _, test = load_address_bench(3000, 1000, log=lambda *_: None)
        assert len({r["bucket"] for r in test}) == 20, "per-bucket accuracy needs all 20 buckets"

    def test_both_labels_are_present_on_both_sides(self):
        train, test = load_address_bench(3000, 1000, log=lambda *_: None)
        assert {r["label"] for r in train} == {ADDRESSED, NOT_ADDRESSED}
        assert {r["label"] for r in test} == {ADDRESSED, NOT_ADDRESSED}


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------

class TestExtraction:
    @pytest.mark.parametrize("raw,want", [
        ("addressed", ADDRESSED),
        ("not_addressed", NOT_ADDRESSED),
        ("  ADDRESSED.  ", ADDRESSED),
        ("`not_addressed`", NOT_ADDRESSED),
        ("The answer is addressed.", ADDRESSED),
        ("<think>hmm</think> not_addressed", NOT_ADDRESSED),
    ])
    def test_reads_the_label(self, raw, want):
        assert scorer.extract_predictions([raw], _rows([])) == [want]

    def test_negative_is_never_read_as_a_positive(self):
        """`addressed` is a substring of `not_addressed`; longest-first ordering is what saves it."""
        for raw in ("not_addressed", "I think not_addressed", "Answer: not_addressed"):
            assert scorer.extract_predictions([raw], _rows([])) == [NOT_ADDRESSED], raw

    @pytest.mark.parametrize("raw", ["", "   ", "I'm not sure what you mean."])
    def test_a_non_answer_is_a_recorded_failure_not_a_lucky_label(self, raw):
        assert scorer.extract_predictions([raw], _rows([])) == ["__EXTRACTION_FAILED__"]

    def test_a_label_buried_mid_paragraph_is_not_an_answer(self):
        rambling = "addressed " + "but then again who can really say for certain about any of it. " * 5
        assert scorer.extract_predictions([rambling], _rows([])) == ["__EXTRACTION_FAILED__"]

    def test_the_token_cap_is_what_bounds_the_tail_heuristic(self):
        """A known, bounded limitation — pinned here so a later cap change has to confront it.

        Step 2 of extraction accepts a label word in the answer's TAIL, so a long paragraph whose
        FINAL sentence happens to contain `addressed` would be read as an answer. That is not
        reachable in practice: `max_new_tokens=16` means a generation is ~60-80 characters, far
        too short to be a paragraph with an incidental trailing match. If someone raises the cap,
        this test fails and the heuristic has to be revisited rather than silently weakened.
        """
        assert get_task("address_bench").max_new_tokens <= 24


# ---------------------------------------------------------------------------------------------
# The metric — the reason this task exists in the shape it does
# ---------------------------------------------------------------------------------------------

class TestF2Asymmetry:
    """A miss must cost more than a false yes. That is the whole point of beta=2."""

    def test_a_miss_costs_more_than_a_false_yes(self):
        gold = [ADDRESSED] * 10 + [NOT_ADDRESSED] * 10
        perfect = list(gold)

        one_miss = list(perfect)
        one_miss[0] = NOT_ADDRESSED          # addressed turn, gate said no

        one_false_yes = list(perfect)
        one_false_yes[10] = ADDRESSED        # quiet turn, gate woke up

        rows = _rows([(g, "P1" if g == ADDRESSED else "N1") for g in gold])
        miss_score = scorer.score_address_gate(rows, one_miss)["f1"]
        false_yes_score = scorer.score_address_gate(rows, one_false_yes)["f1"]

        assert miss_score < false_yes_score, (
            "F2 must punish a missed addressed turn harder than a spurious wake-up; "
            f"got miss={miss_score:.4f} false_yes={false_yes_score:.4f}"
        )

    def test_f2_exceeds_f1_when_recall_leads_precision(self):
        """Sanity check that beta is actually applied and in the right direction."""
        gold = [ADDRESSED] * 10 + [NOT_ADDRESSED] * 10
        preds = [ADDRESSED] * 15 + [NOT_ADDRESSED] * 5   # recall 1.0, precision 0.667
        assert scorer.fbeta(preds, gold, beta=2.0) > scorer.fbeta(preds, gold, beta=1.0)


class TestDegenerateGates:
    """The two collapses, and the diagnostics that must expose them."""

    def test_always_no_scores_zero(self):
        gold = [ADDRESSED] * 5 + [NOT_ADDRESSED] * 15
        rows = _rows([(g, "P1" if g == ADDRESSED else "N1") for g in gold])
        result = scorer.score_address_gate(rows, [NOT_ADDRESSED] * 20)
        assert result["f1"] == 0.0
        assert result["per_class"]["recall"] == 0.0
        assert result["per_class"]["missed"] == 5.0

    def test_always_yes_hits_the_documented_0_625_floor(self):
        """The known weakness of this metric, pinned so nobody rediscovers it mid-run."""
        gold = [ADDRESSED] * 5 + [NOT_ADDRESSED] * 15      # the corpus's 25/75 rate
        rows = _rows([(g, "P1" if g == ADDRESSED else "N1") for g in gold])
        result = scorer.score_address_gate(rows, [ADDRESSED] * 20)
        assert result["f1"] == pytest.approx(0.625, abs=1e-6)

    def test_always_yes_is_unmistakable_from_the_diagnostics(self):
        """F2=0.625 could be mistaken for learning; false_yes_rate=1.0 cannot be."""
        gold = [ADDRESSED] * 5 + [NOT_ADDRESSED] * 15
        rows = _rows([(g, "P1" if g == ADDRESSED else "N1") for g in gold])
        per_class = scorer.score_address_gate(rows, [ADDRESSED] * 20)["per_class"]
        assert per_class["false_yes_rate"] == 1.0
        assert per_class["recall"] == 1.0
        assert per_class["precision"] == pytest.approx(0.25)

    def test_unreadable_answers_cannot_inflate_recall(self):
        """A garbage answer is a predicted negative here, so it never earns credit for a positive.

        Scored end to end from RAW output, because the extraction step is what turns `???` into
        `__EXTRACTION_FAILED__` and the two halves have to agree for `format_valid` to mean
        anything.
        """
        rows = _rows([(ADDRESSED, "P1")] * 10)
        predictions = scorer.extract_predictions(["???"] * 10, rows)
        assert predictions == ["__EXTRACTION_FAILED__"] * 10
        result = scorer.score_address_gate(rows, predictions)
        assert result["f1"] == 0.0
        assert result["per_class"]["recall"] == 0.0
        assert result["per_class"]["format_valid"] == 0.0


class TestDiagnostics:
    def test_per_bucket_accuracy_is_reported(self):
        rows = _rows([(ADDRESSED, "P1"), (ADDRESSED, "P1"), (NOT_ADDRESSED, "N5")])
        per_class = scorer.score_address_gate(
            rows, [ADDRESSED, NOT_ADDRESSED, NOT_ADDRESSED]
        )["per_class"]
        assert per_class["bucket_P1"] == pytest.approx(0.5)
        assert per_class["bucket_N5"] == pytest.approx(1.0)

    def test_synthesized_rows_do_not_join_a_real_buckets_denominator(self):
        rows = _rows([(ADDRESSED, "P1"), (ADDRESSED, "")])
        per_class = scorer.score_address_gate(rows, [ADDRESSED, NOT_ADDRESSED])["per_class"]
        assert per_class["bucket_P1"] == pytest.approx(1.0)
        assert per_class["bucket_synthetic"] == pytest.approx(0.0)

    def test_failure_category_names_direction_and_bucket(self):
        assert scorer.failure_category_of(
            {"predicted": NOT_ADDRESSED, "bucket": "P1"}) == "missed_P1"
        assert scorer.failure_category_of(
            {"predicted": ADDRESSED, "bucket": "N5"}) == "false_yes_N5"
        assert scorer.failure_category_of(
            {"predicted": "__EXTRACTION_FAILED__", "bucket": "P1"}) == "extraction_failed"

    def test_failures_carry_their_category(self):
        rows = _rows([(ADDRESSED, "P3")])
        result = scorer.score_address_gate(rows, [NOT_ADDRESSED])
        assert len(result["failures"]) == 1
        assert result["failures"][0]["error_type"] == "missed_P3"


# ---------------------------------------------------------------------------------------------
# Train / serve parity — B290
# ---------------------------------------------------------------------------------------------

class TestPromptParity:
    def test_training_turn_prompt_is_byte_identical_to_the_eval_prompt(self):
        spec = get_task("address_bench")
        row = {"text": "Conversation so far:\n(nothing yet)\n\nLatest line, User: \"Set a timer.\"",
               "label": ADDRESSED}
        train_prompt, target, marker = spec.build_training_turn(row, None)
        eval_prompt = spec.build_prompts(_EvalSet([row]))[0]
        assert train_prompt == eval_prompt
        assert target == ADDRESSED
        assert marker == "Answer"

    def test_the_prompt_states_the_gate_definition(self):
        """Without the rule, the model is guessing from syntax — see the scorer docstring."""
        prompt = get_task("address_bench").build_prompts(
            _EvalSet([{"text": "x", "label": ADDRESSED}]))[0]
        assert "addressed to the assistant" in prompt
        assert "A line from anyone other than the User is always" in prompt
        assert "has no fixed name" in prompt


class TestSpecWiring:
    def test_metric_name_matches_what_the_scorer_reports(self):
        spec = get_task("address_bench")
        rows = _rows([(ADDRESSED, "P1")])
        assert spec.score(rows, [ADDRESSED])["metric"] == spec.metric_name

    def test_mining_is_off_because_the_corpus_has_no_mirror(self):
        spec = get_task("address_bench")
        assert spec.mining_sources == ()
        assert spec.allow_paid_discovery is False

    def test_both_labels_are_defined_for_the_teacher(self):
        spec = get_task("address_bench")
        assert set(spec.label_definitions) == {ADDRESSED, NOT_ADDRESSED}
