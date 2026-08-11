# tests/test_hypothesis_integrity.py
"""
The orchestrator's hypothesis is its causal reasoning for an intervention. It is the single
most information-dense field in the run and it was being cut in five separate places, the
worst at the source (B238). These tests pin the full text end to end.

Reference failure: in slm-clinc150-cse-38155022 every long hypothesis landed at exactly 240
characters, ending mid-word ("...cancel->freeze_a"), so the confusion-pair list — the
actionable part — was destroyed before it reached the log, the DAG, or the next prompt.
"""
import re

import pytest

from agent.context_manager import _extract_iteration_summary, compact_trajectory
from agent.nodes.iterate import HYPOTHESIS_MAX_CHARS, _validate_decision_json


def _payload(hypothesis: str) -> dict:
    """Minimal valid hyperparameter decision carrying the hypothesis under test."""
    return {
        "intervention": "hyperparameter",
        "hypothesis": hypothesis,
        "hyperparams": {"lora_rank": 32},
    }


def _long_hypothesis() -> str:
    """A realistic hypothesis of the length the orchestrator actually produces."""
    return (
        "Hard-bucket accuracy is 0.683 (n=142) with multiple specific confusion pairs: "
        "change_ai_name<->change_user_name (7 combined), change_language->translate (4), "
        "cancel->freeze_account (3), account_blocked->extraction_failed (3), "
        "accept_reservations->restaurant_reservation (2), and w2->income_tax (2). These are "
        "near-synonym intents whose surface forms overlap heavily, so the decision boundary is "
        "under-determined rather than the model being under-trained."
    )


class TestNoTruncationAtSource:
    """iterate.py:_validate_iteration_response was the source cut: hypothesis.strip()[:240]."""

    def test_realistic_hypothesis_survives_validation_intact(self):
        hypothesis = _long_hypothesis()
        assert len(hypothesis) > 240, "fixture must exceed the old 240-char cap"

        validated = _validate_decision_json(
            _payload(hypothesis),
            state={},
        )

        assert validated["hypothesis"] == hypothesis, (
            "the full hypothesis must reach the log, the DAG and the next prompt"
        )

    def test_no_longer_cut_at_240(self):
        hypothesis = _long_hypothesis()
        validated = _validate_decision_json(
            _payload(hypothesis),
            state={},
        )
        assert len(validated["hypothesis"]) != 240
        assert not validated["hypothesis"].endswith("freeze_a")

    def test_confusion_pairs_are_preserved(self):
        """The tail of the hypothesis is the actionable part; that is exactly what was lost."""
        validated = _validate_decision_json(
            _payload(_long_hypothesis()),
            state={},
        )
        assert "w2->income_tax" in validated["hypothesis"]
        assert "under-determined" in validated["hypothesis"]

    def test_runaway_output_is_still_bounded_and_announced(self, capsys):
        """A bound still exists for adversarial output — but it must not be silent."""
        runaway = "x" * (HYPOTHESIS_MAX_CHARS + 5000)
        validated = _validate_decision_json(
            _payload(runaway),
            state={},
        )
        assert len(validated["hypothesis"]) == HYPOTHESIS_MAX_CHARS
        assert "truncat" in capsys.readouterr().out.lower(), (
            "silent truncation is what hid this bug for the whole CLINC150 run"
        )

    def test_bound_is_generous_enough_for_real_hypotheses(self):
        assert HYPOTHESIS_MAX_CHARS >= 1500


class TestCompactedTrajectoryKeepsFullHypothesis:
    """context_manager applied a second cut at [:100] on the already-cut text."""

    def _section(self, hypothesis: str, iteration: int = 4) -> str:
        return (
            f"## Iteration {iteration} - 2026-08-04T21:08:42\n\n"
            "### Eval results\n"
            f"- f(\u03c0_{iteration}): 0.8294\n\n"
            "### Iteration policy decision\n"
            "- Score band: 0.80-0.95\n"
            "- Next intervention: data_rebuild\n"
            f"- Hypothesis: {hypothesis}\n\n"
            "### Hardware profile (Phase 1: theoretical)\n"
            "- Model: Qwen/Qwen3-0.6B | Weight size: 1200MB | Tier: 0\n"
        )

    def test_summary_keeps_the_whole_hypothesis(self):
        hypothesis = _long_hypothesis()
        summary = _extract_iteration_summary(self._section(hypothesis))
        assert hypothesis in summary

    def test_summary_does_not_cut_at_100_chars(self):
        summary = _extract_iteration_summary(self._section(_long_hypothesis()))
        assert "under-determined" in summary


class TestEmptyHypothesisDoesNotBleed:
    """
    Iteration 1 has no LLM hypothesis, so the field renders empty. The old regex
    `Hypothesis:\\s*(.+)` let \\s* eat the newline and captured the NEXT non-empty line,
    producing the nonsense summary `- Iter 1: ... - ### Hardware profile (Phase 1: theoretical)`.
    """

    def _section_with_empty_hypothesis(self) -> str:
        return (
            "## Iteration 1 - 2026-08-04T18:02:11\n\n"
            "### Eval results\n"
            "- f(\u03c0_1): 0.8261\n\n"
            "### Iteration policy decision\n"
            "- Score band: 0.80-0.95\n"
            "- Next intervention: hyperparameter\n"
            "- Hypothesis: \n\n"
            "### Hardware profile (Phase 1: theoretical)\n"
            "- Model: Qwen/Qwen3-0.6B | Weight size: 1200MB | Tier: 0\n"
        )

    def test_does_not_capture_the_following_heading(self):
        summary = _extract_iteration_summary(self._section_with_empty_hypothesis())
        assert "Hardware profile" not in summary
        assert "Phase 1: theoretical" not in summary

    def test_still_reports_the_real_fields(self):
        summary = _extract_iteration_summary(self._section_with_empty_hypothesis())
        assert "Iter 1" in summary
        assert "0.8261" in summary
        assert "hyperparameter" in summary

    def test_other_fields_do_not_bleed_when_empty(self):
        """Same regex class of bug for every field, not just hypothesis."""
        section = (
            "## Iteration 2 - 2026-08-04T19:00:00\n\n"
            "### Iteration policy decision\n"
            "- Score band: \n"
            "- Next intervention: \n"
            "- Hypothesis: real reasoning here\n\n"
            "### Hardware profile (Phase 1: theoretical)\n"
        )
        summary = _extract_iteration_summary(section)
        assert "Hardware profile" not in summary
        assert "real reasoning here" in summary
