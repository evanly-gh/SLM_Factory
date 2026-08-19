# tests/test_output_budget.py
"""
The orchestrator's decision is lost entirely when its response exceeds the output ceiling.

Reference failure (slm-clinc150-cse-38179864, B240): the first iterate call returned exactly
1536 output tokens — the cap — so the JSON was cut mid-string. The error surfaced as "no
parseable JSON", the reask therefore told the model to "return valid JSON", it wrote another
over-long answer, and the run fell back to the score-band default. The orchestrator had chosen
`data_rebuild`; `hyperparameter` was executed instead.
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

from agent.nodes.iterate import (
    HYPOTHESIS_MAX_CHARS,
    HYPOTHESIS_TARGET_WORDS,
    _ITERATE_MAX_TOKENS,
    _ITERATE_SYSTEM,
    _hit_output_cap,
)


class TestOutputCeilingIsBigEnough:
    def test_ceiling_leaves_room_for_a_plan_plus_full_hypothesis(self):
        """4000 chars of hypothesis is ~1000 tokens; a data_rebuild plan needs the rest."""
        assert _ITERATE_MAX_TOKENS >= 4096

    def test_hard_cap_is_a_runaway_guard_not_a_style_control(self):
        assert HYPOTHESIS_MAX_CHARS >= 4000


class TestModelIsToldItsBudget:
    """Prompting beats truncating: the model can only respect a limit it has been told."""

    def test_prompt_states_a_target_length(self):
        assert f"about {HYPOTHESIS_TARGET_WORDS} words" in _ITERATE_SYSTEM

    def test_prompt_states_the_hard_output_ceiling(self):
        assert f"{_ITERATE_MAX_TOKENS} output tokens" in _ITERATE_SYSTEM

    def test_prompt_warns_that_overrun_discards_the_decision(self):
        assert "DISCARDED" in _ITERATE_SYSTEM

    def test_prompt_explains_the_hypothesis_is_replayed(self):
        """Justifies spending words: the text becomes its own future context."""
        assert "SHOWN THIS TEXT AGAIN" in _ITERATE_SYSTEM

    def test_no_unsubstituted_placeholders(self):
        assert "{hypothesis_target_words}" not in _ITERATE_SYSTEM
        assert "{max_output_tokens}" not in _ITERATE_SYSTEM


class TestTruncationIsDetected:
    def test_detects_max_tokens_stop_reason(self):
        response = MagicMock(response_metadata={"stop_reason": "max_tokens"})
        assert _hit_output_cap(response) is True

    def test_normal_completion_is_not_flagged(self):
        response = MagicMock(response_metadata={"stop_reason": "end_turn"})
        assert _hit_output_cap(response) is False

    def test_missing_metadata_is_not_flagged(self):
        assert _hit_output_cap(MagicMock(response_metadata=None)) is False
        assert _hit_output_cap(object()) is False


class TestReaskCorrectsLengthNotFormat:
    """
    A length failure and a format failure need OPPOSITE corrections. Telling a model that
    overran to "return valid JSON" reproduces the overrun, which is what happened twice.
    """

    def _capture_reask_prompt(self, validation_error):
        from agent.nodes.iterate import _reask_json_only

        captured = {}

        def fake_invoke(_llm, messages, **_kwargs):
            captured["text"] = messages[-1].content
            return MagicMock(
                tool_calls=[],
                response_metadata={"stop_reason": "end_turn"},
                content='{"intervention":"hyperparameter","hypothesis":"h",'
                        '"hyperparams":{"lora_rank":32}}',
            )

        # A stub module, not a patch of the real one: patching would import langchain_anthropic,
        # which costs about a minute of wall clock here, to replace the one class it exports.
        stub = ModuleType("langchain_anthropic")
        stub.ChatAnthropic = lambda *_args, **_kwargs: MagicMock()
        with (
            patch.dict(sys.modules, {"langchain_anthropic": stub}),
            patch("agent.nodes.iterate.tracked_chat_anthropic_invoke", side_effect=fake_invoke),
        ):
            _reask_json_only(
                [MagicMock(content="sys"), MagicMock(content="user")],
                validation_error=validation_error,
                task="clinc150",
                state={},
            )
        return captured["text"]

    def test_length_failure_asks_for_a_shorter_answer(self):
        text = self._capture_reask_prompt(
            ValueError("response was cut off after 4096 output tokens (stop_reason=max_tokens)")
        )
        assert "RAN OUT OF OUTPUT SPACE" in text
        assert "SHORTER" in text

    def test_length_failure_names_a_concrete_word_budget(self):
        text = self._capture_reask_prompt(
            ValueError("cut off (stop_reason=max_tokens), so the JSON is incomplete")
        )
        assert "words" in text

    def test_format_failure_keeps_the_original_diagnostic_wording(self):
        text = self._capture_reask_prompt(ValueError("unsupported field(s): primary_strategy"))
        assert "RAN OUT OF OUTPUT SPACE" not in text
        assert "diagnostic" in text
