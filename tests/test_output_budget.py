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


def test_the_prompt_estimate_covers_the_worst_tokenizing_corpus_in_the_suite():
    """The output budget must never let prompt + budget exceed the served context.

    THE BUG THIS EXISTS FOR, TWICE. `output_budget` subtracts an ESTIMATE of the prompt's token
    count from a hard server limit, so an optimistic estimate produces an HTTP 400 rather than a
    truncation. It has now happened to two different corpora with the same arithmetic:

        xlam run 39311800  requested 4736 out, prompt >= 3457, total >= 8193  (max 8192)
        gec_bea19 audit    requested 15796 out, prompt >= 589,  total >= 16385 (max 16384)

    Both are over by exactly one token, from a ~65-token shortfall against the 64-token margin.
    The first was addressed by lowering chars/token from 4.0 to 3.5; the second shows that moved
    the threshold rather than removing the failure. W&I+LOCNESS is word-tokenized — every
    punctuation mark is its own token and contractions are split — so its prompts tokenize at 3.01
    chars/token and 3.5 underestimated them by 109 tokens. 97 of 112 generations failed.

    Pinned as an ARITHMETIC INVARIANT over the worst ratio actually observed, not as an assertion
    that the constant equals some number: the constant may legitimately change again, but
    `prompt_tokens + output_budget(prompt) <= served_context()` may not stop holding.
    """
    from config.token_budget import _FLOOR_TOKENS, output_budget, served_context

    # The worst ratio measured across real 5-shot generation prompts for all five suite tasks,
    # which is gec_bea19's. A prompt of `chars` that really costs `chars / 3.01` tokens.
    worst_chars_per_token = 3.01
    context = served_context()
    for chars in (500, 1834, 2302, 4298, 12000):
        prompt = "x" * chars
        real_tokens = int(chars / worst_chars_per_token) + 1
        # Only meaningful for a prompt that FITS. A prompt whose own tokens exceed the window
        # cannot be rescued by any output budget, and `_FLOOR_TOKENS` deliberately returns a
        # working-sized request there rather than a zero — trading a loud 400 for a silent
        # truncation is the one thing that module refuses to do. That case is asserted separately.
        if real_tokens >= context - _FLOOR_TOKENS:
            continue
        budget = output_budget(prompt)
        assert real_tokens + budget <= context, (
            f"a {chars}-char prompt really costs ~{real_tokens} tokens; asking for {budget} "
            f"output tokens totals {real_tokens + budget} against a served context of "
            f"{context} — that is an HTTP 400, not a truncation"
        )


def test_the_budget_still_asks_for_something_usable_after_the_conservative_estimate():
    """A conservative prompt estimate must not shrink the budget into uselessness.

    The estimate exists to avoid a 400, and the floor exists so that avoiding one never turns into
    a request too small to carry a reply. Over-asking is free — generation stops at EOS — so the
    only real requirement is that a short prompt still gets a large budget.
    """
    from config.token_budget import _FLOOR_TOKENS, output_budget

    assert output_budget("short prompt") > 4096
    # Even a prompt that has nearly filled the window returns a working request rather than 0.
    assert output_budget("x" * 200_000) >= _FLOOR_TOKENS
