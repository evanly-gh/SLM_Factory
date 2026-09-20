"""`output_budget` must never ask for more than the server can accept.

Reference failures — the same defect at two different context sizes, both "over by exactly one":

    xlam 39311800 (served 8192)
        requested 4736 output tokens, prompt at least 3457 input, total at least 8193
    xlam 39321471 (served 16384, after raising the context to "fix" it)
        requested 13515 output tokens, prompt at least 2870 input, total at least 16385

Both are HTTP 400s, not truncation: the row is never generated and never verified, so the batch
reports "kept 0" and reads like the verifier rejecting everything. Two of five batches were lost
that way on 39311800 (0 of 25 and 4 of 126 kept).

The shortfall was 65 tokens in BOTH cases, which is the point: it is proportional to prompt length,
not constant, so doubling the served context reproduced it exactly and a fixed margin can never
absorb it. These tests pin the conservative prompt estimate rather than the margin.
"""
from __future__ import annotations

import pytest

from config.token_budget import (
    _CONTEXT_MARGIN_TOKENS,
    _FLOOR_TOKENS,
    _PROMPT_CHARS_PER_TOKEN,
    CHARS_PER_TOKEN,
    output_budget,
    served_context,
)


# Worst-case density actually observed on xlam's tool-schema JSON. The estimate must stay at or
# below this, or it will under-count a real prompt again.
_OBSERVED_WORST_CASE_CHARS_PER_TOKEN = 3.9


class TestPromptEstimateIsConservative:
    def test_prompt_divisor_is_smaller_than_the_output_divisor(self):
        """Smaller divisor -> larger estimate -> smaller budget, which is the safe direction."""
        assert _PROMPT_CHARS_PER_TOKEN < CHARS_PER_TOKEN

    def test_prompt_divisor_covers_the_densest_json_measured(self):
        assert _PROMPT_CHARS_PER_TOKEN <= _OBSERVED_WORST_CASE_CHARS_PER_TOKEN

    def test_estimate_is_not_below_a_real_token_count(self):
        """A prompt that really tokenizes at 3.9 chars/token must not be estimated as smaller."""
        real_tokens = 3457
        prompt = "x" * int(real_tokens * _OBSERVED_WORST_CASE_CHARS_PER_TOKEN)
        assert int(len(prompt) / _PROMPT_CHARS_PER_TOKEN) >= real_tokens


class TestBudgetNeverExceedsServedContext:
    @pytest.mark.parametrize("served", [4096, 8192, 16384, 32768])
    @pytest.mark.parametrize("real_prompt_tokens", [512, 2870, 3457, 6000])
    def test_prompt_plus_budget_fits(self, monkeypatch, served, real_prompt_tokens):
        monkeypatch.setenv("SLM_SYNTH_MAX_MODEL_LEN", str(served))
        assert served_context() == served
        prompt = "x" * int(real_prompt_tokens * _OBSERVED_WORST_CASE_CHARS_PER_TOKEN)
        if real_prompt_tokens + _FLOOR_TOKENS + _CONTEXT_MARGIN_TOKENS > served:
            pytest.skip("prompt cannot fit this context at all; the floor is a loud 400 by design")
        budget = output_budget(prompt, needed_chars=40000)
        assert real_prompt_tokens + budget <= served, (
            f"would 400: {real_prompt_tokens} prompt + {budget} output > {served}"
        )

    def test_the_two_recorded_400s_no_longer_reproduce(self, monkeypatch):
        for served, real_prompt_tokens in ((8192, 3457), (16384, 2870)):
            monkeypatch.setenv("SLM_SYNTH_MAX_MODEL_LEN", str(served))
            prompt = "x" * int(real_prompt_tokens * _OBSERVED_WORST_CASE_CHARS_PER_TOKEN)
            assert real_prompt_tokens + output_budget(prompt, needed_chars=40000) <= served


class TestBudgetStaysUseful:
    def test_a_short_prompt_still_gets_a_large_budget(self, monkeypatch):
        """Being conservative must not collapse the budget on ordinary prompts."""
        monkeypatch.setenv("SLM_SYNTH_MAX_MODEL_LEN", "16384")
        assert output_budget("x" * 4000, needed_chars=40000) > 8000

    def test_budget_never_returns_below_the_floor(self, monkeypatch):
        """A budget under the floor should surface as a loud 400, not a silent truncation."""
        monkeypatch.setenv("SLM_SYNTH_MAX_MODEL_LEN", "8192")
        assert output_budget("x" * 100000, needed_chars=40000) == _FLOOR_TOKENS


class TestTheServedContextMatchesTheModeInUse:
    """`config.config` has TWO defaults for the teacher's context — 8192 local, 131072 API — and
    `served_context` used to hard-code the local one.

    That was invisible while every launcher exported `SLM_SYNTH_MAX_MODEL_LEN`. It became a real
    bug the moment the xlam launcher stopped exporting it in API mode (deliberately, so its local
    16384 cap could not throttle DeepSeek): the budget was then computed against an 8,192 window
    the teacher did not have. On run 39361648 that allowed ~4,600 output tokens where 16,384 were
    available, and a reasoning teacher spends most of a budget thinking before it emits content —
    so replies were truncated a few hundred characters in, 19 of one 350-attempt batch.
    """

    def test_api_mode_defaults_to_the_hosted_context(self, monkeypatch):
        monkeypatch.delenv("SLM_SYNTH_MAX_MODEL_LEN", raising=False)
        monkeypatch.setenv("SLM_SYNTH_API_MODE", "1")
        import importlib

        import config.token_budget as tb

        importlib.reload(tb)
        assert tb.served_context() == tb._API_DEFAULT_CONTEXT

    def test_local_mode_is_unchanged(self, monkeypatch):
        """Every local measurement on record was taken at 8192; moving it would move them all."""
        monkeypatch.delenv("SLM_SYNTH_MAX_MODEL_LEN", raising=False)
        monkeypatch.setenv("SLM_SYNTH_API_MODE", "0")
        import importlib

        import config.token_budget as tb

        importlib.reload(tb)
        assert tb.served_context() == 8192

    def test_an_explicit_value_still_wins_in_api_mode(self, monkeypatch):
        """The local profiles trade context against KV cache with this variable; it must remain an
        override rather than a suggestion."""
        monkeypatch.setenv("SLM_SYNTH_API_MODE", "1")
        monkeypatch.setenv("SLM_SYNTH_MAX_MODEL_LEN", "16384")
        import importlib

        import config.token_budget as tb

        importlib.reload(tb)
        assert tb.served_context() == 16384

    def test_api_mode_gives_a_reasoning_teacher_real_room(self, monkeypatch):
        """The symptom was a ~4,600-token budget on a 3,500-token prompt. It must now clear the
        global output ceiling instead."""
        monkeypatch.delenv("SLM_SYNTH_MAX_MODEL_LEN", raising=False)
        monkeypatch.setenv("SLM_SYNTH_API_MODE", "1")
        import importlib

        import config.token_budget as tb

        importlib.reload(tb)
        assert tb.output_budget("x" * 14_000, needed_chars=4_000) == tb.MAX_OUTPUT_TOKENS
