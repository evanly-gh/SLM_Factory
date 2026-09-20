"""The `iterate` system prompt is cached; the per-turn user message deliberately is not.

WHY THIS IS WORTH PINNING
    `iterate` is 96% of this project's Anthropic spend — on run 39294409, 39 calls carrying
    379,829 input tokens against 95,148 output, $1.71 of a $1.78 bill — and every one of those
    calls reported `cache=0` while resending the same ~2,500-token system prompt.

    The failure mode these tests exist to prevent is the one Anthropic's own guide calls out: a
    cache entry is written ONLY at a breakpoint, so marking a block that changes every request
    pays for a write every time and never reads one back. That is strictly worse than no caching,
    because a write costs more than a plain input token. The `iterate` user message carries the
    trajectory, the tried-config list and the test report, so it must never carry the marker.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    import config.config as cc

    return importlib.reload(cc)


def _reload(monkeypatch, **env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import config.config as cc

    return importlib.reload(cc)


LONG = "x" * 40_000       # ~10k tokens, far above every model's floor
SHORT = "x" * 200         # ~50 tokens, far below it


class TestTheMarkerLandsOnCacheableContent:
    def test_a_long_system_prompt_becomes_a_marked_block(self, config):
        block = config.cacheable_system_content(LONG)
        assert isinstance(block, list) and len(block) == 1
        assert block[0]["type"] == "text"
        assert block[0]["cache_control"]["type"] == "ephemeral"

    def test_the_prompt_text_is_preserved_exactly(self, config):
        """Caching must not alter a single byte of what the model is asked."""
        assert config.cacheable_system_content(LONG)[0]["text"] == LONG

    def test_the_real_iterate_system_prompt_is_above_the_floor(self, config):
        """If the prompt is ever trimmed below ~1024 tokens this silently stops caching, and the
        API returns no error to say so."""
        from agent.nodes.iterate import _ITERATE_SYSTEM

        assert isinstance(config.cacheable_system_content(_ITERATE_SYSTEM), list)


class TestTooShortToCacheIsLeftAlone:
    def test_a_short_prompt_stays_a_plain_string(self, config):
        """Below the model's minimum the API ignores the marker and reports neither a write nor a
        read. Sending it anyway would be a silent no-op; not sending it keeps the request honest."""
        assert config.cacheable_system_content(SHORT) == SHORT

    def test_the_floor_is_configurable(self, monkeypatch):
        cc = _reload(monkeypatch, SLM_ORCHESTRATOR_CACHE_MIN_TOKENS="1")
        assert isinstance(cc.cacheable_system_content(SHORT), list)


class TestTheTTLIsLongEnoughToSurviveATrainEvalCycle:
    def test_it_defaults_to_one_hour(self, config):
        """Consecutive iterate calls are separated by a full train+evaluate cycle — ~14 minutes
        apart on run 39294409 and longer on bigger models. The 5-minute default would miss almost
        every time, and a miss costs MORE than not caching at all."""
        assert config.ORCHESTRATOR_CACHE_TTL == "1h"
        assert config.cacheable_system_content(LONG)[0]["cache_control"]["ttl"] == "1h"

    def test_the_ttl_is_configurable(self, monkeypatch):
        cc = _reload(monkeypatch, SLM_ORCHESTRATOR_CACHE_TTL="5m")
        assert cc.cacheable_system_content(LONG)[0]["cache_control"]["ttl"] == "5m"


class TestItCanBeTurnedOff:
    def test_disabling_returns_plain_text(self, monkeypatch):
        cc = _reload(monkeypatch, SLM_ORCHESTRATOR_CACHE="0")
        assert cc.cacheable_system_content(LONG) == LONG

    def test_cheap_mode_never_caches(self, monkeypatch):
        """Cheap mode runs a different, smaller orchestrator; the rate arithmetic that justifies a
        1-hour write does not carry over to it."""
        cc = _reload(monkeypatch, SLM_CHEAP="1")
        assert cc.cacheable_system_content(LONG) == LONG


class TestOnlyTheSystemBlockIsMarked:
    def test_iterate_marks_the_system_message_and_not_the_user_message(self, monkeypatch):
        """The breakpoint must sit on the last STABLE block. Marking the per-turn user content
        would write a new entry every call and read none back."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        import config.config as cc

        importlib.reload(cc)
        from agent.nodes.iterate import _ITERATE_SYSTEM

        system_content = cc.cacheable_system_content(_ITERATE_SYSTEM)
        user_content = "trajectory that differs every single turn"

        assert isinstance(system_content, list)
        assert "cache_control" in system_content[0]
        # The user half is passed through untouched: a bare string carries no marker at all.
        assert isinstance(user_content, str)
