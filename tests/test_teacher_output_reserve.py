"""A reasoning teacher must not be measured under the student's answer budget.

`TaskSpec.max_new_tokens` sizes the STUDENT's answer — 256 on xlam_bfcl, enough for a JSON array
from a small non-reasoning model. A reasoning teacher spends completion tokens on its own reasoning
first, against the same limit, so the same number truncates it mid-answer.

Run 39361189 measured `deepseek-v4-flash` at format_valid=0.7630 with the pipeline's own warning
that the score was "bounded by a formatting failure rather than by competence". Probed on 24 real
xlam eval prompts: 3 of 24 hit finish_reason=length at 256 against 1 of 24 at 2048.

That number gates synthetic data AND sets the run's accuracy goal, so truncating the teacher lowers
the bar the student is then held to.
"""
from __future__ import annotations

import importlib
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("EXA_API_KEY", "x")

import pytest

from tasks import get_task


def _reserve(spec_name: str, monkeypatch, **env) -> int:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import agent.teacher_fitness as tf

    importlib.reload(tf)
    return tf._teacher_output_reserve(get_task(spec_name))


class TestLocalModeIsUnchanged:
    def test_the_local_teacher_keeps_the_task_reserve(self, monkeypatch):
        """The local Qwen is rendered with thinking disabled, so it never spent tokens reasoning and
        was never truncated. Changing its budget would move every teacher measurement on record."""
        assert _reserve("xlam_bfcl", monkeypatch, SLM_SYNTH_API_MODE="0") == 256

    def test_an_unset_api_mode_is_treated_as_local(self, monkeypatch):
        monkeypatch.delenv("SLM_SYNTH_API_MODE", raising=False)
        import agent.teacher_fitness as tf

        importlib.reload(tf)
        assert tf._teacher_output_reserve(get_task("xlam_bfcl")) == 256


class TestApiModeGetsRoomToReason:
    def test_the_floor_applies(self, monkeypatch):
        assert _reserve("xlam_bfcl", monkeypatch, SLM_SYNTH_API_MODE="1") == 2048

    def test_it_is_configurable(self, monkeypatch):
        assert _reserve(
            "xlam_bfcl", monkeypatch,
            SLM_SYNTH_API_MODE="1", SLM_TEACHER_API_OUTPUT_RESERVE="4096",
        ) == 4096

    def test_the_floor_only_ever_raises(self, monkeypatch):
        """A task that already asks for more than the floor keeps its own, larger number."""
        monkeypatch.setenv("SLM_TEACHER_API_OUTPUT_RESERVE", "128")
        assert _reserve("xlam_bfcl", monkeypatch, SLM_SYNTH_API_MODE="1") == 256

    @pytest.mark.parametrize("task", ["xlam_bfcl", "ner_bc5cdr", "calendar_json", "clinc150"])
    def test_no_task_is_measured_below_its_own_reserve(self, task, monkeypatch):
        own = get_task(task).max_new_tokens
        assert _reserve(task, monkeypatch, SLM_SYNTH_API_MODE="1") >= own


class TestItDoesNotImportConfigAtCallTime:
    def test_a_missing_credential_cannot_disable_the_floor(self, monkeypatch):
        """`config.config` raises on an unset API key. Importing it here is the trap that once made
        the token-budget context clamp silently inactive."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("SLM_SYNTH_API_MODE", "1")
        import agent.teacher_fitness as tf

        importlib.reload(tf)
        assert tf._teacher_output_reserve(get_task("xlam_bfcl")) == 2048
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
