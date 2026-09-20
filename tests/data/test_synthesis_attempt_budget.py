"""Synthesis asks for what it needs and tops up once, instead of over-generating 2x.

WHAT IT USED TO DO
    `max_attempts = max(1, n) * 4` with `planned = max(n * 2, n + 32)`, then trim to n. A 175-row
    batch on run 39361648 therefore fired 350 generations: 19 failed and 156 perfectly good rows
    were discarded because the quota was already met. On a paid endpoint that is double the bill as
    insurance against a failure rate that measured 5%.

THE CONTRACT NOW
    Round 1 asks for exactly n. Round 2 asks for exactly the shortfall — the number that actually
    failed, not a guess at how many might. There is no round 3: a systematically broken batch would
    otherwise be paid for over and over, and `run_health` already stops the run after
    MAX_CONSECUTIVE_EMPTY_SYNTHESIS empty ones.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("EXA_API_KEY", "x")

from data.curriculum import _synthesize_new_correct

_ANCHOR = {
    "text": "q",
    "answer": '[{"name": "f", "arguments": {"a": 1}}]',
    "tools": [{"name": "f", "parameters": {"type": "dict",
                                           "properties": {"a": {"type": "int"}},
                                           "required": ["a"]}}],
}
_GOOD = json.dumps({"text": "new q", "answer": '[{"name": "f", "arguments": {"a": 2}}]'})


class _Teacher:
    """Fails the first `n_failures` calls of every round, then succeeds."""

    def __init__(self, fail_first: int = 0):
        self.calls = 0
        self._fail_first = fail_first

    def __call__(self, prompt, temperature=0.7, max_tokens=None) -> str:
        self.calls += 1
        return "TRUNCATED{" if self.calls <= self._fail_first else _GOOD


def _run(n: int, teacher: _Teacher) -> list[dict]:
    return _synthesize_new_correct(
        [_ANCHOR] * (n * 3), task_description="t", n=n,
        generate_fn=teacher, verify_fn=None,
    )


class TestItAsksForWhatItNeeds:
    def test_a_clean_batch_costs_exactly_n_calls(self):
        teacher = _Teacher()
        rows = _run(20, teacher)
        assert len(rows) == 20
        assert teacher.calls == 20, "a batch with no failures must not pay for a second round"

    def test_a_partial_failure_pays_only_for_the_shortfall(self):
        teacher = _Teacher(fail_first=5)
        rows = _run(20, teacher)
        assert len(rows) == 20
        assert teacher.calls == 25, "should be n + shortfall, not 2n"


class TestItStopsAfterOneRetry:
    def test_a_totally_broken_batch_is_not_retried_forever(self):
        """Worst case is bounded at 2n — the same as the old unconditional cost, not more."""
        teacher = _Teacher(fail_first=10**6)
        rows = _run(20, teacher)
        assert rows == []
        assert teacher.calls == 40

    def test_a_short_batch_is_delivered_short_rather_than_padded(self):
        """Round 2 gets exactly the shortfall; whatever survives is what the caller receives."""
        teacher = _Teacher(fail_first=30)
        rows = _run(20, teacher)
        assert 0 < len(rows) < 20


class TestNoGoodRowIsWasted:
    def test_it_never_generates_more_than_it_can_use(self):
        """The old design discarded 156 usable rows from one 175-row batch."""
        teacher = _Teacher()
        rows = _run(50, teacher)
        assert teacher.calls == len(rows) == 50
