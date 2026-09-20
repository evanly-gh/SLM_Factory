"""The synthesis prompt must put everything STABLE before anything that changes.

WHY THIS IS A TEST AND NOT A STYLE PREFERENCE
    Both the local vLLM server and the DeepSeek API cache a request's PREFIX, and a cached entry is
    only reusable by a later request that matches it from the first token. So one per-batch string
    placed early invalidates everything after it.

    The rejection-feedback block used to sit ahead of the demonstrations. Measured on a real
    calendar prompt, that left a cross-batch shared prefix of 81 characters (~20 tokens) against
    7,277 (~1,819) with the demonstrations first — and the minimum cacheable prefix is ~1,024
    tokens, so the old order was not merely worse, it was below the threshold at which any caching
    happens at all.
"""
from __future__ import annotations

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("EXA_API_KEY", "x")

import data.curriculum as cur


DESC = "test task: turn a request into a call"


def _anchor(n: int) -> dict:
    return {"text": f"request {n}", "answer": f'[{{"name": "f", "arguments": {{"i": {n}}}}}]'}


def _demos() -> list[dict]:
    return [_anchor(i) for i in range(3)]


def _shared_prefix(a: str, b: str) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


class TestStableContentComesFirst:
    def test_demonstrations_precede_the_rejection_feedback(self):
        """Demonstrations are identical for every row in a batch; rejections change per batch."""
        cur._RECENT_REJECTIONS.clear()
        cur.record_rejection_reasons(["a distinctive rejection reason"])
        prompt = cur._new_example_prompt(_anchor(9), DESC, demos=_demos())

        assert "real examples of this task" in prompt
        assert "a distinctive rejection reason" in prompt
        assert prompt.index("real examples of this task") < prompt.index(
            "a distinctive rejection reason"
        )

    def test_the_task_description_is_still_first(self):
        cur._RECENT_REJECTIONS.clear()
        assert cur._new_example_prompt(_anchor(1), DESC, demos=_demos()).startswith(DESC)

    def test_the_per_row_schema_comes_last(self):
        """The anchor changes every single call, so nothing cacheable may follow it."""
        cur._RECENT_REJECTIONS.clear()
        cur.record_rejection_reasons(["some reason"])
        prompt = cur._new_example_prompt(_anchor(7), DESC, demos=_demos())
        assert prompt.index("some reason") < prompt.index("EXACTLY this JSON schema")


class TestTheCacheableSpanSurvivesABatchBoundary:
    def test_a_new_rejection_set_does_not_invalidate_the_demonstrations(self):
        """The regression this file exists to prevent: a changed rejection block truncating the
        shared prefix back to the task description."""
        cur._RECENT_REJECTIONS.clear()
        first = cur._new_example_prompt(_anchor(1), DESC, demos=_demos())
        cur.record_rejection_reasons(["batch two learned something new"])
        second = cur._new_example_prompt(_anchor(2), DESC, demos=_demos())

        shared = _shared_prefix(first, second)
        # The demonstrations block must be inside the shared span, not cut off before it.
        demo_end = first.index("real examples of this task") + len("real examples of this task")
        assert shared > demo_end, (
            f"shared prefix {shared} ended before the demonstrations at {demo_end} — a per-batch "
            "string has moved ahead of them again"
        )

    def test_rows_within_one_batch_share_everything_up_to_the_anchor(self):
        cur._RECENT_REJECTIONS.clear()
        cur.record_rejection_reasons(["stable within this batch"])
        a = cur._new_example_prompt(_anchor(1), DESC, demos=_demos())
        b = cur._new_example_prompt(_anchor(2), DESC, demos=_demos())
        assert _shared_prefix(a, b) > a.index("EXACTLY this JSON schema")
