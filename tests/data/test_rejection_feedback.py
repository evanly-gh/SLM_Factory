"""Quoting the generator's own recent rejections back at it.

WHY THIS FILE EXISTS
    On run 38708719 the verifier's rejections clustered hard on ONE systematic error — "includes
    unrequested optional parameter", "includes optional parameters with default values not implied
    by query" — which accounted for roughly half of them. The generation brief ALREADY told the
    teacher not to leave schema defaults in place, and it kept doing it anyway.

    A static instruction the generator demonstrably ignores is worth less than its own recent
    mistakes quoted back at it. That is the same reason the demonstrations are shown rather than
    described (B276): a contract the model has only been told about is a contract it is being
    tested on guessing.

    So `record_rejection_reasons` remembers what the verifier refused and
    `_rejection_feedback_block` puts it at the top of the next generation prompt. Three things have
    to be true of that loop, and each is a section below: the reasons have to REACH the prompt, the
    list has to stay short and non-repetitive, and it must carry only reasons that describe a
    defect in a ROW.

MODULE-LEVEL STATE
    `_RECENT_REJECTIONS` lives at module scope, because generation and verification are separate
    calls in separate passes with no object between them to hang it on. That makes it shared
    mutable state across the whole test session: any other test that runs a verifier with
    rejections leaves entries behind, and pytest orderings are not fixed. The autouse fixture below
    clears it on BOTH sides of every test so neither this file's results nor anyone else's depend
    on collection order.

    The module is also re-imported inside each test rather than bound at file scope, because
    `tests/config/test_curation_config.py` calls `importlib.reload` on `data.curriculum` — after
    which a name captured at import time refers to a dead copy of the module, and
    `_RECENT_REJECTIONS` is a different list object from the one production code appends to.
"""
from __future__ import annotations

import importlib
import json

import pytest


def _curriculum():
    """The LIVE `data.curriculum` module.

    Resolved per call: `importlib.reload` elsewhere in the suite replaces the module object, and a
    reference captured at import time would point at the old one — including at a stale
    `_RECENT_REJECTIONS` list that production code no longer writes to.
    """
    return importlib.import_module("data.curriculum")


@pytest.fixture(autouse=True)
def _clean_rejection_history():
    """Empty the shared rejection history around every test in this file.

    Cleared before AND after: before, so a verifier run in another test file cannot seed this one;
    after, so these fixtures' reasons cannot leak into a test that asserts a generation prompt is
    clean. Randomised collection order makes both directions reachable.
    """
    _curriculum()._RECENT_REJECTIONS.clear()
    yield
    _curriculum()._RECENT_REJECTIONS.clear()


ANCHOR = {
    "text": "what is the distance between Seattle and Portland",
    "answer": '[{"name": "calculate_distance", "arguments": {"origin": "Seattle"}}]',
    "tools": [{"name": "calculate_distance"}],
}
DEFAULT_DESCRIPTION = "turn a request into a JSON function call using only the declared tools"
REAL_REASON = "includes unrequested optional parameter"


def _verdict(valid: bool, reason: str = "because") -> str:
    return json.dumps({"valid": valid, "reason": reason})


# --------------------------------------------------------------------------
# 1. The reasons reach the generation prompt
# --------------------------------------------------------------------------


def test_a_recorded_reason_appears_in_the_next_generation_prompt():
    curriculum = _curriculum()
    curriculum.record_rejection_reasons([REAL_REASON])

    prompt = curriculum._new_example_prompt(ANCHOR, DEFAULT_DESCRIPTION)
    assert REAL_REASON in prompt
    # Framed as the generator's OWN mistakes on this task, not as generic advice: the whole reason
    # this exists is that generic advice about the same defect was already in the brief.
    assert "do not repeat them" in prompt


def test_the_reasons_reach_the_prompt_the_synthesis_path_actually_sends():
    """`_new_example_prompt` carrying the block is necessary but not sufficient — the feedback is
    worthless if the generator loop builds its prompt some other way."""
    curriculum = _curriculum()
    curriculum.record_rejection_reasons([REAL_REASON])

    prompts: list[str] = []

    def generate(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        return json.dumps({"text": "a new request", "answer": "[]"})

    curriculum._synthesize_new_correct(
        [ANCHOR], task_description=DEFAULT_DESCRIPTION, n=1, generate_fn=generate,
    )

    assert prompts
    assert all(REAL_REASON in prompt for prompt in prompts)


def test_an_empty_history_renders_nothing_and_adds_nothing_to_the_prompt():
    """The first batch of a run has no previous round to learn from. An empty heading would read as
    a section the generator should have been given and was not."""
    curriculum = _curriculum()
    assert curriculum._rejection_feedback_block() == ""

    clean = curriculum._new_example_prompt(ANCHOR, DEFAULT_DESCRIPTION)
    assert "do not repeat them" not in clean

    curriculum.record_rejection_reasons([REAL_REASON])
    fed = curriculum._new_example_prompt(ANCHOR, DEFAULT_DESCRIPTION)
    # The block is the ONLY difference, so nothing else in the prompt depends on the history.
    assert fed.replace(curriculum._rejection_feedback_block(), "") == clean


# --------------------------------------------------------------------------
# 2. Short and non-repetitive
# --------------------------------------------------------------------------


def test_repeated_reasons_are_recorded_once():
    """The rejections cluster: half of run 38708719's were the same complaint. Listing the same
    sentence six times spends the whole budget on one mistake and hides the others, while saying
    nothing more than one copy of it does."""
    curriculum = _curriculum()
    curriculum.record_rejection_reasons([REAL_REASON, REAL_REASON, "invented tool name"])
    assert curriculum._RECENT_REJECTIONS == [REAL_REASON, "invented tool name"]


def test_reasons_differing_only_in_whitespace_are_the_same_reason():
    """The reasons are free text from a model, so the same complaint arrives with a stray newline
    or a doubled space. Treating those as distinct would defeat the dedup on real input."""
    curriculum = _curriculum()
    curriculum.record_rejection_reasons([
        "includes  unrequested\noptional parameter", REAL_REASON,
    ])
    assert curriculum._RECENT_REJECTIONS == [REAL_REASON]


def test_the_list_is_capped_and_keeps_the_first_distinct_reasons():
    """A prompt is a budget. Every rejection quoted is context not spent on the demonstrations,
    which are the measured 6.4x intervention (B276), so the feedback has to stay a summary."""
    curriculum = _curriculum()
    cap = curriculum.MAX_FEDBACK_REJECTIONS
    reasons = [f"defect {i}" for i in range(cap + 4)]
    curriculum.record_rejection_reasons(reasons)

    assert curriculum._RECENT_REJECTIONS == reasons[:cap]
    block = curriculum._rejection_feedback_block()
    assert block.count("\n  - ") == cap
    assert reasons[cap] not in block


def test_a_later_round_replaces_the_previous_rounds_reasons():
    """"Your last batch" has to mean the last batch. Accumulating across rounds would keep quoting
    a defect the generator has already stopped making, which is indistinguishable from telling it
    to keep making it."""
    curriculum = _curriculum()
    curriculum.record_rejection_reasons(["stale defect"])
    curriculum.record_rejection_reasons([REAL_REASON])

    assert curriculum._RECENT_REJECTIONS == [REAL_REASON]
    assert "stale defect" not in curriculum._new_example_prompt(ANCHOR, DEFAULT_DESCRIPTION)


# --------------------------------------------------------------------------
# 3. Only defects in a ROW
# --------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [
    "verifier unavailable (kept)",
    "unparseable verifier reply (kept)",
])
def test_verifier_failures_are_not_reported_as_row_defects(noise):
    """Both of those are the FAIL-OPEN path: the endpoint refused the connection, or the reply
    could not be read. The row was KEPT in each case, so there is no defect to avoid — and quoting
    them tells the generator to stop doing something it never did."""
    curriculum = _curriculum()
    curriculum.record_rejection_reasons([noise, REAL_REASON])

    assert curriculum._RECENT_REJECTIONS == [REAL_REASON]
    assert noise not in curriculum._new_example_prompt(ANCHOR, DEFAULT_DESCRIPTION)


def test_blank_reasons_are_dropped():
    """A verdict of `{"valid": false}` with no reason carries no instruction, and an empty bullet
    in the list reads as a defect the generator failed to understand."""
    curriculum = _curriculum()
    curriculum.record_rejection_reasons(["", "   ", None, REAL_REASON])
    assert curriculum._RECENT_REJECTIONS == [REAL_REASON]


# --------------------------------------------------------------------------
# 4. End to end: a real verification pass feeds the next generation
# --------------------------------------------------------------------------


def test_a_real_verification_pass_populates_the_feedback_for_the_next_generation():
    """The wiring, not the pieces. `verify_generated_answers` is what learns the reasons, and it is
    a separate pass from generation with nothing but module state between them — so a change that
    kept both halves working while disconnecting them would leave every other test in this file
    green."""
    curriculum = _curriculum()
    rows = [{"text": f"request {i}", "answer": "[]"} for i in range(3)]

    kept = curriculum.verify_generated_answers(
        rows,
        task_description=DEFAULT_DESCRIPTION,
        generate_fn=lambda *_args, **_kwargs: _verdict(False, REAL_REASON),
    )

    assert kept == []
    assert curriculum._RECENT_REJECTIONS == [REAL_REASON]
    assert REAL_REASON in curriculum._new_example_prompt(ANCHOR, DEFAULT_DESCRIPTION)


def test_a_verification_pass_that_rejected_nothing_clears_the_feedback():
    """The generator fixed it. Continuing to quote the reason would be the same failure as never
    quoting one — the block stops describing the last batch and starts being decoration."""
    curriculum = _curriculum()
    curriculum.record_rejection_reasons([REAL_REASON])

    rows = [{"text": "request", "answer": "[]"}]
    kept = curriculum.verify_generated_answers(
        rows,
        task_description=DEFAULT_DESCRIPTION,
        generate_fn=lambda *_args, **_kwargs: _verdict(True),
    )

    assert kept == rows
    assert curriculum._RECENT_REJECTIONS == []
    assert curriculum._rejection_feedback_block() == ""


def test_an_unreachable_verifier_leaves_no_feedback_behind():
    """Every row is kept by the fail-open contract, so nothing was judged defective. The previous
    round's reasons must not survive as if they had been re-confirmed either."""
    curriculum = _curriculum()
    curriculum.record_rejection_reasons([REAL_REASON])

    def explode(*_args, **_kwargs):
        raise RuntimeError("endpoint refused the connection")

    rows = [{"text": "request", "answer": "[]"}]
    assert curriculum.verify_generated_answers(
        rows, task_description=DEFAULT_DESCRIPTION, generate_fn=explode,
    ) == rows
    assert curriculum._RECENT_REJECTIONS == []
