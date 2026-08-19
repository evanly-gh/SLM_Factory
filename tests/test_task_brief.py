"""The orchestrator writes the task description every teacher prompt is built from.

WHY THIS FILE EXISTS
    Every teacher prompt — generate a row, judge whether a generated row is correct — has to tell the
    teacher what the task IS. That text used to come from a table keyed by task TYPE, which produced
    two failures:

      * It could not distinguish two tasks sharing a type. `xlam_bfcl` and `calendar_json` were both
        described as "converting a request into a JSON function call using only the declared tools",
        which omits every convention that makes a CALENDAR row correct — the 60-minute default, ISO
        8601, resolving relative dates against the request's own reference instant. A verifier asked
        to judge against that description was judging its own guess, and calendar synthesis scored
        0.2176 (B269).
      * It was one line. A teacher asked "is this answer correct?" with one line of context and no
        worked example is being tested on inferring the contract.

    So the orchestrator writes it once, at cold start, having been shown REAL rows from the dataset
    that was actually loaded. The properties worth pinning are: it is built from real rows, it is
    bounded (these prompts are issued once per generated row, thousands of times per rebuild), it
    reaches the synthesis prompts, and an unreachable orchestrator degrades to something HONEST
    rather than to a plausible-looking guess.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.task_brief import (
    CONTRACT_MAX_CHARS,
    MAX_FAILURE_MODES,
    SUMMARY_MAX_CHARS,
    brief_context_block,
    build_task_brief,
    fallback_brief,
    log_task_brief,
)
from tasks import TASKS, get_task

CALENDAR_ROWS = [
    {"text": ("Convert the request into a calendar.events.insert call.\n"
              "Current date and time: 2026-03-01T09:00:00 (Sunday).\n\n"
              'Add "Dentist" on March 3rd at 10:00 am'),
     "answer": json.dumps([{"name": "calendar.events.insert",
                            "arguments": {"summary": "Dentist"}}]),
     "tools": [{"name": "calendar.events.insert"}],
     "_provenance": "train_anchor",
     "_source": "hf:TOPv2/reminder"},
]

GOOD_REPLY = json.dumps({
    "summary": "Turn a scheduling request into one calendar.events.insert call.",
    "output_contract": ("A JSON array with one call. Datetimes are ISO 8601. An unstated duration "
                        "is 60 minutes. Relative dates resolve against the reference instant in "
                        "the prompt."),
    "failure_modes": ["year resolution error", "imperative copied into the summary"],
})


def _stub_anthropic(monkeypatch, reply=GOOD_REPLY, error=None):
    """Replace the `anthropic` client and the cost-tracked call. No network and no real key.

    `build_task_brief` imports both lazily, so the patch has to land on the modules they come from
    rather than on `agent.task_brief`.
    """
    import agent.cost as cost

    captured: dict = {}

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            if error is not None:
                raise error
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=reply)])

    monkeypatch.setitem(
        __import__("sys").modules, "anthropic",
        SimpleNamespace(Anthropic=lambda **_kwargs: SimpleNamespace(messages=_Messages())),
    )
    def _create(client_messages, **kwargs):
        kwargs.pop("stage", None)
        return client_messages.create(**kwargs)

    monkeypatch.setattr(cost, "tracked_anthropic_messages_create", _create)
    return captured


# --------------------------------------------------------------------------
# The brief is built from real rows
# --------------------------------------------------------------------------


def test_the_brief_prompt_shows_the_rows_the_loader_actually_returned(monkeypatch):
    """A wrong worked example is worse than none, so the examples are always REAL rows from the
    task's own training split rather than anything the orchestrator invented."""
    captured = _stub_anthropic(monkeypatch)
    build_task_brief(get_task("calendar_json"), CALENDAR_ROWS, log=lambda _m: None)

    prompt = captured["messages"][0]["content"]
    assert "calendar.events.insert" in prompt
    assert "Current date and time: 2026-03-01T09:00:00" in prompt
    assert "Dentist" in prompt


def test_private_row_fields_are_not_shown_to_the_teacher(monkeypatch):
    """Provenance bookkeeping is not part of the task, and showing it invites the teacher to
    reproduce it in generated rows."""
    captured = _stub_anthropic(monkeypatch)
    build_task_brief(get_task("calendar_json"), CALENDAR_ROWS, log=lambda _m: None)

    prompt = captured["messages"][0]["content"]
    assert "_provenance" not in prompt
    assert "train_anchor" not in prompt


def test_the_prompt_tells_the_orchestrator_to_write_the_contract_from_the_rows(monkeypatch):
    """The instruction that separates this from the one-line table it replaced: a convention the
    benchmark's NAME would not tell you has to come from the data."""
    captured = _stub_anthropic(monkeypatch)
    build_task_brief(get_task("calendar_json"), CALENDAR_ROWS, log=lambda _m: None)

    prompt = captured["messages"][0]["content"]
    assert "from the ROWS, not from the benchmark's reputation" in prompt


def test_the_prompt_states_how_this_task_is_actually_graded(monkeypatch):
    """The teacher is generating rows that will be scored, so it needs the grading rule — and it
    differs per task, which is the distinction the shared table could not express."""
    captured = _stub_anthropic(monkeypatch)
    build_task_brief(get_task("routerbench"), [{"text": "q", "label": "local"}],
                     log=lambda _m: None)

    prompt = captured["messages"][0]["content"]
    assert "minority_f1" in prompt
    assert "fixed set of class labels" in prompt


def test_a_task_with_an_exact_verifier_says_so_in_its_grading_line(monkeypatch):
    captured = _stub_anthropic(monkeypatch)
    build_task_brief(get_task("xlam_bfcl"), [{"text": "q", "answer": "[]"}], log=lambda _m: None)
    assert "exact programmatic well-formedness check" in captured["messages"][0]["content"]


# --------------------------------------------------------------------------
# The brief is bounded
# --------------------------------------------------------------------------


def test_a_verbose_brief_is_clipped(monkeypatch):
    """These prompts are issued once per generated row, thousands of times per rebuild, so an
    unbounded brief is an unbounded bill."""
    _stub_anthropic(monkeypatch, reply=json.dumps({
        "summary": "s" * 5000,
        "output_contract": "c" * 5000,
        "failure_modes": [f"mode {i}" for i in range(40)],
    }))
    brief = build_task_brief(get_task("gsm8k"), [{"text": "q", "answer": "#### 1"}],
                            log=lambda _m: None)

    assert len(brief["summary"]) == SUMMARY_MAX_CHARS
    assert len(brief["output_contract"]) == CONTRACT_MAX_CHARS
    assert len(brief["failure_modes"]) == MAX_FAILURE_MODES


def test_whitespace_is_collapsed_so_the_brief_is_one_block_of_prose(monkeypatch):
    _stub_anthropic(monkeypatch, reply=json.dumps({
        "summary": "a\n\n  b\tc",
        "output_contract": "d   e",
        "failure_modes": ["  f  ", "", "   "],
    }))
    brief = build_task_brief(get_task("gsm8k"), [{"text": "q", "answer": "#### 1"}],
                            log=lambda _m: None)
    assert brief["summary"] == "a b c"
    assert brief["output_contract"] == "d e"
    assert brief["failure_modes"] == ["f"]


def test_a_good_brief_is_recorded_as_the_orchestrator_own(monkeypatch):
    _stub_anthropic(monkeypatch)
    brief = build_task_brief(get_task("calendar_json"), CALENDAR_ROWS, log=lambda _m: None)

    assert brief["source"] == "orchestrator"
    assert "60 minutes" in brief["output_contract"]
    assert brief["failure_modes"]


def test_the_brief_is_logged_in_full(monkeypatch):
    """It is the text every teacher prompt is built from, so a run whose synthesis behaved oddly
    needs it in the log rather than only in the artifacts."""
    _stub_anthropic(monkeypatch)
    logs: list[str] = []
    build_task_brief(get_task("calendar_json"), CALENDAR_ROWS, log=logs.append)

    joined = "\n".join(logs)
    assert "output contract" in joined
    assert "60 minutes" in joined
    assert "expected failure modes" in joined


def test_a_brief_with_no_failure_modes_says_so_rather_than_printing_nothing():
    logs: list[str] = []
    log_task_brief(
        {"summary": "s", "output_contract": "c", "failure_modes": [], "source": "orchestrator"},
        get_task("gsm8k"), log=logs.append,
    )
    assert any("failure modes: none stated" in line for line in logs)


# --------------------------------------------------------------------------
# Degrading honestly
# --------------------------------------------------------------------------


def test_an_unreachable_orchestrator_does_not_stop_a_runnable_run(monkeypatch):
    _stub_anthropic(monkeypatch, error=RuntimeError("no route to host"))
    logs: list[str] = []
    brief = build_task_brief(get_task("gsm8k"), [{"text": "q", "answer": "#### 1"}],
                            log=logs.append)

    assert brief["source"] == "fallback"
    assert any("brief FAILED" in line and "no route to host" in line for line in logs)


def test_an_unparseable_reply_falls_back(monkeypatch):
    _stub_anthropic(monkeypatch, reply="I would rather not answer in JSON.")
    brief = build_task_brief(get_task("gsm8k"), [{"text": "q", "answer": "#### 1"}],
                            log=lambda _m: None)
    assert brief["source"] == "fallback"


def test_a_reply_missing_the_contract_falls_back(monkeypatch):
    """A brief with no output contract is worse than the fallback: it looks authored, so a reader
    assumes the conventions were stated when they were not — which is B269 with extra steps."""
    _stub_anthropic(monkeypatch, reply=json.dumps({"summary": "s", "output_contract": ""}))
    logs: list[str] = []
    brief = build_task_brief(get_task("gsm8k"), [{"text": "q", "answer": "#### 1"}],
                            log=logs.append)

    assert brief["source"] == "fallback"
    assert any("missing summary or output_contract" in line for line in logs)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_the_fallback_admits_that_it_has_no_contract(task):
    """Honest rather than helpful. A reader of the log has to be able to see that synthesis ran
    WITHOUT a real brief, instead of assuming the prose came from the model."""
    brief = fallback_brief(get_task(task))

    assert brief["source"] == "fallback"
    assert "UNAVAILABLE" in brief["output_contract"]
    assert brief["failure_modes"] == []
    assert get_task(task).title in brief["summary"]


# --------------------------------------------------------------------------
# It reaches the prompts
# --------------------------------------------------------------------------


def test_the_context_block_carries_the_task_the_contract_and_the_mistakes():
    block = brief_context_block(
        {"summary": "Turn a request into a call.",
         "output_contract": "ISO 8601, 60-minute default.",
         "failure_modes": ["year resolution error"]},
        get_task("calendar_json"),
    )
    assert "TASK: Turn a request into a call." in block
    assert "OUTPUT CONTRACT: ISO 8601, 60-minute default." in block
    assert "COMMON MISTAKES TO AVOID: year resolution error" in block


def test_the_context_block_falls_back_when_no_brief_was_stored():
    """Reached on a resumed checkpoint written before the field existed. It must produce usable
    prompt text rather than `None`, and it must still be visibly a fallback."""
    block = brief_context_block(None, get_task("xlam_bfcl"))
    assert "TASK:" in block
    assert "UNAVAILABLE" in block


def test_the_two_format_bound_tasks_are_no_longer_described_identically(monkeypatch):
    """The concrete B269 regression. Under the type-keyed table both were `function_call` and got the
    same sentence; now each brief is authored from its OWN rows, so calendar's conventions appear in
    calendar's brief and nowhere else."""
    _stub_anthropic(monkeypatch)
    calendar = build_task_brief(get_task("calendar_json"), CALENDAR_ROWS, log=lambda _m: None)

    _stub_anthropic(monkeypatch, reply=json.dumps({
        "summary": "Turn a request into calls against the declared tools.",
        "output_contract": "A JSON array of {name, arguments}; use only declared functions.",
        "failure_modes": ["undeclared function"],
    }))
    xlam = build_task_brief(get_task("xlam_bfcl"), [{"text": "q", "answer": "[]"}],
                            log=lambda _m: None)

    assert calendar["output_contract"] != xlam["output_contract"]
    assert "60 minutes" in calendar["output_contract"]
    assert "60 minutes" not in xlam["output_contract"]


def test_synthesis_builds_its_teacher_prompt_from_the_brief(monkeypatch):
    """The whole point of authoring it. `synthesize_examples` must insert the brief, so a convention
    the orchestrator recorded reaches the generator."""
    import data.curriculum as curriculum

    prompts: list[str] = []

    def generate(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        return json.dumps({"text": "new request", "answer": "[]"})

    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")
    curriculum.synthesize_examples(
        [{"text": "old request", "answer": "[]", "tools": [{"name": "t"}]}],
        task="xlam_bfcl", n=1, generate_fn=generate, verify_fn=None, log=None,
        brief={"summary": "A very specific benchmark.",
               "output_contract": "One call, ISO 8601 datetimes.",
               "failure_modes": ["year resolution error"]},
    )

    assert prompts, "synthesis made no teacher call"
    assert "A very specific benchmark." in prompts[0]
    assert "One call, ISO 8601 datetimes." in prompts[0]


def test_synthesis_without_a_brief_still_names_the_task(monkeypatch):
    """A run resumed from before the field existed must still produce a usable prompt."""
    import data.curriculum as curriculum

    prompts: list[str] = []

    def generate(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        return json.dumps({"text": "new request", "answer": "[]"})

    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")
    curriculum.synthesize_examples(
        [{"text": "old", "answer": "[]", "tools": [{"name": "t"}]}],
        task="xlam_bfcl", n=1, generate_fn=generate, log=None, brief=None,
    )
    assert get_task("xlam_bfcl").title in prompts[0]
