"""The teacher's own check on the rows it generated: what it is told, and what it may not do.

This is the SECOND gate on a generated row. The first is the task's exact programmatic verifier
(`tests/data/test_synth_verifiers.py`), which is free and unfoolable but only decides FORM. Where no
exact check exists — and for the semantic half where one does — the teacher is asked whether its own
output is right.

Two invariants, both of which failed in production.

WHAT THE TEACHER IS TOLD (B267)
    The prompts used to show the teacher only the label STRING, so it read the label as an ordinary
    English word. On RouterBench — where `local` means "a small on-device model can answer this
    correctly" — it rejected 70% of generated rows with verdicts like *"the utterance is a math
    problem, not a local query"* and *"the utterance describes fog, not clouds"*. Those are correct
    answers to the question the prompt actually asked; the prompt was asking the wrong one. So the
    label DEFINITIONS have to reach the prompt, and the prompt has to say explicitly that topic is
    not the criterion.

WHAT A VERIFIER MAY NEVER DO
    Empty a dataset. Verification is one model call per row against a service that can be
    unreachable, slow, or chatty, and an unparseable reply is not evidence that a row is wrong. So
    the contract is FAIL-OPEN: an endpoint error or an unreadable verdict KEEPS the row. Only an
    explicit `{"valid": false}` drops one, and the teacher's stated reason is logged so a bad
    GENERATOR prompt is visible rather than silently absorbed as a low keep-rate.

WHAT THE TEACHER IS TOLD, PART TWO (B269)
    `task_description` is now a REQUIRED keyword on `verify_generated_labels`, matching
    `verify_generated_answers`. It carries the orchestrator-authored brief — what the benchmark is
    and what its output contract says — rendered by `agent.task_brief.brief_context_block`. It is
    required rather than defaulted because the thing it replaced was a per-task-TYPE table, and a
    default is exactly how a verifier ends up judging a task nobody described to it.
"""
from __future__ import annotations

import json

import pytest

from agent.task_brief import brief_context_block
from data.curriculum import (
    SYNTH_SHOTS,
    _label_context_block,
    _synthesize_new_gold,
    verify_generated_answers,
    verify_generated_labels,
)
from tasks import get_task

ROUTER_DEFINITIONS = dict(get_task("routerbench").label_definitions)

# The brief as the pipeline builds it: the orchestrator's prose, rendered by the one function that
# every synthesis and verification prompt gets its task description from. Written out here rather
# than passing a bare sentence, so a change to the rendering shows up as a failure in the tests that
# assert the brief reaches the prompt.
ROUTER_BRIEF = {
    "summary": (
        "RouterBench: decide whether a request is easy enough for a small on-device model or "
        "must be escalated to a larger cloud model."
    ),
    "output_contract": "Reply with exactly one label word, 'local' or 'route', and nothing else.",
    "failure_modes": ["answering the request instead of classifying it"],
    "source": "orchestrator",
}
TASK_BRIEF = brief_context_block(ROUTER_BRIEF, get_task("routerbench"))


def _verdict(valid, reason="because"):
    return json.dumps({"valid": valid, "reason": reason})


def _rows(n=3, label="local"):
    return [{"text": f"generated utterance {i}", "label": label} for i in range(n)]


def _verify_labels(rows, **kwargs):
    """`verify_generated_labels` with the brief supplied, which it now requires."""
    kwargs.setdefault("task_description", TASK_BRIEF)
    return verify_generated_labels(rows, **kwargs)


# --------------------------------------------------------------------------
# The teacher is told what the label MEANS
# --------------------------------------------------------------------------


def test_the_definitions_reach_the_verification_prompt():
    prompts: list[str] = []

    def generate(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        return _verdict(True)

    _verify_labels(
        _rows(1), generate_fn=generate, label_definitions=ROUTER_DEFINITIONS,
        all_labels=["local", "route"],
    )

    assert prompts
    assert "a SMALL on-device model can answer it correctly" in prompts[0]
    assert "should be escalated to a LARGER cloud model" in prompts[0]


def test_the_prompt_says_topic_is_not_the_criterion():
    """The exact instruction that stops the 70% over-rejection: judge the CLASS as defined, not
    whether the utterance's subject matter resembles the label's wording."""
    prompts: list[str] = []
    _verify_labels(
        _rows(1),
        generate_fn=lambda prompt, *_a, **_k: prompts.append(prompt) or _verdict(True),
        label_definitions=ROUTER_DEFINITIONS,
    )
    assert "Do NOT answer false merely because" in prompts[0]
    assert "TOPIC is unrelated to the label's wording" in prompts[0]


def test_the_label_under_consideration_is_named_with_its_meaning():
    block = _label_context_block("local", ROUTER_DEFINITIONS, ["local", "route"])
    assert "The label under consideration is 'local'" in block
    # Every class is glossed, not only the one being judged: a decision boundary needs both sides.
    assert "'route'" in block


def test_no_definitions_means_no_context_block_rather_than_an_empty_heading():
    """CLINC150's labels ARE descriptions of the utterance's intent, so a gloss for 151 classes
    would be prompt noise. An empty heading would be worse than nothing — it reads as a section the
    teacher should have been given."""
    assert _label_context_block("transfer", {}, ["transfer"]) == ""
    assert _label_context_block("transfer", None, None) == ""


def test_real_in_class_examples_are_shown_alongside_the_definition():
    """A decision boundary is easier to show than to describe (B276), and the examples are real rows
    so the verifier judges the class as it ACTUALLY appears."""
    prompts: list[str] = []
    _verify_labels(
        _rows(1),
        generate_fn=lambda prompt, *_a, **_k: prompts.append(prompt) or _verdict(True),
        label_definitions=ROUTER_DEFINITIONS,
        reference_rows=[{"text": "what is 2 + 2", "label": "local"},
                        {"text": "summarise this filing", "label": "route"}],
    )
    assert "Real, confirmed examples of the 'local' class" in prompts[0]
    assert "what is 2 + 2" in prompts[0]
    # Only the class under judgement, so the teacher is not shown the answer for the other side.
    assert "summarise this filing" not in prompts[0]


def test_the_brief_reaches_the_label_verification_prompt():
    """`task_description` is the orchestrator's brief, and it is the top of the prompt.

    Before it was threaded through, the verifier was told only "you are checking one training
    example for a text classifier" — no benchmark, no output contract — so it judged the task it
    inferred from the label word rather than the task being trained (B267/B269). The summary and the
    output contract are both asserted because they answer different questions: what the rows ARE,
    and what a correct one has to look like.
    """
    prompts: list[str] = []
    _verify_labels(
        _rows(1),
        generate_fn=lambda prompt, *_a, **_k: prompts.append(prompt) or _verdict(True),
        label_definitions=ROUTER_DEFINITIONS,
    )
    assert ROUTER_BRIEF["summary"] in prompts[0]
    assert ROUTER_BRIEF["output_contract"] in prompts[0]
    # At the top: everything after it is read in its context, so a brief buried under the
    # instructions is a brief the teacher applies retroactively or not at all.
    assert prompts[0].startswith(TASK_BRIEF)


def test_task_description_is_required_rather_than_defaulted():
    """A default would let a caller that never learned about the brief silently keep the old,
    context-free prompt — the exact regression the keyword exists to make impossible (B269)."""
    import inspect

    parameter = inspect.signature(verify_generated_labels).parameters["task_description"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_five_in_class_examples_are_shown_to_the_verifier():
    """Five, matching SYNTH_SHOTS: the verifier is judging the same class the generator was shown,
    so showing it fewer examples than the generator got makes the two disagree about where the
    boundary is. Six are offered and exactly the first five appear (B276)."""
    prompts: list[str] = []
    references = [{"text": f"in-class example {i}", "label": "local"} for i in range(6)]
    _verify_labels(
        _rows(1),
        generate_fn=lambda prompt, *_a, **_k: prompts.append(prompt) or _verdict(True),
        label_definitions=ROUTER_DEFINITIONS,
        reference_rows=references,
    )
    assert SYNTH_SHOTS == 5
    for reference in references[:SYNTH_SHOTS]:
        assert reference["text"] in prompts[0]
    assert references[SYNTH_SHOTS]["text"] not in prompts[0]


def test_the_answer_verifier_is_shown_real_request_answer_pairs():
    """On `calendar_json` the conventions ARE the task — the 60-minute default, ISO 8601, resolving
    against the reference instant — and a verifier that has to guess them is judging its own guess
    (B269)."""
    prompts: list[str] = []
    verify_generated_answers(
        [{"text": "book a dentist visit", "answer": "[]"}],
        task_description="turn a request into one calendar call",
        generate_fn=lambda prompt, *_a, **_k: prompts.append(prompt) or _verdict(True),
        reference_rows=[{"text": "remind me on the 3rd", "answer": '[{"name": "x"}]'}],
    )
    assert "Real, confirmed examples of this task" in prompts[0]
    assert "remind me on the 3rd" in prompts[0]
    assert "turn a request into one calendar call" in prompts[0]


# --------------------------------------------------------------------------
# Rejection works, and is explained
# --------------------------------------------------------------------------


def test_a_row_the_teacher_rejects_is_dropped():
    kept = _verify_labels(
        _rows(3), generate_fn=lambda *_a, **_k: _verdict(False, "belongs to the other class"),
    )
    assert kept == []


def test_a_row_the_teacher_accepts_is_kept():
    rows = _rows(3)
    assert _verify_labels(rows, generate_fn=lambda *_a, **_k: _verdict(True)) == rows


def test_the_stated_reason_is_logged_so_a_bad_generator_prompt_is_visible():
    """Without this, a generator prompt asking the wrong question shows up only as a low keep-rate —
    which is indistinguishable from a hard class."""
    logs: list[str] = []
    _verify_labels(
        _rows(2), generate_fn=lambda *_a, **_k: _verdict(False, "the utterance is a math problem"),
        log=logs.append,
    )
    joined = " ".join(logs)
    assert "validated 0/2" in joined
    assert "the utterance is a math problem" in joined


def test_the_rejection_log_is_bounded():
    """A rebuild can reject thousands of rows; quoting all of them buries the summary."""
    logs: list[str] = []
    _verify_labels(
        _rows(40), generate_fn=lambda *_a, **_k: _verdict(False, "nope"), log=logs.append,
    )
    assert any("more rejected" in line for line in logs)


def test_an_answer_the_teacher_rejects_is_dropped():
    kept = verify_generated_answers(
        [{"text": "q", "answer": "wrong"}],
        task_description="a task", generate_fn=lambda *_a, **_k: _verdict(False, "not correct"),
    )
    assert kept == []


def test_a_generated_row_with_an_empty_answer_is_dropped_without_a_teacher_call():
    """Nothing to judge, and the call would cost as much as a real one."""
    calls: list[int] = []
    kept = verify_generated_answers(
        [{"text": "q", "answer": "   "}],
        task_description="a task",
        generate_fn=lambda *_a, **_k: calls.append(1) or _verdict(True),
    )
    assert kept == []
    assert calls == []


# --------------------------------------------------------------------------
# Fail open: a verifier may never empty a dataset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verify", [verify_generated_labels, verify_generated_answers])
def test_an_unreachable_verifier_keeps_every_row(verify):
    def explode(*_args, **_kwargs):
        raise RuntimeError("endpoint refused the connection")

    rows = [{"text": "t", "label": "local", "answer": "a"}]
    kept = _call(verify, rows, explode)
    assert kept == rows


@pytest.mark.parametrize("verify", [verify_generated_labels, verify_generated_answers])
def test_an_unparseable_verdict_keeps_the_row(verify):
    """An unreadable reply is not evidence the row is wrong."""
    rows = [{"text": "t", "label": "local", "answer": "a"}]
    assert _call(verify, rows, lambda *_a, **_k: "I'd rather not say") == rows


@pytest.mark.parametrize("verify", [verify_generated_labels, verify_generated_answers])
def test_a_verdict_with_no_valid_field_keeps_the_row(verify):
    rows = [{"text": "t", "label": "local", "answer": "a"}]
    assert _call(verify, rows, lambda *_a, **_k: json.dumps({"reason": "hmm"})) == rows


@pytest.mark.parametrize("verify", [verify_generated_labels, verify_generated_answers])
def test_a_verdict_wrapped_in_prose_is_still_read(verify):
    """Teachers add preambles. Treating that as unparseable would silently keep wrong rows."""
    rows = [{"text": "t", "label": "local", "answer": "a"}]
    reply = "Sure! Here is my assessment:\n" + _verdict(False, "wrong class")
    assert _call(verify, rows, lambda *_a, **_k: reply) == []


@pytest.mark.parametrize("verify", [verify_generated_labels, verify_generated_answers])
def test_no_generate_function_means_no_verification_rather_than_no_rows(verify):
    """The endpoint being absent must not be indistinguishable from every row being wrong."""
    rows = [{"text": "t", "label": "local", "answer": "a"}]
    assert _call(verify, rows, None) == rows


@pytest.mark.parametrize("verify", [verify_generated_labels, verify_generated_answers])
def test_an_empty_input_is_returned_unchanged(verify):
    assert _call(verify, [], lambda *_a, **_k: _verdict(True)) == []


def _call(verify, rows, generate_fn):
    """Both verifiers now take the same required `task_description` keyword."""
    return verify(rows, task_description="a task", generate_fn=generate_fn)


# --------------------------------------------------------------------------
# Five demonstrations is a contract, and a short prompt says so
# --------------------------------------------------------------------------
#
# SYNTH_SHOTS is not a tuning knob that quietly degrades. The teacher scores 0.1131 span-F1
# zero-shot on BC5CDR and 0.7190 five-shot — a 6.4x difference — so a prompt that went out with two
# demonstrations produced a batch whose keep rate cannot be compared with any other batch's. The
# only way to know that happened is for the short prompt to announce itself, because the batch
# itself looks exactly like a batch on a hard class (B276).


def _short_shot_lines(logs: list[str]) -> list[str]:
    return [line for line in logs if "demonstration(s) available" in line]


def test_a_verification_batch_with_too_few_examples_warns():
    logs: list[str] = []
    _verify_labels(
        _rows(2),
        generate_fn=lambda *_a, **_k: _verdict(True),
        reference_rows=[{"text": f"in-class {i}", "label": "local"} for i in range(3)],
        log=logs.append,
    )
    warnings = _short_shot_lines(logs)
    assert warnings, "a three-demonstration verification prompt went out unannounced"
    assert f"only 3 of {SYNTH_SHOTS}" in warnings[0]
    # Once per label, not once per row: this prompt is issued thousands of times a rebuild.
    assert len(warnings) == 1


def test_a_verification_batch_with_the_full_five_examples_is_silent():
    """The warning has to mean something. A line on every batch is a line nobody reads."""
    logs: list[str] = []
    _verify_labels(
        _rows(2),
        generate_fn=lambda *_a, **_k: _verdict(True),
        reference_rows=[{"text": f"in-class {i}", "label": "local"}
                        for i in range(SYNTH_SHOTS + 2)],
        log=logs.append,
    )
    assert _short_shot_lines(logs) == []


def test_answer_verification_warns_on_a_short_reference_set_too():
    """Same contract on the generation family, where the teacher invented both halves of the row
    and the references are the only statement of the task's conventions (B269)."""
    logs: list[str] = []
    verify_generated_answers(
        [{"text": "book a dentist visit", "answer": "[]"}],
        task_description=TASK_BRIEF,
        generate_fn=lambda *_a, **_k: _verdict(True),
        reference_rows=[{"text": "remind me on the 3rd", "answer": '[{"name": "x"}]'}],
        log=logs.append,
    )
    assert _short_shot_lines(logs)


def test_generation_warns_when_a_class_cannot_supply_five_demonstrations():
    """The generator side of the same contract, and the case that actually happens: a rare class
    has three rows, so its prompts are three-shot while the common classes' are five-shot, and
    without the line the rare class just looks harder to generate for."""
    logs: list[str] = []
    _synthesize_new_gold(
        [{"text": f"rare class row {i}", "label": "route"} for i in range(3)],
        task_description=TASK_BRIEF,
        n=2,
        generate_fn=lambda *_a, **_k: "a newly written utterance",
        log=logs.append,
    )
    warnings = _short_shot_lines(logs)
    assert warnings, "a short in-class generation prompt went out unannounced"
    assert "in-class generation for 'route'" in warnings[0]


def test_generation_is_silent_when_the_class_has_five_demonstrations_to_spare():
    logs: list[str] = []
    _synthesize_new_gold(
        [{"text": f"common class row {i}", "label": "local"}
         for i in range(SYNTH_SHOTS + 2)],
        task_description=TASK_BRIEF,
        n=2,
        generate_fn=lambda *_a, **_k: "a newly written utterance",
        log=logs.append,
    )
    assert _short_shot_lines(logs) == []


# --------------------------------------------------------------------------
# Order is preserved
# --------------------------------------------------------------------------


def test_verification_preserves_input_order():
    """Rows are checked concurrently against the batching synthesis server. A reordered result would
    silently detach a row from whatever the caller pairs it with downstream."""
    rows = [{"text": f"row {i}", "label": "local"} for i in range(20)]
    kept = _verify_labels(rows, generate_fn=lambda *_a, **_k: _verdict(True))
    assert [row["text"] for row in kept] == [row["text"] for row in rows]


def test_the_teacher_pass_can_be_switched_off_for_the_exact_verifier_alone(monkeypatch):
    """`SLM_VERIFY_SYNTH=0` isolates generation plus the programmatic check — used when the exact
    verifier is the thing under test, and when teacher budget is the binding constraint."""
    import importlib

    import data.curriculum as curriculum

    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")
    curriculum = importlib.reload(curriculum)
    assert curriculum._verify_synth_enabled() is False

    monkeypatch.setenv("SLM_VERIFY_SYNTH", "1")
    assert curriculum._verify_synth_enabled() is True
    importlib.reload(curriculum)
