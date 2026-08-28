"""Every verifier is told the task's own vocabulary before it judges a row (B314, generalised).

WHY THIS FILE EXISTS
    A teacher asked to judge a row it has no context for does not decline — it invents the context and
    then judges against that. Both halves of the context had to be supplied before the verdicts became
    trustworthy, and both were missing for different reasons:

      the ROW's own fields   `verify_generated_answers` showed only the request and the proposed
                             answer, so on xlam run 38661753 it rejected 79 of 330 rows (24%) with
                             reasons like "Tool name 'calculate_distance' is not present in the
                             provided tools list" — about a tools list that was never in the prompt.
                             Every one of those rows had already passed the programmatic verifier
                             against that row's own `tools`, so each rejection was provably false.
                             Fixed by `_row_context_block` (B314).

      the TASK's vocabulary  what is true of the whole task and therefore appears on no single row:
                             the closed class list, what each class means, the entity types. Without
                             it the teacher decides what the label space must be from the label's
                             wording, which is how a grade-school arithmetic problem was rejected from
                             RouterBench's `local` class for being "a math problem, not a local
                             query" — 70% of generated rows discarded (B267/B269).

    `_task_context_block` is the second half, and it is passed to BOTH verifiers so the fix is not
    per-task. These tests cover what it emits for each shape of task, that it survives the trip into
    the prompt the teacher actually receives, and that all eight registered tasks get theirs.

WHY THE VOCABULARY COMES FROM THE ROWS
    No spec enumerates its class list; the vocabulary is a property of the loaded data. Reading it
    from the REAL anchor rows rather than from the generated ones also matters: a teacher that
    invented a class must not be able to legitimise it by having written it down.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import tasks
from data.curriculum import (
    _task_context_block,
    synthesize_examples,
    verify_generated_answers,
    verify_generated_labels,
)
from tasks import get_task, task_names

CLINC_ROWS = [
    {"text": "move fifty dollars into savings", "label": "transfer"},
    {"text": "why did my card get declined", "label": "card_declined"},
    {"text": "how much is left on my account", "label": "balance"},
]
ROUTER_ROWS = [
    {"text": "what is 2 + 2", "label": "local"},
    {"text": "summarise this ten-page filing", "label": "route"},
]
NER_ROWS = [
    {
        "text": "Aspirin caused gastric ulcers in two patients.",
        "entities": [
            {"text": "Aspirin", "type": "Chemical"},
            {"text": "gastric ulcers", "type": "Disease"},
        ],
    },
]


def _verdict(valid=True, reason="because"):
    return json.dumps({"valid": valid, "reason": reason})


# --------------------------------------------------------------------------
# What the block says for a closed label space
# --------------------------------------------------------------------------


def test_a_closed_label_space_lists_every_class_it_observed():
    """The whole list, not the row's own class. A verifier shown one class cannot tell whether a row
    belongs to it or to a neighbour it was never told about, which is the decision it is being asked
    to make."""
    block = _task_context_block(get_task("clinc150"), CLINC_ROWS)

    for row in CLINC_ROWS:
        assert row["label"] in block


def test_a_listed_class_is_declared_valid_by_definition():
    """The instruction that stops the 70% over-rejection at the TASK level.

    The teacher's failure mode is not "I cannot see the class list", it is "I would not have named
    this class that", and it answers that question instead of the one asked. So the block has to say
    outright that the enumeration is authoritative and disagreement with a class NAME is not grounds
    for rejecting a row.
    """
    block = _task_context_block(get_task("clinc150"), CLINC_ROWS).lower()

    assert "valid by definition" in block
    assert "never reject" in block


def test_the_definitions_are_shown_beside_the_classes_that_have_them():
    """RouterBench is the case the gloss exists for: `local` means "a small on-device model can
    answer this correctly", and read as an English word it means the opposite of what the class is
    for. Naming the class without its meaning is what produced verdicts like "the utterance is a
    math problem, not a local query"."""
    spec = get_task("routerbench")
    block = _task_context_block(spec, ROUTER_ROWS)

    for label, definition in spec.label_definitions.items():
        assert label in block
        assert definition in block


def test_a_closed_label_space_with_no_definitions_says_that_it_has_none():
    """CLINC150's 151 intents arrive with no written definitions, and silence there is worse than an
    admission: a verifier given only the names falls back to reading them as English words, which is
    the same failure the definitions exist to prevent. So the absence is stated, together with what to
    judge from instead — the class as it is USED in the confirmed examples."""
    spec = get_task("clinc150")
    assert not spec.label_definitions, "fixture assumption: clinc150 ships no definitions"

    block = _task_context_block(spec, CLINC_ROWS).lower()

    assert "no written definitions" in block
    # And what to use instead, or the admission just tells the teacher it is on its own.
    assert "examples" in block


def test_a_task_with_definitions_does_not_claim_to_be_missing_them():
    """The counterpart assertion, so the note above means something. A block that carried it
    unconditionally would tell the teacher to ignore glosses it had just been given."""
    block = _task_context_block(get_task("routerbench"), ROUTER_ROWS).lower()

    assert "no written definitions" not in block


def test_the_observed_rows_are_what_pin_the_class_list():
    """No spec enumerates its classes, so the rows are the only source — and they must be the REAL
    anchors. Reading the vocabulary from generated rows instead would let a teacher that invented a
    class name have that name confirmed back to it as valid.

    Uses clinc150 because it ships no definitions, so the only text in the block is the class names
    themselves and an absent class cannot sneak in through another class's gloss.
    """
    block = _task_context_block(get_task("clinc150"), [CLINC_ROWS[0]])

    assert CLINC_ROWS[0]["label"] in block
    for unobserved in CLINC_ROWS[1:]:
        assert unobserved["label"] not in block


def test_the_definitions_supply_the_class_list_when_no_rows_are_given():
    """`curate` always has anchors, but the block is also built for a spec on its own (reporting, a
    resumed state with no rows to hand). Falling back to the declared classes keeps it useful there
    instead of emitting nothing."""
    spec = get_task("routerbench")

    block = _task_context_block(spec, [])

    for label in spec.label_definitions:
        assert label in block


# --------------------------------------------------------------------------
# What the block says for a span task, and for a task with no vocabulary
# --------------------------------------------------------------------------


def test_a_span_task_lists_its_entity_types():
    """The span equivalent of the class list. `Chemical` and `Disease` are BC5CDR's entire label
    space and they appear only inside the rows' `entities`, so a verifier not told them judges
    against whatever entity taxonomy it happens to know."""
    block = _task_context_block(get_task("ner_bc5cdr"), NER_ROWS)

    assert "Chemical" in block
    assert "Disease" in block


def test_the_entity_types_are_declared_exhaustive():
    """A verifier that treats the list as a sample will accept a fourth type the scorer cannot
    score, which is the span-level version of accepting an out-of-vocabulary class."""
    block = _task_context_block(get_task("ner_bc5cdr"), NER_ROWS).lower()

    assert "only ones" in block


@pytest.mark.parametrize("task", ("gsm8k", "dialogsum"))
def test_an_open_ended_task_with_neither_vocabulary_yields_nothing(task):
    """Empty string, not an empty heading.

    A heading with nothing under it reads as a section the teacher was supposed to be given and is
    worse than silence — and for these tasks there is genuinely nothing to enumerate: the answer is a
    number the question determines, or free prose. Claiming a label space here would be the B267
    mistake manufactured rather than merely inherited.
    """
    rows = [{"text": "a request", "answer": "an answer"}]

    assert _task_context_block(get_task(task), rows) == ""


def test_no_spec_at_all_yields_nothing_rather_than_raising():
    """The block is built on paths that may not have resolved a spec yet, and a verification pass is
    never allowed to be the thing that breaks a run."""
    assert _task_context_block(None, CLINC_ROWS) == ""


def test_rows_that_carry_no_vocabulary_are_tolerated():
    """Malformed and label-less rows reach here from a resumed artifact. Skipping them is right;
    raising would turn a cosmetic gap into a failed rebuild."""
    spec = get_task("clinc150")

    block = _task_context_block(spec, [{"text": "no label at all"}, "not a row", None])

    assert block == ""


# --------------------------------------------------------------------------
# It survives the trip into the prompt the teacher actually receives
# --------------------------------------------------------------------------


def _captured_prompt(call):
    """Run `call` with a capturing `generate_fn` and return the single prompt it received."""
    prompts: list[str] = []

    def generate(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        return _verdict(True)

    call(generate)
    assert len(prompts) == 1, f"expected exactly one verifier call, got {len(prompts)}"
    return prompts[0]


def test_the_class_list_reaches_the_label_verification_prompt():
    spec = get_task("clinc150")
    prompt = _captured_prompt(lambda generate: verify_generated_labels(
        [{"text": "shift some money across", "label": "transfer"}],
        task_description="CLINC150 intent classification",
        generate_fn=generate,
        all_labels=sorted({row["label"] for row in CLINC_ROWS}),
        reference_rows=CLINC_ROWS,
        task_context=_task_context_block(spec, CLINC_ROWS),
    ))

    for row in CLINC_ROWS:
        assert row["label"] in prompt
    assert "valid by definition" in prompt.lower()


def test_the_entity_vocabulary_reaches_the_answer_verification_prompt():
    """The block is independent of the row being judged, which is why it can carry a vocabulary that
    lives across the whole task.

    The row below needs an `answer` because `verify_generated_answers` refuses to spend a teacher call
    on a row with nothing to judge. See the note at the bottom of this file: a synthesized BC5CDR row
    carries only `entities`, and that is a production gap rather than something this test should paper
    over. The answer names only the span TEXTS, so the type names below can only have arrived through
    the task-context block.
    """
    spec = get_task("ner_bc5cdr")
    row = dict(NER_ROWS[0], answer=json.dumps(["Aspirin", "gastric ulcers"]))

    prompt = _captured_prompt(lambda generate: verify_generated_answers(
        [row],
        task_description="BC5CDR chemical and disease span extraction",
        generate_fn=generate,
        reference_rows=NER_ROWS,
        task_context=_task_context_block(spec, NER_ROWS),
    ))

    assert "Chemical" in prompt
    assert "Disease" in prompt


def test_synthesize_examples_supplies_the_task_context_to_the_label_verifier():
    """The wiring, checked separately from the block's content.

    Both are needed: a correct block that never reaches a prompt fixes nothing, and that is exactly
    the state `verify_generated_answers` was in before B314 — the context existed on the row and the
    prompt did not carry it. The expected value is built from the ANCHORS, not from the generated
    rows, so a teacher that invented a class cannot see it confirmed back as valid.
    """
    spec = get_task("clinc150")
    captured: dict = {}

    def fake_verify(rows, **kwargs):
        captured.update(kwargs)
        return rows

    with patch("data.curriculum.verify_generated_labels", side_effect=fake_verify):
        synthesize_examples(
            CLINC_ROWS,
            task="clinc150",
            n=2,
            generate_fn=lambda *_args, **_kwargs: "a newly written utterance",
        )

    assert captured["task_context"] == _task_context_block(spec, CLINC_ROWS)
    assert captured["task_context"], "the closed label space produced no task context at all"


def test_synthesize_examples_supplies_the_task_context_to_the_answer_verifier():
    """Same wiring on the generation family, which is the one path where the teacher invents BOTH
    halves of the row and therefore has the least constraint on it from anywhere else."""
    anchors = [{"text": "Janet has 3 apples and buys 4 more.", "answer": "7"}]
    spec = get_task("gsm8k")
    captured: dict = {}

    def fake_verify(rows, **kwargs):
        captured.update(kwargs)
        return rows

    def generate(_prompt, **_kwargs):
        return json.dumps({"text": "Ravi has 5 pears and eats 2.", "answer": "3"})

    with patch("data.curriculum.verify_generated_answers", side_effect=fake_verify):
        synthesize_examples(anchors, task="gsm8k", n=2, generate_fn=generate)

    assert captured["task_context"] == _task_context_block(spec, anchors)


# --------------------------------------------------------------------------
# All eight registered tasks, not the ones that happened to have a bug
# --------------------------------------------------------------------------
#
# B314 was fixed on xlam because xlam was the run being debugged. The generalisation is the point of
# this change, so the coverage is driven off the registry: a ninth task cannot be added without either
# declaring what its vocabulary is or failing here.

# One row per task, in the shape that task's rows actually have, plus the vocabulary that MUST reach
# the verifier prompt. Written out per task rather than derived from the spec, because deriving it
# from the same code under test is how a test ends up asserting nothing.
TASK_FIXTURES = {
    "clinc150": (CLINC_ROWS, ("transfer", "card_declined", "balance")),
    "routerbench": (ROUTER_ROWS, ("local", "route")),
    # `ham` is the reason this task declares label_definitions at all: it is corpus jargon, not
    # English, so the vocabulary reaching the verifier prompt is what stops the teacher reading it
    # as the food and rejecting good rows — the RouterBench `local` failure (B267) in a new costume.
    "sms_spam": (
        [
            {"text": "hey are we still on for dinner at 7", "label": "ham"},
            {"text": "WINNER!! You have won a free entry to our prize draw. Txt CLAIM to 81010",
             "label": "spam"},
        ],
        ("ham", "spam"),
    ),
    "proactive_listening": (
        [
            {"text": "so anyway, as I was saying about the report", "label": "wait"},
            {"text": "wait, what did you just say the deadline was", "label": "interrupt"},
        ],
        ("wait", "interrupt"),
    ),
    # Span vocabulary rather than a class list: BC5CDR carries its gold in `entities`. The `answer`
    # names only the span TEXTS, so the two type names can only have reached the prompt through the
    # task-context block.
    "ner_bc5cdr": (
        [dict(NER_ROWS[0], answer=json.dumps(["Aspirin", "gastric ulcers"]))],
        ("Chemical", "Disease"),
    ),
    # The two function-calling tasks have no TASK-level vocabulary — the callable surface differs per
    # row — so their vocabulary is the row's own `tools`, carried by `_row_context_block`. That is the
    # original B314 case, and it has to keep working alongside the task-level block. The expected
    # tokens include a parameter the ANSWER does not mention, so the assertion cannot be satisfied by
    # the proposed answer being echoed back.
    "xlam_bfcl": (
        [{
            "text": "find me a flight to Lisbon on Friday",
            "answer": '[{"name": "search_flights", "arguments": {"destination": "Lisbon"}}]',
            "tools": [{
                "name": "search_flights",
                "parameters": {"destination": "string", "cabin_class": "string"},
            }],
        }],
        ("search_flights", "cabin_class"),
    ),
    "calendar_json": (
        [{
            "text": "remind me to call the dentist on Friday at 3",
            "answer": '[{"name": "calendar.events.insert", "arguments": {"summary": "the dentist"}}]',
            "tools": [{
                "name": "calendar.events.insert",
                "parameters": {"summary": "string", "reference_instant": "ISO-8601"},
            }],
        }],
        ("calendar.events.insert", "reference_instant"),
    ),
    # ToolBench's callable surface is per row too, and it is the case that most needs the block:
    # the row declares a handful of APIs drawn from ~16,000, so a teacher judging without the list
    # in front of it will reject a correct call for naming an endpoint it does not recognise —
    # which is B314 exactly, on a namespace where nobody's prior is reliable.
    "toolbench": (
        [{
            "text": "You are AutoGPT... I want the forecast for Lisbon before I pack.",
            "query": "I want the forecast for Lisbon before I pack.",
            "answer": (
                "Thought: I should look up the forecast.\n"
                "Action: get_forecast_for_weather_api\n"
                'Action Input: {"city": "Lisbon"}\n'
                "Thought: I can answer now.\nAction: Finish\n"
                'Action Input: {"return_type": "give_answer", "final_answer": "Warm and clear."}'
            ),
            "tools": [{
                "name": "get_forecast_for_weather_api",
                "parameters": {
                    "properties": {"city": "string", "units": "string"},
                    "required": ["city"],
                    "optional": ["units"],
                },
            }],
        }],
        ("get_forecast_for_weather_api", "units"),
    ),
    # No closed vocabulary of any kind: the answer is a number the question determines, or free
    # prose. The requirement for these two is the opposite one — see the assertion below.
    "gsm8k": ([{"text": "Janet has 3 apples and buys 4 more.", "answer": "7"}], ()),
    "dialogsum": ([{"text": "A: are we still on for 6?\nB: yes", "answer": "They confirm 6pm."}], ()),
}


def test_the_fixture_table_covers_the_whole_registry():
    """The guard that makes the parametrization meaningful: a new task must be declared here rather
    than silently inheriting no coverage, which is the shape of failure the task registry exists to
    prevent (a task inheriting whatever an `else` branch did)."""
    assert sorted(TASK_FIXTURES) == sorted(task_names())
    assert len(TASK_FIXTURES) == 10


def _verifier_prompt_for(task: str, rows: list[dict]) -> str:
    """Send `rows` through the verifier `synthesize_examples` routes this task to, and return the
    prompt. Mirrors that function's own call, including where each argument comes from."""
    spec = get_task(task)
    task_context = _task_context_block(spec, rows)
    if spec.closed_label_space:
        return _captured_prompt(lambda generate: verify_generated_labels(
            [dict(rows[0])],
            task_description=f"the {task} benchmark",
            generate_fn=generate,
            label_definitions=dict(spec.label_definitions) or None,
            all_labels=sorted({
                str(row["label"]) for row in rows if row.get("label") is not None
            }) or None,
            reference_rows=rows,
            task_context=task_context,
        ))
    return _captured_prompt(lambda generate: verify_generated_answers(
        [dict(rows[0])],
        task_description=f"the {task} benchmark",
        generate_fn=generate,
        reference_rows=rows,
        task_context=task_context,
    ))


@pytest.mark.parametrize("task", sorted(TASK_FIXTURES))
def test_every_task_gets_a_verifier_prompt_carrying_its_own_vocabulary(task):
    rows, expected = TASK_FIXTURES[task]
    prompt = _verifier_prompt_for(task, rows)

    for token in expected:
        assert token in prompt, f"{task}: the verifier was not told about {token!r}"

    if not expected:
        # Nothing to enumerate, so nothing is claimed. The prompt still has to carry the request
        # itself, or the teacher is judging blind for a different reason.
        assert _task_context_block(get_task(task), rows) == ""
        assert rows[0]["text"] in prompt


@pytest.mark.parametrize("task", sorted(TASK_FIXTURES))
def test_the_task_context_block_is_carried_into_the_prompt_verbatim(task):
    """Whatever the block builds must arrive intact.

    Asserted for every task rather than only the ones with an interesting vocabulary, because the
    failure this guards against is in the plumbing — a caller that drops the keyword, or a prompt
    that interpolates it into a position where it is truncated — and plumbing breaks uniformly.
    """
    rows, _expected = TASK_FIXTURES[task]
    block = _task_context_block(get_task(task), rows)
    prompt = _verifier_prompt_for(task, rows)

    if block:
        assert block.strip() in prompt


# --------------------------------------------------------------------------
# A production gap this file deliberately does NOT paper over
# --------------------------------------------------------------------------
#
# `verify_generated_answers` drops a row whose `answer`/`response` is empty before it builds a prompt,
# and a synthesized BC5CDR row carries its gold in `entities` with no `answer` at all — so for that
# one task the teacher pass never runs, even though docs/PIPELINE.md states that it does precisely
# because the substring verifier cannot catch a MISSED entity. The fixture above therefore states the
# spans as `answer` as well, so that this file tests the vocabulary plumbing rather than accidentally
# pinning the gap as correct. Fixing it means changing `data/curriculum.py`, which is production code.


# --------------------------------------------------------------------------
# Task-level CONVENTIONS (verifier_notes)
# --------------------------------------------------------------------------


def test_a_task_with_datetime_conventions_states_them_to_the_verifier():
    """Measured on run 38832587: the teacher rejected 22.6% of generated calendar rows (1,965 of
    8,704), including "7pm start plus 60 mins is 8pm, not 20:00" — the same time written two ways —
    and six rejections of the year-rollover the gold itself uses.

    Every one of those rows had already passed `verify_calendar_row`, which re-resolves the request
    with the loader's own grammar. So the teacher was overruling an exact computation with a
    convention it had invented, which is B267/B269/B314 one level up from the tools list: a verifier
    judging without the task's context makes the context up.
    """
    block = _task_context_block(get_task("calendar_json"), [])
    assert "8 pm IS 20:00" in block, "the 12h/24h equivalence must be stated"
    assert "60 minutes" in block, "the default duration must be stated"
    assert "rolls to NEXT year" in block, "the year-rollover convention must be stated"
    assert "TITLE only" in block, "the summary convention must be stated"


def test_a_task_with_no_conventions_adds_nothing():
    """`verifier_notes=""` is an explicit choice and must cost nothing — an empty section would be
    prompt noise the teacher tries to obey."""
    assert _task_context_block(get_task("gsm8k"), []) == ""


def test_conventions_and_the_label_space_coexist():
    """They answer different questions and a task may need both, so one must not displace the
    other."""
    block = _task_context_block(get_task("clinc150"), [{"label": "transfer"}])
    assert "CLOSED label space" in block


@pytest.mark.parametrize("task", sorted(task_names()))
def test_every_task_states_whether_it_has_conventions(task):
    """No defaults on `TaskSpec`, so this is really asserting the field is reachable and a string —
    the value being empty is a decision a reader can see."""
    assert isinstance(get_task(task).verifier_notes, str)
