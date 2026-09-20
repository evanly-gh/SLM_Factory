"""Exact programmatic verifiers for generated format-bound rows.

These are the free, exact checks that STaR and Bansal et al. show are what actually make a mediocre
generator usable. They run BEFORE the model-based pass, so a malformed row never costs a teacher call.

They verify WELL-FORMEDNESS and SELF-CONSISTENCY, not semantic correctness — a well-formed event on
the wrong day passes here and is left to the model pass. That division is deliberate.
"""
import json

import pytest

from data.synth_verifiers import (
    verify_calendar_row,
    verify_function_call_row,
    verify_ner_row,
)
from tasks import TASKS, get_task

TOOL = {
    "name": "calculate_triangle_area",
    "parameters": {
        "type": "dict",
        "required": ["base", "height"],
        "properties": {
            "base": {"type": "integer"},
            "height": {"type": "integer"},
            "unit": {"type": "string"},
        },
    },
}


def _fc_row(answer, tools=None):
    return {"text": "Area of a triangle, base 10 height 5.",
            "answer": answer if isinstance(answer, str) else json.dumps(answer),
            "tools": TOOL if tools is None else tools, "label": "function_call"}


# --------------------------------------------------------------------------
# function_call
# --------------------------------------------------------------------------

def test_well_formed_call_passes():
    ok, why = verify_function_call_row(_fc_row(
        [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}]))
    assert ok, why


def test_optional_argument_is_allowed():
    ok, _ = verify_function_call_row(_fc_row(
        [{"name": "calculate_triangle_area",
          "arguments": {"base": 10, "height": 5, "unit": "cm"}}]))
    assert ok


def test_fenced_json_is_tolerated():
    """The eval-side extractor tolerates fences, so the verifier must too — otherwise it would
    reject rows the scorer would happily grade."""
    ok, _ = verify_function_call_row(_fc_row(
        '```json\n[{"name": "calculate_triangle_area", "arguments": {"base": 1, "height": 2}}]\n```'))
    assert ok


def test_unparseable_answer_is_rejected():
    ok, why = verify_function_call_row(_fc_row("not json at all"))
    assert not ok and "not a JSON list" in why


def test_undeclared_function_is_rejected():
    """BFCL's `simple_363` shape: gold calls a name the row never declared. The eval scorer rejects
    such a call, so the row is unwinnable and would silently cap the ceiling below 1.0."""
    ok, why = verify_function_call_row(_fc_row(
        [{"name": "find_closest", "arguments": {"base": 1, "height": 2}}]))
    assert not ok and "undeclared function" in why


def test_argument_not_in_schema_is_rejected():
    ok, why = verify_function_call_row(_fc_row(
        [{"name": "calculate_triangle_area",
          "arguments": {"base": 1, "height": 2, "colour": "red"}}]))
    assert not ok and "not in its schema" in why


def test_missing_required_argument_is_rejected():
    ok, why = verify_function_call_row(_fc_row(
        [{"name": "calculate_triangle_area", "arguments": {"base": 1}}]))
    assert not ok and "missing required" in why


def test_row_without_tools_is_rejected():
    """Nothing constrains a call with no declared tools, so it cannot be verified at all. This is
    why the generator now PINS `tools` from the anchor instead of hoping the teacher reproduces it."""
    ok, why = verify_function_call_row(_fc_row(
        [{"name": "anything", "arguments": {}}], tools=[]))
    assert not ok and "no tools" in why


# --------------------------------------------------------------------------
# calendar_json
# --------------------------------------------------------------------------

CAL_TOOL = {
    "name": "calendar.events.insert",
    "parameters": {
        "type": "dict",
        "required": ["summary", "start", "end"],
        "properties": {
            "summary": {"type": "string"}, "start": {"type": "dict"},
            "end": {"type": "dict"}, "location": {"type": "string"},
        },
    },
}


def _cal_row(args, reference="2026-03-01T09:00:00", extra_text=""):
    text = (f"Convert the user's scheduling request into a single calendar.events.insert call.\n"
            f"Current date and time: {reference} (Sunday).\n\n"
            f'Add "Dentist" to my calendar on March 3rd at 10:00 am{extra_text}')
    return {"text": text, "tools": [CAL_TOOL], "label": "function_call",
            "answer": json.dumps([{"name": "calendar.events.insert", "arguments": args}])}


def _args(start="2026-03-03T10:00:00", end="2026-03-03T11:00:00", summary="Dentist"):
    return {"summary": summary, "start": {"dateTime": start}, "end": {"dateTime": end}}


def test_coherent_event_passes():
    ok, why = verify_calendar_row(_cal_row(_args()))
    assert ok, why


def test_end_before_start_is_rejected():
    ok, why = verify_calendar_row(_cal_row(
        _args(start="2026-03-03T11:00:00", end="2026-03-03T10:00:00")))
    assert not ok and "not after start" in why


def test_malformed_datetime_is_rejected():
    ok, why = verify_calendar_row(_cal_row(_args(start="March 3rd 10am")))
    assert not ok and "dateTime" in why


def test_wrong_default_duration_is_rejected():
    """The gold convention is 60 minutes when the request states none, and it is stated in the
    prompt — so a 30-minute event for a request with no duration is a real error."""
    ok, why = verify_calendar_row(_cal_row(_args(end="2026-03-03T10:30:00")))
    assert not ok and "60 min" in why


def test_stated_duration_overrides_the_default():
    row = _cal_row(_args(end="2026-03-03T12:00:00"), extra_text=" for 2 hours")
    ok, why = verify_calendar_row(row)
    assert ok, why


def test_year_resolution_error_is_caught():
    """THE defect that made calendar_json score 0.0000. The event is a year off the request's own
    reference instant with everything else identical — previously indistinguishable from a model
    failure, now caught at generation time as a data defect."""
    ok, why = verify_calendar_row(_cal_row(
        _args(start="2029-03-03T10:00:00", end="2029-03-03T11:00:00")))
    assert not ok and "resolution error" in why


def test_imperative_leaked_into_summary_is_caught():
    """`convert_sgd_rows` built the request as "Schedule <title> on <date>", so a model copying the
    phrase put the imperative into the title."""
    ok, why = verify_calendar_row(_cal_row(_args(summary="Schedule Food")))
    assert not ok and "imperative" in why


def test_empty_summary_is_rejected():
    ok, why = verify_calendar_row(_cal_row(_args(summary="   ")))
    assert not ok and "summary" in why


def test_multiple_calls_are_rejected():
    row = _cal_row(_args())
    row["answer"] = json.dumps([
        {"name": "calendar.events.insert", "arguments": _args()},
        {"name": "calendar.events.insert", "arguments": _args()},
    ])
    ok, why = verify_calendar_row(row)
    assert not ok and "exactly one" in why


# --------------------------------------------------------------------------
# ner_bc5cdr
# --------------------------------------------------------------------------
#
# Span synthesis looks unverifiable and is not. What it CANNOT catch is a MISSED entity — a generated
# abstract that mentions three chemicals and labels two passes here — which is why the teacher pass
# still runs and why these return a reason string rather than a bare bool: an unverifiable dimension
# should be visible in the log rather than implied by a pass.

NER_TEXT = "Aspirin therapy induced gastritis and mild nephritis in the cohort."


def _ner_row(entities, text=NER_TEXT):
    return {"text": text, "entities": entities}


def test_spans_that_appear_verbatim_pass():
    ok, why = verify_ner_row(_ner_row([
        {"text": "Aspirin", "type": "Chemical"},
        {"text": "gastritis", "type": "Disease"},
    ]))
    assert ok, why


def test_an_empty_span_list_is_a_legitimate_row():
    """A negative example — an abstract with no entities — is real training signal, and the eval
    prompt explicitly asks for `[]` in that case."""
    ok, why = verify_ner_row(_ner_row([]))
    assert ok, why


def test_a_span_that_does_not_appear_in_the_text_is_rejected():
    """THE defect this verifier exists to catch: the dominant teacher error is a plausible-sounding
    entity that was never actually written down. It is free to check and unfoolable."""
    ok, why = verify_ner_row(_ner_row([{"text": "Ibuprofen", "type": "Chemical"}]))
    assert not ok and "does not appear verbatim" in why


def test_a_case_mismatch_is_rejected_because_the_scorer_matches_exactly():
    """The scorer compares exact (surface, type) pairs, so a span the model can never reproduce
    character-for-character is an unwinnable row."""
    ok, why = verify_ner_row(_ner_row([{"text": "aspirin", "type": "Chemical"}]))
    assert not ok and "verbatim" in why


def test_a_duplicated_span_is_rejected():
    """The scorer compares multisets, so a repeated `(text, type)` pair silently inflates the gold
    and caps the achievable F1 below 1.0."""
    ok, why = verify_ner_row(_ner_row([
        {"text": "Aspirin", "type": "Chemical"},
        {"text": "Aspirin", "type": "Chemical"},
    ]))
    assert not ok and "listed twice" in why


def test_the_same_surface_under_two_types_is_allowed():
    """Not a duplicate: the pair is what the scorer compares, and an ambiguous surface form under
    two types is a real annotation."""
    ok, why = verify_ner_row(_ner_row([
        {"text": "Aspirin", "type": "Chemical"},
        {"text": "Aspirin", "type": "Disease"},
    ]))
    assert ok, why


def test_an_empty_span_or_type_is_rejected():
    ok, why = verify_ner_row(_ner_row([{"text": "   ", "type": "Chemical"}]))
    assert not ok and "empty" in why
    ok, why = verify_ner_row(_ner_row([{"text": "Aspirin", "type": ""}]))
    assert not ok and "empty" in why


def test_a_row_with_no_text_is_rejected():
    """There is nothing for the spans to refer to, so nothing can be verified at all."""
    ok, why = verify_ner_row(_ner_row([{"text": "Aspirin", "type": "Chemical"}], text="  "))
    assert not ok and "no text" in why


def test_entities_that_are_not_a_list_are_rejected():
    ok, why = verify_ner_row({"text": NER_TEXT, "entities": "Aspirin"})
    assert not ok and "not a list" in why
    ok, why = verify_ner_row(_ner_row(["Aspirin"]))
    assert not ok and "not an object" in why


# --------------------------------------------------------------------------
# Which verifier each task names
# --------------------------------------------------------------------------
#
# The verifier used to be chosen by a `(task_type, benchmark)` dispatch table in
# `data/synth_verifiers.py`, one of the five hand-maintained side registries the task specs
# replaced. It is now `TaskSpec.synth_verifier`, so a task cannot inherit another's check by
# sharing a channel — xlam and calendar were both `function_call` and calendar's conventions
# (the 60-minute default, the reference instant) are not xlam's.

def test_the_calendar_task_names_the_calendar_verifier():
    """Behaviourally, not by identity: calendar's verifier must reject a row that is a perfectly
    well-formed function call but breaks the calendar convention. The schema check alone accepts it."""
    row = _cal_row(_args(end="2026-03-03T10:30:00"))
    assert verify_function_call_row(row)[0], "the row is a well-formed call by schema alone"
    assert get_task("calendar_json").synth_verifier(row) is False


def test_the_xlam_task_names_the_schema_verifier():
    good = _fc_row([{"name": "calculate_triangle_area", "arguments": {"base": 1, "height": 2}}])
    assert get_task("xlam_bfcl").synth_verifier(good) is True
    assert get_task("xlam_bfcl").synth_verifier(_fc_row("garbage")) is False


def test_the_ner_task_names_the_span_verifier():
    """Behaviourally: it must reject a span that is not in the row's own text."""
    verifier = get_task("ner_bc5cdr").synth_verifier
    assert verifier is not None
    assert verifier(_ner_row([{"text": "Aspirin", "type": "Chemical"}])) is True
    assert verifier(_ner_row([{"text": "Ibuprofen", "type": "Chemical"}])) is False


@pytest.mark.parametrize(
    "task", ["gsm8k", "routerbench", "proactive_listening", "clinc150"]
)
def test_no_verifier_where_none_is_honest(task):
    """`None` is the correct answer here, not a stub, and the field has no default so the weakness
    is visible at the point the decision was made.

    Deciding whether a generated word problem's stated answer is correct requires solving it, which
    is the task itself; and a classification row inherits a real anchor's label, so there is no
    answer to verify — only the phrasing, which is what the teacher pass checks.

    `dialogsum` WAS on this list until 2026-09-08, on the reasoning that summarization quality is
    not decidable by computation. That reasoning was right about QUALITY and wrong as a conclusion:
    an audit of 1,161 generated rows found three decidable defects going uncaught — fabricated
    extra "human references" (the count came out {1: 877, 2: 127, 3: 157} for rows with one
    author), 76 rows whose trained target disagreed with their stated answer, and 18 whose
    "summary" carried a transcript turn label, which is the B250 continuation failure in training
    data. See `verify_summarization_row`.

    The distinction worth keeping is between "no answer to verify" — genuinely the case for the
    four tasks above — and "the interesting property is not decidable", which does not imply that
    nothing about the row is.
    """
    assert get_task(task).synth_verifier is None


TOOLBENCH_TOOLS = [{
    "name": "get_forecast_for_weather_api",
    "parameters": {"properties": {"city": "string"}, "required": ["city"], "optional": []},
}]


def _tb_row(answer):
    return {"text": "forecast for Lisbon?", "query": "forecast for Lisbon?",
            "answer": answer, "tools": TOOLBENCH_TOOLS}


def _tb_path(action="get_forecast_for_weather_api", args='{"city": "Lisbon"}',
             return_type="give_answer", final="Warm and clear."):
    return (
        f"Thought: I will look it up.\nAction: {action}\nAction Input: {args}\n"
        f"Thought: I can answer.\nAction: Finish\n"
        f'Action Input: {{"return_type": "{return_type}", "final_answer": "{final}"}}'
    )


def test_the_toolbench_task_names_the_path_verifier():
    """Behaviourally: the verifier must reject an invented API even though the path is perfectly
    well-formed text.

    This is the check that earns its keep on this task. The callable surface is per row and drawn
    from ~16,000 real endpoints, so a teacher asked to invent a solution path will produce a
    plausible-sounding endpoint that does not exist far more often than it will produce malformed
    JSON — and the scorer counts that as `undeclared_api`, so the row would be unwinnable.
    """
    verifier = get_task("toolbench").synth_verifier
    assert verifier is not None
    assert verifier(_tb_row(_tb_path())) is True
    assert verifier(_tb_row(_tb_path(action="get_forecast_for_made_up_api"))) is False
    # Complete and declared, but the path gave up rather than answering.
    assert verifier(_tb_row(_tb_path(return_type="give_up_and_restart"))) is False
    # An argument the schema does not have.
    assert verifier(_tb_row(_tb_path(args='{"town": "Lisbon"}'))) is False
    # Truncated: no terminating Finish at all.
    assert verifier(_tb_row(
        "Thought: looking.\nAction: get_forecast_for_weather_api\n"
        'Action Input: {"city": "Lisbon"}'
    )) is False


def test_exactly_the_format_bound_tasks_have_an_exact_verifier():
    """The tasks where correctness of FORM is decidable by computation. Stated as a set so adding a
    verifier to a task where computation cannot decide correctness fails here.

    `multiconer` and `topv2` joined on 2026-09-06 and both earn it. MultiCoNER shares
    `verify_ner_row` with BC5CDR because the row shape and the decidable properties are identical:
    spans must appear verbatim in the text. TOPv2's parse carries almost all of its own
    correctness conditions — balanced brackets, one tree rooted at an intent, labels matching
    IN:NAME/SL:NAME, and leaves that reproduce the command exactly.

    The other two new tasks correctly have NONE. Whether a grammatical correction is right, and
    whether a comment expresses `annoyance` or `disapproval`, are judgements rather than
    computations — so there is nothing free to check and the teacher pass is the only gate.
    """
    verified = {name for name, spec in TASKS.items() if spec.synth_verifier is not None}
    assert verified == {
        "xlam_bfcl", "calendar_json", "ner_bc5cdr", "toolbench", "multiconer", "topv2",
        # Added 2026-09-08. Correctness here is a judgement, but the LABEL SPACE is a fixed list
        # of 28 names the scorer compares exactly — and run 39708679 put 32 invented labels into
        # training because nothing checked it.
        "goemotions",
        # Added 2026-09-08 for the same reason: faithfulness is a judgement, but fabricated
        # reference counts, a trained target disagreeing with the stated answer, and a "summary"
        # that is actually a transcript continuation are all decidable — and an audit found all
        # three in generated rows.
        "dialogsum",
    }


def test_there_is_no_verifier_dispatch_table_left():
    """`programmatic_verifier_for` and `_BY_BENCHMARK` were the first place the channel abstraction
    visibly failed: `calendar_json` and `xlam_bfcl` were both `function_call`, so keying on the task
    type could not tell them apart, and the table was added to work around it."""
    import data.synth_verifiers as verifiers

    for gone in ("programmatic_verifier_for", "_BY_BENCHMARK"):
        assert not hasattr(verifiers, gone), f"{gone} is back"


def test_a_verifier_returns_a_plain_bool_for_synthesize_examples():
    """`_synthesize_new_correct` calls `verify_fn(row)` in a boolean context, so a verifier that
    returned the `(ok, reason)` tuple the checkers return would accept every row — a non-empty
    tuple is always truthy."""
    for spec in TASKS.values():
        if spec.synth_verifier is None:
            continue
        verdict = spec.synth_verifier(_fc_row("garbage"))
        assert verdict is False, f"{spec.name} verifier returned {verdict!r}, not a bool"


def test_the_fine_ner_verifier_checks_taxonomy_membership_not_just_spans():
    """THE GAP THIS CLOSES, from the 2026-09-09 synthesis audit.

    `multiconer` originally borrowed `verify_ner_row` from BC5CDR, which checks that every span
    appears verbatim in the text and says nothing about the TYPE. That is sufficient for two
    guessable types and not for 33: the audit caught generated rows typed `OtherORG`, `Org` and
    `OtherPer`, none of which exist. The scorer compares types exactly, so those rows are targets
    the model can only ever be marked wrong on.

    The teacher was NOT a usable substitute — on the same audit it rejected `OtherLOC`, which is a
    real type. A computation that cannot be talked out of its verdict is the right instrument for
    a fixed list, which is why membership moved here.
    """
    from data.synth_verifiers import verify_fine_ner_row

    def row(entity_type):
        return {"text": "acme ships", "entities": [{"text": "acme", "type": entity_type}]}

    assert verify_fine_ner_row(row("ORG"))[0] is True
    # A real type the teacher wrongly called invalid must still pass.
    assert verify_fine_ner_row(row("OtherLOC"))[0] is True
    for invented in ("OtherORG", "Org", "OtherPer", "PERSON"):
        ok, reason = verify_fine_ner_row(row(invented))
        assert not ok, invented
        assert "33 classes" in reason

    # And it still enforces the span check it inherited.
    assert verify_fine_ner_row(
        {"text": "acme ships", "entities": [{"text": "zzz", "type": "ORG"}]}
    )[0] is False


def test_the_topv2_verifier_notes_tell_the_teacher_that_nesting_is_legitimate():
    """A teacher not told a convention invents one, and this is the third time in this suite.

    TOPv2 genuinely nests: a slot may contain a whole intent, and 21.5% of the corpus's `reminder`
    parses contain more than one. The notes originally illustrated only a flat
    `[IN:X words [SL:Y words ] ]` form, so on the 2026-09-09 audit the teacher rejected correctly
    nested rows with "slots must be flat" and "slots must not contain intents" — 3 of 10 sampled
    rejections, every one a false positive.

    Asserted on the NOTES rather than on a rejection rate, because the rate is a property of the
    teacher and the notes are the thing under our control.
    """
    from tasks import get_task

    notes = get_task("topv2").verifier_notes
    assert "NESTED INTENTS ARE LEGITIMATE" in notes
    # A real nested gold example, not a description of one.
    assert "[SL:RECURRING_DATE_TIME [IN:GET_RECURRING_DATE_TIME" in notes
    assert "never claim slots must be flat" in notes


def test_the_multiconer_notes_tell_the_teacher_the_listed_types_are_all_valid():
    """The enumeration itself lives in `entity_type_vocabulary`, which is asserted separately.

    What belongs in the notes is the INSTRUCTION about that list, because enumerating the 33 types
    was necessary and not sufficient: the teacher has to be told the list is exhaustive and that a
    name on it may not be rejected. On the 2026-09-09 audit it rejected `OtherLOC`, a real type.
    """
    from tasks import get_task

    notes = get_task("multiconer").verifier_notes
    assert "EXHAUSTIVE" in notes
    assert "never reject a row for using one" in notes
    # Exactness still has to be stated, or the opposite error returns: accepting near-miss synonyms.
    assert "OtherPER and not PERSON" in notes
    assert "{types}" not in notes, "an unsubstituted placeholder reached the teacher"


def test_the_teacher_is_told_the_taxonomy_the_task_defines_not_the_one_the_sample_shows():
    """A sample shows what a type space CONTAINS; it is never evidence of where it ENDS.

    `_task_context_block` used to collect the entity types appearing in the rows it was about to
    show the teacher and label them "the only ones that may appear". That holds for BC5CDR, whose
    two types both appear in any sample, and fails for MultiCoNER: a 40-row anchor sample contains
    5 of its 33, so the sentence declared 28 real types invalid. On the 2026-09-09 synthesis audit
    the teacher acted on it and rejected correctly-typed `OtherLOC` rows as invalid.

    The generalisation that matters is the one asserted first: the emitted list must equal the
    task's declared vocabulary for ANY sample, including an empty one and a one-type one.
    """
    from data.curriculum import _task_context_block
    from tasks import get_task

    spec = get_task("multiconer")
    assert len(spec.entity_type_vocabulary) == 33

    one_type_sample = [{"text": "acme ships", "entities": [{"text": "acme", "type": "ORG"}]}]
    for sample in ([], one_type_sample):
        block = _task_context_block(spec, sample)
        missing = [t for t in spec.entity_type_vocabulary if t not in block]
        assert not missing, f"sample of {len(sample)} row(s) hid real types: {missing}"

    # And the sample cannot ADD a type the task does not define.
    invented = _task_context_block(
        spec, [{"text": "x", "entities": [{"text": "x", "type": "NotAType"}]}]
    )
    assert "NotAType" not in invented

    # BC5CDR, the case the old behaviour was written for, is unchanged.
    bc5cdr = _task_context_block(get_task("ner_bc5cdr"), [])
    assert "Chemical" in bc5cdr and "Disease" in bc5cdr

    # A task that extracts no spans says nothing at all about entity types.
    assert "may appear" not in _task_context_block(get_task("topv2"), [])


def test_a_slotless_parse_is_rejected_only_for_intents_that_always_take_arguments():
    """Check 4 of this verifier passes VACUOUSLY when a parse has no slots.

    With everything a leaf under the root intent, "the leaves reproduce the utterance" is true by
    construction. So the 2026-09-09 audit's `[IN:UPDATE_REMINDER set a reminder for 5pm ]` — a
    parse carrying no structure at all — passed, and would have taught the model to echo the
    command back unlabelled.

    The rule has to be intent-conditional, which is the whole reason it is a measured list: 872 of
    5,000 gold parses are slotless and legitimate, `HELP_REMINDER` in 25 of 25 of its rows.
    """
    from data.synth_verifiers import verify_semantic_parse_row

    degenerate = {
        "text": "set a reminder for 5pm",
        "answer": "[IN:UPDATE_REMINDER set a reminder for 5pm ]",
    }
    ok, reason = verify_semantic_parse_row(degenerate)
    assert not ok and "labels no spans" in reason

    # The same utterance WITH structure is fine.
    assert verify_semantic_parse_row({
        "text": "set a reminder for 5pm",
        "answer": "[IN:CREATE_REMINDER set a reminder for [SL:DATE_TIME 5pm ] ]",
    })[0] is True

    # An argument-free intent stays slotless without complaint, however long the utterance.
    assert verify_semantic_parse_row({
        "text": "How do I make a reminder for recurring events ?",
        "answer": "[IN:HELP_REMINDER How do I make a reminder for recurring events ? ]",
    })[0] is True

    # And a short stub is left alone even for a slot-bearing intent.
    assert verify_semantic_parse_row({
        "text": "add to reminder", "answer": "[IN:UPDATE_REMINDER add to reminder ]",
    })[0] is True


def test_the_slot_bearing_intent_set_still_describes_the_corpus_it_was_derived_from():
    """The constant is a measurement, so it has to be re-measurable.

    Pinning a derived set by hand invites it to drift from the data silently — the loader's
    sampling could change and leave the list describing a corpus that no longer exists. This
    re-derives it and, separately, asserts the property the list is FOR: zero false rejects on
    gold. That second check is the one that matters, because it is what makes the rule safe to run
    in enforce mode.
    """
    import re

    from data.loaders.topv2 import load_topv2
    from data.synth_verifiers import SLOT_BEARING_INTENTS, verify_semantic_parse_row

    loaded = load_topv2()
    rows = list(loaded[0] if isinstance(loaded, tuple) else loaded["train"])
    counts: dict[str, list[int]] = {}
    for row in rows:
        match = re.match(r"\[IN:([A-Z0-9_]+)", str(row["answer"]))
        if not match:
            continue
        seen = counts.setdefault(match.group(1), [0, 0])
        seen[0] += 1
        seen[1] += "[SL:" not in str(row["answer"])
    derived = {
        name for name, (total, slotless) in counts.items()
        if total >= 10 and slotless / total < 0.05
    }
    assert derived == set(SLOT_BEARING_INTENTS), (
        f"the measured set has drifted; missing={derived - set(SLOT_BEARING_INTENTS)}, "
        f"stale={set(SLOT_BEARING_INTENTS) - derived}"
    )

    rejected = [r for r in rows if not verify_semantic_parse_row(r)[0]]
    assert not rejected, f"{len(rejected)} GOLD rows now fail their own exact verifier"


def test_topv2_labels_are_checked_for_membership_not_just_for_shape():
    """`[SL:REMINDER_THING ...]` matches `SL:NAME` perfectly and is not a TOPv2 slot.

    The regex this verifier used checked SHAPE, which admits any capitalised invention. TOPv2's
    vocabulary is closed at 82 intents and 84 slots and the scorer is exact match, so a parse
    naming an invented label is a training target the model can only ever be marked wrong on.

    Deciding it here rather than in the teacher pass is not a preference. On the 2026-09-09 audit
    the teacher rejected `SL:DATE_TIME_NEW` as "not a valid TOPv2 label"; it is real and appears in
    gold. A closed list is exactly the thing a computation should own.
    """
    from data.synth_verifiers import verify_semantic_parse_row

    real = "[IN:UPDATE_REMINDER_DATE_TIME change [SL:TODO PTA ] to [SL:DATE_TIME_NEW 7 pm ] ]"
    assert verify_semantic_parse_row({"text": "change PTA to 7 pm", "answer": real})[0] is True

    for invented in (
        "[IN:UPDATE_REMINDER_DATE_TIME change [SL:REMINDER_THING PTA ] to [SL:DATE_TIME_NEW 7 pm ] ]",
        "[IN:DO_THING change [SL:TODO PTA ] to [SL:DATE_TIME_NEW 7 pm ] ]",
    ):
        ok, reason = verify_semantic_parse_row({"text": "change PTA to 7 pm", "answer": invented})
        assert not ok and "closed vocabulary" in reason, invented


def test_the_topv2_verifier_notes_enumerate_the_closed_vocabulary():
    """The teacher cannot recall 166 names, and guessing them is what produced the DATE_TIME_NEW
    false positive. Same failure and same fix as MultiCoNER's 33 entity types."""
    from data.loaders.topv2 import INTENTS, SLOTS
    from tasks import get_task

    notes = get_task("topv2").verifier_notes
    missing = [n for n in (*INTENTS, *SLOTS) if n not in notes]
    assert not missing, f"labels absent from topv2's verifier notes: {missing[:5]}"
    assert "if it reached you, it is in the vocabulary" in notes


def test_the_create_versus_update_convention_reaches_both_the_generator_and_the_verifier():
    """One statement, two consumers — which is only true since `task_context` reached generation.

    CREATE vs UPDATE was the largest single source of bad topv2 rows on the 2026-09-09 audit:
    synthesis emitted "set a reminder for 5pm" as `UPDATE_REMINDER` and the teacher, correctly,
    rejected it. Gold is unambiguous — over the reminder domain, `remind` opens a CREATE in 103 of
    105 rows and `set` in 29 of 29, while `change` is UPDATE in 35 of 35.

    Deliberately NOT an exact-verifier rule. Which intent a command deserves is the unverifiable
    dimension this task's verifier docstring reserves for the teacher, and a first-word heuristic
    would be a guess dressed as a computation. Telling both parties the convention is the honest
    instrument.
    """
    from data.curriculum import _new_example_prompt, _task_context_block
    from tasks import get_task

    notes = get_task("topv2").verifier_notes
    assert "CREATE VERSUS UPDATE" in notes
    assert "`set a reminder for 5pm` is CREATE_REMINDER, not UPDATE_REMINDER" in notes

    # The generator sees it too, via the same block.
    context = _task_context_block(get_task("topv2"), [])
    prompt = _new_example_prompt(
        {"text": "wake me up at six", "answer": "[IN:CREATE_ALARM ...]"},
        "parse commands", task_context=context,
    )
    assert "CREATE VERSUS UPDATE" in prompt
