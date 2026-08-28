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
    "task", ["gsm8k", "dialogsum", "routerbench", "proactive_listening", "clinc150"]
)
def test_no_verifier_where_none_is_honest(task):
    """`None` is the correct answer here, not a stub, and the field has no default so the weakness
    is visible at the point the decision was made.

    Deciding whether a generated word problem's stated answer is correct requires solving it, which
    is the task itself; summarisation quality is not decidable by computation; and a classification
    row inherits a real anchor's label, so there is no answer to verify — only the phrasing, which
    is what the teacher pass checks.
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
    """The four where correctness of FORM is decidable by computation. Stated as a set so adding a
    verifier to a task where computation cannot decide correctness fails here."""
    verified = {name for name, spec in TASKS.items() if spec.synth_verifier is not None}
    assert verified == {"xlam_bfcl", "calendar_json", "ner_bc5cdr", "toolbench"}


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
