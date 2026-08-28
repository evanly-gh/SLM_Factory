"""EXACT, programmatic verifiers for generated format-bound rows (B269 / 08-17 note §8.2).

WHY THIS EXISTS
    `_synthesize_new_correct` is the one synthesis path where the teacher invents BOTH the input and
    the answer, so nothing about a real anchor constrains correctness. It had been running with no
    check at all — `curate._verifier_for` returned None for every task type, so the
    `if verify_fn is not None` branch had never executed and every batch logged `450/450 kept`. On
    `calendar_json` the teacher scores 0.2176.

    The literature is consistent that this is exactly where a verifier pays. STaR (arXiv:2203.14465):
    *"we filter the generated rationales to include only the ones which result in the correct
    answer"* — and it works because the filter is INDEPENDENT of the generator. Bansal et al.
    (ICLR 2025) go further: a generator with a 7% HIGHER false-positive rate produced better students,
    because the filter caught what mattered. That result is about noise surviving a verifier, not
    about there being no verifier.

    For these two tasks the verifier is free and exact — it is pure computation, no model involved.

TWO-STAGE ORDER (this is the part that matters)
    These run FIRST, before the model-based `verify_generated_answers` pass:

        generate → PROGRAMMATIC verify (here, exact) → MODEL verify (teacher judgement) → QC

    That ordering is deliberate. The programmatic check is exact, free, and cannot be fooled, so
    anything it rejects should never cost a teacher call. Running it first also means the model-based
    pass only ever sees rows that are at least well-formed, so its verdicts are about *semantics*
    rather than about syntax it cannot reliably judge anyway.

WHAT THESE CAN AND CANNOT CATCH
    They verify the row is WELL-FORMED and SELF-CONSISTENT: valid JSON, calls a declared function,
    passes arguments the schema actually has, dates that parse and are internally coherent. They
    cannot verify that the answer is the RIGHT answer to the request — "schedule lunch tomorrow"
    answered with a well-formed event next Tuesday passes here and is caught (if at all) by the model
    pass. That division is the point: exact checks do what computation can do, and the model is asked
    only the question that needs judgement.
"""
from __future__ import annotations

import datetime as _dt
import json
import re

# The gold convention: an event with a start but no stated duration lasts this long. Imported from
# the loader so the verifier and the data cannot drift apart.
try:  # pragma: no cover - the loader import is trivial but must not hard-fail a verifier import
    from data.loaders.calendar_json import DEFAULT_DURATION_MINUTES, FUNCTION_NAME
except Exception:  # pragma: no cover
    DEFAULT_DURATION_MINUTES = 60
    FUNCTION_NAME = "calendar.events.insert"

# `Current date and time: 2026-08-09T11:00:00 (Sunday).` — the reference instant the request's
# relative dates must be resolved against. Parsed out of the row's own prompt text.
_REFERENCE_RE = re.compile(
    r"Current date and time:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
)
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

# How far from the reference instant a generated event may fall before it is treated as a
# resolution error rather than a legitimate far-future booking. Two years each way is generous:
# the corpus is "remind me tomorrow" / "schedule X on the 3rd", not multi-year planning.
_MAX_YEARS_FROM_REFERENCE = 2


def _parse_calls(answer: object) -> list[dict] | None:
    """The row's `answer` as a list of ``{name, arguments}`` calls, or None if it is not one."""
    if isinstance(answer, list):
        calls = answer
    else:
        text = str(answer or "").strip()
        if not text:
            return None
        # Tolerate a fenced block, the way the eval-side extractor does.
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        try:
            calls = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return None
            try:
                calls = json.loads(match.group())
            except json.JSONDecodeError:
                return None
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list) or not calls:
        return None
    for call in calls:
        if not isinstance(call, dict):
            return None
        if not isinstance(call.get("name"), str) or not call["name"]:
            return None
        if not isinstance(call.get("arguments"), dict):
            return None
    return calls


def _tool_index(row: dict) -> dict[str, dict]:
    """`{function name -> its parameter schema}` for the tools this row declares."""
    tools = row.get("tools")
    if isinstance(tools, dict):
        tools = [tools]
    index: dict[str, dict] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if isinstance(name, str) and name:
            params = fn.get("parameters")
            index[name] = params if isinstance(params, dict) else {}
    return index


def verify_function_call_row(row: dict) -> tuple[bool, str]:
    """Exact well-formedness check for a generated `function_call` row.

    Five things, all decidable by computation:
      1. `answer` parses as a non-empty list of ``{name, arguments}``;
      2. the row declares at least one tool (otherwise nothing constrains the call);
      3. every called name is one of the DECLARED tools — this is the check the eval scorer also
         applies, so a row failing it is unwinnable by construction and would silently cap the
         ceiling below 1.0 (the same defect as BFCL's `simple_363`);
      4. every argument key exists in that tool's schema `properties`;
      5. every `required` parameter is present.
    """
    calls = _parse_calls(row.get("answer"))
    if calls is None:
        return False, "answer is not a JSON list of {name, arguments} calls"
    tools = _tool_index(row)
    if not tools:
        return False, "row declares no tools, so no call can be validated against a schema"
    for call in calls:
        name = call["name"]
        if name not in tools:
            return False, f"calls undeclared function {name!r} (declared: {sorted(tools)[:4]})"
        schema = tools[name]
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        if properties:
            unknown = sorted(set(call["arguments"]) - set(properties))
            if unknown:
                return False, f"{name!r} got argument(s) not in its schema: {unknown}"
        required = schema.get("required")
        if isinstance(required, list):
            absent = sorted(set(map(str, required)) - set(call["arguments"]))
            if absent:
                return False, f"{name!r} is missing required argument(s): {absent}"
    return True, "well-formed call against the declared tools"


def _iso(value: object) -> _dt.datetime | None:
    """A `{"dateTime": "..."}` block (or bare string) as a datetime, or None if malformed."""
    if isinstance(value, dict):
        value = value.get("dateTime")
    text = str(value or "").strip()
    if not _ISO_RE.match(text):
        return None
    try:
        return _dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def _expected_start(text: str) -> "_dt.datetime | None":
    """The instant the request resolves to, or None when the grammar cannot say.

    The prompt carries both halves this needs: a `Current date and time: <ISO> (<weekday>).` line and,
    after a blank line, the user's own request. Resolution is delegated to the loader's
    `resolve_datetime`, so the verifier and the gold cannot disagree about conventions — a 60-minute
    default, "tonight" meaning 20:00, a bare date meaning 09:00, and a bare ambiguous hour being
    refused outright.
    """
    reference = _REFERENCE_RE.search(text)
    if not reference:
        return None
    try:
        ref = _dt.datetime.fromisoformat(reference.group(1))
    except ValueError:
        return None
    # The request is the last paragraph; everything before it is the instruction and the reference.
    request = text.split("\n\n")[-1].strip()
    if not request:
        return None
    try:
        from data.loaders.calendar_json import resolve_datetime

        return resolve_datetime(request, ref)
    except Exception:  # noqa: BLE001 — an unresolvable request abstains rather than rejecting
        return None


def verify_calendar_row(row: dict) -> tuple[bool, str]:
    """Exact check for a generated `calendar_json` row: well-formed call PLUS coherent datetimes.

    Everything `verify_function_call_row` checks, and then the things that make a calendar event
    actually valid:
      * `start` and `end` are `{"dateTime": "YYYY-MM-DDTHH:MM:SS"}` and parse;
      * `end` is strictly AFTER `start`;
      * the duration is exactly DEFAULT_DURATION_MINUTES unless the request states one;
      * the event is within a couple of years of the request's own reference instant;
      * and — the check that makes this about the task rather than about JSON — the resolved `start`
        MATCHES what the request actually asked for, re-derived with the loader's own grammar.

    That last check is the one worth having. It is precisely the defect that made `calendar_json`
    score 0.0000 and look like a model failure: 82% of the gold answers required rolling the year
    forward, so gold and prediction differed by a year with everything else identical. A verifier
    that compares against the row's OWN stated reference instant catches a year-resolution error as a
    data defect at generation time, instead of leaving it to be discovered as a mystery zero.
    """
    ok, reason = verify_function_call_row(row)
    if not ok:
        return False, reason
    calls = _parse_calls(row.get("answer")) or []
    if len(calls) != 1:
        return False, f"expected exactly one {FUNCTION_NAME} call, got {len(calls)}"
    call = calls[0]
    if call["name"] != FUNCTION_NAME:
        return False, f"function must be {FUNCTION_NAME!r}, got {call['name']!r}"

    args = call["arguments"]
    start = _iso(args.get("start"))
    end = _iso(args.get("end"))
    if start is None:
        return False, 'start is not {"dateTime": "YYYY-MM-DDTHH:MM:SS"}'
    if end is None:
        return False, 'end is not {"dateTime": "YYYY-MM-DDTHH:MM:SS"}'
    if end <= start:
        return False, f"end {end.isoformat()} is not after start {start.isoformat()}"

    text = str(row.get("text") or "")
    minutes = (end - start).total_seconds() / 60.0
    # Only enforce the default when the request does not state a duration of its own.
    states_duration = re.search(
        r"\b(\d+\s*(?:min|minute|minutes|hour|hours|hr|hrs)|half an hour|all day)\b",
        text, re.IGNORECASE,
    )
    if not states_duration and abs(minutes - DEFAULT_DURATION_MINUTES) > 1e-6:
        return False, (
            f"duration is {minutes:.0f} min but the request states none, so the gold convention "
            f"is {DEFAULT_DURATION_MINUTES} min"
        )

    reference = _REFERENCE_RE.search(text)
    if reference:
        ref = _dt.datetime.fromisoformat(reference.group(1))
        if abs((start - ref).days) > _MAX_YEARS_FROM_REFERENCE * 366:
            return False, (
                f"start {start.date()} is more than {_MAX_YEARS_FROM_REFERENCE} years from the "
                f"request's reference date {ref.date()} — a date/year resolution error"
            )
    # THE CONTENT CHECK. Everything above is form: valid JSON, one call, end after start, a plausible
    # duration, a date within two years of the reference. None of it asks the only question this task
    # is actually about — does the resolved instant match what the request asked for?
    #
    # Without it a row whose request says "tomorrow at 3pm" and whose answer says 09:00 the next day
    # passed as a "well-formed calendar event with coherent datetimes", because it IS well-formed and
    # the datetimes ARE coherent. It is simply the wrong answer, which is the one defect that matters
    # for a task scored by exact argument match (B319).
    #
    # Re-resolving with `resolve_datetime` — the SAME grammar the loader used to build the gold — is
    # what makes this an exact verifier rather than a plausibility check: no teacher call, no
    # judgement. When the grammar cannot resolve the request it returns None and this check abstains,
    # so the verifier's own limits never become row rejections; the teacher pass still sees the row.
    expected = _expected_start(text)
    if expected is not None and expected != start:
        return False, (
            f"start {start.isoformat()} does not match the request: resolving it against the stated "
            f"reference gives {expected.isoformat()}"
        )

    summary = args.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return False, "summary is missing or empty"
    # The `convert_sgd_rows` defect, caught at generation time: the synthesised request reads
    # "Schedule <title> on <date>", and a model that copies the whole phrase puts the imperative
    # into the title.
    if summary.strip().lower().startswith(("schedule ", "remind me", "set up ", "book ")):
        return False, f"summary {summary!r} contains the request's imperative, not the event title"
    return True, "well-formed calendar event with coherent datetimes"


def verify_toolbench_row(row: dict) -> tuple[bool, str]:
    """Exact well-formedness check for a generated ToolBench solution path.

    A whole multi-step path looks unverifiable and mostly is not. Five things are decidable by
    computation, and they are the five that a teacher inventing a path gets wrong:

      1. the answer parses as a sequence of ``Action`` / ``Action Input`` turns at all;
      2. every API called is one the row's own ``tools`` DECLARES — this is the check that matters
         most here, because the callable surface is per-row and enormous (2 to 16 APIs drawn from
         ~16,000), so a plausible-sounding invented endpoint is the single easiest mistake to make
         and the scorer counts it as `undeclared_api`;
      3. every argument key exists in that API's schema, and every required one is present;
      4. the path terminates in ``Finish`` with ``return_type == "give_answer"`` and a non-empty
         ``final_answer``, so the row teaches a completed task rather than a truncated one;
      5. the path is within the API-call budget the scorer enforces, so a generated row cannot be
         unwinnable by construction.

    What it cannot check is whether the final answer is TRUE — that would require calling the APIs,
    which is the same limitation the scorer documents. So this is a form check by design, and the
    teacher pass that follows it is asked the semantic question.

    MEASURED FALSE-POSITIVE RATE: it rejects 4 of 274 real gold paths (1.5%), every one for a
    missing REQUIRED argument — e.g. `body_fat_percentage_for_fitness_calculator` called without
    `hip`/`neck`/`waist`. That is looseness in ToolBench, not a defect here: RapidAPI's own schema
    declares those parameters required and the recorded trace omitted them anyway. The check is kept
    because this verifier gates GENERATED rows, where the asymmetry favours strictness — rejecting a
    good row costs one row, while accepting a malformed one puts it in the curriculum — and because
    `verify_function_call_row` applies the same rule to xlam. The rate is recorded so a future
    reader can tell a known 1.5% from a regression.
    """
    from eval.scorers.toolbench import (
        FINISH,
        GIVE_ANSWER,
        MAX_ACTIONS,
        parse_solution_path,
    )

    path = parse_solution_path(row.get("answer"))
    if path is None:
        return False, "answer is not a sequence of Action / Action Input turns"
    tools = _tool_index(row)
    if not tools:
        return False, "row declares no tools, so no call can be validated against a schema"

    api_calls = [step for step in path["steps"] if step["name"] != FINISH]
    if len(api_calls) > MAX_ACTIONS:
        return False, (
            f"path makes {len(api_calls)} API calls, over the {MAX_ACTIONS}-call budget the "
            "scorer treats as unsolved"
        )
    for call in api_calls:
        name = call["name"]
        if name not in tools:
            return False, f"calls undeclared API {name!r} (declared: {sorted(tools)[:4]})"
        schema = tools[name]
        try:
            arguments = json.loads(call["arguments"]) if call["arguments"] else {}
        except json.JSONDecodeError:
            return False, f"{name!r} was called with an Action Input that is not valid JSON"
        if not isinstance(arguments, dict):
            return False, f"{name!r} was called with an Action Input that is not an object"
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        if properties:
            unknown = sorted(set(arguments) - set(properties))
            if unknown:
                return False, f"{name!r} got argument(s) not in its schema: {unknown}"
        required = schema.get("required")
        if isinstance(required, list):
            absent = sorted(set(map(str, required)) - set(arguments))
            if absent:
                return False, f"{name!r} is missing required argument(s): {absent}"

    if not path["actions"] or path["actions"][-1] != FINISH:
        return False, "path does not terminate in a Finish call"
    if path["return_type"] != GIVE_ANSWER:
        return False, f"path finishes with return_type {path['return_type']!r}, not {GIVE_ANSWER!r}"
    if not str(path["final_answer"]).strip():
        return False, "Finish->give_answer carries an empty final_answer"
    return True, "complete, in-budget path calling only declared APIs"


def verify_ner_row(row: dict) -> tuple[bool, str]:
    """Exact well-formedness check for a generated span-extraction row.

    Span synthesis looks unverifiable and is not. Four things are decidable by computation:

      1. `entities` is a list of ``{text, type}`` objects;
      2. every span's surface text actually APPEARS in the row's own text — this catches the
         dominant teacher error, a plausible-sounding entity that was never written down;
      3. no span is empty or whitespace;
      4. no exact ``(text, type)`` pair is repeated, since the scorer compares multisets and a
         duplicated span silently inflates the gold.

    What it cannot catch is a MISSED entity — a generated abstract that mentions three chemicals
    and labels two passes here. That is what the teacher pass is for, and it is why this returns a
    reason string rather than a bare bool: an unverifiable dimension should be visible in the log
    rather than implied by a pass.
    """
    entities = row.get("entities")
    if not isinstance(entities, list):
        return False, "entities is not a list"
    text = str(row.get("text") or "")
    if not text.strip():
        return False, "row has no text for spans to refer to"
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            return False, "an entity is not an object"
        surface = entity.get("text")
        kind = entity.get("type")
        if not isinstance(surface, str) or not surface.strip():
            return False, "an entity has an empty or non-string text"
        if not isinstance(kind, str) or not kind.strip():
            return False, f"entity {surface!r} has an empty or non-string type"
        if surface not in text:
            return False, f"span {surface!r} does not appear verbatim in the row's text"
        key = (surface, kind)
        if key in seen:
            return False, f"span {surface!r}:{kind} is listed twice"
        seen.add(key)
    return True, "every span appears verbatim in the text"


# No dispatch table here any more. Each task names its verifier directly in its spec
# (`TaskSpec.synth_verifier`), which is what this module's `_BY_BENCHMARK` was already working
# around: `calendar_json` and `xlam_bfcl` were both `function_call`, so keying on the task type
# could not tell them apart, and this table was the first place the channel abstraction visibly
# failed. `verify_function_call_row` and `verify_calendar_row` above are the two implementations.
