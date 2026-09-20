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


def verify_fine_ner_row(row: dict) -> tuple[bool, str]:
    """`verify_ner_row` plus TAXONOMY MEMBERSHIP, for MultiCoNER's fixed 33-class label space.

    WHY BC5CDR'S VERIFIER IS NOT ENOUGH HERE. `verify_ner_row` checks that every span appears
    verbatim in the text, which is the right check and says nothing about the TYPE. BC5CDR has two
    types and a teacher does not invent a third; MultiCoNER has 33 with names like `OtherPROD` and
    `Medication/Vaccine`, and the 2026-09-09 synthesis audit caught generated rows typed
    `OtherORG`, `Org` and `OtherPer` — none of which exist. The scorer compares types EXACTLY, so
    those rows are targets the model can only ever be marked wrong on.

    Membership is decidable, so it belongs here rather than in the teacher pass — which, on the
    same audit, was simultaneously rejecting `OtherLOC` as invalid when it is a real type. A
    computation that cannot be talked out of its verdict is the right instrument for a fixed list.
    """
    from data.loaders.multiconer import ENTITY_TYPES

    ok, reason = verify_ner_row(row)
    if not ok:
        return ok, reason
    vocabulary = set(ENTITY_TYPES)
    for entity in row.get("entities") or []:
        kind = entity.get("type") if isinstance(entity, dict) else None
        if kind not in vocabulary:
            return False, (
                f"entity type {kind!r} is not one of MultiCoNER's 33 classes; the scorer compares "
                f"types exactly, so this row can only ever score as wrong"
            )
    return True, "spans appear verbatim and every type is one of the 33 classes"


def verify_multilabel_emotion_row(row: dict) -> tuple[bool, str]:
    """Exact well-formedness check for a generated GoEmotions row.

    WHY THIS EXISTS, having first been declared unnecessary. `goemotions` originally shipped
    `synth_verifier=None` on the reasoning that whether a comment expresses `annoyance` or
    `disapproval` is a judgement rather than a computation. That is true of CORRECTNESS and false
    of the LABEL SPACE, which is a fixed list of 28 names compared exactly by the scorer.

    Run 39708679 paid for the distinction: 32 of 1,214 synthetic rows carried labels the taxonomy
    does not contain — `frustration` among them — and with no exact verifier nothing caught them.
    They went into training as targets the model can only be wrong about, because the scorer will
    never accept a name outside the 28.

    Four things are decidable here:
      1. `labels` is a non-empty list of strings;
      2. every name is one of the 28, spelled exactly — a synonym is a wrong answer;
      3. no name repeats, since the scorer compares label SETS;
      4. `label` names the SAME SET as `labels`, because the trainer targets `label` and the
         scorer grades `labels`, and a row where they disagree teaches one thing and is graded on
         another.

    (4) COMPARES SETS, NOT ORDER, and the first version of this check got that wrong. It required
    `label` to equal `", ".join(sorted(labels))`, which rejected 268 of 285 flagged rows for
    nothing but word order — the loader happens to emit sorted labels and synthesis does not, and
    `eval.scorers.multilabel_emotion.score` compares `set(pred) != set(gold)`, so order cannot
    affect a score. Only 17 of those 285 were real disagreements. A verifier that fails a quarter
    of a synthesis batch over a non-difference is the false-rejection cascade this file's own
    docstring warns about, arriving from the opposite direction.

    What it cannot check is whether those labels FIT the comment. That is the teacher's job, and
    it is why this returns a reason string rather than a bare bool.
    """
    from data.loaders.goemotions import EMOTIONS

    labels = row.get("labels")
    if not isinstance(labels, list) or not labels:
        return False, "labels is not a non-empty list"
    vocabulary = set(EMOTIONS)
    seen: set[str] = set()
    for name in labels:
        if not isinstance(name, str) or not name.strip():
            return False, "a label is empty or not a string"
        if name not in vocabulary:
            return False, f"label {name!r} is not one of the 28 GoEmotions names"
        if name in seen:
            return False, f"label {name!r} is listed twice"
        seen.add(name)
    named = {part.strip() for part in str(row.get("label") or "").split(",") if part.strip()}
    if named != set(labels):
        return False, (
            f"the trained target and the graded labels name different sets: "
            f"label={sorted(named)} against labels={sorted(labels)}"
        )
    return True, "every label is one of the 28 names, unique, and matches the trained target"


def verify_summarization_row(row: dict) -> tuple[bool, str]:
    """Exact well-formedness check for a generated DialogSum row.

    WHY THIS EXISTS. `dialogsum` shipped `synth_verifier=None` on the reasoning that whether a
    summary is accurate and complete is a judgement rather than a computation. That is true, and
    it is not the whole story: an audit of 1,161 synthetic rows from runs 39708679 / 39719569
    found three defects that ARE decidable, and nothing was catching any of them.

      1. INVENTED REFERENCES. The reference count came out `{1: 877, 2: 127, 3: 157}` — synthesis
         was fabricating two or three "human reference summaries" for rows that have exactly one
         author. Only `references[0]` is trained on, so this was mostly inert; it would stop being
         inert the moment such a row reached an eval set, where the metric takes the best of three
         and would be scoring a model against the teacher's own alternatives.

      2. THE TRAINED TARGET DISAGREEING WITH THE STATED ANSWER. 76 rows had
         `references[0] != answer`. `summarization_turn` targets `references[0]` and the scorer
         grades `references`, so a row where those disagree teaches one string and is graded on
         another.

      3. A "SUMMARY" THAT IS ACTUALLY A CONTINUATION. 18 rows carried a transcript TURN LABEL
         (`#Person1#:`) in the answer — the B250 failure, in generated training data. Training on
         those actively teaches the model to reply to the conversation instead of summarizing it,
         which is the single failure this task has already had in production.

         The COLON is the test, not the tag: real DialogSum summaries refer to the speakers by
         name — "Ms. Dawson helps #Person1# to write a memo" — in 78% of the 1,500 test
         references, while none of them contains the `#PersonN#:` turn-label form.

    What it cannot check is whether the summary is FAITHFUL to the dialogue. That is the teacher's
    job, and it is why this returns a reason string rather than a bare bool.
    """
    import re

    text = str(row.get("text") or "")
    if not text.strip():
        return False, "row has no dialogue to summarize"
    references = row.get("references")
    if not isinstance(references, list) or not references:
        return False, "references is not a non-empty list"
    if len(references) != 1:
        return False, (
            f"a generated row has {len(references)} references; it has one author, so it has one "
            f"reference. Multiple references exist only in the official test split."
        )
    target = str(references[0] or "").strip()
    if not target:
        return False, "the reference summary is empty"
    answer = str(row.get("answer") or "").strip()
    if answer != target:
        return False, (
            "the trained target and the stated answer disagree: "
            f"references[0]={target[:60]!r} against answer={answer[:60]!r}"
        )
    if re.search(r"#person\d+#\s*:", target, re.IGNORECASE):
        return False, (
            "the summary carries a transcript turn label (#PersonN#:), so it continues the "
            "conversation instead of summarizing it — naming a speaker is fine, quoting a turn "
            "is not"
        )
    if target.strip() == " ".join(text.split()):
        return False, "the summary is a copy of the dialogue"
    return True, "one reference, matching the trained target, and a summary rather than a reply"


#: TOPv2 intents that are slotless in under 5% of gold parses, so a slotless one is a defect.
#:
#: DERIVED, NOT CHOSEN. Measured over all 5,000 loaded gold rows on 2026-09-09: of the 21 intents
#: with at least 10 rows, these 9 are slotless in under 5% of their parses, `UPDATE_REMINDER` in 1
#: of 65. `HELP_REMINDER` is the clean counter-example and the reason this is a list rather than a
#: blanket rule — it is slotless in 25 of 25, because "how do I make a recurring reminder?" has no
#: arguments to label. A rule reading "a real parse has slots" would reject all 872 slotless gold
#: rows, 17.4% of the corpus.
#:
#: `tests/data/test_synth_verifiers.py` re-derives this set from the corpus and fails if it drifts,
#: so the constant cannot quietly stop describing the data it came from.
SLOT_BEARING_INTENTS = frozenset({
    "CREATE_REMINDER", "DELETE_REMINDER", "GET_REMINDER", "GET_REMINDER_DATE_TIME",
    "GET_REMINDER_LOCATION", "GET_WEATHER", "UPDATE_REMINDER", "UPDATE_REMINDER_DATE_TIME",
    "UPDATE_REMINDER_TODO",
})

#: Below this, a slotless parse is plausible even for a slot-bearing intent. The two gold rows the
#: rule would otherwise misread are both 3-word stubs — "add to reminder", "whats the weather" —
#: and this guard is what takes the rule to 0 false rejects across all 5,000 gold parses.
_SLOTLESS_MIN_WORDS = 4


def verify_semantic_parse_row(row: dict) -> tuple[bool, str]:
    """Exact well-formedness check for a generated TOPv2 parse row.

    A nested parse tree looks like the least verifiable thing in the suite and is close to the
    most, because the format carries almost all of its own correctness conditions:

      1. brackets balance, and the whole answer is ONE tree — an unbalanced or doubled tree is not
         a parse at all and would score zero at eval;
      2. the tree opens on an intent, since `[SL:...]` at the root has no intent to belong to;
      3. every label matches `IN:NAME` / `SL:NAME` in the corpus's own casing, so a plausible
         invention like `[Slot:date]` is caught rather than trained on;
      4. THE LOAD-BEARING ONE — the leaf words, concatenated in order, reproduce the utterance
         EXACTLY, ignoring whitespace. TOPv2 parses are not merely extractive but COMPLETE: every
         word of the command appears in the tree, including trailing punctuation, and nothing else
         does. Verified against the mirror: this holds for 4,000 of 4,000 test parses.

         That exactness is what makes the check worth having, because one comparison catches all
         three ways a generated parse can lie about its input — paraphrase ("set an alarm" for
         "wake me up"), omission (dropping a clause it did not know how to label), and invention
         (a span the command never contained). Each produces a row that is fluent, plausible, and
         teaches the model to hallucinate span text at eval, where exact match scores it zero.

         The comparison ignores whitespace because the parse is TOKENIZED and the utterance is
         not: "Remind Anita, Madi" has the parse leaves "Remind Anita , Madi", so a word-by-word
         match would reject every gold row carrying punctuation.

      5. an intent that always takes arguments in gold actually has some, because check 4 passes
         VACUOUSLY on a parse with no slots at all: with everything a leaf under the root, the
         leaves reproduce the utterance by construction. The 2026-09-09 audit generated
         `[IN:UPDATE_REMINDER set a reminder for 5pm ]` and this function passed it — a parse
         carrying no structure, which teaches the model to emit the utterance back unlabelled.
         See `SLOT_BEARING_INTENTS` for why this is decidable and where the threshold comes from.

    What it cannot catch is whether the labels are the RIGHT ones — whether this utterance is
    `IN:CREATE_REMINDER` or `IN:CREATE_ALARM`, and whether a span should have been `SL:TODO`. That
    is the teacher pass's job, and the reason this returns a reason string is so an unverifiable
    dimension stays visible in the log rather than being implied by a pass. The same audit shows
    why that division is worth keeping: 5 of 7 generated `UPDATE_REMINDER` rows were plain
    creations ("set a reminder for 5pm"), which is a real mislabel that no computation here can
    see, and the teacher caught every one.
    """
    import re

    parse = row.get("answer")
    if not isinstance(parse, str) or not parse.strip():
        return False, "answer is empty or not a string"
    text = str(row.get("text") or "")
    if not text.strip():
        return False, "row has no utterance for the parse to describe"

    depth = 0
    closed_at = None
    for index, char in enumerate(parse):
        if char == "[":
            if closed_at is not None:
                return False, "answer contains more than one tree"
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                return False, "brackets close before they open"
            if depth == 0:
                closed_at = index
    if depth != 0:
        return False, f"brackets do not balance (ends at depth {depth})"
    if closed_at is None:
        return False, "answer contains no bracketed tree"
    if parse.strip()[:4] != "[IN:":
        return False, "the tree does not open on an intent"

    from data.loaders.topv2 import INTENTS, SLOTS

    for label in re.findall(r"\[([^\s\]]+)", parse):
        if not re.fullmatch(r"(IN|SL):[A-Z0-9_]+", label):
            return False, f"label {label!r} is not IN:NAME or SL:NAME"
        # SHAPE IS NOT MEMBERSHIP. The pattern above admits `[SL:REMINDER_THING]`, which is
        # well-formed and not a TOPv2 slot; the scorer is exact match, so a parse naming an
        # invented label is a training target the model can only ever be marked wrong on. The
        # vocabulary is closed at 82 intents and 84 slots, so this is decidable — and it has to be
        # decided here, because the teacher demonstrably cannot: on the 2026-09-09 audit it
        # rejected `SL:DATE_TIME_NEW`, which is a real slot appearing in gold.
        kind, _, name = label.partition(":")
        if name not in (INTENTS if kind == "IN" else SLOTS):
            return False, (
                f"{label} is not in TOPv2's closed vocabulary of 82 intents and 84 slots; the "
                f"scorer compares labels exactly, so this row can only ever score as wrong"
            )

    # Leaf words are everything outside the bracket-and-label tokens, in order.
    leaves = re.sub(r"\[(?:IN|SL):[A-Z0-9_]+", " ", parse).replace("]", " ").split()
    covered = "".join(leaves)
    expected = "".join(text.split())
    if covered != expected:
        return False, (
            "the parse does not reproduce the command: a TOPv2 parse covers every word of the "
            f"utterance and adds none, but its leaves read {' '.join(leaves)!r} against the "
            f"command {text.strip()!r}"
        )

    # Check 4 cannot see the degenerate parse, because with no slots every word is a leaf and the
    # comparison above succeeds by construction. So ask it separately.
    root = re.match(r"\[IN:([A-Z0-9_]+)", parse.strip())
    if (
        root is not None
        and "[SL:" not in parse
        and root.group(1) in SLOT_BEARING_INTENTS
        and len(text.split()) >= _SLOTLESS_MIN_WORDS
    ):
        return False, (
            f"IN:{root.group(1)} takes arguments in essentially every gold parse, but this one "
            f"labels no spans and leaves the whole {len(text.split())}-word command as bare text"
        )
    return True, "one balanced tree, valid labels, leaves reproduce the command exactly"


# No dispatch table here any more. Each task names its verifier directly in its spec
# (`TaskSpec.synth_verifier`), which is what this module's `_BY_BENCHMARK` was already working
# around: `calendar_json` and `xlam_bfcl` were both `function_call`, so keying on the task type
# could not tell them apart, and this table was the first place the channel abstraction visibly
# failed. `verify_function_call_row` and `verify_calendar_row` above are the two implementations.
