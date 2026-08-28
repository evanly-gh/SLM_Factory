"""Calendar NL→JSON loader (2026-08-13).

The task: turn a free-text scheduling request ("Schedule a meeting at 5pm tomorrow") into a
Google Calendar API v3 ``events.insert`` request body the host application can post without
further parsing. Modelled as ``task_type=function_call`` so it reuses
``eval/scorers/function_call.py`` verbatim — including the ``format_valid`` / ``content_correct``
split, which is exactly the "can it emit JSON at all" vs "are the fields right" distinction that
matters for a task whose output feeds an API.

Train and eval come from **independently constructed corpora**, mirroring the `xlam_bfcl` design
so the eval is a transfer test rather than a held-out slice:

- **train** — TOPv2 `reminder` domain (``WillHeld/top_v2``, parquet, 10,353 `CREATE_REMINDER`
  rows). Hierarchical bracket parses with `TODO` / `DATE_TIME` slots.
- **eval**  — Schema-Guided Dialogue `Calendar_1` `AddEvent` frames (Google's own calendar
  schema: `event_name`, `event_date`, `event_time`, `event_location`), read as raw JSON from the
  `google-research-datasets/dstc8-schema-guided-dialogue` GitHub repo. The HuggingFace mirror is
  script-based and therefore dead under ``datasets>=4``.

**Relative dates are resolved, not passed through.** Both corpora annotate the surface string
("at 5 pm", "March 6th"), which is only meaningful against a reference instant. Each row pins a
reference datetime, states it in the prompt, and the gold carries a fully-resolved ISO-8601
timestamp. That is the only version of this task whose output is directly usable, and it is what
makes it non-trivial for a small model.

**Gold correctness is enforced by refusal.** ``resolve_datetime`` implements a deliberately
strict grammar and returns None for anything it does not fully understand — relative-to-event
offsets ("15 minutes before"), vague spans ("this week", "next month"), and timezone-qualified
expressions ("4pm Pacific Time") are all rejected, and their rows are dropped. A smaller corpus
of certainly-correct gold beats a larger one containing rows no model can win.
"""
from __future__ import annotations

import hashlib
import os
import json
import random
import re
from collections.abc import Iterable
from datetime import datetime, timedelta

TOPV2_ID = "WillHeld/top_v2"
SGD_RAW_BASE = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    "dstc8-schema-guided-dialogue/master"
)

FUNCTION_NAME = "calendar.events.insert"

# Default event length when the request states a start but no duration. Stated in the prompt so
# the model is being tested on extraction, not on guessing a convention.
DEFAULT_DURATION_MINUTES = 60

# The tool signature shown to the model. Field names and nesting are the real Google Calendar
# API v3 Events resource, so a passing prediction is a postable request body.
CALENDAR_TOOL = {
    "name": FUNCTION_NAME,
    "description": (
        "Create an event on the user's primary Google Calendar. "
        "Times must be resolved to ISO-8601 (YYYY-MM-DDTHH:MM:SS) against the current "
        "date and time given in the request."
    ),
    "parameters": {
        "type": "dict",
        "required": ["summary", "start", "end"],
        "properties": {
            "summary": {"type": "string", "description": "Title of the event."},
            "start": {
                "type": "dict",
                "description": 'Start, as {"dateTime": "YYYY-MM-DDTHH:MM:SS"}.',
            },
            "end": {
                "type": "dict",
                "description": (
                    'End, as {"dateTime": "YYYY-MM-DDTHH:MM:SS"}. '
                    f"Default to {DEFAULT_DURATION_MINUTES} minutes after start when the "
                    "request gives no duration."
                ),
            },
            "location": {
                "type": "string",
                "description": "Physical location or address. Omit when not stated.",
            },
        },
    },
}

# Every convention the gold labels use, stated in the prompt.
#
# The default HOUR was missing until 2026-08-21, and the omission made most of the task unanswerable.
# 45% of gold rows resolve to 09:00 because the request names a day and no clock time ("remind me to
# pack my lunch tomorrow") — but 09:00 is our convention, not a fact about the request, and nothing
# told the model. The duration convention was stated; this one was not.
#
# The cost was measured. On the pooled eval set the teacher scored 0.3850, and the synthesis gate read
# that as an incapable teacher, when really it was being asked to guess an unstated house rule on
# nearly half the rows (B322). A fine-tuned student can infer it from thousands of examples; a
# zero-shot or five-shot teacher cannot, and neither can a reader deciding whether the gold is right.
#
# The rule of thumb this encodes: if the label generator applies a convention, the prompt has to name
# it, or the task is scoring telepathy.
CALENDAR_INSTRUCTION = (
    "Convert the user's scheduling request into a single calendar.events.insert call. "
    "Resolve every relative date and time against the current date and time given below, "
    f"and make the event {DEFAULT_DURATION_MINUTES} minutes long unless a duration is stated. "
    "When the request gives a day but no clock time, start the event at 09:00; "
    "when it says tonight, start it at 20:00."
)

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3,
    "fri": 4, "sat": 5, "sun": 6,
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sept": 9, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# Named parts of the day, mapped to the hour a calendar app would actually use.
_DAYPARTS = {"morning": 9, "afternoon": 14, "evening": 19, "night": 20, "noon": 12, "midnight": 0}

# Expressions this grammar refuses outright. Each is either relative to an unstated event
# ("15 minutes before" — before WHAT?), a span rather than an instant ("this week"), or carries
# a timezone we have no basis to convert ("Pacific Time"). Rows containing them are dropped.
_UNRESOLVABLE = re.compile(
    r"\b(before|after|between|until|till|by the end|sometime|whenever|"
    r"time|timezone|pacific|eastern|central|mountain|gmt|utc|est|pst|cst)\b"
    r"|\bthis (week|month|year|weekend)\b"
    r"|\bnext (week|month|year|weekend)\b"
    r"|\blast\b|\bevery\b|\beach\b|\bdaily\b|\bweekly\b|\bmonthly\b",
    re.IGNORECASE,
)

_TIME_RE = re.compile(
    r"\b(?P<hour>\d{1,2})\s*(?::\s*(?P<minute>\d{2}))?\s*"
    r"(?P<mer>a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
_TIME_24_RE = re.compile(r"\b(?P<hour>\d{1,2})\s*:\s*(?P<minute>\d{2})\b")

# A BARE clock hour: "at 5", "at 6 tonight". No meridiem, no colon, so `_TIME_RE` and `_TIME_24_RE`
# both miss it — and before 2026-08-21 that silently became "no time was stated", which took the
# 09:00 bare-date default. The result was 123 gold rows whose answer contradicted the utterance
# ("...on Tuesday at 6" labelled 09:00) and a curriculum where 51% of all gold started at exactly
# 09:00, teaching the model to ignore stated times (B318).
#
# Anchored on "at" specifically so it cannot swallow a DATE number: "on the 15th", "March 3",
# "3/15" have their own patterns and must not be read as an hour.
_BARE_HOUR_RE = re.compile(r"\bat\s+(?P<hour>\d{1,2})\b(?!\s*:)(?!\s*(?:a\.?m|p\.?m))", re.I)

# Dayparts that disambiguate a bare hour. "at 7 tonight" is not ambiguous — nobody means 07:00 — so
# the daypart supplies the meridiem the speaker left out.
_PM_CONTEXT = re.compile(r"\b(tonight|this evening|evening|afternoon|night|pm)\b", re.I)
_AM_CONTEXT = re.compile(r"\b(this morning|morning|am)\b", re.I)
_ORDINAL_RE = re.compile(r"\b(?P<day>\d{1,2})\s*(?:st|nd|rd|th)\b", re.IGNORECASE)
_MONTH_DAY_RE = re.compile(
    r"\b(?P<month>" + "|".join(_MONTHS) + r")\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_DAY_MONTH_RE = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+of\s+(?P<month>" + "|".join(_MONTHS) + r")\b",
    re.IGNORECASE,
)


def _normalize_surface(text: str) -> str:
    """TOPv2 tokenizes punctuation apart (``at 8 : 30 am``, ``5 p.m .``). Re-join it."""
    text = str(text or "").strip().lower()
    text = re.sub(r"\s*:\s*", ":", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"([ap])\.\s*m\.?", r"\1m", text)
    return re.sub(r"\s+", " ", text).strip(" .,")


def _next_weekday(reference: datetime, weekday: int, *, allow_today: bool) -> datetime:
    delta = (weekday - reference.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return (reference + timedelta(days=delta)).replace(
        hour=0, minute=0, second=0, microsecond=0)


# Returned when the text clearly states a clock time that this grammar cannot pin down. Distinct from
# None, which means "no clock time was stated at all" and legitimately takes the bare-date default.
# Collapsing the two is what produced B318.
AMBIGUOUS_TIME = "ambiguous"


def _parse_time(text: str) -> tuple[int, int] | str | None:
    """Return ``(hour, minute)``, ``AMBIGUOUS_TIME``, or None.

    Three outcomes, not two:
      * ``(hour, minute)`` — understood: 12-hour with meridiem, bare ``H:MM``, a daypart, or a bare
        hour that an adjacent daypart disambiguates ("at 7 tonight" is 19:00, not 07:00);
      * ``AMBIGUOUS_TIME`` — a clock time IS stated but cannot be resolved ("at 5" with nothing to
        say whether that is morning or afternoon). The caller drops the row;
      * ``None`` — no clock time at all ("remind me tomorrow"), which takes the 09:00 default.
    """
    match = _TIME_RE.search(text)
    if match:
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        if not (1 <= hour <= 12) or minute > 59:
            return None
        meridiem = match.group("mer").replace(".", "")
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        return hour, minute
    match = _TIME_24_RE.search(text)
    if match:
        hour, minute = int(match.group("hour")), int(match.group("minute"))
        if hour > 23 or minute > 59:
            return None
        return hour, minute
    # A bare hour, disambiguated by an adjacent daypart if one is present. Checked BEFORE the
    # daypart-only branch, or "at 7 tonight" would return the generic 20:00 for "night" and throw
    # away the stated 7.
    bare = _BARE_HOUR_RE.search(text)
    if bare:
        hour = int(bare.group("hour"))
        if 0 <= hour <= 23:
            if _PM_CONTEXT.search(text):
                return (hour + 12 if hour < 12 else hour), 0
            if _AM_CONTEXT.search(text):
                return (0 if hour == 12 else hour), 0
            if hour > 12:
                return hour, 0            # 24-hour reading is the only one available
            # "at 5" with no morning/evening cue. A calendar app guesses; a GOLD LABEL must not —
            # this module's contract is that anything not understood with certainty is dropped.
            return AMBIGUOUS_TIME
    for word, hour in _DAYPARTS.items():
        if re.search(rf"\b{word}\b", text):
            return hour, 0
    return None


def _parse_date(text: str, reference: datetime) -> datetime | None:
    """Return midnight of the referenced day, or None if no date is expressed."""
    today = reference.replace(hour=0, minute=0, second=0, microsecond=0)
    if re.search(r"\btomorrow\b", text):
        return today + timedelta(days=1)
    if re.search(r"\b(today|tonight)\b", text):
        return today
    match = _DAY_MONTH_RE.search(text) or _MONTH_DAY_RE.search(text)
    if match:
        month = _MONTHS[match.group("month").lower()]
        day = int(match.group("day"))
        year = reference.year
        try:
            candidate = reference.replace(
                year=year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            return None
        if candidate < today:
            try:
                candidate = candidate.replace(year=year + 1)
            except ValueError:
                return None
        return candidate
    for name, weekday in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", text):
            # The next occurrence, never today: you do not say "on Monday" to mean four hours
            # ago. "next Monday" and a bare "Monday" resolve identically, which matches ordinary
            # usage from midweek and avoids inventing a distinction the gold cannot justify.
            return _next_weekday(reference, weekday, allow_today=False)
    match = _ORDINAL_RE.search(text)
    if match:
        day = int(match.group("day"))
        if not 1 <= day <= 31:
            return None
        candidate = today
        for _ in range(13):
            try:
                candidate = candidate.replace(day=day)
            except ValueError:
                candidate = (candidate.replace(day=1) + timedelta(days=32)).replace(day=1)
                continue
            if candidate >= today:
                return candidate
            candidate = (candidate.replace(day=1) + timedelta(days=32)).replace(day=1)
        return None
    return None


def resolve_datetime(surface: str, reference: datetime) -> datetime | None:
    """Resolve a temporal surface string to an absolute datetime, or None to reject the row.

    Refusal is the point: this builds **gold labels**, so anything not understood with certainty
    must be dropped rather than guessed. A row is only accepted when at least one of date or time
    is explicit; a bare date defaults to 09:00 and a bare time to the reference day (rolling to
    tomorrow if that instant has already passed, which is how a calendar app behaves).

    A time that is STATED but ambiguous ("at 5", with nothing to say which 5) is refused rather than
    defaulted. Treating it as "no time given" is what made 123 rows contradict their own utterance and
    pushed 51% of all gold to exactly 09:00 (B318).
    """
    text = _normalize_surface(surface)
    if not text or _UNRESOLVABLE.search(text):
        return None
    time_part = _parse_time(text)
    if time_part is AMBIGUOUS_TIME:
        # A stated-but-unresolvable time. Dropping is the same policy `_UNRESOLVABLE` applies to
        # "before what?" and "this week": the row cannot be labelled with certainty, so it is not
        # labelled at all.
        return None
    date_part = _parse_date(text, reference)
    if time_part is None and date_part is None:
        return None
    if date_part is None:
        hour, minute = time_part
        candidate = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= reference:
            candidate += timedelta(days=1)
        return candidate
    if time_part is None:
        # A date with no time. "tonight" is the one case that carries its own hour.
        hour = _DAYPARTS["night"] if re.search(r"\btonight\b", text) else 9
        return date_part.replace(hour=hour, minute=0, second=0, microsecond=0)
    hour, minute = time_part
    return date_part.replace(hour=hour, minute=minute, second=0, microsecond=0)


def reference_for(key: str) -> datetime:
    """A stable, per-row reference instant.

    Derived from a hash of the row key so it is reproducible across runs and machines, but varies
    across rows — a single fixed "today" would let the model memorise one date instead of learning
    to do the arithmetic. Constrained to 2026 and to 08:00–17:00 on the hour, so "tomorrow at
    9am" is never ambiguous about which side of midnight the reference sits on.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    day_offset = digest[0] << 8 | digest[1]
    hour = 8 + (digest[2] % 10)
    return datetime(2026, 1, 1, hour, 0, 0) + timedelta(days=day_offset % 365)


def _reference_before_event(key: str, date_text: str, time_text: str) -> datetime | None:
    """A per-row reference instant placed 1-21 days BEFORE the event, so no year rollforward is
    needed and the year is inferable from the prompt.

    Two passes: resolve the date against a neutral probe reference to find out WHEN the event is,
    then place the real reference a hashed 1-21 days earlier and hand that back. Returns None when
    the date cannot be resolved at all, so the caller drops the row rather than guessing.
    """
    probe = datetime(2026, 1, 1, 9, 0, 0)
    when = resolve_datetime(f"{date_text} {time_text}", probe)
    if when is None:
        return None
    digest = hashlib.sha256(f"ref:{key}".encode("utf-8")).digest()
    days_before = 1 + (digest[0] % 21)
    hour = 8 + (digest[1] % 10)
    reference = (when - timedelta(days=days_before)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )
    # Guard the boundary: the reference must be strictly before the event, or the rollforward this
    # exists to prevent would fire anyway.
    if reference >= when:
        reference = when - timedelta(days=1)
    return reference


def build_calendar_row(
    utterance: str, summary: str, when: datetime, *,
    location: str | None = None, key: str = "",
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
) -> dict:
    """Assemble one ``function_call`` row with a resolved ISO-8601 events.insert gold."""
    arguments = {
        "summary": summary,
        "start": {"dateTime": when.strftime("%Y-%m-%dT%H:%M:%S")},
        "end": {"dateTime": (when + timedelta(minutes=duration_minutes)).strftime(
            "%Y-%m-%dT%H:%M:%S")},
    }
    if location:
        arguments["location"] = location
    reference = when  # placeholder; overwritten by callers that own the reference
    return {
        "text": utterance,
            # `name` FIRST, matching the prompt. `sort_keys=True` put it LAST (alphabetically
            # after "arguments"), which contradicted the instruction the model is given —
            # `{"name": ..., "arguments": {...}}` — and cost calendar_json most of its score:
            # `eval/scorers/function_call._parse_calls` REQUIRES `name`, so a prediction that got
            # the whole nested datetime block right and then fumbled the trailing field parsed as
            # None and counted as a FORMAT failure rather than a wrong answer. Measured on run
            # 38832587: content tracked format_valid at ~0.8 and both swung 1.0000 -> 0.0019 -> 1.0000
            # across iterations, with correct-looking output that would not parse. Emitting the
            # discriminative field first makes the long argument block unable to cost the row.
        "answer": json.dumps(
            [{"name": FUNCTION_NAME, "arguments": arguments}],
            ensure_ascii=False),
        "tools": [CALENDAR_TOOL],
        "label": "function_call",
        "_calendar_key": key,
        "_reference": reference.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _with_reference(utterance: str, reference: datetime) -> str:
    """Prepend the reference instant to the request, so the prompt is self-contained.

    Both the trainer and the eval harness build prompts from ``text`` through the same
    ``function_call`` builder, so stating the reference here keeps train/serve parity by
    construction — the class of bug B250 was.
    """
    stamp = reference.strftime("%Y-%m-%dT%H:%M:%S")
    return (
        f"{CALENDAR_INSTRUCTION}\n"
        f"Current date and time: {stamp} ({reference.strftime('%A')}).\n\n"
        f"{utterance.strip()}"
    )


# --------------------------------------------------------------------------------------
# TOPv2 (train side)
# --------------------------------------------------------------------------------------

def _topv2_slot(parse: str, slot: str) -> str | None:
    """Return the text of the first *leaf* ``[SL:<slot> ...]`` (no nested intent), or None."""
    marker = f"[SL:{slot} "
    start = parse.find(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    out: list[str] = []
    while index < len(parse) and depth:
        char = parse[index]
        if char == "[":
            return None  # nested intent — not a plain leaf, refuse it
        if char == "]":
            depth -= 1
            if not depth:
                break
        else:
            out.append(char)
        index += 1
    text = "".join(out).strip()
    return text or None


def convert_topv2_rows(rows: Iterable[dict]) -> list[dict]:
    """Map TOPv2 `CREATE_REMINDER` parses to calendar rows, dropping anything unresolvable.

    Rows are refused when: the parse is not a CREATE_REMINDER, the TODO or DATE_TIME slot is
    absent or nested, or the temporal expression does not resolve under the strict grammar.
    """
    out: list[dict] = []
    for row in rows:
        parse = str(row.get("semantic_parse") or "")
        utterance = str(row.get("utterance") or "").strip()
        if not utterance or not parse.startswith("[IN:CREATE_REMINDER"):
            continue
        if "RECURRING_DATE_TIME" in parse:
            continue  # recurrence needs an RRULE; out of scope for v1
        todo = _topv2_slot(parse, "TODO")
        when_text = _topv2_slot(parse, "DATE_TIME")
        if not todo or not when_text:
            continue
        key = f"topv2:{utterance}"
        reference = reference_for(key)
        when = resolve_datetime(when_text, reference)
        if when is None:
            continue
        summary = re.sub(r"\s+([',.])", r"\1", todo).strip()
        if not summary:
            continue
        row_out = build_calendar_row(
            _with_reference(utterance, reference), summary, when, key=key)
        row_out["_reference"] = reference.strftime("%Y-%m-%dT%H:%M:%S")
        out.append(row_out)
    return out


def load_topv2_calendar(max_train: int = 2000) -> list[dict]:
    """Load and convert the TOPv2 `reminder` domain. Parquet-native, no loading script."""
    from datasets import load_dataset

    raw = load_dataset(TOPV2_ID, split="train")
    reminder = raw.filter(lambda r: r["domain"] == "reminder")
    return convert_topv2_rows(reminder)[:max_train]


# --------------------------------------------------------------------------------------
# Schema-Guided Dialogue Calendar_1 (eval side)
# --------------------------------------------------------------------------------------

def convert_sgd_rows(dialogues: Iterable[dict]) -> list[dict]:
    """Extract `AddEvent` turns from SGD `Calendar_1` frames and shape them as calendar rows.

    SGD carries dialogue state cumulatively, so the LAST user turn whose active intent is
    AddEvent holds the complete slot set. Only that turn is emitted, and only when the frame has
    both a name and a resolvable date/time — a partially-specified turn mid-dialogue is not a
    well-posed single-shot request.
    """
    out: list[dict] = []
    for dialogue in dialogues:
        services = dialogue.get("services") or []
        if not any(str(s).startswith("Calendar") for s in services):
            continue
        best: tuple[str, dict] | None = None
        for turn in dialogue.get("turns") or []:
            if turn.get("speaker") != "USER":
                continue
            for frame in turn.get("frames") or []:
                if not str(frame.get("service", "")).startswith("Calendar"):
                    continue
                state = frame.get("state") or {}
                if state.get("active_intent") != "AddEvent":
                    continue
                slots = state.get("slot_values") or {}
                if slots:
                    best = (str(turn.get("utterance") or ""), slots)
        if best is None:
            continue
        utterance, slots = best

        def first(name: str) -> str | None:
            values = slots.get(name) or []
            return str(values[0]).strip() if values else None

        summary = first("event_name")
        date_text = first("event_date")
        time_text = first("event_time")
        if not summary or not date_text or not time_text:
            continue
        key = f"sgd:{dialogue.get('dialogue_id')}"
        # Anchor the reference instant just BEFORE the event, instead of scattering it across the
        # year (B-cal-year). SGD's calendar dialogues are almost all set in March, while
        # `reference_for` drew uniformly from 2026 — so for most rows the reference fell AFTER the
        # event's month, `_parse_date`'s "if it already passed, roll to next year" rule fired, and
        # the gold landed in 2027. That made **82% of the eval set** depend on a year-rollforward the
        # prompt never states: the model answered 2026-03-02 for "on 2nd of March", which is the more
        # natural reading, and was marked wrong. It scored 0.0000 for a convention, not for the task.
        #
        # Resolving against a reference 1-21 days earlier means no rollforward is ever needed, so the
        # year is unambiguous from the prompt and the row tests date ARITHMETIC rather than
        # convention-guessing. The offset still varies per row, so a fixed "today" cannot be
        # memorised.
        reference = _reference_before_event(key, date_text, time_text)
        if reference is None:
            continue
        when = resolve_datetime(f"{date_text} {time_text}", reference)
        if when is None:
            continue
        # SGD dialogues are multi-turn; the final utterance alone ("You got it.") is not a
        # request. Restate the accumulated state as the single-shot request the task is about.
        #
        # The title is QUOTED. It used to read `Schedule {summary} on {date}`, so a row whose event
        # was called `Food` produced "Schedule Food on March 1st" and the model reasonably extracted
        # `summary="Schedule Food"` — the word added to make it a sentence became part of the thing
        # being extracted. Quoting makes the span unambiguous without changing the task.
        location = first("event_location")
        request = f'Add "{summary}" to my calendar on {date_text} at {time_text}'
        if location:
            request += f", at {location}"
        row = build_calendar_row(
            _with_reference(request, reference), summary, when,
            location=location, key=key)
        row["_reference"] = reference.strftime("%Y-%m-%dT%H:%M:%S")
        row["_source_utterance"] = utterance
        out.append(row)
    return out


def _fetch_sgd_split(split: str, max_files: int = 128, log=print) -> list[dict]:
    """Download SGD dialogue files for one split. The HF mirror is script-based (dead under
    datasets>=4), so this reads the canonical JSON straight from the source repo."""
    import concurrent.futures as futures
    import urllib.error
    import urllib.request

    def grab(index: int):
        url = f"{SGD_RAW_BASE}/{split}/dialogues_{index:03d}.json"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError):
            return None

    dialogues: list[dict] = []
    with futures.ThreadPoolExecutor(16) as pool:
        for chunk in pool.map(grab, range(1, max_files + 1)):
            if chunk:
                dialogues.extend(chunk)
    log(f"      [sgd] {split}: {len(dialogues)} dialogues")
    return dialogues


SGD_LOCAL_BUNDLE = os.path.join("data", "local", "calendar_sgd")


def _read_local_sgd(log=print) -> list[dict] | None:
    """The vendored SGD Calendar dialogues, or None when the bundle is absent.

    Preferred over the network path. The loader used to fetch these from
    raw.githubusercontent.com AT LOAD TIME and never cache them, which meant one upstream commit
    silently changed the eval set (so past scores stopped being comparable and were not
    reproducible) and the run could not start without network access. `data/local/calendar_sgd/`
    holds the 1,602 Calendar-service dialogues with a manifest and a sha256, the same treatment
    `bc5cdr` and `proactive_listening` get. Override with SLM_CALENDAR_SGD_DIR.
    """
    base = os.environ.get("SLM_CALENDAR_SGD_DIR") or SGD_LOCAL_BUNDLE
    path = os.path.join(base, "dialogues.jsonl")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        dialogues = [json.loads(line) for line in fh if line.strip()]
    log(f"      [sgd] {len(dialogues)} Calendar dialogue(s) from the frozen bundle at {base}")
    return dialogues


def load_sgd_calendar(max_test: int = 800, log=print) -> list[dict]:
    """Load SGD `Calendar_1` AddEvent eval rows.

    Reads the vendored bundle when present; otherwise falls back to the network fetch (train split
    holds nearly all calendar dialogues, so dev and test are pulled too and pooled before sampling).
    """
    local = _read_local_sgd(log=log)
    if local is not None:
        rows = convert_sgd_rows(local)
    else:
        log("      [sgd] frozen bundle not found — falling back to a LIVE fetch from GitHub. "
            "Scores from this run are not reproducible against an upstream change; see "
            "data/local/calendar_sgd/manifest.json.")
        rows = []
        for split, count in (("train", 127), ("dev", 20), ("test", 34)):
            rows.extend(convert_sgd_rows(_fetch_sgd_split(split, count, log=log)))
            if len(rows) >= max_test:
                break
    log(f"      [sgd] {len(rows)} usable AddEvent rows")
    return rows[:max_test]


# Share of the combined pool held out for evaluation.
#
# WHY BOTH SPLITS ARE DRAWN FROM ONE POOL, AND NOT "TRAIN=TOPv2, EVAL=SGD"
#     That is what this function used to return, and it made the task unscoreable. The two corpora
#     describe the same activity with structurally different requests:
#
#       TOPv2   "Remind me to pack my lunch for tomorrow."
#               no quoted title, a relative date, no clock time, and NO location — 0.0% of 4,156 rows
#               carry a `location` argument, and 50% resolve to the 09:00 bare-date default.
#       SGD     'Add "Chris Webby concert" to my calendar on March 13th at 12:30 pm, at 2367
#               Shattuck Avenue' — quoted title, explicit date, explicit time, and a location in
#               100% of 478 rows.
#
#     Training on the first and scoring on the second asks the model for a required argument it has
#     never once seen. The metric is exact argument match, so every prediction misses `location` and
#     the score is pinned near zero however capable the model is: run 38732020 measured 0.0084 with
#     format_valid 1.0000 — flawless JSON, 474 of 478 wrong (B321).
#
#     Interleaving SGD into the training pool while leaving the eval set pure SGD is NOT enough, and
#     was tried first: TOPv2 outnumbers SGD nine to one, so train came out 5% location-bearing against
#     an eval that is 100%. The mismatch shrinks but does not go away.
#
#     So the two corpora are pooled and then split. Both halves are random samples of the same
#     distribution, which is the only arrangement under which the metric measures the model rather
#     than the gap between two datasets. Unlike xlam_bfcl — where BFCL is a published leaderboard and
#     must stay the untouched eval — neither corpus here is a canonical benchmark, so there is nothing
#     that has to be preserved whole.
EVAL_FRACTION = 0.12


def load_calendar_json(max_train: int = 2000, max_test: int = 800,
                       log=print) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)``, both random samples of the same pooled TOPv2 + SGD distribution.

    Disjoint by construction, and the eval firewall in `curate` independently blocks any train row
    whose text matches a held-out one, so a leak would have to survive both.
    """
    # Both loaders treat their argument as a hard slice, and 0 means "none" rather than "all", so
    # ask for more than either corpus holds instead of passing a sentinel.
    _ALL = 1_000_000
    topv2 = load_topv2_calendar(_ALL)
    log(f"      [topv2] {len(topv2)} usable calendar rows")
    sgd = load_sgd_calendar(_ALL, log=log)
    if not sgd:
        raise RuntimeError(
            "SGD Calendar_1 produced zero rows — refusing to proceed without the fully-specified "
            "request form, which is the half of this task that carries a location."
        )

    # Deterministic: a fixed seed over a stable sort, so the same rows land in the same half on every
    # machine and across resumes. A checkpointed run that re-derived a different split would score
    # against rows it had already trained on.
    # Deduplicate by request text BEFORE splitting. TOPv2 repeats utterances verbatim across rows, so
    # splitting the raw pool put 29 identical requests on both sides — a leak the eval firewall would
    # later catch at curation time, but which should never be produced here in the first place. The
    # first occurrence of each text wins, under the stable sort below, so which row survives is
    # deterministic rather than dependent on corpus order.
    seen: set[str] = set()
    deduped = []
    for row in sorted(topv2 + sgd,
                      key=lambda r: str(r.get("_calendar_key") or r.get("text") or "")):
        key = " ".join(str(row.get("text") or "").split()).lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)
    pool = deduped
    random.Random(20260821).shuffle(pool)

    # The BOUNDARY is fixed by EVAL_FRACTION alone and never by the caller's arguments. `max_test`
    # truncates what is RETURNED, and must not move the split.
    #
    # Letting it move the boundary is an eval leak: `_reread_known_sources` re-reads this loader with
    # `max_test=60`, which would put the eval boundary at 60 and hand rows 60..535 — real held-out rows
    # of the run's frozen eval set — back as training candidates. The eval firewall in `curate` blocks
    # them, so nothing contaminated the curriculum, but mining would spend its whole yield on rows
    # destined for rejection while reporting them as novel.
    n_eval = max(1, int(len(pool) * EVAL_FRACTION))
    test, train = pool[:n_eval], pool[n_eval:]
    if max_test:
        test = test[:max_test]
    if max_train:
        train = train[:max_train]

    def _share(rows, predicate) -> str:
        return f"{sum(1 for r in rows if predicate(r)) / max(len(rows), 1):.0%}"

    has_location = lambda r: '"location"' in str(r.get("answer") or "")
    log(f"      [calendar] pooled {len(pool)} rows -> train {len(train)} / eval {len(test)}; "
        f"location-bearing {_share(train, has_location)} train vs {_share(test, has_location)} eval "
        f"— the two halves must match, or the metric measures the split rather than the model")
    return train, test
