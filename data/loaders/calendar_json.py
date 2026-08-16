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
import json
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

CALENDAR_INSTRUCTION = (
    "Convert the user's scheduling request into a single calendar.events.insert call. "
    "Resolve every relative date and time against the current date and time given below, "
    f"and make the event {DEFAULT_DURATION_MINUTES} minutes long unless a duration is stated."
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


def _parse_time(text: str) -> tuple[int, int] | None:
    """Return ``(hour, minute)`` or None. 12-hour with meridiem, bare `H:MM`, or a daypart."""
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
    """
    text = _normalize_surface(surface)
    if not text or _UNRESOLVABLE.search(text):
        return None
    time_part = _parse_time(text)
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
        "answer": json.dumps(
            [{"name": FUNCTION_NAME, "arguments": arguments}],
            ensure_ascii=False, sort_keys=True),
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
        reference = reference_for(key)
        when = resolve_datetime(f"{date_text} {time_text}", reference)
        if when is None:
            continue
        # SGD dialogues are multi-turn; the final utterance alone ("You got it.") is not a
        # request. Restate the accumulated state as the single-shot request the task is about.
        location = first("event_location")
        request = f"Schedule {summary} on {date_text} at {time_text}"
        if location:
            request += f" at {location}"
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


def load_sgd_calendar(max_test: int = 800, log=print) -> list[dict]:
    """Load SGD `Calendar_1` AddEvent eval rows. Train split holds nearly all calendar dialogues,
    so dev and test are pulled too and everything is pooled before sampling."""
    rows: list[dict] = []
    for split, count in (("train", 127), ("dev", 20), ("test", 34)):
        rows.extend(convert_sgd_rows(_fetch_sgd_split(split, count, log=log)))
        if len(rows) >= max_test:
            break
    log(f"      [sgd] {len(rows)} usable AddEvent rows")
    return rows[:max_test]


def load_calendar_json(max_train: int = 2000, max_test: int = 800,
                       log=print) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` = (TOPv2 reminder, SGD Calendar_1) as ``function_call`` rows."""
    train = load_topv2_calendar(max_train)
    log(f"      [topv2] {len(train)} usable calendar rows")
    test = load_sgd_calendar(max_test, log=log)
    if not test:
        raise RuntimeError(
            "SGD Calendar_1 produced zero eval rows — refusing to proceed with an empty "
            "held-out set."
        )
    return train, test
