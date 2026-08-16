"""xLAM / BFCL function-calling loader (2026-08-01; BFCL path rewritten 2026-08-13).

Training data comes from Salesforce's xlam-function-calling-60k (query + tool signatures +
gold calls); the held-out eval comes from BFCL (Berkeley Function-Calling Leaderboard). Both
are shaped into the ``function_call`` row schema ``{text, answer, tools}`` where ``answer`` is a
canonical JSON string of the gold call list ``[{"name", "arguments"}]`` — exactly what
``eval/scorers/function_call.py`` parses.

**BFCL is not a `load_dataset`-able repo (B253).** It ships 52 category-named JSON files
(``BFCL_v3_simple.json``, ``BFCL_v3_parallel.json``, …) that match no split-name pattern, so
``load_dataset(BFCL_ID, ...)`` raises ``DataFilesNotFoundError``. Worse, prompts and gold answers
live in *separate* files joined on ``id``, and each gold argument is a **list of acceptable
values** rather than one value, with ``""`` in that list meaning "this argument may be omitted".
A naive load would therefore have produced an empty eval set even if it had resolved.

So the BFCL side names its files explicitly, downloads them through ``hf_hub_download`` (cached
under ``HF_HOME``, handles the redirect and auth), joins the two files on ``id``, and carries the
acceptable-value map through on a ``_accept`` metadata field. The scorer consults ``_accept``
when present and falls back to plain equality when absent, so xLAM rows are unaffected. The
leading underscore keeps it out of the schema shown to the synthesis teacher, same convention as
``_instruction`` on the summarization rows.

The pure converters (``convert_xlam_rows``, ``convert_bfcl_rows``) are unit-tested on in-memory
samples; ``load_xlam_bfcl`` runs the live pulls on the cluster.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

XLAM_ID = "Salesforce/xlam-function-calling-60k"
BFCL_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# The BFCL v3 categories whose gold is a plain AST call list, i.e. the ones this scorer can
# actually grade. Deliberately excluded: `irrelevance`/`live_relevance` (gold is "no call", and
# they ship no possible_answer file), `java`/`javascript`/`sql`/`rest` (non-Python call syntax),
# `exec_*` (requires executing real APIs), and `multi_turn_*` (stateful, multi-step).
BFCL_AST_CATEGORIES = (
    "BFCL_v3_simple",
    "BFCL_v3_multiple",
    "BFCL_v3_parallel",
    "BFCL_v3_parallel_multiple",
    "BFCL_v3_live_simple",
    "BFCL_v3_live_multiple",
)

# BFCL marks an argument optional by including the empty string among its acceptable values.
_BFCL_OMITTABLE = ""


def _as_json_obj(value):
    """Return ``value`` parsed from JSON if it is a string, else ``value`` unchanged."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


def _normalize_calls(answers) -> list[dict] | None:
    """Coerce a gold-answer payload into ``[{"name", "arguments"}]`` or None if unusable."""
    parsed = _as_json_obj(answers)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return None
    calls = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        args = item.get("arguments")
        if args is None:
            args = item.get("args") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        calls.append({"name": name, "arguments": args})
    return calls


def convert_xlam_rows(dataset: Iterable[dict]) -> list[dict]:
    """Map raw xLAM rows to ``{text, answer, tools}`` with a canonical gold-call JSON.

    Accepts the xLAM field names (``query``/``tools``/``answers``) and tolerates rows that
    already use ``text``/``answer``. Rows without a usable gold call list are dropped.
    """
    out: list[dict] = []
    for ex in dataset:
        text = str(ex.get("query") or ex.get("text") or "").strip()
        if not text:
            continue
        calls = _normalize_calls(ex.get("answers", ex.get("answer")))
        if calls is None:
            continue
        tools = ex.get("tools")
        tools_obj = _as_json_obj(tools)
        out.append({
            "text": text,
            "answer": json.dumps(calls, ensure_ascii=False, sort_keys=True),
            "tools": tools_obj if tools_obj is not None else tools,
            "label": "function_call",
        })
    return out


def _bfcl_question_text(question) -> str:
    """Flatten BFCL's ``question`` (a list of turn-lists of chat messages) to a prompt string.

    The AST categories are single-turn, but a handful of `live_*` rows prepend a system message.
    Those are kept — dropping them would hide constraints the gold answer depends on — and only
    role-prefixed when there is more than one message, so the common case stays a bare utterance.
    """
    turns = question
    while isinstance(turns, list) and turns and isinstance(turns[0], list):
        turns = turns[0]
    if isinstance(turns, str):
        return turns.strip()
    if not isinstance(turns, list):
        return ""
    messages = [m for m in turns if isinstance(m, dict) and str(m.get("content") or "").strip()]
    if not messages:
        return ""
    if len(messages) == 1:
        return str(messages[0].get("content")).strip()
    return "\n".join(
        f"{str(m.get('role') or 'user').strip()}: {str(m.get('content')).strip()}"
        for m in messages
    ).strip()


def _canonical_from_accept(accept_calls: list[dict]) -> list[dict] | None:
    """Collapse BFCL's acceptable-value map into one representative gold call list.

    Each BFCL gold call is ``{func_name: {arg: [acceptable, ...]}}``. The representative picks
    the first acceptable value per argument and omits arguments whose only acceptable value is
    "may be omitted". This is what gets written to ``answer`` — used for display, for training
    targets, and as the fallback when ``_accept`` is unavailable. Grading uses ``_accept``.
    """
    calls: list[dict] = []
    for entry in accept_calls:
        if not isinstance(entry, dict) or len(entry) != 1:
            return None
        (name, arg_map), = entry.items()
        if not isinstance(name, str) or not isinstance(arg_map, dict):
            return None
        arguments = {}
        for arg, accepted in arg_map.items():
            values = accepted if isinstance(accepted, list) else [accepted]
            concrete = [v for v in values if v != _BFCL_OMITTABLE]
            if not concrete:
                # Only "" is acceptable → the argument is meant to be absent.
                continue
            arguments[arg] = concrete[0]
        calls.append({"name": name, "arguments": arguments})
    return calls


def convert_bfcl_rows(prompts: Iterable[dict], answers: Iterable[dict]) -> list[dict]:
    """Join BFCL prompt rows to their ``possible_answer`` rows on ``id`` and shape them.

    ``prompts`` rows carry ``{id, question, function}``; ``answers`` rows carry
    ``{id, ground_truth}``. Rows that cannot be joined, or whose gold does not parse, are
    dropped rather than guessed at — a silently mis-shaped eval row is worse than a smaller
    eval set.
    """
    gold_by_id = {
        str(a.get("id")): a.get("ground_truth")
        for a in answers
        if isinstance(a, dict) and a.get("id") is not None
    }
    out: list[dict] = []
    for ex in prompts:
        if not isinstance(ex, dict):
            continue
        row_id = str(ex.get("id"))
        accept = gold_by_id.get(row_id)
        if not isinstance(accept, list) or not accept:
            continue
        text = _bfcl_question_text(ex.get("question"))
        if not text:
            continue
        canonical = _canonical_from_accept(accept)
        if canonical is None:
            continue
        tools = ex.get("function")
        if isinstance(tools, dict):
            tools = [tools]
        if not isinstance(tools, list) or not tools:
            continue
        # Integrity check: BFCL has a small number of rows whose gold calls a function name that
        # is not among the declared tools (e.g. `simple_363` declares
        # `restaurant_search.find_closest` but its ground truth calls `find_closest`). The scorer
        # rejects any call outside the allowed set, so such a row is unscoreable by construction
        # and would silently depress the ceiling below 1.0. Drop it rather than grade against it.
        declared = {
            t.get("name") for t in tools
            if isinstance(t, dict) and isinstance(t.get("name"), str)
        }
        if any(call["name"] not in declared for call in canonical):
            continue
        out.append({
            "text": text,
            "answer": json.dumps(canonical, ensure_ascii=False, sort_keys=True),
            "tools": tools,
            "label": "function_call",
            "_accept": accept,
            "_bfcl_id": row_id,
        })
    return out


def _read_bfcl_jsonl(filename: str) -> list[dict]:
    """Download one BFCL file and parse it as JSON Lines.

    The files carry a ``.json`` extension but are line-delimited, and they are not resolvable
    through ``load_dataset`` (see the module docstring). ``hf_hub_download`` caches under
    ``HF_HOME`` and follows the LFS redirect that a bare ``requests.get`` does not.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=BFCL_ID, filename=filename, repo_type="dataset")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def load_bfcl(max_test: int = 800, categories: Iterable[str] = BFCL_AST_CATEGORIES,
              log=print) -> list[dict]:
    """Return up to ``max_test`` BFCL AST-gradable eval rows, drawn across categories.

    Categories are interleaved round-robin rather than concatenated, so a truncated eval set
    still spans simple / multiple / parallel rather than being 800 rows of `simple`.
    """
    per_category: list[list[dict]] = []
    for category in categories:
        try:
            prompts = _read_bfcl_jsonl(f"{category}.json")
            answers = _read_bfcl_jsonl(f"possible_answer/{category}.json")
        except Exception as exc:  # noqa: BLE001 — one missing category must not kill the run
            log(f"      [bfcl] skipping {category}: {type(exc).__name__}: {exc}")
            continue
        rows = convert_bfcl_rows(prompts, answers)
        log(f"      [bfcl] {category}: {len(rows)} usable of {len(prompts)} prompts")
        if rows:
            per_category.append(rows)

    out: list[dict] = []
    index = 0
    while len(out) < max_test and any(index < len(rows) for rows in per_category):
        for rows in per_category:
            if index < len(rows) and len(out) < max_test:
                out.append(rows[index])
        index += 1
    return out


def load_xlam_bfcl(max_train: int = 2000, max_test: int = 800,
                   log=print) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` = (xLAM-60k, BFCL) as ``function_call`` rows.

    xLAM is a **gated** repo; the pipeline reaches it via the ``HF_TOKEN`` that
    ``tests/pipeline/run.py`` loads from ``.env`` before any loader import.
    """
    from datasets import load_dataset

    train = convert_xlam_rows(load_dataset(XLAM_ID, split=f"train[:{max_train}]"))
    log(f"      [xlam] {len(train)} usable train rows of {max_train} requested")
    test = load_bfcl(max_test, log=log)
    if not test:
        raise RuntimeError(
            "BFCL produced zero eval rows — refusing to proceed with an empty held-out set. "
            "Check network access to the Hub and that the BFCL_v3_* category files still exist."
        )
    return train, test
