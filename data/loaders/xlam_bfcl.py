"""xLAM / BFCL function-calling loader (2026-08-01).

Training data comes from Salesforce's xlam-function-calling-60k (query + tool signatures +
gold calls); the held-out eval comes from BFCL (Berkeley Function-Calling Leaderboard). Both
are shaped into the ``function_call`` row schema ``{text, answer, tools}`` where ``answer`` is a
canonical JSON string of the gold call list ``[{"name", "arguments"}]`` — exactly what
``eval/scorers/function_call.py`` parses.

The pure ``convert_xlam_rows`` is unit-tested on an in-memory sample; ``load_xlam_bfcl`` runs
the live HF pulls on the cluster.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

XLAM_ID = "Salesforce/xlam-function-calling-60k"
BFCL_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


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
    """Map raw xLAM/BFCL rows to ``{text, answer, tools}`` with a canonical gold-call JSON.

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


def load_xlam_bfcl(max_train: int = 2000, max_test: int = 800) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` = (xLAM-60k, BFCL) as ``function_call`` rows."""
    from datasets import load_dataset

    train = convert_xlam_rows(load_dataset(XLAM_ID, split=f"train[:{max_train}]"))
    # BFCL exposes several categories; the simple/parallel splits match the AST scorer. Fall
    # back to the train split slice if a dedicated test split is unavailable.
    try:
        raw_test = load_dataset(BFCL_ID, split=f"test[:{max_test}]")
    except (ValueError, KeyError):
        raw_test = load_dataset(BFCL_ID, split=f"train[:{max_test}]")
    test = convert_xlam_rows(raw_test)
    return train, test
