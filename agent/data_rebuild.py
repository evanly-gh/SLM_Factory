"""Validated, declarative plans for the `data_rebuild` intervention.

There are exactly TWO sub-strategies, and they are the only two ways the curriculum can grow:

    mine_new_real       Add REAL rows. First from datasets this run has already sourced but not
                        exhausted; if all of those are used up, from a new dataset found by web
                        research. Never invents anything.
    surgical_synthesis  Add TEACHER-GENERATED rows aimed at the failure categories the model is
                        actually losing points on. Verified programmatically where an exact check
                        exists (format-bound tasks) and by the teacher otherwise.

WHAT WAS REMOVED, AND WHY (2026-08-19)

    ``resample`` — re-drew rows from the pool the curriculum was already built from. It could change
    WHICH gold rows were present but never add information; one traced rebuild resampled 3,308 rows
    of which 122 were novel.

    the universal gold FILL — the same defect one level down. Because the curriculum was rebuilt to a
    target size every iteration, something had to refill it from the train pool, and with nothing
    else changed it re-selected the identical ~3,235 rows and honestly reported ``0 novel`` eight
    times in one run. The curriculum is now CUMULATIVE: a rebuild adds rows, and rows leave only via
    quality control or the eval firewall.

    ``synthesize`` as a balanced label-space fill — spent most of the teacher budget on rows chosen
    for class balance rather than for anything the model was getting wrong. Targeted generation is
    strictly better use of the same calls, so surgical synthesis is now the whole of it.

    ``target_rows`` — the curriculum no longer has a target. It has a starting size and it grows.

The plan carries no seed and no dedup identity: the orchestrator freely re-picks a strategy each
turn, and escalation-on-no-improvement is the sole stuck-run backstop.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

DATA_REBUILD_SCHEMA_VERSION = 3

MINE_NEW_REAL = "mine_new_real"
SURGICAL_SYNTHESIS = "surgical_synthesis"
DATA_REBUILD_STRATEGIES = (MINE_NEW_REAL, SURGICAL_SYNTHESIS)

# How many consecutive web-research rounds may fail to contribute a single novel row before
# `mine_new_real` is retired for the rest of the run. Two: one failure is a bad search, two in a row
# means the hub does not have another corpus carrying this task's labels, and continuing to pay for
# discovery is spending money to re-learn that.
MAX_FAILED_DISCOVERY_ROUNDS = 2

# Mirrors agent.nodes.iterate.HYPOTHESIS_MAX_CHARS (imported lazily to avoid a circular import).
HYPOTHESIS_MAX_CHARS = int(os.environ.get("SLM_HYPOTHESIS_MAX_CHARS", "2000"))
PATTERN_HINT_MAX_CHARS = 1200
MAX_TARGET_CATEGORIES = 8

# How many rows a single rebuild may add. The floor stops the orchestrator spending a whole
# train+eval cycle on a handful of rows; the ceiling stops one turn dominating the run.
MIN_REBUILD_ROWS = 50
MAX_REBUILD_ROWS = 2000

# Fields the ORCHESTRATOR may send.
_PLAN_FIELDS = frozenset({
    "schema_version",
    "strategy",
    "rows",
    "target_categories",
    "pattern_hint",
})


def _row_text(row: Any) -> str:
    if not isinstance(row, Mapping):
        return ""
    return str(row.get("text", row.get("prompt", "")) or "")


def _normalized_row_texts(rows: Any) -> set[str]:
    """Normalized surface texts for a row list, sharing curate's normalizer."""
    try:
        from data.loaders.dataset_integrity import normalize_text
    except Exception:  # pragma: no cover - normalizer always present in practice
        def normalize_text(value: Any) -> str:  # type: ignore[misc]
            return re.sub(r"\s+", " ", str(value or "")).strip().lower()
    values = {
        normalize_text(_row_text(row))
        for row in (rows or [])
        if isinstance(row, Mapping)
    }
    values.discard("")
    return values


def _plain_text(value: Any, field: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if len(text) > maximum:
        raise ValueError(f"{field} must be at most {maximum} characters")
    return text


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = int(round(float(value)))
    return max(minimum, min(maximum, number))


def _target_categories(raw: Any) -> list[dict[str, Any]]:
    """Validate the failure categories a surgical rebuild should aim at.

    Each entry is ``{"category": <name>, "count": <observed failures>}``. The categories come from
    the task's own failure taxonomy (`TaskSpec.failure_category`) via the test report, so they name
    something the scorer actually measured rather than a class the orchestrator invented.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("data_rebuild.target_categories must be a list")
    out: list[dict[str, Any]] = []
    for index, entry in enumerate(raw[:MAX_TARGET_CATEGORIES]):
        if not isinstance(entry, Mapping):
            raise ValueError(f"data_rebuild.target_categories[{index}] must be an object")
        category = _plain_text(
            entry.get("category", ""),
            f"data_rebuild.target_categories[{index}].category",
            maximum=64,
        )
        if not category:
            continue
        count = entry.get("count", 0)
        out.append({
            "category": category,
            "count": _integer(count, field=f"target_categories[{index}].count",
                              minimum=0, maximum=10**6) if count is not None else 0,
        })
    return out


def normalize_data_rebuild_plan(
    plan: Mapping[str, Any],
    *,
    task: str,
    hypothesis: str = "",
    mining_available: bool = True,
) -> dict[str, Any]:
    """Validate and clamp an orchestrator-authored rebuild plan.

    Unknown fields are rejected rather than ignored: a plan that sets something we removed is a
    plan written against the wrong contract, and silently dropping it would let the orchestrator
    believe it had asked for something.

    `mining_available` is False once every known source is exhausted AND web research has failed
    its allowance. A `mine_new_real` plan is then rewritten to `surgical_synthesis`, because the
    alternative is spending a full train+eval cycle on an intervention that provably cannot add a
    row.
    """
    if not isinstance(plan, Mapping):
        raise ValueError("data_rebuild plan must be an object")
    unknown = set(plan) - _PLAN_FIELDS
    if unknown:
        raise ValueError(
            f"data_rebuild has unknown field(s) {sorted(unknown)}; "
            f"allowed: {sorted(_PLAN_FIELDS)}"
        )
    version = plan.get("schema_version", DATA_REBUILD_SCHEMA_VERSION)
    if int(version) != DATA_REBUILD_SCHEMA_VERSION:
        raise ValueError(
            f"data_rebuild.schema_version must be {DATA_REBUILD_SCHEMA_VERSION}"
        )
    strategy = plan.get("strategy")
    if strategy not in DATA_REBUILD_STRATEGIES:
        raise ValueError(
            f"data_rebuild.strategy {strategy!r} must be one of "
            f"{list(DATA_REBUILD_STRATEGIES)}"
        )
    if strategy == MINE_NEW_REAL and not mining_available:
        strategy = SURGICAL_SYNTHESIS

    hint = plan.get("pattern_hint")
    pattern_hint = (
        _plain_text(hint, "data_rebuild.pattern_hint", maximum=PATTERN_HINT_MAX_CHARS)
        if hint is not None else ""
    )
    rows = plan.get("rows")
    return {
        "schema_version": DATA_REBUILD_SCHEMA_VERSION,
        "strategy": strategy,
        "rows": _integer(
            rows if rows is not None else 300,
            field="data_rebuild.rows",
            minimum=MIN_REBUILD_ROWS,
            maximum=MAX_REBUILD_ROWS,
        ),
        "target_categories": _target_categories(plan.get("target_categories")),
        "pattern_hint": pattern_hint,
        "hypothesis": _plain_text(
            hypothesis or "", "data_rebuild.hypothesis", maximum=HYPOTHESIS_MAX_CHARS
        ),
        "task": task,
    }


def mining_available_for_state(state: Mapping[str, Any]) -> bool:
    """Whether `mine_new_real` can still add rows.

    False only when BOTH are true: every dataset this run has sourced is exhausted, and web
    research has already failed `MAX_FAILED_DISCOVERY_ROUNDS` times without contributing a row.
    Until then mining is offered, because a source with rows left is free to re-read and a
    discovery round that has not yet been tried might find something.
    """
    if unexhausted_sources(state):
        return True
    return int(state.get("failed_discovery_rounds", 0) or 0) < MAX_FAILED_DISCOVERY_ROUNDS


def unexhausted_sources(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sources this run has drawn from that still have rows we have not taken.

    `source_progress` is maintained by curate: ``{source_id: {"consumed": n, "total": m|None}}``.
    A source with an unknown total counts as unexhausted until a re-read returns nothing new —
    most loaders take a head slice of a split whose length we never measured, and assuming
    exhaustion would give up on the largest corpora in the suite.
    """
    out: list[dict[str, Any]] = []
    progress = state.get("source_progress") or {}
    if not isinstance(progress, Mapping):
        return out
    for source_id, record in progress.items():
        if not isinstance(record, Mapping):
            continue
        if record.get("exhausted"):
            continue
        out.append({"source": str(source_id), **dict(record)})
    return out


def fallback_data_rebuild_plan(
    state: Mapping[str, Any],
    *,
    hypothesis: str = "",
) -> dict[str, Any]:
    """The plan used when the orchestrator's own JSON could not be obtained or validated.

    Prefers real rows while any source has them: real data is free of teacher error, and on this
    project a gold-only curriculum produced the best result anyone has measured (BC5CDR, 0.8098).
    Falls back to surgical synthesis, aimed at whatever the last test report says is failing most.
    """
    task = str(state.get("task") or "")
    strategy = (
        MINE_NEW_REAL if mining_available_for_state(state) else SURGICAL_SYNTHESIS
    )
    report = state.get("test_report") or {}
    categories = [
        {"category": str(pair.get("gold")), "count": int(pair.get("count", 0) or 0)}
        for pair in (report.get("confusion_pairs") or [])
        if isinstance(pair, Mapping) and pair.get("gold")
    ][:MAX_TARGET_CATEGORIES]
    return normalize_data_rebuild_plan(
        {
            "schema_version": DATA_REBUILD_SCHEMA_VERSION,
            "strategy": strategy,
            "rows": 300,
            "target_categories": categories,
            "pattern_hint": "",
        },
        task=task,
        hypothesis=hypothesis or "deterministic fallback: no valid orchestrator plan",
        mining_available=mining_available_for_state(state),
    )
