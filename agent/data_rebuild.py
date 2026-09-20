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

class DataInterventionUnavailable(RuntimeError):
    """Neither sub-strategy can add rows, so `data_rebuild` is not a usable intervention.

    Raised by the plan validator rather than handled inside it: the caller (`iterate`) is the only
    place that can do the right thing, which is to choose a hyperparameter intervention instead.
    """


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
# Fields a plan may carry on the way IN. The last two are not things a caller chooses — they are
# supplied as arguments and written by the normalizer — but they are accepted here because the
# normalizer's own OUTPUT carries them, and two callers legitimately feed that output back:
#
#   * `curate_node` re-normalizes `state["data_rebuild_plan"]`, which `iterate` already normalized;
#   * the orchestrator, shown a schema, nests `hypothesis` inside the plan object instead of leaving
#     it at the top level.
#
# Rejecting them cost verification run 38658213 five consecutive iterations: the orchestrator asked
# for `data_rebuild` every time, the plan was thrown out on `unknown field(s) ['hypothesis', 'task']`,
# and the fallback quietly substituted a hyperparameter step — so the log read as though the
# orchestrator had wanted hyperparameters, and no data intervention ran at all (B312). A validator
# that cannot accept what it just produced is a trap regardless of who steps in it.
#
# Their values are IGNORED: `task` and `hypothesis` come from this function's arguments, which are
# authoritative. A genuinely unknown field still raises, which is the point of the check.
_PLAN_FIELDS = frozenset({
    "schema_version",
    "strategy",
    "rows",
    "target_categories",
    "pattern_hint",
    "hypothesis",
    "task",
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
    synthesis_allowed: bool = True,
) -> dict[str, Any]:
    """Validate and clamp an orchestrator-authored rebuild plan.

    Unknown fields are rejected rather than ignored: a plan that sets something we removed is a
    plan written against the wrong contract, and silently dropping it would let the orchestrator
    believe it had asked for something.

    Two availability flags decide what a plan may ask for, and a plan asking for something
    unavailable is REWRITTEN rather than run, because the alternative is spending a full train+eval
    cycle on an intervention that provably cannot add a row:

      `mining_available`   False once every known source is exhausted AND web research has spent its
                           allowance. `mine_new_real` is then rewritten to `surgical_synthesis`.
      `synthesis_allowed`  False when the teacher could not clear the fitness gate on this task
                           (agent/teacher_fitness.py). `surgical_synthesis` is then rewritten to
                           `mine_new_real`.

    When NEITHER is available there is no data intervention left, and `DataInterventionUnavailable`
    is raised so `iterate` can route to a hyperparameter step instead of curating a no-op.

    IDEMPOTENT: normalizing an already-normalized plan returns the same plan. `curate_node` re-checks
    the plan `iterate` stored, so a validator that rejected its own output would fail every rebuild
    at the second gate rather than the first — see `_PLAN_FIELDS`.
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
    if not mining_available and not synthesis_allowed:
        raise DataInterventionUnavailable(
            "no data intervention can add rows: every source is exhausted with web research spent, "
            "and the teacher did not clear the synthesis fitness gate"
        )
    if strategy == MINE_NEW_REAL and not mining_available:
        strategy = SURGICAL_SYNTHESIS
    elif strategy == SURGICAL_SYNTHESIS and not synthesis_allowed:
        strategy = MINE_NEW_REAL

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
            # The argument wins; the plan's own value is the fallback so that re-normalizing does not
            # blank a hypothesis the orchestrator already wrote.
            hypothesis or plan.get("hypothesis") or "",
            "data_rebuild.hypothesis",
            maximum=HYPOTHESIS_MAX_CHARS,
        ),
        "task": task,
    }


def mining_available_for_state(state: Mapping[str, Any]) -> bool:
    """Whether `mine_new_real` can still add rows.

    False when run health has RETIRED the route — two rounds that added no rows, or three that saw
    candidates and accepted none (`run_health._retire_mining`). That check comes first and overrides
    the source bookkeeping below, because it is evidence from actually running mining, which beats
    `source_progress`'s optimistic assumption that a source of unknown length still has rows.

    Otherwise false only when BOTH are true: every dataset this run has sourced is exhausted, and web
    research has already failed `MAX_FAILED_DISCOVERY_ROUNDS` times without contributing a row.
    Until then mining is offered, because a source with rows left is free to re-read and a
    discovery round that has not yet been tried might find something.
    """
    from agent.ablations import mining_disallowed
    from agent.run_health import MINING_RETIRED_KEY

    # ABLATION 4 first, ahead of every other consideration: the arm exists to hold the curriculum
    # at its starved size, and a run that mined its way back to thousands of real rows would be
    # measuring nothing. Checked before run health so the reason reported is the operator's, not a
    # retirement that happens to coincide.
    if mining_disallowed():
        return False
    if state.get(MINING_RETIRED_KEY):
        return False
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


def data_rebuild_available(state: Mapping[str, Any]) -> bool:
    """Whether `data_rebuild` can still add rows by either route."""
    from agent.teacher_fitness import synthesis_allowed

    return mining_available_for_state(state) or synthesis_allowed(state)


def fallback_data_rebuild_plan(
    state: Mapping[str, Any],
    *,
    hypothesis: str = "",
) -> dict[str, Any]:
    """The plan used when the orchestrator's own JSON could not be obtained or validated.

    Prefers real rows while any source has them: real data is free of teacher error, and on this
    project a gold-only curriculum produced the best result anyone has measured (BC5CDR, 0.8098).
    Falls back to surgical synthesis, aimed at whatever the last test report says is failing most.

    Never raises. This is the safety net for a failed orchestrator call, so it must always return a
    usable plan; the caller checks `data_rebuild_available` BEFORE choosing data_rebuild at all, and
    if it somehow did not, the plan returned here names the one route that is still open rather than
    refusing to produce anything.
    """
    task = str(state.get("task") or "")
    # Mining first: real rows carry no teacher error, and on this project a gold-only curriculum
    # produced the best result anyone has measured. Synthesis is the residual, not a peer — so
    # `synthesis_allowed` is deliberately NOT consulted here. The caller checks
    # `data_rebuild_available` before choosing data_rebuild at all, and this function must always
    # return a usable plan rather than refuse.
    strategy = MINE_NEW_REAL if mining_available_for_state(state) else SURGICAL_SYNTHESIS
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
        # Both flags forced open for the fallback itself: the strategy above was already chosen
        # against them, and letting the validator rewrite or reject it here would either flip that
        # choice back or raise from the one code path that is not allowed to fail.
        mining_available=True,
        synthesis_allowed=True,
    )
