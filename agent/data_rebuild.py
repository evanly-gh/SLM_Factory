"""Validated, declarative plans for dataset rebuild interventions.

Redesigned 2026-07-31 (see docs/superpowers/specs/2026-07-31-data-curation-redesign-design.md):

- Exactly three strategies, single-choice, no primary/support composition and no
  task-type or score gating: ``resample`` (reshuffle the existing pool),
  ``acquire`` (add rows from the same or a new provenance), and ``synthesize``
  (task-adaptive synthetic generation).
- Non-deterministic: there is no plan-identity dedup, no untried-plan rotation,
  and no plan-space exhaustion. The orchestrator freely re-picks a strategy each
  turn; escalation-on-no-improvement is the sole stuck-run backstop. Sampling
  entropy lives in curate, not here.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


DATA_REBUILD_SCHEMA_VERSION = 2
DATA_REBUILD_STRATEGIES = ("resample", "acquire", "synthesize")
MAX_CONFUSION_PAIRS = 8
MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN = 3
MAX_PAID_ACQUIRE_ROUNDS_PER_RUN = 9

_PLAN_FIELDS = frozenset({
    "schema_version",
    "strategy",
    "target_rows",
    "resample_fraction",
    "new_real_rows",
    "synth_rows",
    "max_acquire_rounds",
    "difficulty_buckets",
    "confusion_pairs",
    "pattern_hint",
})
_DIFFICULTY_BUCKETS = ("easy", "medium", "hard")


def _data_size_ceiling() -> int:
    """Upper clamp for target_rows — imported lazily to keep this module cheap."""
    try:
        from config.config import DATA_SIZE_CEILING

        return int(DATA_SIZE_CEILING)
    except Exception:
        return 10000


def _plain_text(value: Any, field: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text[:maximum]


def _number(
    value: Any,
    *,
    field: str,
    default: float,
    lower: float,
    upper: float,
    step: float,
) -> float:
    if value is None:
        value = default
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    clamped = min(upper, max(lower, float(value)))
    units = math.floor((clamped - lower) / step + 0.5)
    snapped = min(upper, max(lower, lower + units * step))
    decimals = max(0, len(str(step).partition(".")[2]))
    return round(snapped, decimals)


def _integer(
    value: Any,
    *,
    field: str,
    default: int,
    lower: int,
    upper: int,
    step: int = 1,
) -> int:
    snapped = _number(
        value,
        field=field,
        default=float(default),
        lower=float(lower),
        upper=float(upper),
        step=float(step),
    )
    return int(snapped)


def _normalized_difficulty(raw: Any) -> dict[str, float]:
    supplied = raw is not None
    if raw is None:
        raw = {"easy": 0.2, "medium": 0.3, "hard": 0.5}
    if not isinstance(raw, Mapping):
        raise ValueError("data_rebuild.difficulty_buckets must be an object")
    unsupported = set(raw) - set(_DIFFICULTY_BUCKETS)
    if unsupported:
        raise ValueError(
            "unsupported difficulty bucket(s): " + ", ".join(sorted(unsupported))
        )
    values = []
    for bucket in _DIFFICULTY_BUCKETS:
        values.append(_number(
            raw.get(bucket, 0.0),
            field=f"data_rebuild.difficulty_buckets.{bucket}",
            default=0.0,
            lower=0.0,
            upper=1.0,
            step=0.001,
        ))
    total = sum(values)
    if total <= 0:
        if supplied:
            raise ValueError(
                "data_rebuild requires at least one positive difficulty weight"
            )
        values, total = [0.2, 0.3, 0.5], 1.0

    # Twenty 0.05 units apportioned by largest remainder — bounded, snapped, sums to 1.
    scaled = [value / total * 20 for value in values]
    units = [math.floor(value) for value in scaled]
    remaining = 20 - sum(units)
    order = sorted(
        range(len(units)),
        key=lambda index: (-(scaled[index] - units[index]), index),
    )
    for index in order[:remaining]:
        units[index] += 1
    return {
        bucket: round(units[index] / 20, 2)
        for index, bucket in enumerate(_DIFFICULTY_BUCKETS)
    }


def _confusion_pairs(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("data_rebuild.confusion_pairs must be a list")
    aggregate: dict[tuple[str, str], int] = defaultdict(int)
    for index, pair in enumerate(raw):
        if not isinstance(pair, Mapping):
            raise ValueError(
                f"data_rebuild.confusion_pairs[{index}] must be an object"
            )
        unsupported = set(pair) - {"gold", "predicted", "count"}
        if unsupported:
            raise ValueError(
                "unsupported confusion-pair field(s): "
                + ", ".join(sorted(unsupported))
            )
        gold = _plain_text(
            pair.get("gold"),
            f"data_rebuild.confusion_pairs[{index}].gold",
            maximum=64,
        )
        predicted = _plain_text(
            pair.get("predicted"),
            f"data_rebuild.confusion_pairs[{index}].predicted",
            maximum=64,
        )
        count = _integer(
            pair.get("count"),
            field=f"data_rebuild.confusion_pairs[{index}].count",
            default=1,
            lower=1,
            upper=10_000,
        )
        aggregate[(gold, predicted)] = min(
            10_000,
            aggregate[(gold, predicted)] + count,
        )
    ordered = sorted(
        aggregate.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    return [
        {"gold": gold, "predicted": predicted, "count": count}
        for (gold, predicted), count in ordered[:MAX_CONFUSION_PAIRS]
    ]


def normalize_data_rebuild_plan(
    raw: Any,
    *,
    task_type: str,
    hypothesis: str,
    target_rows: int = 3000,
    default_dataset_version: int = 0,
    remaining_acquire_rounds: int = MAX_PAID_ACQUIRE_ROUNDS_PER_RUN,
    forbidden_eval_texts: list[str] | tuple[str, ...] | set[str] = (),
) -> dict[str, Any]:
    """Validate and normalize one bounded, single-strategy data-rebuild plan.

    The result is JSON-only and strictly allow-listed. Numeric requests are
    clamped and snapped so provider drift cannot create an unbounded action.
    There is no task-type or score gating: any strategy is valid for any task.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("data_rebuild is required and must be a JSON object")
    unsupported = set(raw) - _PLAN_FIELDS
    if unsupported:
        raise ValueError(
            "unsupported data_rebuild field(s): " + ", ".join(sorted(unsupported))
        )
    schema_version = raw.get("schema_version", DATA_REBUILD_SCHEMA_VERSION)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != DATA_REBUILD_SCHEMA_VERSION
    ):
        raise ValueError(
            f"data_rebuild.schema_version must be {DATA_REBUILD_SCHEMA_VERSION}"
        )

    hypothesis_text = _plain_text(hypothesis, "hypothesis", maximum=240)
    strategy = raw.get("strategy")
    if strategy not in DATA_REBUILD_STRATEGIES:
        raise ValueError(
            f"data_rebuild.strategy {strategy!r} must be one of "
            + ", ".join(DATA_REBUILD_STRATEGIES)
        )

    hint_value = raw.get("pattern_hint")
    hint = (
        _plain_text(hint_value, "data_rebuild.pattern_hint", maximum=160)
        if hint_value is not None
        else ""
    )
    hypothesis_clause = f"causal hypothesis: {hypothesis_text}"
    pattern_hint = (
        hint
        if hypothesis_clause in hint
        else (f"{hint}; {hypothesis_clause}" if hint else hypothesis_clause)
    )[:240]
    normalized_hint = re.sub(r"\s+", " ", pattern_hint).strip().lower()
    for eval_text in forbidden_eval_texts:
        normalized_eval = re.sub(r"\s+", " ", str(eval_text)).strip().lower()
        if len(normalized_eval) >= 12 and normalized_eval in normalized_hint:
            raise ValueError(
                "data_rebuild pattern_hint/hypothesis contains raw eval text; "
                "use aggregate counts and categories only"
            )
    allowed_rounds = min(
        MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN,
        max(0, int(remaining_acquire_rounds)),
    )

    plan = {
        "schema_version": DATA_REBUILD_SCHEMA_VERSION,
        "strategy": strategy,
        "target_rows": _integer(
            raw.get("target_rows"),
            field="data_rebuild.target_rows",
            default=target_rows,
            lower=16,
            upper=_data_size_ceiling(),
            step=8,
        ),
        "resample_fraction": _number(
            raw.get("resample_fraction"),
            field="data_rebuild.resample_fraction",
            default=0.65,
            lower=0.10,
            upper=1.0,
            step=0.05,
        ),
        "new_real_rows": _integer(
            raw.get("new_real_rows"),
            field="data_rebuild.new_real_rows",
            default=40,
            lower=0,
            upper=500,
            step=5,
        ),
        "synth_rows": _integer(
            raw.get("synth_rows"),
            field="data_rebuild.synth_rows",
            default=300,
            lower=0,
            upper=2000,
            step=5,
        ),
        "max_acquire_rounds": _integer(
            raw.get("max_acquire_rounds"),
            field="data_rebuild.max_acquire_rounds",
            default=0,
            lower=0,
            upper=allowed_rounds,
        ),
        "difficulty_buckets": _normalized_difficulty(raw.get("difficulty_buckets")),
        "confusion_pairs": _confusion_pairs(raw.get("confusion_pairs")),
        "pattern_hint": pattern_hint,
    }

    # A material strategy must carry a positive budget; auto-fill a sensible one so
    # the orchestrator declaring a strategy without a budget still produces a valid plan.
    if strategy == "synthesize":
        # The synthesize strategy generates 100–500 new synthetic rows at the model's
        # discretion: the orchestrator's requested count is snapped into that band (an
        # unset/zero request defaults to the mid of the range).
        requested = plan["synth_rows"] or 300
        plan["synth_rows"] = min(500, max(100, requested))
    if strategy == "acquire" and plan["new_real_rows"] <= 0:
        plan["new_real_rows"] = min(plan["target_rows"], 40)
    return plan


def remaining_paid_acquire_rounds(state: Mapping[str, Any]) -> int:
    used = max(0, int(state.get("source_acquire_rounds_used", 0) or 0))
    try:
        from data.acquisition_budget import acquisition_budget_snapshot

        used = max(used, int(acquisition_budget_snapshot()["run_spent"]))
    except Exception:
        # No stable run directory exists for pure library/test callers that never
        # request paid acquisition. Reservation itself still fails closed.
        pass
    return max(0, MAX_PAID_ACQUIRE_ROUNDS_PER_RUN - used)


def _best_dataset_version(state: Mapping[str, Any]) -> int:
    candidates = [
        node for node in (state.get("dag") or [])
        if not node.get("pruned", False)
    ]
    if candidates:
        best = max(candidates, key=lambda node: float(node.get("score", 0.0)))
        dataset = ((best.get("pi") or {}).get("D") or {})
        try:
            return max(0, int(dataset.get("version", 0) or 0))
        except (TypeError, ValueError):
            pass
    return max(0, int(state.get("dataset_version", 0) or 0))


def _fallback_strategy_from_signal(
    state: Mapping[str, Any],
    *,
    task_type: str,
    score: float | None = None,
) -> str:
    """Pick a strategy NON-DETERMINISTICALLY, biased by the measured failure signal.

    This is the LLM-unavailable / parse-failure path. Unlike the old chooser it does
    not track "tried" plans or rotate through a bounded space — it draws a weighted
    random strategy so repeated fallbacks still explore. The weights lean on the same
    aggregate signals the orchestrator sees (per-difficulty accuracy, confusion pairs).
    """
    report = state.get("test_report") or {}
    buckets = report.get("by_difficulty") or {}

    def acc(name: str) -> float | None:
        value = (buckets.get(name) or {}).get("accuracy")
        return float(value) if value is not None else None

    easy, medium, hard = acc("easy"), acc("medium"), acc("hard")
    confusion = report.get("confusion_pairs") or []

    weights = {"resample": 1.0, "acquire": 1.0, "synthesize": 1.0}
    # Failing even the easy bucket => the data/labels are wrong; bring in new material.
    if easy is not None and easy < 0.6:
        weights["acquire"] += 2.0
    # Boundary weakness (medium/hard) => targeted synthesis of the confusable region.
    if any(value is not None and value < 0.6 for value in (medium, hard)):
        weights["synthesize"] += 2.0
    if confusion:
        weights["synthesize"] += 1.0

    choices, chance = zip(*weights.items())
    return random.choices(choices, weights=chance, k=1)[0]


def fallback_data_rebuild_plan(
    state: Mapping[str, Any],
    *,
    hypothesis: str,
    score: float | None = None,
) -> dict[str, Any]:
    """Build a non-deterministic, no-paid-call-by-default fallback plan."""
    task_type = str(state.get("task_type") or "classification")
    score = (
        float(score)
        if score is not None
        else float((state.get("scores") or [0.0])[-1])
    )
    strategy = _fallback_strategy_from_signal(
        state,
        task_type=task_type,
        score=score,
    )

    report = state.get("test_report") or {}
    raw_pairs = report.get("confusion_pairs") or []
    target_rows = int(state.get("curriculum_size_target", 3000) or 3000)

    remaining_rounds = remaining_paid_acquire_rounds(state)
    new_real_rows = min(200, max(20, target_rows // 4)) if strategy == "acquire" else 0
    # synthesize generates 100–500 rows (see normalize_data_rebuild_plan); the fallback
    # requests a target-scaled count that the validator snaps into that band.
    synth_rows = min(500, max(100, target_rows // 10)) if strategy == "synthesize" else 0

    # Weight difficulty buckets toward whichever buckets are ACTUALLY failing.
    buckets = report.get("by_difficulty") or {}
    accuracies = {
        name: (buckets.get(name) or {}).get("accuracy")
        for name in ("easy", "medium", "hard")
    }
    if any(value is not None for value in accuracies.values()):
        deficits = {
            name: max(0.0, 1.0 - (value if value is not None else 1.0))
            for name, value in accuracies.items()
        }
        total_deficit = sum(deficits.values())
        if total_deficit > 0:
            difficulty_buckets = {
                name: round(0.1 + 0.7 * (deficit / total_deficit), 3)
                for name, deficit in deficits.items()
            }
        else:
            difficulty_buckets = {"easy": 0.2, "medium": 0.3, "hard": 0.5}
    else:
        difficulty_buckets = {"easy": 0.2, "medium": 0.3, "hard": 0.5}

    raw = {
        "strategy": strategy,
        "target_rows": target_rows,
        "resample_fraction": 0.65,
        "new_real_rows": new_real_rows,
        "synth_rows": synth_rows,
        "max_acquire_rounds": min(1, remaining_rounds) if strategy == "acquire" else 0,
        "difficulty_buckets": difficulty_buckets,
        "confusion_pairs": raw_pairs,
        "pattern_hint": hypothesis,
    }
    return normalize_data_rebuild_plan(
        raw,
        task_type=task_type,
        hypothesis=hypothesis,
        target_rows=target_rows,
        default_dataset_version=_best_dataset_version(state),
        remaining_acquire_rounds=remaining_paid_acquire_rounds(state),
    )
