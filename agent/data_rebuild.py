"""Validated, declarative plans for dataset rebuild interventions."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


class DataRebuildPlanSpaceExhausted(ValueError):
    """Every reachable data_rebuild plan has already been tried.

    A ValueError subclass so existing `except ValueError` handlers still catch it,
    but a distinct type so callers can terminate the run cleanly (preserving the best
    model) instead of surfacing it as an unhandled crash.
    """


DATA_REBUILD_SCHEMA_VERSION = 1
DATA_REBUILD_STRATEGIES = (
    "resample_existing",
    "preserve_elite_resample",
    "mine_new_real_source",
    "source_diversification",
    "difficulty_weighted_sampling",
    "targeted_synth_positive",
)
TARGETED_SYNTH_TASK_TYPES = frozenset({"classification", "NER"})
MAX_SUPPORT_STRATEGIES = 2
MAX_CONFUSION_PAIRS = 8
MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN = 3
MAX_PAID_ACQUIRE_ROUNDS_PER_RUN = 9
QUERY_VARIANTS = 8
SAMPLING_STRATEGIES = frozenset({
    "resample_existing",
    "source_diversification",
    "difficulty_weighted_sampling",
})

_PLAN_FIELDS = frozenset({
    "schema_version",
    "primary_strategy",
    "support_strategies",
    "target_rows",
    "resample_fraction",
    "preserve_elite_fraction",
    "new_real_rows",
    "synth_rows",
    "max_acquire_rounds",
    "query_variant",
    "difficulty_buckets",
    "confusion_pairs",
    "pattern_hint",
    "elite",
})
_DIFFICULTY_BUCKETS = ("easy", "medium", "hard")
_ELITE_PROVENANCE = frozenset({
    "best_non_pruned_dataset",
    "current_dataset",
})


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

    # Twenty 0.05 units are apportioned by largest remainder. This keeps the
    # normalized result deterministic, bounded, snapped, and exactly summing to 1.
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


def _elite_reference(raw: Any, default_version: int) -> dict[str, Any]:
    if raw is None:
        raw = {
            "provenance": "best_non_pruned_dataset",
            "dataset_version": default_version,
        }
    if not isinstance(raw, Mapping):
        raise ValueError("data_rebuild.elite must be an object")
    unsupported = set(raw) - {"provenance", "dataset_version"}
    if unsupported:
        raise ValueError(
            "unsupported elite field(s): " + ", ".join(sorted(unsupported))
        )
    provenance = raw.get("provenance", "best_non_pruned_dataset")
    if provenance not in _ELITE_PROVENANCE:
        raise ValueError(
            "data_rebuild.elite.provenance must be one of "
            + ", ".join(sorted(_ELITE_PROVENANCE))
        )
    return {
        "provenance": provenance,
        "dataset_version": _integer(
            raw.get("dataset_version"),
            field="data_rebuild.elite.dataset_version",
            default=default_version,
            lower=0,
            upper=1_000_000,
        ),
    }


def eligible_data_rebuild_strategies(task_type: str) -> tuple[str, ...]:
    """Return strategies that have an executable positive-label contract."""
    if task_type in TARGETED_SYNTH_TASK_TYPES:
        return DATA_REBUILD_STRATEGIES
    return tuple(
        strategy
        for strategy in DATA_REBUILD_STRATEGIES
        if strategy != "targeted_synth_positive"
    )


def resolve_elite_source_path(
    state: Mapping[str, Any],
    elite: Mapping[str, Any],
) -> str | None:
    """Resolve an elite reference only to an existing declared dataset."""
    version = int(elite.get("dataset_version", 0) or 0)
    if elite.get("provenance") == "current_dataset":
        path = state.get("current_dataset_path")
        if (
            int(state.get("dataset_version", -1) or -1) == version
            and isinstance(path, str)
            and os.path.isfile(path)
        ):
            return path
        return None
    candidates = []
    for node in state.get("dag") or []:
        if not isinstance(node, Mapping) or node.get("pruned", False):
            continue
        dataset = ((node.get("pi") or {}).get("D") or {})
        path = dataset.get("path")
        if (
            int(dataset.get("version", -1) or -1) == version
            and isinstance(path, str)
            and os.path.isfile(path)
        ):
            candidates.append(node)
    if not candidates:
        return None
    winner = max(
        candidates,
        key=lambda node: (
            float(node.get("score", 0.0)),
            int(node.get("iteration", 0) or 0),
        ),
    )
    return ((winner.get("pi") or {}).get("D") or {}).get("path")


def require_resolvable_elite_source(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    strategies = {
        plan["primary_strategy"],
        *plan["support_strategies"],
    }
    if (
        "preserve_elite_resample" in strategies
        and resolve_elite_source_path(state, plan["elite"]) is None
    ):
        raise ValueError(
            "data_rebuild elite source is not resolvable for the declared "
            "provenance/version"
        )


def normalize_data_rebuild_plan(
    raw: Any,
    *,
    task_type: str,
    hypothesis: str,
    target_rows: int = 150,
    default_dataset_version: int = 0,
    remaining_acquire_rounds: int = MAX_PAID_ACQUIRE_ROUNDS_PER_RUN,
    forbidden_eval_texts: list[str] | tuple[str, ...] | set[str] = (),
) -> dict[str, Any]:
    """Validate and normalize one bounded data-rebuild plan.

    The result is JSON-only and strictly allow-listed. Numeric requests are
    clamped and snapped so provider drift cannot create an unbounded action.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("data_rebuild is required and must be a JSON object")
    unsupported = set(raw) - _PLAN_FIELDS
    if unsupported:
        raise ValueError(
            "unsupported data_rebuild field(s): "
            + ", ".join(sorted(unsupported))
        )
    schema_version = raw.get("schema_version", DATA_REBUILD_SCHEMA_VERSION)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != DATA_REBUILD_SCHEMA_VERSION
    ):
        raise ValueError(
            "data_rebuild.schema_version must be "
            f"{DATA_REBUILD_SCHEMA_VERSION}"
        )

    hypothesis_text = _plain_text(hypothesis, "hypothesis", maximum=240)
    eligible = eligible_data_rebuild_strategies(task_type)
    primary = raw.get("primary_strategy")
    if primary not in eligible:
        raise ValueError(
            f"data_rebuild primary strategy {primary!r} is not eligible for "
            f"task_type={task_type!r}; eligible strategies: {', '.join(eligible)}"
        )
    supports = raw.get("support_strategies", [])
    if not isinstance(supports, list) or not all(
        isinstance(value, str) for value in supports
    ):
        raise ValueError("data_rebuild.support_strategies must be a list of strings")
    if len(supports) > MAX_SUPPORT_STRATEGIES:
        raise ValueError(
            f"data_rebuild.support_strategies is bounded to "
            f"{MAX_SUPPORT_STRATEGIES}"
        )
    if len(set(supports)) != len(supports) or primary in supports:
        raise ValueError(
            "data_rebuild strategies must be unique across primary and support"
        )
    ineligible = [strategy for strategy in supports if strategy not in eligible]
    if ineligible:
        raise ValueError(
            f"data_rebuild support strategy {ineligible[0]!r} is not eligible "
            f"for task_type={task_type!r}"
        )

    hint_value = raw.get("pattern_hint")
    hint = (
        _plain_text(
            hint_value,
            "data_rebuild.pattern_hint",
            maximum=160,
        )
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
        normalized_eval = re.sub(
            r"\s+",
            " ",
            str(eval_text),
        ).strip().lower()
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
        "primary_strategy": primary,
        "support_strategies": list(supports),
        "target_rows": _integer(
            raw.get("target_rows"),
            field="data_rebuild.target_rows",
            default=target_rows,
            lower=16,
            upper=2000,
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
        "preserve_elite_fraction": _number(
            raw.get("preserve_elite_fraction"),
            field="data_rebuild.preserve_elite_fraction",
            default=0.20,
            lower=0.0,
            upper=0.80,
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
            default=20,
            lower=0,
            upper=200,
            step=5,
        ),
        "max_acquire_rounds": _integer(
            raw.get("max_acquire_rounds"),
            field="data_rebuild.max_acquire_rounds",
            default=0,
            lower=0,
            upper=allowed_rounds,
        ),
        "query_variant": _integer(
            raw.get("query_variant"),
            field="data_rebuild.query_variant",
            default=0,
            lower=0,
            upper=QUERY_VARIANTS - 1,
        ),
        "difficulty_buckets": _normalized_difficulty(
            raw.get("difficulty_buckets")
        ),
        "confusion_pairs": _confusion_pairs(raw.get("confusion_pairs")),
        "pattern_hint": pattern_hint,
        "elite": _elite_reference(
            raw.get("elite"),
            default_version=default_dataset_version,
        ),
    }
    strategies = [primary, *supports]
    sampling = [
        strategy for strategy in strategies
        if strategy in SAMPLING_STRATEGIES
    ]
    if "resample_existing" in supports:
        raise ValueError(
            "resample_existing must be the primary strategy, not a no-op support"
        )
    if len(sampling) > 1:
        raise ValueError(
            "data_rebuild may declare only one sampling strategy"
        )
    material_budgets = {
        "preserve_elite_resample": plan["preserve_elite_fraction"],
        "mine_new_real_source": plan["new_real_rows"],
        "targeted_synth_positive": plan["synth_rows"],
    }
    for strategy, budget in material_budgets.items():
        if strategy in strategies and budget <= 0:
            raise ValueError(
                f"{strategy} requires a positive material budget"
            )
    total_material_budget = sum(
        (
            round(plan["target_rows"] * plan["preserve_elite_fraction"])
            if strategy == "preserve_elite_resample"
            else plan["new_real_rows"]
            if strategy == "mine_new_real_source"
            else plan["synth_rows"]
            if strategy == "targeted_synth_positive"
            else 0
        )
        for strategy in strategies
    )
    if total_material_budget > plan["target_rows"]:
        raise ValueError(
            "data_rebuild material budgets exceed target_rows "
            f"({total_material_budget} > {plan['target_rows']})"
        )
    return plan


def data_rebuild_plan_identity(plan: Mapping[str, Any]) -> str:
    """Return a deterministic identity for one normalized declarative plan."""
    encoded = json.dumps(
        dict(plan),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def tried_data_rebuild_plans(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every attempted plan, including pruned and zero-yield plans."""
    records = []
    seen = set()
    nodes = list(state.get("dag") or [])
    for tier in state.get("escalation_history") or []:
        if isinstance(tier, Mapping):
            nodes.extend(tier.get("dag") or [])
    for node in nodes:
        dataset = ((node.get("pi") or {}).get("D") or {})
        plan = dataset.get("plan")
        identity = dataset.get("plan_identity")
        if not identity and isinstance(plan, Mapping):
            identity = data_rebuild_plan_identity(plan)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        composition = dataset.get("composition") or {}
        yield_report = composition.get("plan_yield") or {}
        records.append({
            "identity": identity,
            "plan": dict(plan) if isinstance(plan, Mapping) else None,
            "score": node.get("score"),
            "pruned": bool(node.get("pruned", False)),
            "yield_status": yield_report.get("status"),
        })

    current_identity = state.get("data_rebuild_plan_identity")
    current_plan = state.get("data_rebuild_plan")
    current_curation = state.get("last_curation") or {}
    if (
        current_identity
        and current_identity not in seen
        and current_curation.get("data_rebuild_plan_identity") == current_identity
    ):
        records.append({
            "identity": current_identity,
            "plan": dict(current_plan) if isinstance(current_plan, Mapping) else None,
            "score": None,
            "pruned": False,
            "yield_status": (
                (current_curation.get("plan_yield") or {}).get("status")
            ),
        })
    return records


def remaining_paid_acquire_rounds(state: Mapping[str, Any]) -> int:
    used = max(0, int(state.get("source_acquire_rounds_used", 0) or 0))
    try:
        from data.acquisition_budget import acquisition_budget_snapshot

        used = max(used, int(acquisition_budget_snapshot()["run_spent"]))
    except Exception:
        # No stable run directory exists for pure library/test callers that
        # never request paid acquisition. Reservation itself still fails closed.
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
    elite_available: bool,
    score: float | None = None,
) -> tuple[str, list[str]]:
    """Choose a rebuild strategy from MEASURED signal, not from prose keywords.

    This replaces a keyword match over the hypothesis string
    (``"hard" in hypothesis`` -> difficulty_weighted_sampling, etc.), which was a
    structural dead end: on the fallback path the hypothesis is the test-agent
    diagnosis, and the only two diagnoses that ever *suggest* data_rebuild contain
    none of the matched keywords. Every fallback rebuild therefore fell through to
    the catch-all `resample_existing` — 31 of 31 in the NER run, which exhausted the
    (single-strategy) plan space and crashed the run.

    The inputs here are the same aggregate signals the orchestrator sees, so an
    unavailable LLM degrades to a *reasoned* choice rather than a constant.
    """
    report = state.get("test_report") or {}
    buckets = report.get("by_difficulty") or {}

    def acc(name: str) -> float | None:
        value = (buckets.get(name) or {}).get("accuracy")
        return float(value) if value is not None else None

    easy, medium, hard = acc("easy"), acc("medium"), acc("hard")
    # tried_data_rebuild_plans returns {identity, plan, score, pruned, yield_status};
    # the strategy is nested inside `plan`, not at the record's top level.
    tried = {
        (record.get("plan") or {}).get("primary_strategy")
        for record in tried_data_rebuild_plans(state)
    }
    tried.discard(None)
    confusion = report.get("confusion_pairs") or []

    def pick(candidate: str) -> str | None:
        """Prefer a strategy that hasn't already been tried this run."""
        return candidate if candidate not in tried else None

    # 0. Near-ceiling refinement: at a very high score the remaining errors are a thin
    #    tail of confusable cases, and synthesizing targeted positives is the right
    #    move. This branch is PRESERVED from the old chooser — what changed is that it
    #    is no longer the ONLY way to reach synthesis (see branch 2). Requiring
    #    score >= 0.95 as the sole gate put synthesis above both runs' stop thresholds
    #    (0.88 and 0.82), so it could only fire in a run that had already won.
    if (
        score is not None
        and score >= 0.95
        and task_type in TARGETED_SYNTH_TASK_TYPES
    ):
        return "targeted_synth_positive", (
            ["preserve_elite_resample"] if elite_available else []
        )

    # 1. Failing even the easy bucket => the DATA is wrong (labels/format/balance),
    #    not the sampling. Mine genuinely new material before reshuffling.
    if easy is not None and easy < 0.6:
        for candidate in ("mine_new_real_source", "source_diversification"):
            if pick(candidate):
                return candidate, (
                    ["preserve_elite_resample"] if elite_available else []
                )

    # 2. Easy solid but medium/hard weak => reweight toward the failing buckets, and
    #    synthesize targeted examples where the task supports it. The synthesis gate
    #    used to require score >= 0.95, which is ABOVE both runs' stop thresholds
    #    (0.88 and 0.82) — synthesis could therefore only fire in a run that had
    #    already surpassed its goal, i.e. never. It is now driven by the difficulty
    #    gap that actually calls for it.
    weak = [value for value in (medium, hard) if value is not None and value < 0.6]
    if weak:
        if task_type in TARGETED_SYNTH_TASK_TYPES and confusion and pick(
            "targeted_synth_positive"
        ):
            return "targeted_synth_positive", (
                ["difficulty_weighted_sampling"] if elite_available else []
            )
        if pick("difficulty_weighted_sampling"):
            return "difficulty_weighted_sampling", []

    # 3. No single failing bucket, but below goal => broaden the pool with new real
    #    material rather than redrawing the same one.
    for candidate in (
        "difficulty_weighted_sampling",
        "mine_new_real_source",
        "source_diversification",
        "preserve_elite_resample" if elite_available else "resample_existing",
    ):
        if pick(candidate):
            return candidate, []

    # 4. Everything tried at least once — fall back to a reshuffle, which the
    #    query_variant / resample_fraction rotation can still vary.
    return "resample_existing", []


def fallback_data_rebuild_plan(
    state: Mapping[str, Any],
    *,
    hypothesis: str,
    score: float | None = None,
) -> dict[str, Any]:
    """Build a safe, deterministic, no-paid-call fallback plan."""
    task_type = str(state.get("task_type") or "classification")
    score = (
        float(score)
        if score is not None
        else float((state.get("scores") or [0.0])[-1])
    )
    elite = {
        "provenance": "best_non_pruned_dataset",
        "dataset_version": _best_dataset_version(state),
    }
    elite_available = resolve_elite_source_path(state, elite) is not None
    primary, supports = _fallback_strategy_from_signal(
        state,
        task_type=task_type,
        elite_available=elite_available,
        score=score,
    )

    report = state.get("test_report") or {}
    raw_pairs = report.get("confusion_pairs") or []
    hypothesis_digest = hashlib.sha256(
        hypothesis.encode("utf-8")
    ).digest()
    target_rows = int(state.get("curriculum_size_target", 150) or 150)
    strategies = {primary, *supports}

    # Material budgets must match the chosen strategies. These were hard-zero before,
    # which was consistent only because the strategy was always `resample_existing`;
    # a mining or synthesis strategy with a zero budget fails plan validation.
    remaining_rounds = remaining_paid_acquire_rounds(state)
    needs_mining = bool(
        strategies & {"mine_new_real_source", "source_diversification"}
    )
    new_real_rows = min(200, max(20, target_rows // 4)) if needs_mining else 0
    synth_rows = 20 if "targeted_synth_positive" in strategies else 0

    # Weight the difficulty buckets toward whichever buckets are ACTUALLY failing,
    # rather than toward whether the word "hard" appeared in a prose hypothesis.
    buckets = report.get("by_difficulty") or {}
    accuracies = {
        name: (buckets.get(name) or {}).get("accuracy")
        for name in ("easy", "medium", "hard")
    }
    if any(value is not None for value in accuracies.values()):
        # Weight inversely to accuracy: the worse a bucket scores, the more of the
        # rebuilt curriculum it gets. Unmeasured buckets fall back to their accuracy
        # being treated as 1.0 (no extra weight).
        deficits = {
            name: max(0.0, 1.0 - (value if value is not None else 1.0))
            for name, value in accuracies.items()
        }
        total_deficit = sum(deficits.values())
        if total_deficit > 0:
            difficulty_buckets = {
                # Floor each bucket at 0.1 so a rebuild never drops a bucket entirely.
                name: round(0.1 + 0.7 * (deficit / total_deficit), 3)
                for name, deficit in deficits.items()
            }
        else:
            difficulty_buckets = {"easy": 0.2, "medium": 0.3, "hard": 0.5}
    else:
        difficulty_buckets = {"easy": 0.2, "medium": 0.3, "hard": 0.5}

    raw = {
        "primary_strategy": primary,
        "support_strategies": supports,
        "target_rows": target_rows,
        "resample_fraction": 0.65,
        "preserve_elite_fraction": 0.25,
        "new_real_rows": new_real_rows,
        "synth_rows": synth_rows,
        "max_acquire_rounds": min(1, remaining_rounds) if needs_mining else 0,
        "query_variant": hypothesis_digest[0] % QUERY_VARIANTS,
        "difficulty_buckets": difficulty_buckets,
        "confusion_pairs": raw_pairs,
        "pattern_hint": hypothesis,
        "elite": elite,
    }
    return normalize_data_rebuild_plan(
        raw,
        task_type=task_type,
        hypothesis=hypothesis,
        target_rows=int(state.get("curriculum_size_target", 150) or 150),
        default_dataset_version=_best_dataset_version(state),
        remaining_acquire_rounds=remaining_paid_acquire_rounds(state),
    )


def ensure_untried_data_rebuild_plan(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, int]]:
    """Return a deterministic untried plan and tried/pruned counts."""
    records = tried_data_rebuild_plans(state)
    tried = {record["identity"] for record in records}
    stats = {
        "tried": len(records),
        "pruned": sum(1 for record in records if record.get("pruned")),
        "zero_yield": sum(
            1 for record in records
            if record.get("yield_status") == "no_novelty"
        ),
    }
    candidate = dict(plan)
    identity = data_rebuild_plan_identity(candidate)
    if identity not in tried:
        return candidate, identity, stats

    # query_variant is also part of every deterministic sampling seed, so each
    # rotated plan materially changes execution even when no source query runs.
    original_variant = int(candidate.get("query_variant", 0))
    for offset in range(1, QUERY_VARIANTS + 1):
        rotated = dict(candidate)
        rotated["query_variant"] = (
            original_variant + offset
        ) % QUERY_VARIANTS
        identity = data_rebuild_plan_identity(rotated)
        if identity not in tried:
            return rotated, identity, stats

    for fraction in (0.50, 0.75, 0.90, 1.0, 0.35):
        rotated = dict(candidate)
        rotated["resample_fraction"] = fraction
        identity = data_rebuild_plan_identity(rotated)
        if identity not in tried:
            return rotated, identity, stats

    # Rotate the PRIMARY STRATEGY before giving up. Holding it fixed bounded the space
    # at roughly 1 x 8 query_variants x 5 fractions, which the NER run burned through
    # in 31 rebuilds — and the exhaustion then propagated as an uncaught ValueError
    # out of curate_node, killing a 44.8-hour run outright. Varying the strategy is
    # both a far larger space and a more meaningful one: a different strategy changes
    # what data is assembled, whereas a different seed only reshuffles it.
    for strategy in DATA_REBUILD_STRATEGIES:
        if strategy == candidate.get("primary_strategy"):
            continue
        if strategy == "targeted_synth_positive" and str(
            candidate.get("_task_type", "")
        ) not in TARGETED_SYNTH_TASK_TYPES:
            # Eligibility is task-gated; skip rather than emit an invalid plan.
            continue
        rotated = dict(candidate)
        rotated["primary_strategy"] = strategy
        # resample_existing is primary-only and must not linger in supports.
        rotated["support_strategies"] = [
            support
            for support in rotated.get("support_strategies", [])
            if support not in (strategy, "resample_existing")
        ]
        identity = data_rebuild_plan_identity(rotated)
        if identity not in tried:
            return rotated, identity, stats

    raise DataRebuildPlanSpaceExhausted(
        "bounded data_rebuild plan space is exhausted after rotating query_variant, "
        "resample_fraction, and primary_strategy — every reachable plan has been "
        "tried. Treat this as 'no further data intervention available', not as a "
        "pipeline fault: the caller should terminate cleanly and preserve the best "
        "model rather than propagate this as a crash."
    )
