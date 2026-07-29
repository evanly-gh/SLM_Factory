"""Canonical bounded LoRA/optimizer search-space helpers.

The orchestrator boundary is permissive-but-deterministic: numeric suggestions are
snapped or clamped into the declared space, while ambiguous aliases and false derived
values are rejected. Runtime ``TrainingConfig`` validation remains strict.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from itertools import product
from typing import Any


LORA_SEARCH_SPACE_VERSION = 1

VALID_LORA_RANKS = (4, 8, 16, 32, 64)
VALID_LORA_ALPHA_MULTIPLIERS = (1, 2, 4)
# Orchestrator-facing alias for the same axis: alpha = rank x alpha_ratio.
VALID_ALPHA_RATIOS = VALID_LORA_ALPHA_MULTIPLIERS
VALID_LORA_DROPOUTS = (0.0, 0.05, 0.1)
VALID_WEIGHT_DECAYS = (0.0, 0.01, 0.05, 0.1)
VALID_MICRO_BATCH_SIZES = (1, 2, 4, 8)
VALID_GRADIENT_ACCUMULATION_STEPS = (1, 2, 4, 8)
MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 5e-4
MIN_EPOCHS = 1
MAX_EPOCHS = 8
MAX_EFFECTIVE_BATCH_SIZE = 64

DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA_MULTIPLIER = 2
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MICRO_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_EPOCHS = 3

HYPERPARAMETER_IDENTITY_FIELDS = (
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "weight_decay",
    "learning_rate",
    "nr_epochs",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, got boolean {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _integer(value: Any, field: str) -> int:
    number = _number(value, field)
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return int(number)


def _snap(value: Any, allowed: Iterable[float | int], field: str):
    number = _number(value, field)
    ordered = tuple(sorted(allowed))
    # Round the distance used for tie-breaking so values such as 0.075 choose
    # the smaller option deterministically despite binary-float representation.
    return min(ordered, key=lambda item: (round(abs(float(item) - number), 12), item))


def _alpha_value(value: Any, rank: int) -> int:
    if isinstance(value, str):
        alias = value.strip().lower().replace("×", "x").replace("*", "x")
        aliases = {
            "rank": 1,
            "r": 1,
            "1r": 1,
            "1xr": 1,
            "2r": 2,
            "2xr": 2,
            "4r": 4,
            "4xr": 4,
        }
        if alias in aliases:
            return rank * aliases[alias]
    return int(_snap(value, (rank, rank * 2, rank * 4), "lora_alpha"))


def _changed_note(field: str, raw: Any, normalized: Any, verb: str) -> str | None:
    try:
        unchanged = float(raw) == float(normalized)
    except (TypeError, ValueError):
        unchanged = raw == normalized
    if unchanged:
        return None
    return f"{field}={raw!r} {verb} to {normalized}"


def normalize_hyperparams(
    values: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Normalize an orchestrator/legacy config into one canonical bounded config."""
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ValueError(
            f"hyperparams must be a JSON object, got {type(values).__name__}"
        )
    raw = dict(values)
    notes: list[str] = []

    rank_raw = raw.get("lora_rank", DEFAULT_LORA_RANK)
    rank = int(_snap(rank_raw, VALID_LORA_RANKS, "lora_rank"))
    note = _changed_note("lora_rank", rank_raw, rank, "snapped")
    if note:
        notes.append(note)

    # `alpha_ratio` is the orchestrator-facing form: alpha only ever matters relative
    # to rank (LoRA scales its update by alpha/rank), so exposing both invited
    # incoherent pairs. `lora_alpha` is still accepted for checkpoint/DAG replay of
    # configs recorded before the switch.
    if "alpha_ratio" in raw:
        ratio_raw = raw["alpha_ratio"]
        ratio = _snap(ratio_raw, VALID_ALPHA_RATIOS, "alpha_ratio")
        alpha_raw = int(round(rank * float(ratio)))
        note = _changed_note("alpha_ratio", ratio_raw, ratio, "snapped")
        if note:
            notes.append(note)
    else:
        alpha_raw = raw.get(
            "lora_alpha",
            rank * DEFAULT_LORA_ALPHA_MULTIPLIER,
        )
    alpha = _alpha_value(alpha_raw, rank)
    if isinstance(alpha_raw, str):
        numeric_aliases = {
            "rank": rank,
            "r": rank,
            "1r": rank,
            "2r": rank * 2,
            "4r": rank * 4,
        }
        alias_value = numeric_aliases.get(alpha_raw.strip().lower())
        if alias_value != alpha:
            notes.append(f"lora_alpha={alpha_raw!r} resolved to {alpha}")
    else:
        note = _changed_note("lora_alpha", alpha_raw, alpha, "snapped")
        if note:
            notes.append(note)

    dropout_raw = raw.get("lora_dropout", DEFAULT_LORA_DROPOUT)
    dropout = float(
        _snap(dropout_raw, VALID_LORA_DROPOUTS, "lora_dropout")
    )
    note = _changed_note("lora_dropout", dropout_raw, dropout, "snapped")
    if note:
        notes.append(note)

    decay_raw = raw.get("weight_decay", DEFAULT_WEIGHT_DECAY)
    weight_decay = float(
        _snap(decay_raw, VALID_WEIGHT_DECAYS, "weight_decay")
    )
    note = _changed_note("weight_decay", decay_raw, weight_decay, "snapped")
    if note:
        notes.append(note)

    canonical_batch_raw = raw.get("micro_batch_size")
    legacy_batch_raw = raw.get("batch_size")
    if canonical_batch_raw is None and legacy_batch_raw is None:
        batch_raw = DEFAULT_MICRO_BATCH_SIZE
    elif canonical_batch_raw is None:
        batch_raw = legacy_batch_raw
        notes.append("legacy batch_size accepted as micro_batch_size")
    elif legacy_batch_raw is None:
        batch_raw = canonical_batch_raw
    else:
        canonical_batch = int(
            _snap(
                canonical_batch_raw,
                VALID_MICRO_BATCH_SIZES,
                "micro_batch_size",
            )
        )
        legacy_batch = int(
            _snap(
                legacy_batch_raw,
                VALID_MICRO_BATCH_SIZES,
                "batch_size",
            )
        )
        if canonical_batch != legacy_batch:
            raise ValueError(
                "batch_size and micro_batch_size conflict after bounded "
                f"normalization ({legacy_batch} != {canonical_batch}); provide "
                "only micro_batch_size or make both aliases agree"
            )
        batch_raw = canonical_batch_raw
    micro_batch = int(
        _snap(batch_raw, VALID_MICRO_BATCH_SIZES, "micro_batch_size")
    )
    note = _changed_note("micro_batch_size", batch_raw, micro_batch, "snapped")
    if note:
        notes.append(note)

    accumulation_raw = raw.get(
        "gradient_accumulation_steps",
        DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    accumulation = int(
        _snap(
            accumulation_raw,
            VALID_GRADIENT_ACCUMULATION_STEPS,
            "gradient_accumulation_steps",
        )
    )
    note = _changed_note(
        "gradient_accumulation_steps",
        accumulation_raw,
        accumulation,
        "snapped",
    )
    if note:
        notes.append(note)

    effective_batch = micro_batch * accumulation
    if effective_batch > MAX_EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            f"derived effective_batch_size={effective_batch} exceeds bounded "
            f"maximum {MAX_EFFECTIVE_BATCH_SIZE}"
        )
    if raw.get("effective_batch_size") is not None:
        claimed = _integer(
            raw["effective_batch_size"],
            "effective_batch_size",
        )
        if claimed != effective_batch:
            raise ValueError(
                f"effective_batch_size={claimed} conflicts with the derived "
                f"value {micro_batch}*{accumulation}={effective_batch}; omit "
                "the derived field or make it exact"
            )

    lr_raw = raw.get("learning_rate", DEFAULT_LEARNING_RATE)
    learning_rate = _number(lr_raw, "learning_rate")
    bounded_lr = min(max(learning_rate, MIN_LEARNING_RATE), MAX_LEARNING_RATE)
    note = _changed_note(
        "learning_rate",
        lr_raw,
        bounded_lr,
        "clamped",
    )
    if note:
        notes.append(note)

    epochs_raw = raw.get("nr_epochs", DEFAULT_EPOCHS)
    epochs = _integer(epochs_raw, "nr_epochs")
    bounded_epochs = min(max(epochs, MIN_EPOCHS), MAX_EPOCHS)
    note = _changed_note("nr_epochs", epochs_raw, bounded_epochs, "clamped")
    if note:
        notes.append(note)

    config = {
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "weight_decay": weight_decay,
        "learning_rate": bounded_lr,
        "nr_epochs": bounded_epochs,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        # Keep the historical name in state/checkpoints while canonical callers use
        # micro_batch_size. It is deliberately excluded from identity.
        "batch_size": micro_batch,
    }
    memory_rationale = (
        f"micro batch={micro_batch} controls peak activation memory; gradient "
        f"accumulation={accumulation} gives derived effective batch="
        f"{effective_batch} without increasing per-step peak activation memory"
    )
    rationale = "; ".join([*notes, memory_rationale])
    return config, rationale


def hyperparameter_identity(values: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the complete canonical optimizer identity (legacy fields accepted)."""
    normalized, _ = normalize_hyperparams(values)
    return tuple(normalized[field] for field in HYPERPARAMETER_IDENTITY_FIELDS)


def deterministic_neighbor_configs(
    values: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    """Yield stable alternatives, then the remaining bounded Cartesian space."""
    base, _ = normalize_hyperparams(values)
    yielded: set[tuple[Any, ...]] = {hyperparameter_identity(base)}

    def candidate(**changes):
        raw = {
            key: base[key]
            for key in (
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "weight_decay",
                "learning_rate",
                "nr_epochs",
                "micro_batch_size",
                "gradient_accumulation_steps",
            )
        }
        raw.update(changes)
        return normalize_hyperparams(raw)[0]

    def unseen(config):
        identity = hyperparameter_identity(config)
        if identity in yielded:
            return False
        yielded.add(identity)
        return True

    rank = base["lora_rank"]
    multiplier = base["lora_alpha"] // rank
    for alternative in VALID_LORA_RANKS:
        if alternative > rank:
            config = candidate(
                lora_rank=alternative,
                lora_alpha=alternative * multiplier,
            )
            if unseen(config):
                yield config
    for alternative_multiplier in VALID_LORA_ALPHA_MULTIPLIERS:
        alternative = rank * alternative_multiplier
        if alternative != base["lora_alpha"]:
            config = candidate(lora_alpha=alternative)
            if unseen(config):
                yield config
    # Only the five tunable axes are enumerated. dropout / micro_batch_size /
    # gradient_accumulation_steps were removed: the batch-shape fields do not change
    # what the model learns (only how a batch is split to fit VRAM), and stepping
    # them produced iterations that measured run-to-run noise rather than signal.
    for field, allowed in (
        ("weight_decay", VALID_WEIGHT_DECAYS),
        ("nr_epochs", range(MIN_EPOCHS, MAX_EPOCHS + 1)),
        (
            "learning_rate",
            (1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4),
        ),
    ):
        for alternative in allowed:
            if alternative != base[field]:
                config = candidate(**{field: alternative})
                if unseen(config):
                    yield config
    for alternative in reversed(VALID_LORA_RANKS):
        if alternative < rank:
            config = candidate(
                lora_rank=alternative,
                lora_alpha=alternative * multiplier,
            )
            if unseen(config):
                yield config

    def current_first(current, allowed):
        return (current, *(item for item in allowed if item != current))

    learning_rates = current_first(
        base["learning_rate"],
        (1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4),
    )
    # Weight decay is the fastest-changing axis so a multi-axis alternative is
    # reached quickly after all one-axis neighbors are tried. Batch shape and dropout
    # are absent by design — see the one-axis loop above.
    combinations = product(
        current_first(base["lora_rank"], VALID_LORA_RANKS),
        current_first(
            multiplier,
            VALID_LORA_ALPHA_MULTIPLIERS,
        ),
        current_first(
            base["nr_epochs"],
            range(MIN_EPOCHS, MAX_EPOCHS + 1),
        ),
        learning_rates,
        current_first(base["weight_decay"], VALID_WEIGHT_DECAYS),
    )
    for (
        candidate_rank,
        candidate_multiplier,
        epochs,
        learning_rate,
        weight_decay,
    ) in combinations:
        config = candidate(
            lora_rank=candidate_rank,
            lora_alpha=candidate_rank * candidate_multiplier,
            nr_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        if unseen(config):
            yield config
