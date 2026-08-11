"""Deterministic curriculum-size targeting, recomputed per model tier.

The target answers one question: *how much training data does THIS model need for THIS task?*
Two signals drive it, and both are measured rather than guessed:

1. **Task novelty** — how far the task sits outside the model's pretraining distribution. The
   zero-shot baseline measures exactly this: a model that already scores 0.85 on the task has
   largely seen it before and needs little data; one that scores 0.10 needs a lot.
   ``novelty = 1 - zero_shot_baseline_f1``.

2. **Model capacity** — smaller models need more examples to reach the same accuracy, so the
   target scales *inversely* with parameter count.

This is deliberately NOT an LLM decision. The orchestrator has no way to estimate novelty from a
task description — it would be guessing, and the guess would look authoritative. The zero-shot
baseline is an empirical measurement of the same quantity and is already computed every run.

Recomputed on entry to each tier (initial selection, escalation, and downward regression), because
both inputs change when the model changes.
"""
from __future__ import annotations

import os

# Bounds on the capacity multiplier. A 1B model is the reference point (factor 1.0); the clamp
# stops a very small model from demanding an unreachable amount of data, or a very large one from
# collapsing the target to nothing.
_SIZE_FACTOR_MIN = 0.5
_SIZE_FACTOR_MAX = 2.0
_REFERENCE_PARAMS = 1.0e9

# Novelty is folded in as (NOVELTY_BASE + novelty), so a task the model already knows perfectly
# (novelty 0) still gets a real curriculum rather than nothing.
_NOVELTY_BASE = 0.5


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def size_factor_for_params(n_params: float | None) -> float:
    """Capacity multiplier: smaller model -> more data. Unknown size -> neutral 1.0."""
    if not n_params or n_params <= 0:
        return 1.0
    return _clamp(_REFERENCE_PARAMS / float(n_params), _SIZE_FACTOR_MIN, _SIZE_FACTOR_MAX)


def compute_curriculum_target(
    *,
    zero_shot_baseline: float | None,
    n_params: float | None,
    floor: int,
    ceiling: int,
) -> tuple[int, str]:
    """Return ``(target_rows, rationale)`` for the current model/task pair.

    ``zero_shot_baseline`` is the untrained model's score on the frozen eval set. When it is not
    available yet (the very first sizing, before any eval has run) novelty defaults to 0.5 —
    the neutral midpoint — rather than biasing the first target high or low.
    """
    if zero_shot_baseline is None:
        novelty = 0.5
        novelty_note = "no baseline yet, assuming neutral novelty 0.50"
    else:
        novelty = _clamp(1.0 - float(zero_shot_baseline), 0.0, 1.0)
        novelty_note = f"novelty {novelty:.3f} = 1 - baseline {float(zero_shot_baseline):.4f}"

    factor = size_factor_for_params(n_params)
    raw = floor * (_NOVELTY_BASE + novelty) * factor
    target = int(_clamp(round(raw), floor, ceiling))
    params_note = (
        f"{n_params / 1e9:.2f}B params -> size factor {factor:.2f}"
        if n_params else "unknown params -> size factor 1.00"
    )
    rationale = (
        f"{novelty_note}; {params_note}; "
        f"{floor} x ({_NOVELTY_BASE} + {novelty:.3f}) x {factor:.2f} = {raw:.0f} "
        f"-> clamped to [{floor}, {ceiling}] = {target}"
    )
    return target, rationale


def resize_curriculum_for_tier(state, *, log=print) -> int:
    """Recompute and store ``curriculum_size_target`` for the currently selected model.

    Called on entry to every tier. An explicit ``SLM_CURRICULUM_SIZE`` override always wins so a
    test or a manual run can pin the size.
    """
    from config.config import CURRICULUM_SIZE_FLOOR, DATA_SIZE_CEILING

    override = os.environ.get("SLM_CURRICULUM_SIZE")
    if override:
        try:
            pinned = int(override)
        except (TypeError, ValueError):
            pinned = CURRICULUM_SIZE_FLOOR
        state["curriculum_size_target"] = pinned
        log(f"      [sizing] curriculum target pinned by SLM_CURRICULUM_SIZE={pinned}")
        return pinned

    model = state.get("selected_model")
    n_params = _params_for_model(model)
    baseline = _baseline_for_selected_model(state)

    target, rationale = compute_curriculum_target(
        zero_shot_baseline=baseline,
        n_params=n_params,
        floor=CURRICULUM_SIZE_FLOOR,
        ceiling=DATA_SIZE_CEILING,
    )
    previous = state.get("curriculum_size_target")
    state["curriculum_size_target"] = target
    label = getattr(model, "label", None) or getattr(model, "model_id", "?")
    change = f" (was {previous})" if previous and previous != target else ""
    log(f"      [sizing] curriculum target for {label}: {target}{change} — {rationale}")
    return target


def baseline_is_known(state) -> bool:
    """Whether a zero-shot baseline has been measured for the currently selected model."""
    return _baseline_for_selected_model(state) is not None


def _params_for_model(model) -> float | None:
    """Absolute parameter count for a ``config.android_pool.ModelSpec``.

    ModelSpec exposes ``est_params_b`` — parameters in BILLIONS, derived from on-disk weight
    size divided by the quant's bytes-per-parameter. It has no ``params``/``n_params``
    attribute, so looking for those returned None and every model silently got the neutral
    capacity factor 1.0, collapsing the target to the floor (B241).
    """
    if model is None:
        return None
    billions = getattr(model, "est_params_b", None)
    # It is a METHOD on ModelSpec, not a property (unlike `selector`). Reading it without
    # calling yields a truthy bound method, so a naive getattr looks like it succeeded and then
    # silently degrades to the neutral factor — the same way the original bug hid.
    if callable(billions):
        try:
            billions = billions()
        except Exception:  # noqa: BLE001 — sizing must never break the run
            billions = None
    if billions:
        try:
            return float(billions) * 1e9
        except (TypeError, ValueError):
            return None
    explicit = getattr(model, "params", None) or getattr(model, "n_params", None)
    return float(explicit) if explicit else None


def _baseline_for_selected_model(state) -> float | None:
    """Zero-shot baseline recorded for the currently selected model, if one exists yet."""
    model = state.get("selected_model")
    if model is None:
        return None
    selector = getattr(model, "selector", None) or getattr(model, "model_id", None)
    for entry in state.get("model_baselines") or []:
        if entry.get("selector", entry.get("model_id")) == selector:
            value = entry.get("baseline_f1")
            return float(value) if value is not None else None
    return None
