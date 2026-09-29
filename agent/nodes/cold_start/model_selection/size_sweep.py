"""`size_sweep` — the naive exhaustive baseline: every feasible variant, smallest to largest.

WHY THIS EXISTS
    `smallest_first` climbs the pool one TIER at a time and lets the orchestrator pick one variant
    inside each tier, so a run that ends at tier 3 has trained perhaps three of the 27 variants. That
    is the saving the tiering system claims. This strategy removes it: no tiers, no orchestrator
    choice of model, no skipping. The run starts on the smallest variant by on-disk size and, each
    time a model stagnates or hits the eval cap, moves to the next-larger variant in the pool —
    until there are none left.

    So `size_sweep` vs `smallest_first` on the same task measures what the tier grouping and the
    orchestrator's within-tier pick buy: how much wall clock they save, and how much accuracy they
    leave on the table by never training the variants they skipped.

BEHAVIOUR
    Everything inside a model's run is unchanged — hyperparameter search, data rebuilds, rollback,
    the calibrated accuracy goal, the curriculum carried forward on promotion. Only the choice of
    the NEXT model differs, and it is made here, deterministically, rather than by the orchestrator.

    A sweep visits every variant only if the launcher makes the goal unreachable
    (`SLM_STOP_THRESHOLD=0.99`). Left calibrated, a sweep that meets the goal takes the normal
    convergence route and stops there, which is the ordinary pipeline's answer, not an exhaustive
    one.
"""
from __future__ import annotations

import os

from agent.state import AgentState
from config.android_pool import ModelSpec, resolve_model_selector


def _plog(msg: str):
    print(f"[model_selection:size_sweep] {msg}")


def _sweep_key(model: ModelSpec) -> tuple[int, str]:
    return (model.size_mb, model.selector)


def sweep_order(feasible: list[ModelSpec]) -> list[ModelSpec]:
    """Every feasible variant, smallest on-disk size first; ties broken by selector."""
    return sorted(feasible, key=_sweep_key)


def next_in_sweep(feasible: list[ModelSpec], current: ModelSpec) -> ModelSpec | None:
    """The next-larger variant after `current`, or None when the sweep is exhausted.

    Keyed on (size_mb, selector) rather than list position, so the answer does not depend on whether
    `current` is itself still in `feasible` or on how the caller ordered the list.
    """
    current_key = _sweep_key(current)
    later = [model for model in sweep_order(feasible) if _sweep_key(model) > current_key]
    return later[0] if later else None


def size_sweep_node(state: AgentState) -> AgentState:
    """Start the sweep on the smallest feasible variant."""
    feasible = state.get("feasible_models", [])
    if not feasible:
        raise RuntimeError("size_sweep_node: feasible_models is empty.")

    forced = os.environ.get("SLM_FORCE_MODEL")
    if forced:
        match = resolve_model_selector(feasible, forced)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
        _plog(f"SLM_FORCE_MODEL={forced} pinned → {match.selector} (sweep continues upward from it)")
        return state

    order = sweep_order(feasible)
    chosen = order[0]
    _plog(
        f"Sweeping all {len(order)} feasible variant(s) in on-disk size order, no tiers and no "
        f"orchestrator model choice: {[m.selector for m in order]}"
    )
    _plog(
        f"Selected {chosen.model_id} [{chosen.quant or 'bf16'}] "
        f"(tier={chosen.tier}, size={chosen.size_mb}MB) — sweep position 1/{len(order)}"
    )
    state["selected_model"] = chosen
    return state
