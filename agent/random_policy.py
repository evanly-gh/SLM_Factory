# agent/random_policy.py
"""The random-policy ablation: the iterate node's intervention CHOICE, made by a seeded draw.

Selected with SLM_ABLATION_POLICY=random (seed: SLM_ABLATION_POLICY_SEED, default SLM_SEED).

WHAT IT ANSWERS
    The search loop's gains combine the orchestrator's choices with deterministic rules around them
    (rollback on regression, the run-health watchdog, the proportional synthetic-row allocation),
    and most interventions are rolled back. Replacing ONLY the choice with a uniform draw, under
    otherwise identical rules, measures how much of the gain the choice itself contributes.

WHAT IS REPLACED, AND WHAT IS NOT
    Replaced: the `_llm_iterate` call. This module returns a decision dict in exactly the shape the
    orchestrator returns, and `iterate_node` validates, logs, stores and routes it through the same
    code (`_validate_decision_json`, `state["llm_iterate_decision"]`, `last_intervention`,
    `last_hypothesis` -> DAG node `intervention`/`hypothesis` and data-curation.md).

    Not replaced: everything before and after the choice. Convergence routing, stagnation / eval-cap
    handling, rollback, curate's mining ladder, the 25-600-rows-per-category proportional synthesis
    allocation, the exhaustion skip of already-targeted categories, run health — all unchanged.

THE OPTION SET (the same constraints the orchestrator is held to)
    hyperparameter      if the current best config has at least one untried one-step neighbour
    mine_new_real       if `mining_available_for_state` (the validator's own gate)
    surgical_synthesis  if `teacher_fitness.synthesis_allowed` (the validator's own gate)
    One is drawn uniformly. If none is available (no untried neighbour and no data route) the
    hyperparameter step falls back to the first untried config in the deterministic multi-step
    search order (`training.hparams.deterministic_neighbor_configs`), which is the same fallback the
    trainer applies to a config-less hyperparameter step.

HYPERPARAMETERS
    A uniformly drawn neighbour of the current best config (`train._best_prior_config`, the config
    the trainer would otherwise carry forward), one step along one of the five tunable axes on the
    ladders in `training.hparams.one_step_neighbor_configs`, excluding every five-knob identity
    already trained in this run (on any dataset version, rolled-back trials included).

DATA REBUILDS — the defaults the code applies when the orchestrator gives no specifics
    rows               300 — `normalize_data_rebuild_plan`'s default and `fallback_data_rebuild_plan`'s
    target_categories  [] — curate's `_eligible_categories` then takes the latest test report's
                       failure categories, most failures first, skips exhausted ones, keeps <= 5
    pattern_hint       "" — no free-text steer into the generation prompt
    For synthesis the 300 rows are split across the eligible categories in proportion to their
    failure counts and clamped to 25-600 per category (`curate.SURGICAL_*_ROWS_PER_CATEGORY`).

DETERMINISM AND RESUME
    Each decision draws from its own `random.Random(f"...:{seed}:{iteration}")`. `iteration` is
    unique per iterate call on a pinned model (train and the empty-rebuild skip both advance it), so
    a requeued run redraws the same decision for the same turn without any RNG state in the
    checkpoint.
"""
from __future__ import annotations

import random
from typing import Any, Callable

from agent.ablations import policy_seed
from agent.data_rebuild import (
    DATA_REBUILD_SCHEMA_VERSION,
    MINE_NEW_REAL,
    SURGICAL_SYNTHESIS,
    mining_available_for_state,
)
from training.hparams import (
    deterministic_neighbor_configs,
    normalize_hyperparams,
    one_step_neighbor_configs,
    tunable_identity,
    tunable_knobs,
)

HYPERPARAMETER = "hyperparameter"
DEFAULT_REBUILD_ROWS = 300


def _tried_tunable_identities(state) -> set[tuple]:
    """Five-knob identities of every config trained in this run, rolled-back ones included."""
    from agent.nodes.train import _node_tried_configs

    tried: set[tuple] = set()
    for node in state.get("dag") or []:
        for config in _node_tried_configs(node):
            tried.add(tunable_identity(config))
    return tried


def _best_config(state) -> dict:
    """The config the trainer carries forward: best non-pruned trained H, else the default."""
    from agent.nodes.train import _DEFAULT_CONFIG, _best_prior_config

    best = _best_prior_config(state)
    return normalize_hyperparams(best if best is not None else _DEFAULT_CONFIG)[0]


def _label(config: dict) -> str:
    knobs = tunable_knobs(config)
    return (
        f"r={knobs['lora_rank']} ratio={knobs['alpha_ratio']} wd={knobs['weight_decay']:g} "
        f"lr={knobs['learning_rate']:.0e} ep={knobs['nr_epochs']}"
    )


def random_decision(state, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Draw one intervention decision, in the orchestrator's decision-JSON shape.

    The draw is described in the returned `hypothesis`, which iterate_node logs and stores like an
    orchestrator hypothesis; `log` is used only for the no-option fallback.
    """
    from agent.teacher_fitness import synthesis_allowed

    seed = policy_seed()
    iteration = int(state.get("iteration", 0) or 0)
    rng = random.Random(f"slm-random-policy:{seed}:{iteration}")

    best = _best_config(state)
    tried = _tried_tunable_identities(state)
    neighbours = one_step_neighbor_configs(best)
    untried = [entry for entry in neighbours if tunable_identity(entry[3]) not in tried]

    options: list[str] = []
    if untried:
        options.append(HYPERPARAMETER)
    if mining_available_for_state(state):
        options.append(MINE_NEW_REAL)
    if synthesis_allowed(state):
        options.append(SURGICAL_SYNTHESIS)
    unavailable = [
        name for name in (HYPERPARAMETER, MINE_NEW_REAL, SURGICAL_SYNTHESIS) if name not in options
    ]

    header = f"RANDOM POLICY (seed={seed}, iteration={iteration})"
    if not options:
        # Nothing on the menu: no untried one-step neighbour and no data route. Take the first
        # untried config further out, in the deterministic search order the trainer itself uses.
        for candidate in deterministic_neighbor_configs(best):
            if tunable_identity(candidate) not in tried:
                knobs = tunable_knobs(candidate)
                hypothesis = (
                    f"{header}: no option available (every one-step neighbour of the best config "
                    f"{_label(best)} is tried, mining and synthesis are unavailable); fallback to "
                    f"the first untried config in deterministic search order: {_label(candidate)}."
                )
                log(f"  [random-policy] {hypothesis}")
                return {
                    "intervention": HYPERPARAMETER,
                    "hypothesis": hypothesis,
                    "hyperparams": knobs,
                }
        raise RuntimeError(
            "random policy: the bounded hyperparameter space is exhausted and no data route is "
            "available; nothing left to try"
        )

    choice = rng.choice(options)
    menu = (
        f"chose {choice} uniformly from {options}"
        + (f" (unavailable this turn: {unavailable})" if unavailable else "")
    )

    if choice == HYPERPARAMETER:
        axis, old, new, knobs = rng.choice(untried)
        hypothesis = (
            f"{header}: {menu}; one-step neighbour {axis} {old} -> {new} of the best config "
            f"{_label(best)}, drawn uniformly from {len(untried)} untried of {len(neighbours)} "
            f"neighbour(s). No failure analysis was used."
        )
        return {
            "intervention": HYPERPARAMETER,
            "hypothesis": hypothesis,
            "hyperparams": knobs,
        }

    hypothesis = (
        f"{header}: {menu}; plan uses the code defaults (rows={DEFAULT_REBUILD_ROWS}, target "
        f"categories from the latest test report, no pattern hint). No failure analysis was used."
    )
    return {
        "intervention": "data_rebuild",
        "hypothesis": hypothesis,
        "data_rebuild": {
            "schema_version": DATA_REBUILD_SCHEMA_VERSION,
            "strategy": choice,
            "rows": DEFAULT_REBUILD_ROWS,
            "target_categories": [],
            "pattern_hint": "",
        },
    }
