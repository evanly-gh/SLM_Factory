"""`single_model` — the naive baseline: one model, chosen once, never changed.

WHY THIS EXISTS
    Every other strategy climbs (and now descends) the model ladder, so a run's final score mixes
    two contributions that cannot be separated after the fact: what the iteration loop achieved on
    a given model, and what changing models achieved. This strategy removes the second one.

    It is deliberately close to the naive human workflow the project is trying to beat — pick a
    model, pick a curriculum, pick hyperparameters, train — with one difference: everything else in
    the loop still runs. Hyperparameter search, data rebuilds, rollback on regression, the calibrated
    accuracy goal and the stretch-goal machinery are all unchanged. Only the ladder is off.

    So `single_model` vs `smallest_first` on the same task isolates the ladder's contribution, and
    `single_model` vs a hand-run single fine-tune isolates the loop's.

BEHAVIOUR
    Selection reuses `orchestrator_choice_node` verbatim — the orchestrator picks the starting model
    from the hardware-feasible set using task context and benchmark affinities. From then on:

      * failing the goal does NOT escalate — the run terminates when it stagnates or hits the eval
        cap, and reports its best score honestly as "did not converge on this model";
      * meeting the goal does NOT trigger a downward probe — the answer is the model that was
        chosen, not the smallest one that happens to clear the bar.

    Both gates are enforced in `agent/nodes/iterate.py` via `config.model_ladder_enabled()`, not
    here, because that is where the routing decisions are made.
"""
from __future__ import annotations

from agent.nodes.cold_start.model_selection.orchestrator_choice import (
    orchestrator_choice_node,
)
from agent.state import AgentState


def single_model_node(state: AgentState) -> AgentState:
    """Pick one model via the orchestrator and pin the run to it."""
    state = orchestrator_choice_node(state)
    selected = state.get("selected_model")
    label = getattr(selected, "selector", None) or getattr(selected, "model_id", "?")
    print(
        f"      [model_selection] single_model: PINNED to {label} for the whole run — "
        "no escalation on failure, no downward regression on success. The rest of the loop "
        "(hyperparameters, data rebuilds, rollback, accuracy goal) runs unchanged."
    )
    return state
