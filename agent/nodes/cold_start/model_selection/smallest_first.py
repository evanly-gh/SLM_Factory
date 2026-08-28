# agent/nodes/cold_start/model_selection/smallest_first.py
"""
Strategy: Smallest-First

Start in the smallest feasible TIER and let the orchestrator pick the best model within it.
If the training loop stagnates without reaching the goal, escalation moves to the next tier up.
This is the most resource-conservative strategy — it only spends a larger tier when the smaller
one is proven insufficient.

WHY A TIER AND NOT THE SMALLEST MODEL (changed 2026-08-24)
    This used to call `select_smallest(feasible)` and take the single lowest `size_mb` entry.
    That was defensible while the pool was one family, where size ordered capability closely
    enough. It stopped being defensible when the sub-billion models were measured:

        google/gemma-3-270m-it@Q4_K_M          241MB   xlam_bfcl 0.10   ner_bc5cdr 0.65
        HuggingFaceTB/SmolLM2-360M-Instruct@Q4_K_M  258MB   xlam_bfcl 0.45   ner_bc5cdr 0.73

    17MB apart, and on a structured-output task one scores 4.5x the other. `select_smallest`
    takes the 241MB entry every time, on every task, because file size is all it can see.
    Gemma spends 63% of its parameters on a 262k-token vocabulary; that buys rare-token
    extraction and costs compositional depth. No ordering over `size_mb` can express it.

    So the choice inside a tier is now the orchestrator's, made against the capability
    descriptions and our own measured task scores — the same machinery escalation already uses
    to choose within the tier it promotes into. The tier itself is still chosen by pure
    resource conservatism, which is the part of "smallest first" that was actually load-bearing.

    Escalation is unchanged and still the safety net: picking the weaker model in tier 1 costs
    iterations, not the run.

No probing overhead; the very first curate→train→evaluate cycle uses the production training
config (not a 1-epoch probe).
"""
import logging
import os

from agent.state import AgentState
from config.android_pool import resolve_model_selector

logger = logging.getLogger(__name__)


def _plog(msg: str):
    """Print so model-selection reasoning reaches run.log (logger.info is suppressed, B161)."""
    print(f"[model_selection:smallest_first] {msg}")


def smallest_tier_candidates(feasible):
    """Every feasible variant in the lowest occupied tier.

    Returned as a list rather than one model because the tier is the resource decision and the
    model within it is a task-fit decision; they are made by different things.
    """
    lowest = min(model.tier for model in feasible)
    return [model for model in feasible if model.tier == lowest], lowest


def smallest_first_node(state: AgentState) -> AgentState:
    """Select a model from the smallest feasible tier, orchestrator-chosen within it."""
    feasible = state.get("feasible_models", [])
    if not feasible:
        raise RuntimeError("smallest_first_node: feasible_models is empty.")

    forced = os.environ.get("SLM_FORCE_MODEL")
    if forced:
        match = resolve_model_selector(feasible, forced)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
        _plog(f"SLM_FORCE_MODEL={forced} pinned → {match.selector}")
        return state

    candidates, tier = smallest_tier_candidates(feasible)
    _plog(
        f"Smallest feasible tier is {tier} — {len(candidates)} candidate(s), "
        f"{min(m.size_mb for m in candidates)}-{max(m.size_mb for m in candidates)}MB: "
        f"{[m.selector for m in candidates]}"
    )

    if len(candidates) == 1:
        chosen = candidates[0]
        _plog(f"Only one candidate in tier {tier}; no choice to make → {chosen.selector}")
        state["selected_model"] = chosen
        return state

    # Same helper escalation uses, so an initial pick and a promoted pick are made by identical
    # machinery against identical evidence. On any failure it falls back to the smallest candidate
    # in the tier, which is exactly the old behaviour — so this degrades to the previous strategy
    # rather than to something unpredictable.
    from agent.nodes.escalate import _llm_choose_model

    chosen = _llm_choose_model(
        candidates=candidates,
        task=state.get("task", ""),
        task_plan=state.get("task_plan", {}) or {},
        current_best_score=0.0,
        log=_plog,
        direction="initial",
    )
    state["selected_model"] = chosen
    _plog(
        f"Selected {chosen.model_id} [{chosen.quant or 'bf16'}] "
        f"(tier={chosen.tier}, size={chosen.size_mb}MB) from tier {tier}"
    )
    return state
