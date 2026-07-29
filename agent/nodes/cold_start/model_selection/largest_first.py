# agent/nodes/cold_start/model_selection/largest_first.py
"""
Strategy: Largest-First (Feasibility Probe)

Start with the largest feasible model to establish whether the task is solvable
at all within the hardware budget. If the largest model cannot reach the
accuracy goal, no smaller model will either — terminate early.

If the largest model succeeds, switch to the smallest model and escalate from
there (resource-optimized). This two-phase approach spends one extra training
run upfront but avoids wasting many runs on a task that is infeasible.
"""
import logging
import os

from agent.state import AgentState
from agent.nodes.cold_start.model_selection.base import select_largest, select_smallest
from config.android_pool import resolve_model_selector

logger = logging.getLogger(__name__)


def _plog(msg: str):
    """Print so model-selection reasoning reaches run.log (logger.info is suppressed, B161)."""
    print(f"[model_selection:largest_first] {msg}")


def largest_first_node(state: AgentState) -> AgentState:
    """
    Select the largest feasible model as a feasibility probe.

    Sets state["_largest_first_phase"] = "probe" so that iterate_node can
    detect when the probe succeeds and switch to the smallest model.
    """
    feasible = state.get("feasible_models", [])
    if not feasible:
        raise RuntimeError("largest_first_node: feasible_models is empty.")

    forced = os.environ.get("SLM_FORCE_MODEL")
    if forced:
        match = resolve_model_selector(feasible, forced)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
        logger.info("[model_selection:largest_first] SLM_FORCE_MODEL=%s pinned", forced)
        return state

    chosen = select_largest(feasible)
    state["selected_model"] = chosen
    state["_largest_first_phase"] = "probe"
    _plog(f"Probe phase — selected LARGEST of {len(feasible)} feasible: {chosen.model_id} "
          f"[{chosen.quant or 'bf16'}] (tier={chosen.tier}, size={chosen.size_mb}MB). "
          f"If it clears the goal → switch to smallest; if it stalls → task infeasible.")
    return state


def check_probe_result(state: AgentState) -> AgentState:
    """
    Called after the probe model converges or stagnates.

    If the probe model reached the accuracy goal, switch to the smallest
    feasible model and let normal escalation take over.
    If the probe model stagnated without reaching the goal, terminate —
    the task is infeasible within the hardware budget.
    """
    phase = state.get("_largest_first_phase")
    if phase != "probe":
        return state

    current_score = state["scores"][-1] if state["scores"] else 0.0
    threshold = state["stop_threshold"]

    if current_score >= threshold:
        feasible = state.get("feasible_models", [])
        smallest = select_smallest(feasible)
        if smallest.selector == state["selected_model"].selector:
            logger.info(
                "[model_selection:largest_first] Probe succeeded and smallest == largest; done"
            )
            state["_largest_first_phase"] = "done"
            return state

        logger.info(
            "[model_selection:largest_first] Probe succeeded (%.4f >= %.4f). "
            "Switching to smallest model: %s (tier=%d, size=%dMB)",
            current_score, threshold, smallest.model_id, smallest.tier, smallest.size_mb,
        )
        state["selected_model"] = smallest
        state["scores"] = []
        state["dag"] = []
        state["iteration"] = 0
        state["lifetime_best_score"] = max(
            state.get("lifetime_best_score") or 0.0, state["best_score"]
        )
        state["best_score"] = 0.0
        state["best_weights_ref"] = None
        state["last_eval"] = None
        state["last_intervention"] = "data_rebuild"
        state["last_hypothesis"] = ""
        state["llm_iterate_decision"] = None
        state["data_rebuild_plan"] = None
        state["data_rebuild_plan_identity"] = None
        state["consecutive_no_improvement"] = 0
        state["downward_probe_done"] = False
        _plog(f"Probe SUCCEEDED ({current_score:.4f} >= {threshold:.4f}) — task is feasible; "
              f"switching to smallest model {smallest.model_id} [{smallest.quant or 'bf16'}] "
              f"and escalating from there.")
        state["_largest_first_phase"] = "escalate"
        state["next_action"] = "curate"
    else:
        _plog(f"Probe did NOT clear the goal ({current_score:.4f} < {threshold:.4f}) — "
              f"the largest feasible model can't reach it, so no smaller one will either.")
        state["_largest_first_phase"] = "done"

    return state
