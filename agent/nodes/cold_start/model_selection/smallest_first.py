# agent/nodes/cold_start/model_selection/smallest_first.py
"""
Strategy: Smallest-First

Start with the smallest feasible model. If the accuracy goal is not met after
the training loop stagnates, escalation moves to the next tier up. This is
the most resource-conservative strategy — it only uses a larger model when
the smaller one is proven insufficient.

No probing overhead; the very first curate→train→evaluate cycle uses the
production training config (not a 1-epoch probe).
"""
import logging
import os

from agent.state import AgentState
from agent.nodes.cold_start.model_selection.base import select_smallest

logger = logging.getLogger(__name__)


def smallest_first_node(state: AgentState) -> AgentState:
    """Select the smallest feasible model (lowest peak RAM)."""
    feasible = state.get("feasible_models", [])
    if not feasible:
        raise RuntimeError("smallest_first_node: feasible_models is empty.")

    forced = os.environ.get("SLM_FORCE_MODEL")
    if forced:
        match = next((m for m in feasible if m.model_id == forced), None)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
        logger.info("[model_selection:smallest_first] SLM_FORCE_MODEL=%s pinned", forced)
        return state

    chosen = select_smallest(feasible)
    state["selected_model"] = chosen
    logger.info(
        "[model_selection:smallest_first] Selected %s (tier=%d, quant=%s, peak=%dMB)",
        chosen.model_id, chosen.tier, chosen.quant, chosen.peak_memory_mb,
    )
    return state
