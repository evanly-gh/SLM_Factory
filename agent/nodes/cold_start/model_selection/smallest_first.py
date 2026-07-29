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
from config.android_pool import resolve_model_selector

logger = logging.getLogger(__name__)


def _plog(msg: str):
    """Print so model-selection reasoning reaches run.log (logger.info is suppressed, B161)."""
    print(f"[model_selection:smallest_first] {msg}")


def smallest_first_node(state: AgentState) -> AgentState:
    """Select the smallest feasible model (lowest on-disk weight size)."""
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

    chosen = select_smallest(feasible)
    state["selected_model"] = chosen
    _plog(f"Selected smallest of {len(feasible)} feasible: {chosen.model_id} "
          f"[{chosen.quant or 'bf16'}] (tier={chosen.tier}, size={chosen.size_mb}MB)")
    return state
