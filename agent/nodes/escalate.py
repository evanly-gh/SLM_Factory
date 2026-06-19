# agent/nodes/escalate.py
from agent.state import AgentState
from android_pool import filter_pool


def escalate_node(state: AgentState) -> AgentState:
    """
    Node 8: escalate to next model tier or terminate.
    Escalation checks hardware constraints even in Phase 1 (informational, not blocking
    for latency/power, but storage/memory remain hard gates).
    """
    current_model = state["selected_model"]
    feasible = filter_pool(state["hardware_constraints"])

    # Find current model's position in the feasible list
    current_idx = next(
        (i for i, m in enumerate(feasible) if m.model_id == current_model.model_id),
        None,
    )

    if current_idx is None or current_idx >= len(feasible) - 1:
        # No larger model available — terminate
        state["next_action"] = "terminate"
        return state

    next_model = feasible[current_idx + 1]

    # Storage and memory are hard gates even in Phase 1
    hw_ok = (
        next_model.int4_size_mb <= state["hardware_constraints"].storage_mb
        and next_model.peak_memory_mb <= state["hardware_constraints"].memory_mb
    )

    if not hw_ok:
        state["next_action"] = "terminate"
        return state

    state["selected_model"] = next_model
    state["consecutive_no_improvement"] = 0
    state["next_action"] = "train"
    return state
