# agent/nodes/escalate.py
from agent.state import AgentState
from android_pool import filter_pool


def escalate_node(state: AgentState) -> AgentState:
    """
    Node 8: escalate to next model tier or terminate.

    On escalation, resets the training trajectory (scores, dag, iteration) so the new
    model starts fresh — the old model's scores are irrelevant and would corrupt rollback
    logic. The dataset is carried forward (best_dataset_version is preserved) but the
    score history is cleared.

    Escalation checks hardware constraints even in Phase 1 (informational for latency/power,
    hard gate for storage/memory). Design doc §2.4 Node 8.
    """
    current_model = state["selected_model"]
    feasible = filter_pool(state["hardware_constraints"])

    current_idx = next(
        (i for i, m in enumerate(feasible) if m.model_id == current_model.model_id),
        None,
    )

    if current_idx is None or current_idx >= len(feasible) - 1:
        state["next_action"] = "terminate"
        return state

    next_model = feasible[current_idx + 1]

    hw_ok = (
        next_model.int4_size_mb <= state["hardware_constraints"].storage_mb
        and next_model.peak_memory_mb <= state["hardware_constraints"].memory_mb
    )

    if not hw_ok:
        state["next_action"] = "terminate"
        return state

    state["selected_model"] = next_model
    state["consecutive_no_improvement"] = 0
    state["scores"] = []
    state["dag"] = []
    state["iteration"] = 0
    state["best_score"] = 0.0
    state["best_weights_ref"] = None
    state["last_eval"] = None
    state["last_hypothesis"] = ""
    state["llm_iterate_decision"] = None
    state["next_action"] = "curate"
    return state
