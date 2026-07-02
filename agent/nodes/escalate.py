# agent/nodes/escalate.py
from agent.state import AgentState
from config.android_pool import filter_pool, check_hardware_constraints, all_constraints_pass


def escalate_node(state: AgentState) -> AgentState:
    """
    Node 8: escalate to the next model tier or terminate.

    Triggered when iterate_node detects stagnation: the total improvement across
    the last STAGNATION_WINDOW evaluation runs was below STAGNATION_MIN_DELTA
    (defined in iterate.py). This delta-based trigger is more robust than a
    consecutive-zero count — it escalates on genuine plateaus, not just slow progress.

    On escalation: clears score history and DAG so the new model starts from zero.
    The dataset path is carried forward — the curated data is still valid for the new model.
    The stop_threshold and initial_stop_threshold are preserved (the goal doesn't change
    just because we switched models).

    Hardware constraints are checked before promoting the next model (informational in
    Phase 1, hard gate in Phase 2).
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

    hw_check = check_hardware_constraints(next_model, state["hardware_constraints"])
    hw_ok = all_constraints_pass(hw_check) if state.get("hw_gating_enabled") else (
        next_model.int4_size_mb <= state["hardware_constraints"].storage_mb
        and next_model.peak_memory_mb <= state["hardware_constraints"].memory_mb
    )

    if not hw_ok:
        state["next_action"] = "terminate"
        return state

    print(
        f"[escalate] Stagnation detected. "
        f"Promoting {current_model.model_id} → {next_model.model_id}"
    )

    state["selected_model"] = next_model
    # Reset trajectory — old model scores are irrelevant to the new model's rollback gate.
    # Dataset path is preserved: the curated data carries over.
    state["scores"] = []
    state["dag"] = []
    state["iteration"] = 0
    state["best_score"] = 0.0
    state["best_weights_ref"] = None
    state["last_eval"] = None
    state["last_hypothesis"] = ""
    state["llm_iterate_decision"] = None
    state["consecutive_no_improvement"] = 0  # kept for compatibility with evaluate_node logging
    state["next_action"] = "train"
    return state
