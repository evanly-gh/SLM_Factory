# agent/nodes/escalate.py
from agent.state import AgentState
from config.android_pool import filter_pool, check_hardware_constraints, all_constraints_pass


def _log(model_id: str, msg: str):
    print(f"[escalate][{model_id}] {msg}")


def escalate_node(state: AgentState) -> AgentState:
    """
    Node 8: escalate to the next model tier or terminate.

    Triggered when iterate_node detects stagnation via the sliding-window delta check.
    Clears score history and DAG; carries the dataset forward.
    """
    current_model = state["selected_model"]
    current_id = current_model.model_id if current_model else "?"

    # Log stagnation context
    scores = state.get("scores", [])
    from agent.nodes.iterate import STAGNATION_WINDOW, STAGNATION_MIN_DELTA
    window = scores[-STAGNATION_WINDOW:] if len(scores) >= STAGNATION_WINDOW else scores
    delta = max(window) - min(window) if window else 0.0

    _log(current_id, f"STAGNATION DETECTED")
    _log(current_id, f"  Score window (last {len(window)}): {[f'{s:.4f}' for s in window]}")
    _log(current_id, f"  Window delta: {delta:.4f} < threshold {STAGNATION_MIN_DELTA}")
    _log(current_id, f"  Best score achieved: {state['best_score']:.4f}")

    # Update baseline entry with final best score before we leave this model
    baselines = state.get("model_baselines") or []
    for entry in baselines:
        if entry["model_id"] == current_id:
            entry["best_finetuned_f1"] = max(entry.get("best_finetuned_f1", 0.0), state["best_score"])

    feasible = filter_pool(state["hardware_constraints"])
    _log(current_id, f"  Feasible pool ({len(feasible)} models): "
         f"{[m.model_id for m in feasible]}")

    current_idx = next(
        (i for i, m in enumerate(feasible) if m.model_id == current_model.model_id),
        None,
    )

    if current_idx is None:
        _log(current_id, f"  Current model not in feasible pool — TERMINATING")
        state["next_action"] = "terminate"
        return state

    if current_idx >= len(feasible) - 1:
        _log(current_id, f"  Already at largest feasible model (index {current_idx}/{len(feasible)-1}) — TERMINATING")
        state["next_action"] = "terminate"
        return state

    next_model = feasible[current_idx + 1]

    hw_check = check_hardware_constraints(next_model, state["hardware_constraints"])
    hw_ok = all_constraints_pass(hw_check) if state.get("hw_gating_enabled") else (
        next_model.int4_size_mb <= state["hardware_constraints"].storage_mb
        and next_model.peak_memory_mb <= state["hardware_constraints"].memory_mb
    )

    _log(current_id, f"  Next candidate: {next_model.model_id} "
         f"(tier {next_model.tier}, {next_model.int4_size_mb}MB INT4, "
         f"{next_model.peak_memory_mb}MB peak RAM)")
    _log(current_id, f"  Hardware check: storage={next_model.int4_size_mb}MB "
         f"<= {state['hardware_constraints'].storage_mb}MB? {'✓' if next_model.int4_size_mb <= state['hardware_constraints'].storage_mb else '✗'}  "
         f"memory={next_model.peak_memory_mb}MB "
         f"<= {state['hardware_constraints'].memory_mb}MB? {'✓' if next_model.peak_memory_mb <= state['hardware_constraints'].memory_mb else '✗'}")

    if not hw_ok:
        _log(current_id, f"  Next model fails hardware constraints — TERMINATING")
        state["next_action"] = "terminate"
        return state

    _log(current_id, f"  PROMOTING: {current_id} → {next_model.model_id}")
    _log(current_id, f"  Dataset carried forward: {state.get('current_dataset_path')}")
    _log(current_id, f"  Score history + DAG reset for new model")

    state["selected_model"] = next_model
    state["scores"] = []
    state["dag"] = []
    state["iteration"] = 0
    state["best_score"] = 0.0
    state["best_weights_ref"] = None
    state["last_eval"] = None
    state["last_hypothesis"] = ""
    state["llm_iterate_decision"] = None
    state["consecutive_no_improvement"] = 0
    state["next_action"] = "train"
    return state
