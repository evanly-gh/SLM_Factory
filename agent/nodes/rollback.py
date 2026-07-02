# agent/nodes/rollback.py
from agent.state import AgentState


def _log(model_id: str, msg: str):
    print(f"[rollback][{model_id}] {msg}")


def should_rollback(state: AgentState) -> bool:
    """
    Cold-start simple rollback: if f(π_i+1) < f(π_i), revert.
    No dual-gate in cold-start (that is production mode only).
    """
    scores = state["scores"]
    if len(scores) < 2:
        return False
    return scores[-1] < scores[-2]


def rollback_node(state: AgentState) -> AgentState:
    """
    Node 6: revert to previous configuration if score decreased.
    Removes the last score from history.
    Restores best_weights_ref to the best non-pruned DAG node.
    """
    if not should_rollback(state):
        return state

    model_id = state["selected_model"].model_id if state.get("selected_model") else "?"
    regressed_score = state["scores"][-1]
    previous_score = state["scores"][-2] if len(state["scores"]) >= 2 else 0.0

    _log(model_id, f"REGRESSION detected: {regressed_score:.4f} < {previous_score:.4f} "
         f"(Δ={regressed_score - previous_score:+.4f})")

    # Remove the regressing score
    state["scores"].pop()
    state["last_intervention"] = "rollback"
    state["consecutive_no_improvement"] += 1

    # Mark the last DAG node as pruned
    if state["dag"]:
        pruned_node = state["dag"][-1]
        pruned_node["pruned"] = True
        _log(model_id, f"  Pruned DAG node: iteration={pruned_node['iteration']}  "
             f"config={pruned_node.get('best_config', '?')}  "
             f"score={pruned_node['score']:.4f}")

    # Restore best_weights_ref to the most recent non-pruned DAG node
    non_pruned = [n for n in state["dag"] if not n.get("pruned", False)]
    if non_pruned:
        best_node = max(non_pruned, key=lambda n: n["score"])
        state["best_weights_ref"] = best_node["weights_ref"]
        state["best_score"] = best_node["score"]
        _log(model_id, f"  Restored to: iteration={best_node['iteration']}  "
             f"score={best_node['score']:.4f}  "
             f"weights={best_node['weights_ref']}")
    else:
        _log(model_id, "  WARNING: all DAG nodes pruned, no checkpoint to restore")

    _log(model_id, f"  Proceeding to re-train on existing dataset")

    return state
