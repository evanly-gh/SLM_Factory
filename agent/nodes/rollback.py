# agent/nodes/rollback.py
from agent.state import AgentState


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
    Node 7: revert to previous configuration if score decreased.
    Removes the last score from history.
    Restores best_weights_ref to the best non-pruned DAG node.
    """
    if not should_rollback(state):
        return state

    # Remove the regressing score
    state["scores"].pop()
    state["last_intervention"] = "rollback"
    state["consecutive_no_improvement"] += 1

    # Mark the last DAG node as pruned
    if state["dag"]:
        state["dag"][-1]["pruned"] = True

    # Restore best_weights_ref to the most recent non-pruned DAG node
    non_pruned = [n for n in state["dag"] if not n.get("pruned", False)]
    if non_pruned:
        best_node = max(non_pruned, key=lambda n: n["score"])
        state["best_weights_ref"] = best_node["weights_ref"]
        state["best_score"] = best_node["score"]

    return state
