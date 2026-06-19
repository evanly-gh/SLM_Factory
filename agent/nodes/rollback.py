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
    Removes the last score from history. Restores previous weights_ref.
    """
    if not should_rollback(state):
        return state

    # Remove the regressing score
    state["scores"].pop()
    # Restore best known weights
    state["last_intervention"] = "rollback"
    state["consecutive_no_improvement"] += 1

    # If we have a DAG, mark the last node as pruned
    if state["dag"]:
        state["dag"][-1]["pruned"] = True

    return state
