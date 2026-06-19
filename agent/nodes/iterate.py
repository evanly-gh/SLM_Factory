# agent/nodes/iterate.py
from agent.state import AgentState


def apply_iteration_policy(score: float) -> dict:
    """
    Apply the shared iteration policy based on current f(π).
    Returns dict with band and intervention type.
    """
    if score < 0.80:
        return {
            "band": "<0.80",
            "intervention": "data_rebuild",
            "description": "Score below 0.80 — data problem. Rebuild dataset.",
        }
    elif score < 0.95:
        return {
            "band": "0.80-0.95",
            "intervention": "hyperparameter",
            "description": "Score 0.80–0.95 — optimization problem. Tune hyperparameters, hold dataset fixed.",
        }
    else:
        return {
            "band": ">=0.95",
            "intervention": "surgical",
            "description": "Score ≥0.95 — add 2–3 targeted examples per remaining failure pattern.",
        }


def iterate_node(state: AgentState) -> AgentState:
    """Node 5: determine next intervention based on current score."""
    if not state["scores"]:
        state["next_action"] = "train"
        return state

    current_score = state["scores"][-1]
    policy = apply_iteration_policy(current_score)
    state["last_intervention"] = policy["intervention"]

    if current_score >= state["stop_threshold"]:
        state["next_action"] = "terminate"
    elif state["consecutive_no_improvement"] >= 2:
        state["next_action"] = "escalate"
    elif policy["intervention"] == "hyperparameter":
        # Hyperparameter tuning: hold dataset fixed, skip curation, go straight to retrain
        state["next_action"] = "train"
    else:
        # data_rebuild and surgical interventions need curation first
        state["next_action"] = "curate"

    return state
