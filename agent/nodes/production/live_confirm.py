# agent/nodes/production/live_confirm.py
"""
Production Node 3: Live Confirmation (paper §2.6).

Verifies that identified weaknesses are systematic rather than sampling artifacts.
Re-runs failing inputs through M0 to confirm they still fail.
"""
from agent.state import AgentState


def live_confirm_node(state: AgentState) -> AgentState:
    """Confirm failures are systematic by re-running M0 on failing inputs.

    Paper §2.6: 'Before constructing training data, the agent first verifies that
    each identified weakness is systematic rather than an artifact of sampling noise.'

    In Phase 2, this runs actual inference on M0. For now, we filter based on the
    taxonomy's fixability labels — external failures are removed from training targets.
    """
    taxonomy = state.get("failure_taxonomy", {})
    traces = state.get("traces", [])
    failures = [t for t in traces if t.get("verdict") == "fail"]

    fixable_clusters = [c["name"] for c in taxonomy.get("clusters", []) if c.get("fixable")]

    confirmed = []
    external = []
    for t in failures:
        if any(cluster in str(t) for cluster in fixable_clusters) or not fixable_clusters:
            confirmed.append(t)
        else:
            external.append(t)

    state["train_examples"] = [
        {"text": t.get("input", ""), "label": t.get("corrected_output", "")}
        for t in confirmed
    ]

    print(f"[live_confirm] {len(confirmed)} confirmed fixable, {len(external)} external (excluded)")
    return state
