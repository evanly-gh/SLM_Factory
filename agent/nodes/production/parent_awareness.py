# agent/nodes/production/parent_awareness.py
"""
Production Node 4: Parent Model Awareness (paper §2.6, Eq. 15).

Inspects M0's training lineage, builds regression set R and replay buffer D_replay.
Paper: 'D_replay ⊂ D_parent, |D_replay| ≈ (0.1-0.2)|D_parent|'
"""
import json
import random
from agent.state import AgentState


def parent_awareness_node(state: AgentState) -> AgentState:
    """Build regression set and replay buffer from parent model's training data.

    Paper §2.6: 'To reduce catastrophic forgetting, the agent also includes a replay
    buffer D_replay ⊂ D_parent, |D_replay| ≈ (0.1-0.2)|D_parent|, which helps preserve
    previously learned behaviors while adapting the model to new errors.'
    """
    traces = state.get("traces", [])
    t_pass = [t for t in traces if t.get("verdict") == "pass"]

    # Build regression set R: stratified sample of passing examples
    rng = random.Random(42)
    rng.shuffle(t_pass)
    n_regression = min(len(t_pass), max(50, len(t_pass) // 2))
    state["regression_set"] = [
        {"text": t.get("input", ""), "label": t.get("output", t.get("corrected_output", ""))}
        for t in t_pass[:n_regression]
    ]

    # Build replay buffer: 15% of parent's training data (if available)
    parent_data_path = state.get("current_dataset_path")
    if parent_data_path:
        try:
            with open(parent_data_path) as f:
                parent_data = [json.loads(line) for line in f if line.strip()]
            n_replay = max(1, int(len(parent_data) * 0.15))
            rng.shuffle(parent_data)
            state["replay_buffer"] = parent_data[:n_replay]
            print(f"[parent_awareness] Replay buffer: {n_replay}/{len(parent_data)} examples")
        except FileNotFoundError:
            state["replay_buffer"] = []
            print("[parent_awareness] No parent training data found; empty replay buffer")
    else:
        state["replay_buffer"] = []

    print(f"[parent_awareness] Regression set: {len(state['regression_set'])} examples")
    return state
