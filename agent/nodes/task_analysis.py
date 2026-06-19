# agent/nodes/task_analysis.py
from agent.state import AgentState
from android_pool import filter_pool

_VALID_TASK_TYPES = {"classification", "NER", "generation"}


def task_analysis_node(state: AgentState) -> AgentState:
    """
    Node 1: validate task type, select starting model, survey baselines.
    task_type is read from state (set by the caller at graph-entry time).
    All decisions are made before any data is touched.
    """
    # Validate task_type — must be provided by the caller, not hardcoded here
    task_type = state.get("task_type", "")
    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(
            f"task_type must be one of {_VALID_TASK_TYPES!r}, got {task_type!r}. "
            "Set task_type in the initial AgentState before entering the graph."
        )

    # Filter Android pool by hardware constraints
    feasible = filter_pool(state["hardware_constraints"])
    if not feasible:
        raise RuntimeError("No models in Android pool satisfy hardware constraints.")

    # Start with Tier 1 smallest model (sorted by tier then size in filter_pool)
    state["selected_model"] = feasible[0]

    # Use configured stop_threshold if already set, otherwise use sensible default
    if not state.get("stop_threshold"):
        state["stop_threshold"] = 0.96

    return state
