# agent/nodes/task_analysis.py
from agent.state import AgentState
from android_pool import filter_pool

_VALID_TASK_TYPES = {"classification", "NER", "generation"}


def task_analysis_node(state: AgentState) -> AgentState:
    """
    Node 1: classify the task, select starting model, survey baselines.

    In autonomous mode (state["autonomous"] is True, or task_type is missing/invalid),
    the orchestrator LLM derives task_type + a data-acquisition plan from the natural-language
    description. Otherwise task_type must be provided by the caller. All decisions are made
    before any data is touched.
    """
    task_type = state.get("task_type", "")
    need_plan = state.get("autonomous") or task_type not in _VALID_TASK_TYPES

    if need_plan and state.get("task_plan") is None:
        from agent.task_planner import plan_task
        plan = plan_task(state["description"])
        state["task_plan"] = plan
        task_type = plan["task_type"]
        state["task_type"] = task_type
        if plan.get("stop_threshold"):
            state["stop_threshold"] = float(plan["stop_threshold"])

    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(
            f"task_type must be one of {_VALID_TASK_TYPES!r}, got {task_type!r}. "
            "Set task_type in the initial AgentState, or enable autonomous mode."
        )

    # Filter Android pool by hardware constraints
    feasible = filter_pool(state["hardware_constraints"])
    if not feasible:
        raise RuntimeError("No models in Android pool satisfy hardware constraints.")

    # Start with Tier 1 smallest model (sorted by tier then size in filter_pool).
    # SLM_FORCE_MODEL pins the base model (e.g. to one the trainer backend supports) — it
    # must still satisfy the hardware constraints.
    import os
    forced = os.environ.get("SLM_FORCE_MODEL")
    if forced:
        match = next((m for m in feasible if m.model_id == forced), None)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
    else:
        state["selected_model"] = feasible[0]

    # Use configured stop_threshold if already set, otherwise use sensible default
    if not state.get("stop_threshold"):
        state["stop_threshold"] = 0.96

    return state
