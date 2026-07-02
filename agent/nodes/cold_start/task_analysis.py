# agent/nodes/cold_start/task_analysis.py
from agent.state import AgentState
from config.android_pool import filter_pool, filter_pool_by_task

# Canonical task types. Each type differs in at least two of:
#   model selection, supervision format, eval metric, curation strategy.
#
# classification  — binary or multi-class argmax label prediction.
#                   Flags on task_plan: multi_label:bool, multilingual:bool
# NER             — span extraction, or schema-constrained JSON from unstructured text.
#                   Flags on task_plan: schema:dict|None, multilingual:bool
# math_reasoning  — arithmetic, algebra, word problems. CoT mandatory.
#                   Teacher = DeepSeek-R1. Eval = final-answer exact match.
# code_generation — function synthesis, completion, bug-fix, SQL.
#                   Eval = execution pass@1.
# generation      — open-ended summarization, QA, dialogue. Catch-all.
#                   Eval = LLM-as-judge [0,1]. Flag: multilingual:bool
_VALID_TASK_TYPES = {
    "classification",
    "NER",
    "math_reasoning",
    "code_generation",
    "generation",
}

# Maps each task type to the filter_pool_by_task sort key.
_TASK_TYPE_TO_POOL_KEY: dict[str, str | None] = {
    "classification":  "classification",
    "NER":             "ner",
    "math_reasoning":  "math",
    "code_generation": "code",
    "generation":      None,
}


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
        from config.android_pool import ANDROID_POOL
        plan = plan_task(state["description"], model_pool=ANDROID_POOL)
        state["task_plan"] = plan
        task_type = plan["task_type"]
        state["task_type"] = task_type
        if plan.get("stop_threshold"):
            threshold = float(plan["stop_threshold"])
            state["stop_threshold"] = threshold
            # Record the initial threshold as the immutable floor (can never go below this).
            # iterate_node may lower it further at runtime, but never below initial_stop_threshold.
            if not state.get("initial_stop_threshold"):
                state["initial_stop_threshold"] = threshold

    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(
            f"task_type must be one of {_VALID_TASK_TYPES!r}, got {task_type!r}. "
            "Set task_type in the initial AgentState, or enable autonomous mode."
        )

    # Filter Android pool by hardware constraints, preference-sorted for this task type.
    pool_key = _TASK_TYPE_TO_POOL_KEY.get(task_type)
    feasible = filter_pool_by_task(state["hardware_constraints"], task_type=pool_key)
    if not feasible:
        raise RuntimeError("No models in Android pool satisfy hardware constraints.")

    # math_reasoning: further front-load specialized reasoning models.
    if task_type == "math_reasoning":
        preferred = [m for m in feasible if any(
            k in m.model_id for k in ("DeepSeek-R1", "Qwen3", "Phi-4")
        )]
        if preferred:
            others = [m for m in feasible if m not in preferred]
            feasible = preferred + others

    # Start with the smallest feasible model after preference sorting.
    # SLM_FORCE_MODEL pins to a specific model (e.g. for trainer backend compatibility).
    import os
    forced = os.environ.get("SLM_FORCE_MODEL")
    if forced:
        match = next((m for m in feasible if m.model_id == forced), None)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
    else:
        state["selected_model"] = feasible[0]

    if not state.get("stop_threshold"):
        state["stop_threshold"] = 0.96

    return state
