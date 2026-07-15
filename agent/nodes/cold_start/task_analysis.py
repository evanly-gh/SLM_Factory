# agent/nodes/cold_start/task_analysis.py
import os
from agent.state import AgentState
from config.android_pool import ANDROID_POOL
from agent.nodes.cold_start.hardware_filter import run_hardware_filter

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

def task_analysis_node(state: AgentState) -> AgentState:
    """
    Node 1: classify the task, filter hardware pool, set stop threshold.

    Does NOT select the final model — that is done by scaling_curve_node (Node 1b)
    after fine-tuning 3 candidates. Stores the hardware-filtered feasible set in
    state["feasible_models"] for scaling_curve_node to consume.
    """
    task_type = state.get("task_type", "")
    need_plan = state.get("autonomous") or task_type not in _VALID_TASK_TYPES

    if need_plan and state.get("task_plan") is None:
        from agent.task_planner import plan_task
        plan = plan_task(state["description"], model_pool=ANDROID_POOL)
        state["task_plan"] = plan
        task_type = plan["task_type"]
        state["task_type"] = task_type
        if plan.get("stop_threshold"):
            threshold = float(plan["stop_threshold"])
            state["stop_threshold"] = threshold
            # Record the initial threshold as the immutable floor (can never go below this).
            # iterate_node may lower it further at runtime, but never below initial_stop_threshold.
            if state.get("initial_stop_threshold") is None:
                state["initial_stop_threshold"] = threshold

    # Testing/validation override: pin the stop threshold from the environment so a run
    # can be steered deterministically (e.g. set it above the pool's best benchmark to
    # force escalation through every tier). Takes precedence over the planner and sets
    # the immutable floor too, so iterate_node cannot lower it below the pinned value.
    _threshold_override = os.environ.get("SLM_STOP_THRESHOLD")
    if _threshold_override:
        threshold = float(_threshold_override)
        state["stop_threshold"] = threshold
        state["initial_stop_threshold"] = threshold

    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(
            f"task_type must be one of {_VALID_TASK_TYPES!r}, got {task_type!r}. "
            "Set task_type in the initial AgentState, or enable autonomous mode."
        )

    # Stage 1 + 2: hardware filter (inequality + on-device stub)
    feasible = run_hardware_filter(state["hardware_constraints"])
    if not feasible:
        raise RuntimeError("No models in Android pool satisfy hardware constraints.")

    # Sort largest→smallest for scaling_curve_node which relies on
    # feasible[0] = largest and feasible[-1] = smallest.
    feasible = sorted(feasible, key=lambda m: m.size_mb, reverse=True)
    state["feasible_models"] = feasible

    if not state.get("stop_threshold"):
        state["stop_threshold"] = 0.96

    # selected_model is intentionally NOT set here.
    # scaling_curve_node sets it after probing candidates.
    return state
