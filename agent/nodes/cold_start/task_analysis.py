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
#                   Cloud fallback = DeepSeek V4 Flash thinking. Eval = final-answer exact match.
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

def _apply_data_targets(state: AgentState, task_type: str) -> None:
    """Clamp the planner's chosen curriculum/eval sizes to config floors/ceiling and store
    them in state. Env overrides (SLM_CURRICULUM_SIZE / SLM_EVAL_SET_SIZE) win for testing.
    Curriculum floor is deliberately high (small on-device models need more data); eval floor
    keeps macro-F1 statistically stable."""
    from config.config import (
        CURRICULUM_SIZE_FLOOR, EVAL_SET_SIZE, DATA_SIZE_CEILING, DATASET_SIZE_BY_TYPE,
    )
    plan = state.get("task_plan") or {}

    def _clamp(val, floor):
        try:
            v = int(val)
        except (TypeError, ValueError):
            v = 0
        return max(floor, min(v, DATA_SIZE_CEILING))

    _curr_plan = plan.get("curriculum_size")
    if _curr_plan is None:
        # Fall back to the per-type default, but never below the floor.
        _curr_plan = DATASET_SIZE_BY_TYPE.get(task_type, CURRICULUM_SIZE_FLOOR)
    curriculum = _clamp(_curr_plan, CURRICULUM_SIZE_FLOOR)
    eval_size = _clamp(plan.get("eval_size", EVAL_SET_SIZE) or EVAL_SET_SIZE, EVAL_SET_SIZE)

    # Explicit env overrides (testing/determinism).
    if os.environ.get("SLM_CURRICULUM_SIZE"):
        curriculum = _clamp(os.environ["SLM_CURRICULUM_SIZE"], 1)
    if os.environ.get("SLM_EVAL_SET_SIZE"):
        eval_size = _clamp(os.environ["SLM_EVAL_SET_SIZE"], 1)

    state["curriculum_size_target"] = curriculum
    state["eval_size_target"] = eval_size
    print(f"      [planner] data targets (clamped): curriculum={curriculum}  eval={eval_size}  "
          f"(floors {CURRICULUM_SIZE_FLOOR}/{EVAL_SET_SIZE}, ceiling {DATA_SIZE_CEILING})")


def task_analysis_node(state: AgentState) -> AgentState:
    """
    Node 1: classify the task, filter hardware pool, set stop threshold.

    Does NOT select the final model — that is done by the configurable
    `model_selection` node (Node 1b, agent/nodes/cold_start/model_selection/).
    Stores the hardware-filtered feasible set in state["feasible_models"] for the
    model_selection node to consume.
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

    # Data-size targets: take the orchestrator's chosen curriculum/eval sizes, clamped to
    # [floor, ceiling] (B161). Fall back to the per-type default when the planner gave none.
    _apply_data_targets(state, task_type)

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

    # Sort largest→smallest. The model_selection strategies read feasible_models;
    # interpolation relies on feasible[0]=largest / feasible[-1]=smallest.
    feasible = sorted(feasible, key=lambda m: m.size_mb, reverse=True)
    state["feasible_models"] = feasible

    if not state.get("stop_threshold"):
        state["stop_threshold"] = 0.96

    # selected_model is intentionally NOT set here — the model_selection node does it.
    return state
