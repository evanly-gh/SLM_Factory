# agent/nodes/cold_start/model_selection/orchestrator_choice.py
"""
Strategy: Orchestrator Choice

Let the orchestrator LLM choose the starting model directly, using the task
type, task plan, feasible set, and hardware constraints as context. No probing
overhead — the LLM selects based on published benchmarks and model notes.

Falls back deterministically to the lowest-peak-RAM feasible variant.
"""
import json
import logging
import os
import re

from agent.cost import tracked_anthropic_messages_create
from agent.state import AgentState
from config.android_pool import (
    METRIC_COMPARABILITY_CAVEAT,
    format_capability_metrics,
    resolve_model_selector,
)

logger = logging.getLogger(__name__)

_CHOOSE_SYSTEM = (
    "You are the model-selection stage of an autonomous ON-DEVICE fine-tuning agent. "
    "Given a task, hardware constraints, an accuracy goal, and a list of feasible models "
    "(all already verified to fit the device), choose the single best starting model for "
    "LoRA fine-tuning.\n"
    "OPTIMIZE FOR RESOURCE EFFICIENCY: pick the model with the LOWEST resource consumption "
    "(RAM, storage, decode power) that can still PLAUSIBLY REACH THE ACCURACY GOAL after "
    "fine-tuning — NOT simply the largest or highest-benchmark model. A smaller model that "
    "can hit the goal is strictly better on-device (less RAM/power, faster). Only go bigger "
    "when the task genuinely needs the extra capability to reach the goal.\n"
    "Use the CAPABILITY DESCRIPTIONS provided (qualitative) to judge task fit — they are more "
    "reliable than raw benchmark numbers.\n"
    f"METRIC COMPARABILITY CONTRACT: {METRIC_COMPARABILITY_CAVEAT}\n"
    "Output STRICT JSON only: {\"selector\": \"...\", \"reason\": \"...\"}"
)

_BENCHMARK_HINT = {
    "classification": "like-for-like knowledge metrics and instruction-following.",
    "NER": "instruction-following and structured extraction; use only like-for-like metrics.",
    "math_reasoning": "GSM8K when reported; missing GSM8K is unknown rather than zero.",
    "code_generation": "APPS introductory pass@1; do not substitute a different code metric.",
    "generation": "instruction-following and task-specific generation evidence.",
}


def _parse_choice(raw: str) -> tuple[str, str]:
    """Extract (selector-or-legacy-model-id, reason) from the LLM reply."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and ("selector" in obj or "model_id" in obj):
            choice = obj.get("selector", obj.get("model_id"))
            return str(choice).strip().strip('"'), str(obj.get("reason", "")).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and ("selector" in obj or "model_id" in obj):
                choice = obj.get("selector", obj.get("model_id"))
                return str(choice).strip().strip('"'), str(obj.get("reason", "")).strip()
        except (json.JSONDecodeError, ValueError):
            pass
    return text.strip().strip('"'), ""


def orchestrator_choice_node(state: AgentState) -> AgentState:
    """Let the orchestrator LLM choose the starting model."""
    feasible = state.get("feasible_models", [])
    if not feasible:
        raise RuntimeError("orchestrator_choice_node: feasible_models is empty.")

    forced = os.environ.get("SLM_FORCE_MODEL")
    if forced:
        match = resolve_model_selector(feasible, forced)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
        logger.info("[model_selection:orchestrator_choice] SLM_FORCE_MODEL=%s pinned", forced)
        return state

    from config.config import ORCHESTRATOR_MODEL, ANTHROPIC_API_KEY, orchestrator_client_kwargs
    import anthropic

    task_type = state.get("task_type", "classification")
    task_plan = state.get("task_plan") or {}
    hw = state["hardware_constraints"]
    goal = state.get("stop_threshold", 0.9)
    benchmark_hint = _BENCHMARK_HINT.get(
        task_type,
        "task-specific sourced evidence; compare only like-for-like named metrics.",
    )

    candidate_lines = "\n".join(
        f"  - selector: {m.selector}\n"
        f"    model_id: {m.model_id}\n"
        f"    quant: {m.quant or 'none (bf16)'}  on-disk size: {m.size_mb}MB\n"
        f"    capability_metrics: {format_capability_metrics(m)}\n"
        f"    notes: {m.notes or 'n/a'}"
        for m in feasible
    )

    # Offline capability descriptions plus the named-metric comparability contract (B161).
    from config.model_capabilities import capability_sections
    cap_doc = capability_sections([m.model_id for m in feasible])

    prompt = (
        f"Task description: {state.get('description', 'unknown')}\n"
        f"Task type: {task_type}\n"
        f"Task name: {task_plan.get('task_name', task_type)}\n"
        f"Labels/schema: {task_plan.get('labels', [])}\n"
        f"ACCURACY GOAL (stop threshold): {goal:.3f}\n"
        f"Hardware budget: RAM={hw.memory_mb}MB, storage={hw.storage_mb}MB, "
        f"chip={hw.target_chip}, min_tok_s={hw.min_tok_s}\n\n"
        f"For a {task_type} task, prioritise: {benchmark_hint}\n\n"
        f"CAPABILITY DESCRIPTIONS (use these to judge fit; obey their metric-comparability "
        f"contract):\n{cap_doc}\n\n"
        f"Feasible models (all fit the device; smaller on-disk size = cheaper on-device.\n"
        f"Decode speed and peak RAM are NOT listed: they are unmeasured for these\n"
        f"variants, and an invented number is worse than none. Do not assume them):\n"
        f"{candidate_lines}\n\n"
        f"Pick the MOST RESOURCE-EFFICIENT model (lowest RAM/power) that can plausibly reach "
        f"the accuracy goal {goal:.3f} for this task after LoRA fine-tuning — not the largest. "
        f"Reply with STRICT JSON only:\n"
        f'{{"selector": "<exact selector from the list>", "reason": "<one sentence incl. why it can hit the goal at low cost>"}}'
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())
        resp = tracked_anthropic_messages_create(
            client.messages,
            stage="model_selection",
            model=ORCHESTRATOR_MODEL,
            max_tokens=256,
            system=_CHOOSE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        first_block = resp.content[0]
        raw = first_block.text.strip() if isinstance(first_block, anthropic.types.TextBlock) else ""
        chosen_id, reason = _parse_choice(raw)
        match = resolve_model_selector(feasible, chosen_id)
        if match is not None:
            state["selected_model"] = match
            print(f"[model_selection:orchestrator_choice] LLM chose {match.model_id} "
                  f"[{match.quant or 'bf16'}] (tier={match.tier}, size={match.size_mb}MB)")
            print(f"[model_selection:orchestrator_choice]   reason: {reason or '(no reason given)'}")
            return state
        print(f"[model_selection:orchestrator_choice] LLM returned unknown selector "
              f"{chosen_id!r}; using resource-safe fallback")
    except Exception as e:
        from agent.llm_errors import raise_if_fatal
        raise_if_fatal(e, "orchestrator_choice")
        print(f"[model_selection:orchestrator_choice] LLM call failed ({str(e)[:120]}); "
              f"using resource-safe fallback")

    fallback = min(
        feasible,
        key=lambda model: (
            model.size_mb,
            model.size_mb,
            model.selector,
        ),
    )
    state["selected_model"] = fallback
    print(f"[model_selection:orchestrator_choice] Resource-safe fallback: {fallback.model_id} "
          f"[{fallback.quant or 'bf16'}] (size={fallback.size_mb}MB)")
    return state
