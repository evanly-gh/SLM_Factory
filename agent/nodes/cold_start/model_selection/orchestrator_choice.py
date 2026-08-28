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
from agent.llm_text import (
    MIN_THINKING_SAFE_MAX_TOKENS,
    describe_empty_text,
    response_text,
)
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

_FAMILY_HINT = {
    "classification": "like-for-like knowledge metrics and instruction-following.",
    "extraction": "instruction-following and structured extraction; use only like-for-like metrics.",
    "generation": "GSM8K when the task is arithmetic; otherwise instruction-following and "
                  "task-specific generation evidence. Missing metrics are unknown, not zero.",
    "structured_output": "instruction-following and JSON/schema adherence; a general knowledge "
                         "score is not evidence for emitting a valid call.",
}


def _benchmark_hint(task: str) -> str:
    """Steer the model-choice prompt toward metrics that mean something for THIS task."""
    from tasks import TASKS

    spec = TASKS.get(str(task or ""))
    return _FAMILY_HINT.get(spec.family, "") if spec else ""


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

    # The TASK NAME, which is what `_benchmark_hint` looks up in the registry. This read used to be
    # `state.get("task_type", "classification")` — a field deleted with the channels — so the lookup
    # silently missed on every run and the prompt always got the generic fallback hint instead of the
    # family-specific one (B311).
    task = str(state.get("task") or "")
    task_plan = state.get("task_plan") or {}
    hw = state["hardware_constraints"]
    goal = state.get("stop_threshold", 0.9)
    benchmark_hint = _benchmark_hint(task) or (
        "task-specific sourced evidence; compare only like-for-like named metrics."
    )

    # SORTED SMALLEST FIRST, and each entry says how much cheaper it is than the largest option.
    # The list used to come out in pool order, which asks the orchestrator to hold eighteen sizes in
    # its head while being told to pick the cheapest — and on run 38708719 it picked a 2,382MB variant
    # while explicitly reasoning that it was choosing "the smallest model plausibly capable". Ordering
    # the list is not a hint, it is the difference between the instruction being followable and not.
    ranked = sorted(feasible, key=lambda m: (m.size_mb, m.model_id))
    largest_mb = max((m.size_mb for m in ranked), default=1) or 1
    candidate_lines = "\n".join(
        f"  - selector: {m.selector}\n"
        f"    model_id: {m.model_id}\n"
        f"    tier: {m.tier}   quant: {m.quant or 'none (bf16)'}   on-disk size: {m.size_mb}MB "
        f"({largest_mb / max(m.size_mb, 1):.1f}x cheaper than the largest option here)\n"
        f"    capability_metrics: {format_capability_metrics(m)}\n"
        f"    notes: {m.notes or 'n/a'}"
        for m in ranked
    )
    # Which candidates have NO published numbers at all. Unmeasured is not the same as bad, and the
    # asymmetry is what pushes the choice upward: a 0.6B variant whose row reads "not reported" for
    # every benchmark looks strictly worse than a 4B with an MMLU-Pro score, when in fact one has been
    # measured and the other has not.
    # A candidate with no NUMERIC score anywhere in its metrics line. Matching on a decimal figure
    # rather than the absence of "not reported" because a partially-reported model has both.
    unmeasured = [
        m.selector for m in ranked
        if not re.search(r"\d+\.\d", format_capability_metrics(m))
    ]

    # Offline capability descriptions plus the named-metric comparability contract (B161).
    from config.model_capabilities import capability_sections
    cap_doc = capability_sections([m.model_id for m in feasible])

    prompt = (
        f"Task description: {state.get('description', 'unknown')}\n"
        f"Task: {task}\n"
        f"Labels/schema: {task_plan.get('labels', [])}\n"
        f"ACCURACY GOAL (stop threshold): {goal:.3f}\n"
        f"Hardware budget: RAM={hw.memory_mb}MB, storage={hw.storage_mb}MB, "
        f"chip={hw.target_chip}, min_tok_s={hw.min_tok_s}\n\n"
        f"For the {task} task, prioritise: {benchmark_hint}\n\n"
        f"CAPABILITY DESCRIPTIONS (use these to judge fit; obey their metric-comparability "
        f"contract):\n{cap_doc}\n\n"
        f"ALL {len(ranked)} feasible variants in the pool, SMALLEST FIRST (every one fits the "
        f"device). Decode speed and peak RAM are NOT listed: they are unmeasured for these\n"
        f"variants, and an invented number is worse than none. Do not assume them):\n"
        f"{candidate_lines}\n\n"
        f"HOW TO READ THE CAPABILITY METRICS — this matters, because the obvious reading of them is "
        f"wrong:\n"
        f"  * They are GENERAL-KNOWLEDGE benchmarks (MMLU-Pro, MMLU-Redux, GSM8K). None of them "
        f"measures {task}. A high MMLU-Pro score is weak evidence about this task and a low one is "
        f"weak evidence against it; do not treat the ordering as a ranking for this benchmark.\n"
        f"  * They are measured on the BASE model, ZERO-SHOT. Every candidate here will be LoRA "
        f"fine-tuned on thousands of in-domain rows for this exact task, which routinely lets a "
        f"smaller model beat a larger one's zero-shot number by a wide margin. Extrapolating from "
        f"these scores to a post-fine-tuning result systematically overestimates how large a model "
        f"the goal requires.\n"
        + (f"  * {len(unmeasured)} candidate(s) have NO published numbers at all "
           f"({', '.join(unmeasured[:4])}{' …' if len(unmeasured) > 4 else ''}). Unmeasured is not "
           f"the same as incapable. Do not rank them last for lacking a score.\n"
           if unmeasured else "")
        + f"\nPick the MOST RESOURCE-EFFICIENT model (smallest on-disk size, lowest RAM/power) that "
        f"can plausibly reach the accuracy goal {goal:.3f} for this task AFTER LoRA fine-tuning — "
        f"not the largest, and not the one with the best general-knowledge score. The run can rebuild "
        f"data and retune hyperparameters for many iterations, so a model that starts short of the "
        f"goal is not disqualified; a model that costs 5x the RAM for a benchmark score that does not "
        f"measure this task is a bad trade. Prefer the smallest variant you can justify, and say in "
        f"your reason what specifically makes you doubt the ones below it.\n"
        f"Reply with STRICT JSON only:\n"
        f'{{"selector": "<exact selector from the list>", "reason": "<one sentence incl. why it can hit the goal at low cost, and why the cheaper options below it were rejected>"}}'
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())
        resp = tracked_anthropic_messages_create(
            client.messages,
            stage="model_selection",
            model=ORCHESTRATOR_MODEL,
            max_tokens=MIN_THINKING_SAFE_MAX_TOKENS,
            system=_CHOOSE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response_text(resp)
        if not raw:
            print(f"[model_selection:orchestrator_choice] orchestrator produced no text: "
                  f"{describe_empty_text(resp)}")
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
