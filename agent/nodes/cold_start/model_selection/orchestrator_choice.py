# agent/nodes/cold_start/model_selection/orchestrator_choice.py
"""
Strategy: Orchestrator Choice

Let the orchestrator LLM choose the starting model directly, using the task
type, task plan, feasible set, and hardware constraints as context. No probing
overhead — the LLM selects based on published benchmarks and model notes.

Falls back to the largest feasible model if the LLM call fails.
"""
import json
import logging
import os
import re

from agent.state import AgentState
from config.android_pool import ModelSpec

logger = logging.getLogger(__name__)

_CHOOSE_SYSTEM = (
    "You are the model-selection stage of an autonomous on-device fine-tuning agent. "
    "Given a task description, hardware constraints, and a list of feasible models "
    "(all already verified to fit the device), choose the single best starting model "
    "for LoRA fine-tuning. Consider:\n"
    "  - Task type and what benchmarks matter (GSM8K for math, MMLU for classification)\n"
    "  - Model architecture notes and known strengths/weaknesses\n"
    "  - Prefer smaller models when capability is sufficient (less RAM = better UX)\n"
    "  - All models are from the Qwen architecture family\n"
    "Output STRICT JSON only: {\"model_id\": \"...\", \"reason\": \"...\"}"
)

_BENCHMARK_HINT = {
    "classification": "MMLU and instruction-following — reasoning/label discrimination.",
    "NER": "MMLU and instruction-following — structured extraction follows instructions.",
    "math_reasoning": "GSM8K — arithmetic/word-problem reasoning is the primary signal.",
    "code_generation": "code benchmarks (HumanEval/MBPP); MMLU as a secondary signal.",
    "generation": "MMLU and instruction-following (IFEval) — open-ended answer quality.",
}


def _parse_choice(raw: str) -> tuple[str, str]:
    """Extract (model_id, reason) from the LLM reply."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "model_id" in obj:
            return str(obj["model_id"]).strip().strip('"'), str(obj.get("reason", "")).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "model_id" in obj:
                return str(obj["model_id"]).strip().strip('"'), str(obj.get("reason", "")).strip()
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
        match = next((m for m in feasible if m.model_id == forced), None)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
        logger.info("[model_selection:orchestrator_choice] SLM_FORCE_MODEL=%s pinned", forced)
        return state

    from config.config import ORCHESTRATOR_MODEL, ANTHROPIC_API_KEY
    import anthropic

    task_type = state.get("task_type", "classification")
    task_plan = state.get("task_plan") or {}
    hw = state["hardware_constraints"]
    benchmark_hint = _BENCHMARK_HINT.get(task_type, "MMLU as a general capability proxy.")

    candidate_lines = "\n".join(
        f"  - model_id: {m.model_id}\n"
        f"    quant: {m.quant or 'none (bf16)'}  size: {m.size_mb}MB  peak_ram: {m.peak_memory_mb}MB  "
        f"gsm8k: {m.gsm8k:.2f}  mmlu: {m.mmlu:.2f}\n"
        f"    notes: {m.notes or 'n/a'}"
        for m in feasible
    )

    prompt = (
        f"Task description: {state.get('description', 'unknown')}\n"
        f"Task type: {task_type}\n"
        f"Task name: {task_plan.get('task_name', task_type)}\n"
        f"Labels/schema: {task_plan.get('labels', [])}\n"
        f"Hardware: RAM={hw.memory_mb}MB, storage={hw.storage_mb}MB, "
        f"chip={hw.target_chip}, min_tok_s={hw.min_tok_s}\n\n"
        f"For a {task_type} task, prioritise: {benchmark_hint}\n\n"
        f"Feasible models (all fit the device):\n{candidate_lines}\n\n"
        f"Choose the single best starting model for LoRA fine-tuning of this task. "
        f"Reply with STRICT JSON only:\n"
        f'{{"model_id": "<exact model_id>", "reason": "<one sentence>"}}'
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ORCHESTRATOR_MODEL,
            max_tokens=256,
            system=_CHOOSE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        first_block = resp.content[0]
        raw = first_block.text.strip() if isinstance(first_block, anthropic.types.TextBlock) else ""
        chosen_id, reason = _parse_choice(raw)
        match = next((m for m in feasible if m.model_id == chosen_id), None)
        if match is not None:
            state["selected_model"] = match
            logger.info(
                "[model_selection:orchestrator_choice] LLM chose %s — %s",
                chosen_id, reason or "(no reason given)",
            )
            return state
        logger.warning(
            "[model_selection:orchestrator_choice] LLM returned unknown model_id %r; falling back",
            chosen_id,
        )
    except Exception as e:
        logger.warning(
            "[model_selection:orchestrator_choice] LLM call failed (%s); falling back to largest",
            e,
        )

    fallback = max(feasible, key=lambda m: m.size_mb)
    state["selected_model"] = fallback
    logger.info(
        "[model_selection:orchestrator_choice] Fallback to largest: %s (peak=%dMB)",
        fallback.model_id, fallback.peak_memory_mb,
    )
    return state
