# agent/nodes/escalate.py
"""
Node 8: escalate to the next model tier.

On stagnation, finds all feasible models in the tier above the current model,
calls the orchestrator LLM to choose which specific model to try based on task
context, then resets score history for a fresh start with the new model.
Falls back to the largest feasible model in the next tier if the LLM call fails.
"""
import logging
from agent.state import AgentState
from config.android_pool import filter_pool, check_hardware_constraints, all_constraints_pass, ModelSpec

logger = logging.getLogger(__name__)


def _log(model_id: str, msg: str):
    print(f"[escalate][{model_id}] {msg}")


def _llm_choose_model(
    candidates: list[ModelSpec],
    task_type: str,
    task_plan: dict,
    current_best_score: float,
) -> ModelSpec:
    """Ask the orchestrator LLM to choose a model from `candidates` given the task context.
    Falls back to the largest candidate (best chance) on any failure.
    """
    from config.config import ORCHESTRATOR_MODEL, ANTHROPIC_API_KEY
    import anthropic

    if not candidates:
        raise ValueError("No candidates to choose from")

    candidate_lines = "\n".join(
        f"  {i+1}. {m.model_id} (quant={m.quant}, {m.int4_size_mb}MB, "
        f"gsm8k={m.gsm8k:.2f}, mmlu={m.mmlu:.2f})"
        for i, m in enumerate(candidates)
    )
    task_name = task_plan.get("task_name", task_type)
    task_labels = task_plan.get("labels", [])
    task_notes = (
        f"Task type: {task_type}\n"
        f"Task name: {task_name}\n"
        f"Labels: {task_labels}\n"
        f"Current best score: {current_best_score:.4f}"
    )
    prompt = (
        f"You are selecting the best model for a fine-tuning task from the candidates below.\n\n"
        f"Task context:\n{task_notes}\n\n"
        f"Candidates (all satisfy hardware constraints, same tier):\n{candidate_lines}\n\n"
        f"Choose the model ID most likely to solve this task given its benchmarks and architecture. "
        f"Reply with ONLY the exact model_id string, nothing else."
    )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ORCHESTRATOR_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        first_block = resp.content[0]
        chosen_id = first_block.text.strip().strip('"') if isinstance(first_block, anthropic.types.TextBlock) else ""
        match = next((m for m in candidates if m.model_id == chosen_id), None)
        if match is not None:
            return match
        logger.warning("[escalate] LLM returned unknown model_id %r; falling back to largest", chosen_id)
    except Exception as e:
        logger.warning("[escalate] LLM model-choice failed (%s); falling back to largest", e)
    # Fallback: largest model in the tier (highest int4_size_mb = most capable)
    return max(candidates, key=lambda m: m.int4_size_mb)


def escalate_node(state: AgentState) -> AgentState:
    """
    Node 8: escalate to the next model tier.

    1. Find all feasible models in the tier directly above the current model.
    2. Call the orchestrator LLM to choose which specific model to try.
    3. Check hardware constraints for the chosen model.
    4. Reset score history / DAG for a fresh start.
    5. If no next tier exists or hardware fails, terminate.
    """
    current_model = state["selected_model"]
    if current_model is None:
        state["next_action"] = "terminate"
        return state

    current_id = current_model.model_id
    current_tier = current_model.tier

    # Log stagnation context
    scores = state.get("scores", [])
    from agent.nodes.iterate import STAGNATION_WINDOW, STAGNATION_MIN_DELTA
    window = scores[-STAGNATION_WINDOW:] if len(scores) >= STAGNATION_WINDOW else scores
    delta = max(window) - min(window) if window else 0.0
    _log(current_id, "STAGNATION DETECTED")
    _log(current_id, f"  Score window (last {len(window)}): {[f'{s:.4f}' for s in window]}")
    _log(current_id, f"  Window delta: {delta:.4f} < threshold {STAGNATION_MIN_DELTA}")
    _log(current_id, f"  Best score achieved: {state['best_score']:.4f}")
    _log(current_id, f"  Current tier: {current_tier}")

    # Record final best for this model in baselines
    baselines = state.get("model_baselines") or []
    for entry in baselines:
        if entry["model_id"] == current_id:
            entry["best_finetuned_f1"] = max(
                entry.get("best_finetuned_f1", 0.0), state["best_score"]
            )

    # Collect ALL feasible models in the next tier
    next_tier = current_tier + 1
    if next_tier > 3:
        _log(current_id, "  Already at tier 3 (max) — TERMINATING")
        state["next_action"] = "terminate"
        return state

    feasible = filter_pool(state["hardware_constraints"])
    next_tier_candidates = [m for m in feasible if m.tier == next_tier]

    if not next_tier_candidates:
        _log(current_id, f"  No feasible models in tier {next_tier} — TERMINATING")
        state["next_action"] = "terminate"
        return state

    _log(current_id,
         f"  Tier {next_tier} candidates ({len(next_tier_candidates)}): "
         f"{[m.model_id + '/' + str(m.quant) for m in next_tier_candidates]}")

    # LLM picks the best model from the next tier for this task
    chosen = _llm_choose_model(
        candidates=next_tier_candidates,
        task_type=state.get("task_type", "classification"),
        task_plan=state.get("task_plan") or {},
        current_best_score=state["best_score"],
    )

    # Hardware check
    hw_check = check_hardware_constraints(chosen, state["hardware_constraints"])
    hw_ok = all_constraints_pass(hw_check) if state.get("hw_gating_enabled") else (
        chosen.int4_size_mb <= state["hardware_constraints"].storage_mb
        and chosen.peak_memory_mb <= state["hardware_constraints"].memory_mb
    )
    if not hw_ok:
        _log(current_id, f"  Chosen model {chosen.model_id} fails hardware — TERMINATING")
        state["next_action"] = "terminate"
        return state

    _log(current_id,
         f"  PROMOTING: tier {current_tier} → tier {next_tier} | {current_id} → {chosen.model_id} (quant={chosen.quant})")
    _log(current_id, f"  Dataset carried forward: {state.get('current_dataset_path')}")

    state["selected_model"] = chosen
    state["scores"] = []
    state["dag"] = []
    state["iteration"] = 0
    state["best_score"] = 0.0
    state["best_weights_ref"] = None
    state["last_eval"] = None
    state["last_hypothesis"] = ""
    state["llm_iterate_decision"] = None
    state["consecutive_no_improvement"] = 0
    state["downward_probe_done"] = False
    state["next_action"] = "curate"
    return state
