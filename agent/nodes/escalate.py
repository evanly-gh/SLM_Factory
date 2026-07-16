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
from config.android_pool import filter_pool, ModelSpec

logger = logging.getLogger(__name__)


def _log(model_id: str, msg: str):
    print(f"[escalate][{model_id}] {msg}")


_CHOOSE_SYSTEM = (
    "You are the model-selection stage of an autonomous on-device fine-tuning agent. "
    "You choose which small language model to fine-tune next for a specific task. "
    "All candidates already fit the device's hardware budget, so choose purely on "
    "expected task capability after LoRA fine-tuning — not on size. Prefer the model "
    "whose architecture and published benchmarks best match the task type. Output STRICT "
    "JSON only."
)

# Which published benchmark matters most, per task type. Steers the LLM away from
# defaulting to GSM8K (a math benchmark) when the task is, e.g., NER or code.
_BENCHMARK_HINT = {
    "classification": "MMLU and instruction-following (IFEval) — reasoning/label discrimination.",
    "NER": "MMLU and instruction-following — structured extraction follows instructions.",
    "math_reasoning": "GSM8K — arithmetic/word-problem reasoning is the primary signal.",
    "code_generation": "code benchmarks (HumanEval/MBPP); MMLU as a secondary signal.",
    "generation": "MMLU and instruction-following (IFEval) — open-ended answer quality.",
}


def _llm_choose_model(
    candidates: list[ModelSpec],
    task_type: str,
    task_plan: dict,
    current_best_score: float,
    log=print,
) -> ModelSpec:
    """Ask the orchestrator LLM to choose a model from `candidates` given the task context.
    Returns the chosen ModelSpec. Falls back to the largest candidate on any failure.
    The chosen model AND the orchestrator's one-sentence reason are printed via `log`
    (Q13) so the model-selection rationale is visible in the run log, not just logging.

    Direction-neutral: used both for UPWARD escalation (bigger tier) and the DOWNWARD
    probe (smaller tier). The prompt frames the choice as "best task fit", which is
    correct in both directions since all candidates already satisfy the hardware budget.
    """
    from config.config import ORCHESTRATOR_MODEL, ANTHROPIC_API_KEY
    import anthropic
    import json

    if not candidates:
        raise ValueError("No candidates to choose from")

    candidate_lines = "\n".join(
        f"  - model_id: {m.model_id}\n"
        f"    quant: {m.quant or 'none (bf16)'}  size: {m.size_mb}MB  "
        f"gsm8k: {m.gsm8k:.2f}  mmlu: {m.mmlu:.2f}\n"
        f"    notes: {getattr(m, 'notes', '') or 'n/a'}"
        for m in candidates
    )
    task_name = task_plan.get("task_name", task_type)
    task_labels = task_plan.get("labels", [])
    benchmark_hint = _BENCHMARK_HINT.get(task_type, "MMLU as a general capability proxy.")

    prompt = (
        f"Task to fine-tune for:\n"
        f"  type: {task_type}\n"
        f"  name: {task_name}\n"
        f"  labels/schema: {task_labels}\n"
        f"  current best F1 (previous model): {current_best_score:.4f}\n\n"
        f"For a {task_type} task, prioritise: {benchmark_hint}\n\n"
        f"'quant' is the on-device weight format: none/bf16 (highest quality, largest), "
        f"Q8_0 (near-lossless, ~1.9x smaller), Q4_K_M (4-bit, smallest, minor quality loss). "
        f"All listed candidates already fit the device budget.\n\n"
        f"Candidates:\n{candidate_lines}\n\n"
        f"Choose the single candidate most likely to reach the highest task accuracy after "
        f"LoRA fine-tuning. Reply with STRICT JSON only, no prose:\n"
        f'{{"model_id": "<exact model_id from the list>", "reason": "<one sentence>"}}'
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
        match = next((m for m in candidates if m.model_id == chosen_id), None)
        if match is not None:
            log(f"  Orchestrator chose {match.model_id} "
                f"(quant={match.quant or 'bf16'}, {match.size_mb}MB, gsm8k={match.gsm8k:.2f}, "
                f"mmlu={match.mmlu:.2f})")
            log(f"    reason: {reason or '(none given)'}")
            return match
        log(f"  Orchestrator returned unknown model_id {chosen_id!r}; falling back to largest candidate")
    except Exception as e:
        from agent.llm_errors import raise_if_fatal
        raise_if_fatal(e, "escalate")  # billing/auth → stop the run, don't silently fall back
        log(f"  Orchestrator model-choice failed ({e}); falling back to largest candidate")
    # Fallback: largest model in the tier (highest size_mb = most capable)
    fallback = max(candidates, key=lambda m: m.size_mb)
    log(f"  Fallback choice: {fallback.model_id} (quant={fallback.quant or 'bf16'}, {fallback.size_mb}MB)")
    return fallback


def _parse_choice(raw: str) -> tuple[str, str]:
    """Extract (model_id, reason) from the LLM reply. Tolerates JSON, code fences,
    or a bare model_id string (back-compat with the old plain-string protocol)."""
    import json
    import re

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # Try JSON first
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
    # Back-compat: treat the whole reply as a bare model_id string
    return text.strip().strip('"'), ""


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
    mlabel = current_model.label  # log prefix includes quant
    current_tier = current_model.tier

    # Log stagnation context
    scores = state.get("scores", [])
    from agent.nodes.iterate import STAGNATION_WINDOW, STAGNATION_MIN_DELTA
    window = scores[-STAGNATION_WINDOW:] if len(scores) >= STAGNATION_WINDOW else scores
    delta = max(window) - min(window) if window else 0.0
    _log(mlabel, "STAGNATION DETECTED")
    _log(mlabel, f"  Score window (last {len(window)}): {[f'{s:.4f}' for s in window]}")
    _log(mlabel, f"  Window delta: {delta:.4f} < threshold {STAGNATION_MIN_DELTA}")
    _log(mlabel, f"  Best score achieved: {state['best_score']:.4f}")
    _log(mlabel, f"  Current tier: {current_tier}")

    # Record final best for this model in baselines
    baselines = state.get("model_baselines") or []
    for entry in baselines:
        if entry["model_id"] == current_id:
            entry["best_finetuned_f1"] = max(
                entry.get("best_finetuned_f1", 0.0), state["best_score"]
            )

    # Escalate to the nearest higher (more RAM) non-empty tier. Tiers are per-variant
    # RAM buckets, so a bucket can be empty for a given hardware budget — step past gaps
    # rather than terminating on the first empty tier. Exclude variants of the CURRENT
    # model_id at the same tier (no point re-selecting what already stagnated).
    feasible = filter_pool(state["hardware_constraints"])
    higher = [m for m in feasible if m.tier > current_tier]
    if not higher:
        _log(mlabel, f"  No feasible models above tier {current_tier} — TERMINATING")
        state["next_action"] = "terminate"
        return state
    next_tier = min(m.tier for m in higher)
    next_tier_candidates = [m for m in higher if m.tier == next_tier]

    _log(mlabel,
         f"  Tier {next_tier} candidates ({len(next_tier_candidates)}): "
         f"{[m.model_id + '/' + str(m.quant) for m in next_tier_candidates]}")

    # LLM picks the best model from the next tier for this task (reason logged via _log).
    chosen = _llm_choose_model(
        candidates=next_tier_candidates,
        task_type=state.get("task_type", "classification"),
        task_plan=state.get("task_plan") or {},
        current_best_score=state["best_score"],
        log=lambda m: _log(mlabel, m),
    )

    _log(mlabel,
         f"  PROMOTING: tier {current_tier} → tier {next_tier}  |  "
         f"{current_id} (best {state['best_score']:.4f}) → {chosen.model_id} "
         f"[{chosen.quant or 'bf16'}, {chosen.size_mb}MB, peak {chosen.peak_memory_mb}MB]")
    _log(mlabel, f"  Dataset carried forward: {state.get('current_dataset_path')}")

    # Record this model's completed run BEFORE resetting, so the end-of-run summary can
    # show the FULL trajectory across every tier, not just the final model (Q15/B147).
    history = list(state.get("escalation_history") or [])
    history.append({
        "model_id": current_id,
        "quant": getattr(current_model, "quant", None),
        "tier": current_tier,
        "best_score": state["best_score"],
        "iterations": state["iteration"],
        "scores": list(state.get("scores") or []),
    })
    state["escalation_history"] = history

    # Free the previous model's VRAM/cache — we've moved on from it for good, so it must
    # not sit resident competing with the next (usually larger) model (B142).
    try:
        from training.slm_helpers import clear_inference_cache
        clear_inference_cache()
        _log(mlabel, "  Cleared inference cache (freed the previous model's VRAM)")
    except Exception:
        pass

    state["selected_model"] = chosen
    state["scores"] = []
    state["dag"] = []
    state["iteration"] = 0
    state["dataset_version"] = 0
    state["lifetime_best_score"] = max(
        state.get("lifetime_best_score") or 0.0, state["best_score"]
    )
    state["best_score"] = 0.0
    state["best_weights_ref"] = None
    state["last_eval"] = None
    state["last_curation"] = None
    state["last_intervention"] = "data_rebuild"
    state["last_hypothesis"] = ""
    state["llm_iterate_decision"] = None
    state["consecutive_no_improvement"] = 0
    state["downward_probe_done"] = False
    state["next_action"] = "curate"
    return state
