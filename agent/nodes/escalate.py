# agent/nodes/escalate.py
"""
Node 8: escalate to the next model tier.

On stagnation, finds all feasible models in the tier above the current model,
calls the orchestrator LLM to choose which specific model to try based on task
context, then resets score history for a fresh start with the new model.
Falls back to the largest feasible model in the next tier if the LLM call fails.
"""
import logging
from agent.cost import tracked_anthropic_messages_create
from agent.state import AgentState
from config.android_pool import (
    METRIC_COMPARABILITY_CAVEAT,
    ModelSpec,
    filter_pool,
    format_capability_metrics,
    resolve_model_selector,
)

logger = logging.getLogger(__name__)


def _log(model_id: str, msg: str):
    print(f"[escalate][{model_id}] {msg}")


_CHOOSE_SYSTEM = (
    "You are the model-selection stage of an autonomous on-device fine-tuning agent. "
    "You choose which small language model to fine-tune next for a specific task. "
    "All candidates already fit the device's hardware budget, so choose purely on "
    "expected task capability after LoRA fine-tuning — not on size. Prefer the model "
    "whose documented capabilities best match the task type. "
    f"METRIC COMPARABILITY CONTRACT: {METRIC_COMPARABILITY_CAVEAT} "
    "Output STRICT JSON only."
)

# Which published benchmark matters most, per task type. Steers the LLM away from
# defaulting to GSM8K (a math benchmark) when the task is, e.g., NER or code.
_BENCHMARK_HINT = {
    "classification": "like-for-like knowledge metrics and instruction-following.",
    "NER": "instruction-following and structured extraction; use only like-for-like metrics.",
    "math_reasoning": "GSM8K when reported; missing GSM8K is unknown rather than zero.",
    "code_generation": "APPS introductory pass@1; do not substitute a different code metric.",
    "generation": "instruction-following and task-specific generation evidence.",
}


def _llm_choose_model(
    candidates: list[ModelSpec],
    task_type: str,
    task_plan: dict,
    current_best_score: float,
    log=print,
    direction: str = "up",
) -> ModelSpec:
    """Ask the orchestrator LLM to choose a model from `candidates` given the task context.
    Returns the chosen ModelSpec. Falls back to the largest candidate on any failure.
    The chosen model AND the orchestrator's one-sentence reason are printed via `log`
    (Q13) so the model-selection rationale is visible in the run log, not just logging.

    Direction-neutral: used both for UPWARD escalation (bigger tier) and the DOWNWARD
    probe (smaller tier). The prompt frames the choice as "best task fit", which is
    correct in both directions since all candidates already satisfy the hardware budget.
    """
    from config.config import ORCHESTRATOR_MODEL, ANTHROPIC_API_KEY, orchestrator_client_kwargs
    from agent.llm_text import (
        MIN_THINKING_SAFE_MAX_TOKENS,
        describe_empty_text,
        response_text,
    )
    import anthropic

    if not candidates:
        raise ValueError("No candidates to choose from")
    if direction not in ("up", "down"):
        raise ValueError(f"Unknown selection direction: {direction!r}")

    candidate_lines = "\n".join(
        f"  - selector: {m.selector}\n"
        f"    model_id: {m.model_id}\n"
        f"    quant: {m.quant or 'none (bf16)'}  on-disk size: {m.size_mb}MB  "
        f"capability_metrics: {format_capability_metrics(m)}\n"
        f"    notes: {getattr(m, 'notes', '') or 'n/a'}"
        for m in candidates
    )
    task_name = task_plan.get("task_name", task_type)
    task_labels = task_plan.get("labels", [])
    benchmark_hint = _BENCHMARK_HINT.get(
        task_type,
        "task-specific sourced evidence; compare only like-for-like named metrics.",
    )

    # Offline capability descriptions plus the named-metric comparability contract (B161).
    from config.model_capabilities import capability_sections
    cap_doc = capability_sections([m.model_id for m in candidates])

    prompt = (
        f"Task to fine-tune for:\n"
        f"  type: {task_type}\n"
        f"  name: {task_name}\n"
        f"  labels/schema: {task_labels}\n"
        f"  current best F1 (previous model): {current_best_score:.4f}\n\n"
        f"Selection direction: {'upward escalation' if direction == 'up' else 'downward resource probe'}\n"
        f"Target peak-RAM tier: {candidates[0].tier}\n\n"
        f"For a {task_type} task, prioritise: {benchmark_hint}\n\n"
        f"CAPABILITY DESCRIPTIONS (judge task fit from these, not raw numbers):\n{cap_doc}\n\n"
        f"METRIC COMPARABILITY CONTRACT: {METRIC_COMPARABILITY_CAVEAT}\n\n"
        f"'quant' is the on-device weight format: none/bf16 (highest quality, largest), "
        f"Q8_0 (near-lossless, ~1.9x smaller), Q4_K_M (4-bit, smallest, minor quality loss). "
        f"All listed candidates already fit the device budget; among candidates expected to reach "
        f"the goal, prefer the more resource-efficient one (lower peak_ram).\n\n"
        f"Candidates:\n{candidate_lines}\n\n"
        f"Choose the single candidate most likely to reach the highest task accuracy after "
        f"LoRA fine-tuning (breaking ties toward lower RAM). Reply with STRICT JSON only, no prose:\n"
        f'{{"selector": "<exact selector from the list>", "reason": "<one sentence>"}}'
    )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())
        resp = tracked_anthropic_messages_create(
            client.messages,
            stage="escalate",
            model=ORCHESTRATOR_MODEL,
            max_tokens=MIN_THINKING_SAFE_MAX_TOKENS,
            system=_CHOOSE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response_text(resp)
        if not raw:
            log(f"  Orchestrator produced no text: {describe_empty_text(resp)}")
        chosen_id, reason = _parse_choice(raw)
        match = resolve_model_selector(candidates, chosen_id)
        if match is not None:
            log(f"  Orchestrator chose {match.model_id} "
                f"(quant={match.quant or 'bf16'}, {match.size_mb}MB; "
                f"{format_capability_metrics(match)})")
            log(f"    reason: {reason or '(none given)'}")
            return match
        log(f"  Orchestrator returned unknown selector {chosen_id!r}; using resource-safe fallback")
    except Exception as e:
        from agent.llm_errors import raise_if_fatal
        raise_if_fatal(e, "escalate")  # billing/auth → stop the run, don't silently fall back
        log(f"  Orchestrator model-choice failed ({e}); using resource-safe fallback")
    # Candidates already belong to the direction-appropriate target tier. Choose
    # its lowest-footprint exact variant, never arbitrary list order or max-BF16.
    fallback = min(
        candidates,
        key=lambda model: (
            model.size_mb,
            model.size_mb,
            model.selector,
        ),
    )
    log(f"  Resource-safe {direction} fallback: {fallback.selector} "
        f"({fallback.size_mb}MB, size={fallback.size_mb}MB)")
    return fallback


def _parse_choice(raw: str) -> tuple[str, str]:
    """Extract (selector-or-legacy-model-id, reason) from the reply."""
    import json
    import re

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # Try JSON first
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
    current_selector = current_model.selector
    mlabel = current_model.label  # log prefix includes quant
    current_tier = current_model.tier

    # Log stagnation context
    scores = state.get("scores", [])
    from agent.nodes.iterate import (
        STAGNATION_MIN_DELTA,
        STAGNATION_WINDOW,
        _stagnation_gain,
    )
    window = scores[-STAGNATION_WINDOW:] if len(scores) >= STAGNATION_WINDOW else scores
    gain = _stagnation_gain(window)
    _log(mlabel, "STAGNATION DETECTED")
    _log(mlabel, f"  Score window (last {len(window)}): {[f'{s:.4f}' for s in window]}")
    _log(
        mlabel,
        f"  Window chronological gain: {gain:.4f} < threshold "
        f"{STAGNATION_MIN_DELTA}",
    )
    _log(mlabel, f"  Best score achieved: {state['best_score']:.4f}")
    _log(mlabel, f"  Current tier: {current_tier}")

    # Record final best for this model in baselines
    baselines = state.get("model_baselines") or []
    for entry in baselines:
        if entry.get("selector", entry.get("model_id")) in (current_selector, current_id):
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
        direction="up",
    )

    _log(mlabel,
         f"  PROMOTING: tier {current_tier} → tier {next_tier}  |  "
         f"{current_id} (best {state['best_score']:.4f}) → {chosen.model_id} "
         f"[{chosen.quant or 'bf16'}, {chosen.size_mb}MB, size {chosen.size_mb}MB]")
    _log(mlabel, f"  Dataset carried forward: {state.get('current_dataset_path')}")

    # Record this model's completed run BEFORE resetting, so the end-of-run summary can
    # show the FULL trajectory across every tier, not just the final model (Q15/B147).
    # Default to None, never 0.0: a missing or failed baseline must render as "n/a" in
    # the Model Improvement Report rather than as a real zero-shot score of zero, which
    # would credit fine-tuning with the entire final F1 as its improvement.
    _baseline_f1 = next((e.get("baseline_f1") for e in baselines
                         if e.get("selector", e.get("model_id"))
                         in (current_selector, current_id)), None)
    # Same None-not-0.0 rule as the baseline above: this tier's first fine-tuned score, so the
    # Model Improvement Report can separate what one round of fine-tuning bought from what the
    # iteration loop added, for every tier and not just the final one.
    _first_ft = next((e.get("first_finetuned_f1") for e in baselines
                      if e.get("selector", e.get("model_id"))
                      in (current_selector, current_id)), None)
    history = list(state.get("escalation_history") or [])
    history.append({
        "selector": current_selector,
        "model_id": current_id,
        "quant": getattr(current_model, "quant", None),
        "tier": current_tier,
        "baseline_f1": _baseline_f1,
        "first_finetuned_f1": _first_ft,
        "best_score": state["best_score"],
        "iterations": state["iteration"],
        "scores": list(state.get("scores") or []),
        # Full per-iteration DAG for THIS model, so the end-of-run summary can show the
        # traversal of every model tested — not just the final one (the DAG resets on
        # escalation). (B161 reporting request.)
        "dag": list(state.get("dag") or []),
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
    # Stagnation is per-model: a new tier starts with a clean eval history so the previous
    # model's plateau cannot immediately escalate the new one.
    state["eval_history"] = []
    state["dag"] = []
    state["iteration"] = 0
    state["lifetime_best_score"] = max(
        state.get("lifetime_best_score") or 0.0, state["best_score"]
    )
    state["best_score"] = 0.0
    state["best_weights_ref"] = None
    state["last_eval"] = None
    state["last_intervention"] = "data_rebuild"
    state["last_hypothesis"] = ""
    state["llm_iterate_decision"] = None
    state["data_rebuild_plan"] = None
    state["data_rebuild_plan_identity"] = None
    state["consecutive_no_improvement"] = 0
    state["downward_probe_done"] = False
    state["downward_tiers_tried"] = []
    state["converged_model_ref"] = None
    state["downward_probe_history"] = {
        "origin": None,
        "attempts": [],
    }
    state["next_action"] = "curate"
    return state
