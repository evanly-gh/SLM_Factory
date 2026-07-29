# agent/nodes/downward_probe.py
"""
Node 9: active downward re-exploration (B161 change 8).

When the loop converges (goal met), find the SMALLEST feasible model that still clears the
goal, to minimize on-device resource use. The orchestrator first analyzes the trajectory
(tiers tried, steps taken, margin above the goal) and decides whether a lower tier is worth
attempting; if so, it re-trains + re-evaluates progressively smaller tiers (each tier at most
once, tracked in downward_tiers_tried) and ADOPTS the smallest that clears the goal — else it
keeps the converged model. Always terminates.
"""
import copy
import logging
import os
from agent.checkpoint import run_training_atomically
from agent.cost import tracked_anthropic_messages_create
from agent.state import AgentState
from config.android_pool import filter_pool, resolve_model_selector
from training.hparams import (
    hyperparameter_identity,
    normalize_hyperparams,
)
from training.slm_helpers import train as slm_train
from eval.harness import run_eval
from agent.nodes.escalate import _llm_choose_model

logger = logging.getLogger(__name__)


def _plog(msg: str):
    """Print so downward-re-exploration reasoning reaches run.log (logger.info is suppressed)."""
    print(f"[downward_probe] {msg}")


ARTIFACTS_DIR = "artifacts"

DOWNWARD_PROBE_H, _ = normalize_hyperparams({
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "weight_decay": 0.01,
    "learning_rate": 2e-4,
    "nr_epochs": 3,
    "micro_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "effective_batch_size": 8,
})


def _should_reexplore_downward(state, n_lower_tiers: int) -> bool:
    """Orchestrator decision: given the trajectory (tiers tried, steps, margin above goal),
    is a lower-tier attempt worth it? LLM-driven with a safe heuristic fallback.

    Heuristic: attempt if there is an untried lower tier AND we converged with some margin
    (a comfortable win suggests a smaller model might also clear the goal)."""
    if n_lower_tiers <= 0:
        return False
    best = state.get("best_score", 0.0)
    threshold = state.get("stop_threshold", 0.9)
    margin = best - threshold
    iters = state.get("iteration", 0)
    tiers_tried = len(state.get("escalation_history") or []) + 1

    try:
        import anthropic
        from config.config import ORCHESTRATOR_MODEL, ANTHROPIC_API_KEY, orchestrator_client_kwargs
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())
        prompt = (
            f"An on-device fine-tuning run just CONVERGED. Decide whether to try a SMALLER "
            f"(lower-RAM) model that might also reach the goal, to save device resources.\n"
            f"- Converged score: {best:.4f}  (goal {threshold:.4f}, margin {margin:+.4f})\n"
            f"- Tiers/models tried so far: {tiers_tried}\n"
            f"- Iterations on the winning model: {iters}\n"
            f"- Untried lower tiers available: {n_lower_tiers}\n"
            f"A larger margin above the goal suggests a smaller model may also succeed. "
            f"Reply STRICT JSON only: {{\"reexplore\": true|false, \"reason\": \"<one sentence>\"}}"
        )
        resp = tracked_anthropic_messages_create(
            client.messages,
            stage="downward_probe",
            model=ORCHESTRATOR_MODEL,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        import json as _json
        import re as _re
        raw = resp.content[0].text.strip()
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        obj = _json.loads(m.group()) if m else {}
        decision = bool(obj.get("reexplore", False))
        _plog(f"orchestrator downward-re-exploration decision: {decision} — {obj.get('reason','')}")
        return decision
    except Exception as e:  # noqa: BLE001
        from agent.llm_errors import raise_if_fatal
        raise_if_fatal(e, "downward_probe")
        decision = margin >= 0.03
        _plog(f"orchestrator decision unavailable ({str(e)[:60]}); heuristic reexplore={decision} "
              f"(margin {margin:+.4f})")
        return decision


def _train_and_eval(
    model,
    state,
    dataset_path: str,
    hparams: dict | None = None,
):
    """Train `model` on dataset_path and return (weights_ref, EvalResult).
    Uses the honest quantized-eval path when model.quant is set."""
    task_type = state["task_type"]
    model_id = model.model_id
    realized_h, _ = normalize_hyperparams(
        hparams or DOWNWARD_PROBE_H
    )
    def selector_safe(candidate):
        return candidate.selector.replace("/", "_").replace("@", "__")

    final_directory = os.path.join(
        ARTIFACTS_DIR,
        "downward_probe",
        selector_safe(state["selected_model"]),
        selector_safe(model),
    )

    def produce(output_dir):
        return slm_train(
            dataset_path=dataset_path,
            base_model=model_id,
            nr_epochs=realized_h["nr_epochs"],
            learning_rate=realized_h["learning_rate"],
            lora_rank=realized_h["lora_rank"],
            lora_alpha=realized_h["lora_alpha"],
            lora_dropout=realized_h["lora_dropout"],
            weight_decay=realized_h["weight_decay"],
            micro_batch_size=realized_h["micro_batch_size"],
            gradient_accumulation_steps=(
                realized_h["gradient_accumulation_steps"]
            ),
            effective_batch_size=realized_h["effective_batch_size"],
            output_dir=output_dir,
            task_type=task_type,
        )

    weights_ref = run_training_atomically(
        final_directory,
        produce,
    ).weights_ref
    _plog(
        "  fixed probe config: "
        f"r={realized_h['lora_rank']} alpha={realized_h['lora_alpha']} "
        f"dropout={realized_h['lora_dropout']} "
        f"weight_decay={realized_h['weight_decay']} "
        f"lr={realized_h['learning_rate']} "
        f"epochs={realized_h['nr_epochs']} "
        f"micro_batch={realized_h['micro_batch_size']} "
        f"grad_accum={realized_h['gradient_accumulation_steps']} "
        f"effective_batch={realized_h['effective_batch_size']}"
    )
    gguf_path = None
    if model.quant is not None:
        from agent.nodes.evaluate import _build_gguf_for_eval

        gguf_path = _build_gguf_for_eval(
            weights_ref,
            model.model_id,
            model.quant,
            model.label,
        )
    result = run_eval(
        state["eval_set"], weights_ref, model_id,
        task_type=task_type, quant=model.quant, gguf_path=gguf_path,
    )
    return weights_ref, result


def _finish_probe(state: AgentState) -> AgentState:
    state["downward_probe_done"] = True
    state["downward_probe_pending"] = None
    state["next_action"] = "terminate"
    return state


def tiers_already_explored(state: AgentState, feasible) -> set[int]:
    """Tiers already trained during the run's main escalation ladder.

    `downward_tiers_tried` only records tiers the DOWNWARD PROBE itself has already
    tried — it starts empty even when the run's normal escalation already spent many
    real iterations on a lower tier before promoting past it. Without this, the probe
    (or iterate_node's routing gate) can re-select that same tier, retraining a model
    whose true best score is already known, using only a single fixed probe config
    that has no reason to beat what the real search already found.

    `state["model_baselines"]` has one entry per selector that ever reached
    evaluate_node, which is exactly "every tier the main ladder actually tried."
    """
    explored: set[int] = set()
    for baseline in state.get("model_baselines") or []:
        resolved = resolve_model_selector(feasible, baseline.get("selector", ""))
        if resolved is not None:
            explored.add(resolved.tier)
    return explored


def downward_probe_step_node(state: AgentState) -> AgentState:
    """Plan or execute one durable downward-probe step.

    Planning (including LLM choice) and training/evaluation are separate graph
    commits. A failed training process therefore retries the selected candidate
    without repeating adopted probes or paid model-choice calls.
    """
    state["downward_probe_done"] = False
    state["next_action"] = "downward_probe"
    current = state.get("selected_model")
    if current is None:
        return _finish_probe(state)
    threshold = state["stop_threshold"]
    history = state.get("downward_probe_history") or {}
    if history.get("origin") is None:
        history = {
            "origin": {
                "selector": current.selector,
                "model_id": current.model_id,
                "quant": current.quant,
                "tier": current.tier,
                "score": state.get("best_score", 0.0),
                "weights_ref": state.get("best_weights_ref"),
                "iterations": state.get("iteration", 0),
                "scores": list(state.get("scores") or []),
                "dag": copy.deepcopy(state.get("dag") or []),
            },
            "attempts": list(history.get("attempts") or []),
            "fixed_H": dict(DOWNWARD_PROBE_H),
        }
    else:
        history["fixed_H"] = dict(DOWNWARD_PROBE_H)
    state["downward_probe_history"] = history
    state["converged_model_ref"] = {
        "selector": current.selector,
        "model_id": current.model_id,
        "quant": current.quant,
        "tier": current.tier,
        "score": state.get("best_score", 0.0),
    }

    def skip_optional_error(
        stage: str,
        exc: Exception,
        *,
        target_tier=None,
        candidates=(),
    ):
        reason = f"{type(exc).__name__}: {exc}"
        history["termination"] = {
            "stage": stage,
            "result": "skipped_error",
            "target_tier": target_tier,
            "candidate_selectors": [
                candidate.selector for candidate in candidates
            ],
            "reason": reason,
        }
        _plog(
            f"optional downward re-exploration skipped: {stage} failed "
            f"({reason}); preserving converged {current.selector} "
            f"at {state.get('best_score', 0.0):.4f}"
        )
        return _finish_probe(state)

    dataset_path = state.get("current_dataset_path")
    if not dataset_path:
        _plog("no dataset available — cannot re-explore; keeping converged model")
        return _finish_probe(state)
    feasible = filter_pool(state["hardware_constraints"])
    tried = set(state.get("downward_tiers_tried") or []) | tiers_already_explored(
        state, feasible
    )
    pending = state.get("downward_probe_pending")

    if pending:
        try:
            pending_h, _ = normalize_hyperparams(
                pending.get("H") or DOWNWARD_PROBE_H
            )
        except ValueError as exc:
            return skip_optional_error(
                "pending_hyperparameters",
                exc,
                target_tier=pending.get("target_tier"),
            )
        if (
            hyperparameter_identity(pending_h)
            != hyperparameter_identity(DOWNWARD_PROBE_H)
        ):
            return skip_optional_error(
                "pending_hyperparameters",
                ValueError(
                    "pending downward probe H differs from the fixed "
                    "serialized probe contract"
                ),
                target_tier=pending.get("target_tier"),
            )
        chosen = resolve_model_selector(feasible, pending.get("selector", ""))
        if chosen is None or chosen.selector != pending.get("selector"):
            return skip_optional_error(
                "pending_selector",
                ValueError(
                    f"pending exact selector {pending.get('selector')!r} "
                    "is no longer feasible"
                ),
                target_tier=pending.get("target_tier"),
            )
        target_tier = int(pending["target_tier"])
        tried.add(target_tier)
        state["downward_tiers_tried"] = sorted(tried)
        attempt = {
            "selector": chosen.selector,
            "model_id": chosen.model_id,
            "quant": chosen.quant,
            "tier": chosen.tier,
            "score": None,
            "weights_ref": None,
            "result": "error",
            "adopted": False,
            "error": None,
            "H": dict(pending_h),
        }
        try:
            weights_ref, result = _train_and_eval(
                chosen,
                state,
                pending["dataset_path"],
                pending_h,
            )
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            history["attempts"].append(attempt)
            _plog(
                f"  probe failed for {chosen.model_id} "
                f"({str(exc)[:100]}) — keeping current model"
            )
            return _finish_probe(state)
        adopted = result.f1 >= threshold
        attempt.update({
            "score": result.f1,
            "weights_ref": weights_ref,
            "result": "adopted" if adopted else "rejected",
            "adopted": adopted,
        })
        history["attempts"].append(attempt)
        state["downward_probe_pending"] = None
        if not adopted:
            _plog(
                f"  ✗ {chosen.model_id} scored {result.f1:.4f} "
                f"< {threshold:.4f} — stopping downward search"
            )
            return _finish_probe(state)
        state["selected_model"] = chosen
        state["best_weights_ref"] = weights_ref
        state["best_score"] = result.f1
        state["last_eval"] = result
        state["converged_model_ref"] = {
            "selector": chosen.selector,
            "model_id": chosen.model_id,
            "quant": chosen.quant,
            "tier": chosen.tier,
            "score": result.f1,
        }
        _plog(
            f"  ✓ ADOPT smaller {chosen.model_id} "
            f"[{chosen.quant or 'bf16'}] (tier {target_tier}, "
            f"{result.f1:.4f} ≥ {threshold:.4f})"
        )
        return state

    cur_tier = current.tier
    lower_tiers = sorted(
        {
            model.tier
            for model in feasible
            if model.tier < cur_tier and model.tier not in tried
        },
        reverse=True,
    )
    if not lower_tiers:
        _plog(
            f"no untried feasible tiers below tier {cur_tier} — done. "
            f"Final: {current.model_id} [{current.quant or 'bf16'}]"
        )
        return _finish_probe(state)
    try:
        should_reexplore = _should_reexplore_downward(state, len(lower_tiers))
    except Exception as exc:
        return skip_optional_error("reexploration_gate", exc)
    if not should_reexplore:
        _plog(
            "orchestrator declined further downward re-exploration — "
            "keeping current model"
        )
        return _finish_probe(state)
    target_tier = lower_tiers[0]
    candidates = [model for model in feasible if model.tier == target_tier]
    try:
        chosen = _llm_choose_model(
            candidates=candidates,
            task_type=state.get("task_type", "classification"),
            task_plan=state.get("task_plan") or {},
            current_best_score=state["best_score"],
            log=lambda message: _plog(f"  {message}"),
            direction="down",
        )
    except Exception as exc:
        return skip_optional_error(
            "model_chooser",
            exc,
            target_tier=target_tier,
            candidates=candidates,
        )
    state["downward_probe_pending"] = {
        "selector": chosen.selector,
        "target_tier": target_tier,
        "dataset_path": dataset_path,
        "H": dict(DOWNWARD_PROBE_H),
    }
    _plog(
        f"planned tier {cur_tier} → {target_tier}: "
        f"{chosen.selector}; checkpointing choice before training"
    )
    return state


def downward_probe_node(state: AgentState) -> AgentState:
    """Compatibility wrapper that runs durable probe steps to completion."""
    for _ in range(100):
        state = downward_probe_step_node(state)
        if state.get("next_action") != "downward_probe":
            return state
    raise RuntimeError("downward probe exceeded 100 durable steps")
