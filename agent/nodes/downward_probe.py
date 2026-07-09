# agent/nodes/downward_probe.py
"""
Node 9: active downward probe.

When the loop is about to terminate with success, train + eval the best model one
tier BELOW the current model on the current dataset. If that smaller model also
clears stop_threshold, adopt it (minimum-resource terminal model). Always terminates.
"""
import logging
import os
from agent.state import AgentState
from config.android_pool import filter_pool
from training.lora_trainer import TrainingConfig, run_lora_training, merge_for_quantization
from training.quantize import quantize_from_model_spec
from eval.harness import run_eval
from agent.nodes.escalate import _llm_choose_model

logger = logging.getLogger(__name__)

_PROBE_LORA_RANK = 8
_PROBE_LR = 2e-4
_PROBE_BATCH = 8
_PROBE_EPOCHS = 3


def _train_and_eval(model, state, dataset_path: str):
    """Train `model` on dataset_path and return (weights_ref, EvalResult).
    Uses the honest quantized-eval path when model.quant is set."""
    task_type = state["task_type"]
    model_id = model.model_id
    model_id_safe = model_id.replace("/", "_")
    current_safe = state["selected_model"].model_id.replace("/", "_")
    config = TrainingConfig(
        base_model=model_id,
        nr_epochs=_PROBE_EPOCHS,
        learning_rate=_PROBE_LR,
        batch_size=_PROBE_BATCH,
        lora_rank=_PROBE_LORA_RANK,
        task_type=task_type,
    )
    weights_ref = run_lora_training(
        dataset_path, config,
        output_dir=os.path.join("artifacts", "downward_probe", current_safe, model_id_safe),
        task_type=task_type,
    ).weights_ref
    gguf_path = None
    if model.quant is not None:
        merged_path = merge_for_quantization(
            weights_ref,
            os.path.join("artifacts", "merged", model_id_safe, f"downward_probe_from_{current_safe}"),
        )
        gguf_path = quantize_from_model_spec(
            merged_path,
            os.path.join("artifacts", "gguf", model_id_safe, f"downward_probe_from_{current_safe}"),
            model.quant,
        )
    result = run_eval(
        state["eval_set"], weights_ref, model_id,
        task_type=task_type, quant=model.quant, gguf_path=gguf_path,
    )
    return weights_ref, result


def downward_probe_node(state: AgentState) -> AgentState:
    """Node 9: try the best one-tier-down model; adopt it if it clears threshold."""
    state["downward_probe_done"] = True
    state["next_action"] = "terminate"  # this node always terminates

    current = state.get("selected_model")
    if current is None:
        return state
    model_id = current.model_id

    if current.tier <= 0:
        logger.info("[downward_probe][%s] Already at tier 0 — nothing smaller to try", model_id)
        return state

    dataset_path = state.get("current_dataset_path")
    if not dataset_path:
        logger.info("[downward_probe][%s] No dataset available — skipping probe", model_id)
        return state

    feasible = filter_pool(state["hardware_constraints"])
    lower = [m for m in feasible if m.tier == current.tier - 1]
    if not lower:
        logger.info("[downward_probe][%s] No feasible models in tier %d — skipping",
                    model_id, current.tier - 1)
        return state

    # Reuse escalate's LLM-driven model chooser to pick the best lower-tier candidate.
    chosen = _llm_choose_model(
        candidates=lower,
        task_type=state.get("task_type", "classification"),
        task_plan=state.get("task_plan") or {},
        current_best_score=state["best_score"],
    )
    logger.info("[downward_probe][%s] Probing smaller model %s (tier %d)",
                model_id, chosen.model_id, chosen.tier)

    try:
        weights_ref, result = _train_and_eval(chosen, state, dataset_path)
    except Exception as exc:
        logger.warning("[downward_probe][%s] Probe failed for %s: %s — keeping current model",
                       model_id, chosen.model_id, exc)
        return state

    threshold = state["stop_threshold"]
    logger.info("[downward_probe][%s] %s scored %.4f vs threshold %.4f",
                model_id, chosen.model_id, result.f1, threshold)
    if result.f1 >= threshold:
        logger.info("[downward_probe][%s] ADOPTING smaller model %s (%.4f >= %.4f)",
                    model_id, chosen.model_id, result.f1, threshold)
        state["selected_model"] = chosen
        state["best_weights_ref"] = weights_ref
        state["best_score"] = result.f1
        state["last_eval"] = result
    else:
        logger.info("[downward_probe][%s] Smaller model did not clear threshold — keeping current",
                    model_id)
    return state
