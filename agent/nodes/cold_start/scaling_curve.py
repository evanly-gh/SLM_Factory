# agent/nodes/cold_start/scaling_curve.py
"""
Node 1b (runs after task_analysis, before eval_setup is used by the main loop):
Fit an accuracy-vs-log(size) scaling curve from 3 fine-tuned probe runs,
then select the smallest model whose predicted accuracy meets stop_threshold.
"""
import logging
import math
import os
import tempfile

import numpy as np

from agent.state import AgentState
from config.android_pool import ModelSpec
from eval.harness import run_eval
from training.lora_trainer import TrainingConfig, run_lora_training

logger = logging.getLogger(__name__)

# Probe training config: 1 epoch, minimal LoRA — just enough for a ranking signal.
_PROBE_EPOCHS = 1
_PROBE_LORA_RANK = 8
_PROBE_LR = 2e-4
_PROBE_BATCH = 8


def _pick_candidates(models: list[ModelSpec]) -> list[ModelSpec]:
    """Pick up to 3 candidates: largest, middle, smallest."""
    if len(models) <= 1:
        return models
    if len(models) == 2:
        return [models[0], models[-1]]
    mid = len(models) // 2
    seen = set()
    result = []
    for idx in (0, mid, len(models) - 1):
        m = models[idx]
        key = (m.model_id, m.quant)
        if key not in seen:
            result.append(m)
            seen.add(key)
    return result


def _probe_model(
    model: ModelSpec,
    state: AgentState,
    probe_dir: str,
) -> float:
    """Fine-tune one epoch and return eval f1. Returns 0.0 on failure."""
    model_dir = os.path.join(probe_dir, model.model_id.replace("/", "_"))
    config = TrainingConfig(
        base_model=model.model_id,
        nr_epochs=_PROBE_EPOCHS,
        learning_rate=_PROBE_LR,
        batch_size=_PROBE_BATCH,
        lora_rank=_PROBE_LORA_RANK,
        task_type=state["task_type"],
    )
    try:
        dataset_path = state.get("current_dataset_path")
        if not dataset_path:
            logger.warning("[scaling_curve] No dataset path in state; skipping probe for %s", model.model_id)
            return 0.0
        weights_ref = run_lora_training(dataset_path, config, output_dir=model_dir, task_type=state["task_type"]).weights_ref
        result = run_eval(state["eval_set"], weights_ref, model.model_id, state["task_type"])
        logger.info("[scaling_curve] Probe %s → f1=%.4f", model.model_id, result.f1)
        return result.f1
    except Exception as exc:
        logger.warning("[scaling_curve] Probe failed for %s: %s", model.model_id, exc)
        return 0.0


def scaling_curve_node(state: AgentState) -> AgentState:
    """
    Node 1b: fit accuracy-vs-log(size) curve, select smallest model above stop_threshold.

    Reads:  state["feasible_models"], state["stop_threshold"], state["eval_set"],
            state["current_dataset_path"], state["task_type"]
    Writes: state["selected_model"]
    """
    feasible = state.get("feasible_models", [])
    if not feasible:
        raise RuntimeError("scaling_curve_node: feasible_models is empty.")

    forced = os.environ.get("SLM_FORCE_MODEL")
    if forced:
        match = next((m for m in feasible if m.model_id == forced), None)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
        logger.info("[scaling_curve] SLM_FORCE_MODEL=%s pinned — skipping curve fit", forced)
        return state

    stop_threshold = state.get("stop_threshold", 0.96)
    candidates = _pick_candidates(feasible)

    logger.info(
        "[scaling_curve] Probing %d candidates: %s",
        len(candidates), [m.model_id for m in candidates],
    )

    with tempfile.TemporaryDirectory(prefix="slm_probe_") as probe_dir:
        points: list[tuple[float, float]] = []
        for model in candidates:
            f1 = _probe_model(model, state, probe_dir)
            log_size = math.log(model.int4_size_mb)
            points.append((log_size, f1))
            logger.info(
                "[scaling_curve] Point: log(size)=%.3f f1=%.4f (%s, quant=%s)",
                log_size, f1, model.model_id, model.quant,
            )

    if len(points) < 2:
        # Can't fit a line; fall back to smallest feasible model.
        logger.warning("[scaling_curve] Too few probe points (%d); selecting smallest model.", len(points))
        # feasible is largest→smallest; smallest is last
        state["selected_model"] = feasible[-1]
        return state

    log_sizes = np.array([p[0] for p in points])
    f1s = np.array([p[1] for p in points])
    coeffs = np.polyfit(log_sizes, f1s, deg=1)  # [a, b]: f1 = a*log(size) + b
    a, b = float(coeffs[0]), float(coeffs[1])
    logger.info("[scaling_curve] Fit: f1 = %.4f * log(size) + %.4f", a, b)

    # Walk smallest→largest; pick first whose predicted f1 >= stop_threshold
    for model in reversed(feasible):  # feasible is largest→smallest, so reversed = smallest→largest
        predicted = a * math.log(model.int4_size_mb) + b
        logger.info(
            "[scaling_curve] %s: predicted_f1=%.4f vs threshold=%.4f → %s",
            model.model_id, predicted, stop_threshold,
            "SELECT" if predicted >= stop_threshold else "skip",
        )
        if predicted >= stop_threshold:
            state["selected_model"] = model
            logger.info(
                "[scaling_curve] Selected: %s (tier=%d, quant=%s, predicted_f1=%.4f)",
                model.model_id, model.tier, model.quant, predicted,
            )
            return state

    # None predicted to meet threshold — use largest (best chance)
    state["selected_model"] = feasible[0]
    logger.warning(
        "[scaling_curve] No model predicted to meet threshold %.4f; "
        "defaulting to largest: %s", stop_threshold, feasible[0].model_id,
    )
    return state
