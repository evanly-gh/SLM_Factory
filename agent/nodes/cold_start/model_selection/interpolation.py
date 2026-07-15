# agent/nodes/cold_start/model_selection/interpolation.py
"""
Strategy: Interpolation (3-Probe Scaling Curve)

Probe 3 models (smallest, middle, largest from the feasible set) with 1-epoch
LoRA training, fit a log-linear accuracy-vs-size curve, then select the model
whose predicted peak RAM is closest to the hardware memory budget while still
meeting the accuracy goal.

This is the original scaling_curve approach from the paper, with one
refinement: instead of picking the smallest model that meets the threshold,
it picks the one closest to the RAM target — biasing toward fuller use of
the available hardware budget for better accuracy.
"""
import json
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

_PROBE_EPOCHS = 1
_PROBE_LORA_RANK = 8
_PROBE_LR = 2e-4
_PROBE_BATCH = 8


def _pick_candidates(models: list[ModelSpec]) -> list[ModelSpec]:
    """Pick up to 3 candidates: smallest, middle, largest."""
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
    dataset_path = state.get("current_dataset_path")
    _seed_file = None
    if not dataset_path:
        eval_set = state.get("eval_set")
        if eval_set is None:
            logger.warning(
                "[interpolation] No dataset or eval_set; skipping probe for %s", model.model_id
            )
            return 0.0
        seed_examples = list(eval_set.pos) + list(eval_set.neg) + list(eval_set.boundary)
        if not seed_examples:
            return 0.0
        _seed_fd, _seed_file = tempfile.mkstemp(suffix=".jsonl", prefix="slm_probe_seed_")
        with os.fdopen(_seed_fd, "w") as f:
            for ex in seed_examples:
                f.write(json.dumps(ex) + "\n")
        dataset_path = _seed_file
    try:
        weights_ref = run_lora_training(
            dataset_path, config, output_dir=model_dir, task_type=state["task_type"]
        ).weights_ref
        result = run_eval(state["eval_set"], weights_ref, model.model_id, state["task_type"])
        logger.info("[interpolation] Probe %s -> f1=%.4f", model.model_id, result.f1)
        return result.f1
    except Exception as exc:
        logger.warning("[interpolation] Probe failed for %s: %s", model.model_id, exc)
        return 0.0
    finally:
        if _seed_file and os.path.exists(_seed_file):
            os.remove(_seed_file)


def interpolation_node(state: AgentState) -> AgentState:
    """
    Probe 3 models, fit curve, select model closest to RAM target that meets threshold.
    """
    feasible = state.get("feasible_models", [])
    if not feasible:
        raise RuntimeError("interpolation_node: feasible_models is empty.")

    forced = os.environ.get("SLM_FORCE_MODEL")
    if forced:
        match = next((m for m in feasible if m.model_id == forced), None)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
        logger.info("[model_selection:interpolation] SLM_FORCE_MODEL=%s pinned", forced)
        return state

    stop_threshold = state.get("stop_threshold", 0.96)
    ram_target = state["hardware_constraints"].memory_mb
    candidates = _pick_candidates(feasible)

    logger.info(
        "[interpolation] Probing %d candidates: %s (RAM target=%dMB)",
        len(candidates), [m.model_id for m in candidates], ram_target,
    )

    with tempfile.TemporaryDirectory(prefix="slm_probe_") as probe_dir:
        points: list[tuple[float, float]] = []
        for model in candidates:
            f1 = _probe_model(model, state, probe_dir)
            log_params = math.log(max(model.est_params_b(), 1e-3))
            points.append((log_params, f1))
            logger.info(
                "[interpolation] Point: log(params)=%.3f f1=%.4f (%s, ~%.2fB, quant=%s)",
                log_params, f1, model.model_id, model.est_params_b(), model.quant,
            )

    if len(points) < 2:
        chosen = min(feasible, key=lambda m: abs(m.peak_memory_mb - ram_target))
        logger.warning("[interpolation] Too few points; selecting closest to RAM target: %s", chosen.model_id)
        state["selected_model"] = chosen
        return state

    log_params = np.array([p[0] for p in points])
    f1s = np.array([p[1] for p in points])
    coeffs = np.polyfit(log_params, f1s, deg=1)
    a, b = float(coeffs[0]), float(coeffs[1])
    logger.info("[interpolation] Fit: f1 = %.4f * log(params) + %.4f", a, b)

    qualifiers = []
    for model in feasible:
        predicted = a * math.log(max(model.est_params_b(), 1e-3)) + b
        if predicted >= stop_threshold:
            qualifiers.append(model)
            logger.info(
                "[interpolation] %s (peak=%dMB): predicted=%.4f >= %.4f -> qualifies",
                model.model_id, model.peak_memory_mb, predicted, stop_threshold,
            )

    if qualifiers:
        chosen = min(qualifiers, key=lambda m: abs(m.peak_memory_mb - ram_target))
        logger.info(
            "[interpolation] Selected closest to RAM target (%dMB): %s (peak=%dMB)",
            ram_target, chosen.model_id, chosen.peak_memory_mb,
        )
    else:
        chosen = max(feasible, key=lambda m: (m.est_params_b(), m.peak_memory_mb))
        logger.warning(
            "[interpolation] No model predicted to meet threshold %.4f; "
            "defaulting to highest-capability: %s",
            stop_threshold, chosen.model_id,
        )

    state["selected_model"] = chosen
    return state
