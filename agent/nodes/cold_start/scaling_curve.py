# agent/nodes/cold_start/scaling_curve.py
"""
Node 1b (runs after task_analysis, before eval_setup is used by the main loop):
Fit an accuracy-vs-log(size) scaling curve from 3 fine-tuned probe runs,
then select the smallest model whose predicted accuracy meets stop_threshold.
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
    """Fine-tune one epoch and return eval f1. Returns 0.0 on failure.

    When current_dataset_path is None (scaling_curve runs before curate),
    builds a mini seed dataset from eval_set examples to get a real ranking signal.
    """
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
        # No curated dataset yet (scaling_curve runs before curate). Build a
        # minimal seed from the eval set so probes produce a real ranking signal.
        eval_set = state.get("eval_set")
        if eval_set is None:
            logger.warning(
                "[scaling_curve] No dataset or eval_set; skipping probe for %s", model.model_id
            )
            return 0.0
        seed_examples = list(eval_set.pos) + list(eval_set.neg) + list(eval_set.boundary)
        if not seed_examples:
            logger.warning(
                "[scaling_curve] eval_set is empty; skipping probe for %s", model.model_id
            )
            return 0.0
        _seed_fd, _seed_file = tempfile.mkstemp(suffix=".jsonl", prefix="slm_probe_seed_")
        with os.fdopen(_seed_fd, "w") as f:
            for ex in seed_examples:
                f.write(json.dumps(ex) + "\n")
        dataset_path = _seed_file
        logger.info(
            "[scaling_curve] Using %d eval-set examples as probe seed for %s",
            len(seed_examples), model.model_id,
        )
    try:
        weights_ref = run_lora_training(
            dataset_path, config, output_dir=model_dir, task_type=state["task_type"]
        ).weights_ref
        result = run_eval(state["eval_set"], weights_ref, model.model_id, state["task_type"])
        logger.info("[scaling_curve] Probe %s → f1=%.4f", model.model_id, result.f1)
        return result.f1
    except Exception as exc:
        logger.warning("[scaling_curve] Probe failed for %s: %s", model.model_id, exc)
        return 0.0
    finally:
        if _seed_file and os.path.exists(_seed_file):
            os.remove(_seed_file)


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

    # Capability axis is log(params), NOT log(disk size): a model's BF16/Q8/Q4 variants
    # share parameter count and accuracy, so params is the honest capability proxy.
    # (Using disk size would falsely rank a model's own BF16 as "more capable" than its Q4.)
    with tempfile.TemporaryDirectory(prefix="slm_probe_") as probe_dir:
        points: list[tuple[float, float]] = []
        for model in candidates:
            f1 = _probe_model(model, state, probe_dir)
            log_params = math.log(max(model.est_params_b(), 1e-3))
            points.append((log_params, f1))
            logger.info(
                "[scaling_curve] Point: log(params)=%.3f f1=%.4f (%s, ~%.2fB, quant=%s)",
                log_params, f1, model.model_id, model.est_params_b(), model.quant,
            )

    if len(points) < 2:
        # Can't fit a line; fall back to the lowest-RAM feasible variant.
        logger.warning("[scaling_curve] Too few probe points (%d); selecting lowest-RAM model.", len(points))
        state["selected_model"] = min(feasible, key=lambda m: m.peak_memory_mb)
        return state

    log_params = np.array([p[0] for p in points])
    f1s = np.array([p[1] for p in points])
    coeffs = np.polyfit(log_params, f1s, deg=1)  # [a, b]: f1 = a*log(params) + b
    a, b = float(coeffs[0]), float(coeffs[1])
    logger.info("[scaling_curve] Fit: f1 = %.4f * log(params) + %.4f", a, b)

    # Objective: the MINIMUM peak-RAM variant whose predicted f1 clears the threshold.
    # Iterate feasible variants ascending by peak RAM and take the first that qualifies.
    by_ram = sorted(feasible, key=lambda m: m.peak_memory_mb)
    for model in by_ram:
        predicted = a * math.log(max(model.est_params_b(), 1e-3)) + b
        qualifies = predicted >= stop_threshold
        logger.info(
            "[scaling_curve] %s (quant=%s, peak=%dMB): predicted_f1=%.4f vs %.4f → %s",
            model.model_id, model.quant, model.peak_memory_mb, predicted, stop_threshold,
            "SELECT" if qualifies else "skip",
        )
        if qualifies:
            state["selected_model"] = model
            logger.info(
                "[scaling_curve] Selected lowest-RAM qualifier: %s (tier=%d, quant=%s, "
                "peak=%dMB, predicted_f1=%.4f)",
                model.model_id, model.tier, model.quant, model.peak_memory_mb, predicted,
            )
            return state

    # None predicted to meet threshold — pick the highest-capability feasible variant
    # (most params; ties broken by more RAM = less aggressive quant) as the best shot.
    best = max(feasible, key=lambda m: (m.est_params_b(), m.peak_memory_mb))
    state["selected_model"] = best
    logger.warning(
        "[scaling_curve] No variant predicted to meet threshold %.4f; "
        "defaulting to highest-capability: %s (quant=%s)",
        stop_threshold, best.model_id, best.quant,
    )
    return state
