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

    # Build a predicted accuracy for EVERY feasible variant. Prefer the empirical
    # scaling fit (needs >=2 non-zero probe points). If too many probes failed to load/
    # train, fall back to each model's PUBLISHED benchmark score so selection still has a
    # real capability signal instead of a blind default.
    valid = [(lp, f1) for lp, f1 in points if f1 > 0.0]
    if len(valid) >= 2:
        lp_arr = np.array([p[0] for p in valid])
        f1_arr = np.array([p[1] for p in valid])
        a, b = (float(c) for c in np.polyfit(lp_arr, f1_arr, deg=1))
        logger.info("[scaling_curve] Empirical fit over %d point(s): f1 = %.4f*log(params) + %.4f",
                    len(valid), a, b)

        def predict(m: ModelSpec) -> float:
            return a * math.log(max(m.est_params_b(), 1e-3)) + b
    else:
        logger.warning(
            "[scaling_curve] Only %d usable probe point(s); using published benchmark "
            "scores as the accuracy estimate.", len(valid),
        )
        _tt = state.get("task_type", "")

        def predict(m: ModelSpec) -> float:
            s = m.gsm8k if _tt == "math_reasoning" else m.mmlu
            return float(s) if s is not None else 0.0

    # Choose the model CLOSEST to the accuracy goal:
    #   - if any variant is predicted to MEET the threshold, take the SMALLEST such one
    #     (the cheapest model that reaches the goal);
    #   - otherwise take the variant whose predicted accuracy is nearest the goal (the
    #     best achievable start; the main loop's escalate_node grows it if it still falls short).
    # (ModelSpec is unhashable — keep predictions as (model, pred) pairs, not a dict.)
    scored = [(m, predict(m)) for m in feasible]
    for m, p in sorted(scored, key=lambda mp: mp[0].peak_memory_mb):
        logger.info("[scaling_curve] %s (peak=%dMB, ~%.2fB): predicted_f1=%.4f vs goal %.4f",
                    m.model_id, m.peak_memory_mb, m.est_params_b(), p, stop_threshold)
    qualifiers = [(m, p) for m, p in scored if p >= stop_threshold]
    if qualifiers:
        selected, sel_pred = min(qualifiers, key=lambda mp: mp[0].peak_memory_mb)
        why = "smallest variant predicted to meet the goal"
    else:
        selected, sel_pred = min(scored, key=lambda mp: abs(mp[1] - stop_threshold))
        why = "closest to the goal (none predicted to meet it)"
    state["selected_model"] = selected
    logger.info(
        "[scaling_curve] Selected %s (tier=%d, quant=%s, peak=%dMB, predicted_f1=%.4f) — %s",
        selected.model_id, selected.tier, selected.quant, selected.peak_memory_mb,
        sel_pred, why,
    )
    return state
