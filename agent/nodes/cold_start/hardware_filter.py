# agent/nodes/cold_start/hardware_filter.py
"""
Staged hardware filter — runs pre-graph after hardware_research.

Stage 1: inequality filter (storage, memory, throughput) — free.
Stage 2: on-device eval stub (latency, power, thermal) — iterates largest→smallest,
         discards violators. Currently a stub; Phase 2 wires real ADB runs.
"""
import logging
from config.android_pool import (
    HardwareConstraints,
    ModelSpec,
    filter_pool,
    check_hardware_constraints,
    all_constraints_pass,
    ANDROID_POOL,
)
from hardware_eval.on_device_eval import run_on_device_eval

logger = logging.getLogger(__name__)


def run_hardware_filter(constraints: HardwareConstraints) -> list[ModelSpec]:
    """
    Return models that pass both stages, sorted largest→smallest
    (by size_mb descending within tier) so scaling_curve_node
    can slice small/medium/large candidates from the ends and middle.

    Stage 1: filter_pool() inequality checks (storage, memory, min_tok_s).
    Stage 2: on-device eval from largest candidate downward; discard failures.
    """
    # ── Stage 1 ──────────────────────────────────────────────────────────────
    stage1 = filter_pool(constraints)
    logger.info("[hardware_filter] Stage 1: %d/%d models passed inequality checks",
                len(stage1), len(ANDROID_POOL))

    if not stage1:
        logger.warning("[hardware_filter] Stage 1 eliminated all models.")
        return []

    # Sort largest→smallest for Stage 2 iteration
    stage1_desc = sorted(stage1, key=lambda m: m.size_mb, reverse=True)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    # No GGUF exists yet at pre-training screening, so run_on_device_eval falls
    # back to the theoretical backend regardless of SLM_HW_BACKEND. Real measured
    # gating happens post-convergence (see run.py on-device verification step).
    passed: list[ModelSpec] = []
    for model in stage1_desc:
        hw_result = run_on_device_eval(model, constraints)
        hw_check = check_hardware_constraints(
            model, constraints, measured=hw_result.to_measured()
        )
        ok = all_constraints_pass(hw_check)
        logger.info(
            "[hardware_filter] Stage 2: %s %s (ttft=%.0fms, tok/s=%.1f)",
            "✓" if ok else "✗",
            model.model_id,
            hw_result.ttft_ms or 0,
            hw_result.tok_per_s or 0,
        )
        if ok:
            passed.append(model)

    logger.info(
        "[hardware_filter] Stage 2: %d/%d models passed on-device check",
        len(passed), len(stage1_desc),
    )
    return passed  # largest→smallest order preserved
