"""Stop-threshold calibration (B32).

WHAT THIS REPLACES
    The planner used to be told: "anchor stop_threshold to the PUBLISHED STATE-OF-THE-ART for
    this benchmark at ~{param_range} scale ... roughly SOTA - 2 to 5 points", with a hardcoded
    0.96 default when it declined. That made the run's termination target a product of the
    orchestrator's frozen training recall, and the error is asymmetric:

      too LOW  -> a base model's zero-shot already clears it; the run "converges" at iteration
                  1 having learned nothing.
      too HIGH -> every tier fails; the run burns its whole budget concluding "infeasible".

THE MECHANISM — a single source of truth:

    The accuracy target is the separately-hosted Qwen-3.6 reference model's own zero-shot score
    on THIS run's frozen eval set E, measured with the identical metric, floored at 0.8 and
    capped at THRESHOLD_CEILING. "Good enough" therefore means "matches the strong reference,"
    not a recalled leaderboard number. E does not exist at task-analysis time, so the goal is
    parked PENDING and measured in eval_setup right after E is built.

    There is no registry lookup, no late-bound measured anchor, and no headroom knob: the older
    multi-source design was removed in favour of this one measurement. If the Qwen endpoint is
    unreachable, calibration RAISES and the run stops — it does not silently fall back to a
    guessed target.

HAZARD THE DESIGN GUARDS

    An unfalsifiable target. iterate_node can already lower stop_threshold at runtime. Guard:
    eval_setup writes initial_stop_threshold ONCE (the immutable floor), and every input is
    recorded in state["threshold_calibration"] for audit.
"""
from __future__ import annotations

# Never target a perfect score: label noise alone makes 1.0 unreachable on every real benchmark
# (BANKING77's documented label errors are the clearest case).
THRESHOLD_CEILING = 0.99

# Provisional value while the Qwen baseline is deferred to eval_setup. Deliberately above
# THRESHOLD_CEILING so no score can satisfy it — the run cannot converge before E is measured.
UNREACHABLE_PENDING_THRESHOLD = 1.0


def threshold_from_endpoint_baseline(
    measured: float, floor: float = 0.8
) -> tuple[float, str]:
    """Goal = the separately-hosted Qwen-3.6 base score on THIS run's eval set, floored.

    The accuracy target a run must beat is the strong reference model's own zero-shot
    performance on the identical frozen E, scored by the identical metric — so "good enough"
    means "matches the reference," not a recalled leaderboard number. Floored so a weak
    reference cannot set a trivially-low goal, and capped at THRESHOLD_CEILING because label
    noise makes 1.0 unreachable. An UNREACHABLE endpoint is handled by the caller (it raises);
    this function only shapes a measured score.
    """
    try:
        measured_value = float(measured)
    except (TypeError, ValueError):
        measured_value = 0.0
    floor_value = min(max(float(floor), 0.0), THRESHOLD_CEILING)
    value = min(THRESHOLD_CEILING, max(measured_value, floor_value))
    reason = (
        f"Qwen-3.6 baseline {measured_value:.4f} on E, floored at {floor_value:.2f}, "
        f"capped at {THRESHOLD_CEILING}"
    )
    return round(value, 4), reason
