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

    The accuracy target is the reference teacher's own FIVE-SHOT score on THIS run's frozen eval
    set E, measured with the identical metric, floored at 0.8 and capped at THRESHOLD_CEILING.
    "Good enough" therefore means "matches the strong reference," not a recalled leaderboard
    number. E does not exist at task-analysis time, so the goal is parked PENDING and measured in
    eval_setup right after E is built.

    Five-shot, and the SAME measurement that gates synthetic data. It used to be a second,
    zero-shot pass taken separately from the synthesis gate's five-shot one, which meant a run
    measured its own teacher twice with two prompts that disagree by multiples on a format-bound
    task (BC5CDR: 0.1131 against 0.7190) and set the goal from the shape nothing else in the
    pipeline ever sends. See `agent/teacher_fitness.py`, which now owns the measurement.

    There is no registry lookup, no late-bound measured anchor, and no headroom knob: the older
    multi-source design was removed in favour of this one measurement. If the teacher cannot be
    measured, calibration RAISES and the run stops — it does not silently fall back to a
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
    """Goal = the reference teacher's five-shot score on THIS run's eval set, floored.

    The accuracy target a run must beat is the strong reference model's own performance on the
    identical frozen E, scored by the identical metric — so "good enough" means "matches the
    reference," not a recalled leaderboard number. Floored so a weak reference cannot set a
    trivially-low goal, and capped at THRESHOLD_CEILING because label noise makes 1.0 unreachable.
    An UNMEASURABLE teacher is handled by the caller (it raises); this function only shapes a
    measured score.
    """
    try:
        measured_value = float(measured)
    except (TypeError, ValueError):
        measured_value = 0.0
    floor_value = min(max(float(floor), 0.0), THRESHOLD_CEILING)
    value = min(THRESHOLD_CEILING, max(measured_value, floor_value))
    reason = (
        f"teacher baseline {measured_value:.4f} on E, floored at {floor_value:.2f}, "
        f"capped at {THRESHOLD_CEILING}"
    )
    return round(value, 4), reason


def describe_measured_teacher(calibration: dict | None) -> str:
    """`<model> <k>-shot <metric>=<score> (format_valid=<fv>) on <n> eval row(s)`.

    Every field is read from the calibration record rather than assumed, because none of them are
    constant any more: the teacher is Qwen3.6 or a DeepSeek model, the prompt is five-shot or (on a
    task whose rows crowd out the demonstrations) zero-shot, and the row count is the eval set's.
    Older checkpoints carry only the score and metric, so each addition degrades to silence rather
    than to a wrong claim — printing "5-shot" for a record that never stored a shot count would be
    inventing the very provenance this line exists to report.
    """
    measured = float(calibration.get("measured_qwen"))
    metric = str(calibration.get("measured_metric") or "score")
    model = str(calibration.get("measured_model") or "").strip() or "teacher"
    shots = calibration.get("measured_shots")
    shot_text = f"{int(shots)}-shot " if isinstance(shots, int) else ""
    format_valid = calibration.get("measured_format_valid")
    format_text = (
        f" (format_valid={float(format_valid):.4f})"
        if isinstance(format_valid, (int, float))
        else ""
    )
    rows = calibration.get("measured_n")
    rows_text = f" on {int(rows)} eval row(s)" if isinstance(rows, int) and rows else ""
    return f"{model} {shot_text}{measured:.4f} {metric}{format_text}{rows_text}"


def describe_threshold_provenance(calibration: dict | None) -> str:
    """One-line provenance for the accuracy goal, for the run summary.

    The goal is the teacher's own five-shot score on E, floored at 0.8. When the teacher scores
    BELOW the floor the floor wins, and the resulting number looks identical to a goal the teacher
    actually set — BC5CDR converged at "threshold 0.8000" while the teacher had in fact scored
    0.0999, and nothing in the summary said so. Reporting which input won, and the measured teacher
    score either way, is what makes a converged run interpretable: clearing a floor the teacher
    could not reach is a very different result from matching a teacher that scored 0.87.
    """
    if not isinstance(calibration, dict) or not calibration:
        return "provenance unrecorded"
    source = str(calibration.get("source") or "unknown")
    if source == "pending_qwen_baseline":
        return "pending teacher measurement (eval set not yet built)"
    if source == "manual_override":
        return "set explicitly via SLM_STOP_THRESHOLD"
    if calibration.get("measured_qwen") is None:
        return f"source={source}"
    described = describe_measured_teacher(calibration)
    floor = calibration.get("floor")
    if calibration.get("floored"):
        return f"floor {float(floor):.2f} OVERRODE the measured teacher: {described}"
    return (
        f"from the measured teacher: {described} (above the {float(floor):.2f} floor)"
        if floor is not None
        else f"from the measured teacher: {described}"
    )
