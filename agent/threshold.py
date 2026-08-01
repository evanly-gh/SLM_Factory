"""Stop-threshold calibration (B32).

WHAT THIS REPLACES
    The planner used to be told: "anchor stop_threshold to the PUBLISHED STATE-OF-THE-ART for
    this benchmark at ~{param_range} scale ... roughly SOTA - 2 to 5 points", with a hardcoded
    0.96 default when it declined. That made the run's termination target a product of the
    orchestrator's frozen training recall, and the error is asymmetric:

      too LOW  -> a base model's zero-shot already clears it; the run "converges" at iteration
                  1 having learned nothing.
      too HIGH -> every tier fails; the run burns its whole budget concluding "infeasible".

    Both were observed. Benchmark research on 2026-07-28 found BANKING77 SOTA is ~94.8% from a
    110M encoder (so a 1-4B target is nonsense) and that MedQA is saturating.

THE REPLACEMENT — two sources, in order of preference:

    1. REGISTRY. config/benchmark_baselines.md holds sourced rows with an explicit metric name.
       A row calibrates the threshold only when its metric matches what this pipeline actually
       measures for the task type (eval/harness.py::TASK_METRIC_NAMES). Reviewable and diffable;
       cannot be moved by a prompt injection in a scraped page.

    2. MEASURED ANCHOR (late-bound). No usable row -> defer. After the first train+evaluate
       cycle, anchor on max(zero_shot_baseline, first_finetune_score) and add a BOUNDED headroom
       the orchestrator selects. Measured on YOUR eval set with YOUR prompt, so it cannot go
       stale.

THREE HAZARDS THE DESIGN GUARDS

    Anchoring on a bad first config. Iteration 1 uses _DEFAULT_CONFIG; a poor draw would depress
    the anchor and set a goal the run clears instantly. Guard: the anchor is
    max(zero_shot, first_finetune). Zero-shot is config-independent and already measured for
    free at iteration 1, so it floors the anchor.

    An unfalsifiable target. A goal derived from your own first result can always be met by
    lowering it, and iterate_node can already lower stop_threshold at runtime. Guard: this
    writes initial_stop_threshold ONCE (the existing immutable floor), and every input is
    recorded in state["threshold_calibration"] for audit.

    "How much headroom" is what an LLM is worst at. Guard: VALID_HEADROOMS is a small ordinal
    set, snapped like every other bounded field in this repo, with a required reason.
"""
from __future__ import annotations

import os
import re

# Bounded headroom choices. A free-text number invites a confident guess; four rungs with a
# required justification is a decision the orchestrator can actually make well.
VALID_HEADROOMS = (0.02, 0.05, 0.10, 0.15)
DEFAULT_HEADROOM = 0.05

# Never target a perfect score: label noise alone makes 1.0 unreachable on every real benchmark
# in the registry (BANKING77's documented label errors are the clearest case).
THRESHOLD_CEILING = 0.99

# Provisional value while calibration is deferred. Deliberately above THRESHOLD_CEILING so no
# score can satisfy it — the run cannot converge before the anchor has been measured.
UNREACHABLE_PENDING_THRESHOLD = 1.0

_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "benchmark_baselines.md",
)
_REGISTRY_CACHE: dict[str, list[dict]] | None = None


def _normalize_benchmark(name: str) -> str:
    """Loose key so 'GSM8K', 'gsm8k', and 'GSM-8K' collide."""
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _parse_registry(text: str) -> dict[str, list[dict]]:
    """Parse '### <Benchmark> (<task_type>)' sections and their markdown tables."""
    registry: dict[str, list[dict]] = {}
    current_key: str | None = None
    current_task: str | None = None
    for line in text.splitlines():
        heading = re.match(r"^###\s+(.+?)\s*\(([^)]+)\)\s*$", line)
        if heading:
            current_key = _normalize_benchmark(heading.group(1))
            current_task = heading.group(2).strip().split(",")[0].strip()
            registry.setdefault(current_key, [])
            continue
        if line.startswith("###"):
            current_key = current_task = None
            continue
        if current_key is None or not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            continue
        metric, value, model, params, source, checked = cells
        if metric in ("metric", "---") or set(metric) <= {"-", ":"}:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            # 'n/a' rows are deliberate: they record that no comparable figure was found.
            continue
        if not 0.0 < numeric <= 1.0:
            continue
        if not source or source == "—" or not checked or checked == "—":
            # The contract requires provenance; a row without it is ignored, not guessed at.
            continue
        registry[current_key].append({
            "benchmark_task_type": current_task,
            "metric": metric,
            "value": numeric,
            "model": model,
            "params": params,
            "source": source,
            "checked": checked,
        })
    return registry


def load_registry(force: bool = False) -> dict[str, list[dict]]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None or force:
        try:
            with open(_REGISTRY_PATH, encoding="utf-8") as handle:
                _REGISTRY_CACHE = _parse_registry(handle.read())
        except FileNotFoundError:
            _REGISTRY_CACHE = {}
    return _REGISTRY_CACHE


def registry_lookup(benchmark: str | None, task_type: str) -> dict | None:
    """Return the registry row whose metric matches what we measure, else None.

    A row whose metric differs from TASK_METRIC_NAMES[task_type] is INFORMATIONAL only —
    comparing a published `accuracy` to our `macro_f1` would be exactly the incomparable-metric
    error the pool already refuses to make.
    """
    if not benchmark:
        return None
    from eval.harness import TASK_METRIC_NAMES

    expected_metric = TASK_METRIC_NAMES.get(task_type)
    if not expected_metric:
        return None
    for row in load_registry().get(_normalize_benchmark(benchmark), []):
        if row["metric"] == expected_metric:
            return dict(row)
    return None


def snap_headroom(value) -> float:
    """Snap a requested headroom to the nearest allowed rung (lower wins ties)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_HEADROOM
    numeric = float(value)
    return min(VALID_HEADROOMS, key=lambda rung: (abs(rung - numeric), rung))


def threshold_from_registry(row: dict) -> tuple[float, str]:
    """Target just under a published figure, leaving room for a task-specific dataset.

    The 0.03 shortfall is the same intent the old prompt expressed as "SOTA - 2 to 5 points",
    but applied deterministically to a sourced number instead of a recalled one.
    """
    value = min(THRESHOLD_CEILING, max(0.0, float(row["value"]) - 0.03))
    reason = (
        f"registry: {row['metric']}={row['value']} on {row['model']} ({row['params']}), "
        f"checked {row['checked']}, minus 0.03 headroom for a task-specific dataset"
    )
    return round(value, 4), reason


def threshold_from_endpoint_baseline(
    measured: float, floor: float = 0.8
) -> tuple[float, str]:
    """Goal = the separately-hosted Qwen-3.6 base score on THIS run's eval set, floored.

    The accuracy target a run must beat is the strong reference model's own zero-shot
    performance on the identical frozen E, scored by the identical metric — so "good enough"
    means "matches the reference," not a recalled leaderboard number. Floored so a weak
    reference (or an unreachable endpoint measured as 0.0) cannot set a trivially-low goal,
    and capped at THRESHOLD_CEILING because label noise makes 1.0 unreachable.
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


def threshold_from_anchor(
    zero_shot: float | None,
    first_finetune: float | None,
    headroom: float,
) -> tuple[float, str]:
    """Late-bound target: max(zero_shot, first_finetune) + a bounded headroom.

    Using the MAX is the guard against anchoring on a bad first hyperparameter draw. The
    zero-shot baseline is config-independent and already measured at iteration 1, so a weak
    _DEFAULT_CONFIG cannot depress the anchor below what the base model already does.
    """
    candidates = [
        value for value in (zero_shot, first_finetune)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not candidates:
        raise ValueError(
            "threshold calibration needs at least one measured score "
            "(zero-shot baseline or first fine-tune)"
        )
    anchor = max(float(value) for value in candidates)
    snapped = snap_headroom(headroom)
    value = min(THRESHOLD_CEILING, anchor + snapped)
    which = (
        "zero-shot"
        if zero_shot is not None and anchor == float(zero_shot)
        else "first fine-tune"
    )
    reason = (
        f"measured anchor {anchor:.4f} ({which}; "
        f"zero_shot={_fmt(zero_shot)}, first_finetune={_fmt(first_finetune)}) "
        f"+ {snapped:.2f} headroom"
    )
    return round(value, 4), reason


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
