# agent/nodes/cold_start/model_selection/interpolation.py
"""
Strategy: Interpolation (3-Probe Scaling Curve)

Probe 3 variants (smallest, middle, largest from the feasible set), fit F1
against log deployment weight size, then select the variant
whose predicted size is closest to the hardware memory budget while still
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

from agent.state import AgentState
from config.android_pool import ModelSpec, resolve_model_selector
from eval.harness import run_eval
from training.slm_helpers import train as slm_train

logger = logging.getLogger(__name__)


def _plog(msg: str):
    """Print to stdout so the probe/fit/selection steps land in run.log. The model-selection
    nodes previously logged only via logger.info, which the run's logging config suppresses,
    so the 3-probe scaling curve was invisible and interpolation *looked* like it jumped
    straight to the largest model (B161)."""
    print(f"[model_selection:interpolation] {msg}")

# Probes must be REPRESENTATIVE of real fine-tuning, or they systematically underestimate
# fully-trained F1 → the fitted curve sits too low → the goal-crossing extrapolates to a huge
# param count → nothing qualifies → default-to-largest (the failure seen in earlier runs, B161).
# Fix: more epochs (3, like real training) on the actual acquired curriculum, not the eval seed.
_PROBE_EPOCHS = int(os.environ.get("SLM_PROBE_EPOCHS", "3"))
_PROBE_LORA_RANK = 16
_PROBE_LORA_ALPHA = 32
_PROBE_LORA_DROPOUT = 0.0
_PROBE_WEIGHT_DECAY = 0.01
_PROBE_LR = 2e-4
_PROBE_MICRO_BATCH = 8
_PROBE_GRADIENT_ACCUMULATION = 1
_PROBE_EFFECTIVE_BATCH = (
    _PROBE_MICRO_BATCH * _PROBE_GRADIENT_ACCUMULATION
)
_PROBE_MAX_EXAMPLES = int(os.environ.get("SLM_PROBE_MAX_EXAMPLES", "300"))  # cap probe cost


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


def _fit_scaling_curve(
    points: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """Fit F1 against log on-disk weight size, averaging duplicate footprints."""
    grouped: dict[float, list[float]] = {}
    for size_mb, score in points:
        grouped.setdefault(float(size_mb), []).append(float(score))
    if len(grouped) < 2:
        return None

    xs = [math.log(max(size, 1.0)) for size in sorted(grouped)]
    ys = [
        sum(grouped[size]) / len(grouped[size])
        for size in sorted(grouped)
    ]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 1e-12:
        return None
    slope = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    ) / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _probe_model(
    model: ModelSpec,
    state: AgentState,
    probe_dir: str,
) -> float:
    """Fine-tune a probe and score the selected deployment variant. Returns 0 on failure."""
    model_dir = os.path.join(
        probe_dir,
        model.selector.replace("/", "_").replace("@", "__"),
    )
    dataset_path = state.get("current_dataset_path")
    _seed_file = None
    if not dataset_path:
        # Prefer the ACQUIRED real curriculum (train_examples) for a representative probe;
        # fall back to the eval-set seed only if no train data is available. Stratify + cap
        # so the probe is balanced and cheap (B161).
        from data.loaders.web_acquire import _stratified_take
        seed_examples = list(state.get("train_examples") or [])
        src = "train_examples"
        if not seed_examples:
            eval_set = state.get("eval_set")
            if eval_set is None:
                logger.warning(
                    "[interpolation] No dataset or eval_set; skipping probe for %s", model.model_id
                )
                return 0.0
            seed_examples = list(eval_set.pos) + list(eval_set.neg) + list(eval_set.boundary)
            src = "eval_seed"
        if not seed_examples:
            return 0.0
        seed_examples = _stratified_take(seed_examples, _PROBE_MAX_EXAMPLES)
        _plog(f"    (probe data: {len(seed_examples)} examples from {src}, {_PROBE_EPOCHS} epochs)")
        _seed_fd, _seed_file = tempfile.mkstemp(suffix=".jsonl", prefix="slm_probe_seed_")
        with os.fdopen(_seed_fd, "w") as f:
            for ex in seed_examples:
                f.write(json.dumps(ex) + "\n")
        dataset_path = _seed_file
    try:
        weights_ref = slm_train(
            dataset_path=dataset_path,
            base_model=model.model_id,
            nr_epochs=_PROBE_EPOCHS,
            learning_rate=_PROBE_LR,
            lora_rank=_PROBE_LORA_RANK,
            lora_alpha=_PROBE_LORA_ALPHA,
            lora_dropout=_PROBE_LORA_DROPOUT,
            weight_decay=_PROBE_WEIGHT_DECAY,
            micro_batch_size=_PROBE_MICRO_BATCH,
            gradient_accumulation_steps=(
                _PROBE_GRADIENT_ACCUMULATION
            ),
            effective_batch_size=_PROBE_EFFECTIVE_BATCH,
            output_dir=model_dir,
            task_type=state["task_type"],
        ).weights_ref
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
            state["eval_set"],
            weights_ref,
            model.model_id,
            state["task_type"],
            quant=model.quant,
            gguf_path=gguf_path,
        )
        logger.info("[interpolation] Probe %s -> f1=%.4f", model.model_id, result.f1)
        return result.f1
    except Exception as exc:
        _plog(f"  ⚠ probe FAILED for {model.model_id} ({str(exc)[:120]}) → f1=0.0")
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
        match = resolve_model_selector(feasible, forced)
        if match is None:
            raise RuntimeError(f"SLM_FORCE_MODEL={forced!r} is not in the feasible pool.")
        state["selected_model"] = match
        logger.info("[model_selection:interpolation] SLM_FORCE_MODEL=%s pinned", forced)
        return state

    stop_threshold = state.get("stop_threshold", 0.96)
    ram_target = state["hardware_constraints"].memory_mb
    candidates = _pick_candidates(feasible)

    _plog(f"3-probe scaling curve — probing {len(candidates)} candidates "
          f"(smallest/middle/largest of {len(feasible)} feasible): "
          f"{[m.model_id + ' [' + (m.quant or 'bf16') + ']' for m in candidates]}  "
          f"(RAM target={ram_target}MB, threshold={stop_threshold:.3f})")
    logger.info(
        "[interpolation] Probing %d candidates: %s (RAM target=%dMB)",
        len(candidates), [m.model_id for m in candidates], ram_target,
    )

    with tempfile.TemporaryDirectory(prefix="slm_probe_") as probe_dir:
        points: list[tuple[float, float]] = []
        for i, model in enumerate(candidates, 1):
            _plog(f"  probe {i}/{len(candidates)}: fine-tuning {model.model_id} "
                  f"[{model.quant or 'bf16'}] (~{model.est_params_b():.2f}B, "
                  f"{_PROBE_EPOCHS} epochs)…")
            f1 = _probe_model(model, state, probe_dir)
            points.append((model.size_mb, f1))
            _plog(f"  probe {i}/{len(candidates)} result: {model.model_id} "
                  f"[{model.quant or 'bf16'}] → f1={f1:.4f}  "
                  f"(size={model.size_mb}MB)")

    fit = _fit_scaling_curve(points)
    if fit is None:
        chosen = min(feasible, key=lambda m: abs(m.size_mb - ram_target))
        _plog("Too few distinct deployment footprints for a stable curve; selecting closest to RAM "
              f"target: {chosen.model_id} [{chosen.quant or 'bf16'}]")
        state["selected_model"] = chosen
        return state

    a, b = fit
    _plog(f"Fitted deployment curve from {len(points)} probes: "
          f"f1 = {a:.4f}·log(size_MB) + {b:.4f}")

    # Draw the curve at each distinct feasible deployment footprint, so the shape and
    # where it sits relative to the goal is visible in the log (B161).
    _seen_sz = set()
    _plog(f"  curve (predicted f1 by on-disk size, goal={stop_threshold:.3f}):")
    for model in sorted(feasible, key=lambda m: m.size_mb):
        size = model.size_mb
        if size in _seen_sz:
            continue
        _seen_sz.add(size)
        pred = a * math.log(max(size, 1.0)) + b
        mark = "≥goal" if pred >= stop_threshold else "<goal"
        _plog(f"    size={size:>5}MB  predicted f1={pred:.4f}  [{mark}]")

    # Solve the curve for the on-disk footprint where it crosses the accuracy goal.
    if a > 1e-6:
        ram_at_goal = math.exp((stop_threshold - b) / a)
        closest = min(feasible, key=lambda m: abs(m.size_mb - ram_at_goal))
        _plog(f"  curve intersects the accuracy goal ({stop_threshold:.3f}) at "
              f"~{ram_at_goal:.0f}MB on-disk → nearest feasible variant "
              f"{closest.model_id} [{closest.quant or 'bf16'}] at size≈{closest.size_mb}MB RAM")
        if ram_at_goal > max(m.size_mb for m in feasible):
            _plog("  (intersection is ABOVE the largest feasible model — goal likely "
                  "unreachable within the RAM budget on these probes)")
        elif ram_at_goal < min(m.size_mb for m in feasible):
            _plog("  (intersection is BELOW the smallest feasible model — even the smallest "
                  "should clear the goal)")
    else:
        _plog("  curve slope ≤ 0 (accuracy did not increase with size across the probes) — "
              "no meaningful size↔goal intersection; probes may be noisy/underpowered")

    qualifiers = []
    for model in feasible:
        predicted = a * math.log(max(model.size_mb, 1.0)) + b
        if predicted >= stop_threshold:
            qualifiers.append(model)

    if qualifiers:
        chosen = min(qualifiers, key=lambda m: abs(m.size_mb - ram_target))
        _plog(f"{len(qualifiers)} model(s) predicted ≥ {stop_threshold:.3f}; selected the "
              f"one closest to the RAM target ({ram_target}MB): {chosen.model_id} "
              f"[{chosen.quant or 'bf16'}] (size={chosen.size_mb}MB, "
              f"predicted f1={a * math.log(max(chosen.size_mb, 1.0)) + b:.4f})")
    else:
        # No model predicted to clear the goal. Rather than blindly defaulting to the LARGEST
        # (which underpowered probes made happen every time before, B161), pick the model with
        # the HIGHEST PREDICTED f1 on the fitted curve — for a positive slope that's the
        # largest, but for a flat/negative slope (noisy probes) it avoids over-selecting.
        # Downward re-exploration (post-convergence) can still trim to a smaller model.
        def _pred(m):
            return a * math.log(max(m.size_mb, 1.0)) + b
        chosen = max(feasible, key=lambda m: (round(_pred(m), 4), -m.size_mb))
        _plog(f"NO model predicted to meet threshold {stop_threshold:.3f} from the curve "
              f"(probes underperformed) — selecting highest PREDICTED f1: "
              f"{chosen.model_id} [{chosen.quant or 'bf16'}] (predicted {_pred(chosen):.4f}, "
              f"size={chosen.size_mb}MB)")

    state["selected_model"] = chosen
    return state
