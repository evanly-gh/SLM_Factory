"""
Test-data agent (B161 change 9).

Owns the held-out eval set and reports, to the orchestrator, ONLY per-difficulty accuracy
numbers plus a targeted diagnosis — never the raw examples (a contamination firewall). It
rather than clustering or exposing individual failures, it
runs the eval bucketed by difficulty (easy/medium/hard) and turns the per-bucket score
pattern into an actionable improvement suggestion (data / hyperparameters / escalate).

Difficulty is labeled by the base-model zero-shot capability gradient (B161): run the
SMALLEST and LARGEST feasible base models zero-shot once at setup —
  easy   = both get it right
  medium = only the large model gets it right
  hard   = neither gets it right
This captures the small→large capability gap that drives escalation. A text-length heuristic
is the fallback when zero-shot labeling is disabled or fails, so buckets always exist.
"""
import os
from collections import Counter


def _unique_base_endpoints(feasible_models):
    """Return smallest/largest unique base IDs for BF16 zero-shot capacity checks."""
    representatives = {}
    order = []
    for model in feasible_models:
        if model.model_id not in representatives:
            representatives[model.model_id] = model
            order.append(model.model_id)
        elif getattr(model, "quant", None) is None:
            # Prefer the BF16 object as documentation of base-model intent.
            representatives[model.model_id] = model

    unique = [representatives[model_id] for model_id in order]
    if not unique:
        return None, None
    if all(callable(getattr(model, "est_params_b", None)) for model in unique):
        ranked = sorted(
            unique,
            key=lambda model: (model.est_params_b(), model.model_id),
        )
        return ranked[0], ranked[-1]
    # Backward-compatible fallback: task_analysis supplies largest→smallest.
    return unique[-1], unique[0]


def label_difficulty(eval_set, feasible_models, task_type, log=print, correctness_fn=None):
    """Return {"easy":[texts], "medium":[texts], "hard":[texts]} for eval_set.all.

    `correctness_fn(model_id) -> dict[text->bool]` is injectable for testing; by default it
    runs the base model zero-shot via the eval harness. Falls back to a length-tercile
    heuristic if zero-shot labeling is disabled (SLM_DIFFICULTY=heuristic) or errors.
    """
    texts = [e.get("text", "") for e in eval_set.all]
    mode = os.environ.get("SLM_DIFFICULTY", "zeroshot")

    if mode == "zeroshot" and feasible_models:
        try:
            # Quant siblings are deployment variants, not distinct capacity endpoints.
            # Evaluate unique base model IDs through run_eval(model_id, model_id), which
            # intentionally loads the BF16/base artifact for this one-time gradient.
            smallest, largest = _unique_base_endpoints(feasible_models)
            if smallest is None or largest is None:
                raise ValueError("no unique base-model endpoints")
            cf = correctness_fn or _zeroshot_correctness(eval_set, task_type, log)
            small_ok = cf(smallest.model_id)
            large_ok = cf(largest.model_id)
            easy, medium, hard = [], [], []
            for t in texts:
                small_passed = small_ok.get(t, False)
                large_passed = large_ok.get(t, False)
                if small_passed and large_passed:
                    easy.append(t)
                elif large_passed:
                    medium.append(t)
                else:
                    hard.append(t)
            log(f"      [test_agent] difficulty (BF16 base-model gradient "
                f"{smallest.model_id} vs {largest.model_id}): "
                f"easy={len(easy)} medium={len(medium)} hard={len(hard)}")
            # Guard against a degenerate split (e.g. both models score 0) — fall back.
            if easy or medium or hard:
                return {"easy": easy, "medium": medium, "hard": hard}
        except Exception as e:  # noqa: BLE001
            log(f"      [test_agent] zero-shot difficulty labeling failed ({str(e)[:100]}); "
                f"using length heuristic")

    return _length_heuristic_buckets(texts, log)


def _zeroshot_correctness(eval_set, task_type, log):
    """Return correctness using the BF16 base model ID, not a quant sibling."""
    from eval.harness import run_eval

    def cf(model_id):
        res = run_eval(eval_set, model_id, model_id, task_type=task_type)
        failed = {f.get("text") for f in res.failures}
        return {e.get("text", ""): (e.get("text", "") not in failed) for e in eval_set.all}

    return cf


def _length_heuristic_buckets(texts, log=print):
    """Difficulty ≈ text length terciles (longer = harder). Deterministic fallback."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    n = len(texts)
    t1, t2 = n // 3, 2 * n // 3
    easy = [texts[i] for i in order[:t1]]
    medium = [texts[i] for i in order[t1:t2]]
    hard = [texts[i] for i in order[t2:]]
    log(f"      [test_agent] difficulty (length heuristic): easy={len(easy)} "
        f"medium={len(medium)} hard={len(hard)}")
    return {"easy": easy, "medium": medium, "hard": hard}


def score_by_difficulty(eval_set, correctness_by_text, difficulty) -> dict:
    """Per-bucket accuracy = fraction correct among that bucket's examples.
    `correctness_by_text` maps eval text -> bool (correct)."""
    out = {}
    for bucket in ("easy", "medium", "hard"):
        items = difficulty.get(bucket, []) if difficulty else []
        if not items:
            out[bucket] = {"n": 0, "accuracy": None}
            continue
        correct = sum(1 for t in items if correctness_by_text.get(t, False))
        out[bucket] = {"n": len(items), "accuracy": correct / len(items)}
    return out


def diagnose(by_difficulty: dict, overall_f1: float, threshold: float, task_type: str) -> dict:
    """Turn the per-difficulty score PATTERN into a targeted improvement suggestion.

    The pattern is more actionable than a flat score:
      - low even on EASY            → data quality / label format / prompt problem (data_rebuild)
      - easy OK, MEDIUM/HARD weak   → capacity/optimization: tune hyperparameters, then escalate
      - all buckets strong ≥ goal   → converged
    Returns {"suggested_intervention", "diagnosis", "band"}.
    """
    def acc(b):
        v = (by_difficulty.get(b) or {}).get("accuracy")
        return v if v is not None else None

    easy, med, hard = acc("easy"), acc("medium"), acc("hard")

    if overall_f1 >= threshold:
        return {"suggested_intervention": "none",
                "diagnosis": f"overall {overall_f1:.3f} ≥ goal {threshold:.3f} — converged.",
                "band": "converged"}

    # Failing even the easy bucket ⇒ the problem is the DATA/format, not model capacity.
    if easy is not None and easy < 0.6:
        return {"suggested_intervention": "data_rebuild",
                "diagnosis": (f"easy-bucket accuracy is low ({easy:.2f}) — the model misses even "
                              f"simple cases, indicating a data-quality / label-format / prompt "
                              f"problem rather than capacity. Rebuild + balance the data."),
                "band": "data"}

    # Easy solid but medium/hard weak ⇒ capacity/optimization limit.
    weak = [b for b, v in (("medium", med), ("hard", hard)) if v is not None and v < 0.6]
    if weak:
        return {"suggested_intervention": "hyperparameter",
                "diagnosis": (f"easy cases are handled but {'/'.join(weak)} cases are weak "
                              f"(medium={med}, hard={hard}) — an optimization/capacity gap. Tune "
                              f"hyperparameters (rank/lr/epochs); if it plateaus, escalate a tier."),
                "band": "optimization"}

    return {"suggested_intervention": "data_rebuild",
            "diagnosis": (f"below goal (overall {overall_f1:.3f}) with no single failing bucket — "
                          f"add more balanced data and continue."),
            "band": "general"}


def build_test_report(eval_set, best_result, difficulty, threshold, task_type) -> dict:
    """Assemble the report the orchestrator sees: overall + per-difficulty accuracy + diagnosis.
    Correctness per example is reconstructed from best_result.failures (text-matched)."""
    failed = {f.get("text") for f in (best_result.failures or [])}
    correctness = {e.get("text", ""): (e.get("text", "") not in failed) for e in eval_set.all}
    by_diff = score_by_difficulty(eval_set, correctness, difficulty or {})
    diag = diagnose(by_diff, best_result.f1, threshold, task_type)
    confusion: Counter = Counter()
    for failure in best_result.failures or []:
        if task_type == "classification":
            gold = str(failure.get("label", "?"))[:64]
            predicted = str(failure.get("predicted", "?"))[:64]
        elif task_type == "NER":
            gold_types = sorted({
                str(entity.get("type", "?"))[:64]
                for entity in failure.get("entities", [])
                if isinstance(entity, dict)
            })
            prediction = failure.get("predicted")
            prediction_entities = prediction if isinstance(prediction, list) else []
            predicted_types = sorted({
                str(entity.get("type", "?"))[:64]
                for entity in prediction_entities
                if isinstance(entity, dict)
            })
            gold = ",".join(gold_types) or "no_entity"
            predicted = ",".join(predicted_types) or "incorrect_entity_set"
        else:
            # Open-ended targets/predictions may contain the raw held-out answer.
            # Report only an aggregate verifier category for these tasks.
            gold = str(
                failure.get("error_type")
                or failure.get("judge_category")
                or "gold_verifier"
            )[:64]
            predicted = "incorrect"
        confusion[(gold, predicted)] += 1
    confusion_pairs = [
        {"gold": gold, "predicted": predicted, "count": count}
        for (gold, predicted), count in sorted(
            confusion.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )[:8]
    ]
    return {
        "overall": best_result.f1,
        "by_difficulty": by_diff,
        "confusion_pairs": confusion_pairs,
        "diagnosis": diag["diagnosis"],
        "suggested_intervention": diag["suggested_intervention"],
        "band": diag["band"],
    }
