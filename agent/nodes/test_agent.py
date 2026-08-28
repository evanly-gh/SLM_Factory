"""
Test-data agent (B161 change 9).

Owns the held-out eval set and reports, to the orchestrator, ONLY per-difficulty accuracy
numbers plus a targeted diagnosis — never the raw examples (a contamination firewall).
Rather than clustering or exposing individual failures, it runs the eval bucketed by
difficulty (easy/medium/hard) and turns the per-bucket score pattern into an actionable
improvement suggestion (data / hyperparameters / escalate).

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

# Ceiling on the zero-shot difficulty gradient, in generated tokens PER probe model.
#
# WHY THIS EXISTS (measured 2026-08-24, toolbench jobs 38818333/38818334)
#     `label_difficulty` runs two FULL generative evals over the whole eval set — the smallest and
#     largest feasible base models, both in BF16 — before the run has selected a model or trained
#     anything. On every task that existed when it was written that was cheap. On `toolbench` it is
#     not: 760 rows x 1,536 output tokens is 1.17M tokens per model against prompts averaging ~1,250
#     tokens, and the largest feasible model is Qwen3.5-4B at BF16. The smallest model alone took
#     70 minutes; the 4B pass would not have finished inside the 20-hour allocation, so both runs
#     spent their entire budget on a REPORTING aid and never reached a training step.
#
# The bound is generated tokens rather than wall time because it is knowable before any work starts.
# 1,000,000 leaves every other task in the registry untouched — the next highest is 512,000
# (dialogsum, gsm8k, ner_bc5cdr) — while catching toolbench's 1,536,000. It under-counts the true
# cost, since it ignores prefill and toolbench's prompts are several times longer than any other
# task's, so a task that trips this is comfortably over rather than marginally so.
#
# Falling back is cheap in consequence: the difficulty buckets feed the test agent's per-difficulty
# REPORT. They do not affect the curriculum, the training loop, the score, or any routing decision.
MAX_DIFFICULTY_PROBE_TOKENS = 1_000_000


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


def _probe_is_affordable(rows: int, task, log=print) -> bool:
    """Whether the zero-shot difficulty gradient is worth what it costs on this task.

    Two full generative passes over the eval set, before model selection, on a REPORTING aid. See
    `MAX_DIFFICULTY_PROBE_TOKENS` for the measurement that motivated the bound. Any failure to
    resolve the task's budget answers "affordable", so an unknown task keeps the previous behaviour
    rather than silently losing its difficulty labels.
    """
    try:
        from tasks import get_task

        budget = int(rows) * int(get_task(task).max_new_tokens)
    except Exception:  # noqa: BLE001 — an unresolvable budget must not disable the probe
        return True
    if budget <= MAX_DIFFICULTY_PROBE_TOKENS:
        return True
    log(
        f"      [test_agent] SKIPPING the zero-shot difficulty gradient for {task}: it would "
        f"generate {budget:,} tokens per probe model ({rows} rows x "
        f"{get_task(task).max_new_tokens} output tokens) against a "
        f"{MAX_DIFFICULTY_PROBE_TOKENS:,} ceiling, twice, in BF16, before this run selects a model "
        f"or trains anything. Difficulty buckets only feed the per-difficulty REPORT, so the length "
        f"heuristic is used instead. Set SLM_DIFFICULTY=zeroshot-force to override."
    )
    return False


def label_difficulty(eval_set, feasible_models, task, log=print, correctness_fn=None):
    """Return {"easy":[texts], "medium":[texts], "hard":[texts]} for eval_set.all.

    `correctness_fn(model_id) -> dict[text->bool]` is injectable for testing; by default it
    runs the base model zero-shot via the eval harness. Falls back to a length-tercile
    heuristic if zero-shot labeling is disabled (SLM_DIFFICULTY=heuristic) or errors.
    """
    texts = [e.get("text", "") for e in eval_set.all]
    mode = os.environ.get("SLM_DIFFICULTY", "zeroshot")

    # `zeroshot-force` runs the gradient whatever it costs, for the case where the per-difficulty
    # breakdown is the point of the run. Plain `zeroshot` is the default and is cost-bounded.
    if mode == "zeroshot-force":
        mode = "zeroshot"
    elif mode == "zeroshot" and not _probe_is_affordable(len(texts), task, log=log):
        mode = "heuristic"

    if mode == "zeroshot" and feasible_models:
        try:
            # Quant siblings are deployment variants, not distinct capacity endpoints.
            # Evaluate unique base model IDs through run_eval(model_id, model_id), which
            # intentionally loads the BF16/base artifact for this one-time gradient.
            smallest, largest = _unique_base_endpoints(feasible_models)
            if smallest is None or largest is None:
                raise ValueError("no unique base-model endpoints")
            cf = correctness_fn or _zeroshot_correctness(eval_set, task, log)
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


def _zeroshot_correctness(eval_set, task, log):
    """Return correctness using the BF16 base model ID, not a quant sibling."""
    from eval.harness import run_eval

    def cf(model_id):
        res = run_eval(eval_set, model_id, model_id)
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


def diagnose(by_difficulty: dict, overall_f1: float, threshold: float, task: str) -> dict:
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


def _outcome_breakdown(eval_set, best_result, spec) -> list[dict]:
    """Eval rows split into correct / failed, bucketed by whatever is informative for this task.

    Difficulty answers "how hard were the rows it got wrong". This answers "WHICH rows", which is
    the question an intervention is chosen against — `surgical_synthesis` targets exactly these
    buckets, so this is the record of whether targeting worked.

    Two bucketings, because the informative unit differs:

      closed label space  → the gold CLASS. A class at 0.50 on four rows and one at 0.50 on four
                            hundred are the same F1 and completely different problems, so both
                            halves of the count are kept.
      open-ended target   → the FAILURE CATEGORY, with every correct row in one `correct` bucket.
                            There is no class to break down by; what varies is the kind of error.
    """
    failures = list(best_result.failures or [])
    rows = list(getattr(eval_set, "all", []) or [])
    buckets: dict[str, dict[str, int]] = {}

    def _slot(name: str) -> dict[str, int]:
        return buckets.setdefault(str(name)[:64], {"correct": 0, "failed": 0})

    if spec.closed_label_space:
        failed_texts = Counter(str(f.get("text", "")) for f in failures)
        for row in rows:
            slot = _slot(row.get("label", "?"))
            key = str(row.get("text", ""))
            if failed_texts.get(key):
                slot["failed"] += 1
                failed_texts[key] -= 1
            else:
                slot["correct"] += 1
    else:
        _slot("correct")["correct"] = max(0, len(rows) - len(failures))
        for failure in failures:
            category = "uncategorised"
            if spec.failure_category is not None:
                try:
                    category = str(spec.failure_category(failure))
                except Exception:  # noqa: BLE001 — a diagnostic must never break the report
                    category = "uncategorised"
            _slot(category)["failed"] += 1

    return [
        {"bucket": name, "correct": counts["correct"], "failed": counts["failed"]}
        for name, counts in sorted(
            buckets.items(),
            key=lambda item: (-item[1]["failed"], -item[1]["correct"], item[0]),
        )
    ]


def build_test_report(eval_set, best_result, difficulty, threshold, task) -> dict:
    """Assemble the report the orchestrator sees: overall + per-difficulty accuracy + diagnosis.
    Correctness per example is reconstructed from best_result.failures (text-matched)."""
    failed = {f.get("text") for f in (best_result.failures or [])}
    correctness = {e.get("text", ""): (e.get("text", "") not in failed) for e in eval_set.all}
    by_diff = score_by_difficulty(eval_set, correctness, difficulty or {})
    diag = diagnose(by_diff, best_result.f1, threshold, task)
    # The task's own failure taxonomy. Under the previous design this was an `if task_type == ...`
    # chain whose `else` reported the single constant pair `gold_verifier -> incorrect` for every
    # open-ended task, so its count was just the failure count the orchestrator already had — and
    # it wrote pages of reasoning about that constant (B296). A task that declares
    # `failure_category=None` reports one honest aggregate instead of a fake taxonomy.
    from tasks import get_task

    spec = get_task(task)
    confusion: Counter = Counter()
    for failure in best_result.failures or []:
        if spec.failure_category is None:
            confusion[("failed", "incorrect")] += 1
            continue
        try:
            category = str(spec.failure_category(failure))[:64]
        except Exception:  # noqa: BLE001 — a diagnostic must never break the report
            category = "uncategorised"
        # Classification keeps the true class on the left so the pair still names a real
        # confusion; everything else names the error kind, which is what is actionable.
        if spec.closed_label_space:
            confusion[(str(failure.get("label", "?"))[:64], str(failure.get("predicted", "?"))[:64])] += 1
        else:
            # No `predicted` side. This used to be the literal string "incorrect", a placeholder to
            # fill the pair shape — and the orchestrator read it as a real model output. On calendar
            # run 38735780 it wrote about "the degenerate single-token 'incorrect' output" 65 TIMES,
            # building hypotheses on a behaviour that never occurred: the model was emitting `[]` and
            # well-formed calls with wrong arguments, never the word "incorrect" (B326).
            #
            # An open-ended task has a failure CATEGORY, not a confusion between two classes. None
            # says so, and the renderers below print a category line instead of a pair.
            confusion[(category, None)] += 1
    confusion_pairs = [
        {"gold": gold, "predicted": predicted, "count": count}
        for (gold, predicted), count in sorted(
            confusion.items(),
            # `predicted` may be None for an open-ended task, so sort on its string form rather than
            # comparing None with str.
            key=lambda item: (-item[1], item[0][0], str(item[0][1])),
        )[:8]
    ]
    return {
        "overall": best_result.f1,
        "by_difficulty": by_diff,
        "confusion_pairs": confusion_pairs,
        "outcome_breakdown": _outcome_breakdown(eval_set, best_result, spec),
        "diagnosis": diag["diagnosis"],
        "suggested_intervention": diag["suggested_intervention"],
        "band": diag["band"],
    }
