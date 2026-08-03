"""Measure a hosted reference model's zero-shot score on THIS run's frozen eval set.

Used to calibrate the accuracy goal to the separately-hosted Qwen-3.6 (config.SYNTH_ENDPOINT):
the target a run must beat is the reference model's OWN performance on the identical E, scored
by the identical task scorer — not a recalled leaderboard number. See
agent.threshold.threshold_from_endpoint_baseline for how the score becomes the goal.

The endpoint is a LOCAL vLLM OpenAI-compatible server reached via data.synth_client. When it is
unreachable (e.g. a dev box with SLM_SYNTH_ENDPOINT unset), get_generate_fn returns None and
the caller degrades honestly rather than crashing.
"""
import logging
from concurrent.futures import ThreadPoolExecutor

from data.eval_set import EvalSet
from eval.harness import EvalResult, TASK_METRIC_NAMES, eval_output_token_reserve

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = 16


def _scorer_for(task_type: str):
    """Resolve the same scorer module the live harness dispatches to."""
    if task_type == "classification":
        from eval.scorers import classification as scorer
    elif task_type == "NER":
        from eval.scorers import ner as scorer
    elif task_type in ("math_reasoning", "code_generation", "generation"):
        from eval.scorers import generation as scorer
    elif task_type == "function_call":
        from eval.scorers import function_call as scorer
    elif task_type == "diff":
        from eval.scorers import diff as scorer
    else:
        raise ValueError(f"endpoint baseline: unknown task_type {task_type!r}")
    return scorer


def _generate_all(
    generate_fn, prompts: list[str], max_tokens: int, max_workers: int
) -> list[str]:
    """Run generate_fn over every prompt, order-preserving. A per-call failure yields '' so a
    single bad row cannot abort the whole baseline."""
    def _one(prompt: str) -> str:
        try:
            return generate_fn(prompt, temperature=0.0, max_tokens=max_tokens)
        except Exception as error:  # noqa: BLE001 - one row must not kill the measurement
            logger.warning("endpoint baseline generation failed: %s", str(error)[:120])
            return ""

    workers = max(1, min(max_workers, len(prompts) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, prompts))


def measure_endpoint_baseline(
    eval_set: EvalSet,
    task_type: str,
    generate_fn=None,
    *,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    log=print,
) -> EvalResult | None:
    """Score the hosted reference model zero-shot on E, or None if no endpoint is reachable.

    generate_fn is injectable for testing; when omitted it is resolved from
    data.synth_client.get_generate_fn() (returns None when the endpoint is unavailable).
    The eval set is scored by the task's own scorer so the number is directly comparable to
    what the fine-tuned SLM will be judged on.
    """
    if generate_fn is None:
        from data.synth_client import get_generate_fn
        generate_fn = get_generate_fn(log=log)
    if generate_fn is None:
        log("      [baseline] reference endpoint unavailable — no zero-shot baseline measured")
        return None

    scorer = _scorer_for(task_type)
    prompts = scorer.build_prompts(eval_set)
    max_tokens = eval_output_token_reserve(task_type)
    log(f"      [baseline] scoring reference model on {len(prompts)} eval rows "
        f"(task_type={task_type}, max_new_tokens={max_tokens})")
    raw_outputs = _generate_all(generate_fn, prompts, max_tokens, max_workers)
    predictions = scorer.extract_predictions(raw_outputs, eval_set)
    result = scorer.score(eval_set, predictions)

    metric = result.get("metric", TASK_METRIC_NAMES.get(task_type, "f1"))
    log(f"      [baseline] reference {metric}={result['f1']:.4f}")
    return EvalResult(
        f1=result["f1"],
        per_class=result["per_class"],
        failures=result["failures"],
        execution_diagnostics=result.get("execution_diagnostics", []),
        metric=metric,
    )
