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
from eval.harness import EvalResult, eval_output_token_reserve

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = 16


class BaselineGenerationError(RuntimeError):
    """Generation failed on so many rows that the resulting score is not a measurement.

    Tolerating individual failures is right — one truncated response must not abort a 1,000-row
    baseline. Tolerating ALL of them is not: the score then describes the harness rather than the
    model, and it is indistinguishable from a genuinely incapable teacher. On run 38661753 every one
    of 1,000 rows failed with `'str' object is not callable`, because the task name had been passed
    into the `generate_fn` slot; the baseline read 0.0000, and the accuracy goal was quietly floored
    at 0.80 on the strength of it while reporting that the teacher had scored zero (B313).
    """


# Above this share of failed generations the result is discarded rather than scored. Not zero: a few
# refusals or truncations are normal and the surviving rows still measure something. Well under half,
# because a teacher that cannot answer most of the eval set is not being measured either.
MAX_GENERATION_FAILURE_RATE = 0.25


def _generate_all(
    generate_fn, prompts: list[str], max_tokens: int, max_workers: int
) -> list[str]:
    """Run generate_fn over every prompt, order-preserving.

    A per-call failure yields '' so a single bad row cannot abort the whole baseline, but failures are
    COUNTED and a score is refused outright once too many of them fail — see
    `BaselineGenerationError`.
    """
    failures: list[str] = []

    def _one(prompt: str) -> str:
        try:
            return generate_fn(prompt, temperature=0.0, max_tokens=max_tokens)
        except Exception as error:  # noqa: BLE001 - one row must not kill the measurement
            failures.append(f"{type(error).__name__}: {error}")
            logger.warning("endpoint baseline generation failed: %s", str(error)[:120])
            return ""

    workers = max(1, min(max_workers, len(prompts) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(_one, prompts))
    if prompts and len(failures) / len(prompts) > MAX_GENERATION_FAILURE_RATE:
        from collections import Counter

        common = Counter(failures).most_common(3)
        raise BaselineGenerationError(
            f"generation failed on {len(failures)} of {len(prompts)} eval row(s) "
            f"({len(failures) / len(prompts):.0%}), so the resulting score would describe the "
            f"harness rather than the model. Most common errors: "
            + "; ".join(f"{message} (x{count})" for message, count in common)
        )
    return outputs


def measure_endpoint_baseline(
    eval_set: EvalSet,
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
    if not callable(generate_fn):
        # Cheap, and it catches the whole family: `generate_fn` is the second POSITIONAL parameter, so
        # anything a caller passes there while believing it is passing a task lands here silently.
        raise TypeError(
            f"measure_endpoint_baseline needs a callable generate_fn, got "
            f"{type(generate_fn).__name__} ({generate_fn!r}). The task is read from the eval set, so "
            f"there is no task parameter to pass positionally."
        )

    # The SAME spec the live harness uses, so the reference number is directly comparable to what
    # the fine-tuned model will be judged on — previously these were two separate dispatch chains
    # that had to be kept in sync by hand.
    spec = eval_set.spec
    prompts = spec.build_prompts(eval_set)
    max_tokens = eval_output_token_reserve(spec.name)
    log(f"      [baseline] scoring reference model on {len(prompts)} eval rows "
        f"(task={spec.name}, max_new_tokens={max_tokens})")
    raw_outputs = _generate_all(generate_fn, prompts, max_tokens, max_workers)
    predictions = spec.extract_predictions(raw_outputs, eval_set)
    result = spec.score(eval_set, predictions)

    metric = result.get("metric", spec.metric_name)
    log(f"      [baseline] reference {metric}={result['f1']:.4f} "
        f"format_valid={float(result.get('format_valid', 1.0)):.4f}")
    return EvalResult(
        f1=result["f1"],
        per_class=result["per_class"],
        failures=result["failures"],
        metric=metric,
        format_valid=float(result.get("format_valid", 1.0)),
    )
