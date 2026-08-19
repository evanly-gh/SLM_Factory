# eval/harness.py
import os
from dataclasses import dataclass
from data.eval_set import EvalSet
from training.slm_helpers import infer_batch, infer_batch_gguf


_OVERRIDE_MAX_NEW_TOKENS = "SLM_EVAL_MAX_NEW_TOKENS"


def eval_output_token_reserve(task: str) -> int:
    """The task's output-token reserve, validated against its own context window.

    Both numbers come from the task's spec, which validated their relationship at import time; this
    only re-checks after an environment override. Previously the reserve was a dict keyed by
    task_type with `.get(..., 4096)` on the context side, so an unrecognised type silently received
    a generous default rather than an error.
    """
    from tasks import get_task

    spec = get_task(task)
    raw_value = os.environ.get(_OVERRIDE_MAX_NEW_TOKENS)
    if raw_value is None:
        return spec.max_new_tokens
    try:
        reserve = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_OVERRIDE_MAX_NEW_TOKENS} must be a positive integer, got {raw_value!r}."
        ) from exc
    if reserve < 1:
        raise ValueError(
            f"{_OVERRIDE_MAX_NEW_TOKENS} must be a positive integer, got {raw_value!r}."
        )
    if reserve >= spec.max_seq_length:
        raise ValueError(
            f"task={task} reserves {reserve} output tokens, leaving no prompt budget inside its "
            f"{spec.max_seq_length}-token context. Lower {_OVERRIDE_MAX_NEW_TOKENS} or raise the "
            "task's max_seq_length."
        )
    return reserve


def task_metric_name(task: str) -> str:
    """What the comparison scalar measures for this task.

    Per TASK, not per channel: RouterBench and CLINC150 were both `classification` and both
    reported `macro_f1`, but RouterBench's headline number has always been a minority-class F1.
    """
    from tasks import get_task

    return get_task(task).metric_name


@dataclass
class EvalResult:
    # `f1` is the pipeline's universal comparison scalar: it drives best_score, the
    # stagnation window, rollback, and every DAG node. Its NAME is a historical artifact —
    # only classification and NER actually compute an F1. `metric` records what the number
    # really is so reports and papers cannot misattribute it. Never rename `f1` itself;
    # checkpoints and DAG replay depend on the field name.
    f1: float
    per_class: dict
    failures: list[dict]
    metric: str = "f1"
    # What fraction of predictions were even READABLE by the scorer, separately from whether they
    # were right. Every task has a parse step — a JSON call list, a JSON span array, an
    # in-vocabulary label, a final number — so a low score means two very different things depending
    # on this number: a content problem the data can fix, or a format problem the prompt or the chat
    # template can. Reported for all tasks rather than only function-calling, because the run that
    # made this necessary (B290) was diagnosed entirely from the gap between the two.
    format_valid: float = 1.0


_PREDICTION_SAMPLE_N = int(os.environ.get("SLM_EVAL_SAMPLE_LOG_N", "3"))


def _attach_reasoning_to_failures(result, eval_set, raw_outputs, predictions) -> None:
    """Record the chain-of-thought that produced each failed answer.

    A CoT-trained model emits `<reasoning>...</reasoning>` before its answer. Only the ANSWER is
    scored — handing a reasoning block to a judge asked "how good is this summary?" guarantees a
    poor score for output that may contain a perfectly good summary. But the reasoning is the
    part that explains WHY the answer is wrong, so it is kept here rather than discarded: a
    failure record with the answer alone cannot distinguish bad reasoning from bad phrasing.

    Keyed by eval-row index, which is stable — the scorers preserve row order and `predictions`
    is positional. Failures are matched on the row's own text, so this is a no-op for scorers
    that emit no reasoning (B251).
    """
    failures = (result or {}).get("failures") or []
    if not failures:
        return
    try:
        from eval.scorers.generation import split_reasoning
    except Exception:  # noqa: BLE001 — diagnostics must never break scoring
        return
    reasoning_by_answer: dict[str, str] = {}
    for raw, prediction in zip(raw_outputs, predictions):
        reasoning, _answer = split_reasoning(raw)
        if reasoning:
            reasoning_by_answer.setdefault(str(prediction), reasoning)
    if not reasoning_by_answer:
        return
    for failure in failures:
        if isinstance(failure, dict):
            reasoning = reasoning_by_answer.get(str(failure.get("predicted")))
            if reasoning:
                failure["reasoning"] = reasoning


def _gold_for_display(row: dict):
    """The gold answer for a row, whatever field this task type keeps it in.

    NER rows carry their gold in `entities` and have no `answer`/`label` at all, so reading only
    those two printed a BLANK gold on every NER sample — in the baseline block and the fine-tuned
    block alike. That removed the one human check of gold against prediction from exactly the task
    where it matters most: a legitimate `Baseline F1 = 0.0000` (the base model emitting ```json []```
    on every row) was indistinguishable from a broken harness, because there was nothing to compare
    the prediction to. `diff` rows keep gold in `answer`, which already worked, and `src`/`tgt` are
    shown as a fallback for a row that somehow lacks the precomputed diff.
    """
    entities = row.get("entities")
    if entities is not None:
        # Rendered the way the NER scorer compares them — exact (surface, type) pairs — so the
        # display matches what is actually being matched, not a prettier paraphrase of it.
        if isinstance(entities, list):
            return "[" + ", ".join(
                f"{e.get('text')!r}:{e.get('type')}" if isinstance(e, dict) else repr(e)
                for e in entities
            ) + "]"
        return entities
    gold = row.get("answer") or row.get("label")
    if gold:
        return gold
    if row.get("tgt") is not None:
        return f"src={row.get('src')!r} → tgt={row.get('tgt')!r}"
    return gold


def _log_prediction_samples(eval_set, raw_outputs, predictions) -> None:
    """Print a few raw model answers next to the extracted prediction and the gold label.

    Scores alone cannot distinguish "the model picked the wrong class" from "the model answered
    in a format the extractor could not read". Showing the RAW output beside the extracted value
    makes that difference visible — an `__EXTRACTION_FAILED__` next to a chatty answer is a
    prompt/format problem, while a clean wrong label is a real accuracy problem.

    Prefers examples that failed extraction, since those are the diagnostic ones.
    """
    if _PREDICTION_SAMPLE_N <= 0:
        return
    rows = getattr(eval_set, "all", None) or []
    n = min(len(rows), len(raw_outputs), len(predictions))
    if n == 0:
        return
    failed = [i for i in range(n) if str(predictions[i]) == "__EXTRACTION_FAILED__"]
    chosen = (failed + [i for i in range(n) if i not in set(failed)])[:_PREDICTION_SAMPLE_N]

    def _clip(value, limit=110):
        text = " ".join(str(value or "").split())
        return text[:limit] + ("…" if len(text) > limit else "")

    print(
        f"      [eval] sample predictions ({len(chosen)} of {n}"
        + (f"; {len(failed)} extraction failure(s) this eval" if failed else "")
        + "):"
    )
    for i in chosen:
        gold = _gold_for_display(rows[i])
        flag = "  <-- EXTRACTION FAILED" if str(predictions[i]) == "__EXTRACTION_FAILED__" else ""
        print(f"        input : {_clip(rows[i].get('text'))}")
        print(f"        gold  : {_clip(gold, 60)}")
        print(f"        raw   : {_clip(raw_outputs[i])}")
        print(f"        parsed: {_clip(predictions[i], 60)}{flag}")


def run_eval(
    eval_set: EvalSet,
    weights_ref: str,
    base_model: str,
    quant: str | None = None,
    gguf_path: str | None = None,
) -> EvalResult:
    """Run one complete evaluation, isolated in a disposable process when enabled.

    The task comes from the eval set rather than a separate argument: they can no longer be passed
    inconsistently, and the eval set is the thing that knows which rows these are.
    """
    from training.cuda_isolation import isolation_enabled, run_isolated

    if isolation_enabled():
        from training.slm_helpers import clear_inference_cache

        payload = {
            "eval_set": eval_set,
            "weights_ref": weights_ref,
            "base_model": base_model,
            "quant": quant,
            "gguf_path": gguf_path,
        }
        clear_inference_cache()
        try:
            return run_isolated("eval", payload)
        finally:
            clear_inference_cache()

    return _run_eval_local(
        eval_set, weights_ref, base_model, quant=quant, gguf_path=gguf_path,
    )


def _judge_overlap_chunk() -> int:
    """Rows per generate-then-judge chunk. 0 disables the overlap entirely."""
    try:
        return max(0, int(os.environ.get("SLM_EVAL_JUDGE_OVERLAP_CHUNK", "100")))
    except (TypeError, ValueError):
        return 100


def _infer_overlapping_judge(prompts, eval_set, infer) -> list[str]:
    """Generate in chunks, judging each finished chunk while the next one generates.

    Generation runs on the pipeline GPU and the judge on the synthesis GPU, but the two were
    strictly sequential: all 800 predictions were produced, and only then were all 800 judged.
    Measured overlap between the two devices across run 38303490 was 0.0%, with the judge alone
    accounting for 22% of wall time (B258).

    The judge is not called differently here — a background thread simply asks
    ``LocalJudgeClient`` to score the chunk, which populates its process-wide and on-disk caches.
    The scorer's own ``score`` call afterwards is unchanged and finds those rows already cached,
    so ordering, scores and failure records are bit-identical to the sequential path. That is
    also why warm failures are swallowed: anything genuinely broken resurfaces in ``score``,
    which is the authority and raises there.
    """
    from concurrent.futures import ThreadPoolExecutor

    from eval.judge_client import LocalJudgeClient
    from eval.scorers.generation import split_reasoning

    chunk_size = _judge_overlap_chunk()
    rows = getattr(eval_set, "all", None) or []
    client = LocalJudgeClient.from_config()

    def warm(start: int, raw_chunk: list[str]) -> None:
        # Mirrors eval.scorers.generation.score exactly; a divergence here would miss the cache
        # rather than corrupt anything, but it would silently undo the speedup.
        triples = [
            (
                row.get("text", ""),
                row.get("answer", row.get("label", "")),
                split_reasoning(raw)[1],
            )
            for row, raw in zip(rows[start:start + len(raw_chunk)], raw_chunk)
        ]
        try:
            client.score_many(triples)
        except Exception:  # noqa: BLE001 — best-effort warm; score() is the authority
            pass

    raw_outputs: list[str] = []
    # One worker: the point is to overlap judging with the NEXT generation chunk, not to run
    # several judge batches at once against a single vLLM server.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = []
        for start in range(0, len(prompts), chunk_size):
            chunk_raw = infer(prompts[start:start + chunk_size])
            raw_outputs.extend(chunk_raw)
            pending.append(pool.submit(warm, start, chunk_raw))
        for future in pending:
            future.result()
    return raw_outputs


def _run_eval_local(
    eval_set: EvalSet,
    weights_ref: str,
    base_model: str,
    quant: str | None = None,
    gguf_path: str | None = None,
) -> EvalResult:
    """Run inference on the eval set and score it with the task's own scorer.

    When gguf_path is provided (quantized model path), inference uses llama-cpp-python
    (infer_batch_gguf) to get honest on-device accuracy. When gguf_path is None (base/BF16 model),
    uses Unsloth (infer_batch).

    There is no task dispatch here any more. Every choice — which prompt builder, which extractor,
    which scoring rule, how many output tokens, whether to overlap the judge, whether to record
    reasoning — is read off the task's spec, so a task cannot inherit another's behaviour by
    matching the same branch.
    """
    from tasks import get_task

    spec = get_task(eval_set.task)
    max_new_tokens = eval_output_token_reserve(spec.name)
    prompts = spec.build_prompts(eval_set)

    def _infer(chunk: list[str]) -> list[str]:
        if gguf_path is not None:
            return infer_batch_gguf(
                chunk,
                gguf_path,
                max_new_tokens=max_new_tokens,
                base_model=base_model,
            )
        return infer_batch(
            chunk,
            weights_ref,
            base_model,
            max_workers=20,
            max_new_tokens=max_new_tokens,
            task=spec.name,
        )

    if spec.judge_overlap and _judge_overlap_chunk() > 0:
        raw_outputs = _infer_overlapping_judge(prompts, eval_set, _infer)
    else:
        raw_outputs = _infer(prompts)

    predictions = spec.extract_predictions(raw_outputs, eval_set)
    _log_prediction_samples(eval_set, raw_outputs, predictions)
    result = spec.score(eval_set, predictions)
    if spec.attach_reasoning:
        _attach_reasoning_to_failures(result, eval_set, raw_outputs, predictions)

    return EvalResult(
        f1=result["f1"],
        per_class=result["per_class"],
        failures=result["failures"],
        metric=result.get("metric", spec.metric_name),
        format_valid=float(result.get("format_valid", 1.0)),
    )
