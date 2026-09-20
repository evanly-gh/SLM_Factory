# eval/harness.py
import os
from dataclasses import dataclass
from data.eval_set import EvalSet
from training.slm_helpers import infer_batch, infer_batch_gguf, infer_batch_mnn


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

    A REFERENCE-FREE task has no gold at all, and that is not a defect to paper over. ToolEval's pass
    rate asks a judge whether the model's answer addresses the QUERY; the ToolBench test queries ship
    without a reference solution, so `answer` is legitimately empty on every eval row. Printing a bare
    blank there reproduces the B263 failure for a different reason — the reader cannot tell "no
    reference exists" from "the gold went missing" — so the query the row is judged against is shown
    instead, labelled as what it is.
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
    query = row.get("query")
    if query:
        return f"(no reference; judged against the query) {query}"
    return gold


def _log_prediction_samples(eval_set, raw_outputs, predictions) -> None:
    """Print a few raw model answers next to the extracted prediction and the gold label.

    Scores alone cannot distinguish "the model picked the wrong class" from "the model answered
    in a format the extractor could not read". Showing the RAW output beside the extracted value
    makes that difference visible — an `__EXTRACTION_FAILED__` next to a chatty answer is a
    prompt/format problem, while a clean wrong label is a real accuracy problem.

    Prefers examples that failed extraction, since those are the diagnostic ones — and then rows
    the model got WRONG, which is the case this used to be blind to.

    WHY THE SECOND TIER EXISTS
        The ordering was `failures + everything else in index order`, so a run whose outputs all
        parse — the healthy case, and every format-bound task once fine-tuning takes hold — showed
        rows 0, 1 and 2 and nothing else. On gec_bea19 run 39881529 that meant the same three rows
        for all 58 eval rounds at `format_valid=1.0000`: 2 unique rows of diagnostic signal out of
        1,000, and never once a row that was merely SCORED wrong.

        That is backwards for the question these samples exist to answer. An extraction failure is
        visible in `format_valid` already; a correct-looking answer graded wrong is visible nowhere
        else, and it is the only symptom a scorer bug has. Preferring mismatches costs one string
        comparison and turns a fixed window into the rows worth reading.

    Deliberately compares rendered strings rather than calling the scorer. This runs inside a CUDA
    worker on the eval hot path, the scorer may shell out (ERRANT) or load a model (BERTScore), and
    a display helper must not be able to fail or stall an eval it was only meant to describe. An
    approximate match is the right instrument: it over-selects rows a lenient metric would forgive,
    which still lands on a row worth looking at.
    """
    if _PREDICTION_SAMPLE_N <= 0:
        return
    rows = getattr(eval_set, "all", None) or []
    n = min(len(rows), len(raw_outputs), len(predictions))
    if n == 0:
        return
    failed = [i for i in range(n) if str(predictions[i]) == "__EXTRACTION_FAILED__"]
    failed_set = set(failed)

    def _looks_wrong(index: int) -> bool:
        gold = _gold_for_display(rows[index])
        if gold is None:
            return False
        norm = lambda value: " ".join(str(value or "").split()).strip().lower()
        return norm(predictions[index]) != norm(gold)

    mismatched, matched = [], []
    for index in range(n):
        if index in failed_set:
            continue
        (mismatched if _looks_wrong(index) else matched).append(index)
    chosen = (failed + mismatched + matched)[:_PREDICTION_SAMPLE_N]

    # GOLD AND PARSED GET THE MOST ROOM, not the least. They used to be clipped to 60 characters
    # while `input` and `raw` got 110, which is backwards: gold-vs-parsed is the comparison these
    # samples exist to let a reader make, and a structured prediction is where the characters go.
    # On multiconer run 39881531 a two-entity row rendered as
    #   parsed: [{'text': 'china', 'type': 'HumanSettlement'}, {'text': 'ele…
    # so the entity actually in dispute was the one cut off. Measured while auditing that run: a
    # third of sampled rows were unusable because the field under examination was truncated away.
    def _clip(value, limit=110):
        text = " ".join(str(value or "").split())
        return text[:limit] + ("…" if len(text) > limit else "")

    # Wide enough for a handful of NER spans or a nested parse, which are the shapes that overflow.
    _STRUCTURED = 240

    print(
        f"      [eval] sample predictions ({len(chosen)} of {n}"
        + (f"; {len(failed)} extraction failure(s) this eval" if failed else "")
        + "):"
    )
    for i in chosen:
        gold = _gold_for_display(rows[i])
        flag = "  <-- EXTRACTION FAILED" if str(predictions[i]) == "__EXTRACTION_FAILED__" else ""
        print(f"        input : {_clip(rows[i].get('text'))}")
        print(f"        gold  : {_clip(gold, _STRUCTURED)}")
        print(f"        raw   : {_clip(raw_outputs[i], _STRUCTURED)}")
        print(f"        parsed: {_clip(predictions[i], _STRUCTURED)}{flag}")


SCORING_MODES = ("select", "report")


def resolve_scorer(spec, scoring: str):
    """The scorer and metric name for one scoring mode.

    A MODE STRING RATHER THAN A CALLABLE, deliberately. `run_eval` may execute inside a disposable
    CUDA worker, and the boundary is `pickle` — a bare function would either fail to pickle or,
    worse, pickle by qualified name and silently resolve to a different object in the child. A
    two-value enum crosses that boundary as a string and is re-resolved against the registry on
    the far side, so parent and child cannot disagree about which metric was computed.
    """
    if scoring not in SCORING_MODES:
        raise ValueError(f"scoring must be one of {SCORING_MODES}, got {scoring!r}")
    if scoring == "report":
        return spec.report_score, spec.report_metric_name
    return spec.score, spec.metric_name


def run_eval(
    eval_set: EvalSet,
    weights_ref: str,
    base_model: str,
    quant: str | None = None,
    quant_artifact: str | None = None,
    quant_backend: str | None = None,
    scoring: str = "select",
) -> EvalResult:
    """Run one complete evaluation, isolated in a disposable process when enabled.

    The task comes from the eval set rather than a separate argument: they can no longer be passed
    inconsistently, and the eval set is the thing that knows which rows these are.

    `quant_artifact` is the built quantized model to score — a `.gguf` FILE under the llama.cpp
    backend, an MNN model DIRECTORY under the MNN one — and `quant_backend` says which engine to
    load it with. They arrive as a pair because the path alone cannot say: both are just paths, and
    guessing from the extension is the kind of inference that scores an MNN artifact through
    llama.cpp and reports the result under the wrong runtime's name. `None` means score the
    unquantized HF/LoRA weights through Unsloth instead.

    `scoring` picks which of the task's two scorers grades the run. It defaults to `"select"`
    because that is what the agent loop wants on every iteration, and because a default of
    `"report"` would quietly feed a report metric into `best_score` and checkpoint selection.
    `scripts/report_eval.py` is the only caller that passes `"report"`.
    """
    from training.cuda_isolation import isolation_enabled, run_isolated

    if isolation_enabled():
        from training.slm_helpers import clear_inference_cache

        payload = {
            "eval_set": eval_set,
            "weights_ref": weights_ref,
            "base_model": base_model,
            "quant": quant,
            "quant_artifact": quant_artifact,
            "quant_backend": quant_backend,
            "scoring": scoring,
        }
        clear_inference_cache()
        try:
            return run_isolated("eval", payload)
        finally:
            clear_inference_cache()

    return _run_eval_local(
        eval_set, weights_ref, base_model, quant=quant, quant_artifact=quant_artifact,
        quant_backend=quant_backend, scoring=scoring,
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


def _resolve_served_infer(spec, weights_ref: str, base_model: str, max_new_tokens: int):
    """An inference function backed by a purpose-started vLLM server, or None to use the usual path.

    Returns None — rather than raising — whenever the served backend is not asked for or is not
    usable. A run must never die because a speedup was unavailable: the in-process path is slower,
    not wrong, and it is always there.
    """
    from eval.student_server import backend

    if backend() != "vllm":
        return None
    from eval.student_server import StudentServerUnavailable, infer_batch_served
    from training.slm_helpers import task_max_seq_length

    def _served(chunk: list[str]) -> list[str]:
        return infer_batch_served(
            chunk,
            weights_ref,
            base_model,
            max_new_tokens=max_new_tokens,
            max_model_len=task_max_seq_length(spec.name),
        )

    # Probed on ONE prompt before the real pass, so a broken configuration costs one engine startup
    # instead of surfacing after the harness has committed to a backend it cannot use.
    try:
        _served(["ping"])
    except StudentServerUnavailable as error:
        print(f"      [student-server] unavailable, using in-process inference: {error}")
        return None
    except Exception as error:  # noqa: BLE001 - any failure here means fall back, never die
        print(f"      [student-server] probe failed ({type(error).__name__}: {error}), "
              "using in-process inference")
        return None
    return _served


def _run_eval_local(
    eval_set: EvalSet,
    weights_ref: str,
    base_model: str,
    quant: str | None = None,
    quant_artifact: str | None = None,
    quant_backend: str | None = None,
    scoring: str = "select",
) -> EvalResult:
    """Run inference on the eval set and score it with the task's own scorer.

    When `quant_artifact` is provided, inference runs through the ENGINE THAT RUNS ON THE PHONE —
    llama-cpp-python for a GGUF, pymnn for an MNN directory — so the accuracy recorded is the
    deployed artifact's, not a full-precision proxy for it. When it is None (base/BF16 model),
    uses Unsloth (infer_batch).

    There is no task dispatch here any more. Every choice — which prompt builder, which extractor,
    which scoring rule, how many output tokens, whether to overlap the judge, whether to record
    reasoning — is read off the task's spec, so a task cannot inherit another's behaviour by
    matching the same branch.
    """
    from tasks import get_task

    spec = get_task(eval_set.task)
    score_fn, metric_name = resolve_scorer(spec, scoring)
    max_new_tokens = eval_output_token_reserve(spec.name)
    prompts = spec.build_prompts(eval_set)
    _served_infer = _resolve_served_infer(spec, weights_ref, base_model, max_new_tokens)

    def _infer(chunk: list[str]) -> list[str]:
        if _served_infer is not None:
            return _served_infer(chunk)
        if quant_artifact is not None:
            from training.quant_backend import MNN, resolve_backend

            if resolve_backend(quant_backend) == MNN:
                return infer_batch_mnn(
                    chunk,
                    quant_artifact,
                    max_new_tokens=max_new_tokens,
                    base_model=base_model,
                    task=spec.name,
                )
            return infer_batch_gguf(
                chunk,
                quant_artifact,
                max_new_tokens=max_new_tokens,
                base_model=base_model,
                # The task, so the GGUF path can size its scoring concurrency from the spec's
                # `eval_batch_size`. Omitting it left `task=""`, which that function degrades to a
                # concurrency of 1 — correct as a safety net for out-of-loop callers, and silently the
                # old sequential behaviour for the one caller that matters. The bf16 branch below
                # already passed it; only this branch was missed, which is the same one-sided update
                # that produced B313.
                task=spec.name,
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
    result = score_fn(eval_set, predictions)
    if spec.attach_reasoning:
        _attach_reasoning_to_failures(result, eval_set, raw_outputs, predictions)

    return EvalResult(
        f1=result["f1"],
        per_class=result["per_class"],
        failures=result["failures"],
        metric=result.get("metric", metric_name),
        format_valid=float(result.get("format_valid", 1.0)),
    )
