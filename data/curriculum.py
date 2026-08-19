# data/curriculum.py
import json
import logging
import os
import random
import re
from collections import Counter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoT annotation (paper §2.3 quality control #5, §2.5)
# The CoT teacher is the local Qwen3.6 synth model, supplied as ``generate_fn`` — and only
# that model. There is no cloud CoT fallback.
# ---------------------------------------------------------------------------


def annotate_cot(
    examples: list[dict],
    generate_fn=None,
    log=print,
) -> list[dict]:
    """Add chain-of-thought reasoning to generation examples.

    Paper §2.3 quality control #5: 'A teacher model generates step-by-step reasoning
    chains for training examples, teaching the model WHY an answer is correct rather
    than only WHAT the answer is.'

    The CoT teacher is the local Qwen3.6 synth model, supplied as ``generate_fn`` — the sole
    backend, with no cloud fallback. Generation is NON-FATAL: if ``generate_fn`` is absent
    (endpoint unavailable) or every call fails, the original example remains CoT-less.

    Returns examples with an added 'cot_reasoning' field, prepended to the response during
    training formatting. Whether a task wants this at all is `TaskSpec.cot_annotation`.

    Efficiency (B141): examples that ALREADY carry a non-empty 'cot_reasoning' are left
    untouched — e.g. the GSM8K loader ships real gold chains-of-thought, so re-generating
    them via the teacher was both wasteful (300 sequential calls per curate, per tier) and
    quality-reducing (gold CoT replaced by a weaker teacher CoT). The examples that DO need
    annotation are processed concurrently with a bounded thread pool.
    """
    def _build_prompt(prompt_text: str, gold_answer: str) -> str:
        return (
            f"Solve this problem step by step, showing your reasoning clearly.\n\n"
            f"Problem: {prompt_text}\n\n"
            f"The correct answer is: {gold_answer}\n\n"
            f"Provide a clear step-by-step explanation of how to arrive at this answer. "
            f"Reply with only the reasoning steps, not the final answer."
        )

    def _present(value) -> bool:
        return value is not None and bool(str(value).strip())

    def _first_present(ex: dict, *keys: str):
        for key in keys:
            if key in ex and _present(ex[key]):
                return ex[key]
        return ""

    def _annotate_one(ex: dict) -> tuple[dict, str | None]:
        # Preserve any existing gold CoT (e.g. GSM8K) — do not regenerate.
        if _present(ex.get("cot_reasoning")):
            return ex, None
        prompt_text = _first_present(ex, "prompt", "text")
        gold_answer = _first_present(ex, "answer", "response", "label")
        if not _present(prompt_text) or not _present(gold_answer):
            return ex, None
        cot_prompt = _build_prompt(prompt_text, gold_answer)
        if generate_fn is not None:
            try:
                # LOCAL synth model (Qwen3.6-35B via vLLM) authors the CoT. Low temperature
                # for focused reasoning; 512 tokens covers verbose math/code chains.
                cot = (generate_fn(cot_prompt, 0.3, 512) or "").strip()
                if cot:
                    return {**ex, "cot_reasoning": cot}, "Qwen3.6"
            except Exception:
                pass
        return ex, None

    # Only spend calls on examples that actually need a CoT.
    need_idx = [i for i, ex in enumerate(examples)
                if not _present(ex.get("cot_reasoning"))
                and _present(_first_present(ex, "prompt", "text"))
                and _present(_first_present(ex, "answer", "response", "label"))]
    # Without a local synth endpoint there is no CoT teacher; leave examples untouched.
    if not need_idx or generate_fn is None:
        return list(examples)

    annotated = list(examples)
    # The local synth server continuous-batches, so match the synth concurrency.
    max_workers = _synth_concurrency(len(need_idx))
    outcomes = _progress_map(
        lambda i: _annotate_one(examples[i]),
        need_idx,
        label="CoT annotation",
        log=log,
        workers=max_workers,
    )

    backend_counts: Counter = Counter()
    for i, (result, backend) in zip(need_idx, outcomes):
        annotated[i] = result
        if backend:
            backend_counts[backend] += 1
    succeeded = sum(backend_counts.values())
    log(f"      [cot] annotated={succeeded}/{len(need_idx)} "
        f"by_backend={dict(backend_counts)} failed={len(need_idx) - succeeded}")
    return annotated


def apply_quality_controls(
    dataset: list[dict],
    task: str,
    *,
    allowed_labels: set | None = None,
    log=None,
) -> list[dict]:
    """Run the quality-control steps THIS task declared, in the order it declared them.

    Previously an `if task_type == ...` chain ending in `else: return dataset`. Measured on
    2026-08-18, that meant four of the eight tasks were never filtered: `function_call` fell into
    the `else`, and `math_reasoning`/`generation` entered their branch but keyed length and dedup
    on a `"prompt"` field their rows do not carry, so a 100,000-character row survived and nothing
    was logged (B299).

    Every step reports what it removed and why. Quality control is the single biggest consumer of
    rows in this pipeline — it deleted ~1,500 of 1,549 synthesized rows in
    slm-clinc150-cse-38155022 — and doing so silently made a curriculum arriving far under target
    look inexplicable (B228).
    """
    if not dataset:
        return dataset
    from data.quality_controls import apply_quality_controls as _apply
    from tasks import get_task

    return _apply(
        dataset,
        get_task(task).quality_controls,
        task_name=task,
        allowed_labels=allowed_labels,
        log=log,
    )


def _synth_concurrency(n_tasks: int) -> int:
    """How many generation requests to run CONCURRENTLY against the synth endpoint.

    Synthesis is I/O-bound on the vLLM server, which continuous-batches concurrent requests
    (its --max-num-seqs). Firing them one-at-a-time was the single biggest wall-clock cost
    (~15 min/rebuild in run 37372065); at concurrency W it becomes ~n/W round-trips. Tunable
    via SLM_SYNTH_CONCURRENCY (default 16); set higher when the server has more GPUs / a
    larger max-num-seqs. Capped at the number of tasks; 1 restores fully-sequential behavior.
    """
    import os
    try:
        c = int(os.environ.get("SLM_SYNTH_CONCURRENCY", "16"))
    except (TypeError, ValueError):
        c = 16
    return max(1, min(c, max(1, n_tasks)))


def _progress_map(fn, items: list, *, label: str, log, workers: int) -> list:
    """Map ``fn`` over ``items`` (concurrently when possible), reporting coarse progress.

    Synthesis issues one model call per row, so the per-call cost lines used to be the only
    sign of life — thousands of them for one rebuild. A ~10-step progress line replaces that
    with something a human can actually read while keeping the run observable.
    """
    total = len(items)
    if total == 0:
        return []
    step = max(1, total // 10)
    results = []

    def _tick(done: int) -> None:
        if log and (done % step == 0 or done == total):
            log(f"      [synth] {label}: {done}/{total} ({100 * done // total}%)")

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, result in enumerate(pool.map(fn, items), 1):
                results.append(result)
                _tick(done)
    else:
        for done, item in enumerate(items, 1):
            results.append(fn(item))
            _tick(done)
    return results


# Task families whose rows carry a free-form answer rather than a label, so "synthesis" means
# generating a whole NEW CORRECT in-distribution example (verified where a verifier exists)
# in the anchor's JSON schema, rather than a new utterance for a known class.
#
# `function_call` and `diff` were MISSING here until 2026-08-17, and because
# `synthesize_examples` ends in a bare `return []` for anything it does not recognise, the
# `synthesize` strategy was a silent no-op on every format-bound task. In xlam run 38566712 the
# orchestrator chose it for six of eight rebuilds, announced 250-500 rows each time against a
# reachable teacher endpoint, and produced zero rows on all six with nothing in the log to say so —
# and the exact verifiers built for exactly this path (`data/synth_verifiers.py`:
# `verify_function_call_row`, `verify_calendar_row`) had never once executed in production.
# `_synthesize_new_correct` was clearly written with these tasks in mind: it pins `tools` from the
# anchor, a field only a function-calling row has.


# How many worked examples to show the teacher. FIVE, not zero — measured on 2026-08-17: the teacher
# scores 0.1131 span-F1 zero-shot on BC5CDR NER and 0.7190 with five demonstrations, a 6.4x
# difference, and the raw outputs show why (wrong casing, a class the task does not have, a markdown
# fence). Min et al. (arXiv:2202.12837) attribute exactly that to demonstrations supplying "(1) the
# label space, (2) the distribution of the input text, and (3) the overall format of the sequence".
# A generator asked for output in a precise contract it has only been described, not shown, is being
# tested on guessing the contract. B276.
SYNTH_SHOTS = int(os.environ.get("SLM_SYNTH_SHOTS", "5"))


def _demo_block(demos: list[dict], task_description: str) -> str:
    """Worked examples in the exact JSON shape the teacher is about to be asked for."""
    import json

    parts = []
    for demo in demos:
        payload = {k: demo.get(k) for k in demo if not str(k).startswith("_")}
        parts.append(json.dumps(payload, ensure_ascii=False))
    return "\n\n".join(parts)


def _new_example_prompt(anchor: dict, task_description: str, demos: list[dict] | None = None,
                        target_category: str = "") -> str:
    """Prompt to generate ONE new, correct example in the anchor's exact schema.

    Generation-family only. The classification/NER path (`_synthesize_new_gold`) never asks the
    teacher for a label — it copies the anchor's — so an out-of-vocabulary label is impossible
    there by construction, and no label list needs to be stated.

    `demos` are SHOWN, not described. See SYNTH_SHOTS for why that matters (B276).
    """
    import json

    schema = {k: anchor.get(k) for k in anchor if not str(k).startswith("_")}
    shown = ""
    if demos:
        shown = (
            f"Here are {len(demos)} real examples of this task, in the exact output format "
            f"required:\n\n{_demo_block(demos, task_description)}\n\n"
        )
    aimed = (
        f"\nThe model being trained is currently failing on: {target_category}. Favour examples "
        f"that exercise exactly that difficulty.\n"
        if target_category else ""
    )
    return (
        f"{task_description}\n\n"
        f"{shown}"
        f"Generate ONE new, correct example in EXACTLY this JSON schema "
        f"(same keys, same value types): {json.dumps(schema, ensure_ascii=False)}."
        f"{aimed}"
        "\nIt must be a genuinely new, diverse, and CORRECT instance — not a copy or a "
        "paraphrase of the reference, and never a wrong answer. Obey the output contract above "
        "exactly; a well-formed answer that breaks a stated convention is graded wrong. Return "
        "only the JSON object, no preamble or code fences."
    )


# The teacher's description of the task is authored by the ORCHESTRATOR at cold start, from real
# rows, and reaches every prompt here as `task_description` (see agent/task_brief.py). It replaced a
# table keyed by task TYPE, under which `xlam_bfcl` and `calendar_json` were both described as
# "converting a request into a JSON function call using only the declared tools" — which omits every
# convention that makes a calendar row correct (the 60-minute default, ISO-8601, resolving against
# the reference instant), so a verifier asked to judge against it was judging its own guess (B269).


def verify_generated_answers(
    rows: list[dict],
    *,
    task_description: str,
    generate_fn,
    log=None,
    reference_rows: list[dict] | None = None,
) -> list[dict]:
    """Ask the teacher whether each generated (input, answer) pair is actually correct.

    The generation family is the ONE synthesis path where the teacher invents both the input and
    the output, so nothing about the anchor constrains correctness — and it was running completely
    unchecked: `curate._verifier_for` returns None for every one of these task types, so
    `_synthesize_new_correct`'s `if verify_fn is not None` branch had never executed and every batch
    logged `450/450 kept`. On `calendar_json` the teacher scores 0.2176, so that is ~5,700
    machine-invented training targets from a model that gets the task right 22% of the time (B269).

    Same fail-open contract as `verify_generated_labels`: an unparseable reply or endpoint error
    KEEPS the row, because a verifier must never be able to empty a dataset.
    """
    if not rows or generate_fn is None:
        return rows
    task_desc = task_description
    # Real (request, correct answer) pairs, so the verifier has the task's actual conventions in
    # front of it instead of inferring them. On `calendar_json` those conventions ARE the task
    # (60-minute default, ISO-8601, resolve against the reference instant) and a verifier that has
    # to guess them is judging its own guess.
    shown = ""
    if reference_rows:
        pairs = []
        for ref in reference_rows[:SYNTH_SHOTS]:
            q = " ".join(str(ref.get("text") or "").split())[:400]
            a = " ".join(str(ref.get("answer") or "").split())[:400]
            if q and a:
                pairs.append(f"Request: {q}\nCorrect answer: {a}")
        if pairs:
            shown = ("Real, confirmed examples of this task:\n\n"
                     + "\n\n".join(pairs) + "\n\n")

    def _check(row: dict):
        request = str(row.get("text") or "")
        answer = str(row.get("answer") or row.get("response") or "")
        if not answer.strip():
            return row, False, "empty answer"
        prompt = (
            f"You are checking one training example for the task of {task_desc}.\n\n"
            f"{shown}"
            f"User input / question:\n{request}\n\n"
            f"Proposed answer:\n{answer}\n\n"
            f"Does the proposed answer correctly and directly satisfy the user's request, in the "
            f"context of {task_desc}? Answer strictly as JSON: "
            f'{{"valid": true|false, "reason": "<max 15 words>"}}. '
            f"Answer false if the answer is wrong, incomplete, in the wrong format for this task, "
            f"or does not address what was actually asked."
        )
        try:
            raw = generate_fn(prompt, 0.0, 160)
        except Exception:  # noqa: BLE001 — verification must never be fatal
            return row, True, "verifier unavailable (kept)"
        match = re.search(r"\{.*\}", str(raw or ""), re.DOTALL)
        if not match:
            return row, True, "unparseable verifier reply (kept)"
        try:
            verdict = json.loads(match.group())
        except json.JSONDecodeError:
            return row, True, "unparseable verifier reply (kept)"
        return row, bool(verdict.get("valid", True)), str(verdict.get("reason", ""))[:120]

    results = _progress_map(
        _check, rows, label="answer verification", log=log,
        workers=_synth_concurrency(len(rows)),
    )
    kept, rejected = [], []
    for row, valid, reason in results:
        (kept if valid else rejected).append((row, reason))
    if log:
        log(f"      [verify] teacher validated {len(kept)}/{len(rows)} generated answer(s); "
            f"rejected {len(rejected)}")
        for row, reason in rejected[:_VERIFY_LOG_LIMIT]:
            text = " ".join(str(row.get("text") or "").split())[:70]
            log(f"        REJECTED {text!r} — teacher: {reason}")
        if len(rejected) > _VERIFY_LOG_LIMIT:
            log(f"        ... and {len(rejected) - _VERIFY_LOG_LIMIT} more rejected")
    return [row for row, _ in kept]


def _synthesize_new_correct(
    examples: list[dict],
    *,
    task_description: str,
    n: int,
    generate_fn,
    verify_fn=None,
    log=None,
    target_category: str = "",
) -> list[dict]:
    """Generate ``n`` new CORRECT in-distribution examples (never wrong-answer pairs).

    Each candidate is parsed as JSON in the anchor's schema and, when ``verify_fn`` is
    supplied (e.g. a math answer-checker or code test-runner), kept only if it verifies.
    Failures are non-fatal — a bad generation is skipped, bounded by ``n * 4`` attempts.
    """
    import json

    anchors = list(examples)
    if not anchors:
        return []
    rng = random.Random(20260817)
    rng.shuffle(anchors)

    rejections: list[str] = []
    checker = getattr(verify_fn, "checker", None)

    def _one(anchor: dict) -> dict | None:
        demos = rng.sample(anchors, min(SYNTH_SHOTS, len(anchors))) if SYNTH_SHOTS else []
        prompt = _new_example_prompt(anchor, task_description, demos=demos,
                                     target_category=target_category)
        try:
            row = json.loads(generate_fn(prompt, temperature=0.7, max_tokens=512))
        except Exception:  # noqa: BLE001 — a failed generation is skipped, never fatal
            return None
        if not isinstance(row, dict) or not row.get("text"):
            return None
        # PIN the constraint from the anchor rather than trusting the teacher to reproduce it. The
        # tool signature is what the call must satisfy, not something being invented — the same
        # "supply what you can, generate only what you must" principle that makes the
        # classification path safe. It also makes the row VERIFIABLE: a generated row with no
        # `tools` cannot be schema-checked at all.
        for pinned in ("tools", "_instruction"):
            if pinned in anchor and pinned not in row:
                row[pinned] = anchor[pinned]
        # STAGE 1 — exact, programmatic check. Runs BEFORE any model-based verification, because it
        # is free, cannot be fooled, and a row it rejects should never cost a teacher call.
        if verify_fn is not None and not verify_fn(row):
            if checker is not None:
                _ok, reason = checker(row)
                rejections.append(reason)
            return None
        row["_source"] = "synth:generated"
        row["_provenance"] = "synthetic_positive"
        return row

    # Parallel, like every other generator in this module. This loop used to be SERIAL — one
    # blocking generate_fn call at a time — while the classification path (_synthesize_new_gold)
    # and annotate_cot both fan out over _progress_map. The synthesis server is configured for
    # SLM_SYNTH_CONCURRENCY (8 by default) in-flight requests, so the generation family was
    # using an eighth of the capacity it had already paid for: in
    # slm-dialogsum-samsum-cse-38186914 vLLM sat at "Running: 1 reqs" and 21% GPU utilisation
    # for hours while filling 2,678 rows, with no progress line to show it was alive (B252).
    #
    # Over-request by the same 4x the old attempt budget allowed, so rejects still leave enough
    # accepted rows to reach n, then trim.
    max_attempts = max(1, n) * 4
    planned = [anchors[i % len(anchors)] for i in range(min(max_attempts, max(n * 2, n + 32)))]
    produced = _progress_map(
        _one,
        planned,
        label="new-correct synthesis",
        log=log,
        workers=_synth_concurrency(len(planned)),
    )
    out = [row for row in produced if row is not None][:n]
    if log:
        shots = f"{SYNTH_SHOTS}-shot" if SYNTH_SHOTS else "zero-shot"
        log(f"      [synth] new-correct ({shots}): {len(out)}/{n} kept "
            f"({len(planned)} attempts)")
        if rejections:
            counts = Counter(rejections)
            log(f"      [verify:exact] programmatic verifier rejected {len(rejections)} row(s) "
                f"before any teacher call:")
            for reason, count in counts.most_common(_VERIFY_LOG_LIMIT):
                log(f"        x{count}  {reason}")
    return out


# How many rejected rows to quote in the log before summarising the rest.
_VERIFY_LOG_LIMIT = int(os.environ.get("SLM_VERIFY_LOG_LIMIT", "10"))


def _verify_synth_enabled() -> bool:
    """Teacher label-verification of generated rows. On by default; set 0 to skip the pass."""
    return os.environ.get("SLM_VERIFY_SYNTH", "1") == "1"


def _label_context_block(
    label: str,
    label_definitions: dict | None,
    all_labels: list[str] | None,
) -> str:
    """Explain what the labels MEAN, so the teacher judges the task and not the label's wording.

    Without this the prompts named the label and nothing else, and the teacher read the label as an
    ordinary English word. On RouterBench — where `local` means "a small on-device model can answer
    this correctly" — it rejected 70% of generated rows for reasons like *"the utterance is a math
    problem, not a local query"* and *"the utterance describes fog, not clouds"*. Those verdicts are
    correct answers to the question the prompt actually asked; the prompt was asking the wrong one.
    """
    if not label_definitions:
        return ""
    meaning = label_definitions.get(label)
    lines = ["", "What the labels mean for THIS task (judge by these definitions, NOT by the"
             " everyday meaning of the label word):"]
    for name in (all_labels or sorted(label_definitions)):
        definition = label_definitions.get(name)
        if definition:
            lines.append(f"  - {name!r}: {definition}")
    if meaning:
        lines.append("")
        lines.append(f"The label under consideration is {label!r}, which means: {meaning}")
    return "\n".join(lines) + "\n"


def verify_generated_labels(
    rows: list[dict],
    *,
    generate_fn,
    log=None,
    label_definitions: dict | None = None,
    all_labels: list[str] | None = None,
    reference_rows: list[dict] | None = None,
) -> list[dict]:
    """Ask the teacher model to confirm each generated row really belongs to its assigned label.

    Generation and verification are NOT the same task. Writing "an utterance that belongs to
    class X" is open-ended; deciding "does this utterance belong to class X, yes or no" is the
    classification task the reference model is already good at. So a self-check is cheap and
    meaningfully better than nothing, even though it uses the same model.

    `label_definitions` says what each class MEANS. Supply it whenever the label name is not itself
    a plain description of the class, or the teacher will judge the word instead of the task — see
    `_label_context_block`.

    Rows the teacher rejects are dropped, and its stated reason is logged so a bad *generator*
    prompt is visible rather than silently absorbed. Any verification failure (unparseable reply,
    endpoint error) KEEPS the row — the verifier must never be able to empty a dataset.
    """
    if not rows or generate_fn is None:
        return rows

    # Real in-class examples per label, so the verifier judges against the class as it ACTUALLY
    # appears rather than against its own reading of the label word. Same reason the generator is
    # few-shot (B276): a decision boundary is easier to show than to describe.
    by_label: dict[str, list[str]] = {}
    for row in (reference_rows or []):
        if isinstance(row, dict) and row.get("label") is not None:
            text = str(row.get("text") or "").strip()
            if text:
                by_label.setdefault(str(row["label"]), []).append(text)

    def _check(row: dict):
        label = str(row.get("label"))
        text = str(row.get("text") or "")
        context = _label_context_block(label, label_definitions, all_labels)
        examples = by_label.get(label, [])[:SYNTH_SHOTS]
        shown = ""
        if examples:
            joined = "\n".join(f"- {e}" for e in examples)
            shown = f"Real, confirmed examples of the '{label}' class:\n{joined}\n\n"
        prompt = (
            f"You are checking one training example for a text classifier.\n"
            f"{context}\n"
            f"{shown}"
            f"Utterance: {text}\n"
            f"Proposed label: {label}\n\n"
            f"Does this utterance genuinely belong to the '{label}' class? Answer strictly as "
            f'JSON: {{"valid": true|false, "reason": "<max 15 words>"}}. '
            f"Answer false if the utterance actually belongs to a different class, is "
            f"incoherent, or mixes two intents. Do NOT answer false merely because the "
            f"utterance's TOPIC is unrelated to the label's wording — judge only whether the "
            f"class, as defined above, applies."
        )
        try:
            raw = generate_fn(prompt, 0.0, 120)
        except Exception:  # noqa: BLE001 — verification must never be fatal
            return row, True, "verifier unavailable (kept)"
        match = re.search(r"\{.*\}", str(raw or ""), re.DOTALL)
        if not match:
            return row, True, "unparseable verifier reply (kept)"
        try:
            verdict = json.loads(match.group())
        except json.JSONDecodeError:
            return row, True, "unparseable verifier reply (kept)"
        return row, bool(verdict.get("valid", True)), str(verdict.get("reason", ""))[:120]

    workers = _synth_concurrency(len(rows))
    results = _progress_map(
        _check, rows, label="label verification", log=log, workers=workers
    )
    kept, rejected = [], []
    for row, valid, reason in results:
        (kept if valid else rejected).append((row, reason))

    if log:
        log(
            f"      [verify] teacher validated {len(kept)}/{len(rows)} generated row(s); "
            f"rejected {len(rejected)}"
        )
        for row, reason in rejected[:_VERIFY_LOG_LIMIT]:
            text = " ".join(str(row.get("text") or "").split())[:80]
            log(f"        REJECTED [{row.get('label')}] {text!r} — teacher: {reason}")
        if len(rejected) > _VERIFY_LOG_LIMIT:
            log(f"        ... and {len(rejected) - _VERIFY_LOG_LIMIT} more rejected")
    return [row for row, _ in kept]


def _synthesize_new_gold(
    examples: list[dict],
    *,
    task_description: str,
    n: int,
    generate_fn,
    log=None,
    label_definitions: dict | None = None,
    target_category: str = "",
) -> list[dict]:
    """Generate ``n`` NEW CORRECT in-class examples, spread evenly over the label space.

    The generated row keeps its anchor's label — the model is never asked to choose one — so
    this grows in-class coverage without disturbing the class histogram, and an out-of-vocabulary
    label is impossible by construction. Anchors are drawn round-robin across labels so rare
    classes get the same attention as common ones.
    """
    if n <= 0 or not examples:
        return []
    by_label: dict[str, list[dict]] = {}
    for row in examples:
        label = row.get("label")
        if label is not None and str(row.get("text") or "").strip():
            by_label.setdefault(str(label), []).append(row)
    if not by_label:
        # NER rows keep their gold in `entities` and carry NO `label`, so this bucketing is always
        # empty for NER and the function has therefore NEVER produced a row: the BC5CDR run logged
        # `requested 5693 new-gold row(s) -> kept 0` on every call and ran ~7,000 rows below target
        # with no explanation. Say so, because silence here reads as a synthesis failure.
        #
        # This must stay a no-op rather than being "fixed" to generate. The generator returns
        # {text, label} and never entity spans, so a produced row would either have no `entities`
        # (dropped by the NER schema filter) or carry the ANCHOR sentence's spans against NEW text —
        # fabricated gold. Span synthesis needs its own generator, not this one.
        if log:
            log(
                "      [synth] SKIPPED: no anchor row carries both a 'label' and non-empty "
                "'text', so there is nothing to generate in-class from. A task whose gold lives "
                "somewhere else (NER spans, for instance) should declare synthesize=None rather "
                "than reach this."
            )
        return []

    rng = random.Random(20260804)
    labels = sorted(by_label)
    for bucket in by_label.values():
        rng.shuffle(bucket)
    anchors: list[dict] = []
    depth = 0
    while len(anchors) < n:
        progressed = False
        for label in labels:
            bucket = by_label[label]
            if depth < len(bucket):
                anchors.append(bucket[depth])
                progressed = True
                if len(anchors) == n:
                    break
        if not progressed:
            break
        depth += 1

    def _gold_one(anchor: dict) -> dict | None:
        label = str(anchor.get("label"))
        context = _label_context_block(label, label_definitions, labels)
        # SHOW several real examples of this class, not one. The teacher's zero-shot score is a poor
        # guide to what it can produce when the distribution is demonstrated: measured 0.1131 ->
        # 0.7190 on BC5CDR NER between zero- and five-shot (B276). Demonstrations here come from the
        # SAME class, so they convey that class's phrasing and length distribution as well as format.
        same_class = by_label.get(label) or []
        demos = [row for row in same_class[:SYNTH_SHOTS + 1] if row is not anchor][:SYNTH_SHOTS]
        shown = ""
        if demos:
            joined = "\n".join(f"- {str(d.get('text', '')).strip()}" for d in demos)
            shown = (
                f"Real examples of the '{label}' class, showing its typical phrasing and "
                f"length:\n{joined}\n\n"
            )
        aimed = (
            f"The model being trained is currently failing on: {target_category}. Favour "
            f"utterances that exercise exactly that difficulty.\n\n"
            if target_category else ""
        )
        prompt = (
            f"{task_description}\n\n"
            f"Write ONE new, realistic user utterance that belongs to the '{label}' class "
            f"of a text classifier.\n"
            f"{context}\n"
            f"{aimed}"
            f"{shown}"
            f"It must be genuinely NEW and phrased differently from the "
            f"references — not a paraphrase, not a copy — while unambiguously belonging to "
            f"'{label}'.\n\n"
            f"Reference '{label}' example:\n{anchor.get('text', '')}\n\n"
            f"Output ONLY the new utterance — no preamble, no explanation, no quotation marks, "
            f"no label prefix."
        )
        try:
            text = generate_fn(prompt, 1.0, 200)
        except Exception:  # noqa: BLE001 — a failed generation is skipped, never fatal
            return None
        text = str(text or "").strip()
        if not text:
            return None
        return {
            "text": text,
            "label": anchor.get("label"),
            "_source": "synth",
            "_provenance": "synthetic_positive",
        }

    workers = _synth_concurrency(len(anchors))
    produced = _progress_map(
        _gold_one,
        anchors,
        label=f"new gold ({len(labels)} labels, {SYNTH_SHOTS}-shot)",
        log=log,
        workers=workers,
    )
    return [row for row in produced if row is not None]


def synthesize_examples(
    examples: list[dict],
    *,
    task: str,
    n: int,
    generate_fn,
    verify_fn=None,
    log=None,
    label_definitions: dict | None = None,
    brief: dict | None = None,
    target_category: str = "",
) -> list[dict]:
    """Generate `n` new rows anchored on real `examples`.

    Which of two shapes is produced is DERIVED from the task, not chosen by a channel:

      closed label space  → a new INPUT for the anchor's existing class. The generated row inherits
                            a real row's label, so the target cannot be wrong; only the phrasing
                            can. Verified by asking the teacher whether the phrasing really
                            expresses that class.
      open-ended target   → a whole new (input, answer) pair in the anchor's schema. This is the one
                            case where the teacher invents both halves, so it is gated by the task's
                            exact programmatic verifier first (free, unfoolable, available for the
                            format-bound tasks) and by the teacher's own answer check second.

    `brief` is the orchestrator's description of the benchmark, authored once at cold start from
    real rows (see agent/task_brief.py). It replaces a hardcoded one-line description per task
    TYPE, which could not distinguish two tasks sharing a type and told the teacher nothing about
    the conventions that make an answer correct (B269).

    `target_category` names the failure category this batch is aimed at, so the teacher is asked
    for rows that exercise what the model is actually getting wrong rather than for more of the
    same.

    Non-fatal throughout: an unavailable or failing backend yields fewer rows, never raises.
    """
    from tasks import get_task

    spec = get_task(task)
    if n <= 0 or not examples:
        return []

    if spec.closed_label_space:
        rows = _synthesize_new_gold(
            examples,
            task_description=_describe(brief, spec),
            n=n,
            generate_fn=generate_fn,
            log=log,
            label_definitions=label_definitions,
            target_category=target_category,
        )
        if rows and _verify_synth_enabled():
            rows = verify_generated_labels(
                rows,
                generate_fn=generate_fn,
                log=log,
                label_definitions=label_definitions,
                all_labels=sorted({
                    str(row.get("label")) for row in examples
                    if isinstance(row, dict) and row.get("label") is not None
                }) or None,
                reference_rows=examples,
            )
        if log:
            log(f"      [synth] requested {n} in-class row(s) -> kept {len(rows)}")
        return rows

    rows = _synthesize_new_correct(
        examples,
        task_description=_describe(brief, spec),
        n=n,
        generate_fn=generate_fn,
        verify_fn=verify_fn if verify_fn is not None else spec.synth_verifier,
        log=log,
        target_category=target_category,
    )
    # Teacher answer-verification. The programmatic verifier above is exact but only checks FORM;
    # this asks whether the answer is actually right. Weaker than execution feedback, but the
    # alternative was keeping 100% of whatever the teacher produced (B269).
    if rows and _verify_synth_enabled():
        rows = verify_generated_answers(
            rows,
            task_description=_describe(brief, spec),
            generate_fn=generate_fn,
            log=log,
            reference_rows=examples,
        )
    if log:
        verifier = "exact+teacher" if (verify_fn or spec.synth_verifier) else "teacher only"
        log(f"      [synth] requested {n} new-correct row(s) -> kept {len(rows)} ({verifier})")
    return rows


def _describe(brief: dict | None, spec) -> str:
    """The task description inserted into teacher prompts."""
    from agent.task_brief import brief_context_block

    return brief_context_block(brief, spec)
