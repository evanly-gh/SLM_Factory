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
    task_type: str = "generation",
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

    Returns examples with an added 'cot_reasoning' field. The CoT is prepended to the
    response during training formatting. The prompt is task-aware: code-generation gets
    an implementation-plan style explanation rather than a prose math-style derivation.

    Efficiency (B141): examples that ALREADY carry a non-empty 'cot_reasoning' are left
    untouched — e.g. the GSM8K loader ships real gold chains-of-thought, so re-generating
    them via the teacher was both wasteful (300 sequential calls per curate, per tier) and
    quality-reducing (gold CoT replaced by a weaker teacher CoT). The examples that DO need
    annotation are processed concurrently with a bounded thread pool.
    """
    def _build_prompt(prompt_text: str, gold_answer: str) -> str:
        if task_type == "code_generation":
            return (
                f"Explain the reasoning behind this code solution as a concise implementation "
                f"plan a developer would follow: the approach, key steps, and any edge cases "
                f"handled. Do NOT restate the full code.\n\n"
                f"Problem:\n{prompt_text}\n\n"
                f"Correct solution:\n{gold_answer}\n\n"
                f"Reply with only the step-by-step implementation reasoning, not the code and "
                f"not the final answer."
            )
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
    task_type: str = "classification",
    *,
    allowed_labels: set | None = None,
    log=None,
) -> list[dict]:
    """
    Enforce quality controls from the paper (§2.3):
    1. Label balancing: no label exceeds 3x count of any other (classification family)
    2. Context-length matching: remove outliers >3x median length (all types)
    3. Entity diversification: cap any entity value at 3 occurrences (NER)
    4. Surface-form dedup: remove near-duplicate texts (all types, Jaccard >0.9)

    New types route to their natural family:
      classification          → label balancing + dedup
      NER                     → entity diversification + length filter
      math_reasoning          → length filter on prompt + dedup
      code_generation         → length filter on prompt + dedup
      generation              → length filter on prompt
    """
    if not dataset:
        return dataset

    # Every control reports what it removed and why. Quality control is the single biggest
    # consumer of rows in this pipeline — it deleted ~1,500 of 1,549 synthesized rows in
    # slm-clinc150-cse-38155022 — and previously did so completely silently, so a dataset
    # arriving far under target looked inexplicable (B228).
    def _report(stage: str, before: int, after: int, reason: str) -> None:
        if log and after < before:
            log(f"      [qc] {stage}: removed {before - after} row(s) — {reason}")

    if task_type == "classification":
        before = len(dataset)
        clean = [e for e in dataset if "text" in e and "label" in e]
        _report("schema", before, len(clean), "row missing a 'text' or 'label' field")

        # 0. Label-space validation. A row whose label is not one of the task's established
        # classes can never match an eval label, so it is pure training noise. This is where
        # mined sources carrying raw integer class ids ("0"/"1"/"2") get removed (B222/B229).
        if allowed_labels:
            before = len(clean)
            rejected = Counter(
                str(e["label"]) for e in clean if str(e["label"]) not in allowed_labels
            )
            clean = [e for e in clean if str(e["label"]) in allowed_labels]
            if rejected and log:
                sample = ", ".join(
                    f"{lab!r}x{cnt}" for lab, cnt in rejected.most_common(5)
                )
                log(
                    f"      [qc] label-space: removed {before - len(clean)} row(s) — "
                    f"label not in the task's {len(allowed_labels)} established classes "
                    f"[{sample}]"
                )

        # 1. Label balancing (single-label argmax — for multi_label the flag is on
        # the EvalSet, not here; we balance by the primary label field).
        counts = Counter(e["label"] for e in clean)
        if counts:
            min_count = max(min(counts.values()), 1)
            max_allowed = 3 * min_count
            balanced = []
            seen: Counter = Counter()
            for ex in clean:
                if seen[ex["label"]] < max_allowed:
                    balanced.append(ex)
                    seen[ex["label"]] += 1
            _report(
                "label-balance", len(clean), len(balanced),
                f"label over the cap of 3x the smallest class ({max_allowed} rows/label; "
                f"smallest class has {min_count})",
            )
            clean = balanced

        before = len(clean)
        clean = _filter_length_outliers(clean)
        _report("length-outlier", before, len(clean), "text longer than 3x the median length")

        before = len(clean)
        clean = _dedup_surface_forms(clean)
        _report("surface-dedup", before, len(clean), "near-duplicate text (Jaccard > 0.9)")
        return clean

    elif task_type == "NER":
        clean = [e for e in dataset if "text" in e and "entities" in e]

        # 3. Entity diversification: no entity surface value appears >3 times.
        # Also applies to structured_extraction (schema field values).
        entity_counts: Counter = Counter()
        for ex in clean:
            for ent in ex.get("entities", []):
                entity_counts[ent.get("text", "").lower()] += 1
        over_represented = {k for k, v in entity_counts.items() if v > 3}
        if over_represented:
            result = []
            running: Counter = Counter()
            for ex in clean:
                ent_values = [e.get("text", "").lower() for e in ex.get("entities", [])]
                skip = False
                for v in ent_values:
                    if v in over_represented and running[v] >= 3:
                        skip = True
                        break
                if not skip:
                    result.append(ex)
                    for v in ent_values:
                        running[v] += 1
            clean = result

        clean = _filter_length_outliers(clean)
        return clean

    elif task_type in ("math_reasoning", "code_generation", "generation"):
        clean = [
            e for e in dataset
            if ("prompt" in e and "response" in e) or ("text" in e and "label" in e)
        ]
        key = "text" if task_type == "code_generation" else "prompt"
        clean = _filter_length_outliers(clean, key=key)
        # Dedup on math/code — repeated problem templates inflate the dataset
        # without adding coverage. Generation is diverse enough to skip dedup.
        if task_type in ("math_reasoning", "code_generation"):
            clean = _dedup_surface_forms(clean, key=key)
        return clean

    else:
        return dataset


# Row provenances that define the task's true length distribution: real rows from the benchmark's
# own train split. Mined and synthesized rows do not — see _filter_length_outliers.
_TRUSTED_LENGTH_PROVENANCE = frozenset({"train_anchor", "resample"})


def _filter_length_outliers(
    examples: list[dict], key: str = "text", max_ratio: float = 3.0,
) -> list[dict]:
    """Remove examples whose text length exceeds max_ratio × median. Paper §2.3 item 3.

    The median is computed over TRUSTED rows only — the task's own real training data — not over
    the whole dataset (B260). The ratio is relative, so whatever sets the median sets the cutoff,
    and that made the filter destroy the real data it was meant to protect:

    RouterBench's real rows have a median length of 715 characters. Mining substituted a foreign
    dataset (B259) whose rows have a median of 75, which dragged the DATASET median down to 269 and
    the cutoff from 2145 to 807 — a bound that removes 48% of the real benchmark (17,667 of 36,497
    rows) instead of the 0.6% it removes on clean data. It is visible in the artifacts: the median
    length of `train_anchor` rows fell 572 → 432 → 396 across three rebuilds while the pool they
    were sampled from never changed.

    Anchoring on trusted rows keeps the intended behaviour (drop genuine outliers relative to the
    task's own distribution) and makes it impossible for injected rows to move the goalposts. Falls
    back to all rows when nothing is tagged, so untagged callers and unit tests behave as before.
    """
    if not examples:
        return examples
    trusted = [
        e for e in examples
        if str(e.get("_provenance") or "") in _TRUSTED_LENGTH_PROVENANCE
    ]
    basis = trusted or examples
    lengths = sorted(len(e.get(key, "")) for e in basis)
    median = lengths[len(lengths) // 2] or 1
    cutoff = median * max_ratio
    return [e for e in examples if len(e.get(key, "")) <= cutoff]


def _dedup_surface_forms(
    examples: list[dict], threshold: float = 0.9, key: str = "text"
) -> list[dict]:
    """Remove near-duplicate texts using word-set Jaccard similarity."""
    result = []
    seen_word_sets: list[set] = []
    for ex in examples:
        words = set(ex.get(key, "").lower().split())
        if not words:
            result.append(ex)
            continue
        is_dup = False
        for seen in seen_word_sets[-50:]:
            intersection = len(words & seen)
            union = len(words | seen)
            if union > 0 and intersection / union >= threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(ex)
            seen_word_sets.append(words)
    return result


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
_GENERATION_FAMILY = frozenset({
    "math_reasoning",
    "code_generation",
    "generation",
    "multilingual",
    "structured_extraction",
})


def _new_example_prompt(anchor: dict, task_type: str) -> str:
    """Prompt to generate ONE new, correct example in the anchor's exact schema.

    Generation-family only. The classification/NER path (`_synthesize_new_gold`) never asks the
    teacher for a label — it copies the anchor's — so an out-of-vocabulary label is impossible
    there by construction, and no label list needs to be stated.
    """
    import json

    schema = {k: anchor.get(k) for k in anchor if not str(k).startswith("_")}
    return (
        f"Generate ONE new, correct {task_type} example in EXACTLY this JSON schema "
        f"(same keys, same value types): {json.dumps(schema, ensure_ascii=False)}. "
        "It must be a genuinely new, diverse, and CORRECT instance — not a copy or a "
        "paraphrase of the reference, and never a wrong answer. Return only the JSON "
        "object, no preamble or code fences."
    )


_TASK_DESCRIPTIONS = {
    "generation": "summarising a conversation in one to three sentences",
    "function_call": "converting a request into a JSON function call using only the declared tools",
    "diff": "producing a unified diff that applies the requested edit",
    "math_reasoning": "solving a math word problem and giving the final numeric answer",
    "code_generation": "writing code that satisfies the stated problem",
    "multilingual": "responding correctly in the language of the request",
    "structured_extraction": "extracting the requested fields into the given schema",
}


def verify_generated_answers(
    rows: list[dict],
    *,
    task_type: str,
    generate_fn,
    log=None,
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
    task_desc = _TASK_DESCRIPTIONS.get(task_type, task_type)

    def _check(row: dict):
        request = str(row.get("text") or "")
        answer = str(row.get("answer") or row.get("response") or "")
        if not answer.strip():
            return row, False, "empty answer"
        prompt = (
            f"You are checking one training example for the task of {task_desc}.\n\n"
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
        _check, rows, label=f"answer verification ({task_type})", log=log,
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
    task_type: str,
    n: int,
    generate_fn,
    verify_fn=None,
    log=None,
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
    random.shuffle(anchors)

    def _one(anchor: dict) -> dict | None:
        prompt = _new_example_prompt(anchor, task_type)
        try:
            row = json.loads(generate_fn(prompt, temperature=0.7, max_tokens=512))
        except Exception:  # noqa: BLE001 — a failed generation is skipped, never fatal
            return None
        if not isinstance(row, dict) or not row.get("text"):
            return None
        if verify_fn is not None and not verify_fn(row):
            return None
        row["_source"] = f"synth:{task_type}"
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
        label=f"new-correct synthesis ({task_type})",
        log=log,
        workers=_synth_concurrency(len(planned)),
    )
    out = [row for row in produced if row is not None][:n]
    if log:
        log(f"  new-correct synthesis: {len(out)}/{n} kept ({len(planned)} attempts)")
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

    def _check(row: dict):
        label = str(row.get("label"))
        text = str(row.get("text") or "")
        context = _label_context_block(label, label_definitions, all_labels)
        prompt = (
            f"You are checking one training example for a text classifier.\n"
            f"{context}\n"
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
    task_type: str,
    n: int,
    generate_fn,
    log=None,
    label_definitions: dict | None = None,
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
            if task_type == "NER":
                log(
                    "      [synth] SKIPPED: NER rows have no 'label' field to anchor in-class "
                    "generation, and this generator cannot produce entity spans. NER curricula are "
                    "gold-only by design — no rows generated, no teacher calls made."
                )
            else:
                log(
                    f"      [synth] SKIPPED: no anchor row carries both a 'label' and non-empty "
                    f"'text' (task_type={task_type}), so there is nothing to generate in-class from."
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
        prompt = (
            f"Write ONE new, realistic user utterance that belongs to the '{label}' class "
            f"of a text classifier.\n"
            f"{context}\n"
            f"It must be genuinely NEW and phrased differently from the "
            f"reference — not a paraphrase, not a copy — while unambiguously belonging to "
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
        label=f"new gold ({task_type}, {len(labels)} labels)",
        log=log,
        workers=workers,
    )
    return [row for row in produced if row is not None]


def synthesize_examples(
    examples: list[dict],
    *,
    task_type: str,
    n: int,
    generate_fn,
    verify_fn=None,
    log=None,
    label_definitions: dict | None = None,
) -> list[dict]:
    """Unified, task-adaptive synthesis entry.

    - classification / NER: NEW GOLD (in-class) examples only, spread across the label space
      and then label-verified by the teacher model.
    - generation-family (math/code/generation/multilingual/structured): NEW CORRECT
      in-distribution examples, verified when a ``verify_fn`` is supplied.

    Returns synthetic rows in the same format as the real data. Non-fatal: an unavailable or
    failing backend yields fewer rows, never raises.
    """
    if n <= 0 or not examples:
        return []
    if task_type in ("classification", "NER"):
        rows = _synthesize_new_gold(
            examples,
            task_type=task_type,
            n=n,
            generate_fn=generate_fn,
            log=log,
            label_definitions=label_definitions,
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
            )
        if log:
            log(f"      [synth] requested {n} new-gold row(s) -> kept {len(rows)}")
        return rows
    if task_type in _GENERATION_FAMILY:
        rows = _synthesize_new_correct(
            examples,
            task_type=task_type,
            n=n,
            generate_fn=generate_fn,
            verify_fn=verify_fn,
            log=log,
        )
        # Teacher answer-verification for the generation family. `verify_fn` above is a
        # PROGRAMMATIC checker (a math answer-checker, a test runner) and is None for every task
        # type today, which left this path with no check at all. This is the model-based fallback:
        # weaker than execution feedback, but the alternative was keeping 100% of whatever the
        # teacher produced (B269).
        if rows and _verify_synth_enabled():
            rows = verify_generated_answers(
                rows, task_type=task_type, generate_fn=generate_fn, log=log,
            )
        if log:
            log(f"      [synth] requested {n} new-correct row(s) -> kept {len(rows)}")
        return rows
    return []
