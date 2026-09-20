# data/curriculum.py
import json
import logging
import os
import random
import re
from collections import Counter

from config.token_budget import output_budget, prompt_char_budget
from data.synth_client import synth_source_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoT annotation (paper §2.3 quality control #5, §2.5)
# The CoT teacher is the synth model, supplied as ``generate_fn`` — the local Qwen3.6, or
# DeepSeek under SLM_SYNTH_API_MODE=1, and only that model. There is no Claude CoT fallback.
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
                # LOCAL synth model (Qwen3.6-35B via vLLM) authors the CoT. Low temperature for
                # focused reasoning; the length is whatever the context allows, not a guess. At 512
                # a verbose math or code chain was cut mid-sentence and the PARTIAL reasoning was
                # attached to the row as training data, with nothing checking that it terminated.
                cot = (generate_fn(cot_prompt, 0.3, output_budget(cot_prompt)) or "").strip()
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
# Worked examples shown to the teacher, for BOTH generation and verification. Five, and not
# negotiable by accident: measured 2026-08-17, the teacher scores 0.1131 span-F1 zero-shot on BC5CDR
# and 0.7190 with five demonstrations — a 6.4x difference, with raw outputs showing exactly why
# (wrong casing, a class the task does not have, a markdown fence). Min et al. (arXiv:2202.12837)
# attribute that to demonstrations supplying "(1) the label space, (2) the distribution of the input
# text, and (3) the overall format of the sequence". A generator asked for output in a contract it
# has only been described, not shown, is being tested on guessing the contract (B276).
#
# Every generation and verification prompt in this module carries them, and any path that cannot
# assemble five says so in the log rather than quietly sending fewer.
SYNTH_SHOTS = int(os.environ.get("SLM_SYNTH_SHOTS", "5"))


# Verifier rejection reasons from the previous synthesis round, fed back into the next generation
# prompt. Module-level because generation and verification are separate calls in separate passes.
#
# WHY: on run 38708719 the rejections clustered hard on ONE systematic error — "includes unrequested
# optional parameter", "includes optional parameters with default values not implied by query" — which
# accounted for roughly half of them. The brief already says not to leave schema defaults in place, and
# the generator kept doing it anyway. A static instruction it demonstrably ignores is worth less than
# its own recent mistakes quoted back at it, which is the same reason the demonstrations are shown
# rather than described (B276).
_RECENT_REJECTIONS: list[str] = []
MAX_FEDBACK_REJECTIONS = 6

# A bounded sample of the ROWS the exact verifier refused, each carrying the reason it was refused
# for. Module-level for the same reason `_RECENT_REJECTIONS` is: generation happens deep inside a
# thread pool, and `agent.nodes.curate` — which owns the run's artifact directory — is the only place
# that can write them down.
#
# WHY THE REASON COUNTS ARE NOT ENOUGH
#     Run 38985393 generated 519 rows on toolbench and the exact verifier rejected all 519. Two
#     separate blind spots meant nothing survived to explain it: the task's verifier wrapper discarded
#     the reason string with a `[0]` subscript, and `_archive_synthetic_rows` returns early on an empty
#     list, so the audit file that exists to answer "why did these rows fail" is silent in exactly the
#     case where the question matters most. A count tells you WHICH check fired; the row tells you what
#     the teacher actually wrote, which is what you need to fix the prompt.
_RECENT_REJECTED_ROWS: list[dict] = []
# Enough to see a pattern, few enough that a 500-row wipeout does not write a 500-row artifact every
# iteration. The cap is checked from worker threads, so a race can let a couple extra rows through;
# that is harmless for a sample and not worth a lock on the hot path.
MAX_ARCHIVED_REJECTS = int(os.environ.get("SLM_MAX_ARCHIVED_REJECTS", "25"))


# Approximate characters per token for dense JSON. Deliberately conservative (real dense JSON runs
# nearer 3) so the budget errs large: over-asking costs nothing because generation stops at the EOS
# token, while under-asking truncates the row and loses it entirely.
_JSON_CHARS_PER_TOKEN = 4.0


def _row_output_budget(anchor: dict, prompt: str = "") -> int:
    """How many output tokens ONE generated row needs, measured from the anchor it must resemble.

    WHY THIS IS NOT A CONSTANT — this cost three runs and about 40 hours of GPU time
        This call site read `max_tokens=512`, hardcoded, for every task in the registry. That is
        ample for calendar (a row is a few hundred characters) and impossible for toolbench, whose
        row carries the full ReAct system prompt with its callable API list:

            toolbench row-as-JSON   min 3,913 chars (~978 tok)   median 9,997 (~2,499)   p90 14,678
            budget                  512 tokens
            rows that could fit     0 of 4,995

        So every toolbench generation was cut off mid-object, `json.loads` raised, and the row was
        dropped by a bare `except Exception: return None`. Runs 38832586, 38985393 and 39041380 each
        reported "0 kept" from 1,019, 133 and 425 attempts and were read as the VERIFIER rejecting
        everything — the log line even says "verification rejected EVERY generated row". Nothing was
        ever verified. Nothing was ever finished being generated.

        Shot count was never the variable. Zero-shot, one-shot and five-shot all produced exactly
        zero, because the limit that mattered was on the OUTPUT and none of them changed it.

    Sized from the anchor because the prompt asks the teacher to reproduce the anchor's schema with
    new values, so the anchor's own serialized length is a direct measurement of what the reply must
    contain rather than a guess. 1.5x plus a fixed margin, since a "genuinely new and diverse"
    instance is allowed to be somewhat longer than the row it was modelled on.

    Clamped to what the served context can actually return after the prompt is spent, so a large task
    asks for the most it can get instead of exceeding `max_model_len` and taking an HTTP 400.
    """
    schema = {k: v for k, v in anchor.items() if not str(k).startswith("_")}
    try:
        needed_chars = len(json.dumps(schema, ensure_ascii=False))
    except (TypeError, ValueError):
        needed_chars = 0
    return output_budget(prompt, needed_chars=needed_chars)


# Generation failures that happen BEFORE verification, counted by kind. Module-level for the same
# reason the rejection lists are: the failure occurs inside a thread pool and the report is assembled
# by the caller. See `_note_generation_failure`.
_GENERATION_FAILURES: list[tuple[str, str]] = []
MAX_GENERATION_FAILURE_SAMPLES = 400


def _note_generation_failure(kind: str, error: BaseException | None, raw: str = "") -> None:
    """Record a generation that never reached the verifier, and why.

    The bare `except Exception: return None` this replaces is why three separate runs were
    misdiagnosed. A row lost here is NOT a rejected row — it is a row that was never finished — and
    the two call for opposite fixes: raise the output budget versus change the generator prompt. The
    tail of the truncated reply is kept because an object that simply stops mid-string is the
    signature of hitting the token limit, and it is unmistakable once you can see it.
    """
    if len(_GENERATION_FAILURES) >= MAX_GENERATION_FAILURE_SAMPLES:
        return
    detail = f"{type(error).__name__}: {error}" if error is not None else ""
    if raw:
        detail += f" | reply was {len(raw)} chars ending: ...{raw[-90:]!r}"
    _GENERATION_FAILURES.append((kind, detail))


def take_generation_failures() -> list[tuple[str, str]]:
    """Hand over the generation-failure sample and clear it."""
    failures = list(_GENERATION_FAILURES)
    _GENERATION_FAILURES.clear()
    return failures


def note_rejected_row(row: dict, reason: str) -> None:
    """Keep a rejected row, with its reason, for the audit trail. Bounded; never raises."""
    if len(_RECENT_REJECTED_ROWS) >= MAX_ARCHIVED_REJECTS:
        return
    _RECENT_REJECTED_ROWS.append({"_reject_reason": " ".join(str(reason or "").split()), **row})


def take_rejected_rows() -> list[dict]:
    """Hand the rejected-row sample to the caller and clear it.

    Clear-on-read rather than clear-on-write: one rebuild calls synthesis once per targeted failure
    category, and clearing per call would keep only the last category's rejections — which on run
    38985393 would have thrown away the 386-row batch and kept the 28-row one.
    """
    rows = list(_RECENT_REJECTED_ROWS)
    _RECENT_REJECTED_ROWS.clear()
    return rows


def record_rejection_reasons(reasons: list[str]) -> None:
    """Remember why the verifier refused rows, so the next generation prompt can quote it."""
    seen: list[str] = []
    for reason in reasons:
        text = " ".join(str(reason or "").split())
        if text and text not in seen and "unavailable" not in text and "unparseable" not in text:
            seen.append(text)
    _RECENT_REJECTIONS[:] = seen[:MAX_FEDBACK_REJECTIONS]


def _rejection_feedback_block() -> str:
    """The previous round's rejection reasons, as concrete mistakes not to repeat."""
    if not _RECENT_REJECTIONS:
        return ""
    joined = "\n".join(f"  - {reason}" for reason in _RECENT_REJECTIONS)
    return (
        "The verifier REJECTED rows from your last batch for these specific reasons. They are your "
        "own recent mistakes on this exact task — do not repeat them:\n"
        f"{joined}\n\n"
    )


def _warn_short_shots(kind: str, got: int, log=None) -> None:
    """Say when a prompt went out with fewer demonstrations than the contract asks for."""
    if log and got < SYNTH_SHOTS:
        log(
            f"      [synth] {kind}: only {got} of {SYNTH_SHOTS} demonstration(s) available — the "
            "prompt is weaker than the measured five-shot configuration, so expect a lower keep "
            "rate on this batch"
        )


def _demo_block(demos: list[dict], task_description: str) -> str:
    """Worked examples in the exact JSON shape the teacher is about to be asked for."""
    import json

    parts = []
    for demo in demos:
        payload = {k: demo.get(k) for k in demo if not str(k).startswith("_")}
        parts.append(json.dumps(payload, ensure_ascii=False))
    return "\n\n".join(parts)


def _new_example_prompt(anchor: dict, task_description: str, demos: list[dict] | None = None,
                        target_category: str = "", task_context: str = "") -> str:
    """Prompt to generate ONE new, correct example in the anchor's exact schema.

    Generation-family only, which is every task with `closed_label_space=False`: calendar_json,
    xlam_bfcl, toolbench, gsm8k, dialogsum AND ner_bc5cdr. The other path (`_synthesize_new_gold`)
    never asks the teacher for a label — it copies the anchor's — so an out-of-vocabulary label is
    impossible there by construction and no label list needs stating.

    That other path used to be described here as "the classification/NER path", which was wrong and
    actively misleading: `ner_bc5cdr` declares `closed_label_space=False` and is dispatched to THIS
    function. Anyone tracing why BC5CDR synthesis behaved a certain way was sent to the wrong half of
    the module by a comment.

    `demos` are SHOWN, not described. See SYNTH_SHOTS for why that matters (B276).

    `task_context` is the SAME block the teacher gets when it verifies these rows, and it reached
    the verifier long before it reached here. That asymmetry is what the 2026-09-09 topv2 audit
    measured: the exact verifier rejected 62 of 98 generated rows, 33 of them for labels that do
    not exist — `SL:TIME` 12 times where TOPv2 says `DATE_TIME`, `IN:SET_ALARM` 10 times where it
    says `CREATE_ALARM`, `IN:SET_TIMER` 8 times where it says `CREATE_TIMER`. Every one is a
    near-miss synonym of a real label, which is exactly what a generator produces when it has to
    guess a closed vocabulary it was never shown. Holding a row to a convention while withholding
    the convention is a yield cliff that reads like a model quality problem.

    Five demonstrations cannot substitute. They exhibit at most a handful of the 166 names, and
    `tools_rule` below is this same argument already accepted for xLAM's function names — the fix
    there was to state the closed list rather than hope the schema implied it.
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
    # THE TOOLS ARE FIXED. Stated explicitly because "EXACTLY this JSON schema (same keys, same
    # value types)" below asks for the SHAPE, and a teacher reasonably reads that as licence to
    # invent its own tool list — which is what it did on run 39361648, collapsing 3,230 generated
    # calls onto 787 names with 45.8% of them in the top ten against gold's 6.1%.
    #
    # `_synthesize_new_correct` now overwrites `tools` with the anchor's after generation, so a call
    # to anything else is rejected by the exact verifier. Saying so here is what turns that from a
    # yield cliff into a constraint the teacher can actually satisfy.
    tools_rule = ""
    if anchor.get("tools"):
        try:
            names = [
                (t.get("function") if isinstance(t.get("function"), dict) else t).get("name")
                for t in anchor["tools"] if isinstance(t, dict)
            ]
            names = [n for n in names if n]
        except Exception:  # noqa: BLE001 — a malformed tools list must not break generation
            names = []
        if names:
            tools_rule = (
                "\nThe `tools` list above is FIXED and will be restored verbatim after you answer. "
                f"Your `answer` MUST call only these function(s): {', '.join(sorted(set(names)))}. "
                "Do NOT invent a different function name, and do NOT change any tool's parameter "
                "schema — a call to an undeclared function is rejected. Vary the REQUEST and the "
                "ARGUMENT VALUES instead, which is where the useful diversity is.\n"
            )
    # ORDER IS LOAD-BEARING FOR PREFIX CACHING (2026-08-30).
    #
    # Both the local vLLM server and the DeepSeek API cache a request's PREFIX, and a cache entry is
    # only usable by a later request that matches it from the very first token. So everything stable
    # must come before anything that changes.
    #
    #   task_description  — authored once at cold start, identical for the whole run
    #   shown (demos)     — `fit_demonstrations` is deterministic (shortest-first, no rng), so this
    #                       is identical for every row drawn from the same anchor pool
    #   rejection block   — the previous batch's verifier reasons, so it CHANGES between batches
    #   anchor schema     — changes on every single row
    #
    # The rejection block used to sit second, ahead of the demonstrations. That put a per-batch
    # string in front of ~1,800 tokens of stable demonstrations, so every new batch invalidated the
    # demos too and the cacheable prefix collapsed to just the task description. Moving it after the
    # demos lets the brief+demonstrations prefix survive across batches for the whole run.
    #
    # It also reads better where it now is: the rejections are instructions about the row being
    # asked for, and they now sit immediately before the request rather than before the examples.
    # Immediately after the brief and BEFORE the demos, which the ordering note above requires:
    # `task_context` is built from the spec and the anchor pool, so it is identical for every row
    # in the run and belongs in the cacheable prefix. Putting it after the demos would push a
    # stable string behind them for no benefit; putting it after the rejection block would put it
    # behind a per-batch one and invalidate the prefix on every batch.
    context_block = f"{task_context.strip()}\n\n" if task_context.strip() else ""
    return (
        f"{task_description}\n\n"
        f"{context_block}"
        f"{shown}"
        f"{_rejection_feedback_block()}"
        f"Generate ONE new, correct example in EXACTLY this JSON schema "
        f"(same keys, same value types): {json.dumps(schema, ensure_ascii=False)}."
        f"{tools_rule}"
        f"{aimed}"
        "\nIt must be a genuinely new, diverse, and CORRECT instance — not a copy or a "
        "paraphrase of the reference, and never a wrong answer. Vary the SUBSTANCE, not just the "
        "wording: a different scenario, different argument values, and a different number of calls "
        "where the task allows it. Reusing the reference's shape with new nouns adds a row the "
        "curriculum already effectively has. Obey the output contract above exactly; a well-formed "
        "answer that breaks a stated convention is graded wrong. Return only the JSON object, no "
        "preamble or code fences."
    )


# The teacher's description of the task is authored by the ORCHESTRATOR at cold start, from real
# rows, and reaches every prompt here as `task_description` (see agent/task_brief.py). It replaced a
# table keyed by task TYPE, under which `xlam_bfcl` and `calendar_json` were both described as
# "converting a request into a JSON function call using only the declared tools" — which omits every
# convention that makes a calendar row correct (the 60-minute default, ISO-8601, resolving against
# the reference instant), so a verifier asked to judge against it was judging its own guess (B269).


# Fields that are the QUESTION rather than context for judging it.
_INPUT_FIELDS = frozenset({"text", "prompt"})
# Fields that may carry the ANSWER. `_gold_field` narrows this per task; the set exists so that a row
# from a task whose gold lives elsewhere still has its answer kept out of the context block.
_ANSWER_FIELDS = frozenset({"answer", "response", "label", "entities"})


def _gold_field(spec) -> str:
    """The row key holding the gold answer for this task.

    Read off the spec's `required_fields`, which is `(input, gold)` for every task in the registry:
    `("text", "answer")` for xlam/gsm8k/calendar/dialogsum, `("text", "entities")` for BC5CDR,
    `("text", "label")` for the classification tasks.

    WHY THIS IS NOT JUST `row["answer"]`
        `verify_generated_answers` used to read `row.get("answer") or row.get("response")` and reject
        the row outright when both were blank. A synthesized BC5CDR row carries its gold in
        `entities` and has no `answer` at all, so EVERY span row was rejected with "empty answer"
        before a prompt was built — zero teacher calls, zero rows kept, 100% of the generation budget
        wasted. `surgical_synthesis` on `ner_bc5cdr` could not add a single row, while
        `docs/PIPELINE.md` documented the teacher pass as running for spans precisely because the
        substring verifier cannot catch a MISSED entity (B317).
    """
    required = tuple(getattr(spec, "required_fields", ()) or ())
    for field in required:
        if field not in _INPUT_FIELDS:
            return field
    return "answer"


def _render_gold(row: dict, gold_field: str) -> str:
    """The gold answer as text the verifier can read.

    Structured golds (a span list, a tool-call array) are rendered as JSON rather than `str()`, so the
    verifier sees the same shape the scorer parses instead of a Python repr with single quotes.
    """
    value = row.get(gold_field)
    if value is None and gold_field != "answer":
        value = row.get("answer") or row.get("response")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _task_context_block(spec, rows: list[dict] | None = None) -> str:
    """The TASK-level facts a verifier needs, for whichever of the eight tasks this is.

    Distinct from `_row_context_block`, which carries what varies per row (a function-calling row's
    own `tools`, and so on). This carries what is true of the whole task and therefore appears on no
    single row: the closed label space, what each class means, the entity-type vocabulary.

    WHY BOTH ARE REQUIRED
        A verifier judging without the row's context invents it (B314: it rejected calls for using
        tools "not present in the provided tools list" when no tools list was in the prompt). A
        verifier judging without the TASK's context does the same thing one level up — it decides
        what the label space must be from the label's wording, which is how a grade-school math
        problem was rejected from RouterBench's `local` class for being "a math problem, not a local
        query", when `local` means "route this to the local model" (B267/B269).

        So: always show the tools, always show the class list. Neither is optional and neither is
        inferable from the other.

    `rows` supplies the observed label space when the spec does not enumerate one itself, which is the
    case for every task here — the vocabulary is a property of the loaded data, not of the spec.
    """
    parts: list[str] = []
    # Task-level CONVENTIONS, before the label space. A closed vocabulary is not the only fact that
    # is true of the whole task and visible on no single row: calendar_json's datetime rules are
    # decidable and documented, and without them the teacher rejected 22.6% of generated rows on run
    # 38832587 — including "7pm start plus 60 mins is 8pm, not 20:00", which is the same time written
    # two ways. Those rows had already passed the EXACT programmatic verifier, so the teacher was
    # overruling a computation with a guess. See `TaskSpec.verifier_notes`.
    notes = str(getattr(spec, "verifier_notes", "") or "").strip()
    if notes:
        parts.append(notes)
    if spec is not None and getattr(spec, "closed_label_space", False):
        definitions = dict(getattr(spec, "label_definitions", None) or {})
        observed = sorted({
            str(row.get("label")) for row in (rows or [])
            if isinstance(row, dict) and row.get("label") is not None
        })
        labels = observed or sorted(definitions)
        if labels:
            parts.append(
                "This task has a CLOSED label space. These are the only valid classes, and a class "
                "listed here is valid by definition — never reject a row because you would not have "
                "chosen that class name:"
            )
            for label in labels:
                meaning = str(definitions.get(label, "")).strip()
                parts.append(f"  - {label}" + (f" — {meaning}" if meaning else ""))
            if not definitions:
                # Worth stating. 151 CLINC150 intents arrive with no definitions, and a verifier told
                # only the names will fall back to reading them as English words.
                parts.append(
                    "  (no written definitions are available for these classes: judge each row "
                    "against how the class is USED in the confirmed examples above, not against what "
                    "its name sounds like)"
                )
    # The task's OWN taxonomy, never one inferred from the rows in front of the teacher. Sampling
    # the shown rows is what this used to do, and it is only safe when every type is common enough
    # to appear: BC5CDR has two and both always do, MultiCoNER has 33 and a 40-row sample holds 5.
    # For MultiCoNER that made the sentence below a false statement about 28 real types, and on the
    # 2026-09-09 audit the teacher acted on it, rejecting correctly-typed `OtherLOC` rows. A sample
    # shows what a type space contains, never where it ends. See `TaskSpec.entity_type_vocabulary`.
    entity_types = list(getattr(spec, "entity_type_vocabulary", ()) or ())
    if entity_types:
        parts.append(
            "Valid entity types for this task, and the only ones that may appear: "
            + ", ".join(entity_types)
        )
    return ("\n".join(parts) + "\n\n") if parts else ""


def _row_context_block(row: dict) -> str:
    """The row's own task-specific fields, rendered for a verification prompt.

    WHY THIS EXISTS
        `verify_generated_answers` used to show the verifier only the request and the proposed answer.
        For a task whose correctness is defined by per-row context, that makes the question
        unanswerable, and the teacher answers it anyway by inventing the missing context.

        On xlam run 38661753 it rejected 79 of 330 rows (24%) with reasons like "Tool name
        'calculate_distance' is not present in the provided tools list" and "Tool names and arguments
        are invented and not from the provided tools list" — about a tools list that was never in the
        prompt. Every one of those rows had ALREADY passed the programmatic verifier, which checks the
        call's name and arguments against that row's own `tools` schema, so each rejection was
        provably false and each discarded a valid row. Others second-guessed real xLAM conventions
        ("argument key 'is_id' is likely incorrect; schema likely uses 'id'"), which is the same
        mistake in a subtler form: judging its own prior instead of the row (B267/B269/B314).

    Built by exclusion rather than from a per-task list of context fields, so a task that gains a
    field gets it shown automatically instead of silently omitting it — omission is the failure mode
    this function exists to prevent. Internal `_`-prefixed bookkeeping is excluded because it is
    provenance, not task content.
    """
    context = {
        key: value for key, value in row.items()
        if not key.startswith("_")
        and key not in _INPUT_FIELDS
        and key not in _ANSWER_FIELDS
        and value not in (None, "", [], {})
    }
    if not context:
        return ""
    rendered = json.dumps(context, ensure_ascii=False, sort_keys=True)
    # Bounded: a tools array can be several KB, and the verdict does not improve past the point where
    # the schema is legible. Truncation is announced so a clipped schema is never read as a short one.
    limit = 6000
    if len(rendered) > limit:
        rendered = rendered[:limit] + f"  … [truncated, {len(rendered)} chars total]"
    return (
        "Context provided WITH this example (the answer must be consistent with exactly this, and "
        "anything named here is valid by definition):\n"
        f"{rendered}\n\n"
    )


def verify_generated_answers(
    rows: list[dict],
    *,
    task_description: str,
    generate_fn,
    log=None,
    reference_rows: list[dict] | None = None,
    task_context: str = "",
    gold_field: str = "answer",
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
        from agent.teacher_fitness import fit_demonstrations
        from config.config import SYNTH_MAX_MODEL_LEN

        pairs = []
        # Same fitting rule as generation: show real examples, but only as many as the context has
        # room for. Each is additionally clipped to 400 characters below, so this bounds the count
        # while the clip bounds the size.
        _refs = fit_demonstrations(
            list(reference_rows), SYNTH_SHOTS, int(SYNTH_MAX_MODEL_LEN * 2.5 * 0.5)
        )
        _warn_short_shots("answer verification", len(_refs), log=log)
        for ref in _refs:
            # A share of the real context rather than a flat 400 characters. These pairs are what
            # tell the verifier the task's conventions, and on a long-row task the convention being
            # demonstrated fell off the end — leaving the verifier judging its own guess (B269).
            _clip = prompt_char_budget(0.10)
            q = " ".join(str(ref.get("text") or "").split())[:_clip]
            a = " ".join(_render_gold(ref, gold_field).split())[:_clip]
            if q and a:
                pairs.append(f"Request: {q}\nCorrect answer: {a}")
        if pairs:
            shown = ("Real, confirmed examples of this task:\n\n"
                     + "\n\n".join(pairs) + "\n\n")

    def _check(row: dict):
        request = str(row.get("text") or "")
        answer = _render_gold(row, gold_field)
        if not answer.strip():
            # A row with no gold at all cannot be verified OR trained on, so dropping it is right —
            # but it must be the row that is empty, not the field name that is wrong. See `_gold_field`.
            return row, False, f"empty answer (no {gold_field!r} on the row)"
        context = _row_context_block(row)
        prompt = (
            f"You are checking one training example for the task of {task_desc}.\n\n"
            f"{task_context}"
            f"{shown}"
            f"User input / question:\n{request}\n\n"
            f"{context}"
            f"Proposed answer:\n{answer}\n\n"
            f"Does the proposed answer correctly and directly satisfy the user's request, in the "
            f"context of {task_desc}? Answer strictly as JSON: "
            f'{{"valid": true|false, "reason": "<max 15 words>"}}. '
            f"Answer false if the answer is wrong, incomplete, in the wrong format for this task, "
            f"or does not address what was actually asked. Judge ONLY against the request and the "
            f"context above — if a name or field appears in the context, it is valid by definition, "
            f"and you must not reject the answer for using it or claim it was not provided."
        )
        try:
            # Was 160. The verdict is {valid, reason} with a free-text reason, and verification is
            # FAIL-OPEN: an unparseable reply keeps the row. So a verdict truncated mid-reason did
            # not make the verifier strict, it made it structurally unable to reject anything.
            raw = generate_fn(prompt, 0.0, output_budget(prompt))
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
    return _apply_verdicts(
        rows, results, kind="answer", mode=_verify_synth_mode(), log=log,
    )


def _synthesize_new_correct(
    examples: list[dict],
    *,
    task_description: str,
    n: int,
    generate_fn,
    verify_fn=None,
    log=None,
    target_category: str = "",
    task_context: str = "",
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
    # Warned once per batch, not once per row: this prompt is issued thousands of times.
    _shot_warned = {"new_correct": False}

    # Demonstrations are chosen to FIT the teacher's context rather than sampled at random. On a
    # task whose rows are large, five random full-prompt demonstrations exceed the served context and
    # every generation call 400s; dropping to zero-shot instead produced 0 kept rows out of 636, 326
    # and 57 attempts on toolbench. Shortest-first keeps five real demonstrations AND fits. See
    # `agent.teacher_fitness.fit_demonstrations`.
    from agent.teacher_fitness import fit_demonstrations
    from config.config import SYNTH_MAX_MODEL_LEN

    _demo_budget = int(SYNTH_MAX_MODEL_LEN * 2.5 * 0.5)

    def _one(anchor: dict) -> dict | None:
        demos = fit_demonstrations(anchors, SYNTH_SHOTS, _demo_budget, rng=rng)
        if not _shot_warned["new_correct"]:
            _shot_warned["new_correct"] = True
            _warn_short_shots("new-correct generation", len(demos), log=log)
        prompt = _new_example_prompt(anchor, task_description, demos=demos,
                                     target_category=target_category,
                                     task_context=task_context)
        try:
            raw = generate_fn(prompt, temperature=0.7,
                              max_tokens=_row_output_budget(anchor, prompt))
        except Exception as error:  # noqa: BLE001 — a failed generation is skipped, never fatal
            _note_generation_failure("endpoint error", error)
            return None
        try:
            row = json.loads(raw)
        except Exception as error:  # noqa: BLE001
            # Counted BY KIND rather than swallowed. An unterminated object here means the reply hit
            # the output limit mid-JSON, which is a budget problem and reads nothing like a verifier
            # disagreement — see `_row_output_budget` and `_note_generation_failure`.
            _note_generation_failure("unparseable JSON from the generator", error, raw=raw)
            return None
        if not isinstance(row, dict) or not row.get("text"):
            _note_generation_failure("generator returned no 'text' field", None, raw=raw)
            return None
        # PIN the constraint from the anchor rather than trusting the teacher to reproduce it. The
        # tool signature is what the call must satisfy, not something being invented — the same
        # "supply what you can, generate only what you must" principle that makes the
        # classification path safe. It also makes the row VERIFIABLE: a generated row with no
        # `tools` cannot be schema-checked at all.
        #
        # AUTHORITATIVE, not a fallback. This was `and pinned not in row`, so the anchor's tools
        # were used only when the teacher omitted them — and the teacher almost never omits them,
        # because the prompt hands it the anchor's whole JSON schema and asks for "EXACTLY this
        # JSON schema (same keys)". So the teacher invented its own `tools`, called a function from
        # that invented list, and `verify_function_call_row` passed it: the call IS declared, by the
        # row's own fabricated schema. Self-consistent and off-distribution.
        #
        # Measured on run 39361648, that produced a corpus collapse. Gold draws 1,158 distinct
        # function names over 2,536 calls with the top ten accounting for 6.1% of them; the
        # generated rows managed 787 names over 3,230 calls with the top ten at 45.8% — 356
        # `convert_currency`, 347 `get_crypto_price`, 302 `get_weather`. xLAM/BFCL is a long-tail
        # benchmark whose whole question is whether a model can call an UNFAMILIAR API from a
        # declared schema, so training on a handful of invented generic ones teaches a distribution
        # the eval does not measure.
        #
        # It also produced contradictions. Two kept rows carried the identical request "What are
        # the current prices of Bitcoin and Ethereum?" against the same invented `get_crypto_price`
        # with INCOMPATIBLE arguments — `{"coin_id": "bitcoin"}` in one and `{"symbol": "BTC"}` in
        # the other. Nothing could catch that, because each row was internally consistent with the
        # schema it had invented for itself.
        #
        # Overwriting means a generated row must satisfy the anchor's REAL tool signature or be
        # rejected by the exact verifier, and anchors are drawn round-robin across the curriculum —
        # so the synthetic distribution inherits gold's diversity instead of the teacher's priors.
        # Expect a lower keep rate in exchange: rows that ignore the anchor's schema now fail, and
        # their reasons feed back into the next batch's prompt via `_RECENT_REJECTIONS`.
        for pinned in ("tools", "_instruction"):
            if pinned in anchor:
                row[pinned] = anchor[pinned]
        # STAGE 1 — exact, programmatic check. Runs BEFORE any model-based verification, because it
        # is free, cannot be fooled, and a row it rejects should never cost a teacher call.
        if verify_fn is not None and not verify_fn(row):
            if checker is not None:
                _ok, reason = checker(row)
                rejections.append(reason)
                note_rejected_row(row, reason)
            return None
        row["_source"] = "synth:generated"
        # WHICH teacher wrote it, kept separate from `_source` (which records the synthesis KIND).
        # Under API mode this is the only place a row records that it came from a model we do not
        # own, and the audit archive copies every field, so it survives into the evidence file.
        row["_teacher"] = synth_source_label()
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
    # ASK FOR WHAT WE NEED, THEN TOP UP ONCE. Two rounds, never more.
    #
    # This used to fire `max(n * 2, n + 32)` attempts unconditionally and trim to n, so a batch of
    # 175 rows cost 350 generations: 19 failed and 156 perfectly good rows were thrown away because
    # the quota was already met. On a paid endpoint that is double the bill to insure against a
    # failure rate that measured 5%.
    #
    # Round 1 asks for exactly n. Round 2 asks for exactly the shortfall, which is the number of
    # rows that actually failed rather than a guess at how many might. If round 2 also comes up
    # short the batch is delivered short — deliberately. A third round would be paying repeatedly
    # for whatever is systematically broken, and `run_health` already watches for that: a batch
    # that returns nothing counts toward MAX_CONSECUTIVE_EMPTY_SYNTHESIS.
    #
    # Anchor rotation CONTINUES across the two rounds rather than restarting, so the retry draws
    # different anchors than the attempt it is replacing. Retrying the same anchor with the same
    # prompt is the one thing least likely to produce a different outcome.
    def _round(count: int, offset: int) -> list[dict]:
        if count <= 0:
            return []
        batch = [anchors[(offset + i) % len(anchors)] for i in range(count)]
        produced = _progress_map(
            _one,
            batch,
            label="new-correct synthesis",
            log=log,
            workers=_synth_concurrency(len(batch)),
        )
        return [row for row in produced if row is not None]

    out = _round(n, 0)
    attempts = n
    shortfall = n - len(out)
    if shortfall > 0:
        if log:
            log(f"      [synth] {shortfall} of {n} generation(s) produced no usable row; "
                f"retrying exactly that many once")
        out.extend(_round(shortfall, n))
        attempts += shortfall
        out = out[:n]
    if log:
        shots = f"{SYNTH_SHOTS}-shot" if SYNTH_SHOTS else "zero-shot"
        log(f"      [synth] new-correct ({shots}): {len(out)}/{n} kept "
            f"({attempts} attempts)")
        # Reported BEFORE the verifier breakdown and separately from it, because they answer a
        # different question. A row counted here never reached the verifier at all, so reading these
        # as rejections points the fix at the generator prompt when the cause may be the output
        # budget. Three runs were misread that way; see `_row_output_budget`.
        failures = take_generation_failures()
        if failures:
            by_kind = Counter(kind for kind, _detail in failures)
            log(f"      [generate:failed] {len(failures)} generation(s) never produced a usable row "
                f"and were NEVER VERIFIED — this is not a verifier rejection:")
            for kind, count in by_kind.most_common(_VERIFY_LOG_LIMIT):
                example = next(d for k, d in failures if k == kind)
                log(f"        x{count}  {kind}")
                if example:
                    log(f"                {example[:260]}")
            if by_kind.get("unparseable JSON from the generator"):
                log("        ^ an object that stops mid-string means the reply hit the output token "
                    "limit. Check the max_tokens `_row_output_budget` computed against the size of "
                    "this task's rows.")
        if rejections:
            counts = Counter(rejections)
            log(f"      [verify:exact] programmatic verifier rejected {len(rejections)} row(s) "
                f"before any teacher call:")
            for reason, count in counts.most_common(_VERIFY_LOG_LIMIT):
                log(f"        x{count}  {reason}")
    return out


# How many rejected rows to quote in the log before summarising the rest.
_VERIFY_LOG_LIMIT = int(os.environ.get("SLM_VERIFY_LOG_LIMIT", "10"))


# Announced once per process, not once per batch: `synthesize_examples` is called once per targeted
# failure category and a rebuild targets up to five of them, so a per-call notice would repeat the
# same paragraph five times per rebuild.
_VERIFY_DISABLED_ANNOUNCED = [False]


VERIFY_MODE_OFF = "off"
VERIFY_MODE_ENFORCE = "enforce"
VERIFY_MODE_SHADOW = "shadow"


def _verify_synth_mode(log=None) -> str:
    """Whether the teacher re-reads its own generated rows, and whether its vote BINDS.

    Three modes, because there are three genuinely different things to want:

      `enforce`  run the pass and drop what it rejects. The default on the local teacher.
      `off`      do not run it. The default in API mode, where it costs one paid call per row —
                 about a quarter of synthesis spend on the measured runs — to ask a model whether
                 it agrees with itself, which is the weakest signal in the pipeline.
      `shadow`   RUN IT AND KEEP EVERY ROW ANYWAY, recording what it would have rejected and why.

    WHY SHADOW EXISTS
        `enforce` has been measured doing net harm. On run 38832588 the teacher rejected 25 of 25
        `ner_bc5cdr` rows that `verify_ner_row` had just accepted, twice running, and that stopped
        the run; on `calendar_json` it rejected 1,965 of 8,704 (22.6%) including "7pm start plus 60
        mins is 8pm, not 20:00" — 8pm IS 20:00. So both tasks ship with it off, and the note in
        their launchers says the pass should be re-measured once `verifier_notes` exists to tell
        the teacher the conventions it was missing.

        Turning it back on to find out is the expensive way to ask: a false-rejection cascade does
        not just discard good rows, it can end a multi-day run. Shadow mode is the cheap way. The
        rejection rate and the reasons are exactly the measurement, and the trajectory is
        unaffected because nothing is dropped.

    SHADOW MUST NOT FEED ITS REASONS BACK, and that is the subtle part. `record_rejection_reasons`
    injects the last round's rejections into the NEXT generation prompt as mistakes not to repeat.
    In shadow mode that would change what the teacher generates, so the pass would be altering the
    run it is supposed to be passively measuring — and the measured rejection rate would be of a
    trajectory that only exists because we were measuring it. The callers therefore skip the
    feedback recorder in shadow mode.

    `SLM_VERIFY_SYNTH` selects explicitly and always wins over the mode default: `1`/`enforce`,
    `0`/`off`, or `shadow`.
    """
    configured = (os.environ.get("SLM_VERIFY_SYNTH") or "").strip().lower()
    if configured:
        if configured in ("shadow", "dry-run", "dryrun"):
            return VERIFY_MODE_SHADOW
        return VERIFY_MODE_ENFORCE if configured == "1" else VERIFY_MODE_OFF
    try:
        from config.config import SYNTH_API_MODE
    except Exception:  # noqa: BLE001 — config may be unimportable in a bare unit test
        return VERIFY_MODE_ENFORCE
    if not SYNTH_API_MODE:
        return VERIFY_MODE_ENFORCE
    if log and not _VERIFY_DISABLED_ANNOUNCED[0]:
        _VERIFY_DISABLED_ANNOUNCED[0] = True
        log("      [verify] teacher self-check SKIPPED in API mode — it costs one paid call per "
            "generated row to ask the teacher whether it agrees with itself. The exact "
            "programmatic verifiers still run. Set SLM_VERIFY_SYNTH=1 to re-enable it.")
    return VERIFY_MODE_OFF


def _verify_synth_enabled(log=None) -> bool:
    """True when the teacher pass RUNS at all, in either `enforce` or `shadow` mode."""
    return _verify_synth_mode(log=log) != VERIFY_MODE_OFF


def _apply_verdicts(rows: list[dict], results, *, kind: str, mode: str, log=None,
                    describe=None) -> list[dict]:
    """Keep or drop rows per the teacher's verdicts, or keep everything and just report.

    One implementation for both the label and the answer pass, so the two cannot disagree about
    what shadow mode means — which is the kind of drift that would make a measurement taken on one
    task inapplicable to the other.
    """
    kept, rejected = [], []
    for row, valid, reason in results:
        (kept if valid else rejected).append((row, reason))

    shadow = mode == VERIFY_MODE_SHADOW
    if not shadow:
        # Only in enforce mode. See `_verify_synth_mode`: feeding these back would make the
        # shadow pass change the trajectory it is measuring.
        record_rejection_reasons([reason for _row, reason in rejected])

    if log:
        total = len(rows)
        if shadow:
            share = (100.0 * len(rejected) / total) if total else 0.0
            # `[verify:shadow]` is the grep handle for the measurement. Deliberately distinct from
            # the enforce path's `[verify]` so a log cannot be misread as rows having been dropped.
            log(f"      [verify:shadow] teacher would have rejected {len(rejected)}/{total} "
                f"generated {kind}(s) ({share:.1f}%) — ALL {total} KEPT, nothing was dropped")
        else:
            log(f"      [verify] teacher validated {len(kept)}/{total} generated {kind}(s); "
                f"rejected {len(rejected)}")
        marker = "WOULD REJECT" if shadow else "REJECTED"
        for row, reason in rejected[:_VERIFY_LOG_LIMIT]:
            label = describe(row) if describe else ""
            text = " ".join(str(row.get("text") or "").split())[:70]
            log(f"        {marker} {label}{text!r} — teacher: {reason}")
        if len(rejected) > _VERIFY_LOG_LIMIT:
            # "rejected" / "would be rejected" rather than a bare "more": a rebuild can reject
            # thousands of rows and this line is what a reader sees instead of all of them, so it
            # has to say which of the two things happened on its own.
            tail = "would be rejected" if shadow else "rejected"
            log(f"        ... and {len(rejected) - _VERIFY_LOG_LIMIT} more {tail}")
        if shadow and rejected:
            # The reason HISTOGRAM is the actual deliverable. A list of 25 individually logged
            # rejections does not answer "is this one systematic misunderstanding or 25 different
            # ones", and that distinction decides whether `verifier_notes` can fix it.
            counts: dict[str, int] = {}
            for _row, reason in rejected:
                key = " ".join(str(reason or "").split())[:60]
                counts[key] = counts.get(key, 0) + 1
            top = sorted(counts.items(), key=lambda item: -item[1])[:5]
            log(f"      [verify:shadow] would-reject reasons by frequency: {dict(top)}")

    return rows if shadow else [row for row, _ in kept]


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
    task_description: str,
    generate_fn,
    log=None,
    label_definitions: dict | None = None,
    all_labels: list[str] | None = None,
    reference_rows: list[dict] | None = None,
    task_context: str = "",
) -> list[dict]:
    """Ask the teacher model to confirm each generated row really belongs to its assigned label.

    Generation and verification are NOT the same task. Writing "an utterance that belongs to
    class X" is open-ended; deciding "does this utterance belong to class X, yes or no" is the
    classification task the reference model is already good at. So a self-check is cheap and
    meaningfully better than nothing, even though it uses the same model.

    `task_description` is the orchestrator-authored brief (what the benchmark is, the exact output
    contract, the likely failure modes) and `label_definitions` says what each class MEANS. Both are
    REQUIRED context, not decoration: without them the teacher judges the label WORD instead of the
    task, which is how a grade-school math problem was rejected for the `local` class because "the
    utterance is a math problem, not a local query" — 70% of generated rows discarded for the wrong
    reason (B267/B269).

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

    _verify_shot_warned: set[str] = set()

    def _check(row: dict):
        label = str(row.get("label"))
        text = str(row.get("text") or "")
        context = _label_context_block(label, label_definitions, all_labels)
        examples = by_label.get(label, [])[:SYNTH_SHOTS]
        if label not in _verify_shot_warned:
            _verify_shot_warned.add(label)
            _warn_short_shots(f"label verification for {label!r}", len(examples), log=log)
        shown = ""
        if examples:
            joined = "\n".join(f"- {e}" for e in examples)
            shown = f"Real, confirmed examples of the '{label}' class:\n{joined}\n\n"
        prompt = (
            f"{task_description}\n\n"
            f"{task_context}"
            f"You are checking one training example for a text classifier.\n"
            f"{context}\n"
            f"{shown}"
            f"Utterance: {text}\n"
            f"{_row_context_block(row)}"
            f"Proposed label: {label}\n\n"
            f"Does this utterance genuinely belong to the '{label}' class? Answer strictly as "
            f'JSON: {{"valid": true|false, "reason": "<max 15 words>"}}. '
            f"Answer false if the utterance actually belongs to a different class, is "
            f"incoherent, or mixes two intents. Do NOT answer false merely because the "
            f"utterance's TOPIC is unrelated to the label's wording — judge only whether the "
            f"class, as defined above, applies."
        )
        try:
            # Was 120 — the same fail-open truncation as the answer verifier above, 40 tokens tighter.
            raw = generate_fn(prompt, 0.0, output_budget(prompt))
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
    return _apply_verdicts(
        rows, results, kind="row", mode=_verify_synth_mode(), log=log,
        describe=lambda row: f"[{row.get('label')}] ",
    )


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
    # Warned once per LABEL: a rare class legitimately has fewer examples than a common one, and that
    # is exactly the case worth seeing in the log.
    _gold_shot_warned: set[str] = set()
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
        if label not in _gold_shot_warned:
            _gold_shot_warned.add(label)
            _warn_short_shots(f"in-class generation for {label!r}", len(demos), log=log)
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
            # Was 200 (~800 characters). Ample for the classification utterances and BC5CDR
            # sentences measured today, and silently lossy for any future task with longer inputs —
            # which is the whole argument against picking the number by hand.
            text = generate_fn(prompt, 1.0, output_budget(prompt))
        except Exception:  # noqa: BLE001 — a failed generation is skipped, never fatal
            return None
        text = str(text or "").strip()
        if not text:
            return None
        return {
            "text": text,
            "label": anchor.get("label"),
            "_source": "synth",
            "_teacher": synth_source_label(),
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
        if rows and _verify_synth_enabled(log=log):
            rows = verify_generated_labels(
                rows,
                task_description=_describe(brief, spec),
                generate_fn=generate_fn,
                log=log,
                label_definitions=label_definitions,
                all_labels=sorted({
                    str(row.get("label")) for row in examples
                    if isinstance(row, dict) and row.get("label") is not None
                }) or None,
                reference_rows=examples,
                task_context=_task_context_block(spec, examples),
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
        task_context=_task_context_block(spec, examples),
    )
    # Teacher answer-verification. The programmatic verifier above is exact but only checks FORM;
    # this asks whether the answer is actually right. Weaker than execution feedback, but the
    # alternative was keeping 100% of whatever the teacher produced (B269).
    if rows and _verify_synth_enabled(log=log):
        rows = verify_generated_answers(
            rows,
            task_description=_describe(brief, spec),
            task_context=_task_context_block(spec, examples),
            gold_field=_gold_field(spec),
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
