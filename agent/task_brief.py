"""The orchestrator's own description of the benchmark it is about to work on.

WHY THIS EXISTS
    Every teacher prompt in the pipeline — generate a row, judge whether a generated row is correct
    — needs to tell the teacher what the task IS. That text used to come from a hardcoded table
    keyed by task type (`data/curriculum.py::_TASK_DESCRIPTIONS`), which produced two failures:

      * It could not distinguish two tasks sharing a type. `xlam_bfcl` and `calendar_json` were both
        described as "converting a request into a JSON function call using only the declared tools",
        which omits every convention that makes a calendar row correct — the 60-minute default, ISO
        8601, resolving relative dates against the request's own reference instant. A verifier asked
        to judge against that description was judging its own guess, and calendar synthesis scored
        0.2176 (B269).
      * It was one line. A teacher asked "is this answer correct?" with one line of context and no
        worked example is being tested on inferring the contract, which is exactly what Min et al.
        (arXiv:2202.12837) identify as the thing demonstrations supply.

    So the orchestrator writes it, once, at cold start, having been shown real rows from the
    dataset that was actually loaded. It is stored on the run state, logged in full, and every
    synthesis and verification prompt is built from it.

WHAT IT IS NOT
    It is not a source of examples. The worked examples shown to the teacher are always REAL rows
    sampled from the task's own training split — an orchestrator-invented example could be wrong,
    and a wrong example is worse than none. The brief supplies prose: what the task is, what a
    correct answer must look like, and what tends to go wrong.
"""
from __future__ import annotations

import json
import re

# The brief is prose the teacher reads, so it is bounded to keep synthesis prompts affordable:
# these prompts are issued once per generated row, thousands of times per rebuild.
SUMMARY_MAX_CHARS = 700
CONTRACT_MAX_CHARS = 700
MAX_FAILURE_MODES = 6
FAILURE_MODE_MAX_CHARS = 200

_BRIEF_PROMPT = """You are briefing a TEACHER MODEL that will (a) generate new training rows for a \
benchmark and (b) judge whether generated rows are correct. Everything you write here is inserted \
verbatim into those prompts, so it must be precise and self-contained.

Benchmark: {title}
Internal name: {name}
Category: {category}
Scoring metric: {metric}
How a prediction is graded: {grading}

Here are {n_shown} REAL rows from this benchmark's own training split, exactly as the pipeline \
stores them:

{samples}

Reply with STRICT JSON, no prose and no code fences:

{{
  "summary": "What this benchmark is and what the model must do. Name the domain and the input. \
2-4 sentences.",
  "output_contract": "Exactly what a correct answer looks like, including every formatting rule a \
grader would enforce. Be concrete: name fields, types, units, conventions, and anything that must \
be derived from the input rather than invented. 2-5 sentences.",
  "failure_modes": ["Short phrases naming the mistakes most likely to make an answer wrong here. \
3-6 of them."]
}}

Write the output_contract from the ROWS, not from the benchmark's reputation. If the rows show a \
convention the name would not tell you — a default duration, a date resolved against a reference \
instant, an answer marker, a fixed tool name — state it, because a teacher that does not know it \
will produce plausible rows that are graded wrong."""


def _clip(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        raise ValueError("orchestrator returned no JSON object")
    return json.loads(match.group())


def _sample_block(rows: list[dict], n: int) -> str:
    """Real rows, rendered as the JSON the pipeline actually stores, minus private fields."""
    shown = []
    for row in rows[:n]:
        public = {
            key: value for key, value in row.items()
            if not str(key).startswith("_")
        }
        shown.append(json.dumps(public, ensure_ascii=False, default=str)[:1200])
    return "\n\n".join(shown)


def fallback_brief(spec) -> dict:
    """A minimal brief built from the spec alone, for when the orchestrator is unreachable.

    Honest rather than helpful: it says the contract is unknown, so a reader of the log can see
    that synthesis ran without a real brief instead of assuming the prose came from the model.
    """
    return {
        "summary": f"{spec.title} ({spec.name}). Scored by {spec.metric_name}.",
        "output_contract": (
            "UNAVAILABLE — the orchestrator could not be reached, so no output contract was "
            "authored. Generated rows are shaped by the real anchor rows alone."
        ),
        "failure_modes": [],
        "source": "fallback",
    }


def build_task_brief(spec, train_rows: list[dict], *, n_shown: int = 5, log=print) -> dict:
    """Ask the orchestrator to describe this benchmark, and log what it said.

    Never fatal: an unreachable orchestrator degrades to `fallback_brief`, because a run that can
    train and score should not be stopped by a missing prose description.
    """
    from agent.cost import tracked_anthropic_messages_create
    from agent.llm_text import MIN_THINKING_SAFE_MAX_TOKENS, response_text
    from config.config import (
        ANTHROPIC_API_KEY,
        ORCHESTRATOR_MODEL,
        orchestrator_client_kwargs,
    )

    grading = _grading_description(spec)
    prompt = _BRIEF_PROMPT.format(
        title=spec.title,
        name=spec.name,
        category=spec.category,
        metric=spec.metric_name,
        grading=grading,
        n_shown=min(n_shown, len(train_rows)),
        samples=_sample_block(train_rows, n_shown),
    )
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs()
        )
        resp = tracked_anthropic_messages_create(
            client.messages,
            stage="task_brief",
            model=ORCHESTRATOR_MODEL,
            max_tokens=max(1024, MIN_THINKING_SAFE_MAX_TOKENS),
            messages=[{"role": "user", "content": prompt}],
        )
        payload = _extract_json(response_text(resp))
    except Exception as error:  # noqa: BLE001 — a missing brief must not stop a runnable run
        log(f"      [brief] orchestrator brief FAILED ({type(error).__name__}: "
            f"{error}); falling back to a spec-only description")
        return fallback_brief(spec)

    brief = {
        "summary": _clip(payload.get("summary"), SUMMARY_MAX_CHARS),
        "output_contract": _clip(payload.get("output_contract"), CONTRACT_MAX_CHARS),
        "failure_modes": [
            _clip(mode, FAILURE_MODE_MAX_CHARS)
            for mode in (payload.get("failure_modes") or [])[:MAX_FAILURE_MODES]
            if str(mode or "").strip()
        ],
        "source": "orchestrator",
    }
    if not brief["summary"] or not brief["output_contract"]:
        log("      [brief] orchestrator reply was missing summary or output_contract; "
            "falling back to a spec-only description")
        return fallback_brief(spec)
    log_task_brief(brief, spec, log=log)
    return brief


def _grading_description(spec) -> str:
    """How this task's scorer decides a row is right, in one line, for the brief prompt."""
    parts = [f"the comparison scalar is {spec.metric_name}"]
    if spec.closed_label_space:
        parts.append(
            "the prediction must be one of a fixed set of class labels, extracted from the "
            "model's reply"
        )
    if spec.synth_verifier is not None:
        parts.append(
            "generated rows additionally pass an exact programmatic well-formedness check"
        )
    return "; ".join(parts)


def log_task_brief(brief: dict, spec, *, log=print) -> None:
    """Print the brief in full. It is the text every teacher prompt is built from, so a run whose
    synthesis behaved oddly needs it in the log rather than only in the artifacts."""
    log(f"      [brief] task description authored by the orchestrator for {spec.name} "
        f"(source={brief.get('source')}):")
    log(f"      [brief]   summary: {brief.get('summary')}")
    log(f"      [brief]   output contract: {brief.get('output_contract')}")
    modes = brief.get("failure_modes") or []
    if modes:
        log(f"      [brief]   expected failure modes ({len(modes)}):")
        for mode in modes:
            log(f"      [brief]     - {mode}")
    else:
        log("      [brief]   expected failure modes: none stated")


def brief_context_block(brief: dict | None, spec) -> str:
    """The brief rendered for insertion into a synthesis or verification prompt."""
    brief = brief or fallback_brief(spec)
    lines = [
        f"TASK: {brief.get('summary') or spec.title}",
        f"OUTPUT CONTRACT: {brief.get('output_contract') or 'not stated'}",
    ]
    modes = brief.get("failure_modes") or []
    if modes:
        lines.append("COMMON MISTAKES TO AVOID: " + "; ".join(modes))
    return "\n".join(lines)
