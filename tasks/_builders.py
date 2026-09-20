"""Shared building blocks the task modules compose.

Nothing here decides anything. Each function is a reusable implementation that a `TaskSpec` must
name explicitly — two tasks pointing at the same builder is fine and expected (xlam and calendar
both use `function_call_turn`), but neither of them gets it by falling through a branch.

The training-turn builders import their prompt from the EVAL scorer rather than reproducing it.
That is not tidiness: when the two were written separately the model was fine-tuned on one input
shape and scored on another (B250 for generation, and again for NER, where the training copy had
quietly dropped the "Reply with [] if there are no entities" sentence). Importing makes them
incapable of drifting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingContext:
    """Dataset-level context a training turn may need beyond the row itself."""

    labels: tuple[str, ...]
    """Sorted class vocabulary. Empty for tasks with no closed label space."""
    instruction: str
    """The dataset's shared generation instruction. Empty for tasks that do not use one."""


def classification_turn(row: dict, ctx: TrainingContext) -> tuple[str, str, str]:
    from eval.scorers.classification import build_classify_prompt

    return (
        build_classify_prompt(row.get("text", ""), ctx.labels),
        str(row.get("label", "")),
        "Answer",
    )


def ner_turn(row: dict, _ctx: TrainingContext) -> tuple[str, str, str]:
    from eval.scorers.ner import NER_PROMPT

    return (
        NER_PROMPT.format(text=row["text"]),
        json.dumps(row.get("entities", [])),
        "Entities",
    )


def multilabel_emotion_turn(row: dict, _ctx: TrainingContext) -> tuple[str, str, str]:
    """One GoEmotions row as `(prompt, comma-joined labels, marker)`.

    Not `classification_turn`: that builds a single-label prompt from `ctx.labels` and targets one
    label word. This task is multi-label, and its 28-name vocabulary is pinned in the loader
    rather than read off the eval set.
    """
    from eval.scorers.multilabel_emotion import build_prompt

    target = row.get("label") or ", ".join(row.get("labels") or [])
    if not str(target).strip():
        raise ValueError("emotion training row has no labels; nothing to learn")
    return build_prompt(row.get("text", "")), str(target), "Answer"


def semantic_parse_turn(row: dict, _ctx: TrainingContext) -> tuple[str, str, str]:
    """One TOPv2 row as `(prompt, parse string, marker)`.

    The target is the corpus's `semantic_parse` VERBATIM. Re-serializing it here — even
    normalizing bracket spacing — would train the model toward a string the scorer then compares
    against the unmodified gold, so exact match would penalize the model for obeying us.
    """
    from eval.scorers.semantic_parse import build_prompt

    answer = row.get("answer", "")
    if not str(answer).strip():
        raise ValueError("semantic-parse training row has an empty 'answer'; nothing to learn")
    return build_prompt(row.get("text", "")), str(answer), "Parse"


def fine_ner_turn(row: dict, _ctx: TrainingContext) -> tuple[str, str, str]:
    """One MultiCoNER row as `(prompt, JSON span list, marker)`.

    Separate from `ner_turn` because the prompt is: this one enumerates all 33 type names, which
    the model cannot guess and which the scorer then compares exactly.
    """
    from eval.scorers.fine_ner import build_prompt

    return (
        build_prompt(row.get("text", "")),
        json.dumps(row.get("entities", [])),
        "Entities",
    )


def generation_turn(row: dict, ctx: TrainingContext) -> tuple[str, str, str]:
    """Free-form answer, optionally preceded by a chain-of-thought block."""
    from eval.scorers.generation import build_generation_prompt

    prompt = build_generation_prompt(
        row.get("text", row.get("prompt", "")), ctx.instruction
    )
    answer = row.get("answer", row.get("response", row.get("label", "")))
    cot = row.get("cot_reasoning", "")
    target = f"<reasoning>\n{cot}\n</reasoning>\n\n{answer}" if cot else answer
    return str(prompt), str(target), "Answer"


def summarization_turn(row: dict, ctx: TrainingContext) -> tuple[str, str, str]:
    """One DialogSum row as `(prompt, reference summary, marker)`.

    The target is `references[0]` — one human summary, not all three. Training on three targets
    for one dialogue would teach the model to average three people's phrasing; the three exist to
    make SCORING fair, not to triple the training signal. Train and dev are single-reference
    anyway, so this is only a choice on the test split, which is never trained on.
    """
    from eval.scorers.summarization import build_summarization_prompt, resolve_instruction

    references = row.get("references") or []
    target = str(references[0] if references else row.get("answer", "")).strip()
    if not target:
        raise ValueError("summarization training row has no reference summary; nothing to learn")
    instruction = ctx.instruction or resolve_instruction([row])
    return build_summarization_prompt(row.get("text", ""), instruction), target, "Summary"


def gec_turn(row: dict, _ctx: TrainingContext) -> tuple[str, str, str]:
    """One GEC row as `(prompt, corrected sentence, marker)`.

    The target is the TOKENIZED corrected sentence, exactly as the M2 file yields it. ERRANT reads
    tokenized text, so detokenizing the target here would train the model to emit a string the
    scorer then has to align against tokenized gold — misaligning every edit for reasons that have
    nothing to do with grammar.
    """
    from eval.scorers.gec import GEC_PROMPT

    answer = row.get("answer", "")
    if not str(answer).strip():
        raise ValueError("gec training row has an empty 'answer'; nothing to learn")
    return GEC_PROMPT.format(text=row.get("text", "")), str(answer), "Correction"


def function_call_turn(row: dict, _ctx: TrainingContext) -> tuple[str, str, str]:
    from eval.scorers.function_call import build_function_call_prompt

    answer = row.get("answer", "")
    if not str(answer).strip():
        raise ValueError("function-call training row has an empty 'answer'; nothing to learn")
    return str(build_function_call_prompt(row)), str(answer), "Answer"


def toolbench_turn(row: dict, _ctx: TrainingContext) -> tuple[str, str, str]:
    """One ToolBench row as `(prompt, whole solution path, marker)`.

    The target is the ENTIRE path — every `Thought` / `Action` / `Action Input` turn through the
    terminating `Finish` — and not the next action alone, because that is the unit the eval asks
    for: with no API server there are no observations to feed back, so the model is scored on a
    path it produces in one generation. Training on single next actions and evaluating on whole
    paths would be B290 with extra steps.
    """
    from eval.scorers.toolbench import build_toolbench_prompt

    answer = row.get("answer", "")
    if not str(answer).strip():
        raise ValueError("toolbench training row has an empty 'answer'; nothing to learn")
    return str(build_toolbench_prompt(row)), str(answer), "Answer"
