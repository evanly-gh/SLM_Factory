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


def function_call_turn(row: dict, _ctx: TrainingContext) -> tuple[str, str, str]:
    from eval.scorers.function_call import build_function_call_prompt

    answer = row.get("answer", "")
    if not str(answer).strip():
        raise ValueError("function-call training row has an empty 'answer'; nothing to learn")
    return str(build_function_call_prompt(row)), str(answer), "Answer"
