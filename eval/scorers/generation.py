"""LLM-as-a-judge scoring for free-form tasks: prompt building, answer extraction, judge scoring.

This module and `eval/judge_client.py` together are the pipeline's LLM-as-a-judge capability. A task
whose output has no checkable answer (open-ended text judged for quality) sets `needs_judge=True` and
names `build_prompts` / `extract_predictions` / `score_with_judge` from here in its `TaskSpec`. The
local teacher then scores each answer 0-1 against the reference. None of the ten paper tasks uses it:
all of them have exact or reference-based metrics. `split_reasoning` and
`resolve_generation_instruction` below are also used by the live tasks (CoT stripping, row instructions).

History: this module used to serve three `task_type` channels — `generation`, `math_reasoning` and
`code_generation` — and branched on `eval_set.task_type` inside `build_prompts`,
`extract_predictions` and `score`. A task got whichever branch its channel happened to land in,
and the `else` was the LLM judge, so anything routed here without a matching branch was silently
judged rather than checked.

There is now one named scoring function and no branch: a task whose quality is a judgement names
`score_with_judge` in its `TaskSpec`. The numeric exact-match scorer went with the gsm8k task
(removed 2026-11-11). The `code_generation` channel and its execution sandbox were deleted on
2026-08-18 — APPS/MBPP are not in the benchmark suite.
"""
import re

from data.eval_set import EvalSet
from eval.judge_client import LocalJudgeClient

# Fallback instruction, used only when a task's rows carry no `_instruction`. It suits a
# question-answering task, which is what it was written for, and it is actively WRONG for anything
# else: on DialogSum it told the model to "answer" a chat transcript that asks nothing, so the model
# continued the conversation instead of summarizing it. Every task that is not
# question-answering states its own instruction on its rows.
DEFAULT_GENERATION_INSTRUCTION = "Answer the following question:"

# Rows carry the instruction under a leading underscore so it is treated as metadata: it is
# excluded from the JSON schema shown to the synthesis teacher, so generated rows cannot invent
# or reword it.
INSTRUCTION_FIELD = "_instruction"


def resolve_generation_instruction(rows) -> str:
    """The single instruction shared by every row of one dataset.

    Resolved ONCE per dataset rather than per row, and deliberately so: synthetic rows are built
    fresh and do not carry `_instruction`, so a per-row lookup would silently give real and
    synthetic rows different prompts within the same training set. Taking the first instruction
    present makes the whole set consistent.
    """
    for row in rows or []:
        if isinstance(row, dict):
            instruction = str(row.get(INSTRUCTION_FIELD) or "").strip()
            if instruction:
                return instruction
    return DEFAULT_GENERATION_INSTRUCTION


def build_generation_prompt(text: str, instruction: str) -> str:
    """The ONE generation prompt. Used by the eval harness AND the trainer.

    Before this existed the two built their inputs independently: eval wrapped the text in
    "Answer the following question:", while training passed the bare text with no instruction at
    all. The model was therefore fine-tuned on one input distribution and scored on another —
    the train/serve skew that `build_classify_prompt` had already been introduced to prevent on
    the classification side.
    """
    return f"{instruction}\n\n{text}"


def build_prompts(eval_set: EvalSet) -> list[str]:
    instruction = resolve_generation_instruction(eval_set.all)
    return [
        build_generation_prompt(example.get("text", ""), instruction)
        for example in eval_set.all
    ]


_REASONING_TAGS = r"(?:reasoning|think)"
_REASONING_BLOCK_RE = re.compile(
    rf"\s*<\s*{_REASONING_TAGS}\s*>.*?<\s*/\s*(?:{_REASONING_TAGS}|tool_call)\s*>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
_ORPHAN_REASONING_CLOSE_RE = re.compile(
    rf"^.*?<\s*/\s*(?:{_REASONING_TAGS}|tool_call)\s*>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


def split_reasoning(raw: str) -> tuple[str, str]:
    """Split a raw generation into ``(reasoning, answer)``.

    Both halves are kept by the caller: the reasoning is real signal for diagnosing HOW the model
    reached an answer, and discarding it at the source would make that unrecoverable. It simply
    must not reach the judge, which is asked "how good is this summary?" — handing it a
    `<reasoning>` block guarantees a poor score for output that may contain a fine summary.
    """
    text = raw or ""
    match = _REASONING_BLOCK_RE.search(text)
    if match:
        reasoning = match.group(0)
        answer = (text[: match.start()] + text[match.end():]).strip()
        # A model that emits ONLY reasoning has no answer to judge; keep the text rather than
        # hand the judge an empty string, which would score 0 and hide the real failure.
        return reasoning.strip(), (answer or text.strip())
    orphan = _ORPHAN_REASONING_CLOSE_RE.match(text)
    if orphan:
        answer = text[orphan.end():].strip()
        if answer:
            return orphan.group(0).strip(), answer
    return "", text.strip()


def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]:
    return [split_reasoning(raw)[1] for raw in raw_outputs]


def _result(eval_set, predictions, scores, metric_name: str, format_flags) -> dict:
    failures = [
        {**example, "predicted": prediction, "judge_score": value}
        for example, prediction, value in zip(eval_set.all, predictions, scores)
        if value < 0.5
    ]
    average = sum(scores) / len(scores) if scores else 0.0
    format_valid = sum(format_flags) / len(format_flags) if format_flags else 0.0
    # `f1` stays as the pipeline's universal comparison scalar — renaming it would break
    # checkpoints and DAG replay — and `metric` says what the number really is, so a report cannot
    # present a judge mean as an F1.
    return {
        "f1": average,
        "metric": metric_name,
        "per_class": {metric_name: average, "format_valid": format_valid},
        "failures": failures,
        "format_valid": format_valid,
    }


def score_with_judge(eval_set: EvalSet, predictions: list[str]) -> dict:
    """Score with the local LLM judge. Used by tasks whose quality is a judgement."""
    triples = [
        (
            example.get("text", ""),
            example.get("answer", example.get("label", "")),
            prediction,
        )
        for example, prediction in zip(eval_set.all, predictions)
    ]
    scores = LocalJudgeClient.from_config().score_many(triples)
    # A judged task has no output contract to satisfy, so "format" can only mean the model produced
    # something to judge. It is reported anyway so every task's log line has the same two numbers.
    flags = [1.0 if str(prediction or "").strip() else 0.0 for prediction in predictions]
    return _result(eval_set, predictions, scores, "judge_mean_0_1", flags)


def judge_failure_category_of(failure: dict) -> str:
    """Coarse bands for judge-scored failures — the score itself is the only signal available."""
    try:
        value = float(failure.get("judge_score") or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if not str(failure.get("predicted") or "").strip():
        return "empty_output"
    if value <= 0.1:
        return "unrelated_output"
    return "partially_correct"
