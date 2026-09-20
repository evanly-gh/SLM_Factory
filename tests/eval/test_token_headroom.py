"""Every task's output cap must fit its own gold answers.

WHY THIS IS A TEST AND NOT JUST A PROBE

    "The baseline eval must not be depressed by token truncation" is a standing invariant, not a
    one-time measurement. It breaks silently and in the direction that looks like a model problem:
    a gold answer longer than `max_new_tokens` is an answer the model is structurally unable to
    emit, so the eval marks it wrong for a reason that has nothing to do with the model, and on an
    exact-match task one missing bracket is the whole score.

    The evidence available before this was INDIRECT — the harness probe's `format_valid` sits near
    1.0, which is consistent with no truncation but does not measure it, because a truncated answer
    can still be format-valid. This measures it: tokenize the gold with a real tokenizer and
    compare against the cap.

    A real tokenizer rather than a chars/token estimate, because the estimate is exactly what
    failed before. `config/token_budget.py` assumed 3.5 chars/token; GEC's tokenized learner
    English came in at 3.01, which cost a 400 from the endpoint and 45 of 60 rows on the 2026-09-08
    synthesis audit.

    `scripts/probe_token_headroom.py` is the full version — 1,000 rows against two tokenizers, and
    the source of the numbers quoted below. This is the cheap always-on guard.
"""
from __future__ import annotations

import pytest

TASKS = ("topv2", "multiconer", "gec_bea19", "goemotions", "dialogsum")

# Measured 2026-09-09 over 1,000 eval rows per task, Qwen3-1.7B and SmolLM2-360M. Headroom is
# `max_new_tokens / longest gold answer`: goemotions 16.0x, multiconer 5.2x, topv2 5.1x (4.3x on
# SmolLM2's smaller vocabulary), dialogsum 5.0x, gec_bea19 1.57x. GEC is the tight one — 163 tokens
# against a 256 cap, with p99 at 76 — so it is the task this guard exists for.
MIN_HEADROOM = 1.4


@pytest.fixture(scope="module")
def tokenizer():
    """Qwen3-1.7B, one of the two probe models. Skipped rather than downloaded on demand."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B", trust_remote_code=True, local_files_only=True
        )
    except Exception as error:  # noqa: BLE001 — an absent cache is a skip, not a failure
        pytest.skip(f"tokenizer unavailable offline: {type(error).__name__}")


def _gold_text(row: dict) -> str:
    import json

    for field in ("answer", "label"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    if row.get("entities") is not None:
        return json.dumps(row["entities"], ensure_ascii=False)
    return ""


@pytest.mark.parametrize("task", TASKS)
def test_no_gold_answer_is_longer_than_the_task_can_generate(task, tokenizer):
    """The cap has to fit the ANSWER, or the eval is scoring its own budget."""
    from tasks import get_task

    spec = get_task(task)
    _train, eval_rows = spec.load(max_train=8, max_test=200, log=lambda *a: None)
    rows = list(eval_rows)

    lengths = [len(tokenizer(_gold_text(r))["input_ids"]) for r in rows]
    assert lengths, f"{task}: no eval rows to measure"

    longest = max(lengths)
    over = [n for n in lengths if n > spec.max_new_tokens]
    assert not over, (
        f"{task}: {len(over)} of {len(rows)} gold answers exceed max_new_tokens="
        f"{spec.max_new_tokens} (longest {longest}); those rows cannot be answered correctly at "
        f"any model quality"
    )
    headroom = spec.max_new_tokens / max(longest, 1)
    assert headroom >= MIN_HEADROOM, (
        f"{task}: only {headroom:.2f}x headroom (longest gold {longest} vs cap "
        f"{spec.max_new_tokens}); raise max_new_tokens before this becomes truncation"
    )


@pytest.mark.parametrize("task", TASKS)
def test_the_prompt_and_the_output_cap_both_fit_the_sequence_window(task, tokenizer):
    """`max_seq_length` has to hold the prompt AND the reserved output, not one or the other.

    Checked separately from the answer length because it fails differently: the prompt gets
    truncated instead of the answer, so the model is scored on a question it was shown only part
    of. `TaskSpec` already rejects `max_new_tokens >= max_seq_length`, which is the degenerate
    case; this is the real one, measured against actual prompts.
    """
    from data.eval_set import EvalSet
    from tasks import get_task

    spec = get_task(task)
    _train, eval_rows = spec.load(max_train=8, max_test=200, log=lambda *a: None)
    rows = list(eval_rows)

    prompts = [str(p) for p in spec.build_prompts(EvalSet(all=rows, task=task))]
    longest = max(len(tokenizer(p)["input_ids"]) for p in prompts)
    assert longest + spec.max_new_tokens <= spec.max_seq_length, (
        f"{task}: longest prompt is {longest} tokens and the run reserves "
        f"{spec.max_new_tokens} for output, which exceeds max_seq_length="
        f"{spec.max_seq_length}; the prompt would be truncated"
    )
