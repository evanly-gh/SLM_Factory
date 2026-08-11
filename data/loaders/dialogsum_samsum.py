"""DialogSum + SAMSum dialogue-summarization loader (2026-08-01).

Both are dialogue→summary datasets scored as ``generation`` (LLM-as-judge). We pull the two,
shape each into ``{text, answer}`` (text=dialogue, answer=reference summary), and concatenate —
DialogSum for training breadth, SAMSum for a second style. The pure converters are unit-tested;
``load_dialogsum_samsum`` performs the live HF pulls.
"""
from __future__ import annotations

from collections.abc import Iterable

DIALOGSUM_ID = "knkarthick/dialogsum"
# `Samsung/samsum` (and the bare `samsum` alias) were withdrawn from the Hub and now raise
# DatasetNotFoundError, which killed run slm-dialogsum-samsum-cse-38186256 in eval_setup before
# any model was selected. This mirror carries the same `dialogue`/`summary` columns and the same
# split sizes (14,731 train / 819 test), so the converters are unchanged (B249).
SAMSUM_ID = "knkarthick/samsum"


# Carried on every row and read by `eval.scorers.generation.resolve_generation_instruction`,
# which the eval harness AND the trainer both call — so the model is asked to do the same thing
# in training and at eval.
#
# Without it both fall back to the family default, "Answer the following question:", which is
# wrong here in a way that destroys the task: a dialogue transcript asks no question, so the
# model continued the conversation instead of summarizing it. Observed in
# slm-dialogsum-samsum-cse-38186375, e.g. the reply "Shelly: How about you? Any volunteer work?
# Tracy: Nah. Not into that." against a gold summary (B250). The second sentence is aimed
# squarely at that failure.
SUMMARIZATION_INSTRUCTION = (
    "Summarize the following conversation in one to three sentences. "
    "Write only the summary — do not continue the conversation or reply to it."
)


def convert_dialogsum_rows(dataset: Iterable[dict]) -> list[dict]:
    """DialogSum rows carry ``dialogue`` and ``summary`` → ``{text, answer}``."""
    out: list[dict] = []
    for ex in dataset:
        dialogue = str(ex.get("dialogue") or ex.get("text") or "").strip()
        summary = str(ex.get("summary") or ex.get("answer") or "").strip()
        if not dialogue or not summary:
            continue
        out.append({
            "text": dialogue,
            "answer": summary,
            "label": "generation",
            "_instruction": SUMMARIZATION_INSTRUCTION,
        })
    return out


# SAMSum shares DialogSum's dialogue/summary field names.
convert_samsum_rows = convert_dialogsum_rows


def load_dialogsum_samsum(
    max_train: int = 2000, max_test: int = 800
) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` merging DialogSum + SAMSum as ``{text, answer}`` rows."""
    from datasets import load_dataset

    half_train = max(1, max_train // 2)
    half_test = max(1, max_test // 2)
    train = (
        convert_dialogsum_rows(load_dataset(DIALOGSUM_ID, split=f"train[:{half_train}]"))
        + convert_samsum_rows(load_dataset(SAMSUM_ID, split=f"train[:{half_train}]"))
    )
    test = (
        convert_dialogsum_rows(load_dataset(DIALOGSUM_ID, split=f"test[:{half_test}]"))
        + convert_samsum_rows(load_dataset(SAMSUM_ID, split=f"test[:{half_test}]"))
    )
    return train, test
