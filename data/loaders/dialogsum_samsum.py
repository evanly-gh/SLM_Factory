"""DialogSum + SAMSum dialogue-summarization loader (2026-08-01).

Both are dialogue→summary datasets scored as ``generation`` (LLM-as-judge). We pull the two,
shape each into ``{text, answer}`` (text=dialogue, answer=reference summary), and concatenate —
DialogSum for training breadth, SAMSum for a second style. The pure converters are unit-tested;
``load_dialogsum_samsum`` performs the live HF pulls.
"""
from __future__ import annotations

from collections.abc import Iterable

DIALOGSUM_ID = "knkarthick/dialogsum"
SAMSUM_ID = "Samsung/samsum"


def convert_dialogsum_rows(dataset: Iterable[dict]) -> list[dict]:
    """DialogSum rows carry ``dialogue`` and ``summary`` → ``{text, answer}``."""
    out: list[dict] = []
    for ex in dataset:
        dialogue = str(ex.get("dialogue") or ex.get("text") or "").strip()
        summary = str(ex.get("summary") or ex.get("answer") or "").strip()
        if not dialogue or not summary:
            continue
        out.append({"text": dialogue, "answer": summary, "label": "generation"})
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
