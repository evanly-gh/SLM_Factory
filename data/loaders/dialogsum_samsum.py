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


def _dialogue_key(row: dict) -> str:
    """Normalized dialogue text, for cross-source deduplication."""
    return " ".join(str(row.get("text") or "").lower().split())


def load_dialogsum_samsum(
    max_train: int = 2000, max_test: int = 800, log=print
) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` merging DialogSum + SAMSum as ``{text, answer}`` rows.

    **SAMSum ships a handful of dialogues in BOTH its own train and test splits**, each with an
    independently written summary. Measured on a 500+500 draw: 2 collisions, ~0.2% of rows, and both
    were samsum_train x samsum_test — NOT, as first reported, a DialogSum/SAMSum cross-source
    problem (B285, corrected). One example, byte-identical text with two different golds:

        train: "Jeff has a skin allergy. He doesn't take meds all the time..."
        eval : "Serena's skin condition is fine now and she doesn't have to take medication..."

    Reading the dialogue, the training summary is the accurate one. So the eval row is unwinnable.

    At ~0.2% this is NEGLIGIBLE for the score — it did not move the run's 0.7157 — and curate's eval
    firewall already removes the training side before training. Deduplicating here is cheap hygiene
    rather than a fix for a significant defect: it keeps the reported curriculum size honest instead
    of having the firewall silently shrink it. Dedup is EVAL FIRST, so an eval row is never dropped.
    """
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

    # Within-split dedup first (the same dialogue can appear twice in one split across the two
    # sources), then remove any training row whose dialogue is in the eval set.
    def _dedup(rows: list[dict]) -> list[dict]:
        seen, out = set(), []
        for row in rows:
            key = _dialogue_key(row)
            if key and key not in seen:
                seen.add(key)
                out.append(row)
        return out

    test = _dedup(test)
    eval_keys = {_dialogue_key(row) for row in test}
    train_deduped = _dedup(train)
    train_clean = [row for row in train_deduped if _dialogue_key(row) not in eval_keys]

    dropped_dupe = len(train) - len(train_deduped)
    dropped_overlap = len(train_deduped) - len(train_clean)
    if log and (dropped_dupe or dropped_overlap):
        log(f"      [dialogsum_samsum] dedup: dropped {dropped_dupe} duplicate train row(s) and "
            f"{dropped_overlap} train row(s) whose dialogue is in the eval split "
            f"(SAMSum ships a few dialogues in both its own splits, with different summaries)")
    return train_clean, test
