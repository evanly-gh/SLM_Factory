"""MedQA-USMLE (4-option) loader (2026-08-01).

MedQA is a medical multiple-choice QA benchmark. We score it as ``classification`` where the
label is the correct option letter (A/B/C/D) and ``text`` embeds the question plus the rendered
options, so the argmax-label metric applies directly.

The pure ``convert_medqa_rows`` is unit-tested on an in-memory sample; ``load_medqa`` performs
the live HF pull on the cluster.
"""
from __future__ import annotations

from collections.abc import Iterable

HF_ID = "GBaker/MedQA-USMLE-4-options"

_LETTERS = ("A", "B", "C", "D")


def _render_options(options) -> tuple[str, dict]:
    """Return (rendered_options_text, letter→text map) from a MedQA options payload."""
    letter_map: dict[str, str] = {}
    if isinstance(options, dict):
        # Options keyed by letter (e.g. {"A": "...", ...}) or by index string.
        for i, letter in enumerate(_LETTERS):
            value = options.get(letter)
            if value is None:
                value = options.get(str(i)) or options.get(i)
            if value is not None:
                letter_map[letter] = str(value)
    elif isinstance(options, (list, tuple)):
        for i, value in enumerate(options[: len(_LETTERS)]):
            letter_map[_LETTERS[i]] = str(value)
    rendered = "\n".join(f"{letter}. {letter_map[letter]}"
                         for letter in _LETTERS if letter in letter_map)
    return rendered, letter_map


def _resolve_answer_letter(ex: dict, letter_map: dict) -> str | None:
    """Determine the gold option letter from the row's answer fields."""
    idx = ex.get("answer_idx")
    if isinstance(idx, str) and idx.strip().upper() in _LETTERS:
        return idx.strip().upper()
    if isinstance(idx, int) and 0 <= idx < len(_LETTERS):
        return _LETTERS[idx]
    # Fall back to matching the answer text against the options.
    answer_text = str(ex.get("answer") or "").strip()
    if answer_text:
        for letter, value in letter_map.items():
            if value.strip() == answer_text:
                return letter
    return None


def convert_medqa_rows(dataset: Iterable[dict]) -> list[dict]:
    """Map raw MedQA rows to ``{text, label}`` with the option letter as the label."""
    out: list[dict] = []
    for ex in dataset:
        question = str(ex.get("question") or ex.get("text") or "").strip()
        if not question:
            continue
        rendered, letter_map = _render_options(ex.get("options"))
        letter = _resolve_answer_letter(ex, letter_map)
        if letter is None:
            continue
        text = f"{question}\n{rendered}" if rendered else question
        out.append({"text": text, "label": letter})
    return out


def load_medqa(max_train: int = 2000, max_test: int = 800) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` MedQA MCQs shaped as ``{text, label}`` rows."""
    from datasets import load_dataset

    train = convert_medqa_rows(load_dataset(HF_ID, split=f"train[:{max_train}]"))
    try:
        raw_test = load_dataset(HF_ID, split=f"test[:{max_test}]")
    except (ValueError, KeyError):
        raw_test = load_dataset(HF_ID, split=f"validation[:{max_test}]")
    test = convert_medqa_rows(raw_test)
    return train, test
