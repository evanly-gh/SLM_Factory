"""CoEdIT text-editing loader (2026-08-01).

CoEdIT ships instruction + source + target text-edit pairs (grammar, clarity, coherence,
paraphrase). We score it as a ``diff`` task: the model must express the edit as a unified diff.
CoEdIT does not ship diffs, so the loader computes the GOLD unified diff of ``src`` → ``tgt``
with ``difflib`` (free, exact) — the same diff ``eval/scorers/diff.py`` applies with ``git``.

Each row is ``{text, answer, src, tgt}`` where ``text`` is the edit instruction, ``src``/``tgt``
are the original/edited text, and ``answer`` is the gold unified diff. The pure
``convert_coedit_rows`` is unit-tested; ``load_coedit`` performs the live HF pull.
"""
from __future__ import annotations

import difflib
from collections.abc import Iterable

HF_ID = "grammarly/coedit"

# CoEdIT's ``src`` embeds the instruction as a prefix, e.g. "Fix grammar: <sentence>". We split
# on the first colon to recover (instruction, source text).
_INSTRUCTION_SEP = ":"


def gold_unified_diff(src: str, tgt: str) -> str:
    """Unified diff of ``src`` → ``tgt`` with a/b path prefixes the diff scorer expects."""
    return "".join(difflib.unified_diff(
        src.splitlines(keepends=True),
        tgt.splitlines(keepends=True),
        fromfile="a/file.txt", tofile="b/file.txt",
    ))


def _split_instruction(raw_src: str) -> tuple[str, str]:
    """Return (instruction, source_text) from a CoEdIT ``src`` field.

    Falls back to a generic instruction when no colon prefix is present.
    """
    if _INSTRUCTION_SEP in raw_src:
        instruction, _, source = raw_src.partition(_INSTRUCTION_SEP)
        instruction, source = instruction.strip(), source.strip()
        if instruction and source:
            return instruction, source
    return "Edit the text as needed", raw_src.strip()


def convert_coedit_rows(dataset: Iterable[dict]) -> list[dict]:
    """Map raw CoEdIT rows to ``{text, answer, src, tgt}`` with a difflib gold diff.

    Rows whose source and target are identical (no edit) are dropped — an empty diff is not a
    meaningful supervision or eval signal. Callers may pass rows that already provide ``src``
    and ``tgt`` explicitly (instruction in ``text``); those are used verbatim.
    """
    out: list[dict] = []
    for ex in dataset:
        if ex.get("src") is not None and ex.get("tgt") is not None and ex.get("text"):
            instruction = str(ex.get("text")).strip()
            source = str(ex.get("src"))
        else:
            raw_src = str(ex.get("src") or ex.get("text") or "")
            if not raw_src.strip():
                continue
            instruction, source = _split_instruction(raw_src)
        target = str(ex.get("tgt") or ex.get("answer") or "")
        # Ensure trailing newline so difflib emits a well-formed hunk that git can apply.
        src = source if source.endswith("\n") else source + "\n"
        tgt = target if target.endswith("\n") else target + "\n"
        if not source.strip() or not target.strip() or src == tgt:
            continue
        out.append({
            "text": instruction,
            "src": src,
            "tgt": tgt,
            "answer": gold_unified_diff(src, tgt),
            "label": "diff",
        })
    return out


def load_coedit(max_train: int = 2000, max_test: int = 800) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` CoEdIT edits shaped as ``diff`` rows."""
    from datasets import load_dataset

    train = convert_coedit_rows(load_dataset(HF_ID, split=f"train[:{max_train}]"))
    # CoEdIT's held-out split is named 'validation'.
    try:
        raw_test = load_dataset(HF_ID, split=f"validation[:{max_test}]")
    except (ValueError, KeyError):
        raw_test = load_dataset(HF_ID, split=f"test[:{max_test}]")
    test = convert_coedit_rows(raw_test)
    return train, test
