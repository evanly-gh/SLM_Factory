"""BC5CDR biomedical NER loader (2026-08-13).

Promotes BC5CDR from the autonomous path to a pinned `SLM_BENCHMARK_TASK` key. The previous run
(`slm-ner-l40s-37531245`, 44.8h, best span-F1 0.8628 against a 0.88 threshold) reached BC5CDR
through the orchestrator emitting a plan that `web_acquire` happened to resolve to the `bc5cdr`
catalog key. That is not reproducible: a different plan wording gets different data. Pinning the
loader makes the run repeatable.

Rows are ``{text, entities: [{"text", "type"}]}`` — the schema `eval/scorers/ner.py` scores with
exact `(surface, type)` multiset span-F1.

**Source order matters and is not the same as the autonomous ladder's.** Under `datasets 4.3.0`:

- ``tner/bc5cdr`` is script-based → ``RuntimeError: Dataset scripts are no longer supported``.
- ``spyysalo/bc5cdr`` is gone from the Hub → ``DatasetNotFoundError``.
- T-NER's raw ``dataset/{train,test}.json`` files still resolve and are the working remote path.

So this loader reads the **checksummed local bundle first** (``data/local/bc5cdr``), which needs
no network, and only falls back to the raw JSON URLs. The bundle is also strictly better data:
**5,096 train rows against the 3,403 the live loader produced**. The previous run died with
`bounded data_rebuild plan space is exhausted` after running out of gold to resample, so the
extra 1,693 rows address the actual cause of that failure.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable

LOCAL_BUNDLE = "data/local/bc5cdr"
TNER_JSON = {
    "train": "https://huggingface.co/datasets/tner/bc5cdr/resolve/main/dataset/train.json",
    "test": "https://huggingface.co/datasets/tner/bc5cdr/resolve/main/dataset/test.json",
}
# T-NER's integer tag ids for BC5CDR. Pinned rather than read from the repo's label.json so a
# silent upstream reordering cannot relabel every span without anything failing.
TNER_TAG_NAMES = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]
ENTITY_TYPES = ("Chemical", "Disease")


def bio_to_spans(tokens: list[str], tags: list[str]) -> list[dict]:
    """Collapse BIO-tagged tokens into ``{"text", "type"}`` entity spans."""
    spans: list[dict] = []
    current: list[str] = []
    current_type: str | None = None

    def flush():
        nonlocal current, current_type
        if current and current_type:
            spans.append({"text": " ".join(current), "type": current_type})
        current, current_type = [], None

    for token, tag in zip(tokens, tags):
        tag = str(tag)
        if tag.startswith("B-"):
            flush()
            current, current_type = [token], tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current.append(token)
        else:
            flush()
    flush()
    return spans


def convert_tner_rows(rows: Iterable[dict]) -> list[dict]:
    """Map T-NER ``{tokens, tags}`` rows to ``{text, entities}``, resolving integer tag ids."""
    out: list[dict] = []
    for row in rows:
        tokens = row.get("tokens") or []
        raw_tags = row.get("tags")
        if raw_tags is None:
            raw_tags = row.get("ner_tags") or []
        if not tokens or len(tokens) != len(raw_tags):
            continue
        tags = [
            TNER_TAG_NAMES[t] if isinstance(t, int) and 0 <= t < len(TNER_TAG_NAMES) else str(t)
            for t in raw_tags
        ]
        text = " ".join(str(t) for t in tokens).strip()
        if not text:
            continue
        out.append({"text": text, "entities": bio_to_spans(list(tokens), tags)})
    return out


def _read_local_bundle(split: str, root: str) -> list[dict] | None:
    """Read one split of the frozen offline bundle, already in ``{text, entities}`` form."""
    path = os.path.join(root, f"{split}.jsonl")
    if not os.path.isfile(path):
        return None
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("text") and isinstance(row.get("entities"), list):
                rows.append({"text": row["text"], "entities": row["entities"]})
    return rows or None


def load_ner_bc5cdr(max_train: int = 4000, max_test: int = 800,
                    log=print) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` BC5CDR rows, preferring the local checksummed bundle."""
    root = os.environ.get("SLM_BC5CDR_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        LOCAL_BUNDLE,
    )
    train = _read_local_bundle("train", root)
    test = _read_local_bundle("test", root)
    if train and test:
        log(f"      [bc5cdr] local bundle {root}: train={len(train)} test={len(test)}")
        return train[:max_train], test[:max_test]

    log("      [bc5cdr] local bundle unavailable; falling back to T-NER raw JSON")
    from datasets import load_dataset

    raw = load_dataset("json", data_files=TNER_JSON)
    train = convert_tner_rows(raw["train"])
    test = convert_tner_rows(raw["test"])
    log(f"      [bc5cdr] T-NER JSON: train={len(train)} test={len(test)}")
    if not test:
        raise RuntimeError(
            "BC5CDR produced zero eval rows from both the local bundle and the T-NER JSON "
            "fallback — refusing to proceed with an empty held-out set."
        )
    return train[:max_train], test[:max_test]
