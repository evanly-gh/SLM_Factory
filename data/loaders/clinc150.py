"""CLINC150 intent-classification loader (2026-08-01).

CLINC150 is 150 in-scope intents + an out-of-scope (OOS) class across 10 domains. We treat it
as ``classification`` with the intent string as the label; OOS rows keep the literal label
``oos`` so the boundary slice naturally captures the reject class.

The pure ``convert_clinc150_rows`` shapes raw HF rows into ``{text, label}`` and is unit-tested
on an in-memory sample; ``load_clinc150`` performs the live HF pull on the cluster.
"""
from __future__ import annotations

from collections.abc import Iterable

# Fully namespaced: huggingface_hub >= 1.0 rejects the bare canonical name "clinc_oos"
# with HfUriError when it builds the hf://datasets/<id> URI.
HF_ID = "clinc/clinc_oos"
HF_CONFIG = "plus"  # the 'plus' config includes the OOS class.
OOS_LABEL = "oos"


def convert_clinc150_rows(dataset: Iterable[dict]) -> list[dict]:
    """Map raw CLINC150 rows to ``{text, label}``.

    HF rows carry ``text`` and an integer ``intent``; the label names live on the dataset
    feature. Callers that already resolved the intent to a string may pass ``intent`` (or
    ``label``) as a str, which is used verbatim. Rows without usable text are dropped.
    """
    out: list[dict] = []
    for ex in dataset:
        text = str(ex.get("text") or "").strip()
        if not text:
            continue
        label = ex.get("label")
        if label is None:
            label = ex.get("intent")
        if label is None:
            label = ex.get("intent_name")
        label = str(label).strip()
        if not label:
            continue
        out.append({"text": text, "label": label})
    return out


def _resolve_intent_names(split) -> list[str] | None:
    """Best-effort recovery of the intent label names from the HF dataset features."""
    try:
        feature = split.features["intent"]
        return list(feature.names)
    except (AttributeError, KeyError, TypeError):
        return None


def stratified_by_label(rows: list[dict], limit: int, seed: int = 0) -> list[dict]:
    """Draw up to ``limit`` rows round-robin across labels.

    CLINC150's HF splits are grouped by intent, so a prefix of the split covers only a
    fraction of the 151 labels. Cycling the labels keeps every intent represented once the
    limit reaches the label count. The shuffle is seeded so a requeued run rebuilds the
    identical curriculum.
    """
    import random

    if limit <= 0:
        return []
    if limit >= len(rows):
        return list(rows)

    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("label")), []).append(row)

    rng = random.Random(seed)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    picked: list[dict] = []
    labels = sorted(by_label)
    depth = 0
    while len(picked) < limit:
        progressed = False
        for label in labels:
            bucket = by_label[label]
            if depth < len(bucket):
                picked.append(bucket[depth])
                progressed = True
                if len(picked) == limit:
                    return picked
        if not progressed:
            break
        depth += 1
    return picked


def load_clinc150(max_train: int = 2000, max_test: int = 800) -> tuple[list[dict], list[dict]]:
    """Pull CLINC150 (plus) and return ``(train, test)`` as ``{text, label}`` rows.

    Each split is read in full and then sampled round-robin across intents: the HF splits are
    grouped by intent, so slicing the split itself would cover only a fraction of the labels.
    """
    from datasets import load_dataset

    def _conv(split_name: str, limit: int) -> list[dict]:
        split = load_dataset(HF_ID, HF_CONFIG, split=split_name)
        names = _resolve_intent_names(split)
        rows = []
        for ex in split:
            intent = ex.get("intent")
            if names is not None and isinstance(intent, int) and 0 <= intent < len(names):
                label = names[intent]
            else:
                label = intent
            rows.append({"text": ex.get("text", ""), "label": label})
        return stratified_by_label(convert_clinc150_rows(rows), limit)

    return _conv("train", max_train), _conv("test", max_test)
