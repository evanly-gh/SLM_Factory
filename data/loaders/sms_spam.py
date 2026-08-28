"""UCI SMS Spam Collection loader (re-added 2026-08-23).

The original `data/loaders/sms_spam.py` was deleted in the 2026-08-18 registry rebuild along with
the abstract `task_type` channel it depended on. This is a rewrite, not a revert, because the old
one carried three defects that the current task registry makes it possible to fix properly:

  * **B44 — the split was not shuffled.** It took `examples[:80%]` as train. The UCI file is not
    randomly ordered, so the two halves had different class balance. Here the split is stratified
    per class and seeded, so train and eval carry the same base rate by construction.
  * **It fetched a zip over plain HTTP from `archive.ics.uci.edu`** at load time, with no checksum
    and no cache beyond a bare TSV. This reads the checksummed local bundle first and falls back
    to the HuggingFace mirror, which is the treatment `bc5cdr` and `proactive_listening` get.
  * **It never deduplicated.** The corpus contains a few hundred byte-identical messages ("Ok.",
    "Sorry, I'll call later"), and a random split puts copies of the same string on both sides.
    That is train/eval contamination that the preflight would flag and the run would silently
    benefit from, so duplicates are collapsed BEFORE the split rather than filtered after.

SHAPE
    `{text, label}` with `label` in `{"ham", "spam"}` — the schema `eval/scorers/classification.py`
    scores. `spam` is the minority class at roughly 13% of the corpus, which is why the task spec
    scores `minority_f1` rather than accuracy or macro-F1: predicting `ham` for everything is
    right 87% of the time and must not look like learning.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterable

LOCAL_BUNDLE = "data/local/sms_spam"
HF_ID = "ucirvine/sms_spam"
HF_CONFIG = "plain_text"

HAM_LABEL = "ham"
SPAM_LABEL = "spam"
LABELS = (HAM_LABEL, SPAM_LABEL)

# The upstream corpus ships one `train` split, so the held-out half is ours to make. Both the
# fraction and the seed are pinned: a requeued run must rebuild the identical eval set or the
# scores from before and after the requeue are not comparable.
EVAL_FRACTION = 0.20
SPLIT_SEED = 20260823


def _normalize(text: str) -> str:
    """Comparison key for duplicate detection. Mirrors the eval firewall's normalization."""
    return " ".join(str(text or "").lower().split())


def convert_sms_spam_rows(dataset: Iterable[dict]) -> list[dict]:
    """Map raw HuggingFace rows to `{text, label}`.

    Accepts the upstream column names (`sms` plus an integer `label` over the ClassLabel
    `['ham', 'spam']`) as well as already-resolved `{text, label}` rows, so the same converter
    serves the live pull, the local bundle and the unit tests.
    """
    out: list[dict] = []
    for example in dataset:
        text = str(example.get("sms") or example.get("text") or "").strip()
        if not text:
            continue
        label = example.get("label")
        if isinstance(label, bool):
            label = int(label)
        if isinstance(label, int):
            # ClassLabel order is ['ham', 'spam'] and is pinned here rather than read from the
            # dataset features: a silent upstream reordering would otherwise invert every label
            # in the corpus without anything failing.
            if not 0 <= label < len(LABELS):
                continue
            label = LABELS[label]
        label = str(label or "").strip().lower()
        if label not in LABELS:
            continue
        out.append({"text": text, "label": label})
    return out


def deduplicate(rows: list[dict]) -> list[dict]:
    """Drop repeated messages, keeping first occurrence.

    Runs before the split, not after. A message appearing in both halves is contamination no
    filter downstream can undo, because by then the eval set is frozen and the duplicate in train
    is indistinguishable from a legitimate row.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        key = _normalize(row["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def stratified_split(
    rows: list[dict],
    eval_fraction: float = EVAL_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[list[dict], list[dict]]:
    """Split per class so both halves carry the corpus base rate (fixes B44)."""
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    rng = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []
    for label in sorted(by_label):
        bucket = list(by_label[label])
        rng.shuffle(bucket)
        cut = max(1, int(round(len(bucket) * eval_fraction)))
        test.extend(bucket[:cut])
        train.extend(bucket[cut:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _read_local_bundle(split: str, root: str) -> list[dict] | None:
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
            if row.get("text") and row.get("label") in LABELS:
                rows.append({"text": row["text"], "label": row["label"]})
    return rows or None


def _verify_bundle(root: str) -> None:
    """Raise if a bundle file does not match its recorded sha256."""
    checksums = os.path.join(root, "checksums.sha256")
    if not os.path.isfile(checksums):
        return
    with open(checksums, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 2:
                continue
            digest, filename = parts[0], parts[1].lstrip("*")
            path = os.path.join(root, filename)
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as payload:
                actual = hashlib.sha256(payload.read()).hexdigest()
            if actual != digest:
                raise RuntimeError(
                    f"sms_spam bundle file {filename} does not match its recorded checksum "
                    f"({actual} != {digest}); re-run scripts/download_datasets.py"
                )


def _balanced_head(rows: list[dict], limit: int, seed: int = SPLIT_SEED) -> list[dict]:
    """Take up to `limit` rows round-robin across classes.

    A plain `rows[:limit]` on a 13%-spam corpus would hand a small cap a curriculum that is almost
    entirely `ham`. Round-robin keeps the minority class present at any cap, and the task's
    `balance_labels` quality control then enforces the ratio it actually wants.
    """
    if limit <= 0:
        return []
    if limit >= len(rows):
        return list(rows)

    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
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


def load_sms_spam(
    max_train: int = 5000,
    max_test: int = 1000,
    log=print,
) -> tuple[list[dict], list[dict]]:
    """Return `(train, test)` SMS Spam rows, preferring the checksummed local bundle.

    Both halves come from the same corpus and the same stratified split, so this is an
    in-distribution task by construction — unlike `xlam_bfcl`, whose eval is a different corpus.
    """
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), LOCAL_BUNDLE)

    _verify_bundle(root)
    train = _read_local_bundle("train", root)
    test = _read_local_bundle("test", root)
    if train and test:
        log(f"      [sms_spam] local bundle {root}: train={len(train)} test={len(test)}")
    else:
        from datasets import load_dataset

        log(f"      [sms_spam] no local bundle; pulling {HF_ID} ({HF_CONFIG}) from HuggingFace")
        raw = load_dataset(HF_ID, HF_CONFIG, split="train")
        rows = deduplicate(convert_sms_spam_rows(raw))
        train, test = stratified_split(rows)
        log(f"      [sms_spam] {len(rows)} deduplicated row(s) → train={len(train)} test={len(test)}")

    train = _balanced_head(train, max_train)
    test = _balanced_head(test, max_test)

    spam_train = sum(1 for r in train if r["label"] == SPAM_LABEL)
    spam_test = sum(1 for r in test if r["label"] == SPAM_LABEL)
    log(
        f"      [sms_spam] train={len(train)} (spam {spam_train}, "
        f"{spam_train / max(1, len(train)):.1%}) "
        f"test={len(test)} (spam {spam_test}, {spam_test / max(1, len(test)):.1%})"
    )
    return train, test
