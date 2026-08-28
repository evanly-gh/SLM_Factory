"""Materialize the checksummed local SMS Spam bundle under `data/local/sms_spam/`.

Mirrors what `scripts/download_datasets.py` does for `bc5cdr`, kept separate because the SMS Spam
Collection ships a single `train` split and therefore needs the held-out half MADE rather than
copied. The split logic lives in the loader so that the bundle and the live-pull fallback cannot
produce different eval sets; this script only freezes the result and records its lineage.

    python scripts/vendor_sms_spam.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders.sms_spam import (  # noqa: E402
    EVAL_FRACTION,
    HF_CONFIG,
    HF_ID,
    LABELS,
    SPLIT_SEED,
    convert_sms_spam_rows,
    deduplicate,
    stratified_split,
)

ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local", "sms_spam"
)


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main() -> int:
    from datasets import load_dataset

    print(f"pulling {HF_ID} ({HF_CONFIG}) ...")
    raw = load_dataset(HF_ID, HF_CONFIG, split="train")
    revision = None
    try:
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(HF_ID).sha
    except Exception as exc:  # lineage is best-effort; the checksums are the integrity guarantee
        print(f"  (could not resolve dataset revision: {type(exc).__name__})")

    converted = convert_sms_spam_rows(raw)
    rows = deduplicate(converted)
    duplicates = len(converted) - len(rows)
    train, test = stratified_split(rows)

    # The whole point of deduplicating before the split is that this must be zero.
    train_keys = {" ".join(r["text"].lower().split()) for r in train}
    overlap = sum(1 for r in test if " ".join(r["text"].lower().split()) in train_keys)
    if overlap:
        raise RuntimeError(f"train/eval overlap after stratified split: {overlap} row(s)")

    os.makedirs(ROOT, exist_ok=True)
    _write_jsonl(os.path.join(ROOT, "train.jsonl"), train)
    _write_jsonl(os.path.join(ROOT, "test.jsonl"), test)

    counts = {
        "train": len(train),
        "test": len(test),
        "total": len(rows),
        "train_spam": sum(1 for r in train if r["label"] == "spam"),
        "test_spam": sum(1 for r in test if r["label"] == "spam"),
    }
    manifest = {
        "schema_version": 2,
        "name": "sms_spam",
        "task": "sms_spam",
        "hf_id": HF_ID,
        "config": HF_CONFIG,
        "source_revision": revision,
        "source_url": f"https://huggingface.co/datasets/{HF_ID}",
        # Recorded because it is NOT an upstream split: anyone comparing scores across runs needs
        # to know the held-out half is ours and exactly how it was drawn.
        "source_splits": {"train": "train", "test": "train (stratified holdout)"},
        "split_policy": {
            "kind": "stratified_holdout",
            "eval_fraction": EVAL_FRACTION,
            "seed": SPLIT_SEED,
            "deduplicated_before_split": True,
            "duplicates_removed": duplicates,
        },
        "labels": list(LABELS),
        "row_schema": {"required": ["text", "label"], "optional": []},
        "counts": counts,
        "overlap": {
            "field": "text",
            "normalization": "lower_whitespace_v1",
            "normalized_count": 0,
            "removed_from_train": 0,
        },
        "integrity": {
            "algorithm": "sha256",
            "checksum_file": "checksums.sha256",
            "files": {
                "train.jsonl": _sha256(os.path.join(ROOT, "train.jsonl")),
                "test.jsonl": _sha256(os.path.join(ROOT, "test.jsonl")),
            },
        },
    }
    with open(os.path.join(ROOT, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    lines = [
        f"{manifest['integrity']['files'][name]}  {name}"
        for name in ("train.jsonl", "test.jsonl")
    ]
    lines.append(f"{_sha256(os.path.join(ROOT, 'manifest.json'))}  manifest.json")
    with open(os.path.join(ROOT, "checksums.sha256"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"  raw rows        : {len(converted)}")
    print(f"  duplicates      : {duplicates} (removed BEFORE the split)")
    print(f"  train           : {counts['train']} (spam {counts['train_spam']})")
    print(f"  test            : {counts['test']} (spam {counts['test_spam']})")
    print(f"  train/eval leak : {overlap}")
    print(f"  wrote           : {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
