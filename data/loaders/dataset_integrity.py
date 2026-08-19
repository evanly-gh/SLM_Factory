"""Shared integrity rules for downloaded and runtime benchmark splits."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import Iterable

NORMALIZATION_VERSION = "nfkc_casefold_whitespace_v1"
CHECKSUM_FILENAMES = ("train.jsonl", "test.jsonl", "manifest.json")
# Required fields are declared per task (`TaskSpec.required_fields`), not per channel.


def normalize_text(value: object) -> str:
    """Normalize text for contamination checks without changing stored examples."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def normalized_text_overlap(
    train_rows: Iterable[dict], test_rows: Iterable[dict]
) -> set[str]:
    train = {
        normalize_text(row.get("text"))
        for row in train_rows
        if isinstance(row.get("text"), str)
    }
    test = {
        normalize_text(row.get("text"))
        for row in test_rows
        if isinstance(row.get("text"), str)
    }
    train.discard("")
    test.discard("")
    return train & test


def remove_normalized_train_overlap(
    train_rows: list[dict], test_rows: list[dict]
) -> tuple[list[dict], int]:
    """Keep official test rows fixed and remove normalized duplicates from train."""
    test_texts = {
        normalize_text(row.get("text"))
        for row in test_rows
        if isinstance(row.get("text"), str)
    }
    test_texts.discard("")
    clean = [
        row
        for row in train_rows
        if normalize_text(row.get("text")) not in test_texts
    ]
    return clean, len(train_rows) - len(clean)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_fields_for_task(task: str) -> tuple[str, ...]:
    """The fields every row of `task` must carry, from its spec."""
    from tasks import get_task

    return get_task(task).required_fields


def validate_rows(
    rows: Iterable[dict],
    required_fields: Iterable[str],
    *,
    bundle_name: str,
    split: str,
) -> None:
    rows = list(rows)
    required = set(required_fields)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{bundle_name}: {split} row {index} must be an object")
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{bundle_name}: {split} row {index} missing schema fields "
                f"{sorted(missing)}"
            )
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{bundle_name}: {split} row {index} has invalid text")
        if "answer" in required and (
            not isinstance(row.get("answer"), str) or not row["answer"].strip()
        ):
            raise ValueError(f"{bundle_name}: {split} row {index} has invalid answer")
        if "entities" in required and not isinstance(row.get("entities"), list):
            raise ValueError(f"{bundle_name}: {split} row {index} has invalid entities")


def write_checksum_sidecar(
    bundle_dir: str | Path,
    filenames: Iterable[str],
    checksum_filename: str = "checksums.sha256",
) -> Path:
    bundle = Path(bundle_dir)
    checksum_path = bundle / checksum_filename
    checksum_path.write_text(
        "".join(f"{sha256_file(bundle / name)}  {name}\n" for name in filenames),
        encoding="utf-8",
    )
    return checksum_path


def verify_checksum_sidecar(
    bundle_dir: str | Path,
    filenames: Iterable[str],
    checksum_filename: str = "checksums.sha256",
) -> bool:
    """Verify an explicit file set; return False only when the sidecar is absent."""
    bundle = Path(bundle_dir)
    checksum_path = bundle / checksum_filename
    if not checksum_path.exists():
        return False

    filenames = tuple(filenames)
    expected = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        if not separator or len(digest) != 64 or filename not in filenames:
            raise ValueError(f"{bundle.name}: malformed checksum entry {line!r}")
        expected[filename] = digest
    if set(expected) != set(filenames):
        raise ValueError(
            f"{bundle.name}: {checksum_filename} must cover {list(filenames)}"
        )
    for filename, digest in expected.items():
        path = bundle / filename
        if not path.exists() or sha256_file(path) != digest:
            raise ValueError(f"{bundle.name}: checksum mismatch for {filename}")
    return True


def verify_manifest_hashes(bundle_dir: str | Path, hashes: dict[str, str]) -> None:
    bundle = Path(bundle_dir)
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{bundle.name}: manifest integrity file hashes are missing")
    for filename, digest in hashes.items():
        if len(str(digest)) != 64 or sha256_file(bundle / filename) != digest:
            raise ValueError(f"{bundle.name}: manifest hash mismatch for {filename}")


def verify_bundle_checksums(bundle_dir: str | Path) -> bool:
    """Verify a local schema-v2 bundle; return False for explicit legacy v1."""
    return verify_checksum_sidecar(bundle_dir, CHECKSUM_FILENAMES)
