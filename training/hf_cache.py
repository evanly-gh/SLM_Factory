"""Reliable Hugging Face model snapshot prefetching for HPC workers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

_DEFAULT_DOWNLOAD_WORKERS = 1
_DOWNLOAD_WORKERS_ENV = "SLM_HF_DOWNLOAD_WORKERS"
_BACKOFF_SECONDS_ENV = "SLM_HF_DOWNLOAD_BACKOFF_SECONDS"
_DEFAULT_BACKOFF_SECONDS = 2.0
_MODEL_IGNORE_PATTERNS = [
    "*.gguf",
    "original/*",
    "*.pth",
    "consolidated*",
]


class HFSnapshotInfrastructureError(RuntimeError):
    """Raised when a complete model snapshot cannot be cached."""


def _configured_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise HFSnapshotInfrastructureError(
            f"{name} must be a positive integer, got {raw_value!r}."
        ) from exc
    if value < 1:
        raise HFSnapshotInfrastructureError(
            f"{name} must be a positive integer, got {raw_value!r}."
        )
    return value


def _configured_nonnegative_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name, str(default))
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise HFSnapshotInfrastructureError(
            f"{name} must be a non-negative number, got {raw_value!r}."
        ) from exc
    if value < 0:
        raise HFSnapshotInfrastructureError(
            f"{name} must be a non-negative number, got {raw_value!r}."
        )
    return value


def _is_local_model_ref(model_id: str) -> bool:
    path = Path(os.path.expanduser(model_id))
    return (
        path.is_dir()
        or path.is_absolute()
        or model_id.startswith(("./", "../", "~"))
    )


def _required_index_shards(snapshot_path: Path) -> set[str]:
    required: set[str] = set()
    for index_path in snapshot_path.glob("*.index.json"):
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read model weight index {index_path.name}: {exc}"
            ) from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"model weight index {index_path.name} has no nonempty weight_map"
            )
        shard_names = set(weight_map.values())
        if not shard_names or not all(
            isinstance(name, str) and name for name in shard_names
        ):
            raise ValueError(
                f"model weight index {index_path.name} contains invalid shard names"
            )
        required.update(shard_names)
    return required


def _has_valid_tokenizer_json(snapshot_path: Path) -> bool:
    tokenizer_path = snapshot_path / "tokenizer.json"
    if not tokenizer_path.is_file() or tokenizer_path.stat().st_size == 0:
        return False
    try:
        tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(tokenizer_data, dict)
        and isinstance(tokenizer_data.get("model"), dict)
        and bool(tokenizer_data["model"])
    )


def _has_valid_vocab_and_merges(snapshot_path: Path) -> bool:
    vocab_path = snapshot_path / "vocab.json"
    merges_path = snapshot_path / "merges.txt"
    if (
        not vocab_path.is_file()
        or vocab_path.stat().st_size == 0
        or not merges_path.is_file()
        or merges_path.stat().st_size == 0
    ):
        return False
    try:
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        merge_lines = merges_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    valid_vocab = (
        isinstance(vocab, dict)
        and bool(vocab)
        and all(
            isinstance(token, str)
            and isinstance(token_id, int)
            and not isinstance(token_id, bool)
            for token, token_id in vocab.items()
        )
    )
    valid_merges = any(
        line.strip() and not line.lstrip().startswith("#")
        for line in merge_lines
    )
    return valid_vocab and valid_merges


def _verify_tokenizer_representation(snapshot_path: Path) -> None:
    sentencepiece_path = snapshot_path / "tokenizer.model"
    has_sentencepiece = (
        sentencepiece_path.is_file()
        and sentencepiece_path.stat().st_size > 0
    )
    if (
        _has_valid_tokenizer_json(snapshot_path)
        or has_sentencepiece
        or _has_valid_vocab_and_merges(snapshot_path)
    ):
        return
    raise ValueError(
        "snapshot is incomplete; no valid tokenizer representation: expected "
        "a nonempty valid tokenizer.json, a nonempty tokenizer.model, or both "
        "a valid vocab.json and nonempty merges.txt"
    )


def verify_hf_model_snapshot(snapshot_dir: str | os.PathLike[str]) -> None:
    """Verify model weights and one complete tokenizer representation."""
    snapshot_path = Path(snapshot_dir)
    if not snapshot_path.is_dir():
        raise ValueError(f"snapshot directory does not exist: {snapshot_path}")

    required_shards = _required_index_shards(snapshot_path)
    if required_shards:
        missing_or_empty = sorted(
            name
            for name in required_shards
            if not (snapshot_path / name).is_file()
            or (snapshot_path / name).stat().st_size == 0
        )
        if missing_or_empty:
            raise ValueError(
                "snapshot is incomplete; missing or empty required shard(s): "
                + ", ".join(missing_or_empty)
            )
        _verify_tokenizer_representation(snapshot_path)
        return

    weight_candidates = [
        path
        for pattern in (
            "model*.safetensors",
            "pytorch_model*.bin",
        )
        for path in snapshot_path.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not weight_candidates:
        raise ValueError(
            "snapshot is incomplete; no nonempty model weight file or weight index"
        )
    _verify_tokenizer_representation(snapshot_path)


def resolve_cached_hf_snapshot(model_id: str) -> str:
    """Return a verified immutable local snapshot without any remote lookup."""
    if not model_id:
        raise ValueError("model_id must be nonempty")
    if _is_local_model_ref(model_id):
        local_path = Path(os.path.expanduser(model_id)).resolve()
        verify_hf_model_snapshot(local_path)
        return str(local_path)

    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(
        model_id,
        local_files_only=True,
        ignore_patterns=_MODEL_IGNORE_PATTERNS,
    )
    verify_hf_model_snapshot(snapshot_dir)
    return str(Path(snapshot_dir).resolve())


def ensure_model_cached(
    model_id: str,
    retries: int = 3,
    log: Callable[[str], None] = print,
) -> None:
    """Prefetch and verify one pinned HF snapshot without deleting partial data."""
    if not model_id or _is_local_model_ref(model_id):
        return
    if retries < 1:
        raise ValueError(f"retries must be positive, got {retries!r}")

    workers = _configured_positive_int(
        _DOWNLOAD_WORKERS_ENV,
        _DEFAULT_DOWNLOAD_WORKERS,
    )
    backoff_seconds = _configured_nonnegative_float(
        _BACKOFF_SECONDS_ENV,
        _DEFAULT_BACKOFF_SECONDS,
    )
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import model_info, snapshot_download

    try:
        local_snapshot_dir = snapshot_download(
            model_id,
            local_files_only=True,
            ignore_patterns=_MODEL_IGNORE_PATTERNS,
        )
        verify_hf_model_snapshot(local_snapshot_dir)
        return
    except Exception:
        # A cache miss or incomplete partial snapshot both require the pinned
        # online path below. Keep partial files in place so HF can resume them.
        pass

    revision: str | None = None
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if revision is None:
                revision = getattr(model_info(model_id), "sha", None)
                if not revision:
                    raise ValueError(
                        f"Hugging Face returned no commit SHA for {model_id!r}"
                    )
            snapshot_dir = snapshot_download(
                model_id,
                revision=revision,
                max_workers=workers,
                force_download=False,
                local_files_only=False,
                ignore_patterns=_MODEL_IGNORE_PATTERNS,
            )
            verify_hf_model_snapshot(snapshot_dir)
            return
        except Exception as exc:
            last_error = exc
            log(
                f"      [hf-cache] model prefetch {attempt}/{retries} "
                f"for {model_id} at {revision or 'unresolved revision'} failed: "
                f"{str(exc)[:180]}"
            )
            if attempt < retries and backoff_seconds:
                time.sleep(backoff_seconds * attempt)

    raise HFSnapshotInfrastructureError(
        "Hugging Face model-cache infrastructure failure for "
        f"{model_id!r} after {retries} attempt(s): {last_error}"
    ) from last_error
