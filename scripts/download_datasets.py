#!/usr/bin/env python
"""Materialize task-aware Hugging Face benchmarks as offline JSONL bundles.

Each bundle contains only rows derived from the source's official ``train`` and ``test``
splits:

    data/local/<name>/train.jsonl
    data/local/<name>/test.jsonl
    data/local/<name>/manifest.json

The manifest records source/config/split lineage, row counts, schema version, the held-out
evaluation source restriction, and normalized-text contamination checks. Some public mirrors
contain rows duplicated across their official train/test files. Those rows are removed from
train (the official test split is kept intact), recorded in the manifest, and the final
bundle is required to have zero case/whitespace-normalized ``text`` overlap.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from data.loaders.dataset_integrity import (  # noqa: E402
    NORMALIZATION_VERSION,
    normalized_text_overlap,
    remove_normalized_train_overlap,
    required_fields_for_task,
    sha256_file,
    validate_rows,
)
from data.loaders.apps import APPS_SOURCE_REVISION  # noqa: E402

SCHEMA_VERSION = 2
_BC5CDR_TAG_NAMES = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]


def _hf_source(
    hf_id: str,
    *,
    config: str | None = None,
    splits: dict[str, str] | None = None,
    **extra,
) -> dict:
    return {
        "id": hf_id,
        "config": config,
        "splits": splits or {"train": "train", "test": "test"},
        "url": f"https://huggingface.co/datasets/{hf_id}",
        **extra,
    }


# The first source is preferred. Later entries are deterministic mirrors/compatibility
# paths and are attempted only when the preceding source cannot be loaded.
_DATASETS = [
    {
        "name": "emotion",
        "task_type": "classification",
        "converter": "classification",
        "text_col": "text",
        "label_col": "label",
        "row_schema": {"required": ["text", "label"], "optional": []},
        "sources": [_hf_source("dair-ai/emotion")],
    },
    {
        "name": "go_emotions",
        "task_type": "classification",
        "converter": "classification",
        "text_col": "text",
        "label_col": "labels",
        "multilabel_take_first": True,
        "row_schema": {"required": ["text", "label"], "optional": []},
        "sources": [_hf_source("google-research-datasets/go_emotions", config="simplified")],
    },
    {
        "name": "bc5cdr",
        "task_type": "NER",
        "converter": "bc5cdr",
        "row_schema": {"required": ["text", "entities"], "optional": []},
        "sources": [
            _hf_source("tner/bc5cdr", tokens_col="tokens", tags_col="tags"),
            _hf_source("spyysalo/bc5cdr", tokens_col="tokens", tags_col="ner_tags"),
            # datasets>=4 no longer executes dataset scripts. T-NER's official JSON files
            # remain public, so this script-free compatibility path reads the same repo and
            # the same official train/test files.
            _hf_source(
                "tner/bc5cdr",
                loader="json",
                data_files={
                    "train": (
                        "https://huggingface.co/datasets/tner/bc5cdr/"
                        "resolve/main/dataset/train.json"
                    ),
                    "test": (
                        "https://huggingface.co/datasets/tner/bc5cdr/"
                        "resolve/main/dataset/test.json"
                    ),
                },
                tokens_col="tokens",
                tags_col="tags",
            ),
        ],
    },
    {
        "name": "gsm8k",
        "task_type": "math_reasoning",
        "converter": "gsm8k",
        "row_schema": {
            "required": ["text", "answer", "cot_reasoning", "label"],
            "optional": [],
        },
        "sources": [_hf_source("openai/gsm8k", config="main")],
    },
    {
        "name": "apps",
        "task_type": "code_generation",
        "converter": "apps",
        "row_schema": {
            "required": [
                "text",
                "starter_code",
                "difficulty",
                "input_output",
                "execution_mode",
                "label",
            ],
            "optional": [
                "answer",
                "code",
                "solutions",
                "fn_name",
                "entry_point",
                "problem_id",
                "url",
                "gold_validation_status",
                "runner_compatible",
            ],
        },
        "split_row_schema": {
            "train": {"required": ["answer", "solutions"]},
            "test": {"required": []},
        },
        "source_counts": {"train": 2639, "test": 1000},
        "sources": [
            _hf_source(
                "codeparrot/apps",
                config="introductory",
                revision=APPS_SOURCE_REVISION,
            ),
            # datasets>=4 refuses script-based repositories. Stream the same official
            # JSONL files and apply the builder's introductory filter in our converter.
            _hf_source(
                "codeparrot/apps",
                config="introductory",
                loader="json",
                streaming=True,
                data_files={
                    "train": (
                        "https://huggingface.co/datasets/codeparrot/apps/"
                        f"resolve/{APPS_SOURCE_REVISION}/train.jsonl"
                    ),
                    "test": (
                        "https://huggingface.co/datasets/codeparrot/apps/"
                        f"resolve/{APPS_SOURCE_REVISION}/test.jsonl"
                    ),
                },
                revision=APPS_SOURCE_REVISION,
            ),
        ],
    },
    {
        "name": "mbpp",
        "task_type": "code_generation",
        "converter": "mbpp",
        "row_schema": {
            "required": ["text", "answer", "code", "test_list", "label"],
            "optional": ["test_imports", "task_id"],
        },
        "sources": [
            _hf_source("google-research-datasets/mbpp", config="sanitized"),
        ],
    },
    {
        "name": "samsum",
        "task_type": "generation",
        "converter": "samsum",
        "row_schema": {"required": ["text", "answer", "label"], "optional": []},
        "sources": [
            _hf_source("samsum"),
            _hf_source("knkarthick/samsum"),
        ],
    },
]
DATASETS_BY_NAME = {spec["name"]: spec for spec in _DATASETS}


def _label_names(ds, col: str) -> list[str] | None:
    features = getattr(ds, "features", None)
    feat = features.get(col) if features is not None else None
    names = getattr(feat, "names", None)
    if names is None:
        names = getattr(getattr(feat, "feature", None), "names", None)
    return list(names) if names else None


def _bio_to_entities(tokens: list, tags: list, names: list[str] | None) -> tuple[str, list[dict]]:
    text = " ".join(str(token) for token in tokens)
    entities: list[dict] = []
    current_tokens: list[str] = []
    current_type: str | None = None

    def flush() -> None:
        nonlocal current_tokens, current_type
        if current_tokens and current_type:
            entities.append({"text": " ".join(current_tokens), "type": current_type})
        current_tokens, current_type = [], None

    for token, raw_tag in zip(tokens, tags):
        if isinstance(raw_tag, int):
            label = names[raw_tag] if names and raw_tag < len(names) else str(raw_tag)
        else:
            label = str(raw_tag)
        if label in ("", "0", "O"):
            flush()
            continue
        prefix, separator, entity_type = label.partition("-")
        entity_type = entity_type if separator else label
        if prefix == "B" or entity_type != current_type:
            flush()
            current_tokens, current_type = [str(token)], entity_type
        else:
            current_tokens.append(str(token))
    flush()
    return text, entities


def _convert_classification(ds, spec: dict) -> tuple[list[dict], list[str]]:
    text_col, label_col = spec["text_col"], spec["label_col"]
    names = _label_names(ds, label_col)
    out, labels_seen = [], set()
    for example in ds:
        raw = example.get(label_col)
        if spec.get("multilabel_take_first"):
            if not raw:
                continue
            raw = raw[0]
        label = names[raw] if isinstance(raw, int) and names else raw
        text = example.get(text_col)
        if text and label is not None:
            out.append({"text": str(text), "label": str(label)})
            labels_seen.add(str(label))
    return out, sorted(labels_seen)


def _convert_bc5cdr(ds, source: dict | None = None) -> tuple[list[dict], list[str]]:
    source = source or {}
    tokens_col = source.get("tokens_col", "tokens")
    tags_col = source.get("tags_col")
    if tags_col is None:
        tags_col = "tags"
        # Hugging Face Dataset and ordinary test lists are reiterable. Materialize only
        # one-shot iterables so schema detection never consumes or duplicates their first row.
        if not hasattr(ds, "__getitem__"):
            ds = list(ds)
        first = ds[0] if len(ds) else None
        if first is not None and tags_col not in first:
            tags_col = "ner_tags"
    names = _label_names(ds, tags_col) or _BC5CDR_TAG_NAMES
    out, labels_seen = [], set()
    for example in ds:
        tokens, tags = example.get(tokens_col), example.get(tags_col)
        if not tokens or tags is None:
            continue
        text, entities = _bio_to_entities(tokens, tags, names)
        if text.strip():
            out.append({"text": text, "entities": entities})
            labels_seen.update(entity["type"] for entity in entities)
    return out, sorted(labels_seen)


def _split_gsm8k_answer(raw_answer: object) -> tuple[str, str]:
    answer = str(raw_answer or "")
    if "####" not in answer:
        return answer.strip(), ""
    reasoning, _, final = answer.rpartition("####")
    return final.strip(), reasoning.strip()


def _convert_gsm8k(ds) -> tuple[list[dict], list[str]]:
    out = []
    for example in ds:
        question = str(example.get("question") or "").strip()
        if not question or example.get("answer") is None:
            continue
        answer, reasoning = _split_gsm8k_answer(example["answer"])
        if answer:
            out.append(
                {
                    "text": question,
                    "answer": answer,
                    "cot_reasoning": reasoning,
                    "label": "math_reasoning",
                }
            )
    return out, ["math_reasoning"] if out else []


def _convert_mbpp(ds) -> tuple[list[dict], list[str]]:
    out = []
    for example in ds:
        prompt = str(example.get("prompt") or "").strip()
        code = str(example.get("code") or "").strip()
        tests = example.get("test_list")
        if not prompt or not code or not isinstance(tests, (list, tuple)):
            continue
        out.append(
            {
                "text": prompt,
                "answer": code,
                "code": code,
                "test_imports": list(example.get("test_imports") or []),
                "test_list": list(tests),
                "task_id": example.get("task_id"),
                "label": "code_generation",
            }
        )
    return out, ["code_generation"] if out else []


def _convert_apps(
    ds,
    split: str | None = None,
    gold_validator=None,
    conversion_stats: dict | None = None,
) -> tuple[list[dict], list[str]]:
    from data.loaders.apps import convert_apps_rows

    return convert_apps_rows(
        ds,
        split=split,
        gold_validator=gold_validator,
        conversion_stats=conversion_stats,
    )


def _convert_samsum(ds) -> tuple[list[dict], list[str]]:
    out = []
    for example in ds:
        dialogue = str(example.get("dialogue") or "").strip()
        summary = str(example.get("summary") or "").strip()
        if dialogue and summary:
            out.append({"text": dialogue, "answer": summary, "label": "generation"})
    return out, ["generation"] if out else []


def _convert(
    ds,
    spec: dict,
    source: dict | None = None,
    *,
    split: str | None = None,
    gold_validator=None,
    conversion_stats: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """Convert one source split to the pipeline's task-specific row schema."""
    converter = spec.get("converter", "classification")
    if converter == "classification":
        return _convert_classification(ds, spec)
    if converter == "bc5cdr":
        return _convert_bc5cdr(ds, source)
    if converter == "gsm8k":
        return _convert_gsm8k(ds)
    if converter == "apps":
        return _convert_apps(
            ds,
            split=split,
            gold_validator=gold_validator,
            conversion_stats=conversion_stats,
        )
    if converter == "mbpp":
        return _convert_mbpp(ds)
    if converter == "samsum":
        return _convert_samsum(ds)
    raise ValueError(f"unknown converter {converter!r}")


def _drop_train_overlap(
    train_rows: list[dict],
    test_rows: list[dict],
    spec: dict | None = None,
) -> tuple[list[dict], int]:
    """Keep the official test split fixed and remove normalized duplicates from train."""
    removed = 0
    if spec and spec.get("name") == "apps":
        from data.loaders.apps import remove_apps_train_fingerprint_overlap

        train_rows, fingerprint_removed = remove_apps_train_fingerprint_overlap(
            train_rows,
            test_rows,
        )
        removed += fingerprint_removed
    train_rows, text_removed = remove_normalized_train_overlap(
        train_rows,
        test_rows,
    )
    return train_rows, removed + text_removed


def _source_descriptor(source: dict, source_revision: str | None = None) -> dict:
    descriptor = {
        "id": source["id"],
        "config": source.get("config"),
        "splits": dict(source["splits"]),
        "url": source.get("url"),
        "loader": source.get("loader", "datasets"),
    }
    revision = source_revision or source.get("revision")
    if revision:
        descriptor["revision"] = revision
    return descriptor


def _build_manifest(
    spec: dict,
    source: dict,
    train_rows: list[dict],
    test_rows: list[dict],
    labels: list[str],
    removed_train_overlap: int = 0,
    fingerprint_removed_from_train: int = 0,
    conversion_stats: dict | None = None,
    source_revision: str | None = None,
    content_hashes: dict[str, str] | None = None,
) -> dict:
    source_url = source.get("url") or f"https://huggingface.co/datasets/{source['id']}"
    content_hashes = dict(content_hashes or {})
    records = [
        {
            "kind": "hf",
            "id": source["id"],
            "config": source.get("config"),
            "revision": source_revision,
            "split": source["splits"]["train"],
            "url": source_url,
            "role": "curriculum",
        },
        {
            "kind": "hf",
            "id": source["id"],
            "config": source.get("config"),
            "revision": source_revision,
            "split": source["splits"]["test"],
            "url": source_url,
            "role": "eval",
        },
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": spec["name"],
        "task_type": spec["task_type"],
        # Compatibility fields consumed by the pre-manifest-v1 local loader.
        "hf_id": source["id"],
        "config": source.get("config"),
        "source_revision": source_revision,
        "source_url": source_url,
        "source_splits": dict(source["splits"]),
        "labels": sorted(set(labels)),
        "row_schema": spec["row_schema"],
        "counts": {
            "train": len(train_rows),
            "test": len(test_rows),
            "total": len(train_rows) + len(test_rows),
        },
        "source": _source_descriptor(source, source_revision),
        "source_candidates": [_source_descriptor(item) for item in spec["sources"]],
        "provenance": {
            "provider": "huggingface",
            "official_splits_only": True,
            "records": records,
        },
        "eval_ban": [dict(records[1])],
        "overlap": {
            "field": "text",
            "normalization": NORMALIZATION_VERSION,
            "normalized_count": 0,
            "removed_from_train": removed_train_overlap,
        },
        "integrity": {
            "algorithm": "sha256",
            "checksum_file": "checksums.sha256",
            "files": content_hashes,
        },
    }
    if spec.get("name") == "apps":
        manifest["overlap"][
            "fingerprint_removed_from_train"
        ] = fingerprint_removed_from_train
        source_counts = dict(spec["source_counts"])
        manifest["filtering"] = {
            "source_counts": source_counts,
            "removed_unusable": {
                "train": source_counts["train"]
                - len(train_rows)
                - removed_train_overlap,
                "test": source_counts["test"] - len(test_rows),
            },
            "conversion": dict(conversion_stats or {}),
        }
    if spec.get("split_row_schema"):
        manifest["split_row_schema"] = spec["split_row_schema"]
    return manifest


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_bundle(
    output_dir: str | Path,
    spec: dict,
    source: dict,
    train_rows: list[dict],
    test_rows: list[dict],
    *,
    labels: list[str],
    removed_train_overlap: int = 0,
    fingerprint_removed_from_train: int = 0,
    conversion_stats: dict | None = None,
    source_revision: str | None = None,
) -> dict:
    common_required = tuple(spec["row_schema"]["required"])
    canonical_required = required_fields_for_task(spec["task_type"])
    for split, rows in (("train", train_rows), ("test", test_rows)):
        validate_rows(
            rows,
            common_required,
            bundle_name=spec["name"],
            split=split,
        )
        validate_rows(
            rows,
            canonical_required,
            bundle_name=spec["name"],
            split=split,
        )
        split_required = set(
            ((spec.get("split_row_schema") or {}).get(split) or {}).get(
                "required",
                [],
            )
        )
        for index, row in enumerate(rows):
            missing = split_required - set(row)
            if missing:
                raise ValueError(
                    f"{spec['name']}: {split} row {index} missing split schema "
                    f"fields {sorted(missing)}"
                )

    if spec["name"] == "apps":
        from data.loaders.apps import apps_fingerprint_overlap

        fingerprint_overlap = apps_fingerprint_overlap(
            train_rows,
            test_rows,
        )
        if fingerprint_overlap:
            raise ValueError(
                f"{spec['name']}: train/test URL or solution fingerprint "
                f"overlap ({len(fingerprint_overlap)} fingerprints)"
            )
    overlap = normalized_text_overlap(train_rows, test_rows)
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(
            f"{spec['name']}: normalized train/test text overlap ({len(overlap)} rows), "
            f"sample={sample!r}"
        )
    if not train_rows or not test_rows:
        raise ValueError(f"{spec['name']}: train and test must both contain converted rows")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / spec["name"]

    temporary = Path(tempfile.mkdtemp(prefix=f".{spec['name']}-", dir=output_dir))
    try:
        _write_jsonl(temporary / "train.jsonl", train_rows)
        _write_jsonl(temporary / "test.jsonl", test_rows)
        content_hashes = {
            filename: sha256_file(temporary / filename)
            for filename in ("train.jsonl", "test.jsonl")
        }
        manifest = _build_manifest(
            spec,
            source,
            train_rows,
            test_rows,
            labels,
            removed_train_overlap=removed_train_overlap,
            fingerprint_removed_from_train=fingerprint_removed_from_train,
            conversion_stats=conversion_stats,
            source_revision=source_revision,
            content_hashes=content_hashes,
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        checksum_lines = [
            f"{sha256_file(temporary / filename)}  {filename}"
            for filename in ("train.jsonl", "test.jsonl", "manifest.json")
        ]
        (temporary / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        destination.mkdir(parents=True, exist_ok=True)
        # Publish the manifest last so readers never observe a v2 manifest without all
        # integrity-covered files.
        for filename in ("train.jsonl", "test.jsonl", "checksums.sha256", "manifest.json"):
            os.replace(temporary / filename, destination / filename)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return manifest


def _cached_hf_revision(hf_id: str) -> str | None:
    """Read the Hub's cached ``main`` commit without making a network request."""
    cache_root = os.environ.get("HF_HUB_CACHE")
    if not cache_root:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_root = HF_HUB_CACHE
        except Exception:
            return None
    repo = "datasets--" + hf_id.replace("/", "--")
    ref = Path(cache_root) / repo / "refs" / "main"
    try:
        revision = ref.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    if len(revision) < 7 or any(char not in "0123456789abcdef" for char in revision):
        return None
    return revision


def _hf_revision(hf_id: str) -> str | None:
    """Resolve a source commit, preferring the already-populated local Hub ref."""
    cached = _cached_hf_revision(hf_id)
    if cached:
        return cached
    try:
        from huggingface_hub import HfApi

        revision = str(
            HfApi().dataset_info(hf_id, files_metadata=False).sha or ""
        ).strip().lower()
    except Exception:
        return None
    if len(revision) < 7 or any(char not in "0123456789abcdef" for char in revision):
        return None
    return revision


def _load_source_split(
    load_dataset_fn: Callable,
    source: dict,
    our_split: str,
):
    source_split = source["splits"][our_split]
    if source.get("loader") == "json":
        return load_dataset_fn(
            "json",
            data_files={source_split: source["data_files"][our_split]},
            split=source_split,
            streaming=bool(source.get("streaming")),
        )
    config = source.get("config")
    kwargs = {}
    if source.get("revision"):
        kwargs["revision"] = source["revision"]
    if config is not None:
        return load_dataset_fn(
            source["id"],
            config,
            split=source_split,
            **kwargs,
        )
    return load_dataset_fn(source["id"], split=source_split, **kwargs)


def _download_dataset(
    spec: dict,
    output_dir: str | Path,
    load_dataset_fn: Callable | None = None,
) -> dict:
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset

    failures = []
    for source in spec["sources"]:
        source_label = f"{source['id']} (config={source.get('config')}, loader={source.get('loader', 'datasets')})"
        print(f"[download] {spec['name']}: trying {source_label}")
        try:
            raw_train = _load_source_split(load_dataset_fn, source, "train")
            raw_test = _load_source_split(load_dataset_fn, source, "test")
            gold_validator = None
            if spec["name"] == "apps":
                from eval.scorers.generation import _run_apps_tests

                def gold_validator(solution, row):
                    return _run_apps_tests(solution, row)

            train_stats: dict = {}
            test_stats: dict = {}
            train_rows, train_labels = _convert(
                raw_train,
                spec,
                source,
                split="train",
                gold_validator=gold_validator,
                conversion_stats=train_stats,
            )
            test_rows, test_labels = _convert(
                raw_test,
                spec,
                source,
                split="test",
                gold_validator=gold_validator,
                conversion_stats=test_stats,
            )
            fingerprint_removed = 0
            if spec["name"] == "apps":
                from data.loaders.apps import (
                    remove_apps_train_fingerprint_overlap,
                )

                train_rows, fingerprint_removed = (
                    remove_apps_train_fingerprint_overlap(
                        train_rows,
                        test_rows,
                    )
                )
            train_rows, text_removed = remove_normalized_train_overlap(
                train_rows,
                test_rows,
            )
            removed = fingerprint_removed + text_removed
            if removed:
                print(
                    f"    decontamination: removed {removed} held-out text/URL/"
                    "solution-overlap row(s) from official train"
                )
            labels = sorted(set(train_labels) | set(test_labels))
            source_revision = source.get("revision") or _hf_revision(
                source["id"]
            )
            manifest = _write_bundle(
                output_dir,
                spec,
                source,
                train_rows,
                test_rows,
                labels=labels,
                removed_train_overlap=removed,
                fingerprint_removed_from_train=fingerprint_removed,
                conversion_stats={
                    "train": train_stats,
                    "test": test_stats,
                },
                source_revision=source_revision,
            )
            print(
                f"    wrote {spec['name']}: train={manifest['counts']['train']} "
                f"test={manifest['counts']['test']} overlap=0 source={source['id']}"
            )
            return manifest
        except Exception as error:  # noqa: BLE001 - each declared mirror gets a chance
            failures.append(f"{source_label}: {error}")
            print(f"    unavailable: {source_label}: {str(error)[:240]}")
    raise RuntimeError(
        f"{spec['name']}: all declared sources failed:\n  " + "\n  ".join(failures)
    )


def _local_dataset_dir() -> Path:
    return Path(os.environ.get("SLM_LOCAL_DATASET_DIR", PROJ / "data" / "local"))


def _parse_only(raw_groups: list[list[str]] | None, parser: argparse.ArgumentParser) -> list[dict]:
    if not raw_groups:
        return list(_DATASETS)
    names = []
    for group in raw_groups:
        for value in group:
            names.extend(part.strip().lower().replace("-", "_") for part in value.split(","))
    selected, seen = [], set()
    for name in names:
        if name not in DATASETS_BY_NAME:
            parser.error(
                f"unknown dataset {name!r}; choose from {', '.join(DATASETS_BY_NAME)}"
            )
        if name not in seen:
            seen.add(name)
            selected.append(DATASETS_BY_NAME[name])
    return selected


def _list_catalog(output_dir: Path) -> None:
    for spec in _DATASETS:
        manifest_path = output_dir / spec["name"] / "manifest.json"
        installed_manifest = None
        if manifest_path.exists():
            try:
                installed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                installed_manifest = None
        item = {
            "name": spec["name"],
            "task_type": spec["task_type"],
            "sources": [
                {"id": source["id"], "config": source.get("config")}
                for source in spec["sources"]
            ],
            "installed": installed_manifest is not None,
            "counts": (installed_manifest or {}).get("counts"),
        }
        print(json.dumps(item, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        nargs="+",
        metavar="DATASET",
        help="download only these names (space/comma separated; option may repeat)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list supported datasets, task types, sources, and local install status",
    )
    args = parser.parse_args(argv)
    output_dir = _local_dataset_dir()

    if args.list:
        _list_catalog(output_dir)
        return 0

    selected = _parse_only(args.only, parser)
    failures = []
    for spec in selected:
        try:
            _download_dataset(spec, output_dir)
        except Exception as error:  # noqa: BLE001 - report every selected bundle
            failures.append((spec["name"], error))
            print(f"[download] FAILED {spec['name']}: {error}", file=sys.stderr)
    if failures:
        print(
            "[download] failed bundles: " + ", ".join(name for name, _ in failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
