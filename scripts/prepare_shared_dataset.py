#!/usr/bin/env python
"""
Build ONE frozen dataset + eval set for a task and save it, so multiple strategy-comparison
runs load IDENTICAL data (B161). This removes the acquisition-nondeterminism confound that
previously gave each model-selection strategy a different dataset/eval set.

Point the runs at the output with SLM_SHARED_DATASET_DIR=<out_dir>.

The task is a REGISTRY NAME, not a natural-language description. It used to be prose that was fed
to the orchestrator's planner, and the resulting plan's abstract `task_type` was written into the
manifest — but `agent/nodes/cold_start/eval_setup.py::_load_shared_dataset` now checks the
manifest's `task` against the run's task and reads the row schema from that task's spec, so a
bundle built from a channel is unloadable. Naming the task also means the bundle is built by the
same loader, with the same train/eval sizing, as the run it will feed.

Usage:
    python scripts/prepare_shared_dataset.py --task ner_bc5cdr --out <dir>
    # optional: --curriculum 5000 --eval 800
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from data.loaders.dataset_integrity import (  # noqa: E402
    NORMALIZATION_VERSION,
    normalized_text_overlap,
    remove_normalized_train_overlap,
    required_fields_for_task,
    sha256_file,
    validate_rows,
    write_checksum_sidecar,
)

SHARED_SCHEMA_VERSION = 1
SHARED_CONTENT_FILES = (
    "train.jsonl",
    "test.jsonl",
    "difficulty.json",
    "sources.json",
    "eval_ban.json",
    "plan.json",
)
SHARED_CHECKSUM_FILES = SHARED_CONTENT_FILES + ("manifest.json",)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_shared_bundle(out, train, test, *, task, plan, difficulty, meta):
    """Write and integrity-seal one frozen train/eval bundle."""
    out = Path(out)
    required = required_fields_for_task(task)
    validate_rows(train, required, bundle_name=out.name, split="train")
    validate_rows(test, required, bundle_name=out.name, split="test")
    overlap = normalized_text_overlap(train, test)
    if overlap:
        raise ValueError(
            f"{out.name}: normalized train/eval overlap ({len(overlap)} rows), "
            f"sample={sorted(overlap)[:3]!r}"
        )

    out.mkdir(parents=True, exist_ok=True)
    sources = list(meta.get("source_records") or [{
        "kind": "unknown", "id": meta.get("source"), "role": "source",
    }])
    eval_ban = list(meta.get("eval_ban") or [])
    _write_jsonl(out / "train.jsonl", list(train))
    _write_jsonl(out / "test.jsonl", list(test))
    for filename, value in (
        ("difficulty.json", difficulty or {}),
        ("sources.json", sources),
        ("eval_ban.json", eval_ban),
        ("plan.json", plan),
    ):
        (out / filename).write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    content_hashes = {
        filename: sha256_file(out / filename) for filename in SHARED_CONTENT_FILES
    }
    manifest = {
        "schema_version": SHARED_SCHEMA_VERSION,
        "bundle_type": "shared_dataset",
        # The registry task name. `_load_shared_dataset` refuses a bundle whose task does not
        # equal the run's, which is the check that stops one task's frozen data being served to
        # another — the failure the abstract channel could not detect, because several tasks
        # shared each channel.
        "task": task,
        "counts": {
            "train": len(train),
            "test": len(test),
            "total": len(train) + len(test),
        },
        "row_schema": {"required": list(required)},
        "source": meta.get("source", "unknown"),
        "source_records": sources,
        "eval_ban": eval_ban,
        "overlap": {
            "field": "text",
            "normalization": NORMALIZATION_VERSION,
            "normalized_count": 0,
        },
        "integrity": {
            "algorithm": "sha256",
            "checksum_file": "checksums.sha256",
            "files": content_hashes,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_checksum_sidecar(out, SHARED_CHECKSUM_FILES, "checksums.sha256")
    return manifest


def main():
    import tasks

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=tasks.task_names(),
                    help="registry task name to freeze")
    ap.add_argument("--out", required=True, help="output dir for the frozen dataset")
    ap.add_argument("--curriculum", type=int, default=None)
    ap.add_argument("--eval", type=int, default=None)
    args = ap.parse_args()

    # Before any loader import: xLAM is a GATED repo and resolves only via the HF_TOKEN kept in
    # .env, so a task whose loader needs it would otherwise fail at the download rather than here.
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJ, ".env"))

    from agent.nodes.cold_start.eval_setup import _eval_target
    from config.android_pool import ANDROID_POOL
    from data.eval_set import build_eval_set
    from agent.nodes.test_agent import label_difficulty

    spec = tasks.get_task(args.task)
    # The task's own initial cap, overridable for a one-off. There is no global floor or ceiling any
    # more; the curriculum grows by rebuild rather than aiming at a number.
    curriculum = int(args.curriculum or spec.initial_train_cap)
    # The task's own eval cap unless overridden, so the frozen eval set is the size the runs that
    # consume it would have built for themselves.
    eval_size = int(args.eval or spec.eval_cap)
    # Same sizing arithmetic as `eval_setup._load_named_benchmark`: the loader is asked for the
    # task's own share of the curriculum target as gold rows.
    max_train = max(int(curriculum * spec.train_fraction), 60)
    print(f"[shared] task={spec.name} ({spec.title}) curriculum={curriculum} "
          f"(train\u2264{max_train}) eval\u2264{eval_size}")

    train, test = spec.load(max_train=max_train, max_test=eval_size, log=print)
    # Stage-0 decontamination, exactly as the curated path does it: official splits are not
    # guaranteed disjoint (CLINC150 ships the same utterance in both under two intents), and the
    # bundle writer below treats any remaining overlap as fatal. Held-out rows are authoritative
    # and never modified; the training row is dropped.
    train, overlap_removed = remove_normalized_train_overlap(train, test)
    if overlap_removed:
        print(f"[shared] Stage-0 normalized overlap removal: dropped {overlap_removed} train row(s)")
    print(f"[shared] loaded train={len(train)} test={len(test)}")

    meta = {
        "source": spec.title,
        "source_records": [
            {"kind": "hf", "id": spec.name, "split": "train", "role": "curriculum"},
            {"kind": "hf", "id": spec.name, "split": "test", "role": "eval"},
        ],
        "eval_ban": [{"kind": "hf", "id": spec.name, "split": "test", "role": "eval"}],
    }

    eval_set = build_eval_set(test, task=spec.name, target=_eval_target(eval_size))

    # Difficulty labeling is device-independent here, so the whole Android pool is the candidate
    # set. The script used to resolve a device first by passing the TASK description to
    # `research_device`, which researches a phone — it spent an orchestrator call to be told
    # nothing, and the pool it produced was the fallback anyway.
    feasible = sorted(ANDROID_POOL, key=lambda m: m.size_mb, reverse=True)
    difficulty = label_difficulty(eval_set, feasible, spec.name, log=print)

    # Recorded for provenance only: nothing reads plan.json back, but it is covered by the
    # bundle's checksums, so it must exist and must describe what was frozen.
    plan = {
        "task": spec.name,
        "title": spec.title,
        "category": spec.category,
        "family": spec.family,
        "metric": spec.metric_name,
        "curriculum_size": curriculum,
        "eval_size": eval_size,
    }
    out = args.out
    _write_shared_bundle(
        out, train, eval_set.all, task=spec.name, plan=plan, difficulty=difficulty, meta=meta
    )
    print(f"[shared] wrote frozen dataset to {out}. Point runs at it with SLM_SHARED_DATASET_DIR={out}")


if __name__ == "__main__":
    main()
