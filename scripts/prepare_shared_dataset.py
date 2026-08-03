#!/usr/bin/env python
"""
Build ONE frozen dataset + eval set for a task and save it, so multiple strategy-comparison
runs load IDENTICAL data (B161). This removes the acquisition-nondeterminism confound that
previously gave each model-selection strategy a different dataset/eval set.

Point the runs at the output with SLM_SHARED_DATASET_DIR=<out_dir>.

Usage:
    python scripts/prepare_shared_dataset.py --task "<natural-language task>" --out <dir>
    # optional: --curriculum 2000 --eval 800
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


def _build_requested_eval_set(examples, plan, eval_size):
    """Build the frozen eval set at the same dynamic size target as runs."""
    from agent.nodes.cold_start.eval_setup import _eval_target
    from data.eval_set import build_eval_set

    return build_eval_set(
        examples,
        task_type=plan["task_type"],
        target=_eval_target(eval_size),
        multi_label=plan.get("multi_label", False),
        schema=plan.get("schema"),
        multilingual=plan.get("multilingual", False),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_shared_bundle(out, train, test, *, plan, difficulty, meta):
    """Write and integrity-seal one frozen train/eval bundle."""
    out = Path(out)
    task_type = plan["task_type"]
    required = required_fields_for_task(task_type)
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
        "task_type": task_type,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="natural-language task description")
    ap.add_argument("--out", required=True, help="output dir for the frozen dataset")
    ap.add_argument("--curriculum", type=int, default=None)
    ap.add_argument("--eval", type=int, default=None)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJ, ".env"))

    from agent.task_planner import plan_task
    from config.android_pool import ANDROID_POOL
    from config.config import CURRICULUM_SIZE_FLOOR, EVAL_SET_SIZE, DATA_SIZE_CEILING
    from data.loaders.web_acquire import acquire_dataset
    from agent.nodes.test_agent import label_difficulty
    from agent.nodes.cold_start.hardware_filter import run_hardware_filter
    from agent.nodes.cold_start.hardware_research import research_device

    print(f"[shared] planning task: {args.task}")
    plan = plan_task(args.task, model_pool=ANDROID_POOL)
    task_type = plan["task_type"]
    curriculum = args.curriculum or max(int(plan.get("curriculum_size") or CURRICULUM_SIZE_FLOOR),
                                        CURRICULUM_SIZE_FLOOR)
    eval_size = args.eval or max(int(plan.get("eval_size") or EVAL_SET_SIZE), EVAL_SET_SIZE)
    curriculum = min(curriculum, DATA_SIZE_CEILING)
    eval_size = min(eval_size, DATA_SIZE_CEILING)
    gold_target = int(curriculum * 0.65)
    bench_train = min(int(gold_target * 1.15) + 40, DATA_SIZE_CEILING)
    print(f"[shared] task_type={task_type} curriculum={curriculum} (train≤{bench_train}) eval={eval_size}")

    meta = {}
    train, test = acquire_dataset(
        plan, description=args.task, target_examples=max(gold_target, 120),
        benchmark_max_train=bench_train, benchmark_max_test=eval_size, meta=meta,
    )
    print(f"[shared] acquired train={len(train)} test={len(test)} source={meta.get('source')}")

    eval_set = _build_requested_eval_set(
        test,
        plan,
        eval_size,
    )

    # Difficulty labeling needs the feasible pool (device-independent here → use whole pool).
    HW, _ = research_device(args.task, log=print)
    feasible = sorted(run_hardware_filter(HW), key=lambda m: m.size_mb, reverse=True)
    difficulty = label_difficulty(eval_set, feasible, task_type, log=print)

    out = args.out
    _write_shared_bundle(
        out, train, eval_set.all, plan=plan, difficulty=difficulty, meta=meta
    )
    print(f"[shared] wrote frozen dataset to {out}. Point runs at it with SLM_SHARED_DATASET_DIR={out}")


if __name__ == "__main__":
    main()
