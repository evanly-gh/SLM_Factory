"""Baseline + fine-tuned + quantized eval for sub-billion models, outside the agent loop.

WHY A SEPARATE SCRIPT
    The question is narrow — "can these three models train, score and quantize in THIS pipeline,
    and what do they get on ner_bc5cdr and xlam_bfcl" — and the agent loop is the wrong instrument
    for it. A full run picks its own model, sets its own goal from the teacher, and spends hours on
    interventions. None of that is being asked. This does one fixed LoRA fit per (model, task) and
    reports the three numbers that answer the question:

        zero_shot   the base checkpoint, no adapter          (is there anything to build on?)
        finetuned   the same checkpoint after one LoRA fit    (does the pipeline move it?)
        quantized   that adapter merged and Q4_K_M'd          (does the deploy artifact survive?)

    Every one of those goes through the SAME `eval.harness.run_eval` and the SAME task spec the
    pipeline uses, so the numbers are comparable to run scores rather than to a private harness.

    Nothing here touches `config/android_pool.py`. These models are not pool members and selection
    cannot reach them; this measures whether they COULD be.

USAGE
    python scripts/probe_small_models.py --out logs/probes/small-models.json
    python scripts/probe_small_models.py --models google/gemma-3-270m --tasks ner_bc5cdr
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODELS = (
    "google/gemma-3-270m",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-135M",
)
DEFAULT_TASKS = ("ner_bc5cdr", "xlam_bfcl")

# One fixed configuration for every cell, deliberately. Comparing models needs the fit held
# constant; tuning per model would answer a different question. These are the pipeline's own
# cold-start defaults (`training/hparams.py`).
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
EPOCHS = 3


def _log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def _tokenizer_facts(model_id: str) -> dict:
    """What the pipeline needs to know about a candidate before it can train it at all."""
    facts: dict = {"model_id": model_id}
    try:
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        template = getattr(tokenizer, "chat_template", None)
        facts["vocab_size"] = len(tokenizer)
        facts["has_chat_template"] = bool(template)
        facts["eos_token"] = getattr(tokenizer, "eos_token", None)
        if template:
            facts["rendered_user_turn"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": "PROMPT"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        facts["architectures"] = list(getattr(config, "architectures", []) or [])
        facts["model_type"] = getattr(config, "model_type", None)
    except Exception as exc:
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _write_curriculum(rows: list[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _score(eval_set, weights_ref: str, base_model: str, gguf_path: str | None = None) -> dict:
    from eval.harness import run_eval

    result = run_eval(
        eval_set,
        weights_ref,
        base_model,
        quant="Q4_K_M" if gguf_path else None,
        # This probe builds GGUFs specifically (see `_quantize`), so it names its backend rather
        # than inheriting the run's: the artifact in hand is a GGUF whatever SLM_QUANT_BACKEND says.
        quant_artifact=gguf_path,
        quant_backend="llama_cpp",
    )
    return {
        "score": round(float(result.f1), 4),
        "format_valid": round(float(getattr(result, "format_valid", float("nan"))), 4),
        "n": len(getattr(eval_set, "all", []) or []),
    }


def _quantize(checkpoint: str, base_model: str, workdir: str) -> tuple[str | None, dict]:
    """Merge the adapter, build a Q4_K_M GGUF, and load-validate it. Returns (path, report).

    Same three calls `agent/nodes/evaluate._build_or_reuse_quant_artifact` makes, in the same order, minus
    the cache. `validate_and_record_gguf` is the part that answers "can we quantize this" — it
    loads the file through llama-cpp-python, so a converter that emits a structurally broken
    artifact fails here rather than showing up as a mystery zero score.
    """
    # `merge_for_quantization` lives in lora_trainer, not quantize — it needs Unsloth to load and
    # merge the adapter, which is the trainer's dependency, not the converter's.
    from training.lora_trainer import merge_for_quantization
    from training.quantize import quantize_from_model_spec, validate_and_record_gguf

    report: dict = {}
    started = time.time()
    merged = merge_for_quantization(
        checkpoint,
        os.path.join(workdir, "merged"),
        base_model_id=base_model,
    )
    report["merge_seconds"] = round(time.time() - started, 1)

    started = time.time()
    out_dir = os.path.join(workdir, "gguf")
    os.makedirs(out_dir, exist_ok=True)
    gguf_path = quantize_from_model_spec(merged, out_dir, "Q4_K_M")
    report["quantize_seconds"] = round(time.time() - started, 1)
    report["gguf_path"] = gguf_path
    if gguf_path and os.path.isfile(gguf_path):
        report["gguf_mb"] = round(os.path.getsize(gguf_path) / (1024 * 1024), 1)

    started = time.time()
    validate_and_record_gguf(gguf_path, base_model=base_model)
    report["validate_seconds"] = round(time.time() - started, 1)
    report["load_validated"] = True
    return gguf_path, report


def probe(model_id: str, task_name: str, args) -> dict:
    from data.eval_set import build_eval_set
    from tasks import get_task
    from training.slm_helpers import train

    cell: dict = {"model": model_id, "task": task_name}
    spec = get_task(task_name)
    workdir = os.path.join(
        args.workdir, f"{model_id.replace('/', '_')}__{task_name}"
    )
    os.makedirs(workdir, exist_ok=True)

    _log(f"── {model_id} × {task_name} ──")
    _log("  loading task data")
    train_rows, eval_rows = spec.load(max_train=args.train_rows, max_test=args.eval_rows)
    eval_set = build_eval_set(eval_rows, task_name, target=args.eval_rows)
    cell["train_rows"] = len(train_rows)
    cell["eval_rows"] = len(getattr(eval_set, "all", []) or [])
    _log(f"  train={cell['train_rows']} eval={cell['eval_rows']} metric={spec.metric_name}")

    # 1 — zero shot. weights_ref == model_id resolves to the base model with no adapter, which is
    # exactly what `evaluate._baseline` does, so this number is comparable to a run's baseline.
    if args.skip_zero_shot:
        cell["zero_shot"] = {"skipped": True}
    else:
        try:
            started = time.time()
            cell["zero_shot"] = _score(eval_set, model_id, model_id)
            cell["zero_shot"]["seconds"] = round(time.time() - started, 1)
            _log(f"  zero-shot  {spec.metric_name}={cell['zero_shot']['score']}")
        except Exception as exc:
            cell["zero_shot"] = {"error": f"{type(exc).__name__}: {exc}"}
            cell["zero_shot_traceback"] = traceback.format_exc()
            _log(f"  zero-shot  FAILED {type(exc).__name__}: {exc}")

    # 2 — one LoRA fit on the task's own gold rows. No mining, no synthesis, no interventions:
    # this is the cold-start curriculum and nothing else.
    curriculum = _write_curriculum(train_rows, os.path.join(workdir, "curriculum.jsonl"))
    # Reuse a checkpoint this workdir already holds. Training is by far the most expensive step
    # here, so a rerun that only needs the steps AFTER it — a quantization pass over cells that
    # already trained, say — should not pay for it again. Pass --retrain to force a fresh fit.
    existing = os.path.join(workdir, "training", "final_checkpoint")
    if not args.retrain and os.path.isdir(existing):
        checkpoint = existing
        cell["train"] = {"reused": True, "checkpoint": checkpoint}
        _log(f"  reusing existing checkpoint → {checkpoint}")
    else:
        try:
            started = time.time()
            output = train(
                dataset_path=curriculum,
                base_model=model_id,
                nr_epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                lora_rank=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                output_dir=os.path.join(workdir, "training"),
                task=task_name,
            )
            checkpoint = output.weights_ref
            cell["train"] = {
                "seconds": round(time.time() - started, 1),
                "checkpoint": checkpoint,
            }
            _log(f"  trained in {cell['train']['seconds']}s → {checkpoint}")
        except Exception as exc:
            cell["train"] = {"error": f"{type(exc).__name__}: {exc}"}
            cell["train_traceback"] = traceback.format_exc()
            _log(f"  train      FAILED {type(exc).__name__}: {exc}")
            return cell

    # 3 — fine-tuned, bf16. Separated from the quantized number on purpose: if they differ, the
    # question "did the model learn" and the question "did quantization survive" have different
    # answers and conflating them is how a GGUF defect gets read as a capability limit.
    try:
        started = time.time()
        cell["finetuned"] = _score(eval_set, checkpoint, model_id)
        cell["finetuned"]["seconds"] = round(time.time() - started, 1)
        _log(f"  fine-tuned {spec.metric_name}={cell['finetuned']['score']}")
    except Exception as exc:
        cell["finetuned"] = {"error": f"{type(exc).__name__}: {exc}"}
        cell["finetuned_traceback"] = traceback.format_exc()
        _log(f"  fine-tuned FAILED {type(exc).__name__}: {exc}")

    # 4 — the deployed artifact.
    if args.skip_quant:
        cell["quantized"] = {"skipped": True}
        return cell
    try:
        gguf_path, report = _quantize(checkpoint, model_id, workdir)
        cell["quantize"] = report
        _log(f"  quantized  {report.get('gguf_mb')}MB in {report.get('quantize_seconds')}s")
        started = time.time()
        cell["quantized"] = _score(eval_set, checkpoint, model_id, gguf_path=gguf_path)
        cell["quantized"]["seconds"] = round(time.time() - started, 1)
        _log(f"  quantized  {spec.metric_name}={cell['quantized']['score']}")
    except Exception as exc:
        cell["quantized"] = {"error": f"{type(exc).__name__}: {exc}"}
        cell["quantized_traceback"] = traceback.format_exc()
        _log(f"  quantize   FAILED {type(exc).__name__}: {exc}")

    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--eval-rows", type=int, default=300)
    parser.add_argument("--train-rows", type=int, default=3000)
    parser.add_argument("--skip-quant", action="store_true")
    parser.add_argument("--skip-zero-shot", action="store_true",
                        help="Skip the baseline pass. It is the slowest step by far — an untrained "
                             "model rarely emits a stop token, so every row runs the full output "
                             "reserve — and it is pure waste on a rerun that already has it.")
    parser.add_argument("--retrain", action="store_true",
                        help="Fit even when the workdir already holds a checkpoint.")
    parser.add_argument("--workdir", default="logs/probes/small-models")
    parser.add_argument("--out", default="logs/probes/small-models.json")
    args = parser.parse_args()

    results: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "eval_rows": args.eval_rows,
            "train_rows": args.train_rows,
        },
        "models": {},
        "cells": [],
    }

    _log("=== tokenizer / architecture facts ===")
    for model_id in args.models:
        facts = _tokenizer_facts(model_id)
        results["models"][model_id] = facts
        _log(
            f"  {model_id}: arch={facts.get('architectures')} "
            f"vocab={facts.get('vocab_size')} chat_template={facts.get('has_chat_template')}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for model_id in args.models:
        for task_name in args.tasks:
            try:
                cell = probe(model_id, task_name, args)
            except Exception as exc:
                cell = {
                    "model": model_id,
                    "task": task_name,
                    "fatal": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                _log(f"  FATAL {type(exc).__name__}: {exc}")
            results["cells"].append(cell)
            # Written after every cell so a job killed at hour six still reports hours one to five.
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)

    _log("=== SUMMARY ===")
    header = f"  {'model':38s} {'task':14s} {'zero':>8s} {'ft':>8s} {'q4':>8s}"
    _log(header)
    for cell in results["cells"]:
        def _get(key):
            value = cell.get(key) or {}
            if "score" in value:
                return f"{value['score']:.4f}"
            if value.get("skipped"):
                return "skip"
            return "ERR"
        _log(
            f"  {cell['model']:38s} {cell['task']:14s} "
            f"{_get('zero_shot'):>8s} {_get('finetuned'):>8s} {_get('quantized'):>8s}"
        )
    _log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
