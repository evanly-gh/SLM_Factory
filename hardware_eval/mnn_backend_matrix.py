#!/usr/bin/env python3
"""
Does the MNN backend work for EVERY model in the pool, at EVERY precision, on the GPU?

One row per (model, precision) the loop could ever ask MNN to build: export it, load-validate it,
score the same clinc150 rows on the GPU and on the CPU, and report speed and accuracy side by side.
The point is coverage and comparability, not a headline score — the headline accuracy number is
`tests/pipeline/verify_mnn_backend.slurm`, which A/Bs one adapter against llama.cpp.

WHY A MATRIX AND NOT A SPOT CHECK
    The loop walks a ladder: it starts at tier 1 and escalates, so a backend that works on
    SmolLM2-360M and fails on Qwen3-1.7B is a backend that dies three hours into a run. Every
    export is also an independent test of MNN's architecture support — `llmexport.py` has per-model
    branches, and a Qwen3.5 is not a SmolLM2 is not a Gemma.

WHY GPU AND CPU BOTH
    The GPU is there for speed (measured 6x on the 4-bit 360M), and the CPU is what a phone runs.
    Scoring both on the same rows is how "the GPU is faster" and "the GPU computes the same thing"
    become measurements rather than assumptions — and it is the reason the per-artifact build does
    NOT pay for a GPU-vs-CPU probe: this tool answers it once, thoroughly.

ONE SUBPROCESS PER (CELL, DEVICE), which is not an implementation detail but the only faithful
configuration. MNN's CUDA runtime does not survive a second model load in the same process: the
first version of this tool scored all 27 cells in one process and its whole CUDA column came back
0.0000 — cell 1 correct, cell 2 emitting `<|endoftext|>` repeatedly, cells 3-27 returning empty
strings at a nonsensical 878 rows/s. It reported "27/27 passed", because nothing raised. A child
per measurement matches how the pipeline actually runs MNN (one artifact per disposable CUDA
worker) and is why `slm_helpers._note_cuda_load` now refuses the second load outright.

AND A CELL THAT GENERATES NOTHING IS A FAILURE, not a pass. `status: ok` requires non-empty output
on a majority of rows; the first version's only criterion was "scoring did not raise", which is how
an entirely empty column passed review.

PRECISIONS
    q4 (Q4_K_M) and q8 (Q8_0) are the pool's quantized selectors. `FP16` is MNN's full-precision
    export, included because a backend claim should cover the unquantized case too; note the LOOP
    still scores the pool's `bf16` variants through Unsloth on both backends, exactly as before.

Usage:
    python hardware_eval/mnn_backend_matrix.py --rows 100
    python hardware_eval/mnn_backend_matrix.py --models Qwen/Qwen3-0.6B --precisions Q4_K_M
    python hardware_eval/mnn_backend_matrix.py --tier 1 --devices cuda
"""
import argparse
import json
import os
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

# Every precision the matrix can cover. Q4_K_M/Q8_0 are pool selectors; FP16 is MNN's own
# full-precision export and is not reachable from any pool entry (see training/quantize_mnn.py).
ALL_PRECISIONS = ("Q4_K_M", "Q8_0", "FP16")
ALL_DEVICES = ("cuda", "cpu")


def _pool_models() -> list[tuple[int, str]]:
    """(lowest tier, model_id) for every base model in the pool, smallest tier first.

    Tier comes from the pool rather than from a list here so that a model added to
    `config/android_pool.py` is covered by this matrix without anyone remembering to update it.
    """
    from config.android_pool import ANDROID_POOL

    lowest: dict[str, int] = {}
    for spec in ANDROID_POOL:
        if spec.model_id not in lowest or spec.tier < lowest[spec.model_id]:
            lowest[spec.model_id] = spec.tier
    return sorted((tier, model_id) for model_id, tier in lowest.items())


def _eval_rows(path: str, count: int):
    from data.eval_set import EvalSet

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("all") or payload["examples"]
    # The FULL set builds the prompts, then the first `count` are scored: a clinc150 prompt
    # enumerates its label space from the eval set, so truncating the rows first would shrink the
    # prompt from ~1,385 tokens to ~130 and measure a workload that does not exist.
    full = EvalSet(all=rows, task="clinc150")
    return full, EvalSet(all=rows[:count], task="clinc150")


def _score(full_set, eval_set, artifact: str, base_model: str, device: str) -> dict:
    """Score `eval_set` on one device, returning accuracy, speed and a sample prediction.

    Prompts come from `full_set` and are then sliced. A clinc150 prompt enumerates its label space
    from the eval set it is built with, so building from the truncated set produced a 283-token
    prompt instead of the real ~1,385-token one — a fifth of the work, and the wrong measurement.
    """
    os.environ["SLM_MNN_BACKEND_TYPE"] = device
    from tasks import get_task
    from training import slm_helpers

    # Re-resolved per cell: the backend is read at load time, and the cache is keyed on it, but a
    # long-lived process would otherwise hold a model loaded on the other device.
    slm_helpers.clear_inference_cache()
    spec = get_task("clinc150")
    prompts = spec.build_prompts(full_set)[:len(eval_set.all)]
    started = time.time()
    outputs = slm_helpers.infer_batch_mnn(
        prompts, artifact, max_new_tokens=spec.max_new_tokens,
        base_model=base_model, task="clinc150",
    )
    elapsed = time.time() - started
    predictions = spec.extract_predictions(outputs, eval_set)
    result = spec.score(eval_set, predictions)
    empty = sum(1 for out in outputs if not (out or "").strip())
    return {
        "device": device,
        "metric": round(float(result["f1"]), 4),
        "format_valid": round(float(result.get("format_valid", 1.0)), 4),
        "rows_per_s": round(len(prompts) / max(elapsed, 1e-9), 2),
        "seconds": round(elapsed, 1),
        "empty_rows": empty,
        # An all-empty column is the failure this tool shipped with once; recorded per cell so the
        # verdict is derived from evidence rather than from "nothing raised".
        "degenerate": empty > len(outputs) // 2,
        "sample": (outputs[0] or "").strip()[:40],
    }


def _one_cell(model_id: str, tier: int, precision: str, args, full_set, scored_set) -> dict:
    from training.quantize import resolve_hf_snapshot
    from training.quant_backend import artifact_size_mb
    from training.quantize_mnn import export_from_model_spec

    cell = {"tier": tier, "model": model_id, "precision": precision, "status": "failed"}
    safe = model_id.replace("/", "_")
    out_dir = os.path.join(args.out, safe)
    try:
        snapshot = resolve_hf_snapshot(model_id)
        started = time.time()
        artifact = export_from_model_spec(snapshot, out_dir, precision)
        cell["export_s"] = round(time.time() - started, 1)
        cell["size_mb"] = round(artifact_size_mb(artifact, "mnn"), 1)
    except Exception as error:  # noqa: BLE001 - a model that cannot export is a matrix result
        cell["error"] = f"export: {type(error).__name__}: {str(error)[:160]}"
        return cell

    # In a child, on the device it will be scored on, because validation LOADS the model — and a
    # second CUDA load in one process is refused (correctly: MNN's CUDA runtime does not survive
    # it). This also matches the pipeline, where validation happens inside the artifact-build
    # worker and scoring inside a separate eval worker.
    record = _child(["--validate-only", "--artifact", artifact, "--base", model_id,
                     "--precision", precision, "--device", args.devices[0]], args)
    if record.get("error"):
        cell["error"] = f"validate: {record['error']}"
        return cell
    cell["validated_bits"] = record.get("quant_bit")
    cell["validated_backend"] = record.get("validated_backend")

    for device in args.devices:
        cell[device] = _score_in_subprocess(artifact, model_id, device, args)

    # Derived from the measurements, not from the absence of an exception: a device that returned
    # empty text for most rows did not work, however quietly it did so.
    problems = [
        device for device in args.devices
        if cell[device].get("error") or cell[device].get("degenerate")
    ]
    cell["status"] = "ok" if not problems else "failed"
    if problems and not cell.get("error"):
        cell["error"] = "no usable output from: " + ", ".join(problems)
    if args.discard:
        from training.quant_backend import invalidate_cache

        invalidate_cache(artifact, "mnn")
    return cell


def _child(extra_args: list[str], args) -> dict:
    """Run this file in a FRESH interpreter for one measurement and return its JSON result.

    A child per model load, because MNN's CUDA runtime does not survive a second one (see the
    module docstring) — and because it is what the pipeline does anyway: one artifact per
    disposable CUDA worker.
    """
    import subprocess

    command = [
        sys.executable, os.path.abspath(__file__),
        "--eval-set", args.eval_set, "--rows", str(args.rows),
    ] + extra_args
    finished = subprocess.run(
        command, cwd=PROJ, capture_output=True, text=True, timeout=args.cell_timeout,
    )
    for line in finished.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    tail = (finished.stderr or finished.stdout or "").strip().splitlines()
    return {"error": f"child exited {finished.returncode}: "
                     + (tail[-1][:200] if tail else "no output")}


def _score_in_subprocess(artifact: str, model_id: str, device: str, args) -> dict:
    """Score one (artifact, device) in its own interpreter."""
    result = _child(["--score-only", "--artifact", artifact, "--base", model_id,
                     "--device", device], args)
    result.setdefault("device", device)
    return result


def _print_matrix(cells: list[dict], devices: tuple[str, ...]) -> None:
    header = f"{'tier':>4}  {'model':<38}{'prec':<8}{'MB':>8}{'export':>8}"
    for device in devices:
        header += f"{device + ' rows/s':>14}{device + ' metric':>14}"
    print("\n" + header)
    print("-" * len(header))
    for cell in cells:
        line = (f"{cell['tier']:>4}  {cell['model']:<38}{cell['precision']:<8}"
                f"{cell.get('size_mb', 0) or 0:>8.0f}{cell.get('export_s', 0) or 0:>8.0f}")
        for device in devices:
            entry = cell.get(device) or {}
            if entry.get("error") or entry.get("degenerate") or "metric" not in entry:
                line += f"{'FAIL':>14}{'-':>14}"
            else:
                line += f"{entry['rows_per_s']:>14.2f}{entry['metric']:>14.4f}"
        print(line)
        if cell.get("error"):
            print(f"      !! {cell['error']}")
        for device in devices:
            entry = cell.get(device) or {}
            if entry.get("error"):
                print(f"      !! {device}: {entry['error']}")


def parse_args():
    p = argparse.ArgumentParser(description="MNN backend coverage matrix: every tier, every precision")
    p.add_argument("--eval-set", default=os.path.join(
        PROJ, "logs/runs/slm-clinc150-ablation-scarce-synth-l40s-40175895/artifacts/eval_set.json"))
    p.add_argument("--rows", type=int, default=100,
                   help="eval rows scored per cell (prompts are always built from the FULL set, so "
                        "the prompt length is the real one)")
    p.add_argument("--models", default="", help="comma list; default is every pool model")
    p.add_argument("--precisions", default=",".join(ALL_PRECISIONS))
    p.add_argument("--devices", default=",".join(ALL_DEVICES))
    p.add_argument("--tier", type=int, default=0, help="only models whose lowest tier is this")
    p.add_argument("--out", default="/mmfs1/gscratch/intelligentsystems/evanly/tmp/mnn-matrix")
    p.add_argument("--report", default=os.path.join(PROJ, "logs/probes/mnn-backend-matrix.json"))
    p.add_argument("--discard", action="store_true",
                   help="delete each artifact after scoring it (the full matrix is ~40GB otherwise)")
    p.add_argument("--cell-timeout", type=int, default=3600,
                   help="seconds allowed for one (cell, device) child before it is killed")
    # Child modes. One model load per fresh interpreter, one JSON line on stdout.
    p.add_argument("--score-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--artifact", default="", help=argparse.SUPPRESS)
    p.add_argument("--base", default="", help=argparse.SUPPRESS)
    p.add_argument("--device", default="", help=argparse.SUPPRESS)
    p.add_argument("--precision", default="", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # The context a real CLINC150 prompt needs; set before any scoring, parent or child.
    os.environ.setdefault("SLM_MAX_SEQ_LENGTH", "4096")

    if args.validate_only:
        os.environ["SLM_MNN_BACKEND_TYPE"] = args.device
        from training.quant_backend import validate_and_record

        try:
            record = validate_and_record(
                args.artifact, base_model=args.base, quant=args.precision, backend="mnn"
            )
            result = {key: record.get(key) for key in
                      ("quant_bit", "lm_quant_bit", "quant_block",
                       "validated_threads", "validated_backend", "weight_size_mb")}
        except Exception as error:  # noqa: BLE001 - reported to the parent as data
            result = {"error": f"{type(error).__name__}: {str(error)[:200]}"}
        print("RESULT_JSON " + json.dumps(result))
        return 0

    if args.score_only:
        full_set, scored_set = _eval_rows(args.eval_set, args.rows)
        try:
            result = _score(full_set, scored_set, args.artifact, args.base, args.device)
        except Exception as error:  # noqa: BLE001 - reported to the parent as data
            result = {"device": args.device,
                      "error": f"{type(error).__name__}: {str(error)[:200]}"}
        print("RESULT_JSON " + json.dumps(result))
        return 0

    args.devices = tuple(d.strip() for d in args.devices.split(",") if d.strip())
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    models = _pool_models()
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [(tier, model) for tier, model in models if model in wanted]
    if args.tier:
        models = [(tier, model) for tier, model in models if tier == args.tier]

    full_set, scored_set = _eval_rows(args.eval_set, args.rows)
    print(f"MNN backend matrix: {len(models)} model(s) x {len(precisions)} precision(s) "
          f"x {len(args.devices)} device(s); {len(scored_set.all)} clinc150 rows per cell")
    for tier, model in models:
        print(f"  tier {tier}  {model}")

    cells = []
    for tier, model in models:
        for precision in precisions:
            print(f"\n=== tier {tier} | {model} | {precision} ===", flush=True)
            cell = _one_cell(model, tier, precision, args, full_set, scored_set)
            cells.append(cell)
            os.makedirs(os.path.dirname(args.report), exist_ok=True)
            with open(args.report, "w", encoding="utf-8") as handle:
                json.dump({"cells": cells, "rows": len(scored_set.all)}, handle, indent=2)
            print(f"    -> {cell['status']}"
                  + (f"  {cell.get('error')}" if cell.get("error") else ""), flush=True)

    _print_matrix(cells, args.devices)
    ok = sum(1 for cell in cells if cell["status"] == "ok")
    print(f"\n{ok}/{len(cells)} cells passed;  report: {args.report}")
    return 0 if ok == len(cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
