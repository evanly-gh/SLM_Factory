#!/usr/bin/env python3
"""Fill config/measured_metrics.json for every pool variant that has no real measurement.

WHY THIS EXISTS
    The pool carries no modelled runtime metrics — that was removed deliberately. What
    remains is on-disk weight size, and for an unmeasured variant that size is
    bytes-per-parameter ARITHMETIC, not a measurement. On the one family measured so far
    (Qwen3.5-4B) the arithmetic was 20% LOW at Q4_K_M: predicted 2200 MB, actual 2654.5 MB.
    That correction moved the variant from tier 2 to tier 3. The 2026-07-29 sweep then
    verified every other family and found the corrected arithmetic accurate to +4.9%/-1.1%
    with ZERO further tier changes — a negative result, but one that had to be measured
    rather than assumed.

WHAT IT RECORDS
    _sizes         real os.path.getsize of a real GGUF build, and the real safetensors sum
                   for bf16. Chip-independent — this is the primary payload.
    measurements   peak RSS + decode tok/s from llama.cpp's own timing, keyed to the chip
                   the run actually happened on (default "host_cpu"). Never a phone key.

Nothing here is estimated. A variant that fails to build or measure is LEFT ABSENT, which
makes consumers correctly report "unmeasured" rather than trusting a guess.

Usage:
    python hardware_eval/measure_pool.py                 # every unmeasured pool variant
    python hardware_eval/measure_pool.py --sizes-only    # skip runtime timing
    python hardware_eval/measure_pool.py --model Qwen/Qwen3-0.6B
    python hardware_eval/measure_pool.py --force         # re-measure already-recorded ones
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
import sys
import traceback

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

METRICS_PATH = os.path.join(PROJ, "config", "measured_metrics.json")
QUANTS = ("Q4_K_M", "Q8_0")


def _load() -> dict:
    with open(METRICS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _save(doc: dict) -> None:
    """Atomic read-modify-write so an interrupted sweep cannot corrupt the file."""
    doc.setdefault("_sizes", {})
    doc.setdefault("_size_provenance", {})
    doc.setdefault("measurements", {})
    tmp = METRICS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, METRICS_PATH)


def _record_size(model: str, quant_key: str, size_mb: float, method: str, source: str,
                 note: str = "") -> None:
    doc = _load()
    key = f"{model}|{quant_key}"
    doc.setdefault("_sizes", {})[key] = round(size_mb, 1)
    provenance = {
        "source": source,
        "method": method,
        "measured_at": datetime.date.today().isoformat(),
    }
    if note:
        provenance["note"] = note
    doc.setdefault("_size_provenance", {})[key] = provenance
    _save(doc)
    print(f"    recorded size {key} = {round(size_mb, 1)} MB")


def _record_runtime(model: str, quant_key: str, chip: str, result, source: str) -> None:
    """Record only the fields the backend actually produced; omit the rest."""
    from config.android_pool import measured_metrics_key

    fields = {
        "peak_memory_mb": getattr(result, "peak_memory_mb", None),
        "tok_per_s": getattr(result, "tok_per_s", None),
        "ttft_ms": getattr(result, "ttft_ms", None),
        "avg_watts": getattr(result, "avg_watts", None),
    }
    present = {name: value for name, value in fields.items() if value is not None}
    if not present:
        print("    no runtime fields measured — recording nothing")
        return
    doc = _load()
    key = measured_metrics_key(model, None if quant_key == "bf16" else quant_key, chip)
    doc.setdefault("measurements", {})[key] = {
        **present,
        "source": source,
        "method": f"{getattr(result, 'eval_method', 'llama_cpp')}; "
                  "peak RSS via getrusage(RUSAGE_CHILDREN)",
        "measured_at": datetime.date.today().isoformat(),
    }
    _save(doc)
    print(f"    recorded runtime {key}: {present}")


def _bf16_size(model: str) -> tuple[float, str] | None:
    from training.quantize import resolve_hf_snapshot

    snapshot = resolve_hf_snapshot(model)
    shards = sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))
    if not shards:
        return None
    total = sum(os.path.getsize(path) for path in shards) / (1024 * 1024)
    return total, f"sum of {len(shards)} *.safetensors in the immutable HF snapshot"


def _pool_model_ids() -> list[str]:
    from config.android_pool import ANDROID_POOL

    seen: list[str] = []
    for spec in ANDROID_POOL:
        if spec.model_id not in seen:
            seen.append(spec.model_id)
    return seen


def measure_variant(model: str, quant: str, out_root: str, source: str,
                    *, chip: str, sizes_only: bool, keep_gguf: bool) -> bool:
    """Build, size, and (optionally) time one quantized variant. Returns success."""
    from training.quantize import quantize_from_model_spec, resolve_hf_snapshot

    print(f"  --- {model} @ {quant}")
    gguf_dir = os.path.join(out_root, model.replace("/", "_"), quant)
    os.makedirs(gguf_dir, exist_ok=True)
    try:
        # quantize_from_model_spec takes a local checkpoint PATH, not a bare HF id, so the
        # immutable snapshot has to be resolved first (same as evaluate.py's remote-base path).
        snapshot = resolve_hf_snapshot(model)
        gguf = quantize_from_model_spec(snapshot, gguf_dir, quant)
    except Exception as exc:  # noqa: BLE001 - one bad family must not stop the sweep
        print(f"    !! build failed ({type(exc).__name__}: {str(exc)[:200]})")
        print(f"    !! leaving {model}|{quant} UNMEASURED (the correct fallback)")
        return False

    size_mb = os.path.getsize(gguf) / (1024 * 1024)
    _record_size(
        model, quant, size_mb,
        "llama.cpp convert_hf_to_gguf + llama-quantize; os.path.getsize",
        source,
    )

    if not sizes_only:
        try:
            from config.android_pool import HardwareConstraints, resolve_model_selector
            from config.android_pool import ANDROID_POOL
            from hardware_eval.on_device_eval import run_on_device_eval

            spec = resolve_model_selector(ANDROID_POOL, f"{model}@{quant}")
            if spec is None:
                print(f"    !! {model}@{quant} is not a pool variant — skipping runtime")
            else:
                # A generous budget: this measures, it does not gate.
                constraints = HardwareConstraints(
                    storage_mb=1_000_000,
                    memory_mb=1_000_000,
                    latency_ttft_ms=1_000_000,
                    min_tok_s=0.0,
                    target_chip=chip,
                )
                result = run_on_device_eval(
                    spec, constraints, gguf_path=gguf,
                    backend="llama_cpp", log=lambda m: print(f"    {m}"),
                )
                if result.success:
                    _record_runtime(model, quant, chip, result, source)
                else:
                    print(f"    !! runtime measurement failed: {result.error}")
        except Exception as exc:  # noqa: BLE001
            print(f"    !! runtime measurement errored ({type(exc).__name__}: {exc})")

    if not keep_gguf:
        shutil.rmtree(gguf_dir, ignore_errors=True)
        print("    removed GGUF (pass --keep-gguf to retain)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=[],
                       help="Limit to this model id (repeatable). Default: all pool models.")
    parser.add_argument("--out", default=os.path.join(PROJ, "artifacts", "measure_pool"),
                       help="Scratch directory for GGUF builds.")
    parser.add_argument("--chip", default="host_cpu",
                       help="Chip key the runtime numbers are valid FOR. Never a phone name "
                            "for a host run. Default: host_cpu")
    parser.add_argument("--sizes-only", action="store_true",
                       help="Record on-disk sizes only; run no inference.")
    parser.add_argument("--force", action="store_true",
                       help="Re-measure variants that already have a recorded size.")
    parser.add_argument("--keep-gguf", action="store_true",
                       help="Keep the built GGUFs (they are large).")
    parser.add_argument("--source", default="",
                       help="Provenance label. Defaults to the Slurm job id.")
    args = parser.parse_args()

    source = args.source or f"slurm job {os.environ.get('SLURM_JOB_ID', 'manual')}"
    models = args.model or _pool_model_ids()
    existing = _load().get("_sizes", {})

    print(f"=== measure_pool: {len(models)} model family(ies), chip={args.chip}")
    print(f"=== already recorded: {sorted(existing)}")

    failures: list[str] = []
    for model in models:
        print(f"\n{'#' * 70}\n### {model}\n{'#' * 70}")
        for quant in QUANTS:
            if not args.force and f"{model}|{quant}" in existing:
                print(f"  --- {model} @ {quant}: already measured, skipping")
                continue
            if not measure_variant(
                model, quant, args.out, source,
                chip=args.chip, sizes_only=args.sizes_only,
                keep_gguf=args.keep_gguf,
            ):
                failures.append(f"{model}@{quant}")

        if args.force or f"{model}|bf16" not in existing:
            print(f"  --- {model} @ bf16 (safetensors sum, no conversion)")
            try:
                measured = _bf16_size(model)
                if measured is None:
                    print("    !! no *.safetensors in snapshot — leaving unmeasured")
                    failures.append(f"{model}@bf16")
                else:
                    total, method = measured
                    _record_size(model, "bf16", total, method, source)
            except Exception as exc:  # noqa: BLE001
                print(f"    !! bf16 size failed ({type(exc).__name__}: {str(exc)[:200]})")
                failures.append(f"{model}@bf16")
        else:
            print(f"  --- {model} @ bf16: already measured, skipping")

    doc = _load()
    print(f"\n{'#' * 70}")
    if failures:
        print("### LEFT UNMEASURED (correct fallback — consumers report 'unmeasured'):")
        for item in failures:
            print(f"###   {item}")
    else:
        print("### every requested variant measured")
    print(f"{'#' * 70}\n")
    print("=== _sizes:")
    for key, value in sorted(doc.get("_sizes", {}).items()):
        print(f"  {key:44s} {value:>9} MB")
    print("=== measurements (runtime, chip-keyed):")
    for key, value in sorted(doc.get("measurements", {}).items()):
        fields = {k: v for k, v in value.items()
                  if k not in ("source", "method", "measured_at")}
        print(f"  {key}: {fields}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc()
        raise SystemExit(130)
