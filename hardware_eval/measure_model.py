#!/usr/bin/env python3
"""Record REAL measured metrics into config/measured_metrics.json.

This is the only sanctioned way to put runtime numbers in front of the pipeline. The
pool carries no estimated tok/s or peak RAM (see the note atop config/android_pool.py);
whatever this script records is what the hardware gate can use, and everything else
reports "unmeasured".

Nothing here is modelled. Every value comes from running the actual artifact:
  - size_mb          : os.path.getsize of a real GGUF / the real weight files
  - tok_per_s, ttft_ms : llama.cpp's own timing output for this build
  - peak_memory_mb   : peak RSS of the inference process (getrusage / on-device RSS)
  - avg_watts        : only from a device backend that can read battery current

The chip key matters. A host-machine run is recorded under the host's key (default
"host_cpu"), NEVER under a phone's — a number measured on an L40S node says nothing
about a Snapdragon. Use --chip to label a real device run.

Usage:
  # record on-disk size of an existing GGUF (no inference)
  python hardware_eval/measure_model.py --model Qwen/Qwen3.5-4B --quant Q4_K_M \\
      --gguf artifacts/gguf/Qwen_Qwen3.5-4B/<hash>/model-q4_k_m.gguf --size-only

  # full local timing + RSS measurement
  python hardware_eval/measure_model.py --model Qwen/Qwen3.5-4B --quant Q4_K_M \\
      --gguf <path>.gguf --backend llama_cpp --chip host_cpu

  # real phone via ADB
  python hardware_eval/measure_model.py --model Qwen/Qwen3.5-4B --quant Q4_K_M \\
      --gguf <path>.gguf --backend adb_llama --chip snapdragon_8gen3
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

METRICS_PATH = os.path.join(PROJ, "config", "measured_metrics.json")


def _load() -> dict:
    try:
        with open(METRICS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {"_sizes": {}, "_size_provenance": {}, "measurements": {}}


def _save(doc: dict) -> None:
    doc.setdefault("_sizes", {})
    doc.setdefault("_size_provenance", {})
    doc.setdefault("measurements", {})
    tmp = METRICS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, METRICS_PATH)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3.5-4B")
    p.add_argument("--quant", default=None, help="Q4_K_M / Q8_0 / omit for bf16")
    p.add_argument("--gguf", required=True, help="Path to the real GGUF (or weight dir for bf16).")
    p.add_argument("--chip", default="host_cpu",
                   help="Chip key this measurement is valid FOR. Never label a host run "
                        "with a phone's chip name. Default: host_cpu")
    p.add_argument("--backend", default=None,
                   choices=["llama_cpp", "adb_llama", "smolchat"],
                   help="Inference backend for timing/RSS. Omit with --size-only.")
    p.add_argument("--size-only", action="store_true",
                   help="Record on-disk size only; run no inference.")
    p.add_argument("--serial", default="", help="ADB serial for device backends.")
    p.add_argument("--note", default="", help="Free-text provenance note.")
    args = p.parse_args()

    if not args.size_only and not args.backend:
        p.error("either --size-only or --backend is required")
    if not os.path.exists(args.gguf):
        print(f"FAIL: artifact not found: {args.gguf}", file=sys.stderr)
        return 2

    quant_key = args.quant or "bf16"
    doc = _load()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    # --- size: always recorded, always real -------------------------------------
    if os.path.isdir(args.gguf):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(args.gguf) for f in fs
            if f.endswith((".safetensors", ".bin", ".gguf"))
        ) / (1024 * 1024)
        method = "sum of weight files in directory"
    else:
        size_mb = os.path.getsize(args.gguf) / (1024 * 1024)
        method = "os.path.getsize of GGUF"
    size_key = f"{args.model}|{quant_key}"
    doc.setdefault("_sizes", {})[size_key] = round(size_mb, 1)
    doc.setdefault("_size_provenance", {})[size_key] = {
        "source": args.gguf,
        "method": method,
        "measured_at": now,
        "note": args.note,
    }
    print(f"[measure] size {size_key} = {size_mb:.1f} MB  ({method})")

    # --- runtime metrics: only from a real inference run -------------------------
    if not args.size_only:
        from config.android_pool import HardwareConstraints, resolve_model_selector, ANDROID_POOL
        from hardware_eval.on_device_eval import run_on_device_eval

        spec = resolve_model_selector(ANDROID_POOL, f"{args.model}@{quant_key}")
        if spec is None:
            print(f"FAIL: {args.model}@{quant_key} is not in ANDROID_POOL", file=sys.stderr)
            return 2
        constraints = HardwareConstraints(
            storage_mb=10**9, memory_mb=10**9, latency_ttft_ms=10**9, target_chip=args.chip,
        )
        res = run_on_device_eval(spec, constraints, gguf_path=args.gguf,
                                 backend=args.backend, serial=args.serial)
        if not res.success:
            print(f"FAIL: measurement failed: {res.error}", file=sys.stderr)
            return 1
        measured = res.to_measured()
        measured.pop("device", None)
        if not measured:
            print("FAIL: backend returned no measured fields; nothing recorded",
                  file=sys.stderr)
            return 1
        entry = dict(measured)
        entry.update({
            "source": args.gguf,
            "method": f"{res.eval_method} backend",
            "measured_at": now,
            "note": args.note,
        })
        key = f"{args.model}|{quant_key}|{args.chip}"
        doc.setdefault("measurements", {})[key] = entry
        print(f"[measure] runtime {key}:")
        for field_name, value in measured.items():
            print(f"            {field_name} = {value}")

    _save(doc)
    print(f"[measure] wrote {METRICS_PATH}")
    print("[measure] NOTE: values are valid only for the chip key they were recorded "
          "under; the pool never interpolates across chips or quants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
