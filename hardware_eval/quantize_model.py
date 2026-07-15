#!/usr/bin/env python3
"""
Quantization CLI — the SINGLE, standalone quantization step.

This is deliberately separate from the on-device evaluation step
(hardware_eval/run_autobench.py + hardware_eval/on_device_eval.py), which now
only *consume* a ready GGUF and never quantize. The split is:

    1. quantize_model.py   (this file)  HF checkpoint ──► GGUF        [quantization]
    2. run_autobench.py                 GGUF ──► on-device metrics    [measurement]

Both share one quantization engine: training/quantize.py (llama.cpp
convert_hf_to_gguf + llama-quantize). There is no second, home-grown converter.

Requirements: llama.cpp tools (`convert_hf_to_gguf` and `llama-quantize`) on PATH.
Without them the underlying engine returns a clear error and this CLI exits non-zero
(it does not silently produce a partial artifact).

Usage:
    # Merged full-precision HF checkpoint → Q4_K_M GGUF
    python hardware_eval/quantize_model.py --checkpoint artifacts/merged/... --quant Q4_K_M --out artifacts/gguf/model

    # LoRA adapter (+ base) → merge → GGUF
    python hardware_eval/quantize_model.py --adapter artifacts/iterN/final_checkpoint \
        --base Qwen/Qwen2.5-1.5B-Instruct --quant Q8_0 --out artifacts/gguf/model
"""
import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)


def main() -> int:
    p = argparse.ArgumentParser(description="Quantize an HF checkpoint to GGUF (standalone step).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Path to a MERGED full-precision HF checkpoint dir.")
    src.add_argument("--adapter", help="Path to a LoRA adapter-only checkpoint dir (needs --base).")
    p.add_argument("--base", default="", help="Base model id/path (required with --adapter).")
    p.add_argument("--quant", choices=["Q4_K_M", "Q8_0"], default="Q4_K_M")
    p.add_argument("--out", required=True, help="Output directory for the GGUF file.")
    args = p.parse_args()

    from training.quantize import quantize_from_model_spec

    checkpoint = args.checkpoint
    if args.adapter:
        if not args.base:
            p.error("--adapter requires --base (the base model to merge the adapter into).")
        from training.lora_trainer import merge_for_quantization
        print(f"[quantize] merging adapter {args.adapter} onto base {args.base} ...")
        checkpoint = merge_for_quantization(args.adapter, os.path.join(args.out, "_merged"))

    print(f"[quantize] {checkpoint} → {args.quant} GGUF (engine: training/quantize.py / llama.cpp)")
    try:
        gguf = quantize_from_model_spec(checkpoint, args.out, args.quant)
    except (RuntimeError, ValueError) as e:
        print(f"[quantize] FAILED: {e}", file=sys.stderr)
        return 1
    print(f"[quantize] OK → {gguf}")
    print(f"[quantize] Next: run on-device eval on this GGUF, e.g.\n"
          f"    python hardware_eval/run_autobench.py --model {gguf} --device phone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
