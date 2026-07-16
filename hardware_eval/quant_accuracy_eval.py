#!/usr/bin/env python3
"""
Quantization ACCURACY comparison — score a trained checkpoint at Q4_K_M / Q8_0 / bf16 and
print an accuracy table. NO on-device / latency / power measurement (that's run_autobench).

This answers "how much accuracy do I lose at each quantization?" on CPU, using the same
honest path the pipeline uses: merge → quantize to GGUF (llama.cpp) → score the GGUF via
llama-cpp-python with the model's chat template. bf16 is scored full-precision via Unsloth.

Requirements:
  1. llama.cpp tools on PATH: `convert_hf_to_gguf` (+ `.py`) and `llama-quantize`
     (git clone https://github.com/ggml-org/llama.cpp && cmake --build ... ; add build/bin to PATH)
  2. pip install llama-cpp-python   (CPU GGUF inference)
  3. The pipeline venv for the bf16 path (unsloth) — only needed with --include-bf16.

Usage:
  # from a MERGED full-precision HF checkpoint + a run's eval_set.json
  python hardware_eval/quant_accuracy_eval.py \
      --checkpoint logs/runs/<ts>/artifacts/merged/<model>/<label>/iterN/merged \
      --eval-set  logs/runs/<ts>/artifacts/eval_set.json \
      --quants Q4_K_M,Q8_0 --include-bf16 --out /tmp/quant_cmp

  # from a LoRA adapter (+ base) — it merges first
  python hardware_eval/quant_accuracy_eval.py \
      --adapter logs/runs/<ts>/artifacts/iterN/final_checkpoint \
      --base Qwen/Qwen2.5-1.5B-Instruct \
      --eval-set logs/runs/<ts>/artifacts/eval_set.json --out /tmp/quant_cmp
"""
import argparse
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)


def _load_eval_set(path: str, task_type_override: str | None):
    from data.eval_set import EvalSet
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    task_type = task_type_override or d.get("task_type")
    return EvalSet(pos=d.get("pos", []), neg=d.get("neg", []),
                   boundary=d.get("boundary", []), task_type=task_type), task_type


def main() -> int:
    p = argparse.ArgumentParser(description="Compare fine-tuned model accuracy across quantizations.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Merged full-precision HF checkpoint dir.")
    src.add_argument("--adapter", help="LoRA adapter dir (requires --base; merged first).")
    p.add_argument("--base", default="", help="Base model id/path (required with --adapter).")
    p.add_argument("--eval-set", required=True, help="Path to a run's artifacts/eval_set.json.")
    p.add_argument("--task-type", default=None, help="Override task_type (else read from eval_set.json).")
    p.add_argument("--quants", default="Q4_K_M,Q8_0", help="Comma list of GGUF quants to compare.")
    p.add_argument("--include-bf16", action="store_true", help="Also score full-precision (Unsloth).")
    p.add_argument("--out", required=True, help="Output dir for merged/GGUF artifacts.")
    args = p.parse_args()

    from eval.harness import run_eval
    from training.quantize import quantize_from_model_spec, _file_size_mb

    checkpoint = args.checkpoint
    if args.adapter:
        if not args.base:
            p.error("--adapter requires --base")
        from training.lora_trainer import merge_for_quantization
        print(f"[quant-eval] merging {args.adapter} onto {args.base} ...")
        checkpoint = merge_for_quantization(args.adapter, os.path.join(args.out, "_merged"))

    eval_set, task_type = _load_eval_set(args.eval_set, args.task_type)
    base = args.base or checkpoint
    print(f"[quant-eval] task_type={task_type}  eval_set={len(eval_set.all)} examples  checkpoint={checkpoint}")

    rows = []  # (label, size_mb, f1)
    for quant in [q.strip() for q in args.quants.split(",") if q.strip()]:
        try:
            gguf = quantize_from_model_spec(checkpoint, os.path.join(args.out, quant), quant)
            size = round(_file_size_mb(gguf), 1)
            res = run_eval(eval_set, checkpoint, base, task_type=task_type, quant=quant, gguf_path=gguf)
            rows.append((quant, size, res.f1))
            print(f"[quant-eval] {quant}: F1={res.f1:.4f}  ({size} MB)")
        except Exception as e:
            print(f"[quant-eval] {quant}: FAILED — {e}", file=sys.stderr)
            rows.append((quant, None, None))

    if args.include_bf16:
        try:
            res = run_eval(eval_set, checkpoint, base, task_type=task_type, quant=None, gguf_path=None)
            rows.append(("bf16 (full)", None, res.f1))
            print(f"[quant-eval] bf16: F1={res.f1:.4f}  (full precision via Unsloth)")
        except Exception as e:
            print(f"[quant-eval] bf16: FAILED — {e}", file=sys.stderr)

    print("\n" + "=" * 48)
    print(f"{'Quant':<16}{'On-disk MB':>12}{'F1':>10}")
    print("-" * 48)
    for label, size, f1 in rows:
        print(f"{label:<16}{(size if size is not None else 'N/A'):>12}{(f'{f1:.4f}' if f1 is not None else 'FAIL'):>10}")
    print("=" * 48)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
