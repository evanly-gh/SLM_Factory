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
    # Artifact stores rows under "examples"; from_serialized also folds legacy pos/neg/boundary.
    rows = d.get("all") if d.get("all") is not None else d.get("examples")
    payload = {**d, "task_type": task_type}
    if rows is not None:
        payload["all"] = rows
    return EvalSet.from_serialized(payload), task_type


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
    p.add_argument("--report-dir", default=None,
                   help="Write results.json + report.md (with chart) here. Recommended: logs/quant_eval/<name>.")
    p.add_argument("--label", default="", help="Human label for the report header.")
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
    elif not os.path.isdir(checkpoint):
        # --checkpoint may name a bare HF model id (base-model sweep, no adapter). Resolve
        # it to the immutable local snapshot the converter can read.
        from training.quantize import resolve_hf_snapshot
        print(f"[quant-eval] resolving HF snapshot for base model {checkpoint} ...")
        checkpoint = resolve_hf_snapshot(checkpoint)
        print(f"[quant-eval] snapshot: {checkpoint}")

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
            # bf16 has no GGUF, so measure the weight files in the checkpoint dir itself —
            # otherwise the size chart has a hole exactly where the baseline should be.
            bf16_mb = round(sum(
                os.path.getsize(os.path.join(dirpath, name))
                for dirpath, _, names in os.walk(checkpoint)
                for name in names
                if name.endswith((".safetensors", ".bin"))
            ) / (1024 * 1024), 1) or None
            rows.append(("bf16 (full)", bf16_mb, res.f1))
            print(f"[quant-eval] bf16: F1={res.f1:.4f}  ({bf16_mb} MB, full precision via Unsloth)")
        except Exception as e:
            print(f"[quant-eval] bf16: FAILED — {e}", file=sys.stderr)

    _emit_report(rows, args, task_type, len(eval_set.all), checkpoint)
    return 0


def _bar(value: float, vmax: float, width: int = 40) -> str:
    """Proportional bar. Scaled to vmax so small real differences stay visible."""
    filled = int(round((value / vmax) * width)) if vmax > 0 else 0
    return "█" * filled + "·" * (width - filled)


def _emit_report(rows, args, task_type, n_examples, checkpoint) -> None:
    """Print an accuracy/size chart and optionally persist results.json + report.md."""
    import datetime
    import json as _json

    scored = [(lab, sz, f1) for lab, sz, f1 in rows if f1 is not None]
    ref = next((f1 for lab, _, f1 in scored if lab.startswith("bf16")), None)

    lines = []
    add = lines.append
    title = args.label or os.path.basename(str(checkpoint).rstrip("/"))
    add(f"Precision sweep — {title}")
    add(f"task_type={task_type}  eval_set={n_examples} examples")
    add("")

    if scored:
        # --- accuracy chart, zoomed to the observed range -------------------
        # A 0..1 axis would render every bar identical when variants differ by
        # <0.01; anchoring near the observed minimum is what makes the real
        # spread legible. The absolute F1 column keeps that honest.
        lo = min(f1 for _, _, f1 in scored)
        hi = max(f1 for _, _, f1 in scored)
        span = max(hi - lo, 1e-9)
        floor = max(0.0, lo - span * 0.35)
        add(f"ACCURACY (F1)   bars scaled {floor:.4f}–{hi:.4f}, not 0–1")
        add(f"{'variant':<16}{'F1':>9}  {'Δ vs bf16':>10}  chart")
        add("-" * 84)
        for lab, _, f1 in scored:
            delta = "—" if ref is None or lab.startswith("bf16") else f"{f1 - ref:+.4f}"
            add(f"{lab:<16}{f1:>9.4f}  {delta:>10}  {_bar(f1 - floor, hi - floor)}")
        add("")

        # --- size chart ------------------------------------------------------
        sized = [(lab, sz) for lab, sz, _ in scored if sz]
        if sized:
            smax = max(sz for _, sz in sized)
            add(f"{'ON-DISK SIZE':<16}{'MB':>9}  {'vs largest':>10}  chart")
            add("-" * 84)
            for lab, sz in sized:
                add(f"{lab:<16}{sz:>9.1f}  {sz / smax:>9.0%}  {_bar(sz, smax)}")
            add("")

        # --- the tradeoff, stated numerically --------------------------------
        if ref is not None:
            add("TRADEOFF vs bf16")
            bf_sz = next((sz for lab, sz, _ in rows if lab.startswith("bf16") and sz), None)
            for lab, sz, f1 in scored:
                if lab.startswith("bf16"):
                    continue
                acc_cost = (ref - f1) / ref * 100 if ref else 0.0
                shrink = f"{bf_sz / sz:.2f}x smaller" if (bf_sz and sz) else "size n/a"
                verb = "loses" if acc_cost >= 0 else "GAINS"
                add(f"  {lab:<12} {verb} {abs(acc_cost):.3f}% accuracy for {shrink}")
            add("")

    for lab, sz, f1 in rows:
        if f1 is None:
            add(f"  !! {lab}: FAILED to score")

    report = "\n".join(lines)
    print("\n" + report)

    if not args.report_dir:
        return
    os.makedirs(args.report_dir, exist_ok=True)
    payload = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        "checkpoint": str(checkpoint),
        "base_model": args.base or None,
        "adapter": args.adapter or None,
        "eval_set": args.eval_set,
        "task_type": task_type,
        "n_examples": n_examples,
        "results": [
            {"variant": lab, "on_disk_mb": sz, "f1": f1}
            for lab, sz, f1 in rows
        ],
    }
    with open(os.path.join(args.report_dir, "results.json"), "w") as fh:
        _json.dump(payload, fh, indent=2)
    with open(os.path.join(args.report_dir, "report.md"), "w") as fh:
        fh.write(f"# {title}\n\n```\n{report}\n```\n")
    print(f"[quant-eval] report written to {args.report_dir}/")


if __name__ == "__main__":
    raise SystemExit(main())
