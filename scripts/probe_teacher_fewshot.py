"""Measure the Qwen3.6 teacher zero-shot vs few-shot on a task's own eval set.

WHY
    The accuracy goal, and every argument about whether the teacher is fit to generate synthetic
    data, currently rests on ONE number: its ZERO-SHOT score
    (`eval/endpoint_eval.py::measure_endpoint_baseline`). On BC5CDR that number is 0.0999, which
    reads as "the teacher cannot do biomedical NER".

    That conflates two very different claims. Exact-span NER is dominated by output-format
    compliance, so a zero-shot score mostly measures whether the model guessed our JSON contract
    and our span conventions — not whether it can find entities. Few-shot prompting supplies the
    convention explicitly. If the gap is large, the teacher's competence was never the problem and
    the "is the teacher good enough to generate data" question needs a different input than the
    zero-shot eval score.

WHAT IT DOES
    Scores the same frozen eval set three ways through the same scorer the pipeline uses, so the
    numbers are directly comparable to the run logs:
      * 0-shot  — byte-identical to what `measure_endpoint_baseline` does today
      * k-shot  — the same prompt with k demonstrations prepended, drawn from TRAIN only
    Demonstrations come from the task's train split, never from the eval set, so this cannot leak.

USAGE (needs a reachable vLLM endpoint; see scripts/probe_teacher_fewshot.slurm)
    python scripts/probe_teacher_fewshot.py --task ner_bc5cdr --shots 0 1 3 5 --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fmt_gold(row: dict, task_type: str) -> str:
    """The gold answer rendered exactly as the scorer expects the model to emit it."""
    if task_type == "NER":
        return json.dumps(
            [{"text": e.get("text"), "type": e.get("type")} for e in row.get("entities") or []],
            ensure_ascii=False,
        )
    if row.get("answer") is not None:
        return str(row["answer"])
    return str(row.get("label", ""))


def _demo_block(demos: list[dict], task_type: str, scorer) -> str:
    """k demonstrations in the same prompt/answer shape the model is about to be asked for."""
    from data.eval_set import EvalSet

    parts = []
    for row in demos:
        one = EvalSet(all=[row], task_type=task_type)
        prompt = scorer.build_prompts(one)[0]
        parts.append(f"{prompt}\n{_fmt_gold(row, task_type)}")
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ner_bc5cdr")
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--n", type=int, default=200, help="eval rows to score per condition")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from agent.nodes.cold_start.eval_setup import NAMED_BENCHMARK_TASK_TYPES
    from data.eval_set import EvalSet, build_eval_set
    from data.synth_client import get_generate_fn, is_available
    from eval.endpoint_eval import _generate_all, _scorer_for
    from eval.harness import TASK_METRIC_NAMES, eval_output_token_reserve

    if args.task not in NAMED_BENCHMARK_TASK_TYPES:
        print(f"unknown task {args.task!r}; known: {sorted(NAMED_BENCHMARK_TASK_TYPES)}")
        return 2
    task_type = NAMED_BENCHMARK_TASK_TYPES[args.task][0]

    if not is_available(log=print):
        print("FATAL: no reachable synth endpoint (set SLM_SYNTH_ENDPOINT).")
        return 3
    generate_fn = get_generate_fn(log=print)

    # Load the task through its own loader, so the eval set is the one the pipeline would score.
    from agent.nodes.cold_start.eval_setup import _named_benchmark_loaders

    loader = _named_benchmark_loaders()[args.task][0]
    train, test = loader(max_train=2000, max_test=max(args.n, 200), log=print)
    eval_set = build_eval_set(test, task_type=task_type, target=args.n)
    rows = list(eval_set.all)[: args.n]
    eval_set = EvalSet(all=rows, task_type=task_type)

    scorer = _scorer_for(task_type)
    base_prompts = scorer.build_prompts(eval_set)
    max_tokens = eval_output_token_reserve(task_type)
    metric = TASK_METRIC_NAMES.get(task_type, "f1")
    rng = random.Random(args.seed)

    print(f"\n=== teacher few-shot probe: {args.task} (task_type={task_type}, metric={metric}) ===")
    print(f"    model  : {os.environ.get('SLM_SYNTH_MODEL', 'Qwen/Qwen3.6-35B-A3B')}")
    print(f"    rows   : {len(rows)} eval, {len(train)} train available for demos")
    print(f"    tokens : max_new_tokens={max_tokens}\n")

    results = []
    for k in args.shots:
        if k > 0 and not train:
            print(f"  skip k={k}: no train rows for demonstrations")
            continue
        prompts = base_prompts
        if k > 0:
            # Fresh demonstrations per eval row, drawn from TRAIN only — never the eval set, so a
            # demonstration can never be the row being scored.
            prompts = []
            for base in base_prompts:
                demos = rng.sample(train, min(k, len(train)))
                prompts.append(
                    "Here are worked examples showing the exact output format expected:\n\n"
                    + _demo_block(demos, task_type, scorer)
                    + "\n\nNow do the same for this one.\n\n"
                    + base
                )
        t0 = time.perf_counter()
        raw = _generate_all(generate_fn, prompts, max_tokens, 16)
        preds = scorer.extract_predictions(raw, eval_set)
        scored = scorer.score(eval_set, preds)
        elapsed = time.perf_counter() - t0
        # An empty prediction is the format-failure signature for NER and classification alike.
        empty = sum(1 for p in preds if not p or p == "__EXTRACTION_FAILED__")
        results.append({
            "shots": k, "score": scored["f1"], "metric": scored.get("metric", metric),
            "empty_or_failed": empty, "n": len(rows), "seconds": round(elapsed, 1),
        })
        print(f"  k={k}:  {scored.get('metric', metric)}={scored['f1']:.4f}   "
              f"empty/failed={empty}/{len(rows)}   ({elapsed:.0f}s)")
        if k == 0 and raw:
            print(f"        sample raw 0-shot output: {str(raw[0])[:160]!r}")
        elif raw:
            print(f"        sample raw {k}-shot output: {str(raw[0])[:160]!r}")

    if results:
        zero = next((r for r in results if r["shots"] == 0), None)
        best = max(results, key=lambda r: r["score"])
        print("\n=== summary ===")
        for r in results:
            delta = f"  (Δ vs 0-shot {r['score'] - zero['score']:+.4f})" if zero else ""
            print(f"  {r['shots']}-shot: {r['score']:.4f}{delta}")
        if zero and best["shots"] != 0:
            print(f"\n  BEST is {best['shots']}-shot at {best['score']:.4f}, "
                  f"{best['score'] - zero['score']:+.4f} over zero-shot "
                  f"({best['score'] / zero['score']:.1f}x)" if zero["score"] > 0 else
                  f"\n  BEST is {best['shots']}-shot at {best['score']:.4f}, "
                  f"up from {zero['score']:.4f} zero-shot")

    out = args.out or f"logs/probes/teacher_fewshot_{args.task}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"task": args.task, "task_type": task_type,
                   "model": os.environ.get("SLM_SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B"),
                   "results": results}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
