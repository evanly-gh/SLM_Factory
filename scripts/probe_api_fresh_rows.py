"""Score the frontier models on the SAME fresh rows `rescore_unseen.py --chunk-rows` scored the factory on.

WHY. `scripts/rescore_unseen.py` showed what each factory model scores on test rows outside its
run's held-out set. Those numbers can only be set against a frontier model scored on the same rows:
fresh rows can be harder or easier than the held-out set, for Claude as much as for the factory.
`scripts/probe_api_baselines.py` scored the frontier models on the held-out set only.

HOW IT STAYS COMPARABLE
  * Rows: rebuilt exactly as `rescore_unseen.py` built its chunks — every test row whose normalized
    text is in neither the run's held-out set nor any training dataset, kept only if its prompt is
    no longer than the longest held-out prompt, deduplicated, shuffled with random.Random(seed), cut
    into `--chunk-rows` sets, at most `--max-chunks`. The chunk sizes are checked against the
    factory's `_chunks.json` report, and the run stops if they differ.
  * Prompts, parsing and scoring: the task's own TaskSpec, through probe_api_baselines' own API
    calls (thinking off on both; Anthropic through the Message Batches API).
  * Each chunk is scored separately, like the factory's, and the mean is reported next to it.

Usage:
  python scripts/probe_api_fresh_rows.py --task goemotions --run-dir logs/runs/slm-goemotions-ckpt-39708679
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fresh_chunks(task: str, run_dir: str, chunk_rows: int, max_chunks: int, seed: int):
    """The chunks `rescore_unseen.py` scored, rebuilt step for step (see that script's main)."""
    from agent.nodes.curate import _normalized_texts, _row_text
    from data.eval_set import build_eval_set
    from data.loaders.dataset_integrity import normalize_text
    from scripts.report_eval import load_report_rows
    from tasks import get_task

    spec = get_task(task)
    selection = json.load(open(os.path.join(run_dir, "artifacts", "eval_set.json")))["examples"]
    train_rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "artifacts", "dataset_v*.jsonl"))):
        with open(path) as fh:
            train_rows.extend(json.loads(line) for line in fh if line.strip())
    sel_keys = _normalized_texts([r for r in selection if isinstance(r, dict)])
    train_keys = _normalized_texts([r for r in train_rows if isinstance(r, dict)])
    keyed = [(normalize_text(_row_text(r)), r)
             for r in load_report_rows(task, log=lambda *a: None) if isinstance(r, dict)]
    seen, unseen = set(), []
    for k, r in keyed:
        if not k or k in sel_keys or k in train_keys or k in seen:
            continue
        seen.add(k)
        unseen.append(r)

    def lengths(rows):
        es = build_eval_set(rows, task=task, target=len(rows))
        return [len(p) for p in spec.build_prompts(es)], es.all

    limit = max(lengths(selection)[0])
    lens, ordered = lengths(unseen)
    pool = [r for r, n in zip(ordered, lens) if n <= limit]
    rng = random.Random(seed)
    rng.shuffle(pool)
    size = chunk_rows if len(pool) >= 2 * chunk_rows else len(pool) // 2
    return spec, [pool[i * size:(i + 1) * size] for i in range(min(max_chunks, len(pool) // size))]


def main(argv=None) -> int:
    from data.eval_set import EvalSet
    from scripts.probe_api_baselines import (ANTHROPIC_MODEL, DEEPSEEK_MODEL, _log, price,
                                             run_anthropic, run_deepseek)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", nargs="+", default=["anthropic", "deepseek"])
    ap.add_argument("--chunk-rows", type=int, default=1000)
    ap.add_argument("--max-chunks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    run = os.path.basename(os.path.normpath(args.run_dir))
    factory_path = f"logs/reports/unseen/{args.task}__{run}_chunks.json"
    factory = json.load(open(factory_path))
    spec, chunks = fresh_chunks(args.task, args.run_dir, args.chunk_rows, args.max_chunks, args.seed)
    fs = factory["chunk_summary"]
    if [len(c) for c in chunks] != [c["factory"]["n"] for c in factory["chunks"]]:
        raise SystemExit(f"chunk sizes {[len(c) for c in chunks]} differ from {factory_path}; "
                         "the rows would not be the factory's rows — stopping")
    _log(f"{args.task}: {len(chunks)} chunk(s) of {len(chunks[0])} rows, matching {factory_path}")

    rows = [r for c in chunks for r in c]
    prompts = [str(p) for p in spec.build_prompts(EvalSet(all=rows, task=args.task))]
    out = {"task": args.task, "run_dir": args.run_dir, "chunk_rows": len(chunks[0]),
           "n_chunks": len(chunks), "factory_scores": fs["factory_scores"],
           "factory_mean": fs["factory_mean"], "factory_selection": fs["selection_score"],
           "models": {}}
    total = 0.0
    for which in args.models:
        started = time.monotonic()
        if which == "anthropic":
            raw, usage = run_anthropic(spec, prompts)
            model_id, provider, discounted = ANTHROPIC_MODEL, "anthropic", True
        else:
            raw, usage = run_deepseek(spec, prompts, args.concurrency)
            model_id, provider, discounted = DEEPSEEK_MODEL, "deepseek", False
        scores, valid, i = [], [], 0
        for c in chunks:
            es = EvalSet(all=c, task=args.task)
            scored = spec.score(es, spec.extract_predictions(raw[i:i + len(c)], es))
            scores.append(round(float(scored.get(spec.metric_name, scored.get("f1"))), 6))
            valid.append(scored.get("format_valid"))
            i += len(c)
        usd = price(provider, model_id, usage, batch_discount=discounted)
        total += 0.0 if usd != usd else usd
        out["models"][model_id] = {"scores": scores, "mean": round(statistics.mean(scores), 6),
                                   "format_valid": valid, "usd": usd,
                                   "seconds": round(time.monotonic() - started, 1),
                                   **{f"tok_{k}": v for k, v in usage.items()}}
        _log(f"  {model_id}: chunks {scores} mean {statistics.mean(scores):.4f} "
             f"(factory mean {fs['factory_mean']:.4f}) ${usd:.2f}, {usage.get('errors', 0)} error(s)")
        path = args.out or f"logs/reports/unseen/{args.task}__{run}_frontier.json"
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
    _log(f"wrote {path} — total ${total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
