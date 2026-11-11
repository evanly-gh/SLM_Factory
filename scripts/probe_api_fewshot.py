"""Five-shot API baselines: Claude and DeepSeek with the same five demonstrations the teacher saw.

WHY. Table 3's frontier columns are zero-shot. A reviewer's first objection is that a frontier model
given a few examples of the output format would close much of the gap. This scores both frontier
models five-shot on every task's frozen held-out set.

HOW IT STAYS COMPARABLE
  * Rows: the same held-out set as the zero-shot baselines (`probe_api_baselines.load_task`, which
    matches every run's `artifacts/eval_set.json` exactly).
  * Demonstrations: the teacher's own selection and framing (`agent/teacher_fitness.py`):
    `fit_demonstrations` (shortest-first from the training pool the runs load at cold start,
    `spec.load(max_train=spec.initial_train_cap)`), each rendered with the task's
    `build_training_turn`, fenced by numbered headers, then the "answer THIS request only" header.
    The only difference from the teacher is the budget: the teacher's 8K context fits no
    demonstrations on CLINC150, while the frontier models get the full five there too.
  * Calls, parsing and scoring: `probe_api_baselines`' own (thinking off; Anthropic via Batches).

USAGE
  python scripts/probe_api_fewshot.py --estimate            # token counts and cost, no API calls
  python scripts/probe_api_fewshot.py --models deepseek     # one provider
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHOTS = 5


def fewshot_prompts(task: str):
    from agent.teacher_fitness import _QUESTION_HEADER, _demo_block, _training_context, fit_demonstrations
    from scripts.probe_api_baselines import load_task
    from tasks import get_task

    spec, eval_set, base = load_task(task, None)
    full = get_task(task)
    train, _ = full.load(max_train=full.initial_train_cap, max_test=1, log=lambda *a: None)
    pool = [r for r in train if isinstance(r, dict)]
    demos = fit_demonstrations(pool, SHOTS, budget_chars=10**9)
    ctx = _training_context(pool, spec)
    prefix = _demo_block(demos, spec, ctx) + "\n\n" + _QUESTION_HEADER + "\n"
    return spec, eval_set, [prefix + p for p in base], prefix, len(demos)


def run_anthropic_cached(spec, prefix: str, prompts: list[str], batch_id: str | None = None):
    """probe_api_baselines.run_anthropic, with the shared demonstration block cached.

    The demonstrations are ~83% of every five-shot prompt and identical across rows, so the prompt
    is sent as two text blocks, [demonstrations (cache_control), question], whose concatenation is
    the same text. Cache hits inside a batch are best-effort; usage reports what was cached.
    """
    import anthropic

    from config.config import ANTHROPIC_API_KEY, orchestrator_client_kwargs
    from scripts.probe_api_baselines import ANTHROPIC_MODEL, _BATCH_TIMEOUT_S, _POLL_S, _log

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())
    requests = [{
        "custom_id": f"row-{i}",
        "params": {
            "model": ANTHROPIC_MODEL,
            "max_tokens": spec.max_new_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt[len(prefix):]},
            ]}],
        },
    } for i, prompt in enumerate(prompts)]
    if batch_id:  # results of a finished batch are kept 29 days: fetch instead of paying again
        batch = client.messages.batches.retrieve(batch_id)
        _log(f"    anthropic batch {batch.id} REUSED ({batch.processing_status})")
    else:
        batch = client.messages.batches.create(requests=requests)
        _log(f"    anthropic batch {batch.id} submitted ({len(requests)} request(s), prefix cached)")
    deadline = time.monotonic() + _BATCH_TIMEOUT_S
    while time.monotonic() < deadline:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        time.sleep(_POLL_S)
    else:
        raise TimeoutError(f"batch {batch.id} did not finish within {_BATCH_TIMEOUT_S}s")
    outputs = [""] * len(prompts)
    usage = {"input": 0, "output": 0, "cached": 0, "cache_write": 0, "errors": 0}
    for result in client.messages.batches.results(batch.id):
        index = int(str(result.custom_id).split("-")[1])
        if result.result.type != "succeeded":
            usage["errors"] += 1
            continue
        message = result.result.message
        outputs[index] = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
        usage["input"] += message.usage.input_tokens
        usage["output"] += message.usage.output_tokens
        usage["cached"] += getattr(message.usage, "cache_read_input_tokens", 0) or 0
        usage["cache_write"] += getattr(message.usage, "cache_creation_input_tokens", 0) or 0
    return outputs, usage


def main() -> int:
    from scripts.probe_api_baselines import (ANTHROPIC_MODEL, DEEPSEEK_MODEL, TASKS, _log, price,
                                             run_anthropic, run_deepseek)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--models", nargs="+", default=["anthropic", "deepseek"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--estimate", action="store_true", help="count tokens and price; no API calls")
    ap.add_argument("--out", default="logs/probes/api-baselines-5shot.json")
    ap.add_argument("--reuse-batch", nargs="*", default=[], metavar="TASK=BATCH_ID",
                    help="fetch an already-finished Anthropic batch for TASK instead of submitting")
    ap.add_argument("--deepseek-cap", type=float, default=7.0,
                    help="stop calling DeepSeek once this run has spent this much (USD)")
    args = ap.parse_args()

    results = {"model_ids": {"anthropic": ANTHROPIC_MODEL, "deepseek": DEEPSEEK_MODEL},
               "shots": SHOTS, "cells": []}
    if os.path.exists(args.out) and not args.estimate:
        results = json.load(open(args.out))
    done = {(c["task"], c["model"]) for c in results["cells"] if "score" in c}
    est_total = {"anthropic": 0.0, "deepseek": 0.0}
    for task in args.tasks:
        spec, eval_set, prompts, prefix, n_demos = fewshot_prompts(task)
        chars = sum(len(p) for p in prompts)
        _log(f"── {task}: {len(prompts)} rows, {n_demos}-shot, prefix {len(prefix):,} chars, "
             f"prompts {chars:,} chars (~{chars / 3.5 / 1e6:.2f}M tokens) ──")
        if args.estimate:
            tok_in, tok_prefix = chars / 3.5, len(prefix) / 3.5 * len(prompts)
            out_tok = len(prompts) * min(spec.max_new_tokens, 200)
            a = price("anthropic", ANTHROPIC_MODEL, {"input": tok_in, "output": out_tok}, batch_discount=True)
            d = price("deepseek", DEEPSEEK_MODEL, {"input": tok_in, "output": out_tok,
                                                   "cached": tok_prefix}, batch_discount=False)
            d_nocache = price("deepseek", DEEPSEEK_MODEL, {"input": tok_in, "output": out_tok},
                              batch_discount=False)
            est_total["anthropic"] += a
            est_total["deepseek"] += d_nocache
            _log(f"    est. anthropic ${a:.2f}; deepseek ${d:.2f} with prefix cache, "
                 f"${d_nocache:.2f} without (output capped at {min(spec.max_new_tokens, 200)} tok/row)")
            continue
        for which in args.models:
            model_id = ANTHROPIC_MODEL if which == "anthropic" else DEEPSEEK_MODEL
            spent_ds = sum(c.get("usd", 0) or 0 for c in results["cells"] if c.get("model") == DEEPSEEK_MODEL)
            if which == "deepseek" and spent_ds >= args.deepseek_cap:
                _log(f"    deepseek: cap ${args.deepseek_cap:.2f} reached (${spent_ds:.2f}); skipping")
                continue
            if (task, model_id) in done:
                _log(f"    {which}: already scored, skipping")
                continue
            started = time.monotonic()
            try:
                if which == "anthropic":
                    reuse = dict(x.split("=", 1) for x in args.reuse_batch)
                    raw, usage = run_anthropic_cached(spec, prefix, prompts, reuse.get(task))
                    provider, discounted = "anthropic", True
                else:
                    raw, usage = run_deepseek(spec, prompts, args.concurrency)
                    provider, discounted = "deepseek", False
            except Exception as error:  # noqa: BLE001
                _log(f"    {which}: FAILED {type(error).__name__}: {str(error)[:160]}")
                results["cells"].append({"task": task, "model": model_id, "error": str(error)[:300]})
                continue
            # Raw outputs first: a scorer failure must not cost a paid batch again.
            raw_dir = os.path.join(os.path.dirname(args.out), "fewshot_raw")
            os.makedirs(raw_dir, exist_ok=True)
            with open(os.path.join(raw_dir, f"{task}__{model_id}.json"), "w") as fh:
                json.dump({"usage": usage, "outputs": raw}, fh)
            scored = spec.score(eval_set, spec.extract_predictions(raw, eval_set))
            headline = scored.get(spec.metric_name, scored.get("f1"))
            usd = price(provider, model_id, usage, batch_discount=discounted)
            if usage.get("cache_write"):  # 1.25x base input, at the batch discount
                usd += price(provider, model_id, {"input": usage["cache_write"] * 1.25},
                             batch_discount=discounted)
            results["cells"].append({
                "task": task, "model": model_id, "metric": spec.metric_name, "shots": n_demos,
                "score": headline, "format_valid": scored.get("format_valid"), "rows": len(prompts),
                "usd": usd, "seconds": round(time.monotonic() - started, 1),
                **{f"tok_{k}": v for k, v in usage.items()}})
            _log(f"    {which:<10} {spec.metric_name}={headline}  format_valid="
                 f"{scored.get('format_valid')}  ${usd:.2f}  ({usage.get('errors', 0)} error(s))")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=2)
    if args.estimate:
        _log(f"ESTIMATE: anthropic ${est_total['anthropic']:.2f}; deepseek up to "
             f"${est_total['deepseek']:.2f} (no cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
