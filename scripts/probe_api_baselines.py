"""Zero-shot API baselines: how do the frontier models score on each task's own eval set?

WHAT THIS ANSWERS
    Every number in the suite so far compares a fine-tuned small model against the LOCAL teacher
    (Qwen3.6-35B-A3B, 5-shot). That leaves an obvious question unanswered: what would you get by
    simply calling a frontier model instead of fine-tuning anything? This measures exactly that,
    for the orchestrator's own model and for the cheap API teacher.

HOW IT STAYS COMPARABLE
    Prompts, extraction and scoring all come from the task's own `TaskSpec` — `build_prompts`,
    `extract_predictions`, `score` — over the SAME frozen eval set the runs used. So a number here
    is directly comparable to a `Score:` line in a run log rather than to a re-implementation of
    the task that happens to share its name. Nothing about the prompt is adjusted for the API.

    That constraint is load-bearing and it costs money. Anthropic will not cache a prefix below
    1,024 tokens, and no task's shared prompt prefix reaches it — topv2 comes closest at 873 of an
    882-token prompt, clinc150 has 1,041 tokens of prompt but only 538 shared because its label
    enumeration follows the input. Reordering either prompt would buy caching and would also mean
    measuring a prompt no student was ever evaluated on. Not worth it; see the batch API below for
    the discount taken instead.

THINKING IS OFF ON BOTH, and the two spell it differently — see `_sampling_extra_body` in
`data/synth_client.py` for what leaving DeepSeek's flag unset cost on run 39562029 (10.95M output
tokens for ~1,440 rows, 92% of that arm's bill). Anthropic's extended thinking is opt-in, so the
absence of a `thinking` argument is the disable.

USAGE
    python scripts/probe_api_baselines.py                     # all ten tasks, both models
    python scripts/probe_api_baselines.py --tasks gec_bea19 --models deepseek
    python scripts/probe_api_baselines.py --limit 50          # a cheap smoke test first
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The ten the suite reports on. Ordered cheapest-first so a failure surfaces before the expensive
# tasks run: clinc150 and topv2 are 60% of all input tokens between them.
TASKS = (
    "ner_bc5cdr", "gec_bea19", "sms_spam", "goemotions", "multiconer",
    "dialogsum", "xlam_bfcl", "calendar_json", "clinc150", "topv2",
)

ANTHROPIC_MODEL = os.environ.get("SLM_BASELINE_ANTHROPIC_MODEL", "claude-sonnet-5")
DEEPSEEK_MODEL = os.environ.get("SLM_BASELINE_DEEPSEEK_MODEL", "deepseek-v4-flash")

# Batch polling. The API returns `ended` when every request in the batch has resolved; these jobs
# are typically minutes, and the documented ceiling is 24h.
_POLL_S = 20.0
_BATCH_TIMEOUT_S = float(os.environ.get("SLM_BASELINE_BATCH_TIMEOUT_S", str(6 * 3600)))


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_task(task: str, limit: int | None):
    """The task's frozen eval rows and the prompts the students were scored on."""
    from data.eval_set import EvalSet
    from tasks import get_task

    spec = get_task(task)
    _train, eval_rows = spec.load(max_train=8, max_test=spec.select_cap, log=lambda *a: None)
    rows = list(eval_rows)[: limit or None]
    eval_set = EvalSet(all=rows, task=task)
    prompts = [str(p) for p in spec.build_prompts(eval_set)]
    return spec, EvalSet(all=rows, task=task), prompts[: len(rows)]


def run_anthropic(spec, prompts: list[str]) -> tuple[list[str], dict]:
    """One Message Batch per task, at half the synchronous price.

    BATCHES RATHER THAN 9,406 SEPARATE CALLS. The discount is 50% and this workload is the case it
    exists for: every request is independent, none is latency-sensitive, and caching — the other
    lever — is unavailable here for the prefix-length reason in the module docstring.
    """
    import anthropic

    from config.config import ANTHROPIC_API_KEY, orchestrator_client_kwargs

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())
    requests = [
        {
            "custom_id": f"row-{i}",
            "params": {
                "model": ANTHROPIC_MODEL,
                "max_tokens": spec.max_new_tokens,
                # No `thinking` key: extended thinking is opt-in, so omitting it disables it.
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for i, prompt in enumerate(prompts)
    ]
    batch = client.messages.batches.create(requests=requests)
    _log(f"    anthropic batch {batch.id} submitted ({len(requests)} request(s))")

    deadline = time.monotonic() + _BATCH_TIMEOUT_S
    while time.monotonic() < deadline:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        time.sleep(_POLL_S)
    else:
        raise TimeoutError(f"batch {batch.id} did not finish within {_BATCH_TIMEOUT_S}s")

    # Results arrive unordered, so they are placed back by `custom_id` rather than by arrival.
    outputs = [""] * len(prompts)
    usage = {"input": 0, "output": 0, "errors": 0}
    for result in client.messages.batches.results(batch.id):
        index = int(str(result.custom_id).split("-")[1])
        if result.result.type != "succeeded":
            usage["errors"] += 1
            continue
        message = result.result.message
        outputs[index] = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        usage["input"] += message.usage.input_tokens
        usage["output"] += message.usage.output_tokens
    return outputs, usage


def run_deepseek(spec, prompts: list[str], concurrency: int) -> tuple[list[str], dict]:
    """Synchronous fan-out. DeepSeek caches prefixes automatically at a far finer granularity than
    Anthropic's 1,024-token floor, so these prompts DO get cache reads at $0.007/Mtok."""
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    from openai import OpenAI

    from config.config import DEEPSEEK_API_KEY, SYNTH_API_BASE_URL

    client = OpenAI(
        base_url=SYNTH_API_BASE_URL, api_key=DEEPSEEK_API_KEY,
        timeout=180.0,
        http_client=httpx.Client(limits=httpx.Limits(max_connections=concurrency + 8)),
    )
    usage = {"input": 0, "output": 0, "cached": 0, "errors": 0}

    def one(prompt: str) -> str:
        try:
            reply = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=spec.max_new_tokens,
                temperature=0.0,
                # THE FLAG. Unset, DeepSeek reasons by default and bills the trace inside
                # max_tokens — which on run 39562029 meant HTTP 200s carrying no answer at all.
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as error:  # noqa: BLE001 — one bad row must not lose the task
            usage["errors"] += 1
            _log(f"      deepseek error: {type(error).__name__}: {str(error)[:90]}")
            return ""
        u = getattr(reply, "usage", None)
        if u is not None:
            usage["input"] += getattr(u, "prompt_tokens", 0) or 0
            usage["output"] += getattr(u, "completion_tokens", 0) or 0
            details = getattr(u, "prompt_tokens_details", None)
            usage["cached"] += getattr(details, "cached_tokens", 0) or 0 if details else 0
        return (reply.choices[0].message.content or "").strip()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outputs = list(pool.map(one, prompts))
    return outputs, usage


def price(provider: str, model: str, usage: dict, *, batch_discount: bool) -> float:
    """USD, from the same pricing table the runs bill against.

    `batch_discount` is applied here rather than in the table because it is a property of HOW the
    call was made, not of the model: the Message Batches API is half price for the same tokens.
    """
    from agent.cost import estimate_cost_usd

    try:
        cost = estimate_cost_usd(
            provider, model,
            input_tokens=usage.get("input", 0),
            output_tokens=usage.get("output", 0),
            cache_read_tokens=usage.get("cached", 0),
        )
    except Exception:  # noqa: BLE001 — a missing price must not lose the measurement
        return float("nan")
    cost = float(cost or 0.0)
    return cost * 0.5 if batch_discount else cost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--models", nargs="+", default=["anthropic", "deepseek"])
    parser.add_argument("--limit", type=int, default=None, help="rows per task (smoke tests)")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", default="logs/probes/api-baselines.json")
    args = parser.parse_args()

    results = {"model_ids": {"anthropic": ANTHROPIC_MODEL, "deepseek": DEEPSEEK_MODEL},
               "shots": 0, "cells": []}
    total_cost = 0.0
    for task in args.tasks:
        _log(f"── {task} ──")
        spec, eval_set, prompts = load_task(task, args.limit)
        _log(f"    {len(prompts)} eval row(s), metric={spec.metric_name}")
        for which in args.models:
            started = time.monotonic()
            try:
                if which == "anthropic":
                    raw, usage = run_anthropic(spec, prompts)
                    model_id, provider, discounted = ANTHROPIC_MODEL, "anthropic", True
                else:
                    raw, usage = run_deepseek(spec, prompts, args.concurrency)
                    model_id, provider, discounted = DEEPSEEK_MODEL, "deepseek", False
            except Exception as error:  # noqa: BLE001
                _log(f"    {which}: FAILED {type(error).__name__}: {str(error)[:120]}")
                results["cells"].append({"task": task, "model": which, "error": str(error)[:200]})
                continue

            predictions = spec.extract_predictions(raw, eval_set)
            scored = spec.score(eval_set, predictions)
            headline = scored.get(spec.metric_name, scored.get("f1"))
            usd = price(provider, model_id, usage, batch_discount=discounted)
            total_cost += 0.0 if usd != usd else usd
            cell = {
                "task": task, "model": model_id, "metric": spec.metric_name,
                "score": headline, "format_valid": scored.get("format_valid"),
                "rows": len(prompts), "usd": usd, "seconds": round(time.monotonic() - started, 1),
                **{f"tok_{k}": v for k, v in usage.items()},
            }
            results["cells"].append(cell)
            _log(f"    {which:<10} {spec.metric_name}={headline}  "
                 f"format_valid={scored.get('format_valid')}  ${usd:.2f}  "
                 f"({usage.get('errors', 0)} error(s))")

            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)

    _log(f"wrote {args.out} — total ${total_cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
