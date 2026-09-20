"""Measure real token lengths against each task's caps, offline.

WHY THIS EXISTS AS ITS OWN PROBE

    "The baseline eval must not be depressed by token truncation" is one of the six failure modes
    the suite is audited for, and the evidence available so far was INDIRECT: the harness probe's
    `format_valid` sits near 1.0 after fine-tuning, which is consistent with no truncation but does
    not measure it. A format-valid answer can still be a truncated one, and on a task scored by
    exact match a single missing bracket is the difference between 1.0 and 0.0.

    The direct question is arithmetic and needs no GPU: tokenize the GOLD answer for every eval row
    and compare it against that task's `max_new_tokens`. A gold answer longer than the cap is an
    answer the model is structurally unable to emit — the eval would be scoring it wrong for a
    reason that has nothing to do with the model. Likewise `prompt + max_new_tokens` against
    `max_seq_length`: if the sum exceeds it, generation starts by evicting the prompt.

    Run against a real tokenizer rather than a chars/token estimate, because the estimate is what
    failed before: `config/token_budget.py` assumed 3.5 chars/token and GEC's tokenized learner
    English came in at 3.01, which cost a 400 from the endpoint and 45 of 60 rows on the 2026-09-08
    synthesis audit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASKS = ("topv2", "multiconer", "gec_bea19", "goemotions", "dialogsum")
# Both probe models, because the cap is in tokens and tokenizers disagree. SmolLM2 is the smaller
# vocabulary of the two and so the more pessimistic on the same string.
MODELS = ("Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM2-360M-Instruct")


def _gold_text(spec, row: dict) -> str:
    """The string the model has to produce for this row, rendered as the eval compares it."""
    for field in ("answer", "label"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    # Entity tasks carry structured gold; the model emits its JSON form.
    if row.get("entities") is not None:
        return json.dumps(row["entities"], ensure_ascii=False)
    return ""


def measure(task: str, tokenizer, cap_rows: int) -> dict:
    from tasks import get_task

    from data.eval_set import EvalSet

    spec = get_task(task)
    # `select_cap` rows, which is exactly what the in-loop eval scores.
    _train, eval_rows = spec.load(max_train=8, max_test=cap_rows, log=lambda *a: None)
    rows = list(eval_rows)[:cap_rows]

    # `build_prompts` takes the EvalSet, not a row list — the prompt is a property of the whole
    # sample for tasks that draw demonstrations from it.
    prompts = [str(p) for p in spec.build_prompts(EvalSet(all=rows, task=task))]

    answer_lens = [len(tokenizer(_gold_text(spec, r))["input_ids"]) for r in rows]
    prompt_lens = [len(tokenizer(p)["input_ids"]) for p in prompts]

    over_answer = [n for n in answer_lens if n > spec.max_new_tokens]
    over_window = [
        p for p in prompt_lens if p + spec.max_new_tokens > spec.max_seq_length
    ]
    ranked = sorted(answer_lens)

    def pct(values, q):
        return values[min(int(len(values) * q), len(values) - 1)] if values else 0

    return {
        "task": task,
        "rows": len(rows),
        "max_new_tokens": spec.max_new_tokens,
        "max_seq_length": spec.max_seq_length,
        "gold_answer_tokens": {
            "p50": pct(ranked, 0.50), "p99": pct(ranked, 0.99), "max": max(answer_lens or [0]),
        },
        "prompt_tokens": {"max": max(prompt_lens or [0])},
        # THE TWO NUMBERS THAT MATTER. Both must be zero.
        "gold_answers_over_cap": len(over_answer),
        "prompts_leaving_no_output_room": len(over_window),
        "headroom_factor": round(
            spec.max_new_tokens / max(max(answer_lens or [1]), 1), 2
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--out", default="logs/probes/token-headroom.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    results = {"cells": []}
    failures = 0
    for model in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        print(f"\n=== {model} ===", flush=True)
        print(f"{'task':<12}{'cap':>6}{'gold p50':>10}{'gold p99':>10}{'gold max':>10}"
              f"{'over cap':>10}{'no room':>9}{'headroom':>10}")
        for task in args.tasks:
            cell = measure(task, tokenizer, args.rows)
            cell["model"] = model
            results["cells"].append(cell)
            failures += cell["gold_answers_over_cap"] + cell["prompts_leaving_no_output_room"]
            g = cell["gold_answer_tokens"]
            print(f"{task:<12}{cell['max_new_tokens']:>6}{g['p50']:>10}{g['p99']:>10}"
                  f"{g['max']:>10}{cell['gold_answers_over_cap']:>10}"
                  f"{cell['prompts_leaving_no_output_room']:>9}"
                  f"{cell['headroom_factor']:>9}x", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {args.out}")
    print("VERDICT:", "no truncation exposure" if failures == 0
          else f"{failures} cell(s) EXCEED their cap — baselines are truncation-depressed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
