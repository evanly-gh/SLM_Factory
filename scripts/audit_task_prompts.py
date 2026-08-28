"""Does each task's PROMPT describe the format its GOLD actually uses?

WHY THIS EXISTS
    A prompt and a gold answer are two halves of one contract, written in different files, and
    nothing checked that they agree. `calendar_json` shipped for months with a prompt saying
    `{"name": ..., "arguments": {...}}` while every gold answer read
    `{"arguments": {...}, "name": ...}` — because the loader called `json.dumps(sort_keys=True)` and
    "arguments" sorts before "name". Key order does not affect PARSING, but it does decide the order
    the model learns to EMIT, and `name` is the field the scorer requires. Trained last, it is the
    most fragile token in the output.

    This prints the two halves side by side per task so a mismatch is visible rather than inferred,
    and asserts the ones that are machine-checkable.

USAGE
    python scripts/audit_task_prompts.py                    # all tasks, hand-written sample rows
    python scripts/audit_task_prompts.py calendar_json
    python scripts/audit_task_prompts.py --real calendar_json   # key order of REAL loaded gold

    The default pass uses a declared sample row, so it checks the prompt against the BUILDER. The
    key-order bug lived one layer lower, in the loader, so `--real` runs the loader and reports the
    key order of gold as it actually arrives. It is slow (a real dataset load per task), which is why
    it is opt-in rather than the default.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.eval_set import EvalSet
from tasks import TASKS, get_task
from tasks._builders import TrainingContext

# One representative row per task, in that task's own row schema. Enough to render the real prompt
# and the real training target without a network load.
SAMPLES: dict[str, dict] = {
    "calendar_json": {
        "text": ("Convert the request into a calendar.events.insert call.\n"
                 "Current date and time: 2026-03-01T09:00:00 (Sunday).\n\nAdd Dentist March 3 10am"),
        "answer": json.dumps([{"name": "calendar.events.insert", "arguments": {
            "summary": "Dentist", "start": {"dateTime": "2026-03-03T10:00:00"},
            "end": {"dateTime": "2026-03-03T11:00:00"}}}]),
        "tools": [{"name": "calendar.events.insert",
                   "parameters": {"properties": {"summary": {}}, "required": ["summary"]}}],
    },
    "xlam_bfcl": {
        "text": "weather in Paris?",
        "answer": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
        "tools": [{"name": "get_weather",
                   "parameters": {"properties": {"city": {}}, "required": ["city"]}}],
    },
    "toolbench": {
        "text": "You are AutoGPT...\nwhat is the weather in Paris?\nBegin!\n",
        "query": "what is the weather in Paris?",
        "answer": ("Thought: look it up.\nAction: get_weather_for_weather_api\n"
                   'Action Input: {"city": "Paris"}\nThought: done.\nAction: Finish\n'
                   'Action Input: {"return_type": "give_answer", "final_answer": "18C"}'),
        "tools": [{"name": "get_weather_for_weather_api",
                   "parameters": {"properties": {"city": "string"},
                                  "required": ["city"], "optional": []}}],
    },
    "ner_bc5cdr": {
        "text": "Aspirin induced gastritis.",
        "entities": [{"text": "Aspirin", "type": "Chemical"},
                     {"text": "gastritis", "type": "Disease"}],
    },
    "gsm8k": {"text": "Janet has 3 apples and buys 15. How many?", "answer": "Add.\n#### 18"},
    "dialogsum": {"text": "#Person1#: lunch? #Person2#: yes", "answer": "They agree to lunch."},
    "clinc150": {"text": "move money to savings", "label": "transfer"},
    "routerbench": {"text": "what is 2+2", "label": "local"},
    "sms_spam": {"text": "WINNER claim your prize", "label": "spam"},
    "proactive_listening": {"text": "A: the code was, um...", "label": "interrupt"},
}


def audit(name: str) -> list[str]:
    spec = get_task(name)
    row = SAMPLES[name]
    eval_set = EvalSet(all=[dict(row)], task=name)
    prompt = spec.build_prompts(eval_set)[0]
    labels = tuple(sorted({str(r["label"]) for r in eval_set.all if "label" in r}))
    instruction = ""
    if spec.build_prompts.__module__ == "eval.scorers.generation":
        from eval.scorers.generation import resolve_generation_instruction

        instruction = resolve_generation_instruction(eval_set.all)
    train_prompt, target, _marker = spec.build_training_turn(
        dict(row), TrainingContext(labels=labels, instruction=instruction),
    )

    print("=" * 78)
    print(f"{name}   ({spec.build_prompts.__module__})   metric={spec.metric_name}")
    print("-" * 78)
    print("PROMPT (what the model is told):")
    for line in prompt.splitlines()[:8]:
        print(f"  {line[:150]}")
    if len(prompt.splitlines()) > 8:
        print(f"  … ({len(prompt.splitlines())} lines total)")
    print("\nGOLD TARGET (what it is trained to emit):")
    print(f"  {str(target)[:220]}")

    problems: list[str] = []
    if train_prompt != prompt:
        problems.append("train prompt differs from the eval prompt (B250/B290)")

    # The machine-checkable half: for a task whose gold is a JSON call list, does the KEY ORDER the
    # prompt advertises match the order the gold actually emits?
    if spec.build_prompts.__module__ == "eval.scorers.function_call":
        try:
            calls = json.loads(str(target))
            keys = list(calls[0].keys()) if calls else []
        except (ValueError, TypeError, IndexError):
            keys = []
        advertised = prompt.find('"name"') < prompt.find('"arguments"')
        gold_first = keys[0] if keys else "?"
        print(f"\n  prompt advertises name-first: {advertised}    gold emits: {keys}")
        if advertised and gold_first != "name":
            problems.append(
                f"prompt says name-first, gold emits {gold_first}-first — the field the scorer "
                f"REQUIRES is trained last, after the whole argument block"
            )
        if "name" not in keys:
            problems.append("gold has no `name` key at all; the scorer would score it unparseable")
    print()
    for problem in problems:
        print(f"  ✗ {problem}")
    if not problems:
        print("  ✓ prompt and gold agree")
    return problems


def audit_real_gold(name: str) -> list[str]:
    """Load the task for real and report the key order gold actually arrives in."""
    from collections import Counter

    spec = get_task(name)
    train, eval_rows = spec.load(max_train=40, max_test=20)

    def key_orders(rows: list[dict]) -> Counter:
        counts: Counter = Counter()
        for row in rows:
            try:
                calls = json.loads(str(row.get("answer", "")))
                counts[tuple(calls[0].keys())] += 1
            except (ValueError, TypeError, IndexError, AttributeError):
                counts[("<unparseable>",)] += 1
        return counts

    print("=" * 78)
    print(f"{name}   REAL loaded gold")
    print("-" * 78)
    problems: list[str] = []
    for half, rows in (("train", list(train)), ("eval", list(eval_rows))):
        counts = key_orders(rows)
        print(f"  {half:5s}: {dict(counts)}")
        for keys, count in counts.items():
            if keys[0] != "name":
                problems.append(f"{half}: {count} row(s) emit {keys[0]} before name")
    for problem in problems:
        print(f"  ✗ {problem}")
    if not problems:
        print("  ✓ all real gold is name-first")
    return problems


def main() -> int:
    argv = sys.argv[1:]
    real = "--real" in argv
    names = [a for a in argv if not a.startswith("--")] or sorted(TASKS)
    if real:
        failed = {n: p for n in names if (p := audit_real_gold(n))}
        print("=" * 78)
        if failed:
            for name, problems in failed.items():
                for problem in problems:
                    print(f"  {name}: {problem}")
            return 1
        print(f"real gold for {len(names)} task(s) is name-first")
        return 0
    failed: dict[str, list[str]] = {}
    for name in names:
        if name not in SAMPLES:
            print(f"(no sample row declared for {name})")
            continue
        problems = audit(name)
        if problems:
            failed[name] = problems
    print("=" * 78)
    if failed:
        print("MISMATCHES:")
        for name, problems in failed.items():
            for problem in problems:
                print(f"  {name}: {problem}")
        return 1
    print(f"all {len(names)} audited task(s) agree with their gold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
