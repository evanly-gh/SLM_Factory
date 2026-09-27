#!/usr/bin/env python3
"""
extract_eval_questions.py — Extracts fixed, reproducible question slices from
eval_set.json (a math-reasoning benchmark) for two downstream uses:
  - phase2: a plain-text question list, for run_mnn_autobench.py's --questions
  - phase3: a JSON list of {text, answer} pairs, for later accuracy scoring

eval_set.json's actual structure (confirmed by inspection of the real file)
has one list per category - "pos", "neg", "boundary" - each holding dicts
with at least "text" and "answer" (plus "cot_reasoning"/"label", not needed
here). A category being a flat list of plain strings instead (answer-less)
is handled gracefully below, even though it wasn't observed in the real file,
since the input format may vary. A "difficulty" key also exists at the top
level, but it maps difficulty level -> list of question-text strings only -
duplicating pos/neg/boundary content rather than being its own category - so
it is excluded from the combined question list, the same as task_type/
counts/difficulty_counts.

Usage:
    python3 extract_eval_questions.py --eval-set eval_set.json
"""

import argparse
import json
import sys
from pathlib import Path

# Top-level keys that are metadata/indexing, not a category of questions.
NON_CATEGORY_KEYS = {"task_type", "counts", "difficulty_counts", "difficulty"}


def normalize_entry(item):
    """Return {"text": ..., "answer": ...} for one raw category entry, or
    None if it's a type this script doesn't know how to interpret (neither
    dict nor string) - such entries are skipped rather than crashing the
    whole extraction."""
    if isinstance(item, dict):
        text = item.get("text")
        if text is None:
            return None
        return {"text": text, "answer": item.get("answer")}
    if isinstance(item, str):
        return {"text": item, "answer": None}
    return None


def load_questions(eval_set_path):
    """Read eval_set.json and return (combined, per_category_counts).

    combined is every normalized {text, answer} question across every
    category found, in file order (whatever order dict keys/list items
    appear in the JSON) - never shuffled, so slicing from the start of this
    list is reproducible run to run. Any top-level key that isn't in
    NON_CATEGORY_KEYS and whose value is a list is treated as a category;
    everything else (unrecognized metadata) is silently ignored rather than
    guessed at.
    """
    with open(eval_set_path) as f:
        data = json.load(f)

    combined = []
    per_category_counts = {}

    for key, value in data.items():
        if key in NON_CATEGORY_KEYS:
            continue
        if not isinstance(value, list):
            continue

        found = 0
        skipped = 0
        for item in value:
            normalized = normalize_entry(item)
            if normalized is None:
                skipped += 1
                continue
            combined.append(normalized)
            found += 1

        per_category_counts[key] = {"found": found, "skipped": skipped}

    return combined, per_category_counts


def write_phase2(questions, output_path):
    """Plain text, one question per line, no answers - matches
    run_mnn_autobench.py's load_questions() expectations (one non-empty
    line per question)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            # Collapse embedded newlines so each question stays exactly one
            # line in the output file.
            f.write(" ".join(q["text"].split()) + "\n")


def write_phase3(questions, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)


def preview(text, limit=100):
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract fixed, reproducible question slices from eval_set.json."
    )
    p.add_argument("--eval-set", default="eval_set.json", help="Path to eval_set.json")
    p.add_argument(
        "--phase2-count", type=int, default=15,
        help="Number of questions in the phase2 plain-text slice (default: 15).",
    )
    p.add_argument(
        "--phase3-count", type=int, default=100,
        help="Number of questions in the phase3 JSON slice (default: 100).",
    )
    p.add_argument("--phase2-output", default="eval_questions_phase2.txt")
    p.add_argument("--phase3-output", default="eval_questions_phase3.json")
    return p.parse_args()


def main():
    args = parse_args()

    eval_set_path = Path(args.eval_set)
    if not eval_set_path.exists():
        print(f"[ERROR] eval set not found: {eval_set_path}")
        sys.exit(1)

    try:
        combined, per_category_counts = load_questions(eval_set_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to read/parse {eval_set_path}: {exc}")
        sys.exit(1)

    total_found = len(combined)
    print("=" * 60)
    print("Eval Question Extraction")
    print("=" * 60)
    print(f"Source: {eval_set_path}")
    for key, counts in per_category_counts.items():
        skip_note = (
            f" ({counts['skipped']} skipped: unrecognized entry type/missing text)"
            if counts["skipped"] else ""
        )
        print(f"  category '{key}': {counts['found']} questions{skip_note}")
    print(f"Total questions found across all categories: {total_found}")

    if total_found == 0:
        print("[ERROR] No usable questions found in the eval set.")
        sys.exit(1)

    phase2_count = min(args.phase2_count, total_found)
    phase3_count = min(args.phase3_count, total_found)
    if phase2_count < args.phase2_count:
        print(f"[WARN] Only {total_found} questions available; phase2 slice truncated to {phase2_count} (requested {args.phase2_count})")
    if phase3_count < args.phase3_count:
        print(f"[WARN] Only {total_found} questions available; phase3 slice truncated to {phase3_count} (requested {args.phase3_count})")

    # Fixed slice from the start of the combined list - not random - so
    # re-running this script (or re-running it later against the same
    # eval_set.json) always produces identical phase2/phase3 question sets.
    phase2_questions = combined[:phase2_count]
    phase3_questions = combined[:phase3_count]

    write_phase2(phase2_questions, args.phase2_output)
    write_phase3(phase3_questions, args.phase3_output)

    print(f"\n[OUTPUT] Phase 2 (plain text, no answers): {args.phase2_output} ({len(phase2_questions)} questions)")
    for q in phase2_questions[:2]:
        print(f"    - {preview(q['text'])}")

    print(f"\n[OUTPUT] Phase 3 (JSON, text+answer): {args.phase3_output} ({len(phase3_questions)} questions)")
    for q in phase3_questions[:2]:
        print(f"    - text: {preview(q['text'])}")
        print(f"      answer: {q['answer']!r}")

    print("\nDone.")


if __name__ == "__main__":
    main()
