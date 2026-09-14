#!/usr/bin/env python3
"""
compare_engines.py — Per-model+quant MNN vs. GGUF comparison report.

Reads mnn_results.json and gguf_results.json (both flat lists of per-question
result entries, produced by run_fallback_agent_mnn.py and
run_fallback_agent_gguf.py's --full-run mode respectively - confirmed via
their actual output: each entry carries "model_id", "quant", "question_number",
"is_correct", and a "metrics" dict with "power_ma"/"power_ma_monsoon"/
"peak_rss_kb"/"ttft_ms" - GGUF's own metrics dict is a strict subset of
MNN's field names, already normalized to match in
run_fallback_agent_gguf.py's own extract_response_and_metrics()).

For every (model_id, quant) pair present in BOTH files, aggregates across all
of that pair's recorded questions on each engine, then compares the two
engines across 4 categories:

  - power:    mean power_ma_monsoon when available, else power_ma  (lower better)
  - memory:   mean peak_rss_kb                                      (lower better)
  - latency:  mean ttft_ms - see LATENCY METRIC CHOICE below         (lower better)
  - accuracy: % of questions with is_correct == True                (higher better)

LATENCY METRIC CHOICE: ttft_ms (time-to-first-token), not cold_load_ms and
not a combination of the two. cold_load_ms reflects one-time model-load
overhead - in this pipeline's design it's re-paid on every single attempt
(each retry/warmup attempt reloads the model fresh), which does not reflect
how a real deployment would behave (load once, serve many requests), so
averaging it in would overweight an artifact of the benchmark's own retry
mechanism rather than a property of the model+engine combination. ttft_ms
isolates genuine per-request responsiveness and is directly comparable
across both engines already.

The engine winning MORE of the 4 categories is "recommended" for that
model+quant - simple plurality, no 3-of-4 majority required (e.g. 2-1
decides it). A tie in a category (equal values, or one/both sides missing
data) does not count toward either engine's win total. When win counts end
up exactly equal (2-2, or 1-1 with 2 category-level ties, etc.), a tiebreak
priority order decides it instead: Accuracy > Power > Memory > Latency -
whichever engine won the highest-priority category among those still
undecided becomes the recommendation. Only when every single category is a
tie/N/A is the pair genuinely undecidable, reported as "true tie" - a rare,
real edge case, not the normal outcome of an equal win count.

Usage:
    python3 compare_engines.py
    python3 compare_engines.py --mnn-results mnn_results.json --gguf-results gguf_results.json --report-json comparison_report.json
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# (category, metric_label, unit, higher_is_better)
CATEGORIES = [
    ("power", "Power", "mA", False),
    ("memory", "Memory", "KB", False),
    ("latency", "Latency", "ms", False),
    ("accuracy", "Accuracy", "%", True),
]

# Values within this fraction of each other are treated as a tie rather than
# forcing a winner on floating-point noise (e.g. 100.00000001% vs 100%).
TIE_RELATIVE_TOLERANCE = 1e-6


def load_results(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list of result entries, got {type(data).__name__}")
    return data


def group_by_model_quant(entries: list) -> dict:
    """{(model_id, quant): [entry, ...]}"""
    grouped = {}
    for e in entries:
        key = (e["model_id"], e["quant"])
        grouped.setdefault(key, []).append(e)
    return grouped


def _mean(values: list):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def resolve_power_ma(metrics: dict):
    """Prefer the Monsoon reading when this question actually captured one;
    otherwise fall back to BatteryManager's power_ma. Either can be None on
    a given question (Monsoon hardware not attached; BatteryManager
    reporting invalid while charging/full) - resolved independently per
    question, not per model, since availability can vary question to
    question within the same run (confirmed in real output: power_ma is
    None on some questions and a real value on others within the same
    model+quant)."""
    monsoon = metrics.get("power_ma_monsoon")
    if monsoon is not None:
        return monsoon
    return metrics.get("power_ma")


def aggregate_engine(entries: list) -> dict:
    """One engine's aggregate stats for one (model_id, quant) pair, across
    all its recorded questions."""
    power_values = [resolve_power_ma(e.get("metrics") or {}) for e in entries]
    memory_values = [(e.get("metrics") or {}).get("peak_rss_kb") for e in entries]
    latency_values = [(e.get("metrics") or {}).get("ttft_ms") for e in entries]
    correct_count = sum(1 for e in entries if e.get("is_correct"))
    total = len(entries)

    return {
        "power": _mean(power_values),
        "memory": _mean(memory_values),
        "latency": _mean(latency_values),
        "accuracy": (100.0 * correct_count / total) if total > 0 else None,
        "n": total,
    }


def compare_category(mnn_value, gguf_value, higher_is_better: bool) -> str:
    """Returns 'mnn', 'gguf', 'tie', or 'N/A' (insufficient data on either side)."""
    if mnn_value is None or gguf_value is None:
        return "N/A"
    if mnn_value == 0 and gguf_value == 0:
        return "tie"
    denom = max(abs(mnn_value), abs(gguf_value), 1e-12)
    if abs(mnn_value - gguf_value) / denom <= TIE_RELATIVE_TOLERANCE:
        return "tie"
    mnn_is_better = (mnn_value > gguf_value) if higher_is_better else (mnn_value < gguf_value)
    return "mnn" if mnn_is_better else "gguf"


# When category wins end up exactly equal between engines, break the tie by
# walking categories in this priority order and taking the first one with a
# decisive (non-tie, non-N/A) winner. Accuracy leads because it's the
# closest proxy for "does the model actually work" - the other three are
# resource-efficiency metrics, useful only once the model is producing
# usable answers at all.
TIEBREAK_PRIORITY = ["accuracy", "power", "memory", "latency"]


def determine_recommended(categories: dict, wins: dict) -> str:
    """Plurality wins: whichever engine wins MORE categories overall is
    recommended - no 3-of-4 majority required (e.g. 2-1 decides it). Only
    when wins are exactly equal (2-2, or 1-1 with 2 category-level ties,
    etc.) does the TIEBREAK_PRIORITY order kick in: walk it in order and
    hand the recommendation to whichever engine won the first category in
    that list that wasn't itself a tie/N/A. "true tie" only when every
    single category is a tie/N/A too - a rare, genuine edge case, not the
    normal outcome of an equal win count.
    """
    if wins["mnn"] > wins["gguf"]:
        return "mnn"
    if wins["gguf"] > wins["mnn"]:
        return "gguf"
    for key in TIEBREAK_PRIORITY:
        winner = categories[key]["winner"]
        if winner in ("mnn", "gguf"):
            return winner
    return "true tie"


def compare_pair(model_id: str, quant: str, mnn_entries: list, gguf_entries: list) -> dict:
    mnn_agg = aggregate_engine(mnn_entries)
    gguf_agg = aggregate_engine(gguf_entries)

    categories = {}
    wins = {"mnn": 0, "gguf": 0}
    for key, label, unit, higher_is_better in CATEGORIES:
        winner = compare_category(mnn_agg[key], gguf_agg[key], higher_is_better)
        categories[key] = {
            "label": label,
            "unit": unit,
            "higher_is_better": higher_is_better,
            "mnn": mnn_agg[key],
            "gguf": gguf_agg[key],
            "winner": winner,
        }
        if winner in wins:
            wins[winner] += 1

    recommended = determine_recommended(categories, wins)

    return {
        "model_id": model_id,
        "quant": quant,
        "mnn_n": mnn_agg["n"],
        "gguf_n": gguf_agg["n"],
        "categories": categories,
        "wins": wins,
        "recommended_engine": recommended,
    }


def _fmt(v, nd=1):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "N/A"


def _cell(cat: dict) -> str:
    # Power specifically gets 3 decimal places, not the default 1 - at 1
    # decimal, genuinely different near-zero mA readings (e.g. 0.03 vs 0.52)
    # both round to "0.0" and look tied in the table even though the
    # underlying comparison (which uses the full-precision float, not this
    # display string) correctly resolved a real winner. Memory/latency/
    # accuracy formatting is untouched.
    if cat["unit"] == "mA":
        nd = 3
    elif cat["unit"] == "KB":
        nd = 0
    else:
        nd = 1
    mnn_s = _fmt(cat["mnn"], nd)
    gguf_s = _fmt(cat["gguf"], nd)
    winner_s = {"mnn": "MNN", "gguf": "GGUF", "tie": "tie", "N/A": "N/A"}[cat["winner"]]
    return f"{mnn_s}/{gguf_s}→{winner_s}"


def print_report_table(comparisons: list) -> None:
    print("\n" + "=" * 130)
    print("ENGINE COMPARISON (values shown as MNN/GGUF→winner; power=mA, memory=KB, latency=ms, accuracy=%)")
    print("=" * 130)
    col_widths = {"model": 26, "quant": 8, "cat": 22, "wins": 9, "rec": 16}
    header = (
        f"{'Model':<{col_widths['model']}}{'Quant':<{col_widths['quant']}}"
        f"{'Power':<{col_widths['cat']}}{'Memory':<{col_widths['cat']}}"
        f"{'Latency':<{col_widths['cat']}}{'Accuracy':<{col_widths['cat']}}"
        f"{'Wins(M-G)':<{col_widths['wins']}}{'Recommended':<{col_widths['rec']}}"
    )
    print(header)
    print("-" * len(header))
    for c in comparisons:
        cats = c["categories"]
        wins_s = f"{c['wins']['mnn']}-{c['wins']['gguf']}"
        row = (
            f"{str(c['model_id']):<{col_widths['model']}}{str(c['quant']):<{col_widths['quant']}}"
            f"{_cell(cats['power']):<{col_widths['cat']}}{_cell(cats['memory']):<{col_widths['cat']}}"
            f"{_cell(cats['latency']):<{col_widths['cat']}}{_cell(cats['accuracy']):<{col_widths['cat']}}"
            f"{wins_s:<{col_widths['wins']}}{c['recommended_engine']:<{col_widths['rec']}}"
        )
        print(row)
    print("=" * 130)


def parse_args():
    p = argparse.ArgumentParser(description="Per-model+quant MNN vs. GGUF comparison report.")
    p.add_argument("--mnn-results", default=str(SCRIPT_DIR / "logs" / "mnn_results.json"),
                    help="Path to mnn_results.json (default: ./logs/mnn_results.json).")
    p.add_argument("--gguf-results", default=str(SCRIPT_DIR / "logs" / "gguf_results.json"),
                    help="Path to gguf_results.json (default: ./logs/gguf_results.json).")
    p.add_argument("--report-json", default=str(SCRIPT_DIR / "logs" / "comparison_report.json"),
                    help="Where to write the machine-readable JSON report (default: ./logs/comparison_report.json).")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        mnn_entries = load_results(args.mnn_results)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] Failed to load '{args.mnn_results}': {exc}")
        sys.exit(1)

    try:
        gguf_entries = load_results(args.gguf_results)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] Failed to load '{args.gguf_results}': {exc}")
        sys.exit(1)

    mnn_by_pair = group_by_model_quant(mnn_entries)
    gguf_by_pair = group_by_model_quant(gguf_entries)

    common_pairs = sorted(set(mnn_by_pair) & set(gguf_by_pair))
    if not common_pairs:
        print(f"[ERROR] No (model_id, quant) pair is present in BOTH '{args.mnn_results}' and '{args.gguf_results}' - nothing to compare.")
        sys.exit(1)

    mnn_only = sorted(set(mnn_by_pair) - set(gguf_by_pair))
    gguf_only = sorted(set(gguf_by_pair) - set(mnn_by_pair))
    if mnn_only:
        print(f"[NOTE] {len(mnn_only)} pair(s) only in MNN results (skipped, no GGUF counterpart): {mnn_only}")
    if gguf_only:
        print(f"[NOTE] {len(gguf_only)} pair(s) only in GGUF results (skipped, no MNN counterpart): {gguf_only}")

    comparisons = [
        compare_pair(model_id, quant, mnn_by_pair[(model_id, quant)], gguf_by_pair[(model_id, quant)])
        for (model_id, quant) in common_pairs
    ]

    print_report_table(comparisons)

    report = {
        "mnn_results_file": args.mnn_results,
        "gguf_results_file": args.gguf_results,
        "total_pairs_compared": len(comparisons),
        "comparisons": comparisons,
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OUTPUT] JSON report saved: {args.report_json}")


if __name__ == "__main__":
    main()
