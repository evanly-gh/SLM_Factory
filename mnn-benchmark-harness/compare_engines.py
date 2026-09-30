#!/usr/bin/env python3
"""
compare_engines.py - choose which (model, quantization, engine, backend) to use, from recorded
benchmark results.

After the full sweep, every configuration (one model + quantization on one engine + backend) has a
result file written by run_autobench.py (SmolChat / llama.cpp) or run_mnn_autobench.py (MNN). This
script turns each file into one row of metrics, keeps the rows that satisfy ALL of your constraints
(--where), and prints them best-first, with the engine and backend to use for each.

A row's numbers are that file's SUMMARY: the mean over all questions of each question's reported
(last) run. The cold-start numbers are the mean of run 1 of each question (the run right after a
force-stop and page-cache eviction). Energy is only present when it was measured validly (phone
unplugged); a constraint on a metric a row does not have excludes that row.

Metrics you can constrain (names for --where):
  cold_start_ms  cold_load_ms  ttft_ms  ttlt_ms  prefill_tps  decode_tps
  native_prefill_tps  native_decode_tps  gen_tokens  memory_mb
  energy_mj_per_token  energy_net_mj_per_token

Examples:
  python3 compare_engines.py --where "decode_tps>=30" --where "cold_start_ms<=500" --where "memory_mb<=600"
  python3 compare_engines.py --where "ttft_ms<=250" --sort decode_tps --best-per-model
  python3 compare_engines.py --list                  # every configuration, no filtering
  python3 compare_engines.py --results-dir path/to/results --report-json picks.json

The older two-list MNN-vs-GGUF head-to-head report (which read files produced by the archived
run_fallback_agent_*.py scripts) was replaced by this; it is in git history.
"""

import argparse
import json
import operator
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "Benchmark-Harness"))
import bench_common  # noqa: E402  (model_and_quant / backend_label for files without labels)

# name -> (label, unit, lower_is_better)
METRICS = {
    "cold_start_ms": ("Cold start", "ms", True),
    "cold_load_ms": ("Cold load", "ms", True),
    "ttft_ms": ("TTFT", "ms", True),
    "ttlt_ms": ("TTLT", "ms", True),
    "prefill_tps": ("Prefill", "t/s", False),
    "decode_tps": ("Decode", "t/s", False),
    "native_prefill_tps": ("Native prefill", "t/s", False),
    "native_decode_tps": ("Native decode", "t/s", False),
    "gen_tokens": ("Tokens/answer", "", False),
    "memory_mb": ("Memory", "MB", True),
    "energy_mj_per_token": ("Energy", "mJ/tok", True),
    "energy_net_mj_per_token": ("Net energy", "mJ/tok", True),
}
OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}
WHERE_RE = re.compile(r"^\s*([a-z_]+)\s*(>=|<=|==|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")


def _mean(block):
    return block.get("mean") if isinstance(block, dict) else None


def _first(*values):
    return next((v for v in values if v is not None), None)


def row_from_file(path: Path, data: dict) -> dict:
    """One configuration's row: identity (model, quant, engine, backend) + metrics."""
    ri, s = data["run_info"], data["summary"]
    engine = ri.get("engine") or ("mnn" if "model_path" in ri else "smolchat")
    model, quant = (ri.get("model_name"), ri.get("quant_label")) if ri.get("model_name") else \
        bench_common.model_and_quant(ri.get("model_path") or ri.get("model") or path.stem, engine)
    backend = ri.get("backend") or bench_common.backend_label(engine, ri.get("backend_type"), data.get("results", []))
    cold = s.get("cold_start") or {}
    energy = s.get("energy_aggregate") or {}
    mem_kb = _first(_mean(s.get("memory_kb")), _mean(s.get("peak_rss_kb")))
    metrics = {
        "cold_start_ms": _mean(cold.get("cold_start_ms")),
        "cold_load_ms": _mean(cold.get("cold_load_ms")),
        "ttft_ms": _mean(s.get("ttft_ms")),
        "ttlt_ms": _mean(s.get("ttlt_ms")),
        "prefill_tps": _mean(s.get("prefill_tps")),
        "decode_tps": _mean(s.get("decode_tps")),
        "native_prefill_tps": _mean(s.get("native_prefill_tps")),
        "native_decode_tps": _mean(s.get("native_decode_tps")),
        "gen_tokens": _mean(s.get("gen_tokens")),
        "memory_mb": mem_kb / 1024 if mem_kb is not None else None,
        "energy_mj_per_token": _first(energy.get("energy_mj_per_token"), _mean(s.get("energy_mj_per_token"))),
        "energy_net_mj_per_token": _first(energy.get("energy_net_mj_per_token"),
                                          _mean(s.get("energy_net_mj_per_token"))),
    }
    return {
        "model": model, "quant": quant or "?", "engine": engine, "backend": backend,
        "questions": (s.get("ttft_ms") or {}).get("n_completed"),
        "file": str(path), "metrics": metrics,
    }


def load_rows(results_dir: Path, verbose: bool) -> list:
    rows = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
            if not (isinstance(data, dict) and "run_info" in data and "summary" in data):
                raise ValueError("no run_info/summary")
            rows.append(row_from_file(path, data))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if verbose:
                print(f"[skip] {path}: {exc}")
    return rows


def parse_constraints(specs: list) -> list:
    out = []
    for spec in specs:
        m = WHERE_RE.match(spec)
        if not m or m.group(1) not in METRICS:
            sys.exit(f"[ERROR] bad --where {spec!r}. Use METRIC OP NUMBER with OP one of >= <= > < == and "
                     f"METRIC one of: {', '.join(METRICS)}")
        out.append((m.group(1), m.group(2), float(m.group(3))))
    return out


def failures(row: dict, constraints: list) -> list:
    reasons = []
    for name, op, threshold in constraints:
        value = row["metrics"].get(name)
        if value is None:
            reasons.append(f"{name}: no valid measurement")
        elif not OPS[op](value, threshold):
            reasons.append(f"{name}={value:.4g} (needs {op} {threshold:g})")
    return reasons


def sort_key(sort_by: str):
    lower = METRICS[sort_by][2]

    def key(row):
        v = row["metrics"].get(sort_by)
        return (v is None, (v if lower else -v) if v is not None else 0)
    return key


def fmt(v, nd=0):
    return "-" if v is None else f"{v:.{nd}f}"


COLUMNS = [("MODEL", None), ("QUANT", None), ("ENGINE", None), ("BACKEND", None),
           ("COLD START ms", ("cold_start_ms", 0)), ("TTFT ms", ("ttft_ms", 0)), ("TTLT ms", ("ttlt_ms", 0)),
           ("PREFILL t/s", ("prefill_tps", 0)), ("DECODE t/s", ("decode_tps", 1)),
           ("MEM MB", ("memory_mb", 0)), ("mJ/TOK (net)", ("energy_net_mj_per_token", 1))]


def print_table(rows: list):
    cells = []
    for r in rows:
        cells.append([r["model"], r["quant"], r["engine"], r["backend"]] +
                     [fmt(r["metrics"].get(spec[0]), spec[1]) for _, spec in COLUMNS[4:]])
    widths = [max(len(COLUMNS[i][0]), *(len(c[i]) for c in cells)) for i in range(len(COLUMNS))]
    line = lambda cols: "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(line([c[0] for c in COLUMNS]))
    print("  ".join("-" * w for w in widths))
    for c in cells:
        print(line(c))


def main():
    default_dir = SCRIPT_DIR.parent / "Benchmark-Harness" / "sweep_results"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=default_dir,
                   help=f"Folder searched (recursively) for result JSONs (default: {default_dir}).")
    p.add_argument("--where", action="append", default=[], metavar="METRIC>=N",
                   help="A constraint on a recorded metric; repeat for several (all must hold).")
    p.add_argument("--sort", default="decode_tps", choices=list(METRICS),
                   help="Rank passing configurations by this metric, best first (default: decode_tps).")
    p.add_argument("--best-per-model", action="store_true",
                   help="Show only the best-ranked passing configuration for each model.")
    p.add_argument("--top", type=int, default=0, help="Show at most this many rows (0 = all).")
    p.add_argument("--list", action="store_true", help="Ignore constraints and list every configuration.")
    p.add_argument("--show-excluded", action="store_true", help="Also list rejected configurations and why.")
    p.add_argument("--report-json", type=Path, default=None, help="Also write the selection to this JSON file.")
    p.add_argument("--verbose", action="store_true", help="Say which files were skipped and why.")
    args = p.parse_args()

    constraints = [] if args.list else parse_constraints(args.where)
    rows = load_rows(args.results_dir, args.verbose)
    if not rows:
        sys.exit(f"[ERROR] no result files with run_info + summary found under {args.results_dir}")

    passing, excluded = [], []
    for row in rows:
        reasons = failures(row, constraints)
        (excluded if reasons else passing).append((row, reasons))
    passing_rows = sorted((r for r, _ in passing), key=sort_key(args.sort))
    if args.best_per_model:
        seen, best = set(), []
        for r in passing_rows:
            if r["model"] not in seen:
                seen.add(r["model"])
                best.append(r)
        passing_rows = best
    if args.top:
        passing_rows = passing_rows[:args.top]

    label = METRICS[args.sort]
    print(f"{len(rows)} configuration(s) found under {args.results_dir}")
    if constraints:
        print("Constraints: " + " AND ".join(f"{n} {o} {t:g}" for n, o, t in constraints))
    print(f"{len(passing)} pass; ranked by {label[0]} ({'lowest' if label[2] else 'highest'} first)"
          + ("; best per model" if args.best_per_model else "") + "\n")
    if passing_rows:
        print_table(passing_rows)
    else:
        print("No configuration satisfies all constraints.")
    if args.show_excluded or (constraints and not passing):
        print("\nExcluded:")
        for row, reasons in excluded:
            print(f"  {row['model']} {row['quant']} on {row['engine']}/{row['backend']}: " + "; ".join(reasons))

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps({
            "results_dir": str(args.results_dir), "constraints": [list(c) for c in constraints],
            "sort": args.sort, "best_per_model": args.best_per_model,
            "selected": passing_rows,
            "excluded": [{"row": r, "reasons": why} for r, why in excluded],
        }, indent=2, default=str))
        print(f"\n[OUTPUT] selection saved: {args.report_json}")


if __name__ == "__main__":
    main()
