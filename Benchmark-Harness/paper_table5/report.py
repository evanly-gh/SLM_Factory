#!/usr/bin/env python3
"""
Build a Table 5-style report from one or more run_paper_table5.py result directories (any number of
phones / sessions; later sessions override earlier ones for the same configuration).

  python report.py results/galaxy_s23_pinned_20261005_101500 results/oneplus15_pinned_20261006_090000
  python report.py results/*  --out results/report

Writes TABLE5.md, TABLE5.html and TABLE5.csv:
  * the main table: model / backend / framework / quantization rows; prefill and decode throughput
    (tokens/s, mean +/- std) and energy (uJ/token) per phone, as in the paper;
  * a comparison with the paper for phones that are in it: our value, the paper's, and the ratio. A cell
    passes if it is within --tolerance (default 25%) of the paper's value for that phone, or inside the
    paper's own spread across its four phones.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = json.loads((HERE / "paper_reference.json").read_text())
PAPER_DEVICES = REFERENCE["devices"]
METRICS = [("prefill", "tps", "Prefill t/s"), ("prefill", "uj", "Prefill uJ/token"),
           ("decode", "tps", "Decode t/s"), ("decode", "uj", "Decode uJ/token")]
BACKEND_ORDER = {"cpu": 0, "gpu": 1}
FRAMEWORK_ORDER = {"llama.cpp": 0, "mnn": 1}
FRAMEWORK_LABEL = {"llama.cpp": "llama.cpp", "mnn": "MNN"}


def quant_order(q: str) -> tuple:
    base = q.split(" @")[0]
    bits = 16 if base in ("F16", "BF16") else int(base[1]) if base[1:2].isdigit() else 99
    return (bits, q)


def load(dirs: list[Path]) -> tuple[dict, dict]:
    """-> ({device_id: {config key: result}}, {device_id: {"label", "paper_column", "energy_kind", ...}})."""
    data: dict = {}
    devices: dict = {}
    for d in dirs:
        sess_file = d / "session.json"
        if not sess_file.exists():
            print(f"[skip] {d}: no session.json")
            continue
        sess = json.loads(sess_file.read_text())
        prof = sess["profile"]
        dev = devices.setdefault(prof["id"], {"label": prof.get("label", prof["id"]),
                                              "paper_column": prof.get("paper_column"), "sessions": [],
                                              "energy_methods": set(), "energy_kinds": set()})
        dev["sessions"].append(d.name)
        for f in sorted((d / "configs").glob("*.json")):
            r = json.loads(f.read_text())
            c = r["config"]
            quant = c["quant"] if c.get("ref", "pinned") == "pinned" else f"{c['quant']} @{c['ref']}"
            key = (c["model"], c["backend"], c["framework"], quant)
            data.setdefault(prof["id"], {})[key] = r
            if r.get("energy_headline_method"):
                dev["energy_methods"].add(r["energy_headline_method"])
            if r.get("energy_kind"):
                dev["energy_kinds"].add(r["energy_kind"])
    return data, devices


def value(r: dict | None, phase: str, kind: str):
    """(mean, std, note) for one cell; note marks failures / invalid energy."""
    if not r:
        return None, None, None
    if r.get("status") != "ok":
        return None, None, "failed"
    p = r.get(phase) or {}
    if kind == "tps":
        return p.get("tps"), p.get("tps_std"), None
    if p.get("uj_per_token") is None:
        return None, None, None
    return p["uj_per_token"], None, (None if p.get("energy_valid") else "invalid")


def not_cooled(r: dict | None) -> bool:
    """True if any benchmark invocation of the configuration started before the cool-down gate passed."""
    return bool(r) and any((inv.get("gate") or {}).get("passed") is False
                           for inv in (r.get("invocations") or {}).values())


def fmt_tps(m, s):
    if m is None:
        return "-"
    return f"{m:.1f}" if s is None else f"{m:.1f} +/- {s:.1f}"


def fmt_uj(m):
    if m is None:
        return "-"
    return f"{m:.1e}".replace("e+0", "e").replace("e+", "e")


def paper_value(model, backend, framework, phase, kind, device):
    row = REFERENCE["rows"].get(f"{model}/{backend}/{framework}")
    if not row or device not in PAPER_DEVICES:
        return None, None
    vals = row[f"{phase}_{kind}"]
    i = PAPER_DEVICES.index(device)
    return (vals[i] if i < len(vals) else None), vals


def build(data, devices, tolerance):
    keys = sorted({k for dev in data.values() for k in dev},
                  key=lambda k: (k[0], BACKEND_ORDER.get(k[1], 9), FRAMEWORK_ORDER.get(k[2], 9), quant_order(k[3])))
    dev_ids = list(devices)
    main_rows, cmp_rows, csv_rows = [], [], []
    for key in keys:
        model, backend, framework, quant = key
        cells = []
        for phase, kind, _ in METRICS:
            for dev in dev_ids:
                m, s, note = value(data.get(dev, {}).get(key), phase, kind)
                text = fmt_tps(m, s) if kind == "tps" else fmt_uj(m)
                if note == "failed":
                    text = "FAIL"
                elif note == "invalid":
                    text += "*"
                if text not in ("-", "FAIL") and not_cooled(data.get(dev, {}).get(key)):
                    text += "^"
                cells.append(text)
                csv_rows.append({"model": model, "backend": backend, "framework": framework, "quant": quant,
                                 "device": dev, "metric": f"{phase}_{kind}", "value": m, "std": s, "note": note})
        main_rows.append((key, cells))
        is_w4 = quant.upper().startswith("Q4")
        for dev in dev_ids:
            col = devices[dev]["paper_column"]
            r = data.get(dev, {}).get(key)
            if not col or not is_w4 or not r:
                continue
            for phase, kind, label in METRICS:
                ours, _, note = value(r, phase, kind)
                ref, spread = paper_value(model, backend, framework, phase, kind, col)
                if ours is None or ref is None:
                    continue
                ratio = ours / ref
                in_spread = bool(spread) and kind == "tps" and min(spread) <= ours <= max(spread)
                ok = abs(ratio - 1) <= tolerance or in_spread
                cmp_rows.append({"model": model, "backend": backend, "framework": framework, "quant": quant,
                                 "device": devices[dev]["label"], "metric": label, "ours": ours, "paper": ref,
                                 "ratio": ratio, "paper_spread": spread if kind == "tps" else None,
                                 "pass": ok and note != "invalid", "note": note})
    return dev_ids, main_rows, cmp_rows, csv_rows


def render_md(dev_ids, devices, main_rows, cmp_rows, tolerance) -> str:
    labels = [devices[d]["label"] for d in dev_ids]
    head = ["Model", "Backend", "Framework", "Quant."] + [f"{m[2]} ({lab})" for m in METRICS for lab in labels]
    lines = [f"# Table 5 replication (arXiv 2607.05475): 256-token prefill and decode",
             "", f"Generated {datetime.now().isoformat(timespec='seconds')}.", ""]
    for d in dev_ids:
        dv = devices[d]
        lines.append(f"- **{dv['label']}**: sessions {', '.join(dv['sessions'])}; paper column: "
                     f"{dv['paper_column'] or 'none (not in the paper)'}; energy: "
                     f"{', '.join(sorted(dv['energy_kinds'])) or 'n/a'} via {', '.join(sorted(dv['energy_methods'])) or 'n/a'}")
    lines += ["", "Throughput in tokens/s (mean +/- std over the recorded trials); energy in uJ/token. "
              "`*` = energy recorded but invalid (e.g. phone on USB power with no chip counters); "
              "`^` = started before the phone cooled to its gate temperature (gate timed out; see the result JSON); "
              "FAIL = configuration did not run (see the result JSON).", "",
              "| " + " | ".join(head) + " |", "|" + "---|" * 4 + "--:|" * (len(head) - 4)]
    prev = None
    for (model, backend, framework, quant), cells in main_rows:
        shown = model if model != prev else ""
        prev = model
        lines.append(f"| {shown} | {backend.upper()} | {FRAMEWORK_LABEL.get(framework, framework)} | {quant} | "
                     + " | ".join(cells) + " |")
    if cmp_rows:
        passed = sum(r["pass"] for r in cmp_rows)
        lines += ["", f"## Comparison with the paper ({passed}/{len(cmp_rows)} cells pass)", "",
                  f"Pass = within {tolerance:.0%} of the paper's value for the same phone, or (throughput) inside the "
                  "paper's own spread across its four phones.", "",
                  "| Model | Backend | Framework | Quant. | Phone | Metric | Ours | Paper | Ratio | Paper spread | Pass |",
                  "|---|---|---|---|---|---|--:|--:|--:|---|:-:|"]
        for r in cmp_rows:
            fmt = (lambda v: f"{v:.1f}") if "t/s" in r["metric"] else fmt_uj
            spread = (f"{min(r['paper_spread']):.1f}-{max(r['paper_spread']):.1f}" if r["paper_spread"] else "")
            lines.append(f"| {r['model']} | {r['backend'].upper()} | {FRAMEWORK_LABEL.get(r['framework'], r['framework'])} | "
                         f"{r['quant']} | {r['device']} | {r['metric']} | {fmt(r['ours'])}{'*' if r['note'] else ''} | "
                         f"{fmt(r['paper'])} | {r['ratio']:.2f}x | {spread} | {'yes' if r['pass'] else 'NO'} |")
    return "\n".join(lines) + "\n"


def render_html(dev_ids, devices, main_rows, cmp_rows) -> str:
    labels = [html.escape(devices[d]["label"]) for d in dev_ids]
    n = len(labels)
    h = ["<!doctype html><meta charset='utf-8'><title>Table 5 replication</title>",
         "<style>body{font-family:Georgia,serif;margin:24px}table{border-collapse:collapse;font-size:13px}"
         "td,th{padding:2px 8px;text-align:right}th{border-bottom:1px solid #000}td.l{text-align:left}"
         "tr.sep td{border-top:1px solid #000}caption{font-weight:bold;font-size:15px;margin-bottom:8px}"
         ".fail{color:#b00}.inv{color:#888}</style>",
         "<table><caption>Averaged throughput and energy across devices, models, frameworks and backends with 256 tokens"
         "</caption>",
         f"<tr><th rowspan=3>Model</th><th rowspan=3>Backend</th><th rowspan=3>Framework</th><th rowspan=3>Quantization</th>"
         f"<th colspan={2 * n}>Prefill</th><th colspan={2 * n}>Decode</th></tr>",
         f"<tr><th colspan={n}>Throughput (tokens/s)</th><th colspan={n}>Energy (&micro;J/token)</th>"
         f"<th colspan={n}>Throughput (tokens/s)</th><th colspan={n}>Energy (&micro;J/token)</th></tr>",
         "<tr>" + "".join(f"<th>{lab}</th>" for _ in range(4) for lab in labels) + "</tr>"]
    prev = None
    for (model, backend, framework, quant), cells in main_rows:
        sep = " class=sep" if model != prev else ""
        name = html.escape(model) if model != prev else ""
        prev = model
        tds = "".join(f"<td class={'fail' if c == 'FAIL' else 'inv' if c.endswith('*') else 'v'}>{html.escape(c)}</td>"
                      for c in cells)
        h.append(f"<tr{sep}><td class=l>{name}</td><td class=l>{backend.upper()}</td>"
                 f"<td class=l>{FRAMEWORK_LABEL.get(framework, framework)}</td><td class=l>{quant}</td>{tds}</tr>")
    h.append("</table>")
    if cmp_rows:
        h.append("<h3>Comparison with the paper</h3><table><tr><th>Model</th><th>Backend</th><th>Framework</th>"
                 "<th>Quant.</th><th>Phone</th><th>Metric</th><th>Ours</th><th>Paper</th><th>Ratio</th><th>Pass</th></tr>")
        for r in cmp_rows:
            fmt = (lambda v: f"{v:.1f}") if "t/s" in r["metric"] else fmt_uj
            h.append(f"<tr><td class=l>{r['model']}</td><td class=l>{r['backend'].upper()}</td>"
                     f"<td class=l>{FRAMEWORK_LABEL.get(r['framework'], r['framework'])}</td><td class=l>{r['quant']}</td>"
                     f"<td class=l>{html.escape(r['device'])}</td><td class=l>{r['metric']}</td><td>{fmt(r['ours'])}</td>"
                     f"<td>{fmt(r['paper'])}</td><td>{r['ratio']:.2f}x</td>"
                     f"<td class={'v' if r['pass'] else 'fail'}>{'yes' if r['pass'] else 'NO'}</td></tr>")
        h.append("</table>")
    return "\n".join(h) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="output directory (default: the first results directory)")
    ap.add_argument("--tolerance", type=float, default=0.25)
    args = ap.parse_args()
    data, devices = load(args.dirs)
    if not data:
        raise SystemExit("no results found")
    dev_ids, main_rows, cmp_rows, csv_rows = build(data, devices, args.tolerance)
    out = args.out or args.dirs[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / "TABLE5.md").write_text(render_md(dev_ids, devices, main_rows, cmp_rows, args.tolerance), encoding="utf-8")
    (out / "TABLE5.html").write_text(render_html(dev_ids, devices, main_rows, cmp_rows), encoding="utf-8")
    with open(out / "TABLE5.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print((out / "TABLE5.md").read_text(encoding="utf-8"))
    print(f"-> {out / 'TABLE5.md'}, TABLE5.html, TABLE5.csv")


if __name__ == "__main__":
    main()
