#!/usr/bin/env python3
"""Combine smolchat_capital_france_results/<model>_<backend>.json (run_autobench.py output)
into SUMMARY.md and summary.csv: mean ± std over the recorded (non-warm-up) trials."""
import csv
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "new_mnn_paper_table_5_results" / "smolchat_capital_france_results"
MODELS = ["qwen2.5-1.5b-q4_k_m", "llama-3.2-1b-q4_k_m", "llama-3.2-3b-q4_k_m", "qwen2.5-7b-q4_k_m"]
BACKENDS = ["cpu", "opencl"]
METRICS = ["prefill_tps", "decode_tps", "ttft_ms", "energy_mj_sampled", "power_ma"]


def stat(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt(m, s, nd=1):
    return "n/a" if m is None else f"{m:.{nd}f} ± {s:.{nd}f}"


rows = []
for model in MODELS:
    for backend in BACKENDS:
        f = OUT / f"{model}_{backend}.json"
        if not f.exists():
            rows.append({"model": model, "backend": backend, "status": "missing"})
            continue
        d = json.loads(f.read_text())
        ok = [r for r in d["results"] if r.get("status") == "success" and r.get("metrics")]
        row = {"model": model, "backend": backend, "status": f"{len(ok)}/{len(d['results'])} ok",
               "backend_verified": ",".join(sorted({r["metrics"].get("backend_verified") or "?" for r in ok})),
               "start_time": d["run_info"].get("start_time"),
               "battery_status": d["run_info"].get("battery_status")}
        for k in METRICS:
            m, s, n = stat([r["metrics"].get(k) for r in ok])
            row[f"{k}_mean"], row[f"{k}_std"], row[f"{k}_n"] = m, s, n
        row["response_chars_mean"] = stat([len(r.get("response") or "") for r in ok])[0]
        rows.append(row)

cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "backend", "status"), k))
with open(OUT / "summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

lines = [
    "# SmolChat CPU vs OpenCL — \"What is the capital of France?\"",
    "",
    "- App: SmolChat (~/SLM_Factory-SmolChat, branch llama-pipeline + OpenCL fix), vendored llama.cpp 3018a11e7",
    "- Models: Q4_K_M (verified files), device: OnePlus CPH2749 / SM8850 (Adreno 840)",
    "- Protocol: run_autobench.py --trials 3 (1 discarded warm-up + 3 recorded), output ≤ 256 tokens (app default), "
    "CPU = n_gpu_layers 0, OpenCL = n_gpu_layers 99",
    "- Values: mean ± std over recorded trials",
    "",
    "| Model | Backend | Runs | Verified | Prefill t/s | Decode t/s | TTFT ms | Energy mJ* | Avg mA* |",
    "|---|---|---|---|--:|--:|--:|--:|--:|",
]
for r in rows:
    if r["status"] == "missing":
        lines.append(f"| {r['model']} | {r['backend']} | missing | | | | | | |")
        continue
    lines.append(
        f"| {r['model']} | {r['backend']} | {r['status']} | {r['backend_verified']} "
        f"| {fmt(r['prefill_tps_mean'], r['prefill_tps_std'])} | {fmt(r['decode_tps_mean'], r['decode_tps_std'])} "
        f"| {fmt(r['ttft_ms_mean'], r['ttft_ms_std'], 0)} | {fmt(r['energy_mj_sampled_mean'], r['energy_mj_sampled_std'])} "
        f"| {fmt(r['power_ma_mean'], r['power_ma_std'])} |")
lines += [
    "",
    "\\* Energy/current come from the app's BatteryManager sampler. The phone was USB-powered during the run, so "
    "battery current follows the charger's cycle rather than the model's load: these columns are recorded for "
    "completeness but are NOT valid inference energy measurements.",
    "",
    "Prefill is measured on the chat-templated prompt (a few dozen tokens), so it is dominated by fixed overhead and "
    "is not comparable to the paper's 256-token prefill.",
]
(OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print((OUT / "SUMMARY.md").read_text())
