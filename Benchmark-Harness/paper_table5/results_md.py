#!/usr/bin/env python3
"""
Write docs/RESULTS.md: the Galaxy S23's measurements laid out like the paper's Table 5 (arXiv 2607.05475), next to
the paper's own values, plus the Q4_K_M rows, per-configuration run conditions and the protocol differences.

  python results_md.py results/galaxy_s23_pinned_20261007 [--out ../../docs/RESULTS.md]

Throughput is compared with the paper's Xiaomi 14 column (the S23's proxy_column in devices.json) and the paper's
range across its four phones. The paper reports energy only for the Xiaomi 17 and OnePlus 15, so the S23's energy
is shown next to those two values as context, without a pass/fail.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import report

HERE = Path(__file__).resolve().parent
MODEL_LABEL = {"llama3.2-1b": "Llama-3.2-1B", "qwen2.5-1.5b": "Qwen2.5-1.5B", "llama3.2-3b": "Llama-3.2-3B",
               "qwen2.5-7b": "Qwen2.5-7B"}
MODEL_ORDER = ["qwen2.5-1.5b", "qwen2.5-7b", "llama3.2-1b", "llama3.2-3b"]  # the paper's row order
W4 = {"llama.cpp": "Q4_0", "mnn": "Q4"}  # main rows: the 4-bit type each framework runs natively on its GPU path


def cell(r, phase, kind):
    m, s, note = report.value(r, phase, kind)
    if r and r.get("status") != "ok":
        return "FAIL"
    if m is None:
        return "–" if kind == "tps" or not r else "n/a"
    text = (f"{m:.1f} ± {s:.1f}" if s is not None else f"{m:.1f}") if kind == "tps" else report.fmt_uj(m)
    return text + ("*" if note == "invalid" else "") + ("^" if report.not_cooled(r) else "") \
        + ("~" if r.get("gpu_partial_offload") else "")


def ratio(r, phase, model, backend, framework):
    m, _, _ = report.value(r, phase, "tps")
    ref, _ = report.paper_value(model, backend, framework, phase, "tps", "xiaomi14")
    return f"{m / ref:.2f}×" if m is not None and ref else "–"


def paper_tps(model, backend, framework, phase):
    ref, spread = report.paper_value(model, backend, framework, phase, "tps", "xiaomi14")
    if ref is None:
        return "–", "–"
    return f"{ref:.1f}", f"{min(spread):.1f}–{max(spread):.1f}"


def paper_uj(model, backend, framework, phase):
    row = report.REFERENCE["rows"].get(f"{model}/{backend}/{framework}") or {}
    vals = row.get(f"{phase}_uj") or []
    return " / ".join(report.fmt_uj(v) for v in vals) if vals else "–"


def start_temps(r):
    temps = [(inv.get("gate") or {}).get("battery_temp_c") for inv in (r or {}).get("invocations", {}).values()]
    return [t for t in temps if t is not None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=HERE.parent.parent / "docs" / "RESULTS.md")
    args = ap.parse_args()

    data, devices = report.load(args.dirs)
    if "galaxy_s23" not in data:
        raise SystemExit("no Galaxy S23 results in the given directories")
    res = data["galaxy_s23"]
    sess = json.loads((args.dirs[0] / "session.json").read_text())
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in res)] + \
        sorted({k[0] for k in res} - set(MODEL_ORDER))
    combos = [(b, f) for b in ("cpu", "gpu") for f in ("llama.cpp", "mnn")]

    L = ["# Table 5 replication on a Galaxy S23", "",
         f"Measured {', '.join(sorted({r['started'][:10] for r in res.values()}))}; generated "
         f"{datetime.now().isoformat(timespec='minutes')} by `Benchmark-Harness/paper_table5/results_md.py` from "
         f"`{', '.join(str(d.as_posix()) for d in args.dirs)}`.", "",
         "Reference: *arXiv 2607.05475*, Table 5 (256-token prefill and 256-token decode, w4, llama.cpp and MNN on "
         "CPU and OpenCL GPU). Paper values are from `paper_reference.json` (its Table 9, which holds Table 5's "
         "rows per phone).", "",
         "## Setup", "",
         f"- **Phone:** Samsung Galaxy S23 (SM-S911U, Snapdragon 8 Gen 2 / SM8550, Adreno 740, 8 GB), Android "
         f"{sess['device'].get('android', sess['device'].get('sdk', '?'))}, unrooted, on battery over wireless adb.",
         "- **Builds:** llama.cpp `eadc418` and MNN `510ac8f`, the paper's versions (Table 3), built by "
         "`build_binaries.py`; tools `llama-bench` and `llm_bench`.",
         "- **Models:** the paper's four instruct models; llama.cpp GGUF **Q4_0** (main rows) and Q4_K_M "
         "(supplementary), MNN 4-bit with `llmexport` defaults.",
         "- **Protocol:** 1 warm-up + 3 recorded repetitions, mean ± std; llama.cpp `pp256` and `tg256` at depth 256; "
         "MNN `-p 256 -n 256 -kv true` with EOS ignored.",
         "- **The paper's closest phone is the Xiaomi 14** (Snapdragon 8 Gen 3, one generation newer), so throughput "
         "is compared with that column; *ratio* = S23 / Xiaomi 14. The paper's range over its four phones "
         "(Xiaomi 17, OnePlus 15, Xiaomi 15, Xiaomi 14) is shown for context.", ""]

    L += ["## Throughput (tokens/s), w4", "",
          "| Model | Backend | Framework | Prefill S23 | Prefill Xiaomi 14 (paper) | ratio | paper range | "
          "Decode S23 | Decode Xiaomi 14 (paper) | ratio | paper range |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for m in models:
        first = True
        for b, f in combos:
            r = res.get((m, b, f, W4[f]))
            pp, pps = paper_tps(m, b, f, "prefill")
            dp, dps = paper_tps(m, b, f, "decode")
            L.append(f"| {MODEL_LABEL.get(m, m) if first else ''} | {b.upper()} | {report.FRAMEWORK_LABEL[f]} | "
                     f"{cell(r, 'prefill', 'tps')} | {pp} | {ratio(r, 'prefill', m, b, f)} | {pps} | "
                     f"{cell(r, 'decode', 'tps')} | {dp} | {ratio(r, 'decode', m, b, f)} | {dps} |")
            first = False

    L += ["", "## Energy (µJ/token), w4", "",
          "S23: whole-phone battery energy (Perfetto battery current × voltage) **net of idle power** measured right "
          "before each run. Paper: SoC-only energy from Qualcomm powercap counters on rooted phones, reported for the "
          "Xiaomi 17 / OnePlus 15 only. The two are not the same quantity (see *Differences* below), so the paper's "
          "values are context, not a pass/fail target.", "",
          "| Model | Backend | Framework | Prefill S23 | Prefill paper (X17 / OP15) | Decode S23 | "
          "Decode paper (X17 / OP15) |", "|---|---|---|--:|--:|--:|--:|"]
    for m in models:
        first = True
        for b, f in combos:
            r = res.get((m, b, f, W4[f]))
            L.append(f"| {MODEL_LABEL.get(m, m) if first else ''} | {b.upper()} | {report.FRAMEWORK_LABEL[f]} | "
                     f"{cell(r, 'prefill', 'uj')} | {paper_uj(m, b, f, 'prefill')} | {cell(r, 'decode', 'uj')} | "
                     f"{paper_uj(m, b, f, 'decode')} |")
            first = False

    extra = sorted({k[3] for k in res if k[2] == "llama.cpp" and k[3] != W4["llama.cpp"]}, key=report.quant_order)
    if extra:
        L += ["", "## Supplementary: other llama.cpp 4-bit types", "",
              "The paper says only \"w4\". `Q4_K_M`: at `eadc418` llama.cpp's OpenCL backend has no Q4_K matmul, so a "
              "Q4_K_M GPU run does most of its matmuls on the CPU (marked `~`). `Q4_0-PURE`: Q4_0 with every tensor in "
              "Q4_0 (`llama-quantize --pure`, as Qualcomm's OpenCL instructions do); the default Q4_0 keeps the output / "
              "tied-embedding matrix at Q6_K, which the OpenCL backend runs unoptimized.", "",
              "| Model | Quant. | Backend | Prefill t/s | Prefill µJ/token | Decode t/s | Decode µJ/token |",
              "|---|---|---|--:|--:|--:|--:|"]
        for m in models:
            for q in extra:
                for b in ("cpu", "gpu"):
                    r = res.get((m, b, "llama.cpp", q))
                    if r:
                        L.append(f"| {MODEL_LABEL.get(m, m)} | {q} | {b.upper()} | {cell(r, 'prefill', 'tps')} | "
                                 f"{cell(r, 'prefill', 'uj')} | {cell(r, 'decode', 'tps')} | {cell(r, 'decode', 'uj')} |")

    L += ["", "Markers: `FAIL` = did not run (e.g. out of memory; see the configuration's JSON), `–` = not run yet, "
          "`n/a` = no valid energy, `*` = energy recorded but invalid, `^` = started before the cool-down gate passed, "
          "`~` = GPU run whose weight type has no OpenCL matmul kernel.", "",
          "## Run conditions", "",
          "| Configuration | Battery °C at each run start | CPU busy during runs | Finished |", "|---|--:|--:|---|"]
    all_t = []
    for k in sorted(res, key=lambda k: (MODEL_ORDER.index(k[0]) if k[0] in MODEL_ORDER else 9, k[1], k[2], k[3])):
        r = res[k]
        t = start_temps(r)
        all_t += t
        busy = [inv.get("cpu_busy") for inv in r.get("invocations", {}).values() if inv.get("cpu_busy") is not None]
        L.append(f"| {r['config']['id']} | {min(t):.1f}–{max(t):.1f} | "
                 f"{min(busy):.0%}–{max(busy):.0%} | {(r.get('finished') or '')[:16]} |" if t and busy else
                 f"| {r['config']['id']} | – | – | {(r.get('finished') or '')[:16]} |")
    if all_t:
        L += ["", f"Across all runs the battery was at {min(all_t):.1f}–{max(all_t):.1f} °C "
              f"(median {statistics.median(all_t):.1f} °C) when a timed run started."]

    failed = [r for r in res.values() if r.get("status") != "ok"]
    if failed:
        L += ["", "## Configurations that did not run", ""]
        L += [f"- `{r['config']['id']}`: {r.get('error')}" for r in sorted(failed, key=lambda r: r["config"]["id"])]
    L += ["", "## Sessions", "", "| Session | Models | Screen | Radios | Cool-down gate |", "|---|---|---|---|---|"]
    for d in args.dirs:
        recs = [json.loads(f.read_text()) for f in sorted((d / "sessions").glob("*.json"))] or             [json.loads((d / "session.json").read_text())]
        starts = [s["started"] for s in recs] + ["9999"]
        ran = [r["started"] for r in res.values()]
        for i, s in enumerate(recs):
            if not any(starts[i] <= t0 < starts[i + 1] for t0 in ran):
                continue  # stopped before finishing a configuration
            screen = ("on, minimum brightness" if s.get("screen_on", True) and not s.get("wake_lock")
                      else "off, CPU kept awake by a partial wake lock" if s.get("wake_lock") else "off")
            radios = s.get("radio_mode") or "mobile data, Bluetooth and location off (Wi-Fi kept for adb)"
            prof = s.get("profile", {})
            gate = f"≤ {s.get('max_temp_c')} °C" + (f", or stopped cooling ({prof['gate_plateau_s']} s)"
                                                    if prof.get("gate_plateau_s") else "")
            models = ", ".join(s["params"].get("models") or ["all prepared"])
            quants = s["params"].get("quants")
            L.append(f"| {s['started'][:16]} | {models}{' (' + ', '.join(quants) + ')' if quants else ''} | {screen} | "
                     f"{radios} | {gate} |")
    L += ["", "## Differences from the paper's protocol", "",
          "1. **Energy is whole-phone, net of idle, not SoC-only.** The paper reads Qualcomm powercap counters, which "
          "need root; this phone is unrooted and has no Power Stats rails, so energy comes from the battery fuel "
          "gauge (`perfetto:current`, chosen by `energy_probe.py`, run `SM-S911U_20261004_235137`). Idle power "
          "(screen at minimum brightness, Wi-Fi, OS) is measured for 12 s before every run and subtracted, which "
          "removes the constant draw but not the extra DRAM, regulator and board power the workload causes, so "
          "these values should read higher than SoC-only energy.",
          "2. **Wireless adb.** Energy needs the USB cable unplugged (it powers the phone), so adb runs over Wi-Fi and "
          "Wi-Fi stays on. A screen-off phone with no USB connection suspends every few seconds and freezes the "
          "benchmark; the first sessions kept the screen on at minimum brightness, later ones keep it off with a "
          "partial wake lock held by the adb shell (`tools/PbWake.java`), as the paper's screen-off protocol. "
          "Airplane mode is used when the phone keeps Wi-Fi on in it; otherwise mobile data, Bluetooth and "
          "location are switched off. Each session's setting is in *Sessions* above.",
          "3. **Cool-down gate.** As in the paper, every run starts with the battery at or below 28 °C (and the CPU "
          "frequency caps back at their maximum). With the screen off the phone rests at about 24.4 °C, so the gate "
          "passes on the paper's rule; a fallback (pass once the battery has stopped falling, ≤ 0.2 °C over 5 minutes, "
          "at most 32.5 °C) exists for a phone that rests above 28 °C, and was needed in the earlier screen-on runs, "
          "where the phone rested at 31.6–32.2 °C. Start temperatures are listed above.",
          "4. **Battery level.** Battery energy is current × voltage at the battery terminals, so a lower charge does "
          "not bias it directly; sessions stop below 20% charge (`--min-battery`).",
          "5. **llama.cpp CPU uses 4 threads.** `llama-bench`'s default on this phone is 8 threads (one per core), "
          "which puts work on the three Cortex-A510 little cores and makes every thread wait for them: "
          "Llama-3.2-1B Q4_0 prefill measured 68 tokens/s at 8 threads standalone and 20.8 tokens/s during a "
          "session, against 180 tokens/s at 4 threads (`-t 1,4,8` on 2026-10-07). MNN's default is 4 threads.",
          "6. **Phone generation.** The S23 is not in the paper; the Xiaomi 14 column is its closest comparison. Its "
          "Adreno 740 is also a generation older than the Adreno 750 that llama.cpp's Adreno-optimized OpenCL "
          "kernels were validated on, which is the likely reason llama.cpp GPU rows trail the paper the most.",
          "7. **MNN conversion** uses MNN 3.4.0's converter (28 commits after `510ac8f`) with `llmexport` defaults "
          "(quant block 64, no HQQ), since the paper gives no export settings."]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
