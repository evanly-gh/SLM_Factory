"""
All four phones in the paper's Table 5 layout: docs/RESULTS.md and docs/figures/all_*.png.

Merges the Galaxy S23 results (results_md.py's single-phone document) with the OnePlus 15, OnePlus 8 Pro and
Redmi 9A results (results/all_results.md, results/graphs/make_table.py) into one document with one format. Every
phone is shown on the same candidate set: the 104 configurations of the GPU phones' table (9 models x 3 precisions x
2 engines x 2 backends, without Qwen3.5-4B F16, which was never prepared), 52 on the CPU-only Redmi 9A.

Data: results/slm_pool_<phone>_{pinned,head}/configs/*.json (benchmarks + energy) and .../memory/*.json
(memory_usage.py peaks); a phone's later results folders override earlier ones for the configurations they ran
(e.g. the OnePlus 8 Pro energy reruns replace its first, energy-less runs).

    python results_all_md.py
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

import report
from results_md import (C, COMBO_COLOR, COMBO_LABEL, COMBOS, DOCS, GRID, INK, INK2, LABEL, MUTED, POOL_ORDER,
                        QUANT_MARKER, QUANT_ORDER, QUANT_ROWS, RESULTS, SURFACE, T5_MODELS, W4, _combo_quant_legend,
                        load_dir, load_memory, paper, save, style, tps, uj, val)

HERE = Path(__file__).resolve().parent
PHASES = ("prefill", "decode")
T5_ALL = T5_MODELS + ["qwen2.5-7b"]  # Qwen2.5-7B: OnePlus 15 only (does not fit the S23's 8 GB)
SUPPLEMENT = RESULTS / "oneplus15_supplement.json"  # OnePlus 15 values whose raw files are not in this repository
EXCLUDED = {("qwen3.5-4b", "F16")}  # never prepared on any phone: not part of the 104-configuration table

PHONES = [
    {"id": "s23", "label": "Galaxy S23", "short": "S23",
     "hw": "Snapdragon 8 Gen 2 (SM8550), Adreno 740, 8 GB (6.9 GiB usable), Android 14", "gpu": True,
     "pool": ["slm_pool_s23_pinned", "slm_pool_s23_head"], "t5": ["galaxy_s23_pinned_20261007_screenoff"],
     "paper_idx": 3, "paper_rel": "proxy: the Xiaomi 14, one chip generation newer"},
    {"id": "oneplus15", "label": "OnePlus 15", "short": "OP15",
     "hw": "Snapdragon 8 Elite Gen 5 (SM8850), 16 GB (14.8 GiB usable), Android 16", "gpu": True,
     "pool": ["slm_pool_oneplus15_pinned", "slm_pool_oneplus15_head"], "t5": ["oneplus15_pinned_20261009_primed"],
     "paper_idx": 1, "paper_rel": "the same phone: the OnePlus 15 is one of the paper's four"},
    {"id": "oneplus8pro", "label": "OnePlus 8 Pro", "short": "OP8P",
     "hw": "Snapdragon 865 (SM8250), Adreno 650, 12 GB (11.2 GiB usable), Android 13", "gpu": True,
     "pool": ["slm_pool_oneplus8pro_pinned", "slm_pool_oneplus8pro_head",
              "slm_pool_oneplus8pro_energy_pinned", "slm_pool_oneplus8pro_energy_head"], "t5": [],
     "paper_idx": 3, "paper_rel": "loose proxy: the Xiaomi 14 is four chip generations newer"},
    {"id": "redmi9a", "label": "Redmi 9A", "short": "R9A",
     "hw": "Helio G25 (MT6762), 8x Cortex-A53, no OpenCL, 4 GB (3.7 GiB usable), 32-bit Android 11", "gpu": False,
     "pool": ["slm_pool_redmi9a_pinned", "slm_pool_redmi9a_head",
              "slm_pool_redmi9a_energy_pinned", "slm_pool_redmi9a_energy_head"], "t5": [],
     "paper_idx": None, "paper_rel": "no comparable phone in the paper"},
]
# Configurations with no result file, and why (recorded by the teammate's runs, results/all_results.md).
NOT_RUN = {
    "oneplus8pro": {("qwen3.5-0.8b", "gpu", "llama.cpp"): "head llama.cpp OpenCL build needs OpenCL symbols this "
                    "phone's driver lacks",
                    ("qwen3.5-2b", "gpu", "llama.cpp"): "same",
                    ("qwen3.5-4b", "gpu", "llama.cpp"): "same"},
}


def candidates(phone) -> list[tuple]:
    backends = ("cpu", "gpu") if phone["gpu"] else ("cpu",)
    out = []
    for m in POOL_ORDER:
        for qname, quant in QUANT_ROWS:
            if (m, quant["llama.cpp"]) in EXCLUDED:
                continue
            for b in backends:
                for f in ("llama.cpp", "mnn"):
                    out.append((m, b, f, quant[f]))
    return out


def merge_dirs(dirs: list[Path], loader) -> dict:
    """Later folders override earlier ones; an ok result is never replaced by a failed one."""
    out: dict = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for k, r in loader(d).items():
            if k in out and out[k].get("status") == "ok" and r.get("status") != "ok":
                continue
            out[k] = dict(r, _dir=d.name)
    return out


def load_phone(phone) -> dict:
    pool_dirs = [RESULTS / n for n in phone["pool"]]
    pool = merge_dirs(pool_dirs, lambda d: load_dir(d) if (d / "configs").is_dir() else {})
    t5 = merge_dirs([RESULTS / n for n in phone["t5"]], load_dir)
    memory = load_memory([d for d in pool_dirs if (d / "memory").is_dir()])
    sessions = []
    for d in pool_dirs:
        for f in sorted((d / "sessions").glob("*.json")) if d.is_dir() else []:
            try:
                sessions.append(json.loads(f.read_text()))
            except ValueError:
                pass
    if phone["id"] == "oneplus15" and SUPPLEMENT.exists():
        apply_supplement(json.loads(SUPPLEMENT.read_text(encoding="utf-8")), pool, t5, memory)
    return {"pool": {k: v for k, v in pool.items() if k in set(candidates(phone))}, "t5": t5, "memory": memory,
            "sessions": sessions, "cands": candidates(phone)}


def apply_supplement(sup: dict, pool: dict, t5: dict, memory: dict):
    """OnePlus 15 peak memory and energy cells (pool), and the paper-model reruns with energy (t5), from
    results/oneplus15_supplement.json. Fills only what the raw results here lack; the throughput in it is checked to match."""
    for x in sup["pool"]:
        k = (x["model"], x["backend"], x["framework"], x["quant"])
        if x.get("peak_total_mib") and k not in memory:
            memory[k] = {"peak_total_mib": x["peak_total_mib"], "peak_gpu_mib": x.get("peak_gpu_mib"),
                         "exit": 139 if x.get("crashed_after_workload") else 0,
                         "stderr_tail": "gen=256" if x.get("crashed_after_workload") else None, "source": "supplement"}
        r = pool.get(k)
        if not r or r.get("status") != "ok":
            continue
        for ph in PHASES:
            mj = x.get(f"{ph}_mj")
            if mj is not None and not (r.get(ph) or {}).get("energy_valid"):
                r[ph] = dict(r[ph], uj_per_token=mj * 1e3, energy_valid=True, energy_reason=None, energy_source="supplement")
    for x in sup.get("reruns", []):  # replace a first run that showed the slow-governor symptom
        k = (x["model"], x["backend"], x["framework"], x["quant"])
        if k in pool:
            for ph in PHASES:
                pool[k][ph] = dict(pool[k][ph], tps=x[f"{ph}_tps"], tps_std=x.get(f"{ph}_std"),
                                   uj_per_token=x[f"{ph}_mj"] * 1e3, energy_valid=True, energy_reason=None,
                                   energy_source="supplement rerun")
            pool[k]["rerun_note"] = (f"rerun; first run {x['first_run']['prefill_tps']} / {x['first_run']['decode_tps']} "
                                     "tokens/s (slow governor)")
    for x in sup["t5"]:
        k = (x["model"], x["backend"], x["framework"], x["quant"])
        t5[k] = {"status": "ok", "config": {"id": f"{x['model']}__{x['quant']}__{x['framework']}__{x['backend']}",
                                            "model": x["model"], "backend": x["backend"], "framework": x["framework"],
                                            "quant": x["quant"]},
                 **{ph: {"tps": x[f"{ph}_tps"], "tps_std": x.get(f"{ph}_std"), "energy_valid": x.get(f"{ph}_mj") is not None,
                         "uj_per_token": x[f"{ph}_mj"] * 1e3 if x.get(f"{ph}_mj") is not None else None}
                    for ph in PHASES}, "note": x.get("note")}


def manifest_storage() -> dict:
    out = {}
    for e in json.loads((HERE / "models" / "manifest.json").read_text())["models"]:
        out[(e["name"], e["framework"], e["quant"])] = sum(e["files"].values())
    return out


STORAGE = manifest_storage()


def storage(k) -> float | None:
    b = STORAGE.get((k[0], k[2], k[3]))
    return b / 1e6 if b else None


def short_reason(r: dict | None, phone_id: str, k) -> str:
    if r is None:
        note = NOT_RUN.get(phone_id, {}).get(k[:3])
        return "not run" + (" (no OpenCL build)" if note else "")
    e = (r.get("error") or "").lower()
    if "does not fit" in e or "swapped" in e:
        return "does not fit"
    if "exit 139" in e or "segmentation" in e:
        return "crashed"
    if "exit 137" in e or "oom" in e:
        return "killed (out of memory)"
    if "load" in e or "opencl" in e:
        return "load failed"
    return "failed"


def mem_cell(mem: dict | None, r: dict | None, phone_id: str, k) -> str:
    if r is None or r.get("status") != "ok":
        return short_reason(r, phone_id, k)
    if not mem:
        return "–"
    if mem.get("exit") == "skipped":
        return "does not fit"
    if mem.get("exit") == -9:
        return "> RAM (swapped)"
    finished = "gen=256" in (mem.get("stderr_tail") or "")
    if mem.get("exit") != 0 and not (finished and mem.get("peak_total_mib")):
        return "crashed"
    if mem.get("peak_total_mib") is None:
        return "–"
    gpu = mem.get("peak_gpu_mib")
    return (f"{mem['peak_total_mib'] * 1.048576:,.0f}" + (f" ({gpu * 1.048576:,.0f} GPU)" if gpu else "")
            + ("†" if mem.get("exit") != 0 else ""))


def peak_mb(mem: dict | None) -> float | None:
    if not mem or mem.get("peak_total_mib") is None:
        return None
    finished = mem.get("exit") == 0 or "gen=256" in (mem.get("stderr_tail") or "")
    return mem["peak_total_mib"] * 1.048576 if finished else None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _models_axis(ax, models):
    for x, m in enumerate(models):
        if m.startswith("qwen3.5"):
            ax.axvspan(x - 0.5, x + 0.5, color="#efeeea", zorder=0, linewidth=0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABEL[m] for m in models], color=INK2, fontsize=7.5, rotation=35, ha="right")


def fig_grid(data, metric, ylabel, fname, title):
    """Rows: 4/8/16-bit; columns: phones; x: the 9 models smallest to largest; one line per engine and backend.
    One shared log y-scale, so every panel compares directly. A gap = that configuration has no value."""
    models = POOL_ORDER
    fig, axes = plt.subplots(3, len(PHONES), figsize=(4.1 * len(PHONES), 10.5), sharex=True, sharey=True)
    for j, phone in enumerate(PHONES):
        d = data[phone["id"]]
        for i, (qname, quant) in enumerate(QUANT_ROWS):
            ax = axes[i][j]
            for combo in COMBOS:
                b, f = combo
                if b == "gpu" and not phone["gpu"]:
                    continue
                ys = [metric(d, (m, b, f, quant[f])) for m in models]
                ys = [float("nan") if y is None else y for y in ys]
                if all(y != y for y in ys):
                    continue
                ax.plot(range(len(models)), ys, color=COMBO_COLOR[combo], linewidth=1.8, marker=QUANT_MARKER[i],
                        markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.0, label=COMBO_LABEL[combo])
            ax.set_yscale("log")
            style(ax, ylabel=f"{qname}: {ylabel}" if j == 0 else None, title=phone["label"] if i == 0 else None)
            if not phone["gpu"] and i == 0:
                ax.text(0.98, 0.97, "CPU only (no OpenCL)", transform=ax.transAxes, ha="right", va="top",
                        color=MUTED, fontsize=7.5)
            if not ax.lines:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)
            _models_axis(ax, models)
    handles = [plt.Line2D([0], [0], color=COMBO_COLOR[c], linewidth=2, marker="o", markersize=5) for c in COMBOS]
    fig.legend(handles, [COMBO_LABEL[c] for c in COMBOS], loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, x=0.01, y=1.03, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return save(fig, fname)


def fig_storage(fname):
    """Storage per model and precision: the same files on every phone (GGUF for llama.cpp, MNN export directory)."""
    models = POOL_ORDER
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (qname, quant) in zip(axes, QUANT_ROWS):
        for k, f in enumerate(("llama.cpp", "mnn")):
            ys = [storage((m, "cpu", f, quant[f])) or 0 for m in models]
            ys = [0 if (m, quant["llama.cpp"]) in EXCLUDED else y for m, y in zip(models, ys)]
            ax.bar([x + (k - 0.5) * 0.38 for x in range(len(models))], ys, 0.36, color=(C[0], C[2])[k],
                   label={"llama.cpp": "llama.cpp (GGUF file)", "mnn": "MNN (export directory)"}[f])
        ax.set_yscale("log")
        style(ax, ylabel="Storage, MB (log scale)" if qname == "4-bit" else None, title=qname)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        _models_axis(ax, models)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, fname)


def fig_scatter_row(data, fname, xfun, yfun, xlabel, ylabel, guides=None, nrows_phases=False):
    """One panel per phone (and per phase if nrows_phases); color = engine/backend, shape = precision."""
    rows = PHASES if nrows_phases else (None,)
    fig, axes = plt.subplots(len(rows), len(PHONES), figsize=(4.1 * len(PHONES), 4.0 * len(rows)),
                             sharex=True, sharey=True, squeeze=False)
    allpts = []
    for i, ph in enumerate(rows):
        for j, phone in enumerate(PHONES):
            ax = axes[i][j]
            d = data[phone["id"]]
            for k in d["cands"]:
                x, y = xfun(d, k, ph), yfun(d, k, ph)
                if not x or not y:
                    continue
                allpts.append((x, y))
                ax.scatter(x, y, s=34, color=COMBO_COLOR[(k[1], k[2])], marker=QUANT_MARKER[QUANT_ORDER[k[3]]],
                           zorder=3, edgecolor=SURFACE, linewidth=0.9, label=COMBO_LABEL[(k[1], k[2])])
            ax.set_xscale("log")
            ax.set_yscale("log")
            title = phone["label"] if i == 0 else None
            yl = (f"{ph.capitalize()}: " if ph else "") + ylabel if j == 0 else None
            style(ax, ylabel=yl, title=title)
            ax.xaxis.grid(True, color=GRID, linewidth=0.8)
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
            if not ax.collections:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)
    if allpts and guides:
        lo, hi = min(p[0] for p in allpts) / 1.4, max(p[0] for p in allpts) * 1.4
        for ax in axes.flat:
            guides(ax, lo, hi)
            ax.set_xlim(lo, hi * 1.6)
    _combo_quant_legend(fig, list(axes.flat), loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.02 + 0.02 * (len(rows) == 1)))
    fig.tight_layout(rect=(0, 0, 1, 0.94 if len(rows) == 1 else 0.965))
    return save(fig, fname)


def _lines(values, unit, fmt=lambda v: f"{v:g}"):
    def draw(ax, lo, hi):
        for w in values:
            ax.plot([lo, hi], [w / lo, w / hi], color=GRID, linewidth=1, zorder=1)
            ax.text(hi * 1.04, w / hi, f"{fmt(w)} {unit}", color=MUTED, fontsize=7, va="center")
    return draw


def fig_t5(data):
    """The paper's models: our phones that ran them next to the paper's four phones (bars) - throughput."""
    ran = [p for p in PHONES if data[p["id"]]["t5"]]
    fig, axes = plt.subplots(2, len(T5_ALL), figsize=(4 * len(T5_ALL), 5.8), sharex=True)
    xs = range(len(COMBOS))
    n = len(ran) + 1
    wdt = 0.8 / n
    for j, m in enumerate(T5_ALL):
        for i, ph in enumerate(PHASES):
            ax = axes[i][j]
            for k, p in enumerate(ran):
                ys = [val(data[p["id"]]["t5"].get((m, b, f, W4[f])), ph, "tps") or 0 for b, f in COMBOS]
                ax.bar([x - 0.4 + wdt * (k + 0.5) for x in xs], ys, wdt * 0.92, color=C[[0, 2][k % 2]],
                       label=f"{p['label']} (ours)")
            ref = [paper(m, b, f, ph, "tps") for b, f in COMBOS]
            px = [x - 0.4 + wdt * (n - 0.5) for x in xs]
            ax.bar(px, [statistics.median(r) if r else 0 for r in ref], wdt * 0.92, color=C[1],
                   label="paper, median of its four phones")
            for x, r in zip(px, ref):
                if r:
                    ax.plot([x, x], [min(r), max(r)], color=INK2, linewidth=1.2, zorder=3)
                    for y in (min(r), max(r)):
                        ax.plot([x - wdt * 0.3, x + wdt * 0.3], [y, y], color=INK2, linewidth=1.2, zorder=3)
            style(ax, ylabel=f"{ph.capitalize()} tokens/s" if j == 0 else None, title=LABEL[m] if i == 0 else None)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([COMBO_LABEL[c].replace(" ", "\n") for c in COMBOS], color=INK2, fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    h.append(plt.Line2D([0], [0], color=INK2, linewidth=1.2))
    l.append("range over the paper's four phones")
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "all_t5_throughput_vs_paper.png")


def fig_t5_ratio(data):
    ran = [p for p in PHONES if data[p["id"]]["t5"] and p["paper_idx"] is not None]
    rows = [(m, b, f) for m in T5_ALL for b, f in COMBOS]
    fig, axes = plt.subplots(1, len(ran), figsize=(5.2 * len(ran), 5.4), sharey=True, squeeze=False)
    ys = list(range(len(rows)))[::-1]
    for ax, p in zip(axes[0], ran):
        ax.axvspan(0.75, 1.25, color="#f0efec", zorder=0)
        ax.axvline(1.0, color=MUTED, linewidth=1)
        for y, (m, b, f) in zip(ys, rows):
            pts = []
            for ph, col, mk in (("prefill", C[0], "o"), ("decode", C[1], "D")):
                o = val(data[p["id"]]["t5"].get((m, b, f, W4[f])), ph, "tps")
                r = paper(m, b, f, ph, "tps")
                if o and len(r) > p["paper_idx"]:
                    pts.append(o / r[p["paper_idx"]])
                    ax.scatter(pts[-1], y, s=46, color=col, marker=mk, zorder=3, edgecolor=SURFACE, linewidth=1.1,
                               label=ph.capitalize() if y == ys[0] else None)
            if len(pts) == 2:
                ax.plot(pts, [y, y], color=GRID, linewidth=2, zorder=1)
        style(ax, title=p["label"])
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ref = ("Xiaomi 17", "OnePlus 15", "Xiaomi 15", "Xiaomi 14")[p["paper_idx"]]
        ax.set_xlabel(f"{p['label']} throughput ÷ paper's {ref}", color=INK2, fontsize=9)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0][0].set_yticks(ys)
    axes[0][0].set_yticklabels([f"{LABEL[m]}  ·  {COMBO_LABEL[(b, f)]}" for m, b, f in rows], color=INK2, fontsize=8)
    axes[0][-1].legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return save(fig, "all_t5_ratio_vs_paper.png")


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

def tps_metric(ph):
    return lambda d, k: val(d["pool"].get(k), ph, "tps")


def mj_metric(ph):
    return lambda d, k: (lambda v: v / 1e3 if v else None)(val(d["pool"].get(k), ph, "uj"))


def session_facts(d) -> dict:
    s = d["sessions"]
    methods = sorted({r["energy_headline_method"] for k, r in d["pool"].items()
                      if r.get("energy_headline_method") and val(r, "decode", "uj") is not None})
    screen = sorted({"panel off (system awake)" if x.get("panel_off") else
                     "on, minimum brightness" if x.get("screen_on") else "off + wake lock" if x.get("wake_lock")
                     else "off" for x in s if "screen_on" in x})
    return {"methods": methods, "screen": screen}


def main():
    data = {p["id"]: load_phone(p) for p in PHONES}
    figs = {
        "prefill": fig_grid(data, tps_metric("prefill"), "tokens/s", "all_prefill_tps.png",
                            "Prefill throughput (tokens/s, log scale), 256-token prompt"),
        "decode": fig_grid(data, tps_metric("decode"), "tokens/s", "all_decode_tps.png",
                           "Decode throughput (tokens/s, log scale), 256 generated tokens"),
        "e_prefill": fig_grid(data, mj_metric("prefill"), "mJ/token", "all_prefill_energy.png",
                              "Prefill energy (mJ/token, log scale), whole phone net of idle"),
        "e_decode": fig_grid(data, mj_metric("decode"), "mJ/token", "all_decode_energy.png",
                             "Decode energy (mJ/token, log scale), whole phone net of idle"),
        "mem": fig_grid(data, lambda d, k: peak_mb(d["memory"].get(k)) if (d["pool"].get(k) or {}).get("status") == "ok"
                        else None, "MB", "all_peak_memory.png",
                        "Peak memory during inference (MB, log scale; PSS incl. GPU buffers)"),
        "storage": fig_storage("all_storage.png"),
        "mem_vs_storage": fig_scatter_row(
            data, "all_memory_vs_storage.png", lambda d, k, ph: storage(k),
            lambda d, k, ph: peak_mb(d["memory"].get(k)) if (d["pool"].get(k) or {}).get("status") == "ok" else None,
            "Model storage, MB (log scale)", "Peak memory, MB",
            guides=lambda ax, lo, hi: (ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1, zorder=1),
                                       ax.text(hi, hi * 1.05, "memory = file size", color=MUTED, fontsize=7,
                                               ha="right", va="bottom"))),
        "bw": fig_scatter_row(data, "all_decode_vs_model_size.png", lambda d, k, ph: storage(k) / 1e3,
                              lambda d, k, ph: val(d["pool"].get(k), "decode", "tps"),
                              "Model file size, GB (log scale)", "Decode tokens/s",
                              guides=_lines((2, 5, 10, 20, 40), "GB/s")),
        "power": fig_scatter_row(data, "all_energy_vs_speed.png", lambda d, k, ph: val(d["pool"].get(k), ph, "tps"),
                                 lambda d, k, ph: (lambda v: v / 1e3 if v else None)(val(d["pool"].get(k), ph, "uj")),
                                 "tokens/s (log scale)", "mJ/token",
                                 guides=_lines((500, 1000, 2000, 4000, 8000), "W", lambda w: f"{w / 1e3:g}"),
                                 nrows_phases=True),
        "t5": fig_t5(data), "ratio": fig_t5_ratio(data),
    }

    L = ["# Results: four phones in the paper's Table 5 layout", "",
         f"Generated {datetime.now().isoformat(timespec='minutes')} by `Benchmark-Harness/paper_table5/results_all_md.py` "
         "from `results/slm_pool_<phone>_*`. Reference: arXiv 2607.05475, Table 5: 256-token prefill and 256-token "
         "decode; throughput in tokens/s, energy in µJ/token (tables) or mJ/token (figures; 1 mJ = 1000 µJ).", "",
         "## Phones and status", "",
         "| Phone | Hardware | Configurations | Measured | With energy | With peak memory | Cannot run | Not run | "
         "Energy method | Screen during runs |",
         "|---|---|--:|--:|--:|--:|--:|--:|---|---|"]
    for p in PHONES:
        d = data[p["id"]]
        cands = d["cands"]
        ok = [k for k in cands if (d["pool"].get(k) or {}).get("status") == "ok"]
        e = [k for k in ok if val(d["pool"][k], "prefill", "uj") is not None and val(d["pool"][k], "decode", "uj") is not None]
        mem = [k for k in ok if peak_mb(d["memory"].get(k)) is not None]
        fail = [k for k in cands if k in d["pool"] and d["pool"][k].get("status") != "ok"]
        miss = [k for k in cands if k not in d["pool"]]
        sf = session_facts(d)
        L.append(f"| {p['label']} | {p['hw']} | {len(cands)} | {len(ok)} | {len(e)} | {len(mem)} | {len(fail)} | "
                 f"{len(miss)} | {', '.join(f'`{m}`' for m in sf['methods']) or 'none'} | {'; '.join(sf['screen']) or '–'} |")
        p["_counts"] = {"ok": len(ok), "e": len(e), "mem": len(mem), "fail": len(fail), "miss": len(miss)}
    L += ["",
          "- **Configurations:** 9 models × 3 precisions × 2 engines × 2 backends without Qwen3.5-4B F16 (never "
          "prepared) = 104 per GPU phone; 52 on the Redmi 9A, which has no OpenCL. The Galaxy S23's extra Qwen3.5-4B "
          "F16 rows (all four do not fit) are left out so every phone has the same table.",
          "- **Runs:** 1 warm-up + 3 recorded repetitions (mean ± std; slow configurations sometimes 1), llama.cpp "
          "with 4 threads, MNN with its default 4; MNN decode timed on the wall clock; every run starts at ≤ 28 °C "
          "battery with the CPU frequency caps at their maximum.",
          "- **Energy** is whole-phone battery energy during the timed work minus the idle power measured just "
          "before it. The paper's is SoC-only (Qualcomm powercap counters, root), reported for the Xiaomi 17 and "
          "OnePlus 15, so ours reads higher (section 6). None of our phones is rooted.",
          "- **Builds:** `pinned` = the paper's llama.cpp `eadc418` / MNN `510ac8f` (Qwen3, SmolLM2, Gemma 3); "
          "`head` = llama.cpp `9c2e0e4` / MNN `a8ec497` (Qwen3.5; text decoder only).", ""]

    # ---- 1. paper models ----
    ran = [p for p in PHONES if data[p["id"]]["t5"]]
    hdr_ours = " | ".join(p["short"] for p in ran)
    L += ["## 1. The paper's models next to the paper's four phones", "",
          "### Throughput (tokens/s), 4-bit", "",
          f"| Model | Backend | Framework | Prefill {hdr_ours} | paper X17 | OP15 | X15 | X14 | Decode {hdr_ours} | "
          "paper X17 | OP15 | X15 | X14 |",
          "|---|---|---|" + "--:|" * (2 * (len(ran) + 4))]
    for m in T5_ALL:
        first = True
        for b, f in [("cpu", "llama.cpp"), ("cpu", "mnn"), ("gpu", "llama.cpp"), ("gpu", "mnn")]:
            cells = []
            for ph in PHASES:
                for p in ran:
                    r = data[p["id"]]["t5"].get((m, b, f, W4[f]))
                    cells.append(tps(val(r, ph, "tps"), ((r or {}).get(ph) or {}).get("tps_std")))
                ref = paper(m, b, f, ph, "tps")
                cells += [tps(ref[i]) if len(ref) > i else "–" for i in range(4)]
            L.append(f"| {LABEL[m] if first else ''} | {b.upper()} | {report.FRAMEWORK_LABEL[f]} | " + " | ".join(cells) + " |")
            first = False
    not_ran = [p["label"] for p in PHONES if not data[p["id"]]["t5"]]
    L += ["", "X17 = Xiaomi 17, OP15 = OnePlus 15, X15 = Xiaomi 15, X14 = Xiaomi 14 (paper values). "
          f"{' and '.join(not_ran)} did not run the paper's models (only the SLM_Factory pool, sections 3–4). "
          "Qwen2.5-7B does not fit in the S23's 8 GB (–). OnePlus 15 paper-model rows: 1 warm-up + 3 repetitions, "
          "wake/sleep key pair before each run; four cells (Llama-3.2-1B and Qwen2.5-1.5B llama.cpp CPU, Qwen2.5-1.5B and "
          "Qwen2.5-7B MNN CPU) are reruns after a slow-governor first pass or a swap-guard stop (first-pass prefill: "
          "92.3, 28.8, 111.7 tokens/s and failed).",
          "", "### Energy (µJ/token), 4-bit", "",
          f"| Model | Backend | Framework | Prefill {hdr_ours} | paper X17 | OP15 | Decode {hdr_ours} | paper X17 | OP15 |",
          "|---|---|---|" + "--:|" * (2 * (len(ran) + 2))]
    for m in T5_ALL:
        first = True
        for b, f in [("cpu", "llama.cpp"), ("cpu", "mnn"), ("gpu", "llama.cpp"), ("gpu", "mnn")]:
            cells = []
            for ph in PHASES:
                cells += [uj(val(data[p["id"]]["t5"].get((m, b, f, W4[f])), ph, "uj")) for p in ran]
                ref = paper(m, b, f, ph, "uj")
                cells += [uj(ref[i]) if len(ref) > i else "–" for i in range(2)]
            L.append(f"| {LABEL[m] if first else ''} | {b.upper()} | {report.FRAMEWORK_LABEL[f]} | " + " | ".join(cells) + " |")
            first = False
    L += ["", "Energy is whole-phone net of idle (ours) against SoC-only (paper), so ours should read higher; on the "
          "OnePlus 15 most cells read lower instead (section 2).", ""]

    # ---- 2. comparison ----
    L += ["## 2. How our phones compare with the paper", "", f"![Throughput vs paper]({figs['t5']})", "",
          "*Prefill (top) and decode (bottom) throughput per framework and backend for the phones that ran the "
          "paper's models, next to the median of the paper's four phones; the dark bracket is the paper's range.*", "",
          f"![Ratio vs paper]({figs['ratio']})", "",
          "*Each row is one configuration: our throughput divided by the paper's value for the same phone (OnePlus 15) "
          "or its closest phone (S23: Xiaomi 14). The shaded band is ±25%.*", ""]
    for p in ran:
        if p["paper_idx"] is None:
            continue
        rs = {ph: [] for ph in PHASES}
        for m in T5_ALL:
            for b, f in COMBOS:
                for ph in PHASES:
                    o, ref = val(data[p["id"]]["t5"].get((m, b, f, W4[f])), ph, "tps"), paper(m, b, f, ph, "tps")
                    if o and len(ref) > p["paper_idx"]:
                        rs[ph].append(o / ref[p["paper_idx"]])
        allr = rs["prefill"] + rs["decode"]
        es = {ph: [] for ph in PHASES}  # energy against the same paper phone (the paper reports X17 and OP15 only)
        if p["paper_idx"] == 1:
            for m in T5_ALL:
                for b, f in COMBOS:
                    for ph in PHASES:
                        o, ref = val(data[p["id"]]["t5"].get((m, b, f, W4[f])), ph, "uj"), paper(m, b, f, ph, "uj")
                        if o and len(ref) > 1:
                            es[ph].append(o / ref[1])
        if allr:
            L.append(f"- **{p['label']}** ({p['paper_rel']}): {sum(0.75 <= x <= 1.25 for x in allr)} of {len(allr)} "
                     f"throughput cells within ±25%; median ratio {statistics.median(rs['prefill']):.2f} for prefill, "
                     f"{statistics.median(rs['decode']):.2f} for decode."
                     + (f" Energy per token against the paper's own OnePlus 15 values: median {statistics.median(es['prefill']):.2f}× "
                        f"for prefill, {statistics.median(es['decode']):.2f}× for decode; ours is whole-phone net of idle and "
                        "the paper's SoC-only, so ours should not read lower, yet prefill does in most cells. The paper's "
                        "SoC counters and our idle-subtracted battery reading split the phone's power differently, so "
                        "treat the absolute comparison as approximate." if es["prefill"] else ""))
    L += ["- **OnePlus 8 Pro and Redmi 9A** are not in the paper and have no close proxy (the Snapdragon 865 is four "
          "generations older than the Xiaomi 14; the Helio G25 is an entry-level chip), so they are compared with the "
          "other phones on the pool in sections 3–4.", ""]

    # ---- 3. pool tables ----
    L += ["## 3. SLM_Factory model pool on every phone", "",
          "Pool: `config/android_pool.py` on SLM_Factory `main`. Quantizations: llama.cpp Q4_0 / Q8_0 / F16, MNN 4 / 8 / "
          "16-bit. **Storage** is the size of the model files on the phone (GGUF file or MNN export directory), in MB "
          "(10^6 bytes), the same on every phone. **Peak memory** is the inference process's largest total memory "
          "(PSS from `dumpsys meminfo`, GPU buffers included) during one 256 + 256-token run, measured by "
          "`memory_usage.py`; the GPU share is in brackets; † = crashed after completing the workload. `–` = not "
          "measured on that phone. FAIL = the configuration cannot run there; the reason is in the memory column and "
          "section 7.", "",
          f"![Prefill throughput, all phones]({figs['prefill']})", "",
          f"![Decode throughput, all phones]({figs['decode']})", "",
          "*Every configuration: one column per phone, one row per precision, models smallest to largest; one line "
          "per engine and backend, all panels on one log scale. A gap means that configuration could not run (does not "
          "fit, crashed, no OpenCL). Shaded columns are Qwen3.5, which runs on the newer builds, so jumps there partly "
          "reflect the build.*", "",
          f"![Storage]({figs['storage']})", "",
          f"![Peak memory, all phones]({figs['mem']})", "",
          f"![Memory vs storage]({figs['mem_vs_storage']})", "",
          "*Storage per model and precision (identical on every phone), peak memory during inference per phone, and "
          "the two against each other: points above the grey line need more memory than the model's file size (GPU "
          "runs allocate buffers for weights, KV cache and work space, and llama.cpp GPU keeps a CPU copy too).*", "",
          f"![Decode vs model size]({figs['bw']})", "",
          "*Decode speed against model file size per phone. Decode reads all weights once per token, so points on a "
          "grey line share one effective memory bandwidth.*", ""]
    for n, p in enumerate(PHONES, 1):
        d = data[p["id"]]
        c = p["_counts"]
        L += [f"### 3.{n} {p['label']}", "",
              f"{p['hw']}. {len(d['cands'])} configurations: {c['ok']} measured ({c['e']} with energy, {c['mem']} with "
              f"peak memory), {c['fail']} cannot run, {c['miss']} not run.", "",
              "| Model | Backend | Framework | Quant. | Storage MB | Peak memory MB | Prefill t/s | Prefill µJ/token | "
              "Decode t/s | Decode µJ/token |", "|---|---|---|---|--:|--:|--:|--:|--:|--:|"]
        for m in POOL_ORDER:
            first = True
            keys = sorted((k for k in d["cands"] if k[0] == m),
                          key=lambda k: (QUANT_ORDER[k[3]], report.BACKEND_ORDER[k[1]], report.FRAMEWORK_ORDER[k[2]]))
            for k in keys:
                r = d["pool"].get(k)
                if not r or r.get("status") != "ok":
                    row = ["FAIL" if r else "not run"] * 4
                else:
                    row = [tps(val(r, "prefill", "tps"), r["prefill"].get("tps_std")), uj(val(r, "prefill", "uj")),
                           tps(val(r, "decode", "tps"), r["decode"].get("tps_std")), uj(val(r, "decode", "uj"))]
                sz = storage(k)
                row = [f"{sz:,.0f}" if sz else "–", mem_cell(d["memory"].get(k), r, p["id"], k)] + row
                L.append(f"| {LABEL[m] if first else ''} | {k[1].upper()} | {report.FRAMEWORK_LABEL[k[2]]} | {k[3]} | "
                         + " | ".join(row) + " |")
                first = False
        noe = [k for k in d["cands"] if (d["pool"].get(k) or {}).get("status") == "ok"
               and (val(d["pool"][k], "prefill", "uj") is None or val(d["pool"][k], "decode", "uj") is None)]
        if noe:
            reasons = []
            for k in noe:
                r = d["pool"][k]
                why = "; ".join(sorted({x for x in (r["prefill"].get("energy_reason"), r["decode"].get("energy_reason"))
                                        if x})) or "no energy reading"
                reasons.append(f"`{r['config']['id']}` ({why})")
            L += ["", f"`–` energy ({len(noe)} rows): " + ", ".join(reasons) + "."]
        if p["id"] == "oneplus8pro":
            aborted = []
            for n in p["pool"]:
                for f in sorted((RESULTS / n / "configs").glob("*.json")) if (RESULTS / n / "configs").is_dir() else []:
                    x = json.loads(f.read_text())
                    k = (x["config"]["model"], x["config"]["backend"], x["config"]["framework"], x["config"]["quant"])
                    if x.get("status") != "ok" and "exit 134" in (x.get("error") or "") \
                            and (d["pool"].get(k) or {}).get("status") == "ok":
                        aborted.append(f"`{x['config']['id']}`")
            if aborted:
                L += ["", f"**Energy not measurable ({len(aborted)} llama.cpp GPU rows):** " + ", ".join(sorted(set(aborted)))
                      + ". Their throughput above comes from the first (screen-on) runs. The energy runs abort inside the Adreno "
                      "650 OpenCL driver (`cl_a6x_cmdbuf_mgr_submit_ibs`, exit 134) minutes into the first prefill. These rows "
                      "prefill at 0.5–3 tokens/s, so one GPU call runs for minutes; a driver hang watchdog is the likely cause "
                      "(not confirmed). The same kind of row kernel-panicked the phone three times with the display panel off."]
            first_runs =merge_dirs([RESULTS / "slm_pool_oneplus8pro_pinned", RESULTS / "slm_pool_oneplus8pro_head"],
                                    lambda dd: load_dir(dd) if (dd / "configs").is_dir() else {})
            diffs = []
            for k in d["cands"]:
                r, o = d["pool"].get(k), first_runs.get(k)
                if not r or r.get("_dir", "").find("_energy_") < 0 or (o or {}).get("status") != "ok" or r.get("status") != "ok":
                    continue
                ratios = [val(r, ph, "tps") / val(o, ph, "tps") for ph in PHASES]
                if any(abs(x - 1) > 0.10 for x in ratios):
                    diffs.append(f"`{r['config']['id']}` ({ratios[0]:.2f}× / {ratios[1]:.2f}×, rerun {r['started'][5:16].replace('T', ' ')})")
            if diffs:
                L += ["", f"**Throughput differs by more than 10% from the first (screen-on, energy-less) OnePlus 8 Pro runs** "
                      f"in {len(diffs)} rows (rerun ÷ first run, prefill / decode): " + ", ".join(diffs) + ". Q4 rows "
                      "reproduce within 5%; most differences are 8-bit and 16-bit CPU rows, 1.5–2.5× faster in the rerun. "
                      "Cause (tested 2026-10-20): the moment the OnePlus 8 Pro falls asleep it caps policy4 at 1.48 of "
                      "2.42 GHz, and the same benchmark (SmolLM2-135M Q8_0 CPU prefill) runs at 124–228 tokens/s asleep "
                      "against 907–921 awake. OxygenOS sleeps after 30 min whatever the screen timeout says, and the "
                      "harness checked the caps only at the start and end of each run, so runs that fell asleep midway "
                      "were slowed without a flag. The faster rerun values are the uncapped ones (they also match the S23, "
                      "where Q8_0 CPU is about as fast as Q4_0); the first runs' 8/16-bit CPU values were partly capped. "
                      "The harness now keeps the phone awake (checked every 2 s) and records the lowest cap during every "
                      "run; rows measured before that fix with only a once-a-minute wake (2026-10-20 12:39–19:00) are "
                      "being rerun."]
        rr = [f"`{d['pool'][k]['config']['id']}` ({d['pool'][k]['rerun_note']})" for k in d["cands"]
              if (d["pool"].get(k) or {}).get("rerun_note")]
        if rr:
            L += ["", "**Rerun after a slow-governor first run** (prefill far below the phone's other measurements, "
                  "decode normal; the rerun replaces it): " + ", ".join(rr) + "."]
        odd = [d["pool"][k]["config"]["id"] for k in d["cands"] if (d["pool"].get(k) or {}).get("status") == "ok"
               and k[1] == "gpu" and k[2] == "llama.cpp"
               and (val(d["pool"][k], "prefill", "tps") or 0) < (val(d["pool"][k], "decode", "tps") or 0)]
        if odd:
            L += ["", f"Prefill slower than decode ({len(odd)} llama.cpp GPU rows): " + ", ".join(f"`{i}`" for i in odd)
                  + ". A slow OpenCL prefill path of that llama.cpp build for these weight types on this GPU (section 5)."]
        L += [""]

    # ---- 4. energy ----
    L += ["## 4. Energy report", "",
          f"![Prefill energy, all phones]({figs['e_prefill']})", "",
          f"![Decode energy, all phones]({figs['e_decode']})", "",
          "*Energy per token, one column per phone, one row per precision (log scale, mJ/token). A gap means that "
          "configuration could not run or has no valid energy.*", "",
          f"![Energy vs speed]({figs['power']})", "",
          "*Energy per token against speed per phone. Energy per token = power ÷ speed, so points on a grey line "
          "draw the same net power; lower right is better.*", "",
          "### Net power while running (W above idle, median and range)", "",
          "| Backend | Framework | " + " | ".join(f"{p['short']} prefill | {p['short']} decode" for p in PHONES) + " |",
          "|---|---|" + "--:|" * (2 * len(PHONES))]
    for c in [("cpu", "llama.cpp"), ("cpu", "mnn"), ("gpu", "llama.cpp"), ("gpu", "mnn")]:
        cells = []
        for p in PHONES:
            for ph in PHASES:
                w = [val(r, ph, "uj") * val(r, ph, "tps") / 1e6 for k, r in data[p["id"]]["pool"].items()
                     if (k[1], k[2]) == c and val(r, ph, "uj") and val(r, ph, "tps")]
                cells.append(f"{statistics.median(w):.1f} ({min(w):.1f}–{max(w):.1f})" if w else "–")
        L.append(f"| {c[0].upper()} | {report.FRAMEWORK_LABEL[c[1]]} | " + " | ".join(cells) + " |")
    L += [""]
    for p in PHONES:
        d = data[p["id"]]
        best = []
        for m in POOL_ORDER:
            for ph in PHASES:
                c = sorted((val(d["pool"].get((m, b, f, W4[f])), ph, "uj"), COMBO_LABEL[(b, f)]) for b, f in COMBOS
                           if val(d["pool"].get((m, b, f, W4[f])), ph, "uj"))
                if c:
                    best.append((LABEL[m], ph, c[0]))
        if best:
            pre = {(n): e for n, ph, e in best if ph == "prefill"}
            dec = {(n): e for n, ph, e in best if ph == "decode"}
            L.append(f"- **{p['label']}, cheapest 4-bit configuration per model:** " + "; ".join(
                f"{n}: prefill {pre[n][1]} ({uj(pre[n][0])}), decode {dec[n][1]} ({uj(dec[n][0])})"
                for n in pre if n in dec) + " (µJ/token).")
    L += ["", "- **Energy per token is set by speed far more than by power.** Net power stays within roughly 0.5–10 W "
          "on every phone while speed spans three orders of magnitude, so the slow configurations are the expensive "
          "ones, and the slow phones pay several times more per token for the same model.",
          "- **Decode costs several times more per token than prefill** on every phone: prefill processes the 256 "
          "tokens in batched passes, decode makes one pass per token.", ""]
    L += [""] + TRUST + [""] + PROTOCOL + [""]

    # ---- 7. could not run ----
    L += ["## 7. Configurations that could not run", ""]
    for p in PHONES:
        d = data[p["id"]]
        bad = [(k, d["pool"].get(k)) for k in d["cands"] if (d["pool"].get(k) or {}).get("status") != "ok"]
        if not bad:
            continue
        L += [f"**{p['label']}** ({len(bad)}):", ""]
        for k, r in bad:
            cid = f"{k[0]}__{k[3]}__{k[2]}__{k[1]}"
            why = (r.get("error") or "failed") if r else ("not run: " + NOT_RUN.get(p["id"], {}).get(k[:3], "no result")
                                                       .replace("same", "head llama.cpp OpenCL build needs OpenCL "
                                                                "symbols this phone's driver lacks"))
            L.append(f"- `{cid}`: {why[:260]}")
        L += [""]
    (DOCS / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {DOCS / 'RESULTS.md'} and {len([f for f in figs.values() if f])} figures")


TRUST = [
    "## 5. How far to trust these numbers", "",
    "- **Repeatable.** The recorded repetitions of a configuration vary by 0.1–5% on every phone (MNN CPU prefill on "
    "the OnePlus phones up to ~30%: its first repetition runs before the big cores ramp up). On the S23 a standalone "
    "`llama-bench` run outside the harness gave the same Llama-3.2-1B prefill (179.9 vs 179.6 tokens/s).",
    "- **Physically consistent.** Decode reads every weight once per token, so each engine reaches a roughly constant "
    "effective memory bandwidth across models on each phone (section 3 figure).",
    "- **Energy is reproducible but absolute values are approximate.** On the S23 the same configuration measured "
    "hours apart, screen on and screen off, differed by −7% to +2%. Absolute values carry roughly ±10–20% from each "
    "phone's battery gauge, and gauges differ between phones (sampled current on the S23 and OnePlus phones, a "
    "slowly updated average current on the Redmi 9A), so compare configurations within a phone more confidently "
    "than across phones.",
    "- **OnePlus 15 slow-governor cells.** A OnePlus 15 benchmark that started 20 s or more after the last screen "
    "change ran with its CPU clocks at the floor; the harness now sends a wake/sleep key pair right before each "
    "run. A few cells may still be affected (Qwen3-0.6B llama.cpp Q8_0 CPU, Qwen3.5-4B llama.cpp Q4_0 CPU: prefill "
    "far below their neighbours).",
    "- **GPU decode varies between sessions by 10–20%** (S23 tests ruled out the screen, the energy trace and the "
    "battery level). Compare GPU decode rows with that margin in mind.",
    "",
    "**Known slow paths (real measurements of these builds, not errors):**", "",
    "- **llama.cpp GPU (OpenCL) on older Adreno GPUs.** On the S23's Adreno 740 llama.cpp GPU prefill is no faster "
    "than the CPU; on the OnePlus 8 Pro's Adreno 650 it is slower than the CPU in every cell (0.5–115 tokens/s). "
    "The OpenCL backend's kernels target newer Adreno GPUs; on the OnePlus 15 it is the fastest prefill.",
    "- **llama.cpp GPU with Q8_0 or F16 weights** at `eadc418` has prefill slower than decode on the S23 and the "
    "OnePlus 8 Pro (flagged per phone in section 3).",
    "- **llama.cpp \"Q4_0\" files are not all 4-bit.** `llama-quantize` keeps the output matrix at Q6_K by default "
    "(28% of Qwen3-0.6B's Q4_0 file), and the pinned OpenCL Q6_K kernel is unoptimized, so on the GPU several small "
    "models decode faster at Q8_0 than at Q4_0.",
    "- **MNN GPU decode on the paper's MNN version** costs about 3–4 ms per layer per token whatever the model size "
    "(about 9 tokens/s for the small models on the S23 and OnePlus 8 Pro). The newer MNN used for Qwen3.5 does not.",
    "- **MNN GPU with 16-bit weights** crashes (segfault) on every phone with a GPU backend, in both MNN versions.",
    "- **llama.cpp CPU with F16 weights** is slow because the Android build leaves half-precision vector arithmetic "
    "off (llama.cpp's own recipe); on the Redmi 9A's 32-bit build it is 7–10× slower than Q4_0.",
]

PROTOCOL = [
    "## 6. Differences from the paper's protocol", "",
    "1. **Energy is whole-phone, net of idle, not SoC-only.** The paper reads Qualcomm powercap counters, which need "
    "root; none of our phones is rooted or exposes Power Stats rails. Energy comes from each phone's battery gauge, "
    "sampled every 100 ms: Perfetto's battery current × voltage on the S23 and OnePlus 15; the battery sysfs files "
    "(average current) on the Redmi 9A; Android's battery service (BatteryManager current, read by "
    "`tools/PbBattery.java`) on the OnePlus 8 Pro, whose battery files are hidden from the shell and whose Perfetto "
    "(v25) cannot record the battery voltage. Idle power is measured for 12 s before every run and subtracted; that "
    "removes the constant draw (including the display's, where it is on) but not the extra memory and regulator "
    "power the workload causes, so values read higher than SoC-only energy.",
    "2. **Wireless adb, phone on battery.** adb runs over Wi-Fi with the other radios off. Screen: the S23 and Redmi "
    "9A run with the screen off and a partial wake lock held by the adb shell (`tools/PbWake.java`) so the phone "
    "does not suspend mid-run. The OnePlus 8 Pro caps its CPU (policy4 at 1.48 of 2.42 GHz) whenever the screen is "
    "asleep, so it runs awake with only the display panel switched off (`tools/PbPanel.java`, as scrcpy does) or, "
    "in its first sessions, with the screen on at minimum brightness. The OnePlus 15 gets a wake/sleep key pair "
    "before each run (section 5).",
    "3. **Cool-down gate.** Every run starts with the CPU frequency caps at their maximum and the battery at or below "
    "28 °C as in the paper (the S23 also passes once its battery has stopped cooling). The OnePlus 8 Pro and Redmi 9A "
    "energy reruns (2026-10-20) used 32 °C: in that room both phones rested at 29–31 °C, so a 28 °C gate would have "
    "timed out before every run. Each run's start temperature is recorded in its result file.",
    "4. **llama.cpp CPU uses 4 threads.** `llama-bench`'s default of one thread per core includes the little cores "
    "and ran 3–9× slower on the S23. MNN's default is 4.",
    "5. **Phones.** Only the OnePlus 15 is in the paper. The S23 is compared with the Xiaomi 14 (one chip generation "
    "newer); the OnePlus 8 Pro and Redmi 9A have no comparable paper phone. Phones with less RAM cannot load the "
    "larger 8-bit and 16-bit models (the harness refuses a run whose weights exceed 55% of RAM, or that swaps more "
    "than 512 MB).",
    "6. **Builds per phone.** llama.cpp is built with `-march=armv8.7a` (llama.cpp's Android recipe) for the S23 and "
    "OnePlus 15, `armv8.2-a+dotprod+fp16` for the OnePlus 8 Pro (no i8mm), and 32-bit `armv7-a` NEON for the Redmi "
    "9A (armeabi-v7a only). MNN models use `llmexport` defaults (quant block 64, no HQQ) from MNN 3.4.0; llama.cpp "
    "4-bit is Q4_0, because the pinned OpenCL backend has no Q4_K kernel.",
    "7. **Two builds.** Qwen3.5 is newer than the paper's llama.cpp `eadc418` and MNN `510ac8f`, so it runs on "
    "llama.cpp `9c2e0e4` and MNN `a8ec497` (text decoder only).",
]


if __name__ == "__main__":
    main()
