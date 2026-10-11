#!/usr/bin/env python3
"""
Write docs/RESULTS2.md and docs/figures/*.png: every Galaxy S23 result so far in the layout of the paper's Table 5
(arXiv 2607.05475), the same rows next to the paper's four phones, the SLM_Factory pool, and comparison figures.

  python results2.py            # reads the result folders listed in SOURCES; rerun as more results land

Colors: the dataviz skill's validated categorical palette (light surface), assigned in fixed order.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import report  # noqa: E402
import run_paper_table5 as rpt  # noqa: E402

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent / "docs"
FIGS = DOCS / "figures"
RESULTS = HERE / "results"
T5_DIR = RESULTS / "galaxy_s23_pinned_20261007_screenoff"
POOL_DIRS = [RESULTS / "slm_pool_s23_pinned", RESULTS / "slm_pool_s23_head"]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
COMBOS = [("cpu", "llama.cpp"), ("gpu", "llama.cpp"), ("cpu", "mnn"), ("gpu", "mnn")]
COMBO_LABEL = {("cpu", "llama.cpp"): "llama.cpp CPU", ("gpu", "llama.cpp"): "llama.cpp GPU",
               ("cpu", "mnn"): "MNN CPU", ("gpu", "mnn"): "MNN GPU"}
COMBO_COLOR = dict(zip(COMBOS, C))
T5_MODELS = ["qwen2.5-1.5b", "llama3.2-1b", "llama3.2-3b"]
LABEL = {"qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-7b": "Qwen2.5-7B", "llama3.2-1b": "Llama-3.2-1B",
         "llama3.2-3b": "Llama-3.2-3B", "smollm2-135m": "SmolLM2-135M", "smollm2-360m": "SmolLM2-360M",
         "gemma3-270m": "Gemma-3-270M", "qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B",
         "qwen3-4b-instruct-2507": "Qwen3-4B", "qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-2b": "Qwen3.5-2B",
         "qwen3.5-4b": "Qwen3.5-4B"}
POOL_ORDER = ["smollm2-135m", "gemma3-270m", "smollm2-360m", "qwen3-0.6b", "qwen3.5-0.8b", "qwen3-1.7b",
              "qwen3.5-2b", "qwen3-4b-instruct-2507", "qwen3.5-4b"]
W4 = {"llama.cpp": "Q4_0", "mnn": "Q4"}
QUANT_ORDER = {"Q4_0": 0, "Q4": 0, "Q8_0": 1, "Q8": 1, "F16": 2}
PAPER_DEV = {"xiaomi17": "Xiaomi 17", "oneplus15": "OnePlus 15", "xiaomi15": "Xiaomi 15", "xiaomi14": "Xiaomi 14"}


def load_dir(d: Path, sub: str = "configs", energy: bool = True) -> dict:
    out = {}
    for f in sorted((d / sub).glob("*.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        if r.get("status") == "ok" and c["framework"] == "mnn" and "mnn_decode_window" not in r:
            # older result: decode timed with llm_bench's decode_us; use the wall-clock window (see DEBUG.md)
            raw = d / "raw" / f"{c['id']}__kv.stderr.txt"
            if raw.exists():
                w = rpt.mnn_windows(rpt.parse_marks(raw.read_text(encoding="utf-8", errors="replace")), "main")
                for ph in ("prefill", "decode"):
                    r[ph].update(rpt.throughput(w, ph))
        if not energy:
            for ph in ("prefill", "decode"):
                (r.get(ph) or {}).pop("uj_per_token", None)
        out.setdefault((c["model"], c["backend"], c["framework"], c["quant"]), r)
    return out


def val(r, phase, kind):
    if not r or r.get("status") != "ok":
        return None
    p = r.get(phase) or {}
    if kind == "tps":
        return p.get("tps")
    return p.get("uj_per_token") if p.get("energy_valid") else None


def paper(model, backend, framework, phase, kind):
    row = report.REFERENCE["rows"].get(f"{model}/{backend}/{framework}") or {}
    return row.get(f"{phase}_{kind}") or []


def tps(v, s=None):
    return "–" if v is None else (f"{v:.1f}" if s is None else f"{v:.1f} ± {s:.1f}")


def uj(v):
    return "–" if v is None else report.fmt_uj(v)


def style(ax, ylabel=None, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGS / name, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return f"figures/{name}"


def fig_t5_throughput(t5):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    xs = range(len(COMBOS))
    for j, m in enumerate(T5_MODELS):
        for i, ph in enumerate(("prefill", "decode")):
            ax = axes[i][j]
            ours = [val(t5.get((m, b, f, W4[f])), ph, "tps") or 0 for b, f in COMBOS]
            ref = [paper(m, b, f, ph, "tps") for b, f in COMBOS]
            x14 = [r[3] if len(r) > 3 else 0 for r in ref]
            ax.bar([x - 0.19 for x in xs], ours, 0.34, color=C[0], label="Galaxy S23 (ours)")
            ax.bar([x + 0.19 for x in xs], x14, 0.34, color=C[1], label="Xiaomi 14 (paper)")
            for x, r in zip(xs, ref):
                if r:
                    ax.plot([x + 0.19, x + 0.19], [min(r), max(r)], color=INK2, linewidth=1.2, zorder=3)
                    ax.plot([x + 0.12, x + 0.26], [max(r)] * 2, color=INK2, linewidth=1.2, zorder=3)
                    ax.plot([x + 0.12, x + 0.26], [min(r)] * 2, color=INK2, linewidth=1.2, zorder=3)
            style(ax, ylabel=f"{ph.capitalize()} tokens/s" if j == 0 else None, title=LABEL[m] if i == 0 else None)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([COMBO_LABEL[c].replace(" ", "\n") for c in COMBOS], color=INK2, fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    h.append(plt.Line2D([0], [0], color=INK2, linewidth=1.2))
    l.append("range over the paper's four phones")
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "t5_throughput_s23_vs_paper.png")


def fig_ratio(t5):
    rows = [(m, b, f) for m in T5_MODELS for b, f in COMBOS]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.axvspan(0.75, 1.25, color="#f0efec", zorder=0)
    ax.axvline(1.0, color=MUTED, linewidth=1)
    ys = list(range(len(rows)))[::-1]
    for y, (m, b, f) in zip(ys, rows):
        pts = []
        for ph, col, mk in (("prefill", C[0], "o"), ("decode", C[1], "D")):
            o = val(t5.get((m, b, f, W4[f])), ph, "tps")
            r = paper(m, b, f, ph, "tps")
            if o and len(r) > 3:
                pts.append(o / r[3])
                ax.scatter(o / r[3], y, s=52, color=col, marker=mk, zorder=3, edgecolor=SURFACE, linewidth=1.2,
                           label={"prefill": "Prefill", "decode": "Decode"}[ph] if y == ys[0] else None)
        if len(pts) == 2:
            ax.plot(pts, [y, y], color=GRID, linewidth=2, zorder=1)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{LABEL[m]}  ·  {COMBO_LABEL[(b, f)]}" for m, b, f in rows], color=INK2, fontsize=8)
    ax.set_xlabel("Galaxy S23 throughput ÷ Xiaomi 14 throughput (paper)", color=INK2, fontsize=9)
    ax.set_xlim(0.2, 1.3)
    ax.text(1.0, len(rows) - 0.35, "±25% of the paper", color=MUTED, fontsize=8, ha="center")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return save(fig, "t5_ratio_s23_vs_xiaomi14.png")


def fig_energy(t5):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    xs = range(len(COMBOS))
    names = ["Galaxy S23 (ours, whole phone net of idle)", "Xiaomi 17 (paper, SoC)", "OnePlus 15 (paper, SoC)"]
    for j, m in enumerate(T5_MODELS):
        for i, ph in enumerate(("prefill", "decode")):
            ax = axes[i][j]
            series = [[(val(t5.get((m, b, f, W4[f])), ph, "uj") or 0) / 1e3 for b, f in COMBOS]]
            for k in (0, 1):
                series.append([(paper(m, b, f, ph, "uj")[k] if len(paper(m, b, f, ph, "uj")) > k else 0) / 1e3
                               for b, f in COMBOS])
            for k, s in enumerate(series):
                ax.bar([x + (k - 1) * 0.27 for x in xs], s, 0.25, color=C[k], label=names[k])
            style(ax, ylabel=f"{ph.capitalize()} mJ/token" if j == 0 else None, title=LABEL[m] if i == 0 else None)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([COMBO_LABEL[c].replace(" ", "\n") for c in COMBOS], color=INK2, fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "t5_energy_s23_vs_paper.png")


def fig_pool(pool, models):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ph in zip(axes, ("prefill", "decode")):
        for combo in COMBOS:
            b, f = combo
            # a model without this configuration breaks the line instead of being bridged
            ys = [val(pool.get((m, b, f, W4[f])), ph, "tps") or float("nan") for m in models]
            if all(y != y for y in ys):
                continue
            ax.plot(range(len(models)), ys, color=COMBO_COLOR[combo], linewidth=2, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label=COMBO_LABEL[combo])
        ax.set_yscale("log")
        style(ax, ylabel=f"{ph.capitalize()} tokens/s (log scale)", title=f"{ph.capitalize()}, 4-bit")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABEL[m] for m in models], color=INK2, fontsize=8, rotation=20, ha="right")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "pool_throughput_by_model.png")


def fig_pool_energy(pool, models):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ph in zip(axes, ("prefill", "decode")):
        for combo in COMBOS:
            b, f = combo
            ys = [(val(pool.get((m, b, f, W4[f])), ph, "uj") or float("nan")) / 1e3 for m in models]
            if all(y != y for y in ys):
                continue
            ax.plot(range(len(models)), ys, color=COMBO_COLOR[combo], linewidth=2, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label=COMBO_LABEL[combo])
        ax.set_yscale("log")
        style(ax, ylabel=f"{ph.capitalize()} mJ/token (log scale)", title=f"{ph.capitalize()} energy, 4-bit")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABEL[m] for m in models], color=INK2, fontsize=8, rotation=20, ha="right")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "pool_energy_by_model.png")


def fig_power(allres):
    """Energy per token against speed: energy/token = power / speed, so constant power is a straight line."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, ph in zip(axes, ("prefill", "decode")):
        pts = [((b, f), val(r, ph, "tps"), val(r, ph, "uj")) for (m, b, f, q), r in allres.items()]
        pts = [(c, t, e / 1e3) for c, t, e in pts if t and e]
        lo, hi = min(t for _, t, _ in pts) / 1.5, max(t for _, t, _ in pts) * 1.5
        for w in (2, 4, 8):
            ax.plot([lo, hi], [w * 1e3 / lo, w * 1e3 / hi], color=GRID, linewidth=1, zorder=1)
            ax.text(hi * 1.04, w * 1e3 / hi, f"{w} W", color=MUTED, fontsize=8, va="center")
        seen = set()
        for c, t, e in pts:
            ax.scatter(t, e, s=44, color=COMBO_COLOR[c], zorder=3, edgecolor=SURFACE, linewidth=1.1,
                       label=None if c in seen else COMBO_LABEL[c])
            seen.add(c)
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, ylabel="mJ/token (log scale)" if ph == "prefill" else None, title=ph.capitalize())
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_xlabel(f"{ph.capitalize()} tokens/s (log scale)", color=INK2, fontsize=9)
        ax.set_xlim(lo, hi * 1.5)
    order = [COMBO_LABEL[c] for c in COMBOS]
    hl = {}
    for ax in axes:
        hl.update(zip(ax.get_legend_handles_labels()[1], ax.get_legend_handles_labels()[0]))
    fig.legend([hl[k] for k in order if k in hl], [k for k in order if k in hl], loc="upper center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "energy_vs_speed.png")


def fig_bandwidth(allres):
    fig, ax = plt.subplots(figsize=(7.6, 5))
    lo, hi = 0.07, 6.0
    for gb in (5, 10, 20, 40):
        ax.plot([lo, hi], [gb / lo, gb / hi], color=GRID, linewidth=1, zorder=1)
        ax.text(hi * 1.03, gb / hi, f"{gb} GB/s", color=MUTED, fontsize=8, va="center")
    seen = set()
    for (m, b, f, q), r in allres.items():
        v = val(r, "decode", "tps")
        if not v:
            continue
        size = sum(r["model_file"]["files"].values()) / 1e9
        combo = (b, f)
        ax.scatter(size, v, s=44, color=COMBO_COLOR[combo], zorder=3, edgecolor=SURFACE, linewidth=1.1,
                   label=None if combo in seen else COMBO_LABEL[combo])
        seen.add(combo)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, ylabel="Decode tokens/s (log scale)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Model file size, GB (log scale)", color=INK2, fontsize=9)
    ax.set_xlim(lo, hi * 1.6)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    return save(fig, "decode_vs_model_size.png")


def main():
    t5 = load_dir(T5_DIR)
    pool = {}
    for d in POOL_DIRS:
        if (d / "configs").is_dir():
            pool.update(load_dir(d))
        for sub in sorted(p.name for p in d.glob("configs_noenergy_*")):
            for k, r in load_dir(d, sub, energy=False).items():
                if pool.get(k, {}).get("status") != "ok":
                    pool[k] = dict(r, energy_lost=True)  # throughput only; kept until the energy rerun replaces it
    models = [m for m in POOL_ORDER if any(k[0] == m for k in pool)]
    figs = {"t5": fig_t5_throughput(t5), "ratio": fig_ratio(t5), "energy": fig_energy(t5),
            "pool": fig_pool(pool, models), "bw": fig_bandwidth({**t5, **pool}),
            "pool_e": fig_pool_energy(pool, models), "power": fig_power({**t5, **pool})}

    L = ["# Galaxy S23 results in the paper's Table 5 layout", "",
         f"Generated {datetime.now().isoformat(timespec='minutes')} by `Benchmark-Harness/paper_table5/results2.py`. "
         "Reference: arXiv 2607.05475, Table 5: 256-token prefill and 256-token decode; throughput in tokens/s, energy "
         "in µJ/token.", "",
         "- **Phone:** Galaxy S23 (Snapdragon 8 Gen 2, Adreno 740, 8 GB), unrooted, on battery, airplane mode with Wi-Fi "
         "kept for adb, screen off, every run started at ≤ 28 °C.",
         "- **Runs:** 1 warm-up + 3 recorded repetitions (mean ± std); llama.cpp CPU with 4 threads; MNN decode timed "
         "on the wall clock.",
         "- **S23 energy** is whole-phone battery energy net of idle; the paper's is SoC-only, reported for the "
         "Xiaomi 17 and OnePlus 15. They are not the same quantity; see [RESULTS.md](RESULTS.md) for the protocol "
         "differences and [DEBUG.md](DEBUG.md) for the audit.", ""]

    # ---- Table 5 models next to the paper ----
    L += ["## 1. The paper's models: S23 next to the paper's four phones", "", "### Throughput (tokens/s), 4-bit", "",
          "| Model | Backend | Framework | Prefill S23 | X17 | OP15 | X15 | X14 | Decode S23 | X17 | OP15 | X15 | X14 |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for m in T5_MODELS:
        first = True
        for b, f in [("cpu", "llama.cpp"), ("cpu", "mnn"), ("gpu", "llama.cpp"), ("gpu", "mnn")]:
            r = t5.get((m, b, f, W4[f]))
            cells = []
            for ph in ("prefill", "decode"):
                p = (r or {}).get(ph) or {}
                cells.append(tps(val(r, ph, "tps"), p.get("tps_std")))
                ref = paper(m, b, f, ph, "tps")
                cells += [tps(ref[i]) if len(ref) > i else "–" for i in range(4)]
            L.append(f"| {LABEL[m] if first else ''} | {b.upper()} | {report.FRAMEWORK_LABEL[f]} | " + " | ".join(cells) + " |")
            first = False
    L += ["", "X17 = Xiaomi 17, OP15 = OnePlus 15, X15 = Xiaomi 15, X14 = Xiaomi 14 (paper values). Qwen2.5-7B is not "
          "shown: it does not fit in the S23's 8 GB of RAM.", "",
          "### Energy (µJ/token), 4-bit", "",
          "| Model | Backend | Framework | Prefill S23 | X17 | OP15 | Decode S23 | X17 | OP15 |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|"]
    for m in T5_MODELS:
        first = True
        for b, f in [("cpu", "llama.cpp"), ("cpu", "mnn"), ("gpu", "llama.cpp"), ("gpu", "mnn")]:
            r = t5.get((m, b, f, W4[f]))
            cells = []
            for ph in ("prefill", "decode"):
                ref = paper(m, b, f, ph, "uj")
                cells += [uj(val(r, ph, "uj"))] + [uj(ref[i]) if len(ref) > i else "–" for i in range(2)]
            L.append(f"| {LABEL[m] if first else ''} | {b.upper()} | {report.FRAMEWORK_LABEL[f]} | " + " | ".join(cells) + " |")
            first = False

    # ---- comparison numbers ----
    ratios = {"prefill": {}, "decode": {}}
    for m in T5_MODELS:
        for b, f in COMBOS:
            for ph in ratios:
                o, ref = val(t5.get((m, b, f, W4[f])), ph, "tps"), paper(m, b, f, ph, "tps")
                if o and len(ref) > 3:
                    ratios[ph][(m, b, f)] = o / ref[3]
    def med(ph, b=None, f=None):
        v = [x for (m, bb, ff), x in ratios[ph].items() if (b is None or bb == b) and (f is None or ff == f)]
        return statistics.median(v) if v else float("nan")
    inband = sum(0.75 <= x <= 1.25 for ph in ratios for x in ratios[ph].values())
    total = sum(len(ratios[ph]) for ph in ratios)
    e_ratio = []
    for m in T5_MODELS:
        for b, f in COMBOS:
            for ph in ("prefill", "decode"):
                o, ref = val(t5.get((m, b, f, W4[f])), ph, "uj"), paper(m, b, f, ph, "uj")
                if o and len(ref) >= 2:
                    e_ratio.append(o / statistics.mean(ref[:2]))
    p_ratio, s_ratio = [], []
    for m in T5_MODELS:
        for b, f in COMBOS:
            for ph in ("prefill", "decode"):
                r = t5.get((m, b, f, W4[f]))
                o, ot = val(r, ph, "uj"), val(r, ph, "tps")
                ru, rt = paper(m, b, f, ph, "uj"), paper(m, b, f, ph, "tps")
                if o and ot and len(ru) >= 2:  # power = energy per token x tokens per second
                    p_ratio.append(o * ot / statistics.mean(ru[i] * rt[i] for i in (0, 1)))
                    s_ratio.append(statistics.mean(rt[:2]) / ot)
    L += ["", "## 2. How the S23 compares with the paper", "",
          f"![Throughput, S23 vs Xiaomi 14]({figs['t5']})", "",
          "*Prefill (top) and decode (bottom) throughput per framework and backend. Blue: Galaxy S23. Orange: the "
          "paper's Xiaomi 14, the closest phone (one chip generation newer). The dark bracket is the range over the "
          "paper's four phones.*", "",
          f"![Ratio S23 / Xiaomi 14]({figs['ratio']})", "",
          "*Each row is one configuration; a point at 1.0 means the S23 matches the Xiaomi 14. The shaded band is ±25%.*",
          "",
          f"- **{inband} of {total} throughput cells are within ±25% of the Xiaomi 14.**",
          f"- **CPU is close.** llama.cpp CPU: median ratio {med('prefill', 'cpu', 'llama.cpp'):.2f} for prefill and "
          f"{med('decode', 'cpu', 'llama.cpp'):.2f} for decode. MNN CPU: {med('prefill', 'cpu', 'mnn'):.2f} and "
          f"{med('decode', 'cpu', 'mnn'):.2f}; MNN's CPU prefill trails more than one chip generation explains.",
          f"- **llama.cpp GPU prefill is the largest gap** (median ratio {med('prefill', 'gpu', 'llama.cpp'):.2f}): on "
          "the S23's Adreno 740 it is no faster than the CPU, while the paper's GPUs are 2–3× faster than their CPUs. "
          f"MNN GPU prefill holds up (median {med('prefill', 'gpu', 'mnn'):.2f}).",
          f"- **GPU decode is slower than CPU decode on the S23 and on the paper's Xiaomi 14 alike.** S23 GPU decode "
          f"ratios: llama.cpp {med('decode', 'gpu', 'llama.cpp'):.2f}, MNN {med('decode', 'gpu', 'mnn'):.2f} (the "
          "paper's MNN version does not batch OpenCL commands during decode, which costs ~3–4 ms per layer per token).",
          "", f"![Energy, S23 vs paper]({figs['energy']})", "",
          "*Energy per token, in mJ/token (1 mJ = 1000 µJ). Blue: S23, whole phone net of idle. Orange and aqua: the "
          "paper's SoC-only values.*", "",
          f"- **S23 energy per token is {min(e_ratio):.1f}–{max(e_ratio):.1f}× the paper's** (median "
          f"{statistics.median(e_ratio):.1f}× the mean of its two phones). Two things multiply: those "
          f"phones are a median {statistics.median(s_ratio):.1f}× faster, and the S23's measured power is a median "
          f"{statistics.median(p_ratio):.1f}× theirs (energy per token × tokens per second). The S23 figure is "
          "whole-phone power, which includes memory, storage and regulator losses that SoC counters leave out.",
          "- **The patterns match the paper:** decode costs several times more per token than prefill; GPU prefill is "
          "cheaper per token than CPU prefill; GPU decode saves nothing over CPU decode.",
          "- Absolute S23 energy carries roughly ±10–20% uncertainty from the battery gauge; comparisons between S23 "
          "configurations do not."]

    # ---- pool ----
    L += ["", "## 3. SLM_Factory model pool on the S23", "",
          "Pool: `config/android_pool.py` on SLM_Factory `main`. Qwen3, SmolLM2 and Gemma 3 run on the paper's "
          "llama.cpp `eadc418` / MNN `510ac8f`; Qwen3.5 needs newer versions (llama.cpp `9c2e0e4`, MNN `a8ec497`, "
          "text decoder only). Quantizations: llama.cpp Q4_0 / Q8_0 / F16, MNN 4 / 8 / 16-bit.", "",
          "| Model | Backend | Framework | Quant. | Prefill t/s | Prefill µJ/token | Decode t/s | Decode µJ/token |",
          "|---|---|---|---|--:|--:|--:|--:|"]
    failed, lost, pending, odd = [], 0, 0, []
    for m in models:
        first = True
        keys = sorted((k for k in pool if k[0] == m),
                      key=lambda k: (QUANT_ORDER.get(k[3], 9), report.BACKEND_ORDER[k[1]], report.FRAMEWORK_ORDER[k[2]]))
        for k in keys:
            r = pool[k]
            if r.get("status") != "ok":
                failed.append(r)
                row = ["FAIL"] * 4
            else:
                row = [tps(val(r, "prefill", "tps"), r["prefill"].get("tps_std")), uj(val(r, "prefill", "uj")),
                       tps(val(r, "decode", "tps"), r["decode"].get("tps_std")), uj(val(r, "decode", "uj"))]
                if val(r, "prefill", "uj") is None:
                    lost += bool(r.get("energy_lost"))
                    pending += not r.get("energy_lost")
                if val(r, "prefill", "tps") < val(r, "decode", "tps"):
                    odd.append(r["config"]["id"])
            L.append(f"| {LABEL.get(m, m) if first else ''} | {k[1].upper()} | {report.FRAMEWORK_LABEL[k[2]]} | {k[3]} | "
                     + " | ".join(row) + " |")
            first = False
    done = sum(r.get("status") == "ok" for r in pool.values())
    L += ["", f"{done} configurations measured, {len(failed)} could not run. `–` in an energy column = throughput "
          f"measured, energy not yet: {lost} rows lost their battery trace when the phone reset and are queued to "
          "rerun" + (f"; {pending} rows of the latest session get theirs from its battery trace, which is still on the phone" if pending
                     else "") + ". Models and configurations still to run are not listed yet.", ""]
    if odd:
        L += ["**Treat with caution:** " + ", ".join(f"`{i}`" for i in odd) + " — prefill is slower than decode. "
              "All are llama.cpp GPU rows with Q8_0 or F16 weights on the paper's llama.cpp `eadc418`; the same weight "
              "types on the newer llama.cpp (the Qwen3.5 rows) do not show it, so this looks like a slow OpenCL prefill "
              "path in that version for these models. Not yet confirmed; do not cite these prefill numbers.", ""]
    L += [
          f"![Pool throughput by model]({figs['pool']})", "",
          "*4-bit throughput by model, smallest to largest (log scale). One line per framework and backend; a gap means "
          "that configuration has not run yet. Qwen3.5 uses newer llama.cpp and MNN versions than the smaller models, "
          "so its points are not strictly comparable with theirs (MNN GPU decode in particular is much faster in the "
          "newer MNN).*", "",
          f"![Decode vs model size]({figs['bw']})", "",
          "*Every measured configuration: decode speed against model file size. Decode reads all weights once per "
          "token, so points on a grey line share the same effective memory bandwidth. CPU runs follow the "
          "20–40 GB/s lines; GPU runs fall below them, most of all for small models, where per-token overhead "
          "dominates.*", ""]
    # ---- energy report ----
    allres = {**t5, **pool}
    watts = {ph: {c: [] for c in COMBOS} for ph in ("prefill", "decode")}
    for (m, b, f, q), r in allres.items():
        for ph in watts:
            t, e = val(r, ph, "tps"), val(r, ph, "uj")
            if t and e:
                watts[ph][(b, f)].append(e * t / 1e6)
    n_energy = sum(len(v) for v in watts["decode"].values())
    def rng(v):
        return f"{min(v):.1f}–{max(v):.1f} (median {statistics.median(v):.1f})" if v else "–"
    def cheapest(ph, m):
        c = [(val(pool.get((m, b, f, W4[f])), ph, "uj"), COMBO_LABEL[(b, f)]) for b, f in COMBOS]
        c = sorted(x for x in c if x[0])
        return c[0] if len(c) == 4 else None
    L += ["## 4. Energy report", "",
          f"{n_energy} configurations have valid energy so far (the Table 5 models plus the pool rows above). All S23 "
          "energy is whole-phone battery energy during the timed work minus the idle power measured just before it.",
          "", f"![Pool energy by model]({figs['pool_e']})", "",
          "*4-bit energy per token by model, smallest to largest (log scale). A gap means that configuration has no "
          "energy yet.*", "",
          f"![Energy vs speed]({figs['power']})", "",
          "*Every configuration with energy: energy per token against speed. Energy per token = power ÷ speed, so "
          "points on a grey line draw the same net power; lower right is better.*", "",
          "### Net power while running (W above idle)", "",
          "| Backend | Framework | Prefill W | Decode W | Configurations |", "|---|---|--:|--:|--:|"]
    for c in [("cpu", "llama.cpp"), ("cpu", "mnn"), ("gpu", "llama.cpp"), ("gpu", "mnn")]:
        L.append(f"| {c[0].upper()} | {report.FRAMEWORK_LABEL[c[1]]} | {rng(watts['prefill'][c])} | "
                 f"{rng(watts['decode'][c])} | {len(watts['decode'][c])} |")
    best = [(LABEL[m], cheapest("prefill", m), cheapest("decode", m)) for m in models]
    best = [x for x in best if x[1] and x[2]]
    L += ["", "- **Energy per token is set by speed far more than by power.** Net power stays between about 1 "
          "and 10 W across every configuration, while speed spans more than two orders of magnitude, so the slow "
          "configurations are the expensive ones.",
          "- **Decode costs several times more per token than prefill** (the one exception is the flagged "
          "SmolLM2-135M Q8_0 llama.cpp GPU row): prefill processes 256 tokens in one batched pass, decode makes one "
          "pass per token.",
          "- **MNN GPU prefill is the cheapest way to process a prompt:** it is faster than the CPU and draws less "
          "power. llama.cpp GPU prefill also draws less power but is slower than the CPU on this phone, so its "
          "energy per token is about the same as the CPU's for the smallest models and lower for the 1B+ models.",
          "- **GPU decode draws less power but is so slow** that its energy per token is no better than the CPU's, "
          "and for the smallest models it is the most expensive option."]
    if best:
        L += ["- **Cheapest 4-bit configuration per model** (models with all four measured): "
              + "; ".join(f"{n}: prefill {p[1]} ({uj(p[0])} µJ/token), decode {d[1]} ({uj(d[0])} µJ/token)"
                          for n, p, d in best) + "."]
    L += ["- Energy for the current session's rows comes from a snapshot of the battery trace taken while the "
          "session is still running; the harness recomputes nothing for them at session end, and the values do not "
          "change.", ""]
    if failed:
        L += ["### Configurations that could not run", ""]
        L += [f"- `{r['config']['id']}`: {r.get('error')}" for r in sorted(failed, key=lambda r: r["config"]["id"])]
    (DOCS / "RESULTS2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {DOCS / 'RESULTS2.md'} and {len(figs)} figures in {FIGS}")


if __name__ == "__main__":
    main()
