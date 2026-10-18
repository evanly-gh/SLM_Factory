"""Graphs of every Table-5-style result per phone (prefill, decode, energy, memory) plus paper comparisons.

Reads results/slm_pool_<phone>_{pinned,head}/configs/*.json and ../../paper_reference.json; writes PNGs and
plotted_data.csv next to this file.   Usage: python make_graphs.py
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)                       # results/
PT5 = os.path.dirname(RES)                        # paper_table5/

# ---- design tokens (reference palette slots 1-3, validated all-pairs, light surface) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BITS_COLOR = {4: BLUE, 8: ORANGE, 16: AQUA}      # color = weight precision
ENGINE_STYLE = {                                  # engine = marker + line style (secondary encoding)
    "llama.cpp": dict(marker="o", linestyle="-"),
    "mnn": dict(marker="s", linestyle="--"),
}
ENGINE_LABEL = {"llama.cpp": "llama.cpp", "mnn": "MNN"}

PHONES = {  # key -> (label, result-dir stems, GPU available, RAM GiB, paper column, paper label)
    "redmi9a": ("Redmi 9A", ["redmi9a_pinned", "redmi9a_head"], False, 3.72, None, None),
    "oneplus8pro": ("OnePlus 8 Pro", ["oneplus8pro_pinned", "oneplus8pro_head"], True, 11.24, 3, "Xiaomi 14 (proxy)"),
    "oneplus15": ("OnePlus 15", ["oneplus15_pinned", "oneplus15_head"], True, 14.76, 1, "OnePlus 15"),
}
RAM_LIMIT_FRAC = 0.55

# parameter counts (billions) and display order (by size; ties: pinned build first)
MODELS = [  # key, label, params_B, head-build?
    ("smollm2-135m", "SmolLM2-135M", 0.135, False),
    ("gemma3-270m", "Gemma3-270M", 0.27, False),
    ("smollm2-360m", "SmolLM2-360M", 0.36, False),
    ("qwen3-0.6b", "Qwen3-0.6B", 0.6, False),
    ("qwen3.5-0.8b", "Qwen3.5-0.8B†", 0.8, True),
    ("qwen3-1.7b", "Qwen3-1.7B", 1.7, False),
    ("qwen3.5-2b", "Qwen3.5-2B†", 2.0, True),
    ("qwen3-4b-instruct-2507", "Qwen3-4B", 4.0, False),
    ("qwen3.5-4b", "Qwen3.5-4B†", 4.3, True),  # 4.3 = x offset so it does not overlap Qwen3-4B on the log-size axis
]
MODEL_IDX = {m[0]: i for i, m in enumerate(MODELS)}
SERIES = [(fw, bits) for fw in ("llama.cpp", "mnn") for bits in (4, 8, 16)]
PAPER_PARAMS = {"llama3.2-1b": 1.24, "qwen2.5-1.5b": 1.54, "llama3.2-3b": 3.21, "qwen2.5-7b": 7.62}


def load():
    """rows: phone -> list of dict(model, fw, bits, backend, status, prefill, decode, pe, de, gib, reason)"""
    data = defaultdict(list)
    for phone, (_, stems, *_rest) in PHONES.items():
        for stem in stems:
            for f in glob.glob(os.path.join(RES, f"slm_pool_{stem}", "configs", "*.json")):
                j = json.load(open(f))
                c, mf = j["config"], j.get("model_file") or {}
                row = dict(model=c["model"], fw=c["framework"], bits=c["bits"], backend=c["backend"],
                           quant=c["quant"], status=j.get("status"), build=stem.rsplit("_", 1)[1],
                           gib=None, reason=str(j.get("error") or j.get("reason") or ""))
                if mf.get("files"):
                    row["gib"] = sum(mf["files"].values()) / 2**30
                if j.get("status") == "ok":
                    p, d = j["prefill"], j["decode"]
                    row.update(prefill=p.get("tps"), decode=d.get("tps"))
                    row["pe"] = p["uj_per_token"] / 1000 if p.get("energy_valid") and p.get("uj_per_token") else None
                    row["de"] = d["uj_per_token"] / 1000 if d.get("energy_valid") and d.get("uj_per_token") else None
                data[phone].append(row)
    return data


def style_ax(ax, ylabel=None, log=True):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(axis="both", which="both", colors=INK2, labelsize=8)
    if log:
        ax.set_yscale("log")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def series_points(rows, backend, fw, bits, key):
    pts = {}
    for r in rows:
        if r["backend"] == backend and r["fw"] == fw and r["bits"] == bits and r["model"] in MODEL_IDX \
                and r.get(key) is not None:
            pts[MODEL_IDX[r["model"]]] = r[key]
    return sorted(pts.items())


def dodge(fw, bits):
    """Horizontal offset inside a model's slot so the six series never sit on top of each other."""
    return (SERIES.index((fw, bits)) - 2.5) * 0.115


def draw_lines(ax, pts_xy, color, style, label=None, hollow=False, ms=5.5, offset=0.0):
    """Connect only consecutive x positions; a gap (missing cell) breaks the line."""
    runs, cur = [], []
    for x, y in pts_xy:
        if cur and x != cur[-1][0] + 1:
            runs.append(cur)
            cur = []
        cur.append((x, y))
    if cur:
        runs.append(cur)
    for k, run in enumerate(runs):
        xs, ys = zip(*run)
        xs = [x + offset for x in xs]
        ax.plot(xs, ys, color=color, linewidth=1.8, markersize=ms, markeredgewidth=1.4,
                markeredgecolor=color, markerfacecolor=SURFACE if hollow else color,
                label=label if k == 0 else None, **style)


def model_axis(ax):
    for i in range(len(MODELS)):          # one band per model; alternate shading
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#f1f0ec", zorder=0, linewidth=0)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[1] for m in MODELS], rotation=40, ha="right", fontsize=8, color=INK2)
    ax.set_xlim(-0.5, len(MODELS) - 0.5)


def series_legend(fig, extra=None):
    handles = [Line2D([], [], color=BITS_COLOR[b], marker="s", linestyle="", markersize=7, label=f"{b}-bit" if b < 16 else "16-bit (F16)")
               for b in (4, 8, 16)]
    handles += [Line2D([], [], color=INK2, linewidth=1.8, label=ENGINE_LABEL[fw], **ENGINE_STYLE[fw])
                for fw in ("llama.cpp", "mnn")]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))


def title(fig, text, sub):
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.925, sub, ha="left", fontsize=8.5, color=INK2)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


CSV = []


def metric_figure(phone, rows, key, ylabel, fname, name_metric, scale=1.0, note=""):
    label, _, has_gpu, *_ = PHONES[phone]
    backends = ["cpu", "gpu"] if has_gpu else ["cpu"]
    fig, axes = plt.subplots(1, len(backends), figsize=(6.4 * len(backends) + 0.4, 5.3), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, be in zip(axes[0], backends):
        style_ax(ax, ylabel if ax is axes[0][0] else None)
        any_pts = False
        for fw, bits in SERIES:
            pts = [(x, y * scale) for x, y in series_points(rows, be, fw, bits, key)]
            if not pts:
                continue
            any_pts = True
            draw_lines(ax, pts, BITS_COLOR[bits], ENGINE_STYLE[fw], offset=dodge(fw, bits))
            for x, y in pts:
                CSV.append([label, name_metric, be.upper(), MODELS[x][1].rstrip("†"), ENGINE_LABEL[fw],
                            f"{bits}-bit", round(y, 3)])
        ax.set_title(be.upper(), loc="left", fontsize=10.5, color=INK, fontweight="bold")
        model_axis(ax)
        if not any_pts:
            ax.text(0.5, 0.5, "no valid measurements", transform=ax.transAxes, ha="center", color=INK2)
    title(fig, f"{label}: {name_metric}", f"4 threads, 256-token prompt, 256-token decode. † = head llama.cpp/MNN build (Qwen3.5). "
          f"Lines break where a cell failed or did not fit. {note}")
    series_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    save(fig, fname)


def energy_figure(phone, rows, fname):
    label, _, has_gpu, *_ = PHONES[phone]
    backends = ["cpu", "gpu"] if has_gpu else ["cpu"]
    fig, axes = plt.subplots(2, len(backends), figsize=(6.4 * len(backends) + 0.4, 8.2), sharey="row", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r_i, (key, nm) in enumerate((("pe", "prefill"), ("de", "decode"))):
        for ax, be in zip(axes[r_i], backends):
            style_ax(ax, f"mJ per token ({nm})" if ax is axes[r_i][0] else None)
            for fw, bits in SERIES:
                pts = series_points(rows, be, fw, bits, key)
                if not pts:
                    continue
                draw_lines(ax, pts, BITS_COLOR[bits], ENGINE_STYLE[fw], offset=dodge(fw, bits))
                for x, y in pts:
                    CSV.append([label, f"energy {nm} (mJ/token)", be.upper(), MODELS[x][1].rstrip("†"),
                                ENGINE_LABEL[fw], f"{bits}-bit", round(y, 2)])
            ax.set_title(f"{be.upper()} — {nm}", loc="left", fontsize=10.5, color=INK, fontweight="bold")
            model_axis(ax)
    title(fig, f"{label}: energy per token", "Whole-device battery energy net of idle (mJ/token); only cells the harness marked valid. "
          "† = head build.")
    series_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(fig, fname)


def memory_figure(phone, rows, fname):
    label, _, has_gpu, ram, *_ = PHONES[phone]
    limit = ram * RAM_LIMIT_FRAC
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax, "weight file size (GiB, log scale)")
    # size and outcome per (model, fw, bits); sizes are identical on every phone
    size, outcome = {}, defaultdict(set)
    for r in rows:
        if r["model"] in MODEL_IDX and r["gib"]:
            size[(r["model"], r["fw"], r["bits"])] = r["gib"]
            outcome[(r["model"], r["fw"], r["bits"])].add(r["status"] == "ok")
    for fw, bits in SERIES:
        pts = sorted((MODEL_IDX[m], g) for (m, f, b), g in size.items() if f == fw and b == bits)
        draw_lines(ax, pts, BITS_COLOR[bits], ENGINE_STYLE[fw], offset=dodge(fw, bits))
        for x, g in pts:
            key = (MODELS[x][0], fw, bits)
            ran = any(outcome[key])
            CSV.append([label, "weight size (GiB)", "", MODELS[x][1].rstrip("†"), ENGINE_LABEL[fw], f"{bits}-bit",
                        round(g, 3)])
            if not ran:  # no backend of this format produced a result on this phone -> mark with an open ring
                ax.plot([x + dodge(fw, bits)], [g], marker="o", markersize=11, markerfacecolor="none", markeredgecolor="#c0392b", markeredgewidth=1.6)
    ax.axhline(limit, color="#c0392b", linewidth=1.2, linestyle=":")
    ax.text(len(MODELS) - 0.55, limit * 1.06, f"RAM guard: 55% of {ram:.2f} GiB = {limit:.2f} GiB", ha="right", fontsize=8,
            color="#c0392b")
    model_axis(ax)
    title(fig, f"{label}: model weights vs. memory limit",
          "File size per model/format (memory use was not logged per run). Red ring = no result on this phone. † = head build.")
    ring = Line2D([], [], marker="o", markersize=10, markerfacecolor="none", markeredgecolor="#c0392b", linestyle="", label="no result on this phone")
    series_legend(fig, [ring])
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    save(fig, fname)


# ------------------------------------------------------------------ paper comparison
PAPER_LABEL = {"llama3.2-1b": "Llama-3.2-1B (paper)", "qwen2.5-1.5b": "Qwen2.5-1.5B (paper)",
               "llama3.2-3b": "Llama-3.2-3B (paper)", "qwen2.5-7b": "Qwen2.5-7B (paper)"}
# x axis of the comparison charts: every model used, ours and the paper's, ordered by size (ties keep our order)
COMPARE_SLOTS = sorted([("ours", m[0], m[1], m[2]) for m in MODELS] +
                       [("paper", k, PAPER_LABEL[k], PAPER_PARAMS[k]) for k in PAPER_PARAMS], key=lambda t: t[3])
SLOT_IDX = {(t[0], t[1]): i for i, t in enumerate(COMPARE_SLOTS)}


def paper_points(paper, metric, backend, fw, col):
    out = []
    for k, v in paper["rows"].items():
        m, be, f = k.split("/")
        if be == backend and f == fw and v.get(metric) and len(v[metric]) > col and v[metric][col] is not None:
            out.append((SLOT_IDX[("paper", m)], v[metric][col] / (1000.0 if "uj" in metric else 1.0)))
    return sorted(out)


def ours_w4(rows, backend, fw, key):
    pts = {}
    for r in rows:
        if r["backend"] == backend and r["fw"] == fw and r["bits"] == 4 and r["model"] in MODEL_IDX and r.get(key) is not None:
            pts[SLOT_IDX[("ours", r["model"])]] = r[key]
    return sorted(pts.items())


def compare_axis(ax):
    for i, (src, _, label, _) in enumerate(COMPARE_SLOTS):
        if src == "paper":                       # paper models get a warm band, ours stay unshaded
            ax.axvspan(i - 0.5, i + 0.5, color="#f6ece3", zorder=0, linewidth=0)
    ax.set_xticks(range(len(COMPARE_SLOTS)))
    labels = ax.set_xticklabels([t[2] for t in COMPARE_SLOTS], rotation=40, ha="right", fontsize=8, color=INK2)
    for lab, t in zip(labels, COMPARE_SLOTS):
        if t[0] == "paper":
            lab.set_style("italic")
            lab.set_color("#9a4a1f")
    ax.set_xlim(-0.5, len(COMPARE_SLOTS) - 0.5)


def compare_figure(data, paper, metric_key, paper_key, ylabel, fname, name, phones):
    fig, axes = plt.subplots(len(phones), 2, figsize=(13.4, 5.6 * len(phones)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r_i, phone in enumerate(phones):
        plabel, _, _, _, col, pcol_label = PHONES[phone]
        for ax, be in zip(axes[r_i], ("cpu", "gpu")):
            style_ax(ax, ylabel if ax is axes[r_i][0] else None)
            for fw, color in (("llama.cpp", BLUE), ("mnn", ORANGE)):
                o = ours_w4(data[phone], be, fw, metric_key)
                if o:
                    xs, ys = zip(*o)
                    ax.plot(xs, ys, color=color, linewidth=1.8, markersize=5.5, **ENGINE_STYLE[fw])
                    for x, y in o:
                        CSV.append([f"{plabel} vs paper", name, be.upper(), COMPARE_SLOTS[x][2].rstrip("\u2020"), ENGINE_LABEL[fw], "4-bit (ours)", round(y, 2)])
                pp = paper_points(paper, paper_key, be, fw, col)
                if pp:
                    xs, ys = zip(*pp)
                    ax.plot(xs, ys, color=color, linewidth=1.8, markersize=7, markerfacecolor=SURFACE, markeredgewidth=1.6,
                            marker="D", linestyle=":")
                    for x, y in pp:
                        CSV.append([f"{plabel} vs paper", name, be.upper(), COMPARE_SLOTS[x][2], ENGINE_LABEL[fw], "4-bit (paper)", round(y, 2)])
            ax.set_title(f"{plabel} {be.upper()}  vs  paper {pcol_label or ''}".replace("  vs  paper None", ""), loc="left",
                         fontsize=10.5, color=INK, fontweight="bold")
            compare_axis(ax)
    handles = [Line2D([], [], color=BLUE, linewidth=1.8, label="llama.cpp (Q4_0)", **ENGINE_STYLE["llama.cpp"]),
               Line2D([], [], color=ORANGE, linewidth=1.8, label="MNN (4-bit)", **ENGINE_STYLE["mnn"]),
               Line2D([], [], color=INK2, marker="o", linestyle="-", label="our models (solid, unshaded)"),
               Line2D([], [], color=INK2, marker="D", markerfacecolor=SURFACE, linestyle=":", label="paper models (hollow, dotted, shaded)")]
    title(fig, f"{name}: our runs vs. the paper (w4)",
          "Models are ordered by size; the paper used different models, so compare the trend, not model-for-model. \u2020 = head build.")
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    save(fig, fname)


def main():
    data = load()
    paper = json.load(open(os.path.join(PT5, "paper_reference.json")))
    for phone in ("redmi9a", "oneplus8pro", "oneplus15"):
        rows = data[phone]
        metric_figure(phone, rows, "prefill", "prefill tokens/s (log scale)", f"{phone}_prefill_tps.png", "prefill throughput")
        metric_figure(phone, rows, "decode", "decode tokens/s (log scale)", f"{phone}_decode_tps.png", "decode throughput")
        if any(r.get("pe") or r.get("de") for r in rows):
            energy_figure(phone, rows, f"{phone}_energy.png")
        else:
            print("no valid energy data for", phone)
        memory_figure(phone, rows, f"{phone}_memory.png")
    # energy values in the paper are uJ/token; ours are mJ/token -> paper_points divides uJ by 1000 as well
    compare_figure(data, paper, "prefill", "prefill_tps", "prefill tokens/s", "compare_paper_prefill_tps.png", "Prefill throughput",
                   ["oneplus15", "oneplus8pro"])
    compare_figure(data, paper, "decode", "decode_tps", "decode tokens/s", "compare_paper_decode_tps.png", "Decode throughput",
                   ["oneplus15", "oneplus8pro"])
    compare_figure(data, paper, "pe", "prefill_uj", "mJ per token (prefill)", "compare_paper_energy_prefill.png",
                   "Prefill energy", ["oneplus15"])
    compare_figure(data, paper, "de", "decode_uj", "mJ per token (decode)", "compare_paper_energy_decode.png",
                   "Decode energy", ["oneplus15"])
    with open(os.path.join(HERE, "plotted_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phone", "metric", "backend", "model", "engine", "precision", "value"])
        w.writerows(CSV)
    print("rows in csv:", len(CSV))


if __name__ == "__main__":
    main()
