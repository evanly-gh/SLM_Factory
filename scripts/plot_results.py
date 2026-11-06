"""Figures for docs/RESULTS.md, one per numeric table.

Every number is parsed from the tables in `docs/RESULTS.md` itself (located by a unique anchor
string), so the figures follow the document: edit a table, rerun this script. PNGs are written to
`docs/img/results/`.

Palette: the dataviz reference palette, in its validated categorical order (blue / orange / aqua /
yellow) and an ordinal blue ramp for quantization precision. Roles are fixed across figures:
hyperparameters / mining / synthesis are blue / orange / aqua; zero-shot / first fine-tune / best
are gray / orange / blue; full data / scarce + synthesis / scarce without synthesis are blue / aqua
/ gray; ladder / one shot / sweep are blue / orange / aqua.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "RESULTS.md"
FIGS = ROOT / "docs" / "img" / "results"
LINES = DOC.read_text().splitlines()

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e4e3de"
NEUTRAL = "#b9b8b1"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
LIGHT_BLUE = "#86b6ef"
PRECISION = {"Q4_K_M": "#86b6ef", "Q8_0": "#2a78d6", "bf16": "#104281"}

HP, MINE, SYNTH = BLUE, ORANGE, AQUA
ZERO, FIRST, BEST = NEUTRAL, ORANGE, BLUE
FULL, SCARCE, NOSYNTH = BLUE, AQUA, NEUTRAL
LADDER, SINGLE, SWEEP = BLUE, ORANGE, AQUA

TASK_ORDER = ["clinc150", "sms_spam", "calendar_json", "xlam_bfcl", "ner_bc5cdr",
              "goemotions", "gec_bea19", "multiconer", "dialogsum", "topv2"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "legend.frameon": False,
    "legend.fontsize": 9, "font.size": 10, "text.color": INK,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


# ───────────────────────────────────────────────────────────────────────────────────── parsing
def clean(cell: str) -> str:
    return re.sub(r"\*\*|`|★|✓", "", cell).strip()


def table(anchor: str) -> tuple[list[str], list[list[str]]]:
    """The first markdown table at or after the one line containing ``anchor``."""
    hits = [k for k, line in enumerate(LINES) if anchor in line]
    if len(hits) != 1:
        raise SystemExit(f"anchor {anchor!r} matched {len(hits)} lines")
    i = hits[0]
    while not LINES[i].startswith("|"):
        i += 1
    rows = []
    while i < len(LINES) and LINES[i].startswith("|"):
        rows.append([clean(c) for c in LINES[i].strip().strip("|").split("|")])
        i += 1
    return rows[0], [r for r in rows[2:] if r[0].lower() != "total" and r[1].lower() != "total"]


def col(header: list[str], name: str) -> int:
    return next(k for k, h in enumerate(header) if name in h)


def nums(cell: str) -> list[float]:
    out = []
    for sign, digits in re.findall(r"([+\-−]?)(\d[\d,]*(?:\.\d+)?)", cell):
        v = float(digits.replace(",", ""))
        out.append(-v if sign in ("-", "−") else v)
    return out


def num(cell: str) -> float | None:
    if cell.strip() in ("", "—", "-"):
        return None
    found = nums(cell)
    return found[0] if found else None


def score(cell: str) -> float | None:
    """The first four-decimal score in a cell that also names jobs or models."""
    m = re.search(r"[+\-−]?\d\.\d{4}", cell)
    return num(m.group(0)) if m else None


def hours(cell: str) -> float | None:
    if m := re.search(r"(\d+)h(\d+)m", cell):
        return int(m.group(1)) + int(m.group(2)) / 60
    if m := re.search(r"(\d+(?:\.\d+)?)h", cell):
        return float(m.group(1))
    return None


def size(model: str) -> str:
    name = model.split("@")[0].split(" ")[0]
    found = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)([MmBb])(?![a-zA-Z])", name)
    label = (found[0][0] + found[0][1].upper()) if found else name
    quant = re.search(r"Q4_K_M|Q8_0|bf16", model)
    return label + (f" {quant.group(0).split('_')[0]}" if quant and quant.group(0) != "Q4_K_M" else "")


def short(model: str) -> str:
    return model.replace("-Instruct-2507", "-Inst").replace("-Instruct", "")


def section_tables(prefix: str) -> dict[str, tuple[list[str], list[list[str]]]]:
    """Appendix tables keyed by task, from headings like ``### A.3 calendar_json — …``."""
    out = {}
    for line in LINES:
        if m := re.match(rf"### {prefix}\.\d+ (\S+)", line):
            out[m.group(1)] = table(line)
    return out


# ───────────────────────────────────────────────────────────────────────────────────── drawing
def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote img/results/{name}", file=sys.stderr)


def legend_top(ax, ncol: int | None = None, handles=None, labels=None) -> None:
    kw = {"handles": handles, "labels": labels} if handles else {}
    ax.legend(**kw, ncol=ncol or 4, loc="lower left", bbox_to_anchor=(0, 1.0),
              borderaxespad=0.3, handlelength=1.2, columnspacing=1.2)


def legend_fig(fig, handles, labels) -> None:
    fig.legend(handles, labels, ncol=len(labels), loc="outside lower left", handlelength=1.2,
               columnspacing=1.2)


def tip(ax, x, y, text, horizontal=True, size=7.5) -> None:
    """Value label just past a bar end, on the side the bar grows toward."""
    if horizontal:
        ax.annotate(text, (x, y), xytext=(3 if x >= 0 else -3, 0), textcoords="offset points",
                    va="center", ha="left" if x >= 0 else "right", fontsize=size, color=INK_2)
    else:
        ax.annotate(text, (y, x), xytext=(0, 2 if x >= 0 else -2), textcoords="offset points",
                    ha="center", va="bottom" if x >= 0 else "top", fontsize=size, color=INK_2)


def hbars(ax, labels, series, fmt=None, label_series=None, group=0.8):
    """Grouped horizontal bars; ``series`` is [(name, values, color)], first series on top."""
    n = len(series)
    h = min(group / n, 0.26)
    y = np.arange(len(labels))
    for k, (name, vals, color) in enumerate(series):
        off = (k - (n - 1) / 2) * h
        v = np.array([np.nan if x is None else x for x in vals], dtype=float)
        ax.barh(y + off, v, height=h * 0.86, color=color, label=name)
        if fmt and (label_series is None or k in label_series):
            for yi, x in zip(y + off, v):
                if not np.isnan(x):
                    tip(ax, x, yi, fmt(x))
    ax.set_yticks(y, labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def vbars(ax, labels, series, fmt=None, group=0.8):
    n = len(series)
    w = min(group / n, 0.26)
    x = np.arange(len(labels))
    for k, (name, vals, color) in enumerate(series):
        off = (k - (n - 1) / 2) * w
        v = np.array([np.nan if s is None else s for s in vals], dtype=float)
        ax.bar(x + off, v, width=w * 0.86, color=color, label=name)
        if fmt:
            for xi, s in zip(x + off, v):
                if not np.isnan(s):
                    tip(ax, s, xi, fmt(s), horizontal=False)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)


def dumbbell(ax, labels, points, connect=True, sizes=None):
    """One row per label; ``points`` is [(name, values, color)] drawn as dots on a shared row."""
    y = np.arange(len(labels))
    if connect:
        for yi, *vals in zip(y, *[p[1] for p in points]):
            vals = [v for v in vals if v is not None]
            if len(vals) > 1:
                ax.plot([min(vals), max(vals)], [yi, yi], color=GRID, lw=3, zorder=1,
                        solid_capstyle="round")
    for k, (name, vals, color) in enumerate(points):
        v = np.array([np.nan if s is None else s for s in vals], dtype=float)
        ax.scatter(v, y, s=sizes[k] if sizes else 60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=name)
    ax.set_yticks(y, labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def tier_lines(ax, xlabels, series):
    x = np.arange(len(xlabels))
    for name, vals, color in series:
        v = np.array([np.nan if s is None else s for s in vals], dtype=float)
        ax.plot(x, v, color=color, lw=2, solid_capstyle="round", label=name, zorder=2)
        ax.scatter(x, v, s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xticks(x, xlabels)
    ax.grid(axis="x", visible=False)


f3 = "{:.3f}".format
f4 = "{:.4f}".format


def signed(v: float) -> str:
    return f"{v:+.3f}".replace("-", "−")


# ──────────────────────────────────────────────────────────────────────────────────── Part I
def part_i() -> None:
    h, rows = table("| # | Task | What it is | Metric | Claude Sonnet 5")
    labels = [f"{r[1]}\n{r[col(h, 'Metric')]}" for r in rows]
    get = lambda name: [score(r[col(h, name)]) for r in rows]  # noqa: E731
    fig, ax = plt.subplots(figsize=(8.5, 10), layout="constrained")
    hbars(ax, labels, [("Best student", get("Best student"), BLUE),
                       ("Claude Sonnet 5, 0-shot", get("Claude Sonnet 5"), ORANGE),
                       ("DeepSeek v4 Flash, 0-shot", get("DeepSeek"), AQUA),
                       ("Local teacher, 5-shot", get("Local teacher (5-shot)"), YELLOW)],
          fmt=f3, label_series={0})
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("score on the task's own metric")
    ax.set_title("I.1  The fine-tuned student against both APIs and its teacher", pad=26)
    legend_top(ax)
    save(fig, "i1-scores.png")

    h, rows = table("| Task | Job | Outcome | Iterations")
    tasks = [r[0] for r in rows]
    conv = ["converged" in r[col(h, "Outcome")] for r in rows]
    colors = [ORANGE if c else BLUE for c in conv]
    fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharey=True, layout="constrained")
    for ax, name, unit, fmt in zip(axes, ("Iterations", "Wall clock", "Claude"),
                                   ("iterations", "wall clock (h)", "Claude spend ($)"),
                                   ("{:.0f}".format, "{:.1f}h".format, "${:.2f}".format)):
        vals = [num(r[col(h, name)]) for r in rows]
        y = np.arange(len(tasks))
        ax.barh(y, vals, height=0.6, color=colors)
        for yi, v in zip(y, vals):
            tip(ax, v, yi, fmt(v))
        ax.set_yticks(y, tasks)
        ax.set_ylim(len(tasks) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(unit)
        ax.margins(x=0.18)
    from matplotlib.patches import Patch
    legend_top(axes[0], 2, [Patch(color=BLUE), Patch(color=ORANGE)],
               ["full five-tier ladder", "converged at tier 2"])
    axes[0].set_title("I.1  How the main runs ended, and what they cost", pad=26)
    save(fig, "i1-cost.png")

    h, rows = table("| Task | Was | Now | Why it was rerun")
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    was, now = [score(r[1]) for r in rows], [score(r[2]) for r in rows]
    dumbbell(ax, [r[0] for r in rows], [("09-16 note (superseded run)", was, NEUTRAL),
                                         ("current run", now, BLUE)])
    for yi, (a, b) in enumerate(zip(was, now)):
        ax.annotate(f"{a:.4f} → {b:.4f}", (max(a, b), yi), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0.6, 0.95)
    ax.set_xlabel("best score")
    ax.set_title("I.1  What the four revised runs changed", pad=26)
    legend_top(ax, 2)
    save(fig, "i1-changed.png")

    h, rows = table("| # | Task | Metric | Highest-gain tier")
    parts = [("hyperparameters", "…from hyperparameters", HP), ("mining", "…from mine_new_real", MINE),
             ("synthesis", "…from synthesis", SYNTH)]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for name, c, color in parts:
        vals = np.array([num(r[col(h, c)]) or 0.0 for r in rows])
        ax.barh(y, vals, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=1.2,
                label=name)
        left += vals
    for yi, r, total in zip(y, rows, left):
        fv = num(r[col(h, "Format valid")])
        tier, model = r[col(h, "Highest-gain tier")].split(" ", 1)
        note = f"{tier} {size(model.replace(' @', '@'))}"
        if fv < 0.5:
            note += f" · format_valid {fv:.2f}, unusable"
        ax.annotate(f"{signed(total)}  {note}", (total, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [r[1] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Δ search at the highest-gain tier, split by the strategy of each kept step")
    ax.set_title("I.2  Where the search paid: the highest-gain tier per task", pad=26)
    legend_top(ax, 3)
    save(fig, "i2-gain.png")

    h, rows = table("| Task | Base model | Fine-tuned | Claude |")
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    vbars(ax, [r[0] for r in rows], [("base model, zero-shot", [num(r[1]) for r in rows], ZERO),
                                     ("Claude Sonnet 5, 0-shot", [num(r[3]) for r in rows], ORANGE),
                                     ("fine-tuned student", [num(r[2]) for r in rows], BLUE)],
          fmt=f3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("I.3  Where the base model cannot do the task at all", pad=26)
    legend_top(ax, 3)
    save(fig, "i3-base.png")

    h, rows = table("| Task | Corpus available | Real gold")
    tasks = [r[0] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True, width_ratios=[2.2, 1],
                                 layout="constrained")
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for name, color in (("Real gold", NEUTRAL), ("Mined", MINE), ("Synthetic", SYNTH)):
        vals = np.array([num(r[col(h, name)]) for r in rows])
        a1.barh(y, vals, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=1.2,
                label=name.lower())
        left += vals
    for yi, total in zip(y, left):
        tip(a1, total, yi, f"{total:,.0f}")
    a1.set_yticks(y, tasks)
    a1.set_ylim(len(rows) - 0.5, -0.5)
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("rows in the final curriculum")
    a1.set_xlim(0, 11500)
    a1.set_title("I.4  Final curriculum, by source", pad=26)
    legend_top(a1, 3)
    yld = [num(r[col(h, "Synthesis yield")]) for r in rows]
    a2.barh(y, yld, height=0.6, color=SYNTH)
    for yi, v in zip(y, yld):
        tip(a2, v, yi, f"{v:.1f}%")
    a2.grid(axis="y", visible=False)
    a2.set_xlim(0, 125)
    a2.set_xlabel("synthesis yield (% of written rows kept)")
    a2.set_title("Synthesis yield", pad=26)
    save(fig, "i4-data.png")


# ─────────────────────────────────────────────────────────────────────────────────── Part II
def part_ii() -> None:
    h, rows = table("| Teacher | `span_f1` 5-shot")
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, layout="constrained")
    names = ["Qwen3.6-35B-A3B\n(three runs)", "deepseek-v4-flash"]
    for ax, c, label in zip(axes, (1, 2), ("span_f1, 5-shot", "format_valid")):
        vals = [nums(r[c]) for r in rows]
        y = np.arange(2)
        ax.barh(y, [np.mean(v) for v in vals], height=0.55, color=[BLUE, ORANGE])
        for yi, v in zip(y, vals):
            if len(v) > 1:
                ax.scatter(v, [yi] * len(v), s=14, color=INK, zorder=3)
            ax.annotate(" / ".join(f"{x:.4f}" for x in v), (max(v), yi), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
        ax.set_yticks(y, names)
        ax.set_ylim(1.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel(label)
    axes[0].set_title("II.1  Teacher swap on ner_bc5cdr: the format gap, not the score gap")
    save(fig, "ii1-teacher.png")

    h, rows = table("| | tier-1 model | iters | tier-1 best")
    names = [r[0].split(" ")[0] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.4), sharey=True, layout="constrained")
    for ax, c, label, fmt in zip(axes, (3, 2, 4), ("tier-1 best span_f1", "iterations",
                                                    "synthesis rounds"),
                                 (f4, "{:.0f}".format, "{:.0f}".format)):
        vals = [num(r[c]) for r in rows]
        ax.barh([0, 1], vals, height=0.55, color=[SYNTH, NEUTRAL])
        for yi, v in enumerate(vals):
            tip(ax, v, yi, fmt(v))
        ax.set_yticks([0, 1], ["baseline 39311801", "no-synth 39562030"])
        ax.set_ylim(1.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.3)
        ax.set_xlabel(label)
    axes[0].set_title("II.2  No synthesis, same tier-1 model (SmolLM2-360M)")
    save(fig, "ii2-nosynth.png")

    h, rows = table("| Task | Original run | No-convergence rerun")
    fig, ax = plt.subplots(figsize=(8, 2.4), layout="constrained")
    orig, rerun = [nums(r[1]) for r in rows], [nums(r[2]) for r in rows]
    dumbbell(ax, [r[0] for r in rows], [("original run (converged at tier 1)", [o[0] for o in orig], NEUTRAL),
                                        ("goal pinned at 0.99", [o[0] for o in rerun], BLUE)])
    for yi, (o, n) in enumerate(zip(orig, rerun)):
        ax.annotate(f"{o[0]:.4f}, {o[1]:.0f} iters", (o[0], yi), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
        ax.annotate(f"{n[0]:.4f}, {n[1]:.0f} iters, 5 tiers", (n[0], yi), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
    ax.set_xlim(0.88, 1.0)
    ax.set_xlabel("best score")
    ax.set_title("II.4  The two converged tasks, not allowed to stop", pad=26)
    legend_top(ax, 2)
    save(fig, "ii4-noconv.png")


# ────────────────────────────────────────────────────────────────────────────────── Part III
def part_iii() -> None:
    h, rows = table("| Task | Real rows | Best, scarce")
    rows.sort(key=lambda r: -num(r[col(h, "Scarce ÷ full")]))
    labels = [f"{r[0]} ({r[col(h, 'Real rows')].split()[0]} real rows)" for r in rows]
    full = [num(r[col(h, "Full-data best")]) for r in rows]
    scarce = [num(r[col(h, "Best, scarce")]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    dumbbell(ax, labels, [("full data (I.1)", full, FULL), ("scarce + synthesis", scarce, SCARCE)],
             sizes=(110, 50))
    for yi, r in enumerate(rows):
        ax.annotate(f"{num(r[col(h, 'Scarce ÷ full')]):.1f}%", (1.0, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("best score (any tier)        right margin: scarce ÷ full")
    ax.set_title("III.0  100 real rows against full data, all ten tasks", pad=26)
    legend_top(ax, 2)
    save(fig, "iii0-retention.png")

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    yld = [num(r[col(h, "Synthesis yield")]) for r in rows]
    ret = [num(r[col(h, "Scarce ÷ full")]) for r in rows]
    ax.axhline(100, color=MUTED, lw=1)
    ax.scatter(yld, ret, s=60, color=SCARCE, edgecolor=SURFACE, linewidth=2, zorder=3)
    last = None
    for r, x, yv in sorted(zip(rows, yld, ret), key=lambda t: -t[2]):
        if x < 90:
            ax.annotate(r[0], (x, yv), xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=8, color=INK_2)
            continue
        ly = yv if last is None else min(yv, last - 2.6)
        last = ly
        ax.annotate(r[0], (x, yv), xytext=(80, ly), textcoords="data", ha="right", va="center",
                    fontsize=8, color=INK_2,
                    arrowprops={"arrowstyle": "-", "color": NEUTRAL, "lw": 0.8,
                                "shrinkA": 2, "shrinkB": 4})
    ax.set_xlim(-5, 108)
    ax.set_ylim(65, 108)
    ax.set_xlabel("synthesis yield (% of teacher-written rows that survived QC)")
    ax.set_ylabel("scarce ÷ full (%)")
    ax.set_title("III.0  Synthesis yield predicts which tasks hold at 100 rows")
    save(fig, "iii0-yield.png")

    gaps = section_tables("B")
    tiers = 5
    grid = np.full((len(TASK_ORDER), tiers), np.nan)
    for i, task in enumerate(TASK_ORDER):
        hb, rb = gaps[task]
        for r in rb:
            grid[i, int(r[0]) - 1] = num(r[col(hb, "Scarce − full")]) if num(
                r[col(hb, "Scarce − full")]) is not None else np.nan
    lim = np.nanmax(np.abs(grid))
    cmap = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])
    cmap.set_bad("#e1e0d9")
    fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(tiers):
            v = grid[i, j]
            txt = "not trained\nin full run" if np.isnan(v) else signed(v)
            dark = not np.isnan(v) and abs(v) > lim * 0.55
            ax.text(j, i, txt, ha="center", va="center", fontsize=7 if np.isnan(v) else 8,
                    color="#ffffff" if dark else (MUTED if np.isnan(v) else INK))
    ax.set_xticks(range(tiers), [f"T{t}" for t in range(1, tiers + 1)])
    ax.set_yticks(range(len(TASK_ORDER)), TASK_ORDER)
    ax.grid(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.set_label("scarce − full, same model (blue: scarce higher)")
    ax.set_title("III.0  Scarce minus full data at every tier (Appendix B)")
    save(fig, "iii0-gap-heatmap.png")

    h, rows = table("| Task | Full data: hyperparameters / mining / synthesis")
    split = lambda cell: [num(p) or 0.0 for p in cell.split("/")]  # noqa: E731
    fullv = [split(r[1]) for r in rows]
    scarcev = [split(r[2]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, sharex=True, layout="constrained")
    for ax, vals, keys, title in ((axes[0], fullv, ("hyperparameters", "mining", "synthesis"),
                                   "Full data"),
                                  (axes[1], scarcev, ("hyperparameters", "synthesis"),
                                   "100 real rows, mining refused")):
        y = np.arange(len(rows))
        left = np.zeros(len(rows))
        colors = {"hyperparameters": HP, "mining": MINE, "synthesis": SYNTH}
        for k, key in enumerate(keys):
            v = np.array([x[k] for x in vals])
            ax.barh(y, v, left=left, height=0.6, color=colors[key], edgecolor=SURFACE,
                    linewidth=1.2, label=key)
            left += v
        ax.set_yticks(y, [r[0] for r in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("kept-step gain summed over all tiers")
        ax.set_title(title, fontsize=10, pad=6)
    fig.suptitle("III.0  Where the gains came from, full data vs scarce", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    legend_fig(fig, *axes[0].get_legend_handles_labels())
    save(fig, "iii0-attribution.png")

    hs, srows = table("| Task | Real rows | Best, scarce")
    hf, frows = table("| Task | Job | Outcome | Iterations")
    fmap = {r[0]: r for r in frows}
    smap = {r[0]: r for r in srows}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, layout="constrained")
    for ax, name, unit, fmt in zip(axes, ("Wall clock", "Claude"), ("wall clock (h)", "Claude ($)"),
                                   ("{:.0f}h".format, "${:.2f}".format)):
        hbars(ax, TASK_ORDER, [("full data", [num(fmap[t][col(hf, name)]) for t in TASK_ORDER], FULL),
                               ("scarce", [num(smap[t][col(hs, name)]) for t in TASK_ORDER], SCARCE)],
              fmt=fmt)
        ax.set_xlabel(unit)
        ax.margins(x=0.2)
    axes[0].set_title("III.0  Scarcity does not save compute", pad=26)
    legend_top(axes[0], 2)
    save(fig, "iii0-cost.png")

    h, rows = table("| | sms_spam | clinc150 | xlam_bfcl |")
    by = {r[0]: r[1:] for r in rows}
    tasks = h[1:]
    ab = [nums(c) for c in by["arm A → arm B, final tier"]]
    ref = [nums(c)[0] for c in by["full-data reference"]]
    wall = [[hours(p) for p in c.split("/")] for c in by["wall clock, A / B"]]
    usd = [nums(c) for c in by["Claude cost, A / B"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True, layout="constrained")
    dumbbell(axes[0], tasks, [("arm A, no synthesis", [v[0] for v in ab], NOSYNTH),
                              ("arm B, synthesis", [v[1] for v in ab], SCARCE),
                              ("full-data reference", ref, FULL)], sizes=(120, 80, 40))
    axes[0].set_xlim(0.84, 0.97)
    axes[0].set_xlabel("final-tier best")
    for ax, data, unit, fmt in ((axes[1], wall, "wall clock (h)", "{:.1f}h".format),
                                (axes[2], usd, "Claude ($)", "${:.2f}".format)):
        hbars(ax, tasks, [("arm A", [d[0] for d in data], NOSYNTH),
                          ("arm B", [d[1] for d in data], SCARCE)], fmt=fmt)
        ax.set_xlabel(unit)
        ax.margins(x=0.25)
    axes[0].set_title("III  The A/B pairs: what synthesis adds at fixed scarcity", pad=26)
    legend_top(axes[0], 3)
    save(fig, "iii-ab.png")

    h, rows = table("| Tier | Model | Zero-shot | arm A best (format)")
    xl = [f"T{r[0]}\n{size(r[1])}" for r in rows]
    a = [nums(r[3]) for r in rows]
    b = [nums(r[4]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    tier_lines(a1, xl, [("zero-shot", [num(r[2]) for r in rows], ZERO),
                        ("arm A, no synthesis", [v[0] for v in a], MUTED),
                        ("arm B, synthesis", [v[0] for v in b], SCARCE)])
    a1.set_ylim(0, 1)
    a1.set_ylabel("ast_arg_match")
    a1.set_title("III.4  xlam_bfcl at 100 real rows, per tier", pad=26)
    legend_top(a1, 3, *a1.get_legend_handles_labels())
    tier_lines(a2, xl, [("arm A", [v[1] for v in a], MUTED), ("arm B", [v[1] for v in b], SCARCE)])
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("format_valid")
    legend_top(a2, 2)
    a2.set_title("Format validity: synthesis taught the 270M the format", pad=26)
    save(fig, "iii4-xlam.png")

    h, rows = table("| | T2 | T3 | T4 | T5 |")
    fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
    tier_lines(ax, h[1:], [("first pass (kept)", [score(c) for c in rows[1][1:]], SCARCE),
                           ("second pass (resume 40388434)", [score(c) for c in rows[0][1:]], ORANGE)])
    ax.set_ylim(0.65, 0.9)
    ax.set_ylabel("ast_arg_match")
    ax.set_title("III.4  xlam_bfcl scarce run, first vs second pass", pad=26)
    legend_top(ax, 2)
    save(fig, "iii4-passes.png")

    h, rows = table("| check (arm B's final curriculum)")
    by = {r[0]: r[1:] for r in rows}
    tasks = h[1:]
    picks = [("near-duplicates (4-gram Jaccard ≥ 0.6; gold 0%)", "near-duplicates (%)\ngold: 0%"),
             ("lexical diversity vs gold (size-matched TTR)", "lexical diversity vs gold (%)\ngold: 100%"),
             ("teacher would have rejected (shadow mode)", "teacher would have rejected (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True, layout="constrained")
    for ax, (key, label) in zip(axes, picks):
        vals = [num(c) for c in by[key]]
        ax.barh(range(3), vals, height=0.55, color=SCARCE)
        for yi, v in enumerate(vals):
            tip(ax, v, yi, f"{v:g}%")
        if "diversity" in key:
            ax.axvline(100, color=MUTED, lw=1)
        ax.set_yticks(range(3), tasks)
        ax.set_ylim(2.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
        ax.set_xlabel(label)
    axes[0].set_title("III.5  Synthetic-row quality in arm B's final curriculum")
    save(fig, "iii5-quality.png")

    h, rows = table("| smallest tier | with full real data")
    fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    vbars(ax, [r[0] for r in rows], [("full real data", [num(r[1]) for r in rows], FULL),
                                     ("scarce, no synthesis", [num(r[2]) for r in rows], NOSYNTH),
                                     ("scarce + synthesis", [num(r[3]) for r in rows], SCARCE)],
          fmt=f3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("best score")
    ax.set_title("III.6  The smallest tier: a data limit or a capacity limit?", pad=26)
    legend_top(ax, 3)
    save(fig, "iii6-smallest.png")

    h, rows = table("| same two adapters, scored…")
    fig, ax = plt.subplots(figsize=(7.5, 2.6), layout="constrained")
    gaps_ = [num(r[3]) for r in rows]
    ax.barh(range(len(rows)), gaps_, height=0.55, color=BLUE)
    for yi, v in enumerate(gaps_):
        tip(ax, v, yi, f"{v:.4f}")
    ax.axvline(0.015, color=MUTED, lw=1)
    ax.annotate("±0.015 noise floor", (0.015, -0.45), xytext=(3, 0), textcoords="offset points",
                fontsize=8, color=MUTED, va="top")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.02)
    ax.set_xlabel("score gap between two identically trained adapters")
    ax.set_title("III.7  Quantization and a small eval amplify training noise")
    save(fig, "iii7-noise.png")


# ─────────────────────────────────────────────────────────────────────────────────── Part IV
def part_iv() -> None:
    h, rows = table("| Tier | Model | Iters | Zero-shot (format)")
    xl = [f"T{r[0]}\n{size(r[1])}" for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), width_ratios=[1.4, 1], layout="constrained")
    tier_lines(a1, xl, [("zero-shot", [num(r[3]) for r in rows], ZERO),
                        ("first fine-tune", [num(r[4]) for r in rows], FIRST),
                        ("best", [num(r[5]) for r in rows], BEST)])
    a1.set_ylim(0, 1)
    a1.set_ylabel("macro_f1 (MNN 4-bit)")
    a1.set_title("IV.1  clinc150 loop run on MNN, per tier", pad=26)
    legend_top(a1, 3)
    cur = [num(r[col(h, "Curriculum")]) for r in rows]
    a2.bar(range(len(rows)), cur, width=0.55, color=NEUTRAL)
    for x, v in enumerate(cur):
        tip(a2, v, x, f"{v:,.0f}", horizontal=False)
    a2.set_xticks(range(len(rows)), xl)
    a2.grid(axis="x", visible=False)
    a2.set_ylabel("curriculum rows at tier end")
    a2.set_title("Curriculum (from 600 gold rows)", pad=26)
    save(fig, "iv1-mnn-loop.png")

    h, rows = table("| Tier | Model | llama.cpp Q4_K_M | MNN 4-bit")
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    vbars(ax, [f"T{r[0]} {short(r[1])}" for r in rows],
          [("llama.cpp Q4_K_M", [num(r[2]) for r in rows], BLUE),
           ("MNN 4-bit (default recipe)", [num(r[3]) for r in rows], ORANGE)], fmt=f3)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("macro_f1, zero-shot")
    ax.set_title("IV.2  Same base weights, two backends: zero-shot clinc150", pad=26)
    legend_top(ax, 2)
    save(fig, "iv2-zeroshot.png")

    h, rows = table("| Model | MNN FP16")
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    vbars(ax, [short(r[0]) for r in rows],
          [("MNN FP16", [num(r[1]) for r in rows], PRECISION["bf16"]),
           ("MNN 8-bit", [num(r[2]) for r in rows], PRECISION["Q8_0"]),
           ("MNN 4-bit", [num(r[3]) for r in rows], PRECISION["Q4_K_M"])], fmt=f3)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("macro_f1, zero-shot (100 rows)")
    ax.set_title("IV.2  The loss is in MNN's 4-bit export", pad=26)
    legend_top(ax, 3)
    save(fig, "iv2-matrix.png")

    h, rows = table("| Weights | llama.cpp Q4_K_M")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True, layout="constrained")
    for ax, kind in zip(axes, ("fine-tuned", "zero-shot")):
        sub = [r for r in rows if kind in r[0]]
        hbars(ax, [r[0].split(",")[0] for r in sub],
              [(h[k], [num(r[k]) for r in sub], c)
               for k, c in zip(range(1, 5), (BLUE, ORANGE, AQUA, YELLOW))],
              fmt=f3)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel(f"macro_f1, {kind}")
        ax.set_title(kind.capitalize(), fontsize=10, pad=6)
    fig.suptitle("IV.2  Closing the gap: HQQ with 32-weight blocks against Q4_K_M", x=0.01,
                 ha="left", fontweight="bold", fontsize=12)
    legend_fig(fig, *axes[0].get_legend_handles_labels())
    save(fig, "iv2-recipes.png")

    h, rows = table("| Task | Raw input, median chars")
    rows.sort(key=lambda r: num(r[col(h, "Train + eval")]))
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    vals = [num(r[col(h, "Train + eval")]) for r in rows]
    ax.barh(range(len(rows)), vals, height=0.6, color=BLUE)
    for yi, v in enumerate(vals):
        tip(ax, v, yi, f"~{v:g} min")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)
    ax.set_xlabel("minutes per train + eval iteration (llama.cpp, Part I runs as of 09-16)")
    ax.set_title("IV.3  Cost per iteration, cheapest first")
    save(fig, "iv3-iteration-cost.png")


# ──────────────────────────────────────────────────────────────────────────────────── Part V
def part_v() -> None:
    h, rows = table("| Task | What it is | Metric | Teacher (5-shot)")
    labels = [f"{r[0]}\n{r[2]}" for r in rows]
    best = [score(r[col(h, "Best student")]) for r in rows]
    zero = [num(r[col(h, "Zero-shot")]) for r in rows]
    teach = [num(r[col(h, "Teacher")]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    dumbbell(ax, labels, [("zero-shot, same model", zero, ZERO), ("teacher, 5-shot", teach, ORANGE),
                          ("best student", best, BEST)])
    for yi, (r, b) in enumerate(zip(rows, best)):
        model = re.search(r"\(([^)]+)\)", r[col(h, "Best student")]).group(1)
        ax.annotate(f"{b:.4f} {model}", (b, yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_2)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("score")
    ax.set_title("V.1  Tasks run outside the main ten", pad=26)
    legend_top(ax, 3)
    save(fig, "v1-other.png")


# ─────────────────────────────────────────────────────────────────────────────────── Part VI
def part_vi() -> None:
    h, rows = table("| Arm | Job | Strategy | Variants trained")
    arms = [r[0] for r in rows]
    colors = [LIGHT_BLUE, LADDER, SWEEP, SINGLE]
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.8), sharey=True, layout="constrained")
    cols = [("Best span_f1", "best span_f1", score, f4),
            ("Iterations", "iterations", num, "{:.0f}".format),
            ("Wall clock", "wall clock (h, log scale)", hours, "{:.1f}h".format),
            ("Claude", "Claude ($)", num, "${:.2f}".format)]
    for ax, (name, unit, parse, fmt) in zip(axes, cols):
        vals = [parse(r[col(h, name)]) for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, [np.nan if v is None else v for v in vals], height=0.6, color=colors)
        for yi, v in zip(y, vals):
            if v is None:
                ax.annotate("not recorded", (0, yi), xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=MUTED)
            else:
                tip(ax, v, yi, fmt(v))
        if name == "Wall clock":
            ax.set_xscale("log")
            ax.set_xlim(1, 1500)
        else:
            ax.margins(x=0.3)
        if name == "Best span_f1":
            ax.set_xlim(0, 1.2)
        ax.set_yticks(y, arms)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(unit)
    axes[0].set_title("VI.1  ner_bc5cdr: ladder vs exhaustive sweep vs one shot")
    save(fig, "vi1-arms.png")

    h, rows = table("| Size class | Best |")
    fig, ax = plt.subplots(figsize=(6, 2.6), layout="constrained")
    vals = [num(r[1]) for r in rows]
    ax.scatter(vals, range(len(rows)), s=70, color=SWEEP, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, v in enumerate(vals):
        tip(ax, v, yi, f"{v:.4f}")
        ax.plot([0.74, v], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.74, 0.88)
    ax.set_xlabel("best span_f1 in the size class (sweep, all 27 variants)")
    ax.set_title("VI.1  Size buys accuracy up to about 1–2B")
    save(fig, "vi1-size-class.png")

    h, rows = table("| Base model | Q4_K_M | Q8_0 | bf16")
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    dumbbell(ax, [short(r[0]) for r in rows],
             [(q, [num(r[k]) for r in rows], PRECISION[q]) for k, q in
              ((1, "Q4_K_M"), (2, "Q8_0"), (3, "bf16"))], sizes=(130, 80, 40))
    ax.set_xlim(0.6, 0.88)
    ax.set_xlabel("best span_f1")
    ax.set_title("VI.1  Precision buys a little, and not consistently", pad=26)
    legend_top(ax, 3)
    save(fig, "vi1-precision.png")

    h, rows = table("| Threshold | Sweep reached it at")
    arms_ = [("sweep", 1, SWEEP), ("ladder, original", 3, LIGHT_BLUE),
             ("ladder, current", 4, LADDER), ("one shot", 5, SINGLE)]
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    labels = [r[0] for r in rows]
    n = len(arms_)
    hh = 0.8 / n
    for k, (name, c, color) in enumerate(arms_):
        off = (k - (n - 1) / 2) * hh
        for yi, r in enumerate(rows):
            v = hours(r[c])
            if v is not None:
                ax.barh(yi + off, v, height=hh * 0.86, color=color,
                        label=name if yi == next(i for i, rr in enumerate(rows) if hours(rr[c])) else None)
                tip(ax, v, yi + off, f"{v:.1f}h")
            elif "not reached" in r[c]:
                ax.annotate(f"{name}: not reached", (1, yi + off), xytext=(3, 0),
                            textcoords="offset points", va="center", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(1, 400)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("hours until the arm's best first reached the threshold (log scale)")
    ax.set_title("VI.1  Time to the same score, like for like", pad=26)
    legend_top(ax, 4)
    save(fig, "vi1-time-to-score.png")

    h, rows = table("| Task | Job | Ladder best (I.1)")
    rows.sort(key=lambda r: -num(r[col(h, "Δ vs ladder")]))
    ladder = [score(r[col(h, "Ladder best")]) for r in rows]
    one = [score(r[col(h, "One-shot best")]) for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), layout="constrained")
    dumbbell(ax, [r[0] for r in rows], [("ladder best (I.1)", ladder, LADDER),
                                        ("one-shot pick", one, SINGLE)])
    for yi, r in enumerate(rows):
        d = num(r[col(h, "Δ vs ladder")])
        ax.annotate(f"{signed(d)}  pick: {short(r[col(h, 'pick')])}", (1.0, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("best score")
    ax.set_title("VI.2  One informed pick against the ladder, all ten tasks", pad=26)
    legend_top(ax, 2)
    save(fig, "vi2-oneshot.png")

    hf, frows = table("| Task | Job | Outcome | Iterations")
    fmap = {r[0]: r for r in frows}
    omap = {r[0]: r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, layout="constrained")
    for ax, name, unit, parse, fmt in ((axes[0], "Wall clock", "wall clock (h)", hours,
                                        "{:.1f}h".format),
                                       (axes[1], "Claude", "Claude ($)", num, "${:.2f}".format)):
        hbars(ax, TASK_ORDER,
              [("ladder", [parse(fmap[t][col(hf, name)]) for t in TASK_ORDER], LADDER),
               ("one shot", [parse(omap[t][col(h, name)]) for t in TASK_ORDER], SINGLE)], fmt=fmt)
        ax.set_xlabel(unit)
        ax.margins(x=0.2)
    axes[0].set_title("VI.2  The one-shot arm costs about a fifth of the ladder", pad=26)
    legend_top(axes[0], 2)
    save(fig, "vi2-cost.png")

    h, rows = table("| Task | Model | One shot | Ladder | Difference |")
    diff = [num(r[4]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.axvspan(-0.015, 0.015, color="#f0efec", zorder=0)
    ax.barh(range(len(rows)), diff, height=0.55, color=[BLUE if d > 0 else RED for d in diff])
    for yi, d in enumerate(diff):
        tip(ax, d, yi, signed(d))
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [f"{r[0]} · {r[1]}" for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.09, 0.13)
    ax.set_xlabel("one shot − ladder on the same model (shaded: ±0.015 noise floor)")
    ax.set_title("VI.2  Same model, two runs: agreement within noise, except where data differed")
    save(fig, "vi2-same-model.png")


# ────────────────────────────────────────────────────────────────────────────────── Part VII
def part_vii() -> None:
    h, rows = table("## VII.5 First result")
    hm, mrows = table("| Metric | Value |")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
    vbars(a1, [f"T{r[0]} {size(r[1])}" for r in rows],
          [("zero-shot", [num(r[3]) for r in rows], ZERO),
           ("first fine-tune", [num(r[4]) for r in rows], FIRST),
           ("best", [num(r[5]) for r in rows], BEST)], fmt=f3)
    a1.set_ylim(0, 1.05)
    a1.set_ylabel("f2_addressed")
    a1.set_title("VII.5  Address gate, per tier", pad=26)
    legend_top(a1, 3)
    collapse = {"f2_addressed": 0.625, "recall": 1.0, "precision": 0.25, "false_yes_rate": 1.0}
    names = [r[0] for r in mrows]
    vals = [num(r[1]) for r in mrows]
    a2.barh(range(len(names)), vals, height=0.55, color=BEST, label="converged gate")
    a2.scatter([collapse[n] for n in names], range(len(names)), marker="|", s=260, color=INK,
               linewidth=2, zorder=3, label="answer “addressed” to everything")
    for yi, v in enumerate(vals):
        if v > 0.3:
            a2.annotate(f"{v:g}", (v, yi), xytext=(-4, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=7.5, color="#ffffff")
        else:
            tip(a2, v, yi, f"{v:g}")
    a2.set_yticks(range(len(names)), names)
    a2.set_ylim(len(names) - 0.5, -0.5)
    a2.grid(axis="y", visible=False)
    a2.set_xlim(0, 1.15)
    a2.set_title("The headline with its three guard metrics", pad=26)
    legend_top(a2, 2)
    save(fig, "vii5-address-gate.png")


# ─────────────────────────────────────────────────────────────────────────────────── Appendices
def small_multiples(name, title, tables, draw, legend_handles, ncols=5):
    nrows = -(-len(TASK_ORDER) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows), sharey=True,
                             layout="constrained")
    for ax, task in zip(axes.flat, TASK_ORDER):
        draw(ax, *tables[task])
        ax.set_title(task, fontsize=10, pad=4)
        ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=12)
    legend_fig(fig, *legend_handles)
    save(fig, name)


def appendices() -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    dot = lambda c: Line2D([], [], color=c, lw=2, marker="o", markersize=5)  # noqa: E731
    a = section_tables("A")

    def tiers_a(ax, h, rows):
        tier_lines(ax, [f"T{r[0]}\n{size(r[1])}" for r in rows],
                   [("zero-shot", [num(r[3]) for r in rows], ZERO),
                    ("first fine-tune", [num(r[4]) for r in rows], FIRST),
                    ("best", [num(r[5]) for r in rows], BEST)])
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.4, len(rows) - 0.6)

    small_multiples("appA-tiers.png", "Appendix A  Per-tier scores, main runs", a, tiers_a,
                    ([dot(ZERO), dot(FIRST), dot(BEST)], ["zero-shot", "first fine-tune", "best"]))

    def gains_a(ax, h, rows):
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for key, color in (("hyperparameters", HP), ("mining", MINE), ("synthesis", SYNTH)):
            v = np.array([num(r[col(h, key)]) or 0.0 for r in rows])
            ax.bar(x, v, bottom=bottom, width=0.55, color=color, edgecolor=SURFACE, linewidth=1)
            bottom += v
        ax.set_xticks(x, [f"T{r[0]}\n{size(r[1])}" for r in rows])
        ax.set_xlim(-0.6, max(len(rows), 2) - 0.4)
        ax.grid(axis="x", visible=False)

    small_multiples("appA-gains.png", "Appendix A  Kept-step gain per tier, by strategy", a,
                    gains_a, ([Patch(color=HP), Patch(color=MINE), Patch(color=SYNTH)],
                              ["hyperparameters", "mining", "synthesis"]))

    b = section_tables("B")

    def tiers_b(ax, h, rows):
        tier_lines(ax, [f"T{r[0]}\n{size(r[1])}" for r in rows],
                   [("zero-shot", [num(r[3]) for r in rows], ZERO),
                    ("full data, same model", [num(r[col(h, "Full data")]) for r in rows], FULL),
                    ("scarce", [num(r[col(h, "Best (scarce)")]) for r in rows], SCARCE)])
        ax.set_ylim(0, 1)

    small_multiples("appB-tiers.png", "Appendix B  Data-scarce runs against full data, per tier", b,
                    tiers_b, ([dot(ZERO), dot(FULL), dot(SCARCE)],
                              ["zero-shot", "full data, same model", "100 real rows + synthesis"]))

    h, rows = table("| # | Variant | Pool tier")
    fig, ax = plt.subplots(figsize=(8.5, 8), layout="constrained")
    y = np.arange(len(rows))
    first = [num(r[col(h, "First fine-tune")]) for r in rows]
    best = [num(r[col(h, "Best")]) for r in rows]
    for yi, f, bv in zip(y, first, best):
        ax.plot([f, bv], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(first, y, s=28, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.2, zorder=2,
               label="first fine-tune")
    for q, color in PRECISION.items():
        idx = [k for k, r in enumerate(rows) if r[1].endswith(q)]
        ax.scatter([best[k] for k in idx], idx, s=60, color=color, edgecolor=SURFACE, linewidth=2,
                   zorder=3, label=f"best, {q}")
    for v, label, ha in ((0.80, "goal 0.80 ", "right"), (max(best), f" ceiling {max(best):.4f}", "left")):
        ax.axvline(v, color=MUTED, lw=1, zorder=0)
        ax.annotate(label, (v, -0.8), ha=ha, fontsize=8, color=MUTED)
    ax.set_yticks(y, [f"{r[0]}. {short(r[1])}" for r in rows], fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.5, 0.9)
    ax.set_xlabel("span_f1 (ner_bc5cdr), in the order the sweep trained them")
    ax.set_title("Appendix C  The size sweep, all 27 variants", pad=26)
    legend_top(ax, 4)
    save(fig, "appC-sweep.png")


def main() -> int:
    for part in (part_i, part_ii, part_iii, part_iv, part_v, part_vi, part_vii, appendices):
        part()
    return 0


if __name__ == "__main__":
    sys.exit(main())
