#!/usr/bin/env python3
"""
Write docs/COST.md and docs/figures/cost_*.png: GPU hours and money for the SLM Factory's ten main suite runs.

  python scripts/cost_md.py                       # per-task totals (from the sources below)
  python scripts/cost_md.py --tiers main10_tiers.json   # also the per-tier breakdown, once extracted from the run logs

Per-task sources (SLM_Factory origin/main 7f0a9a5):
  * wall clock and train+eval iterations: docs/Evan's Notes/09-16-baselines-ablations-and-probes.md, Part IV.3
  * orchestrator calls / cost, local calls, outcome, the xlam_bfcl DeepSeek synthesis bill: SLMF.tex, Table "tab:cost"
    (provenance there: the per-run "cost:" summary lines in logs/slurm/Final Runs/*.out)
  * best tier, tiers climbed, job ids: the 09-16 note, Part I.1
GPU hours = wall clock x 2: every suite job requests --gres=gpu:l40s:2 (training plus the local vLLM teacher).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS, FIGS = ROOT / "docs", ROOT / "docs" / "figures"
GPUS_PER_JOB = 2
USD_PER_GPU_H = 0.99  # RunPod on-demand L40S, US, Sept 2026 (Hostinger's comparison); range $0.47 (Vast) to ~$3.50
PRICE_LOW, PRICE_HIGH = 0.47, 3.50
API_BASELINES_USD = 9.67  # zero-shot Claude + DeepSeek baselines, all 20 cells (09-16 note, Part I caveats)

# task, wall h, iterations, orchestrator calls, orchestrator USD, local calls (None = paid API teacher), outcome,
# tiers climbed, best tier, extra API synthesis USD, job ids
RUNS = [
    ("sms_spam", 9.7, 85, 90, 5.23, 9842, "ceiling (no-stop rerun)", 5, "T3 Qwen3-1.7B Q4_K_M", 0.0, "40105478"),
    ("ner_bc5cdr", 8.8, 55, 62, 3.43, 17831, "converged", 3, "T3 Qwen3.5-2B Q4_K_M", 0.0, "39311801"),
    ("goemotions", 21.7, 90, 91, 6.04, 20794, "ceiling", 5, "T5 Qwen3-4B Q8_0", 0.0, "39708679"),
    ("calendar_json", 9.0, 41, 68, 1.89, 1798, "converged", 3, "T3 Qwen3.5-2B Q4_K_M", 0.0, "39294409"),
    ("gec_bea19", 23.2, 79, 80, 5.19, 22270, "ceiling", 5, "T5 Qwen3-4B Q8_0", 0.0, "39881529"),
    ("multiconer", 31.3, 98, 101, 6.60, 27704, "ceiling", 5, "T5 Qwen3-4B Q8_0", 0.0, "39881531"),
    ("dialogsum", 29.9, 80, 83, 5.04, 21236, "ceiling", 5, "T5 Qwen3-4B Q8_0", 0.0, "39969104"),
    ("clinc150", 67.6, 103, 95, 6.92, 14467, "ceiling (no-stop rerun)", 5, "T5 Qwen3-4B Q8_0", 0.0,
     "40105479 + 40214984"),
    ("xlam_bfcl", 37.3, 73, 75, 4.83, None, "converged", 5, "T5 Qwen3-4B Q8_0", 38.05, "39361648"),
    ("topv2", 66.8, 109, 112, 7.73, 46049, "ceiling", 5, "T5 Qwen3-4B Q8_0", 0.0, "39969102"),
]
TIER_MODELS = {1: "SmolLM2-360M / Gemma-3-270M (Q4_K_M)", 2: "Qwen3.5-0.8B (Q4_K_M)",
               3: "Qwen3-1.7B / Qwen3.5-2B (Q4_K_M)", 4: "Qwen3-4B-Instruct-2507 (Q4_K_M)",
               5: "Qwen3-4B-Instruct-2507 (Q8_0)"}

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style(ax, xlabel=None, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGS / name, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return f"figures/{name}"


def rows():
    out = []
    for t, wall, it, oc, ousd, local, outcome, tiers, best, synth, jobs in RUNS:
        gpu_h = wall * GPUS_PER_JOB
        out.append(dict(task=t, wall=wall, iters=it, orch_calls=oc, orch=ousd, local=local, outcome=outcome,
                        tiers=tiers, best=best, synth=synth, jobs=jobs, gpu_h=gpu_h, gpu_usd=gpu_h * USD_PER_GPU_H,
                        min_per_iter=wall * 60 / it))
    for r in out:
        r["total"] = r["gpu_usd"] + r["orch"] + r["synth"]
    return sorted(out, key=lambda r: r["total"])


def fig_gpu_hours(R):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = range(len(R))
    ax.barh(y, [r["gpu_h"] for r in R], color=C[0], height=0.62)
    for i, r in enumerate(R):
        ax.text(r["gpu_h"] + 2, i, f"{r['gpu_h']:.0f} GPU-h  ·  {r['tiers']} tiers  ·  {r['outcome']}",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["task"] for r in R], color=INK2, fontsize=9)
    ax.set_xlim(0, max(r["gpu_h"] for r in R) * 1.75)
    style(ax, xlabel=f"L40S GPU-hours ({GPUS_PER_JOB} GPUs × wall clock)")
    fig.tight_layout()
    return save(fig, "cost_gpu_hours_by_task.png")


def fig_dollars(R):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = list(range(len(R)))
    parts = [("gpu_usd", f"GPU time at ${USD_PER_GPU_H:.2f}/GPU-h (estimate)", C[0]),
             ("orch", "Orchestrator (Claude Sonnet 5)", C[1]), ("synth", "Synthesis (DeepSeek v4 Flash)", C[2])]
    left = [0.0] * len(R)
    for key, label, color in parts:
        vals = [r[key] for r in R]
        if not any(vals):
            continue
        ax.barh(y, vals, left=left, color=color, height=0.62, label=label, edgecolor=SURFACE, linewidth=1.5)
        left = [a + b for a, b in zip(left, vals)]
    for i, r in enumerate(R):
        ax.text(left[i] + 2, i, f"${r['total']:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["task"] for r in R], color=INK2, fontsize=9)
    ax.set_xlim(0, max(left) * 1.15)
    style(ax, xlabel="US dollars per run")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, "cost_dollars_by_task.png")


LABEL_OFFSET = {"clinc150": (-6, 6, "right"), "topv2": (6, -12, "left"), "calendar_json": (6, 6, "left"),
                "ner_bc5cdr": (6, -12, "left"), "dialogsum": (-6, 6, "right"), "multiconer": (6, 4, "left")}


def fig_per_iteration(R):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for r in R:
        ax.scatter(r["iters"], r["gpu_h"], s=60, color=C[0] if r["tiers"] == 5 else C[3], zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
        dx, dy, ha = LABEL_OFFSET.get(r["task"], (6, 4, "left"))
        ax.annotate(f"{r['task']} ({r['min_per_iter']:.0f} min/iter)", (r["iters"], r["gpu_h"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8, color=INK2)
    style(ax, xlabel="Train + eval iterations in the run")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_ylabel("L40S GPU-hours", color=INK2, fontsize=9)
    ax.scatter([], [], color=C[0], s=50, label="climbed all 5 tiers")
    ax.scatter([], [], color=C[3], s=50, label="stopped at tier 3")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_xlim(30, 130)
    fig.tight_layout()
    return save(fig, "cost_iterations_vs_gpu_hours.png")


def fig_tiers(T, R):
    """T: {task: {tier: {"gpu_h", "usd", "iters"}}}; stacked GPU-hours per task by tier."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    tasks = [r["task"] for r in R if r["task"] in T]
    tier_colors = ["#c9dcf4", "#8db7ea", "#2a78d6", "#1b56a0", "#0f3466"]  # one hue, light to dark: tier order
    for ax, key, xl in ((axes[0], "gpu_h", "L40S GPU-hours"), (axes[1], "usd", "US dollars (orchestration + API)")):
        left = [0.0] * len(tasks)
        for tier in range(1, 6):
            vals = [T[t].get(str(tier), T[t].get(tier, {})).get(key, 0) for t in tasks]
            ax.barh(range(len(tasks)), vals, left=left, color=tier_colors[tier - 1], height=0.62,
                    edgecolor=SURFACE, linewidth=1.5, label=f"Tier {tier}")
            left = [a + b for a, b in zip(left, vals)]
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks, color=INK2, fontsize=9)
        style(ax, xlabel=xl)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, "cost_by_tier.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=Path, help="per-tier JSON extracted from the run logs (see COST.md)")
    a = ap.parse_args()
    R = rows()
    figs = {"gpu": fig_gpu_hours(R), "usd": fig_dollars(R), "iter": fig_per_iteration(R)}
    gpu_h = sum(r["gpu_h"] for r in R)
    orch = sum(r["orch"] for r in R)
    synth = sum(r["synth"] for r in R)
    wall = sum(r["wall"] for r in R)
    gpu_usd = gpu_h * USD_PER_GPU_H

    L = ["# Cost of the ten main suite runs", "",
         "GPU time and money for the SLM Factory's ten main runs (the suite in `SLMF.tex` and the 09-16 note, Part "
         "I). Generated by `scripts/cost_md.py`.", "",
         "## In aggregate", "",
         "| | Total for the 10 runs |", "|---|--:|",
         f"| Wall clock | {wall:.0f} h ({wall / 24:.1f} days of single-job time) |",
         f"| GPU time | **{gpu_h:.0f} L40S GPU-hours** ({GPUS_PER_JOB} GPUs per job) |",
         f"| GPU time, rented-equivalent | **${gpu_usd:,.0f}** at ${USD_PER_GPU_H:.2f}/GPU-h "
         f"(${gpu_h * PRICE_LOW:,.0f}–${gpu_h * PRICE_HIGH:,.0f} across $0.47–$3.50/GPU-h) |",
         f"| Orchestrator API (Claude Sonnet 5) | ${orch:.2f} ({sum(r['orch_calls'] for r in R)} calls) |",
         f"| Synthesis API (DeepSeek v4 Flash, xlam_bfcl only) | ${synth:.2f} |",
         f"| **Total, rented-equivalent** | **${gpu_usd + orch + synth:,.0f}**, of which "
         f"{gpu_usd / (gpu_usd + orch + synth):.0%} is GPU time |",
         f"| Actually paid | **${orch + synth:.2f}** in API fees: the GPUs are the university's Hyak cluster |",
         f"| Train + eval iterations | {sum(r['iters'] for r in R)} |",
         f"| Local teacher / judge / inference calls (free) | {sum(r['local'] or 0 for r in R):,} (9 runs) |", "",
         f"Not included: the zero-shot API baselines (${API_BASELINES_USD:.2f} for all 20 Claude/DeepSeek cells), "
         "ablations, probes and the reruns they replaced.", "",
         f"![Dollars by task]({figs['usd']})", "",
         f"*Rented-equivalent cost per run, cheapest first. Blue is GPU time at ${USD_PER_GPU_H:.2f} per L40S "
         "GPU-hour, orange the Claude orchestrator, green xlam_bfcl's DeepSeek synthesis bill.*", "",
         "## By task", "",
         "| Task | Wall clock | GPU-hours | GPU $ (est.) | Orchestrator | Synthesis API | Total (est.) | Iterations | "
         "Min / iteration | Tiers | Best model | Outcome | Job |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|---|"]
    for r in R:
        L.append(f"| {r['task']} | {r['wall']:.1f} h | {r['gpu_h']:.1f} | ${r['gpu_usd']:.0f} | ${r['orch']:.2f} "
                 f"({r['orch_calls']} calls) | {'$%.2f' % r['synth'] if r['synth'] else '–'} | **${r['total']:.0f}** | "
                 f"{r['iters']} | {r['min_per_iter']:.0f} | {r['tiers']} | {r['best']} | {r['outcome']} | {r['jobs']} |")
    top2 = sorted(R, key=lambda r: -r["gpu_h"])[:2]
    L += ["", f"![GPU-hours by task]({figs['gpu']})", "",
          f"![Iterations vs GPU-hours]({figs['iter']})", "",
          "*Each run's GPU-hours against its train+eval iterations; the label gives minutes per iteration.*", "",
          "## What drives the cost", "",
          f"- **GPU time is {gpu_usd / (gpu_usd + orch + synth):.0%} of the rented-equivalent bill.** The orchestrator "
          f"costs $1.89–$7.73 a run; GPU time costs ${min(r['gpu_usd'] for r in R):.0f}–${max(r['gpu_usd'] for r in R):.0f}.",
          f"- **Two tasks are {sum(r['gpu_h'] for r in top2) / gpu_h:.0%} of all GPU time:** "
          + " and ".join(f"{r['task']} ({r['gpu_h']:.0f} GPU-h)" for r in top2)
          + ". Both climb all five tiers over 100+ iterations, and both are slow per iteration: clinc150's prompt lists "
          "all 151 intents (~1,385 tokens), and topv2 generates nested bracketed parses.",
          f"- **Cost per iteration, not iteration count, separates the tasks.** Iterations range "
          f"{min(r['iters'] for r in R)}–{max(r['iters'] for r in R)}, but wall-clock minutes per iteration range "
          f"{min(r['min_per_iter'] for r in R):.0f}–{max(r['min_per_iter'] for r in R):.0f}: short single-label or "
          "span answers (sms_spam, ner_bc5cdr) are cheap; long prompts and structured outputs (dialogsum, "
          "xlam_bfcl, topv2, clinc150) are not.",
          "- **The runs that stop early are cheapest.** ner_bc5cdr and calendar_json converged at tier 3 in about "
          "nine hours each.",
          "- **The one paid API teacher outspent every orchestrator.** xlam_bfcl's $38.05 of DeepSeek synthesis "
          "(52.7M output tokens, because the thinking-disable flag was only sent on the local path) is more than "
          "the other nine runs' orchestration combined, and 7.9× its own.", ""]

    if a.tiers and a.tiers.exists():
        T = json.loads(a.tiers.read_text())
        fig = fig_tiers(T, R)
        L += ["## By model tier", "", "| Tier | Model | " + " | ".join(r["task"] for r in R) + " | Total |",
              "|---|---|" + "--:|" * (len(R) + 1)]
        for tier in range(1, 6):
            cells, tot = [], 0.0
            for r in R:
                v = T.get(r["task"], {}).get(str(tier), {})
                cells.append(f"{v['gpu_h']:.1f} h / ${v['usd']:.2f}" if v else "–")
                tot += v.get("gpu_h", 0) if v else 0
            L.append(f"| {tier} | {TIER_MODELS[tier]} | " + " | ".join(cells) + f" | {tot:.0f} GPU-h |")
        L += ["", f"![Cost by tier]({fig})", ""]
    else:
        L += ["## By model tier", "",
              "The model tiers the runs climb:", "",
              "| Tier | Model |", "|---|---|"] + [f"| {t} | {m} |" for t, m in TIER_MODELS.items()] + [
              "", "**Per-tier GPU-hours and dollars are not in this document yet.** They are only in each run's own "
              "files on Hyak: the per-iteration timeline in `logs/runs/<run>/checkpoint.json`, the API ledger "
              "(`cost.json` and the cost-event JSONL, one timestamped event per call) and the SLURM accounting. The "
              "notes record only per-run totals. Once those files are copied here, `scripts/cost_md.py --tiers` "
              "adds a per-tier table and figure.", ""]
    L += ["## Method", "",
          f"- **GPU-hours** = SLURM wall clock × {GPUS_PER_JOB}: every suite job holds two L40S GPUs (training and "
          "the local Qwen3.6-35B-A3B teacher, judge and generator) for its whole run.",
          "- **Wall clock and iterations** come from the 09-16 note (Part IV.3), which measured them from the run "
          "logs; clinc150's 67.6 h covers its no-convergence rerun and its resume.",
          "- **API dollars** are the per-run `cost:` summary lines (SLMF.tex, cost table), priced by "
          "`agent/cost.py` at public list rates.",
          f"- **GPU dollars are an estimate:** the runs used the university's Hyak cluster, which bills no per-hour "
          f"rate. ${USD_PER_GPU_H:.2f} per L40S GPU-hour is RunPod's US on-demand price in Hostinger's September 2026 "
          "comparison; specialist clouds range from $0.47 (Vast.ai) to about $1, and bare-metal or hyperscaler "
          "nodes cost several times more.", ""]
    (DOCS / "COST.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {DOCS / 'COST.md'}")


if __name__ == "__main__":
    main()
