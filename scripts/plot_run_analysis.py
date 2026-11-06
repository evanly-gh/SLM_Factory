"""Figures and derived tables for docs/analysis/{COST,EXPLORATION,ESCALATION}.md.

Reads `docs/analysis/data/run_metrics.json` (written by `scripts/analyze_runs.py`) and writes
PNGs to `docs/analysis/figures/` plus `docs/analysis/data/summary.json`, which holds every number
the three docs quote, so a doc table can be regenerated rather than retyped.

Palette: the dataviz reference palette, validated with its `validate_palette.js` — categorical
blue / orange / aqua for ladder / single-model / size-sweep (all-pairs PASS, aqua below 3:1 so it
is always direct-labelled), and a five-step ordinal blue ramp for tiers T1-T5 (ordinal PASS).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "analysis" / "data"
FIGS = ROOT / "docs" / "analysis" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
NEUTRAL = "#b9b8b1"
LADDER = "#2a78d6"
SINGLE = "#eb6834"
SWEEP = "#1baf7a"
TIER = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab", 5: "#104281"}

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
})


def short(selector: str) -> str:
    model, _, quant = selector.partition("@")
    return f"{model.split('/')[-1].replace('-Instruct-2507', '-Inst').replace('-Instruct', '')}@{quant}"


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}", file=sys.stderr)


def hours_to_reach(run: dict, target: float) -> float | None:
    """GPU-hours of active time until the run's best-so-far first reaches ``target``."""
    setup = run["setup_active_seconds"] / 3600
    for ev in run["evals"]:
        if ev["best_so_far"] >= target - 1e-9:
            return round((ev["cumulative_active_hours"]) * run["gpus"], 3)
    return None


def best_eval(run: dict) -> dict:
    return max(run["evals"], key=lambda e: e["score"])


# ─────────────────────────────────────────────────────────────────────────────────────────── COST
def cost(metrics: dict, summary: dict) -> None:
    main = metrics["families"]["main"]
    rate = metrics["gpu_hour_usd"]
    rows = []
    for task in TASK_ORDER:
        r = main[task]
        tiers = {}
        for m in r["models"]:
            t = tiers.setdefault(m["tier"], {"selectors": [], "iterations": 0, "gpu_hours": 0.0,
                                             "api_usd": 0.0, "best": 0.0})
            t["selectors"].append(m["selector"])
            t["iterations"] += m["iterations"]
            t["gpu_hours"] += m["gpu_hours"]
            t["api_usd"] += m["usd"]
            t["best"] = max(t["best"], m["best"])
        other_gpu = r["gpu_hours_total"] - sum(t["gpu_hours"] for t in tiers.values())
        other_usd = r["usd_total"] - sum(t["api_usd"] for t in tiers.values())
        rows.append({
            "task": task, "run_dir": r["run_dir"], "gpus": r["gpus"],
            "wall_hours": round(r["allocation"]["elapsed_hours"], 2),
            "gpu_hours": round(r["gpu_hours_total"], 2),
            "gpu_usd": round(r["gpu_hours_total"] * rate, 2),
            "api_usd": round(r["usd_total"], 2),
            "api_by_provider": r["usd_by_provider"],
            "total_usd": round(r["gpu_hours_total"] * rate + r["usd_total"], 2),
            "iterations": len(r["evals"]),
            "tiers": {str(k): {**v, "gpu_hours": round(v["gpu_hours"], 2),
                               "gpu_usd": round(v["gpu_hours"] * rate, 2),
                               "api_usd": round(v["api_usd"], 2),
                               "gpu_min_per_iter": round(60 * v["gpu_hours"] / v["iterations"], 1)}
                      for k, v in sorted(tiers.items())},
            "setup_overhead_gpu_hours": round(other_gpu, 2),
            "setup_overhead_api_usd": round(other_usd, 2),
        })
    summary["cost"] = rows

    # Aggregate by tier across tasks.
    agg = {}
    for row in rows:
        for k, v in row["tiers"].items():
            a = agg.setdefault(k, {"gpu_hours": 0.0, "api_usd": 0.0, "iterations": 0, "tasks": 0})
            a["gpu_hours"] += v["gpu_hours"]
            a["api_usd"] += v["api_usd"]
            a["iterations"] += v["iterations"]
            a["tasks"] += 1
    for a in agg.values():
        a["gpu_usd"] = round(a["gpu_hours"] * rate, 2)
        a["gpu_min_per_iter"] = round(60 * a["gpu_hours"] / a["iterations"], 1)
        a["gpu_hours"] = round(a["gpu_hours"], 2)
        a["api_usd"] = round(a["api_usd"], 2)
    summary["cost_by_tier"] = agg

    labels = TASK_ORDER[::-1]
    by_task = {row["task"]: row for row in rows}

    # Figure 1: GPU-hours per task, stacked by tier.
    for metric, fname, xlabel, title, other_key in (
        ("gpu_hours", "cost_gpu_hours_by_tier.png", "GPU-hours (L40S)",
         "GPU-hours per task, split by model tier", "setup_overhead_gpu_hours"),
        ("api_usd", "cost_api_usd_by_tier.png", "API spend (USD)",
         "API spend per task, split by model tier", "setup_overhead_api_usd"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for y, task in enumerate(labels):
            left = 0.0
            row = by_task[task]
            for k in sorted(row["tiers"], key=int):
                w = row["tiers"][k][metric]
                ax.barh(y, w, left=left, height=0.62, color=TIER[int(k)],
                        edgecolor=SURFACE, linewidth=1.5)
                left += w
            w = max(0.0, row[other_key])
            ax.barh(y, w, left=left, height=0.62, color=NEUTRAL, edgecolor=SURFACE, linewidth=1.5)
            left += w
            text = f"{left:.1f}" if metric == "gpu_hours" else f"${left:.2f}"
            ax.text(left + ax.get_xlim()[1] * 0.004, y, " " + text, va="center", fontsize=8.5,
                    color=INK_2)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        handles = [Patch(color=TIER[i], label=f"T{i}") for i in range(1, 6)]
        handles.append(Patch(color=NEUTRAL, label="setup / overhead"))
        ax.legend(handles=handles, ncol=6, loc="lower right", bbox_to_anchor=(1, -0.2))
        ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
        save(fig, fname)

    # Figure 3: total dollars per task — GPU at the reference rate + API.
    order = sorted(rows, key=lambda r: r["total_usd"])
    fig, ax = plt.subplots(figsize=(9, 5))
    for y, row in enumerate(order):
        ax.barh(y, row["gpu_usd"], height=0.62, color=LADDER, edgecolor=SURFACE, linewidth=1.5)
        ax.barh(y, row["api_usd"], left=row["gpu_usd"], height=0.62, color=SINGLE,
                edgecolor=SURFACE, linewidth=1.5)
        ax.text(row["total_usd"], y, f"  ${row['total_usd']:.0f}", va="center", fontsize=8.5,
                color=INK_2)
    ax.set_yticks(range(len(order)), [r["task"] for r in order])
    ax.set_xlabel(f"USD  (GPU-hours priced at ${rate:.2f}/L40S-hour)")
    ax.set_title("Total cost per task: GPU time + API spend")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=LADDER, label=f"GPU @ ${rate:.2f}/h"),
                       Patch(color=SINGLE, label="API (orchestrator + any API teacher)")],
              loc="lower right")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
    save(fig, "cost_total_usd_by_task.png")

    # Figure 4: GPU-minutes per iteration by tier, one line per task — the per-step price of scale.
    fig, ax = plt.subplots(figsize=(9, 5))
    ends = []
    for task in TASK_ORDER:
        row = by_task[task]
        ks = sorted(row["tiers"], key=int)
        xs = [int(k) for k in ks]
        ys = [row["tiers"][k]["gpu_min_per_iter"] for k in ks]
        ax.plot(xs, ys, color=NEUTRAL, linewidth=1.5, marker="o", markersize=4, zorder=2)
        ends.append([xs[-1], ys[-1], task])
    # De-collide the end labels: within one x column, push labels apart to a 3.2-minute gap.
    for x in {e[0] for e in ends}:
        col = sorted((e for e in ends if e[0] == x), key=lambda e: e[1])
        placed = []
        for e in col:
            y = e[1] if not placed else max(e[1], placed[-1] + 3.2)
            placed.append(y)
            ax.text(x + 0.08, y, e[2], fontsize=8, color=INK_2, va="center")
    agg_x = sorted(int(k) for k in agg)
    agg_y = [agg[str(k)]["gpu_min_per_iter"] for k in agg_x]
    ax.plot(agg_x, agg_y, color=LADDER, linewidth=2.5, marker="o", markersize=7, zorder=3,
            label="all ten tasks (pooled)")
    for x, y in zip(agg_x, agg_y):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(range(1, 6), [f"T{i}" for i in range(1, 6)])
    ax.set_xlim(0.7, 5.9)
    ax.set_ylabel("GPU-minutes per train→eval iteration")
    ax.set_title("What one iteration costs at each tier")
    ax.legend(loc="upper left")
    save(fig, "cost_gpu_minutes_per_iteration.png")


# ──────────────────────────────────────────────────────────────────────────────────── EXPLORATION
EXPLORATION_RUNS = [
    ("main", "calendar_json", "calendar_json — main suite"),
    ("main", "ner_bc5cdr", "ner_bc5cdr — main suite"),
    ("single_model", "sms_spam", "sms_spam — single model"),
    ("single_model", "calendar_json", "calendar_json — single model"),
    ("single_model", "ner_bc5cdr", "ner_bc5cdr — single model"),
    ("other_converged", "sms_spam (original)", "sms_spam — original run"),
    ("other_converged", "ner_bc5cdr deepseek-ablation", "ner_bc5cdr — DeepSeek-teacher abl."),
    ("other_converged", "ner_bc5cdr reset-ablation", "ner_bc5cdr — reset abl."),
    ("other_converged", "ner_bc5cdr nosynth-ablation", "ner_bc5cdr — no-synth abl."),
    ("other_converged", "address_bench", "address_bench"),
    ("superseded", "calendar_json (08-29)", "calendar_json — superseded 08-29 run"),
    ("superseded", "ner_bc5cdr (08-30)", "ner_bc5cdr — superseded 08-30 run"),
    ("superseded", "xlam_bfcl (DeepSeek teacher)", "xlam_bfcl — superseded DeepSeek run"),
    ("default_goal", "sms_spam", "sms_spam — default-goal rerun"),
    ("default_goal", "clinc150", "clinc150 — default-goal rerun"),
]


def exploration(metrics: dict, summary: dict) -> None:
    out = []
    for fam, key, label in EXPLORATION_RUNS:
        r = metrics["families"][fam][key]
        evs = r["evals"]
        goal_events = r["goal_events"]
        first_met = next(g for g in goal_events if g["kind"] == "met")
        k = first_met["eval_index"] - 1
        at = evs[k]
        last = evs[-1]
        gpus = r["gpus"]
        raises = [g for g in goal_events if g["kind"] == "raise"]
        decisions = []
        for met in goal_events:
            if met["kind"] != "met":
                continue
            # Every "Goal X met … asking the orchestrator" is one threshold_raise call. Runs before
            # the decline reason was logged (`_maybe_raise_threshold`, "previously silent") show
            # the ask and then TERMINATE with nothing between: that is a decline, reason unlogged.
            g = next((d for d in goal_events if d["kind"] in ("raise", "decline")
                      and d["eval_index"] == met["eval_index"]),
                     {"kind": "decline", "eval_index": met["eval_index"],
                      "reason": "(declined — this run predates logging the reason)"})
            e = evs[g["eval_index"] - 1]
            decisions.append({
                "decision": g["kind"], "eval": g["eval_index"],
                "tier_iteration": met["tier_iteration"],
                "turns_shown_remaining": 1500 - (met["tier_iteration"] + 1) * 2,
                "goal": met["goal"], "score": met["score"],
                "margin": round(met["score"] - met["goal"], 4),
                "to": g.get("to"), "selector": e["selector"],
                "active_hours_so_far": e["cumulative_active_hours"],
                "evals_so_far": g["eval_index"],
                "reason": g["reason"],
            })
        s = r["scores_json"]
        out.append({
            "label": label, "family": fam, "key": key, "run_dir": r["run_dir"],
            "initial_goal": s["initial_stop_threshold"], "final_goal": s["stop_threshold"],
            "raises": len(raises),
            "declines": sum(d["decision"] == "decline" for d in decisions),
            "lowers": s["threshold_lowers"],
            "first_met_eval": first_met["eval_index"], "total_evals": len(evs),
            "first_met_score": at["score"], "best_at_first_met": at["best_so_far"],
            "final_best": max(e["score"] for e in evs),
            "score_gain_after_met": round(max(e["score"] for e in evs) - at["best_so_far"], 4),
            "extra_evals": len(evs) - first_met["eval_index"],
            "hours_at_first_met": at["cumulative_active_hours"],
            "total_active_hours": last["cumulative_active_hours"],
            "extra_active_hours": round(last["cumulative_active_hours"]
                                        - at["cumulative_active_hours"], 3),
            "extra_gpu_hours": round((last["cumulative_active_hours"]
                                      - at["cumulative_active_hours"]) * gpus, 3),
            "extra_api_usd": round(sum(e["usd"] for e in evs[first_met["eval_index"]:]), 3),
            "selector_at_met": at["selector"], "tier_at_met": at["tier"],
            "final_selector": max(evs, key=lambda e: e["score"])["selector"],
            "decisions": decisions,
            "trajectory": [(e["cumulative_active_hours"], e["score"], e["best_so_far"])
                           for e in evs],
        })
    summary["exploration"] = out

    # Figure 1: small multiples of the runs that RAISED the goal — best-so-far against time,
    # with the goal as a step line.
    raised = [x for x in out if x["raises"]]
    cols = 3
    rows_n = (len(raised) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 3.3 * rows_n), squeeze=False)
    for ax, x in zip(axes.flat, raised):
        traj = x["trajectory"]
        hrs = [t[0] for t in traj]
        ax.scatter(hrs, [t[1] for t in traj], s=9, color=NEUTRAL, zorder=2, label="each eval")
        ax.step(hrs, [t[2] for t in traj], where="post", color=LADDER, linewidth=2, zorder=3,
                label="best so far")
        # Goal line: initial until each raise, then the raised value (lowers applied too).
        goal = x["initial_goal"]
        gx, gy = [0.0], [goal]
        changes = [(d["active_hours_so_far"], d["to"]) for d in x["decisions"]
                   if d["decision"] == "raise"]
        # Both lowers in this set were logged on the run's final eval (the lowered goal is what
        # let the run converge), so they are drawn there.
        changes += [(hrs[-1], lower["to"]) for lower in x["lowers"]]
        for h, to in sorted(changes):
            gx += [h, h]
            gy += [goal, to]
            goal = to
        gx.append(hrs[-1])
        gy.append(goal)
        ax.plot(gx, gy, color=SINGLE, linewidth=1.5, linestyle="--", zorder=2, label="goal")
        ax.axvline(x["hours_at_first_met"], color=INK_2, linewidth=0.8, linestyle=":")
        ax.set_title(x["label"], fontsize=10)
        ax.set_xlabel("active hours", fontsize=8.5)
        lo = min(min(t[2] for t in traj), x["initial_goal"]) - 0.03
        hi = max(max(t[2] for t in traj), max(gy)) + 0.02
        ax.set_ylim(max(0, lo), min(1.0, hi))
        ax.annotate(f"+{x['extra_active_hours']:.1f} h after goal met\n"
                    f"Δbest {x['score_gain_after_met']:+.4f}",
                    xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=8.5, color=INK)
    for ax in list(axes.flat)[len(raised):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Runs where the orchestrator raised the goal after meeting it  "
                 "(dotted line = goal first met)", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "exploration_trajectories.png")

    # Figure 2: dumbbell — best score when the goal was first met → final best, per run.
    order = sorted(out, key=lambda x: x["score_gain_after_met"])
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for y, x in enumerate(order):
        c = LADDER if x["raises"] else NEUTRAL
        ax.plot([x["best_at_first_met"], x["final_best"]], [y, y], color=c, linewidth=2, zorder=2)
        ax.scatter([x["best_at_first_met"]], [y], s=40, color=SURFACE, edgecolor=c, linewidth=2,
                   zorder=3)
        ax.scatter([x["final_best"]], [y], s=40, color=c, zorder=3)
        ax.text(max(x["final_best"], x["best_at_first_met"]) + 0.004, y,
                f"{x['score_gain_after_met']:+.4f}  ·  +{x['extra_evals']} evals, "
                f"+{x['extra_gpu_hours']:.1f} GPU-h",
                va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(order)), [x["label"] for x in order])
    ax.set_xlabel("score   (○ best when the goal was first met   ● final best)")
    ax.set_title("What continuing past the first met goal bought")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=LADDER, linewidth=2, label="orchestrator raised goal"),
                       Line2D([], [], color=NEUTRAL, linewidth=2, label="no raise")],
              loc="lower right")
    lo = min(x["best_at_first_met"] for x in out)
    ax.set_xlim(lo - 0.01, max(x["final_best"] for x in out) + 0.09)
    save(fig, "exploration_score_gain.png")

    # Figure 3: every raise/decline decision — margin over the goal vs. tier-local iteration.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    # Only the outliers are labelled: the low-margin corner is crowded, and EXPLORATION.md
    # tabulates every decision. The counts in the annotations are computed, not typed.
    decisions = [(x, d) for x in out for d in x["decisions"]]
    for x, d in decisions:
        c = LADDER if d["decision"] == "raise" else SINGLE
        m = "o" if d["decision"] == "raise" else "X"
        ax.scatter(d["tier_iteration"], d["margin"], s=70, color=c, marker=m,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, alpha=0.9)
        if d["tier_iteration"] > 6 and d["decision"] == "raise":
            ax.annotate(f"{x['label']}: the one late raise\n(bought nothing; goal lowered back)",
                        (d["tier_iteration"], d["margin"]), xytext=(-8, 26),
                        textcoords="offset points", ha="right", fontsize=8, color=INK_2,
                        arrowprops={"arrowstyle": "-", "color": NEUTRAL, "lw": 0.8})
    early = [d for _, d in decisions if d["tier_iteration"] <= 6]
    late = [d for _, d in decisions if d["tier_iteration"] > 6]
    ax.annotate(f"iteration ≤ 6: {sum(d['decision'] == 'raise' for d in early)} of {len(early)} "
                "decisions were raises", (1, 0.075), fontsize=8.5, color=INK_2)
    ax.annotate(f"iteration > 6: {sum(d['decision'] == 'decline' for d in late)} of {len(late)} "
                f"were declines,\nall at margin ≤ {max(d['margin'] for d in late):.3f}",
                (7.2, 0.045), fontsize=8.5, color=INK_2)
    ax.set_xlabel("tier-local iteration at which the goal was met")
    ax.set_ylabel("score − goal at that moment")
    ax.set_title("When the orchestrator raised the goal, and when it declined")
    ax.legend(handles=[Line2D([], [], color=LADDER, marker="o", linestyle="", markersize=8,
                              label="raised"),
                       Line2D([], [], color=SINGLE, marker="X", linestyle="", markersize=8,
                              label="declined")], loc="upper right")
    ax.set_xlim(0, max(d["tier_iteration"] for _, d in decisions) + 1.5)
    save(fig, "exploration_raise_decisions.png")


# ───────────────────────────────────────────────────────────────────────────────────── ESCALATION
def escalation(metrics: dict, summary: dict) -> None:
    # Ladder arm: the main runs, except that clinc150 and sms_spam use their default-goal reruns, so
    # the ladder and the one-shot pick share the default goal on every task.
    main = {**metrics["families"]["main"], **metrics["families"].get("default_goal", {})}
    single = metrics["families"]["single_model"]
    sweep = metrics["families"]["size_sweep"]["ner_bc5cdr"]
    rate = metrics["gpu_hour_usd"]
    rows = []
    for task in TASK_ORDER:
        L, S = main[task], single[task]
        lb, sb = best_eval(L), best_eval(S)
        rows.append({
            "task": task,
            "ladder_run": L["run_dir"], "single_run": S["run_dir"],
            "ladder_best": lb["score"], "ladder_best_selector": lb["selector"],
            "ladder_best_tier": lb["tier"],
            "single_best": sb["score"], "single_selector": sb["selector"],
            "single_tier": sb["tier"],
            "delta": round(lb["score"] - sb["score"], 4),
            "ladder_gpu_hours": round(L["gpu_hours_total"], 2),
            "single_gpu_hours": round(S["gpu_hours_total"], 2),
            "ladder_wall_hours": round(L["allocation"]["elapsed_hours"], 2),
            "single_wall_hours": round(S["allocation"]["elapsed_hours"], 2),
            "ladder_api_usd": round(L["usd_total"], 2),
            "single_api_usd": round(S["usd_total"], 2),
            "ladder_total_usd": round(L["gpu_hours_total"] * rate + L["usd_total"], 2),
            "single_total_usd": round(S["gpu_hours_total"] * rate + S["usd_total"], 2),
            "ladder_evals": len(L["evals"]), "single_evals": len(S["evals"]),
            "single_converged": S["scores_json"]["converged"],
            "ladder_converged": L["scores_json"]["converged"],
            "ladder_gpu_hours_to_single_best": hours_to_reach(L, sb["score"]),
            "single_gpu_hours_to_ladder_best": hours_to_reach(S, lb["score"]),
            "single_gpu_hours_to_own_best": hours_to_reach(S, sb["score"]),
            "ladder_gpus": L["gpus"], "single_gpus": S["gpus"],
        })
    summary["escalation"] = rows
    summary["sweep"] = {
        "run_dir": sweep["run_dir"], "evals": len(sweep["evals"]),
        "variants": len(sweep["models"]),
        "gpu_hours": round(sweep["gpu_hours_total"], 2),
        "wall_hours": round(sweep["allocation"]["elapsed_hours"], 2),
        "api_usd": round(sweep["usd_total"], 2),
        "best": best_eval(sweep)["score"], "best_selector": best_eval(sweep)["selector"],
        "jobs": sweep["allocation"]["jobs"],
        "gpu_hours_to": {str(t): hours_to_reach(sweep, t) for t in (0.80, 0.81, 0.8532, 0.8586)},
        "per_variant": [{"selector": m["selector"], "tier": m["tier"], "iterations": m["iterations"],
                         "gpu_hours": m["gpu_hours"], "best": round(m["best"], 4)}
                        for m in sweep["models"]],
    }

    # Figure 1: GPU-hours, ladder vs single model, per task (dumbbell on a log axis).
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    labels = TASK_ORDER[::-1]
    by = {r["task"]: r for r in rows}
    for y, task in enumerate(labels):
        r = by[task]
        ax.plot([r["single_gpu_hours"], r["ladder_gpu_hours"]], [y, y], color=GRID, linewidth=3,
                zorder=1)
        ax.scatter(r["single_gpu_hours"], y, s=60, color=SINGLE, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.scatter(r["ladder_gpu_hours"], y, s=60, color=LADDER, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ratio = r["ladder_gpu_hours"] / r["single_gpu_hours"]
        ax.text(r["ladder_gpu_hours"] * 1.12, y, f"{ratio:.1f}×", va="center", fontsize=8.5,
                color=INK_2)
    ny = len(labels)
    ax.scatter(sweep["gpu_hours_total"], ny, s=60, color=SWEEP, zorder=3, edgecolor=SURFACE,
               linewidth=1.5)
    ax.plot([by["ner_bc5cdr"]["single_gpu_hours"], sweep["gpu_hours_total"]], [ny, ny],
            color=GRID, linewidth=3, zorder=1)
    ax.scatter(by["ner_bc5cdr"]["single_gpu_hours"], ny, s=60, color=SINGLE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(by["ner_bc5cdr"]["ladder_gpu_hours"], ny, s=60, color=LADDER, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.text(sweep["gpu_hours_total"] * 1.12, ny,
            f"size sweep, {len(sweep['models'])} of 27 variants", va="center", fontsize=8.5,
            color=INK_2)
    ax.set_yticks(range(ny + 1), labels + ["ner_bc5cdr (all three)"])
    ax.set_xscale("log")
    ax.set_xlabel("GPU-hours (L40S, log scale)")
    ax.set_title("Compute: escalation ladder vs. one orchestrator-picked model")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=SINGLE, marker="o", linestyle="", markersize=8,
                              label="single model"),
                       Line2D([], [], color=LADDER, marker="o", linestyle="", markersize=8,
                              label="ladder (main run)"),
                       Line2D([], [], color=SWEEP, marker="o", linestyle="", markersize=8,
                              label="exhaustive size sweep")],
              loc="lower right")
    ax.set_xlim(2, sweep["gpu_hours_total"] * 6)
    save(fig, "escalation_gpu_hours.png")

    # Figure 2: best score, ladder vs single model.
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for y, task in enumerate(labels):
        r = by[task]
        ax.plot([r["single_best"], r["ladder_best"]], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.scatter(r["single_best"], y, s=60, color=SINGLE, zorder=3, edgecolor=SURFACE,
                   linewidth=1.5)
        ax.scatter(r["ladder_best"], y, s=60, color=LADDER, zorder=3, edgecolor=SURFACE,
                   linewidth=1.5)
        ax.text(max(r["ladder_best"], r["single_best"]) + 0.012, y,
                f"{r['delta']:+.4f}   ladder T{r['ladder_best_tier']} vs single "
                f"{short(r['single_selector'])}",
                va="center", fontsize=8, color=INK_2)
    ax.scatter(sweep_best := best_eval(sweep)["score"], ny, s=60, color=SWEEP, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(by["ner_bc5cdr"]["ladder_best"], ny, s=60, color=LADDER, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(by["ner_bc5cdr"]["single_best"], ny, s=60, color=SINGLE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.text(sweep_best + 0.012, ny, f"sweep {sweep_best:.4f} ({short(best_eval(sweep)['selector'])})",
            va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(ny + 1), labels + ["ner_bc5cdr (all three)"])
    ax.set_xlabel("best score on the task's own metric")
    ax.set_title("Quality: escalation ladder vs. one orchestrator-picked model")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.3, 1.22)
    ax.legend(handles=[Line2D([], [], color=SINGLE, marker="o", linestyle="", markersize=8,
                              label="single model"),
                       Line2D([], [], color=LADDER, marker="o", linestyle="", markersize=8,
                              label="ladder"),
                       Line2D([], [], color=SWEEP, marker="o", linestyle="", markersize=8,
                              label="size sweep")], loc="lower right")
    save(fig, "escalation_best_score.png")

    # Figure 3: the trade-off — GPU-hours the single pick saved vs. score it gave up.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    nudge = {"sms_spam": (7, -9), "ner_bc5cdr": (7, 5), "calendar_json": (7, -8),
             "goemotions": (7, 4)}
    for r in rows:
        saved = r["ladder_gpu_hours"] - r["single_gpu_hours"]
        ax.scatter(saved, r["delta"], s=70, color=LADDER, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3)
        ax.annotate(r["task"], (saved, r["delta"]), textcoords="offset points",
                    xytext=nudge.get(r["task"], (7, 3)), fontsize=8.5, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.axhspan(-0.015, 0.015, color=GRID, alpha=0.6, zorder=0)
    ax.text(0.99, 0.02, "shaded: ±0.015 run-to-run noise band (09-16 III.7)",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK_2)
    ax.set_xlabel("GPU-hours saved by the single pick (ladder − single)")
    ax.set_ylabel("score the ladder gained over the single pick")
    ax.set_title("What the ladder's extra compute bought, per task")
    save(fig, "escalation_tradeoff.png")

    # Figure 4: ner_bc5cdr — best-so-far vs GPU-hours for all three strategies.
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for run, color, label in ((sweep, SWEEP, "size sweep"),
                              (main["ner_bc5cdr"], LADDER, "ladder (main run)"),
                              (single["ner_bc5cdr"], SINGLE, "single model")):
        xs = [e["cumulative_active_hours"] * run["gpus"] for e in run["evals"]]
        ys = [e["best_so_far"] for e in run["evals"]]
        ax.step(xs, ys, where="post", color=color, linewidth=2, label=label)
        ax.scatter(xs[-1], ys[-1], s=40, color=color, zorder=3)
        ax.annotate(f"{ys[-1]:.4f} @ {xs[-1]:.0f} GPU-h", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(6, 7 if color == SINGLE else -12),
                    fontsize=8.5, color=INK_2)
    ax.axhline(0.80, color=INK_2, linewidth=0.8, linestyle="--")
    ax.text(1.05, 0.802, "0.80 goal floor", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("GPU-hours of active time (log scale)")
    ax.set_ylabel("best span_f1 so far")
    ax.set_ylim(0.45, 0.88)
    ax.set_title("ner_bc5cdr: three ways to search the same 27-variant pool")
    ax.legend(loc="lower right")
    save(fig, "escalation_ner_three_strategies.png")


def main() -> int:
    metrics = json.loads((DATA / "run_metrics.json").read_text())
    summary = {"gpu_hour_usd": metrics["gpu_hour_usd"],
               "gpu_hour_usd_low": metrics["gpu_hour_usd_low"],
               "gpu_hour_usd_high": metrics["gpu_hour_usd_high"],
               "generated_at": metrics["generated_at"]}
    cost(metrics, summary)
    exploration(metrics, summary)
    escalation(metrics, summary)
    (DATA / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote data/summary.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
