"""Post-run graphics: summarize a finished pipeline run as PNG charts.

A run's per-iteration history is durably recorded in ``dag.json`` (the final-tier DAG) and,
in memory at run end, across every escalation tier via
:func:`agent.pipeline_status.build_run_progression`. Each DAG node carries the score, the
per-difficulty test report (easy/medium/hard), the dataset composition (gold / synthetic /
mined), the chosen intervention and the orchestrator's free-text hypothesis. This module reads
that structure and emits four artifacts into ``logs/graphics/<run_id>/``:

    accuracy.png             metric across iterations (+ baseline & stop-threshold lines,
                             tier-escalation boundaries)
    difficulty.png           easy / medium / hard accuracy across iterations
    dataset_composition.png  stacked gold / synthetic / mined rows per iteration (+ total line)
    hypotheses.md            per-iteration hypothesis + intervention (prose, not a chart)
    summary.png              the three charts as one combined panel

Two entry points:

* :func:`generate_run_graphics` — called at the end of ``tests/pipeline/run.py`` with the live
  ``state`` (so it sees every tier via ``build_run_progression``). Wrapped by the caller so a
  plotting failure never fails an otherwise-successful run.
* ``python -m agent.run_graphics [run_dir]`` — re-graph any past run from its on-disk JSON
  (falls back to the final-tier ``dag.json`` when the cross-tier history is not on disk).

matplotlib is imported lazily (inside the plotting helpers, ``Agg`` backend) so importing this
module never hard-requires the dependency.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# What the comparison scalar actually measures per task type (name-only; the value lives in the
# EvalResult.f1 field). Imported lazily-safe: this is a plain dict, no heavy deps.
from eval.harness import TASK_METRIC_NAMES


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
def _iteration_records(progression: list[dict]) -> tuple[list[dict], dict]:
    """Flatten a progression (list of per-model entries) into ordered iteration records.

    Every ``model_trajectory`` entry contributes its DAG nodes in order; the first node of each
    entry after the first marks a tier-escalation boundary. ``downward_probe`` entries carry no
    DAG and are skipped for the per-iteration line/bar charts.

    Returns ``(records, meta)`` where ``meta`` describes the final model (metric name, baseline,
    selector) used to label the charts.
    """
    records: list[dict] = []
    tier_boundaries: list[int] = []
    metric_name = "f1"
    final_baseline: float | None = None
    final_selector: str | None = None

    for entry in progression:
        if entry.get("kind") != "model_trajectory":
            continue
        dag = entry.get("dag") or []
        if not dag:
            continue
        if records:
            # A new trajectory after the first means an escalation happened here.
            tier_boundaries.append(len(records))
        final_baseline = entry.get("baseline_f1")
        final_selector = entry.get("selector")
        for node in dag:
            eval_state = node.get("evaluation_state") or {}
            last_eval = eval_state.get("last_eval") or {}
            task_type = ((node.get("pi") or {}).get("S") or {}).get("task_type", "")
            metric_name = (
                last_eval.get("metric")
                or TASK_METRIC_NAMES.get(task_type)
                or metric_name
            )
            report = eval_state.get("test_report") or {}
            by_diff = report.get("by_difficulty") or {}
            comp = (((node.get("pi") or {}).get("D") or {}).get("composition")) or {}
            records.append({
                # 1-based: iterations are counted from 1 everywhere else in the run (logs, DAG,
                # curation entries), and a 0-based axis made matplotlib pad the left edge into
                # negative territory, showing an "iteration -0.5" that does not exist.
                "global_idx": len(records) + 1,
                "iteration": node.get("iteration"),
                "selector": entry.get("selector"),
                "tier": entry.get("tier"),
                "score": node.get("score"),
                "per_class": last_eval.get("per_class") or {},
                "by_difficulty": by_diff,
                "n_gold": int(comp.get("n_gold", 0) or 0),
                "n_generated": int(
                    comp.get("n_hard_generated", comp.get("n_hard", 0)) or 0
                ),
                "n_source": int(comp.get("n_hard_source", 0) or 0),
                "total": int(comp.get("total_examples", 0) or 0),
                "hypothesis": (node.get("hypothesis") or "").strip(),
                "intervention": (node.get("intervention") or "").strip(),
                # "data_rebuild" alone does not say what actually changed. The rebuild plan
                # records which sub-strategy ran (resample / acquire / synthesize), and those
                # are three very different interventions.
                "substrategy": (
                    ((((node.get("pi") or {}).get("D") or {}).get("plan")) or {})
                    .get("strategy")
                    or ""
                ).strip(),
            })

    meta = {
        "metric_name": metric_name,
        "baseline_f1": final_baseline,
        "selector": final_selector,
        "tier_boundaries": tier_boundaries,
    }
    return records, meta


def _progression_from_run_dir(run_dir: str | os.PathLike) -> tuple[list[dict], float | None]:
    """Reconstruct a progression from a run directory's on-disk JSON.

    Cross-tier ``escalation_history`` is not persisted as a standalone artifact, so the standalone
    path graphs the final-tier ``dag.json`` (plus the baseline for its selector). Returns
    ``(progression, stop_threshold)``.
    """
    run_dir = Path(run_dir)

    def _load(name: str, default):
        path = run_dir / name
        if not path.is_file():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default

    dag = _load("dag.json", [])
    scores_blob = _load("scores.json", {})
    baselines = _load("baselines.json", [])

    selector = dag[-1].get("selector") if dag else None
    baseline_f1 = next(
        (
            b.get("baseline_f1")
            for b in baselines
            if b.get("selector", b.get("model_id")) == selector
        ),
        None,
    )
    progression = [{
        "kind": "model_trajectory",
        "selector": selector,
        "tier": dag[-1].get("tier", "?") if dag else "?",
        "baseline_f1": baseline_f1,
        "dag": dag,
    }] if dag else []
    stop_threshold = (
        scores_blob.get("stop_threshold") if isinstance(scores_blob, dict) else None
    )
    return progression, stop_threshold


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _nan(value):
    return float("nan") if value is None else float(value)


def _mark_tiers(ax, records: list[dict], boundaries: list[int]) -> None:
    for idx in boundaries:
        if 0 <= idx < len(records):
            ax.axvline(records[idx]["global_idx"], color="0.6", linestyle=":", linewidth=1)
            ax.annotate(
                f"→ {records[idx].get('selector') or 'tier'}",
                xy=(records[idx]["global_idx"], 0),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=7,
                color="0.4",
                rotation=90,
                va="bottom",
            )


def _iteration_axis(ax, xs) -> None:
    """Label the x-axis as whole-numbered iterations.

    Iterations are a counting index, so matplotlib's default float locator was wrong twice
    over: it invented fractional ticks ("1.5", "2.5") for iterations that cannot exist, and
    its automatic margin pushed the left edge below the first iteration.
    """
    from matplotlib.ticker import MaxNLocator

    ax.set_xlabel("iteration")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    if xs:
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)


def _plot_accuracy(ax, records, meta, stop_threshold) -> None:
    xs = [r["global_idx"] for r in records]
    ys = [_nan(r["score"]) for r in records]
    ax.plot(xs, ys, marker="o", color="#1f77b4", label=meta["metric_name"])
    if meta.get("baseline_f1") is not None:
        ax.axhline(
            meta["baseline_f1"], color="#d62728", linestyle="--", linewidth=1,
            label=f"base-model {meta['baseline_f1']:.3f}",
        )
    if stop_threshold is not None:
        ax.axhline(
            stop_threshold, color="#2ca02c", linestyle="--", linewidth=1,
            label=f"threshold {stop_threshold:.3f}",
        )
    _mark_tiers(ax, records, meta.get("tier_boundaries") or [])
    _iteration_axis(ax, xs)
    ax.set_ylabel(meta["metric_name"])
    ax.set_title(f"Accuracy across iterations — {meta.get('selector') or ''}".strip(" —"))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def _plot_difficulty(ax, records, meta) -> None:
    xs = [r["global_idx"] for r in records]
    bands = ["easy", "medium", "hard"]
    colors = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728"}
    for band in bands:
        ys = [_nan((r["by_difficulty"].get(band) or {}).get("accuracy")) for r in records]
        # Bucket sizes are fixed for the run; label with n from the first record that has it.
        n = next(
            ((r["by_difficulty"].get(band) or {}).get("n") for r in records
             if (r["by_difficulty"].get(band) or {}).get("n") is not None),
            None,
        )
        label = f"{band} (n={n})" if n is not None else band
        ax.plot(xs, ys, marker="o", color=colors[band], label=label)
    _mark_tiers(ax, records, meta.get("tier_boundaries") or [])
    _iteration_axis(ax, xs)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by difficulty (easy / medium / hard)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def _plot_composition(ax, records) -> None:
    xs = [r["global_idx"] for r in records]
    gold = [r["n_gold"] for r in records]
    gen = [r["n_generated"] for r in records]
    src = [r["n_source"] for r in records]
    # The three attributed buckets do NOT always sum to the dataset total: rows added by the
    # synth-fill top-up are recorded under an "unattributed" strategy, and in
    # slm-clinc150-cse-38180646 that was 2,063 of 5,758 rows at iteration 1 — more than a third
    # of the curriculum, absent from the chart. Show the remainder as its own band so the stack
    # height really is the dataset size, which is what makes a separate "total" series redundant.
    other = [max(0, r["total"] - g - e - s) for r, g, e, s in zip(records, gold, gen, src)]
    bottom_gen = gold
    bottom_src = [g + e for g, e in zip(gold, gen)]
    bottom_other = [g + e + s for g, e, s in zip(gold, gen, src)]
    ax.bar(xs, gold, color="#1f77b4", label="gold")
    ax.bar(xs, gen, bottom=bottom_gen, color="#ff7f0e", label="synthetic (generated)")
    ax.bar(xs, src, bottom=bottom_src, color="#9467bd", label="mined (source)")
    if any(other):
        ax.bar(xs, other, bottom=bottom_other, color="#bbbbbb", label="synth-fill (unattributed)")
    # No separate "total" series: the stack height IS the total, so plotting it again drew a
    # line exactly along the top of the bars and added a legend entry for information already
    # on screen. (An earlier version put it on a twin right-hand axis, which was worse — the
    # different scale made it look like an unrelated quantity.)
    _iteration_axis(ax, xs)
    ax.set_ylabel("rows")
    ax.set_title("Dataset composition per iteration")
    ax.grid(True, alpha=0.3, axis="y")
    # Bars run to the top of the axes, so reserve headroom rather than letting the legend sit
    # on top of the first stack.
    stack_tops = [g + e + s + o for g, e, s, o in zip(gold, gen, src, other)]
    if any(stack_tops):
        ax.set_ylim(0, max(stack_tops) * 1.18)
    ax.legend(fontsize=8, loc="upper left")


def _write_hypotheses(records, out_path: Path, meta: dict) -> None:
    """Write the per-iteration decision table, forward-looking.

    The DAG stores a node's ``hypothesis`` as the rationale that PRODUCED that node, so
    iteration 1 (built by cold-start, not by an orchestrator decision) had an empty cell and
    every rationale appeared one row below the result that motivated it. Here the column is
    shifted to read forward instead: the row for iteration N shows the score N achieved and the
    hypothesis the orchestrator then formed to try to beat it — which is what a reader actually
    wants when scanning for "what did it try next, and did it work".
    """
    from agent.nodes.iterate import HYPOTHESIS_MAX_CHARS

    metric = meta["metric_name"]
    baseline = meta.get("baseline_f1")
    baseline_text = "n/a" if baseline is None else f"{baseline:.4f}"
    lines = [
        f"# Orchestrator hypotheses — {meta.get('selector') or 'run'}",
        "",
        f"- **Untrained baseline ({metric}):** {baseline_text} — the zero-shot score of the "
        "selected base model before any fine-tuning; every iteration below is measured against "
        "this starting point.",
    ]
    scored = [r["score"] for r in records if r["score"] is not None]
    if scored:
        lines.append(f"- **Best achieved ({metric}):** {max(scored):.4f} over {len(records)} iteration(s).")
    lines += [
        "",
        "Each row: the score that iteration achieved, the intervention it then chose, and the",
        "data_rebuild sub-strategy that intervention actually ran. The decision is FORWARD-looking",
        "— it is what the orchestrator decided to try NEXT after seeing that row's score.",
        "",
        "Full reasoning is given below the table rather than in a cell. Hypotheses run to "
        f"{HYPOTHESIS_MAX_CHARS} characters and are never truncated (B238), which a markdown "
        "table cannot display legibly.",
        "",
        f"| iter | {metric} | next intervention | sub-strategy |",
        "|---|---|---|---|",
    ]

    def _clean(value: str) -> str:
        return (value or "—").replace("|", "\\|").replace("\n", " ")

    reasoning: list[tuple[int, str]] = []
    for position, record in enumerate(records):
        score = "n/a" if record["score"] is None else f"{record['score']:.3f}"
        # The decision recorded on the NEXT node is the one this iteration's score produced.
        nxt = records[position + 1] if position + 1 < len(records) else None
        if nxt is None:
            interv = sub = "— (run ended here)"
        else:
            interv = _clean(nxt["intervention"])
            sub = _clean(nxt.get("substrategy")) if nxt["intervention"] == "data_rebuild" else "—"
            if (nxt["hypothesis"] or "").strip():
                reasoning.append((record["iteration"], nxt["hypothesis"].strip()))
        lines.append(f"| {record['iteration']} | {score} | {interv} | {sub} |")

    lines += ["", "## Reasoning in full", ""]
    if reasoning:
        for iteration, hypothesis in reasoning:
            lines += [f"**After iteration {iteration} it reasoned:**", "", hypothesis, ""]
    else:
        lines.append("(no orchestrator reasoning recorded for this run)")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(records, meta, stop_threshold, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_accuracy(axes[0][0], records, meta, stop_threshold)
    _plot_difficulty(axes[0][1], records, meta)
    _plot_composition(axes[1][0], records)
    axes[1][1].axis("off")
    latest = records[-1] if records else {}
    note = [
        f"selector: {meta.get('selector') or 'n/a'}",
        f"iterations: {len(records)}",
        f"final {meta['metric_name']}: "
        + ("n/a" if not records or latest.get("score") is None else f"{latest['score']:.3f}"),
        f"baseline: "
        + ("n/a" if meta.get("baseline_f1") is None else f"{meta['baseline_f1']:.3f}"),
        f"threshold: " + ("n/a" if stop_threshold is None else f"{stop_threshold:.3f}"),
    ]
    axes[1][1].text(
        0.02, 0.98, "\n".join(note), va="top", ha="left", fontsize=11, family="monospace",
    )
    fig.suptitle(f"Run summary — {meta.get('selector') or ''}".strip(" —"), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_one(plot_fn, out_path: Path, *args) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_fn(ax, *args)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def _resolve_out_dir(run_dir: str | os.PathLike, out_dir=None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    run_dir = Path(run_dir)
    run_id = run_dir.name
    # logs/graphics/<run_id>/ — a single browsable place for all run visuals, keyed by run id.
    logs_root = run_dir.parent.parent if run_dir.parent.name == "runs" else run_dir.parent
    return logs_root / "graphics" / run_id


def generate_run_graphics(
    run_dir: str | os.PathLike,
    *,
    state: dict | None = None,
    baselines: list[dict] | None = None,
    out_dir: str | os.PathLike | None = None,
    stop_threshold: float | None = None,
) -> list[Path]:
    """Render the run-summary graphics. Returns the list of written file paths.

    When ``state`` is given (inline call from the driver), every escalation tier is graphed via
    :func:`build_run_progression`. Otherwise the run's on-disk JSON is read (final-tier DAG).
    Raising is left to the caller to decide on — the driver wraps this so graphics never fail a run.
    """
    # Force a headless backend before pyplot is imported anywhere in this process.
    import matplotlib
    matplotlib.use("Agg")

    if state is not None:
        from agent.pipeline_status import build_run_progression

        progression = build_run_progression(state, baselines or [])
        if stop_threshold is None:
            stop_threshold = state.get("stop_threshold")
    else:
        progression, disk_threshold = _progression_from_run_dir(run_dir)
        if stop_threshold is None:
            stop_threshold = disk_threshold

    out = _resolve_out_dir(run_dir, out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    # Top level = the whole run, every tier on one axis.
    written.extend(_render_set(_iteration_records(progression), out, stop_threshold))

    # Per tier. A run escalates through several models and each one has its own baseline,
    # difficulty profile and hypothesis chain; collapsing them onto shared axes hides the
    # trajectory of every model but the last. One subdirectory per tier keeps both views.
    for entry in progression:
        if entry.get("kind") != "model_trajectory" or not (entry.get("dag") or []):
            continue
        tier_records, tier_meta = _iteration_records([entry])
        if not tier_records:
            continue
        written.extend(
            _render_set(
                (tier_records, tier_meta),
                out / _tier_dir_name(entry),
                stop_threshold,
            )
        )

    return written


def _tier_dir_name(entry: dict) -> str:
    """Filesystem-safe ``tier<N>_<selector>`` label for one model's subdirectory."""
    tier = entry.get("tier", "x")
    selector = str(entry.get("selector") or entry.get("model_id") or "unknown")
    for bad, good in (("/", "_"), ("@", "__"), (":", "_"), (" ", "")):
        selector = selector.replace(bad, good)
    return f"tier{tier}_{selector}"


def _render_set(
    records_and_meta: tuple[list, dict],
    out: Path,
    stop_threshold: float | None,
) -> list[Path]:
    """Write the full artifact set for one trajectory into ``out``."""
    records, meta = records_and_meta
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    # hypotheses.md is written even for an empty run so the folder is never silently blank.
    hyp_path = out / "hypotheses.md"
    _write_hypotheses(records, hyp_path, meta)
    written.append(hyp_path)

    if not records:
        return written

    acc = out / "accuracy.png"
    _plot_one(_plot_accuracy, acc, records, meta, stop_threshold)
    written.append(acc)

    diff = out / "difficulty.png"
    _plot_one(_plot_difficulty, diff, records, meta)
    written.append(diff)

    comp = out / "dataset_composition.png"
    _plot_one(_plot_composition, comp, records)
    written.append(comp)

    summary = out / "summary.png"
    _plot_summary(records, meta, stop_threshold, summary)
    written.append(summary)

    return written


def _newest_run_dir() -> Path | None:
    proj = Path(__file__).resolve().parent.parent
    runs = proj / "logs" / "runs"
    if not runs.is_dir():
        return None
    candidates = [d for d in runs.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render post-run summary graphics.")
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="Run directory (logs/runs/<ts>). Defaults to $SLM_RUN_DIR, else the newest run.",
    )
    parser.add_argument("--out", default=None, help="Override output directory.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir or os.environ.get("SLM_RUN_DIR")
    if not run_dir:
        newest = _newest_run_dir()
        if newest is None:
            parser.error("no run_dir given and no runs found under logs/runs/")
        run_dir = str(newest)

    written = generate_run_graphics(run_dir, out_dir=args.out)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
