"""Post-run graphics: summarize a finished pipeline run as PNG charts.

A run's per-iteration history is durably recorded in ``dag.json`` (the final-tier DAG) and,
in memory at run end, across every escalation tier via
:func:`agent.pipeline_status.build_run_progression`. Each DAG node carries the score, the
per-difficulty test report (easy/medium/hard), the dataset composition (gold / synthetic /
mined), the chosen intervention and the orchestrator's free-text hypothesis. This module reads
that structure and emits four artifacts into ``logs/graphics/<run_id>/``:

    accuracy.png             metric across iterations (+ baseline line, the stop threshold as a
                             STEP line since it moves mid-run, tier-escalation boundaries)
    difficulty.png           easy / medium / hard accuracy across iterations
    dataset_composition.png  stacked gold / synthetic / mined rows per iteration, labelled with
                             row counts, gold subdivided by originating dataset
    label_performance.png    per-label / per-failure-category score at the final iteration, worst
                             first, with failure counts — what the model is getting wrong, as
                             opposed to how hard the things it got wrong were
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
from collections import Counter
from pathlib import Path



# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
def _metric_for(task: str) -> str | None:
    """The task's metric name, or None for a DAG written by a task the registry no longer has."""
    if not task:
        return None
    from tasks import TASKS

    spec = TASKS.get(str(task))
    return spec.metric_name if spec else None


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
            task = ((node.get("pi") or {}).get("S") or {}).get("task", "")
            metric_name = last_eval.get("metric") or _metric_for(task) or metric_name
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
                # The goal THIS iteration was judged against, stamped by evaluate_node. The
                # threshold moves mid-run in both directions, so a single run-level value drawn as
                # one flat line misrepresented every iteration that was held to a different bar.
                # Older DAGs predate the field; those records carry None and fall back to the
                # run-level constant.
                "stop_threshold": node.get("stop_threshold"),
                "per_class": last_eval.get("per_class") or {},
                # Per-label / per-category performance. Easy-medium-hard says HOW HARD the failures
                # were; this says WHAT they were, which is what an intervention can act on.
                "confusion_pairs": report.get("confusion_pairs") or [],
                "outcome_breakdown": report.get("outcome_breakdown") or [],
                "format_valid": node.get("format_valid"),
                # Whether the iteration was ROLLED BACK. Half of a typical tier's iterations are
                # (run 39311801's tier 3 kept 3 of 9), and a plain line through every score reads
                # as a search that wandered when it was actually a search that tried and reverted.
                # Only the per-tier accuracy chart draws this; the run-wide charts have too many
                # points for the distinction to survive.
                "pruned": bool(node.get("pruned")),
                "by_difficulty": by_diff,
                "n_gold": int(comp.get("n_gold", 0) or 0),
                # Gold rows per originating dataset, so the gold band can be divided when a run
                # draws its real data from more than one corpus.
                "gold_by_source": dict(comp.get("gold_by_source") or {}),
                # Prefer `n_synth_total`, which counts every teacher-generated row. The older
                # `n_hard_generated` counted only the plan's `synthesize` strategy, so
                # generation-family positives fell into the grey untagged band and the synthetic
                # share of the curriculum was drawn far smaller than it really was. Older runs
                # predate the field, so fall back for them.
                "n_generated": int(
                    comp.get(
                        "n_synth_total",
                        comp.get("n_hard_generated", comp.get("n_hard", 0)),
                    )
                    or 0
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
                # The strategy the CURRICULUM records, which is the only place the first build
                # names itself. Iteration 1 has intervention="data_rebuild" and no plan at all —
                # there is nothing to add to yet — so without this it reads as an unrecorded
                # intervention rather than as the initial gold load.
                "composition_strategy": str(comp.get("strategy") or "").strip(),
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
    """Label the x-axis with one tick per iteration.

    Iterations are a counting index, so matplotlib's default float locator was wrong twice
    over: it invented fractional ticks ("1.5", "2.5") for iterations that cannot exist, and
    its automatic margin pushed the left edge below the first iteration. `MaxNLocator(integer=True)`
    fixed the fractions but still thinned the ticks to a round stride, so a 16-iteration tier was
    labelled 2, 4, 6, … and reading a point off the chart meant counting markers. Every iteration
    now gets its own tick; past ~24 the labels are shrunk and rotated rather than dropped.
    """
    from matplotlib.ticker import MultipleLocator

    ax.set_xlabel("iteration")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    if xs:
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
        if len(xs) > 24:
            ax.tick_params(axis="x", labelsize=6, labelrotation=90)
        elif len(xs) > 12:
            ax.tick_params(axis="x", labelsize=7)


def _threshold_series(records, stop_threshold):
    """Per-iteration stop threshold, or None if the run never recorded one.

    Missing values (older DAGs, or an iteration written before the field existed) are carried
    forward from the previous known value and, failing that, backfilled from the run-level
    constant — so the line is continuous and never silently drops to zero.
    """
    raw = [r.get("stop_threshold") for r in records]
    if not any(value is not None for value in raw):
        return None
    series, last = [], stop_threshold
    for value in raw:
        if value is not None:
            last = float(value)
        series.append(last)
    # A leading run of unknowns has nothing to carry forward; take the first known value back.
    first_known = next((v for v in series if v is not None), None)
    return [first_known if v is None else v for v in series]


def _plot_accuracy(ax, records, meta, stop_threshold) -> None:
    xs = [r["global_idx"] for r in records]
    ys = [_nan(r["score"]) for r in records]
    ax.plot(xs, ys, marker="o", color="#1f77b4", label=meta["metric_name"])
    if meta.get("baseline_f1") is not None:
        ax.axhline(
            meta["baseline_f1"], color="#d62728", linestyle="--", linewidth=1,
            label=f"base-model {meta['baseline_f1']:.3f}",
        )
    # The goal is not a constant. iterate_node lowers it when the failure profile looks like a
    # model-capacity limit and raises it as a stretch goal once one is cleared, so drawing the
    # final value as a single horizontal line claimed every earlier iteration had been measured
    # against a bar that did not yet exist — and on a run that lowered its goal, that line sat
    # BELOW iterations the loop had actually judged as failures. Plotted as a step: the threshold
    # holds its value until the iteration that changed it.
    thresholds = _threshold_series(records, stop_threshold)
    if thresholds is not None:
        moved = len(set(thresholds)) > 1
        label = (
            f"threshold {thresholds[0]:.3f}→{thresholds[-1]:.3f}"
            if moved
            else f"threshold {thresholds[-1]:.3f}"
        )
        ax.step(
            xs, thresholds, where="post", color="#2ca02c", linestyle="--", linewidth=1,
            label=label,
        )
        if moved:
            for prev, curr, x in zip(thresholds, thresholds[1:], xs[1:]):
                if curr == prev:
                    continue
                ax.annotate(
                    f"{'▲' if curr > prev else '▼'} {curr:.3f}",
                    xy=(x, curr), xytext=(2, 4 if curr > prev else -12),
                    textcoords="offset points", fontsize=7, color="#2ca02c",
                )
    elif stop_threshold is not None:
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


# What produced each iteration, and how it is drawn. Marker shape AND colour both carry the
# intervention because the two ride together: shape survives a greyscale print and a colourblind
# reader, colour is what makes a cluster ("every kept step here was synthesis") visible without
# reading anything. The one-letter code is what goes under the iteration number on the x-axis.
_INTERVENTION_STYLE: dict[str, dict] = {
    # NOT brown/red (#8c564b). This marker sits on iteration 1, exactly where the removed
    # base-model diamond used to be, and a dark red-brown plus there reads as that diamond still
    # being present. It is a real fine-tuned evaluation — the first one — and its colour needs to
    # say so at a glance. Red is now reserved for the best-iteration star.
    "initial_gold": {"code": "G", "color": "#e377c2", "marker": "P", "label": "initial gold load"},
    "mine_new_real": {"code": "M", "color": "#ff7f0e", "marker": "o", "label": "mine_new_real"},
    "surgical_synthesis": {
        "code": "S", "color": "#9467bd", "marker": "s", "label": "surgical_synthesis",
    },
    "hyperparameter": {"code": "H", "color": "#17becf", "marker": "^", "label": "hyperparameter"},
}
_INTERVENTION_UNKNOWN = {
    "code": "?", "color": "#7f7f7f", "marker": "X", "label": "not recorded",
}


def _intervention_kind(record) -> str:
    """The intervention that produced this iteration, at the granularity that matters.

    `intervention` is only ever "data_rebuild" or "hyperparameter", and "data_rebuild" covers two
    acts with nothing in common — mining REAL rows and generating SYNTHETIC ones. Distinguishing
    those is the whole point of the ablation suite, so a data_rebuild is refined by its plan.

    `intervention` IS CHECKED FIRST, AND IT DECIDES. Refining by the plan unconditionally looks
    tempting and is wrong: `curate_node` SKIPS on a hyperparameter iteration ("dataset held
    fixed"), which leaves both `pi.D.plan` and `pi.D.composition` holding the PREVIOUS rebuild's
    values. On run 39311801's tier 3 every one of the nine nodes carries
    plan.strategy="surgical_synthesis", including the four the run log records as
    `hyperparameter` — so a plan-first reading labelled that tier 9-of-9 synthesis when it was
    5-of-9, and tier 1's hyperparameter iterations came out as `initial_gold`. The plan is only
    evidence about the iteration that ran it.
    """
    intervention = str(record.get("intervention") or "").strip()
    if intervention == "hyperparameter":
        return "hyperparameter"
    if intervention == "data_rebuild":
        # Now the plan is trustworthy: curate ran this iteration and rewrote both fields. The
        # composition is the fallback because the FIRST build has no plan to name itself with.
        for candidate in (record.get("substrategy"), record.get("composition_strategy")):
            name = str(candidate or "").strip()
            if name in _INTERVENTION_STYLE:
                return name
        return ""
    return intervention if intervention in _INTERVENTION_STYLE else ""


def _intervention_style(record) -> dict:
    return _INTERVENTION_STYLE.get(_intervention_kind(record), _INTERVENTION_UNKNOWN)


def _score_range(score_sets, *, margin=0.05) -> tuple[float, float]:
    """5% below the worst point, 5% above whichever is higher of the best point and the goal.

    A fixed 0-1 axis is what the run-wide charts use, and on a per-tier chart it wastes the plot:
    run 39311801's tier 1 lives between 0.5674 and 0.7694, so three quarters of the axis is empty
    and the climb this chart exists to show is a nearly flat line in a band.

    The margin is proportional, not absolute — `worst * 0.95` and `best * 1.05` — so the headroom
    scales with the numbers rather than swallowing a low-scoring tier.

    THE GOAL IS PART OF THE UPPER BOUND, not just the scores. A tier that never reached its goal
    would otherwise have the goal line clipped off the top of its own chart, which is the one
    comparison the chart is always making.

    Callers pass every tier's scores at once, so the range is computed ONCE and shared: a point at
    the same height means the same score on whichever tier you are looking at. That is the only
    reason not to autoscale each tier independently.
    """
    values = [float(v) for s in score_sets for v in s if v is not None and v == v]
    if not values:
        return (0.0, 1.0)
    low = min(values) * (1.0 - margin)
    high = max(values) * (1.0 + margin)
    # A metric cannot leave [0, 1], and an axis implying otherwise invites the reading that
    # headroom above 1.0 was available.
    low, high = max(0.0, low), min(1.0, high)
    if high - low < 1e-6:
        # Every value identical (a one-iteration tier, or a metric that never moved). Not a
        # correction to the rule above — matplotlib cannot render a zero-height axis at all.
        low, high = max(0.0, low - 0.01), min(1.0, high + 0.01)
    return (low, high)


def _tier_iteration_values(records) -> list[int]:
    """Tier-LOCAL iteration numbers for the x-axis.

    `escalate_node` resets `iteration` to 0 for each new tier, so a tier's DAG nodes already carry
    1..N and those are the numbers the run log prints in its per-tier tables — reading a point off
    this chart should agree with reading a row off that table. Falls back to position when the
    stored numbers are absent, duplicated or out of order, which would otherwise draw a line that
    doubles back on itself.
    """
    stored = [r.get("iteration") for r in records]
    if all(isinstance(v, int) and v > 0 for v in stored):
        if len(set(stored)) == len(stored) and stored == sorted(stored):
            return list(stored)
    return list(range(1, len(records) + 1))


def _tier_accuracy_axis(ax, xs, records) -> None:
    """One tick per iteration, with that iteration's intervention code on a second line.

    The code rides on the tick rather than as text beside each marker. A tier can run 26
    iterations, and 26 free-floating labels either overlap each other or overlap the line; the tick
    already has one label per iteration and vertical room underneath it, so the code sits where it
    cannot collide with anything and stays readable at any iteration count.

    The axis starts at the first FINE-TUNED iteration. There is no x=0 base-model slot — see
    `_plot_tier_accuracy` for why the base model is not on this chart.
    """
    from matplotlib.ticker import FixedLocator

    ax.set_xlabel("iteration / intervention")
    ax.xaxis.set_major_locator(FixedLocator(list(xs)))
    ax.set_xticklabels([
        f"{x}\n{_intervention_style(r)['code']}" for x, r in zip(xs, records)
    ])
    if xs:
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    if len(xs) > 24:
        ax.tick_params(axis="x", labelsize=6)
    elif len(xs) > 12:
        ax.tick_params(axis="x", labelsize=7)


def _plot_tier_accuracy(ax, records, meta, stop_threshold, ylim=None) -> None:
    """One tier's fine-tuning trajectory: every iteration, what produced it, and what survived.

    The run-wide `accuracy.png` puts every tier on one axis, which hides the trajectory of every
    model but the last, and draws one line through all iterations, which hides that most of them
    were rolled back. This chart is per tier and draws two lines, so the search's actual path and
    the path of its accepted state are both visible — the gap between them is the cost of search.

    It starts at the FIRST FINE-TUNED evaluation. The base model is not on it; see the note in the
    body for why.
    """
    metric = meta["metric_name"]
    xs = _tier_iteration_values(records)
    ys = [_nan(r["score"]) for r in records]

    # THE BASE MODEL IS NOT PLOTTED. This chart starts at the first fine-tuned evaluation.
    #
    # It used to open at x=0 with the tier's zero-shot score and a dashed segment into iteration 1.
    # That step is real and large — on run 39311801's tier 1 it was 0.0000 to 0.6885 — but it is
    # also the whole y-axis, and it flattens the thing this chart is actually for: the iteration
    # trajectory then occupies the top fifth of the plot and differences of a few points between
    # iterations become unreadable. The base-model comparison already lives in the run-wide
    # `summary.png` and the Model Improvement Report, where it is the point rather than the
    # backdrop. `meta["baseline_f1"]` is deliberately left unread here.

    # TWO LINES, BECAUSE THERE ARE TWO TRAJECTORIES AND THEY ARE NOT THE SAME SHAPE.
    #
    # The dotted line is every evaluation in the order it happened — the search's actual path,
    # including the attempts it threw away. The solid line joins only the KEPT iterations, which is
    # the trajectory of the tier's accepted state: what the model would have scored if you had
    # followed only the decisions that survived rollback.
    #
    # One line through all of them, which is what this chart used to draw, is neither. It reads as a
    # model whose accuracy swung up and down for 26 iterations, when in fact 19 of those swings were
    # reverted and the accepted state only ever moved 7 times. The gap between the two lines IS the
    # cost of search, and it is not visible with one line.
    kept_pts = [(x, y) for x, y, r in zip(xs, ys, records) if not r.get("pruned") and y == y]
    ax.plot(
        xs, ys, linestyle=":", linewidth=1.3, color="#1f77b4", zorder=2,
        label=f"all {len(records)} iteration(s), in order",
    )
    if len(kept_pts) > 1:
        ax.plot(
            [p[0] for p in kept_pts], [p[1] for p in kept_pts],
            linestyle="-", linewidth=2.0, color="#1f77b4", zorder=3,
            label=f"kept trajectory ({len(kept_pts)} of {len(records)})",
        )

    # Marker shape and colour name the intervention; fill says whether it survived.
    #
    # Drawn unlabelled and described by PROXY legend entries instead, because labelling each
    # (intervention, outcome) group directly produces up to ten near-duplicate rows —
    # "S surgical_synthesis - kept", "S surgical_synthesis - rolled back" — for two independent
    # facts. One row per intervention plus two rows explaining the fill says the same thing in
    # half the space, and keeps an intervention that was ALWAYS rolled back in the legend, which
    # per-group labelling would have listed only under its rolled-back variant.
    for kind, style in (*_INTERVENTION_STYLE.items(), ("", _INTERVENTION_UNKNOWN)):
        for pruned in (False, True):
            pts = [
                (x, y) for x, y, r in zip(xs, ys, records)
                if _intervention_kind(r) == kind and bool(r.get("pruned")) is pruned and y == y
            ]
            if not pts:
                continue
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                marker=style["marker"], markersize=7, linestyle="none", zorder=5,
                markerfacecolor="white" if pruned else style["color"],
                markeredgecolor=style["color"], markeredgewidth=1.4,
            )

    # The goal, stepped for the same reason `_plot_accuracy` steps it: it moves mid-run, and one
    # flat line would hold early iterations to a bar that did not exist yet.
    thresholds = _threshold_series(records, stop_threshold)
    if thresholds is not None:
        moved = len(set(thresholds)) > 1
        ax.step(
            xs, thresholds, where="post", color="#2ca02c", linestyle="--", linewidth=1,
            zorder=2,
            label=(
                f"goal {thresholds[0]:.3f}→{thresholds[-1]:.3f}" if moved
                else f"goal {thresholds[-1]:.3f}"
            ),
        )
    elif stop_threshold is not None:
        ax.axhline(
            float(stop_threshold), color="#2ca02c", linestyle="--", linewidth=1, zorder=2,
            label=f"goal {float(stop_threshold):.3f}",
        )

    # THE BEST ITERATION IS MARKED, NOT LABELLED IN PLACE. Free-floating text does not work here:
    # the best score is by definition the point nearest the goal line, and on a converged tier it
    # is also the last one, so "just above the marker" collides with the green dashed goal and
    # "just right of it" leaves the axes. Both happened. A star plus a legend entry carries the
    # same two numbers, cannot overlap anything, and needs no placement heuristic.
    scored = [(x, y) for x, y in zip(xs, ys) if y == y]
    best = max(scored, key=lambda p: p[1]) if scored else None
    if best is not None:
        ax.plot(
            [best[0]], [best[1]], marker="*", markersize=15, linestyle="none",
            color="#d62728", markeredgecolor="#7f1416", markeredgewidth=0.6, zorder=7,
            label=f"best {best[1]:.4f} @ iter {best[0]}",
        )

    tier = meta.get("tier")
    selector = meta.get("selector") or ""
    kept_n = len(records) - sum(1 for r in records if r.get("pruned"))
    subtitle = f"{len(records)} iteration(s), {kept_n} kept"
    if best is not None:
        first = next((y for y in ys if y == y), None)
        subtitle += f", first {first:.4f} → best {best[1]:.4f}" if first is not None else ""
    ax.set_title(
        f"Tier {tier} accuracy — {selector}".strip(" —") + f"\n{subtitle}",
        fontsize=10,
    )
    _tier_accuracy_axis(ax, xs, records)
    ax.set_ylabel(metric)
    # `ylim` is supplied by the caller so every tier of a run shares one range; falling back to
    # this tier's own scores keeps the function usable on its own.
    ax.set_ylim(*(ylim or _score_range([ys, thresholds or []])))
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    counts = Counter(_intervention_kind(r) for r in records)
    for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        style = _INTERVENTION_STYLE.get(kind, _INTERVENTION_UNKNOWN)
        handles.append(Line2D(
            [], [], marker=style["marker"], markersize=7, linestyle="none",
            markerfacecolor=style["color"], markeredgecolor=style["color"],
        ))
        labels.append(f"{style['code']}  {style['label']} ({count})")
    n_pruned = sum(1 for r in records if r.get("pruned"))
    handles += [
        Line2D([], [], marker="o", markersize=7, linestyle="none",
               markerfacecolor="0.25", markeredgecolor="0.25"),
        Line2D([], [], marker="o", markersize=7, linestyle="none",
               markerfacecolor="white", markeredgecolor="0.25", markeredgewidth=1.4),
    ]
    labels += [
        f"filled = kept ({len(records) - n_pruned})",
        f"hollow = rolled back ({n_pruned})",
    ]
    # Outside the axes. With the baseline, both lines, the goal, the best iteration, one row per
    # intervention and two fill rows, `loc="best"` had nowhere to put ~11 entries without covering
    # data — and on this chart the points sit in a narrow band just under the goal, so there is no
    # empty corner to find.
    ax.legend(
        handles, labels, fontsize=7, loc="center left",
        bbox_to_anchor=(1.01, 0.5), borderaxespad=0,
    )


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


def _gold_source_order(records) -> list[str]:
    """Gold source datasets across the run, largest total contribution first."""
    totals: dict[str, int] = {}
    for record in records:
        for source, rows in (record.get("gold_by_source") or {}).items():
            totals[str(source)] = totals.get(str(source), 0) + int(rows or 0)
    return sorted(totals, key=lambda s: (-totals[s], s))


# How many iterations one composition chart may hold. Past this the bars are narrower than the
# row-count text that goes inside them, so the numbers overlap their neighbours and the chart stops
# being readable — a 41-iteration run rendered as one image was unreadable end to end.
COMPOSITION_PAGE_SIZE = 25


def _band_fontsize(n_bars: int) -> int:
    """Text size that still fits INSIDE a bar at this bar count."""
    if n_bars <= 12:
        return 8
    if n_bars <= 20:
        return 7
    return 6


def _annotate_band(ax, x, bottom, height, total, *, color="white", fontsize=6) -> None:
    """Write a band's row count inside the band, when it is tall enough to hold the text.

    "How much of each type of data went into this iteration" was previously only answerable by
    measuring a bar against the y-axis by eye, which for a 3,235-row gold band under a 90-row
    synthetic band is not a readable comparison. The number is now on the band.

    Kept horizontal at every page size. Rotating it to fit more bars per image trades one
    unreadable failure for another; paging the chart is what actually makes room.

    The cutoff is 3% of the tallest stack rather than 6%. At 6% a 192-row synthetic band on a
    4,092-row curriculum went unlabelled — 3.6% of the bar, roughly 16 pixels, comfortably enough
    for 6pt text — and the synthetic band is the one a reader is most often looking for, because it
    is the smallest and the one an intervention just added. Bands in the tight range drop a point
    of font rather than dropping their number.
    """
    if not height or not total or height < total * 0.03:
        return
    if height < total * 0.06:
        fontsize = max(5, fontsize - 1)
    ax.text(
        x, bottom + height / 2, f"{height:,}", ha="center", va="center",
        fontsize=fontsize, color=color,
    )


def _plot_composition(ax, records) -> None:
    band_fontsize = _band_fontsize(len(records))
    xs = [r["global_idx"] for r in records]
    gold = [r["n_gold"] for r in records]
    gen = [r["n_generated"] for r in records]
    src = [r["n_source"] for r in records]
    # The attributed buckets do NOT always sum to the dataset total (a producer that forgets to
    # tag `_provenance` lands here), and in slm-clinc150-cse-38180646 the untagged remainder was
    # 2,063 of 5,758 rows at iteration 1 — more than a third of the curriculum, absent from the
    # chart. Show it as its own band so the stack height really is the dataset size, which is what
    # makes a separate "total" series redundant.
    other = [max(0, r["total"] - g - e - s) for r, g, e, s in zip(records, gold, gen, src)]

    # Gold subdivided by originating dataset. A run whose real data comes from one corpus gets one
    # solid band; a run mixing corpora gets one sub-band per source with a divider between them,
    # because "3,235 gold rows" says nothing about whether they are all the same distribution.
    sources = _gold_source_order(records)
    multi_source = len(sources) > 1
    gold_shades = ["#1f77b4", "#4c94c7", "#7fb3d8", "#a6cae4", "#c6dcef", "#dbe9f6"]

    tops = [g + e + s + o for g, e, s, o in zip(gold, gen, src, other)]
    peak = max(tops) if tops else 0

    if multi_source:
        bottoms = [0.0] * len(records)
        for position, source in enumerate(sources):
            heights = [
                int((r.get("gold_by_source") or {}).get(source, 0) or 0) for r in records
            ]
            if not any(heights):
                continue
            ax.bar(
                xs, heights, bottom=bottoms, color=gold_shades[position % len(gold_shades)],
                label=f"gold — {source}",
            )
            for x, bottom, height in zip(xs, bottoms, heights):
                _annotate_band(ax, x, bottom, height, peak, fontsize=band_fontsize)
                if bottom > 0:
                    # The divider is what makes two adjacent shades read as two datasets rather
                    # than as a gradient.
                    ax.hlines(bottom, x - 0.4, x + 0.4, color="white", linewidth=1.2)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
    else:
        label = f"gold — {sources[0]}" if sources else "gold"
        ax.bar(xs, gold, color=gold_shades[0], label=label)
        for x, height in zip(xs, gold):
            _annotate_band(ax, x, 0, height, peak, fontsize=band_fontsize)

    bottom_gen = gold
    bottom_src = [g + e for g, e in zip(gold, gen)]
    bottom_other = [g + e + s for g, e, s in zip(gold, gen, src)]
    ax.bar(xs, gen, bottom=bottom_gen, color="#ff7f0e", label="synthetic (teacher-generated)")
    ax.bar(xs, src, bottom=bottom_src, color="#9467bd", label="mined (new real rows)")
    if any(other):
        ax.bar(xs, other, bottom=bottom_other, color="#bbbbbb", label="untagged")
    for x, base, height in zip(xs, bottom_gen, gen):
        _annotate_band(ax, x, base, height, peak, color="black", fontsize=band_fontsize)
    for x, base, height in zip(xs, bottom_src, src):
        _annotate_band(ax, x, base, height, peak, fontsize=band_fontsize)
    for x, base, height in zip(xs, bottom_other, other):
        _annotate_band(ax, x, base, height, peak, color="black", fontsize=band_fontsize)
    # Above each stack: the total, and — only where it MOVED — how much it moved by.
    #
    # WHY ONLY WHERE IT MOVED
    #     Every bar used to carry its total plus a "+N syn +N mined" line. On a run where the
    #     curriculum is unchanged for fifteen consecutive iterations that printed the identical
    #     three numbers fifteen times, and at 41 bars the strings were wider than the bars, so
    #     adjacent labels overlapped into an unreadable smear. A number repeated on every bar is
    #     also the one number a reader does not need: what a rebuild DID is the change.
    #
    #     The "+" was wrong as well. `n_generated` and `n_source` are how many synthetic and mined
    #     rows are in the curriculum NOW, not what this iteration added, so "+900 mined" on fifteen
    #     bars in a row described a single 900-row rebuild as fifteen of them. Those totals are
    #     already written inside their own bands; the delta below is computed against the previous
    #     iteration and is a real addition.
    previous_total = None
    for position, (x, top, generated, mined) in enumerate(zip(xs, tops, gen, src)):
        if not top:
            continue
        delta = None if previous_total is None else top - previous_total
        # The total is worth restating when it changed, and on the first bar of the page so the
        # reader has an anchor even when the page opens mid-plateau.
        if position == 0 or delta:
            ax.annotate(
                f"{top:,}", xy=(x, top), xytext=(0, 11), textcoords="offset points",
                ha="center", fontsize=band_fontsize, color="0.25",
            )
        if delta:
            # Attribute the change to whichever source actually grew, so the colour says where the
            # rows came from without needing a fourth legend entry.
            grew_syn = generated > (records[position - 1]["n_generated"] if position else 0)
            ax.annotate(
                f"{delta:+,}", xy=(x, top), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=band_fontsize,
                color="#c05000" if grew_syn else "#6a3d9a",
                fontweight="bold",
            )
        previous_total = top
    _iteration_axis(ax, xs)
    ax.set_ylabel("rows")
    first, last = (xs[0], xs[-1]) if xs else (0, 0)
    ax.set_title(
        f"Dataset composition per iteration (rows by origin) — iterations {first}–{last}"
    )
    ax.grid(True, alpha=0.3, axis="y")
    # Bars run to the top of the axes, so reserve headroom rather than letting the legend sit
    # on top of the first stack.
    if peak:
        ax.set_ylim(0, peak * 1.30)
    ax.legend(fontsize=7, loc="upper left", ncol=1 if not multi_source else 2)


# Bands used to colour a per-label bar. Chosen so the eye lands on the problem: anything under half
# is red, and only near-ceiling classes are green.
def _score_colour(value: float) -> str:
    if value >= 0.9:
        return "#2ca02c"
    if value >= 0.7:
        return "#98c379"
    if value >= 0.5:
        return "#ff7f0e"
    return "#d62728"


def _plot_label_performance(ax, records, meta) -> None:
    """Correct vs failed eval rows at the final iteration, bucketed and worst-first.

    The difficulty chart answers "how hard were the rows it got wrong"; this answers "WHICH rows",
    which is the question an intervention is actually chosen against. For a task with classes the
    buckets are the classes; otherwise they are the task's own failure categories, with every correct
    row in one bucket. `surgical_synthesis` targets exactly these buckets, so this is the chart that
    says whether targeting worked.

    Counts, not rates, deliberately: a class at 50% on four rows and one at 50% on four hundred are
    the same rate and completely different problems, and the whole point of choosing a target is to
    spend the budget on the second one.
    """
    final = records[-1] if records else {}
    breakdown = [
        entry for entry in (final.get("outcome_breakdown") or [])
        if isinstance(entry, dict) and entry.get("bucket")
    ]
    if not breakdown:
        # Fall back to the confusion pairs, which older DAGs carry even without a breakdown.
        breakdown = [
            {"bucket": str(pair.get("gold")), "correct": 0,
             "failed": int(pair.get("count", 0) or 0)}
            for pair in (final.get("confusion_pairs") or [])
            if isinstance(pair, dict) and pair.get("gold")
        ]
    if not breakdown:
        ax.text(0.5, 0.5, "no per-label outcome breakdown recorded",
                ha="center", va="center", fontsize=10, color="0.4")
        ax.axis("off")
        return

    # WHICH BUCKETING IS THIS?
    #     Two shapes arrive here and they need different charts, which is what the old single
    #     rendering got wrong. A CLASSIFICATION task buckets by gold class, and a class genuinely
    #     holds a mix of correct and failed rows — so a stacked green/red bar and a hit rate are
    #     both meaningful. An OPEN-ENDED task (calendar, xlam, NER) buckets by the scorer's own
    #     FAILURE CATEGORY plus one bucket literally named `correct`, and there the split is
    #     degenerate by construction: a failure category is 0% correct because that is what makes
    #     it a failure category, and `correct` is 100% correct for the same reason.
    #
    #     Drawn as one stacked chart that produced a bar reading "wrong_arguments 0/66 (0%)" beside
    #     "correct 468/468 (100%)", which looks like a contradiction — a big red bar labelled 0% —
    #     and is really just the category's SIZE next to a rate that could never have been anything
    #     else. The numbers were right and the chart was not answering a question.
    category_mode = any(
        str(entry.get("bucket")).lower() == "correct" for entry in breakdown
    )
    total_rows = sum(
        int(e.get("correct", 0) or 0) + int(e.get("failed", 0) or 0) for e in breakdown
    )
    where = _final_eval_caption(records, meta)

    if category_mode:
        _plot_failure_categories(ax, breakdown, total_rows, where)
        return

    # A 151-class task cannot be read as 151 bars; show the worst 25 and say so. The list arrives
    # already sorted failures-first, so truncating keeps the informative end.
    shown, truncated = breakdown[:25], max(0, len(breakdown) - 25)
    names = [entry["bucket"] for entry in shown]
    correct = [int(entry.get("correct", 0) or 0) for entry in shown]
    failed = [int(entry.get("failed", 0) or 0) for entry in shown]
    positions = list(range(len(shown)))

    ax.barh(positions, correct, color="#2ca02c", label="correct")
    ax.barh(positions, failed, left=correct, color="#d62728", label="failed")
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("eval rows")
    ax.grid(True, alpha=0.3, axis="x")

    totals = [c + f for c, f in zip(correct, failed)]
    peak = max(totals) if totals else 0
    for index, (c, f, total) in enumerate(zip(correct, failed, totals)):
        if not total:
            continue
        rate = c / total
        ax.text(total + peak * 0.01, index, f"{c}/{total} ({rate:.0%})",
                va="center", fontsize=6, color="0.25")
    if peak:
        ax.set_xlim(0, peak * 1.22)
    title = f"Accuracy by gold class — {where}"
    if truncated:
        title += f" (worst 25 of {len(breakdown)})"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")


def _final_eval_caption(records, meta) -> str:
    """Which eval this chart is of: the iteration, the tier and the model.

    The chart is of ONE evaluation — the last iteration of the last tier — and said so nowhere, so
    a reader could not tell whether they were looking at the converged model or an early one.
    """
    final = records[-1] if records else {}
    selector = final.get("selector") or (meta or {}).get("selector") or "model"
    tier = final.get("tier")
    iteration = final.get("iteration")
    parts = []
    if iteration is not None:
        parts.append(f"final eval, iteration {iteration}")
    else:
        parts.append("final eval")
    parts.append(f"tier {tier}" if tier is not None else "tier ?")
    parts.append(str(selector))
    return ", ".join(parts)


def _plot_failure_categories(ax, breakdown, total_rows: int, where: str) -> None:
    """Failure categories by size, with the passing share stated rather than drawn as a peer bar.

    `correct` is not a failure category and is 20x the size of any of them, so plotting it in the
    same axis both dwarfs the bars that matter and invites reading the chart as a per-category hit
    rate. It becomes the caption instead: the bars are then all the same kind of thing — how many
    eval rows each way of being wrong cost — which is exactly what `surgical_synthesis` targets.
    """
    correct_rows = sum(
        int(e.get("correct", 0) or 0) for e in breakdown
        if str(e.get("bucket")).lower() == "correct"
    )
    failures = [
        (str(e["bucket"]), int(e.get("failed", 0) or 0))
        for e in breakdown
        if str(e.get("bucket")).lower() != "correct" and int(e.get("failed", 0) or 0)
    ]
    failures.sort(key=lambda item: -item[1])
    shown, truncated = failures[:25], max(0, len(failures) - 25)

    if not shown:
        rate = correct_rows / total_rows if total_rows else 0.0
        ax.text(0.5, 0.5, f"no failures recorded — {correct_rows:,}/{total_rows:,} correct "
                          f"({rate:.1%})",
                ha="center", va="center", fontsize=11, color="#2ca02c")
        ax.axis("off")
        ax.set_title(f"Failures by category — {where}", fontsize=10)
        return

    names = [name for name, _ in shown]
    counts = [count for _, count in shown]
    positions = list(range(len(shown)))
    ax.barh(positions, counts, height=0.6, color="#d62728")
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=8)
    # Keep room for at least five rows even when fewer categories fired. Otherwise a task whose
    # failures collapse to two categories draws them as two slabs filling the whole axis, which
    # reads as "enormous" rather than as "two thirteenths of the eval set".
    ax.set_ylim(-0.6, max(len(shown), 5) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("eval rows failed")
    ax.grid(True, alpha=0.3, axis="x")

    peak = max(counts)
    for index, count in enumerate(counts):
        share = f"  ({count / total_rows:.1%} of eval set)" if total_rows else ""
        ax.text(count + peak * 0.02, index, f"{count:,}{share}",
                va="center", fontsize=7, color="0.25")
    ax.set_xlim(0, peak * 1.35)

    failed_rows = sum(count for _, count in failures)
    accuracy = correct_rows / total_rows if total_rows else 0.0
    title = f"Failures by category — {where}"
    if truncated:
        title += f" (worst 25 of {len(failures)})"
    ax.set_title(title, fontsize=10)
    # The passing share as a caption, not a bar: it is the denominator these categories are carved
    # out of, and drawing it alongside them is what made the old chart unreadable.
    ax.annotate(
        f"{correct_rows:,} of {total_rows:,} eval rows correct ({accuracy:.1%});  "
        f"{failed_rows:,} failed across {len(failures)} categor"
        f"{'y' if len(failures) == 1 else 'ies'}",
        xy=(0.5, -0.13), xycoords="axes fraction", ha="center", fontsize=8, color="0.3",
    )


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

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    _plot_accuracy(axes[0][0], records, meta, stop_threshold)
    _plot_difficulty(axes[0][1], records, meta)
    _plot_label_performance(axes[0][2], records, meta)
    _plot_composition(axes[1][0], records)
    axes[1][1].axis("off")
    axes[1][2].axis("off")
    latest = records[-1] if records else {}
    thresholds = _threshold_series(records, stop_threshold)
    if thresholds and len(set(thresholds)) > 1:
        threshold_text = (
            f"{thresholds[0]:.3f} → {thresholds[-1]:.3f} "
            f"({len(set(thresholds)) - 1} change(s))"
        )
    elif thresholds:
        threshold_text = f"{thresholds[-1]:.3f}"
    else:
        threshold_text = "n/a" if stop_threshold is None else f"{stop_threshold:.3f}"
    note = [
        f"selector: {meta.get('selector') or 'n/a'}",
        f"iterations: {len(records)}",
        f"final {meta['metric_name']}: "
        + ("n/a" if not records or latest.get("score") is None else f"{latest['score']:.3f}"),
        f"baseline: "
        + ("n/a" if meta.get("baseline_f1") is None else f"{meta['baseline_f1']:.3f}"),
        f"threshold: {threshold_text}",
    ]
    axes[1][1].text(
        0.02, 0.98, "\n".join(note), va="top", ha="left", fontsize=11, family="monospace",
    )
    fig.suptitle(f"Run summary — {meta.get('selector') or ''}".strip(" —"), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_one(plot_fn, out_path: Path, *args, figsize=(9, 6), tight_bbox=False) -> None:
    """Render one chart to one file.

    ``tight_bbox`` is needed by any plot whose legend sits OUTSIDE the axes: `tight_layout` only
    accounts for artists inside them, so an anchored legend is silently clipped at the right edge
    of the canvas without it.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax, *args)
    if tight_bbox:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    else:
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_composition_pages(records, out: Path) -> list[Path]:
    """Write the composition chart, split into pages of `COMPOSITION_PAGE_SIZE` iterations.

    A 41-iteration run drawn as one image gave each bar about 20 pixels: too narrow for the row
    counts inside the bands, and far too narrow for the totals above them, so labels collided with
    their neighbours and the chart could not be read at any zoom. Paging is the fix that keeps the
    numbers horizontal and legible rather than shrinking or rotating them until they technically fit.

    Page 1 keeps the plain `dataset_composition.png` name so existing readers and the summary sheet
    are unaffected; later pages are suffixed with the iteration range they cover.
    """
    pages: list[Path] = []
    chunks = [
        records[i:i + COMPOSITION_PAGE_SIZE]
        for i in range(0, len(records), COMPOSITION_PAGE_SIZE)
    ] or [records]
    for index, chunk in enumerate(chunks):
        if index == 0:
            path = out / "dataset_composition.png"
        else:
            first = chunk[0]["global_idx"]
            last = chunk[-1]["global_idx"]
            path = out / f"dataset_composition_iters_{first}-{last}.png"
        # Width scales with the bar count so a short final page is not stretched into a few
        # enormous bars, and a full page still gets room per bar.
        width = max(7.0, min(14.0, 1.6 + 0.42 * len(chunk)))
        _plot_one(_plot_composition, path, chunk, figsize=(width, 6))
        pages.append(path)
    return pages


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

    Three views of the same run, because they answer different questions:
      ``accuracy.png`` etc.        every tier on one axis — where did the run go overall
      ``accuracy_tier<N>.png``     one chart per tier, from its first fine-tuned evaluation — what
                                   the search loop did on this model, labelled by intervention,
                                   with kept and rolled-back iterations distinguished
      ``tier<N>_<selector>/``      the full artifact set for one model, in depth
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

    # One accuracy chart PER TIER, at the top level beside the combined one. Separate from the
    # per-tier subdirectories below: those hold a full artifact set each, which is the right place
    # to study one model in depth but a poor place to compare tiers, because it means opening three
    # folders to see three curves. These sit together, are named for their tier, and each one is
    # anchored on its own base model — see `_plot_tier_accuracy`.
    written.extend(_tier_accuracy_pages(progression, out, stop_threshold))

    # Per tier. A run escalates through several models and each one has its own baseline,
    # difficulty profile and hypothesis chain; collapsing them onto shared axes hides the
    # trajectory of every model but the last. One subdirectory per tier keeps both views.
    for entry in _model_trajectories(progression):
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


def _model_trajectories(progression: list[dict]) -> list[dict]:
    """Progression entries that actually have a trajectory to draw.

    `downward_probe` entries are excluded because a probe runs one config and carries no DAG, and
    the post-convergence probe ORIGIN is excluded by the same emptiness test — it is a pointer back
    to a tier already graphed from its own entry, not a fourth tier.
    """
    return [
        entry for entry in progression
        if entry.get("kind") == "model_trajectory" and (entry.get("dag") or [])
    ]


def _tier_accuracy_pages(
    progression: list[dict],
    out: Path,
    stop_threshold: float | None,
) -> list[Path]:
    """Write ``accuracy_tier<N>.png``, one per model tier the run traversed."""
    written: list[Path] = []
    used: set[str] = set()
    trajectories = [
        (entry, *_iteration_records([entry])) for entry in _model_trajectories(progression)
    ]
    # ONE y-range for the whole run, computed before anything is drawn. Every tier's scores and
    # every tier's goal go into it, so the charts are zoomed to the data AND remain comparable with
    # each other — a height on tier 1 means the same score as the same height on tier 3.
    shared_ylim = _score_range([
        [r.get("score") for r in records] + (_threshold_series(records, stop_threshold) or [])
        for _, records, _ in trajectories
    ])
    for entry, records, meta in trajectories:
        if not records:
            continue
        # `_iteration_records` reports the LAST entry's identity in meta, which is this entry's
        # because it was passed alone — but tier is not among the fields it copies, and the chart
        # is titled by tier.
        meta = {**meta, "tier": entry.get("tier")}
        stem = f"accuracy_tier{entry.get('tier', 'x')}"
        # Tiers are unique across escalations (promotion is always to a strictly higher tier), so
        # this is a guard rather than an expectation. It exists because silently overwriting one
        # tier's chart with another's is indistinguishable, in the output directory, from a tier
        # that was never graphed.
        if stem in used:
            stem = f"{stem}_{_tier_dir_name(entry)}"
        used.add(stem)
        path = out / f"{stem}.png"
        # Wider than the 9x6 default and saved with a tight bounding box: the legend is anchored
        # outside the axes, so the extra width is what keeps the plot itself from being squeezed
        # into a strip once the legend takes its share.
        _plot_one(
            _plot_tier_accuracy, path, records, meta, stop_threshold, shared_ylim,
            figsize=(12, 6), tight_bbox=True,
        )
        written.append(path)
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

    written.extend(_plot_composition_pages(records, out))

    labels_png = out / "label_performance.png"
    _plot_one(_plot_label_performance, labels_png, records, meta)
    written.append(labels_png)

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
