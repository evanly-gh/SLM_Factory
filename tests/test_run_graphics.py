"""Tests for post-run summary graphics (agent/run_graphics.py)."""
import json

import pytest

from agent import run_graphics


def _node(iteration, score, easy, medium, hard, comp, hypothesis, intervention, selector="tierA"):
    """A DAG node with just the fields the grapher reads."""
    return {
        "iteration": iteration,
        "selector": selector,
        "score": score,
        "intervention": intervention,
        "hypothesis": hypothesis,
        "evaluation_state": {
            "last_eval": {"metric": "macro_f1", "per_class": {"pos": score}},
            "test_report": {
                "by_difficulty": {
                    "easy": {"n": 10, "accuracy": easy},
                    "medium": {"n": 8, "accuracy": medium},
                    "hard": {"n": 6, "accuracy": hard},
                },
                # What `label_performance.png` is drawn from. Supplied so the chart takes its real
                # branch rather than the "nothing recorded" placeholder every other fixture here
                # would give it.
                "outcome_breakdown": [
                    {"bucket": "neg", "correct": 4, "failed": 6},
                    {"bucket": "pos", "correct": 9, "failed": 1},
                ],
            },
        },
        "pi": {"S": {"task_type": "classification"}, "D": {"composition": comp}},
    }


def _comp(gold, generated, source):
    return {
        "n_gold": gold,
        "n_hard_generated": generated,
        "n_hard_source": source,
        "total_examples": gold + generated + source,
    }


def _two_tier_progression():
    tier_a = [
        _node(0, 0.40, 0.6, 0.4, 0.1, _comp(100, 0, 0), "start", "acquire", "tierA"),
        _node(1, 0.55, 0.7, 0.5, 0.2, _comp(100, 40, 0), "add synthetic hard cases", "synthesize", "tierA"),
    ]
    tier_b = [
        _node(0, 0.72, 0.85, 0.7, 0.4, _comp(120, 40, 30), "escalated to a bigger model", "escalate", "tierB"),
    ]
    return [
        {"kind": "model_trajectory", "selector": "tierA", "tier": 1,
         "baseline_f1": 0.30, "dag": tier_a},
        {"kind": "model_trajectory", "selector": "tierB", "tier": 2,
         "baseline_f1": 0.45, "dag": tier_b},
    ]


def test_iteration_records_maps_fields_and_tier_boundary():
    records, meta = run_graphics._iteration_records(_two_tier_progression())

    # 1-based, matching how iterations are numbered in the logs, DAG and curation entries.
    assert [r["global_idx"] for r in records] == [1, 2, 3]
    assert [r["score"] for r in records] == [0.40, 0.55, 0.72]
    # Composition folded from pi.D.composition.
    assert records[1]["n_generated"] == 40
    assert records[2]["n_source"] == 30
    assert records[2]["total"] == 190
    # Difficulty carried through.
    assert records[0]["by_difficulty"]["hard"]["accuracy"] == 0.1
    # Hypothesis + intervention prose carried through.
    assert records[1]["hypothesis"] == "add synthetic hard cases"
    assert records[2]["intervention"] == "escalate"
    # Escalation to the second trajectory marks a tier boundary at its first record.
    assert meta["tier_boundaries"] == [2]
    # Metric + final-model metadata describe the last trajectory.
    assert meta["metric_name"] == "macro_f1"
    assert meta["baseline_f1"] == 0.45
    assert meta["selector"] == "tierB"


def test_generate_from_state_writes_all_artifacts(tmp_path):
    # The state path graphs every tier via build_run_progression. Provide escalation_history
    # (tierA) + a live final model (tierB) so both trajectories are present.
    class _Model:
        selector = "tierB"
        model_id = "org/tierB"
        quant = None
        tier = 2

    prog = _two_tier_progression()
    state = {
        "escalation_history": [prog[0]],
        "selected_model": _Model(),
        "best_score": 0.72,
        "iteration": 1,
        "scores": [0.72],
        "dag": prog[1]["dag"],
        "stop_threshold": 0.70,
    }
    baselines = [{"selector": "tierB", "baseline_f1": 0.45}]

    written = run_graphics.generate_run_graphics(
        tmp_path, state=state, baselines=baselines, out_dir=tmp_path / "graphics"
    )
    names = {p.name for p in written}
    assert names == {
        "hypotheses.md", "accuracy.png", "difficulty.png",
        "dataset_composition.png", "label_performance.png", "summary.png",
        # One baseline-anchored accuracy chart per tier, at the top level. The rest of the names
        # collapse because each tier subdirectory repeats them; these do not, by design — they are
        # meant to sit side by side so tiers can be compared without opening three folders.
        "accuracy_tier1.png", "accuracy_tier2.png",
    }
    for path in written:
        assert path.is_file() and path.stat().st_size > 0

    hyp = (tmp_path / "graphics" / "hypotheses.md").read_text(encoding="utf-8")
    assert "add synthetic hard cases" in hyp
    assert "escalated to a bigger model" in hyp


def test_label_performance_is_written_even_with_no_breakdown_recorded(tmp_path):
    """An artifact that appears only on some runs is one a reader cannot trust the absence of.

    Runs from before `outcome_breakdown` existed, and any run whose last iteration never produced a
    test report, still get the file — with the placeholder text inside it rather than no file at
    all, so "the chart is missing" always means the grapher failed.
    """
    dag = _two_tier_progression()[0]["dag"]
    for node in dag:
        node["evaluation_state"]["test_report"].pop("outcome_breakdown")

    written = run_graphics.generate_run_graphics(
        tmp_path,
        state={
            "escalation_history": [],
            "selected_model": type("M", (), {"selector": "tierA", "model_id": "org/tierA",
                                             "quant": None, "tier": 1})(),
            "best_score": 0.55, "iteration": 1, "scores": [0.55], "dag": dag,
        },
        baselines=[], out_dir=tmp_path / "g",
    )
    labels_png = tmp_path / "g" / "label_performance.png"
    assert labels_png in written
    assert labels_png.stat().st_size > 0


def test_generate_from_run_dir_reads_disk_json(tmp_path):
    run_dir = tmp_path / "logs" / "runs" / "20260803_120000_1234"
    run_dir.mkdir(parents=True)
    dag = _two_tier_progression()[0]["dag"]  # single-tier final DAG on disk
    (run_dir / "dag.json").write_text(json.dumps(dag), encoding="utf-8")
    (run_dir / "scores.json").write_text(json.dumps({"stop_threshold": 0.7}), encoding="utf-8")
    (run_dir / "baselines.json").write_text(
        json.dumps([{"selector": "tierA", "baseline_f1": 0.3}]), encoding="utf-8"
    )

    written = run_graphics.generate_run_graphics(run_dir)

    # Default output location: logs/graphics/<run_id>/.
    out = tmp_path / "logs" / "graphics" / "20260803_120000_1234"
    assert out.is_dir()
    assert {p.name for p in written} >= {"hypotheses.md", "accuracy.png"}
    assert (out / "accuracy.png").stat().st_size > 0


def test_empty_run_writes_only_hypotheses(tmp_path):
    # A run with no DAG (e.g. crashed before the first eval) must not crash the grapher; it
    # writes hypotheses.md so the folder is never silently blank, and no charts.
    written = run_graphics.generate_run_graphics(
        tmp_path, state={"dag": [], "selected_model": None}, baselines=[],
        out_dir=tmp_path / "g",
    )
    assert [p.name for p in written] == ["hypotheses.md"]
    assert (tmp_path / "g" / "hypotheses.md").is_file()


class TestPerTierAccuracyCharts:
    """`accuracy_tier<N>.png` — one chart per tier, from its first fine-tuned evaluation.

    The run-wide `accuracy.png` puts every tier on one axis, which hides the trajectory of every
    model but the last, and draws one line through all iterations, which hides that most of them
    were rolled back (19 of 26 on run 39311801's tier 1). These charts are per tier, label each
    iteration with the intervention that produced it, and draw the search path and the accepted
    path separately.
    """

    def _written(self, tmp_path, progression, stop_threshold=0.7):
        return run_graphics._tier_accuracy_pages(
            progression, tmp_path, stop_threshold
        )

    def test_one_chart_per_tier_named_for_its_tier(self, tmp_path):
        written = self._written(tmp_path, _two_tier_progression())
        assert [p.name for p in written] == ["accuracy_tier1.png", "accuracy_tier2.png"]
        for path in written:
            assert path.stat().st_size > 0

    def test_a_tier_with_no_dag_is_skipped(self, tmp_path):
        """An entry with nothing to draw must produce no file, not an empty one."""
        progression = _two_tier_progression()
        progression[1]["dag"] = []
        assert [p.name for p in self._written(tmp_path, progression)] == ["accuracy_tier1.png"]

    def test_downward_probes_are_not_charted_as_tiers(self, tmp_path):
        """A probe runs one config and carries no DAG; it is not a fourth tier."""
        progression = _two_tier_progression() + [
            {"kind": "downward_probe", "selector": "probe", "tier": 1,
             "baseline_f1": None, "dag": []},
        ]
        assert len(self._written(tmp_path, progression)) == 2

    @pytest.mark.parametrize("baseline", [None, 0.0, 0.45])
    def test_the_base_model_is_never_plotted(self, tmp_path, baseline):
        """The chart starts at the first fine-tuned evaluation, whatever the baseline is.

        The zero-shot step is real and large — 0.0000 to 0.6885 on run 39311801's tier 1 — but it
        is also the whole y-axis, and including it squeezed the iteration trajectory this chart
        exists to show into the top fifth of the plot. Asserted across an unmeasured baseline, a
        measured zero and a mid-range one so no value can reintroduce an x=0 point.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        progression = _two_tier_progression()[:1]
        progression[0]["baseline_f1"] = baseline
        records, meta = run_graphics._iteration_records(progression)
        fig, ax = plt.subplots()
        run_graphics._plot_tier_accuracy(
            ax, records, {**meta, "tier": 1}, 0.7,
        )
        # Only marker-bearing artists are inspected. `axhline` reports xdata (0, 1) because it is
        # in AXES coordinates, not data coordinates, so scanning every line would flag the goal
        # line as a base-model point that is not there.
        marker_xs = [
            x
            for line in ax.get_lines()
            if line.get_marker() not in (None, "None", "", " ")
            for x in line.get_xdata()
        ]
        xlim_left = ax.get_xlim()[0]
        plt.close(fig)
        assert marker_xs, "no data markers were plotted"
        assert min(marker_xs) >= 1, f"a marker was drawn at x={min(marker_xs)}"
        # The axis itself must not reserve the slot either.
        assert xlim_left > 0, f"the x-axis still opens at {xlim_left}, leaving room for x=0"

    def test_an_unmeasured_baseline_still_renders(self, tmp_path):
        """`baseline_f1=None` is now simply unread, so it cannot break the chart."""
        progression = _two_tier_progression()[:1]
        progression[0]["baseline_f1"] = None
        written = self._written(tmp_path, progression)
        assert written[0].stat().st_size > 0

    def test_tier_number_reaches_the_plot(self, tmp_path, monkeypatch):
        """`_iteration_records` does not copy `tier` into meta, and the chart is titled by it."""
        seen = []
        monkeypatch.setattr(
            run_graphics, "_plot_tier_accuracy",
            lambda ax, records, meta, thr, ylim=None: seen.append(meta.get("tier")),
        )
        self._written(tmp_path, _two_tier_progression())
        assert seen == [1, 2]

    def test_colliding_tier_numbers_do_not_overwrite_each_other(self, tmp_path):
        """Two entries at one tier must yield two files, not one silently clobbered."""
        progression = _two_tier_progression()
        progression[1]["tier"] = 1
        written = self._written(tmp_path, progression)
        assert len(written) == 2
        assert len({p.name for p in written}) == 2
        assert written[0].name == "accuracy_tier1.png"


class TestInterventionLabelling:
    """Which intervention produced an iteration — the label under each x tick.

    `intervention` decides, and the rebuild plan only refines a `data_rebuild`. Reading the plan
    first is the bug this class exists to prevent; see `test_a_stale_plan_does_not_relabel...`.
    """

    def _record(self, intervention, substrategy="", composition_strategy=""):
        return {
            "intervention": intervention,
            "substrategy": substrategy,
            "composition_strategy": composition_strategy,
        }

    def test_a_rebuild_is_named_by_its_plan(self):
        for strategy in ("mine_new_real", "surgical_synthesis"):
            assert run_graphics._intervention_kind(
                self._record("data_rebuild", substrategy=strategy)
            ) == strategy

    def test_the_first_build_is_named_by_its_composition(self):
        """Iteration 1 has no plan — there is nothing to add to yet — so it names itself."""
        assert run_graphics._intervention_kind(
            self._record("data_rebuild", composition_strategy="initial_gold")
        ) == "initial_gold"

    def test_a_stale_plan_does_not_relabel_a_hyperparameter_iteration(self):
        """THE REGRESSION. `curate_node` skips on a hyperparameter iteration ("dataset held
        fixed"), which leaves `pi.D.plan` and `pi.D.composition` holding the PREVIOUS rebuild's
        values. Every one of run 39311801's tier-3 nodes carries plan.strategy=surgical_synthesis,
        including the four the run log records as hyperparameter — so refining by the plan
        unconditionally reported that tier as 9-of-9 synthesis when it was 5-of-9, and turned
        tier 1's hyperparameter iterations into `initial_gold`.
        """
        stale = self._record(
            "hyperparameter",
            substrategy="surgical_synthesis",
            composition_strategy="surgical_synthesis",
        )
        assert run_graphics._intervention_kind(stale) == "hyperparameter"

    def test_an_unusable_rebuild_plan_is_unknown_not_mislabelled(self):
        assert run_graphics._intervention_kind(self._record("data_rebuild")) == ""
        assert run_graphics._intervention_style(self._record("data_rebuild"))["code"] == "?"

    def test_every_style_has_a_distinct_code_marker_and_colour(self):
        styles = [*run_graphics._INTERVENTION_STYLE.values(), run_graphics._INTERVENTION_UNKNOWN]
        for field in ("code", "marker", "color"):
            values = [s[field] for s in styles]
            assert len(set(values)) == len(values), f"duplicate {field}: {values}"

    def test_the_code_is_written_under_each_iteration_tick(self):
        """The label rides on the tick, where 26 of them cannot collide with the data."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        records = [
            self._record("data_rebuild", composition_strategy="initial_gold"),
            self._record("data_rebuild", substrategy="mine_new_real"),
            self._record("hyperparameter", substrategy="mine_new_real"),
        ]
        fig, ax = plt.subplots()
        run_graphics._tier_accuracy_axis(ax, [1, 2, 3], records)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        plt.close(fig)
        # No "base" slot: the axis begins at the first fine-tuned iteration.
        assert labels == ["1\nG", "2\nM", "3\nH"]


class TestKeptAndSearchLines:
    """Two lines: dotted through every iteration, solid through the kept ones only."""

    def _lines(self, records, baseline=0.3):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        run_graphics._plot_tier_accuracy(
            ax, records,
            {"metric_name": "macro_f1", "baseline_f1": baseline, "selector": "m", "tier": 1},
            0.9,
        )
        styles = {}
        for line in ax.get_lines():
            if line.get_linestyle() in (":", "-") and len(line.get_xdata()) > 1:
                styles.setdefault(line.get_linestyle(), []).append(
                    list(zip(line.get_xdata(), line.get_ydata()))
                )
        plt.close(fig)
        return styles

    def _recs(self, scores_and_pruned):
        return [
            {"iteration": i, "score": score, "pruned": pruned, "stop_threshold": 0.9,
             "intervention": "hyperparameter", "substrategy": "", "composition_strategy": ""}
            for i, (score, pruned) in enumerate(scores_and_pruned, start=1)
        ]

    def test_dotted_covers_every_iteration_and_solid_only_the_kept_ones(self):
        records = self._recs([(0.40, False), (0.30, True), (0.55, False), (0.20, True)])
        styles = self._lines(records)
        dotted = max(styles[":"], key=len)
        assert [p[0] for p in dotted] == [1, 2, 3, 4]
        solid = max(styles["-"], key=len)
        # Only iterations 1 and 3 survived rollback, so the accepted state moved twice.
        assert [p[0] for p in solid] == [1, 3]
        assert [p[1] for p in solid] == [0.40, 0.55]

    def test_no_solid_line_when_a_single_iteration_was_kept(self):
        """A one-point trajectory is a point, not a line; drawing it would imply a segment."""
        records = self._recs([(0.40, False), (0.30, True)])
        styles = self._lines(records)
        assert all(len(seg) != 1 for seg in styles.get("-", []))


class TestSharedYRange:
    """The per-tier charts are zoomed to the data but share ONE range across the run.

    Zooming matters because a fixed 0-1 axis wastes three quarters of the plot on this task (run
    39311801's tier 1 spans 0.5674-0.7694) and flattens the trajectory the chart exists to show.
    Sharing the range matters because the charts are read side by side: if each tier autoscaled
    independently, tier 3's flat band near 0.85 and tier 1's climb to 0.77 would occupy the same
    vertical space and look like the same result.
    """

    def test_the_margin_is_five_percent_of_the_extremes(self):
        low, high = run_graphics._score_range([[0.40, 0.55, 0.72]])
        assert low == pytest.approx(0.40 * 0.95)
        assert high == pytest.approx(0.72 * 1.05)

    def test_the_goal_raises_the_ceiling_when_no_iteration_reached_it(self):
        """A tier that fell short must still show the bar it fell short of."""
        scores, goal = [0.60, 0.71], 0.80
        low, high = run_graphics._score_range([scores + [goal]])
        assert high == pytest.approx(0.80 * 1.05)
        assert high > 0.80, "the goal line would be clipped off the top of its own chart"

    def test_the_range_never_leaves_zero_to_one(self):
        """A metric cannot exceed 1.0, and an axis implying headroom above it misleads."""
        low, high = run_graphics._score_range([[0.0, 0.99]])
        assert (low, high) == (0.0, 1.0)

    def test_identical_values_still_produce_a_renderable_axis(self):
        """matplotlib cannot draw a zero-height axis; a one-iteration tier must not crash."""
        low, high = run_graphics._score_range([[0.85]])
        assert high > low

    def test_no_scores_falls_back_to_the_full_axis(self):
        assert run_graphics._score_range([[None, float("nan")]]) == (0.0, 1.0)
        assert run_graphics._score_range([]) == (0.0, 1.0)

    def test_every_tier_chart_gets_the_same_range(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        seen = []

        def _capture(ax, records, meta, stop_threshold, ylim=None):
            seen.append(ylim)

        original = run_graphics._plot_tier_accuracy
        try:
            run_graphics._plot_tier_accuracy = _capture
            run_graphics._tier_accuracy_pages(_two_tier_progression(), tmp_path, 0.7)
        finally:
            run_graphics._plot_tier_accuracy = original
        assert len(seen) == 2
        assert seen[0] == seen[1], f"tiers got different ranges: {seen}"
        assert seen[0] is not None

    def test_the_shared_range_covers_the_lowest_and_highest_tier(self):
        """Computed across ALL tiers, so neither is clipped."""
        prog = _two_tier_progression()          # tierA 0.40-0.55, tierB 0.72
        sets = [
            [r.get("score") for r in run_graphics._iteration_records([e])[0]]
            for e in prog
        ]
        low, high = run_graphics._score_range(sets)
        assert low < 0.40, "tier A's floor is clipped"
        assert high > 0.72, "tier B's ceiling is clipped"


class TestTierIterationAxis:
    def test_stored_iteration_numbers_are_used_when_clean(self):
        records = [{"iteration": 1}, {"iteration": 2}, {"iteration": 3}]
        assert run_graphics._tier_iteration_values(records) == [1, 2, 3]

    @pytest.mark.parametrize("stored", [
        [0, 1, 2],        # 0-based, as the older DAGs wrote it
        [1, 1, 2],        # duplicated — would draw a line doubling back on itself
        [3, 1, 2],        # out of order
        [None, 2, 3],     # missing
    ])
    def test_unusable_numbering_falls_back_to_position(self, stored):
        records = [{"iteration": v} for v in stored]
        assert run_graphics._tier_iteration_values(records) == [1, 2, 3]


def test_pruned_flag_is_carried_into_the_records(tmp_path):
    """The per-tier chart draws kept and rolled-back iterations differently, so it needs the flag.

    A rolled-back iteration still cost a full train+eval cycle and belongs on the chart, but it
    contributed nothing to the tier's result — on 39311801's tier 3, 6 of 9 were reverted.
    """
    progression = _two_tier_progression()
    progression[0]["dag"][0]["pruned"] = True
    records, _ = run_graphics._iteration_records(progression)
    assert [r["pruned"] for r in records] == [True, False, False]


def test_plot_error_propagates_to_caller(tmp_path, monkeypatch):
    # The driver wraps generate_run_graphics in try/except; verify a plotting failure surfaces
    # as an exception here (so that wrapper is what makes it non-fatal, by design).
    monkeypatch.setattr(
        run_graphics, "_plot_one",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        run_graphics.generate_run_graphics(
            tmp_path, state={
                "escalation_history": [],
                "selected_model": type("M", (), {"selector": "tierA", "model_id": "x",
                                                  "quant": None, "tier": 1})(),
                "best_score": 0.4, "iteration": 0, "scores": [0.4],
                "dag": _two_tier_progression()[0]["dag"],
            },
            baselines=[], out_dir=tmp_path / "g",
        )
