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
                }
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

    assert [r["global_idx"] for r in records] == [0, 1, 2]
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
        "dataset_composition.png", "summary.png",
    }
    for path in written:
        assert path.is_file() and path.stat().st_size > 0

    hyp = (tmp_path / "graphics" / "hypotheses.md").read_text(encoding="utf-8")
    assert "add synthetic hard cases" in hyp
    assert "escalated to a bigger model" in hyp


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
