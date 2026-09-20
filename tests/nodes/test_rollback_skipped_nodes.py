"""`rollback_node` against a DAG containing SKIPPED (scoreless) iterations.

THE BUG THIS EXISTS FOR (2026-09-04). `curate._record_skipped_iteration` puts a node in the DAG for
a rebuild that added no rows, deliberately with `score: None` — nothing was trained or evaluated, and
inventing a score would put a fabricated measurement into the trajectory that rollback and stagnation
both read. Its docstring names the consumers taught to skip it: `_dag_rows` and `agent.nodes.iterate`.

There were three. `rollback_node` filtered on `pruned` alone, and a skipped node is not pruned — it
was never rejected, it was never measured — so it reached

    best_node = max(non_pruned, key=lambda n: n["score"])

and comparing None against a float raised `TypeError`. It also carries no `weights_ref` key at all,
so the next line would `KeyError` even with the score handled.

That killed all three ablation runs of 2026-09-04 (39562028/29/30) at tier 1 iteration 8, costing
5h40m of L40S time between them. BC5CDR guarantees the shape: the local bundle holds 5,096 train rows
against the 5,000 `eval_setup` takes, so `mine_new_real` yields +96 once and +0 for ever after —
two consecutive skipped iterations, then the first regression, then rollback.

The feature shipped with two of three consumers updated and no test putting a skipped node in front
of the third, which is exactly why it looked finished.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.nodes.rollback import rollback_node, should_rollback
from agent.state import SKIPPED_NO_ROWS


def _measured(iteration, score, *, pruned=False):
    return {
        "iteration": iteration,
        "score": score,
        "pruned": pruned,
        "weights_ref": f"weights-{iteration}",
        "best_config": {"lora_rank": 16},
        "intervention": "hyperparameter",
        "pi": {"D": {"path": f"artifacts/dataset_v{iteration}.jsonl", "version": iteration}},
    }


def _skipped(iteration):
    """Exactly what `curate._record_skipped_iteration` writes: no score, no weights, not pruned."""
    return {
        "iteration": iteration,
        "status": SKIPPED_NO_ROWS,
        "score": None,
        "pruned": False,
        "intervention": "data_rebuild",
        "pi": {"D": {"path": "artifacts/dataset_v1.jsonl", "version": 1}},
    }


def _state(dag, scores):
    return {
        "dag": dag,
        "scores": list(scores),
        "selected_model": SimpleNamespace(label="tierX", selector="org/m@Q4_K_M"),
        "best_weights_ref": None,
        "best_score": 0.0,
        "current_dataset_path": "artifacts/dataset_v9.jsonl",
        "dataset_version": 9,
        "last_curation": None,
        "data_rebuild_plan": None,
    }


class TestRollbackSurvivesSkippedNodes:
    def test_the_exact_shape_that_killed_the_ablation_runs(self):
        """Two consecutive skipped iterations, then a regression. This used to raise TypeError."""
        dag = [
            _measured(1, 0.6541),
            _measured(2, 0.6243, pruned=True),
            _measured(3, 0.6113, pruned=True),
            _measured(4, 0.6380, pruned=True),
            _measured(5, 0.5458, pruned=True),
            _skipped(6),
            _skipped(7),
            _measured(8, 0.4105),
        ]
        state = _state(dag, [0.6541, 0.4105])
        assert should_rollback(state) is True

        out = rollback_node(state)

        # Rolled back to the only measured, unpruned node — iteration 1.
        assert out["best_score"] == pytest.approx(0.6541)
        assert out["best_weights_ref"] == "weights-1"
        assert dag[-1]["pruned"] is True, "the regressing node must be pruned"

    def test_a_skipped_node_is_never_chosen_as_the_rollback_target(self):
        """Even when it is the most recent unpruned node, it has no weights to restore."""
        dag = [_measured(1, 0.70), _skipped(2), _measured(3, 0.55)]
        out = rollback_node(_state(dag, [0.70, 0.55]))
        assert out["best_weights_ref"] == "weights-1"
        assert out["best_score"] == pytest.approx(0.70)

    def test_no_restorable_node_raises_a_message_naming_BOTH_reasons(self):
        """Nothing measured survives, so there is genuinely nothing to restore.

        Raising here is the pre-existing intended behaviour and is correct — a rollback with no
        checkpoint to roll back to is an inconsistent state, not something to paper over. What the
        fix changes is the DIAGNOSIS: the old message said "all DAG nodes pruned", which is false
        for a DAG that is mostly unmeasured, and would have sent the next reader looking for a
        pruning bug that does not exist.
        """
        dag = [_skipped(1), _skipped(2), _measured(3, 0.40)]
        with pytest.raises(RuntimeError) as excinfo:
            rollback_node(_state(dag, [0.55, 0.40]))
        message = str(excinfo.value)
        assert "1 of 3 are pruned" in message
        assert "2 were never evaluated" in message

    def test_the_pruning_log_line_handles_a_scoreless_node(self, capsys):
        """The log formats `score` too; an unmeasured node must not raise inside a print."""
        dag = [_measured(1, 0.70), _skipped(2)]
        rollback_node(_state(dag, [0.70, 0.55]))
        assert "unmeasured" in capsys.readouterr().out

    def test_no_rollback_leaves_the_dag_untouched(self):
        """Guard against the fix changing behaviour when there is nothing to roll back."""
        dag = [_measured(1, 0.60), _skipped(2), _measured(3, 0.75)]
        out = rollback_node(_state(dag, [0.60, 0.75]))
        assert [n.get("pruned") for n in out["dag"]] == [False, False, False]
        assert out["best_weights_ref"] is None


class TestNoReportSiteFormatsAScoreUnsafely:
    """No reporting site may apply a numeric format spec to `.get('score', <number>)`.

    That expression is the same bug three times over. `dict.get(key, default)` does not substitute
    the default when the key exists holding None, and `curate._record_skipped_iteration` stores
    `score: None` for a rebuild that added no rows. Found and fixed at, in order:

      1. `agent/nodes/rollback.py`        — killed three ablation runs mid-flight (2026-09-04)
      2. `tests/pipeline/run.py` dag_summary writer — masked (1)'s traceback with its own
      3. `tests/pipeline/run.py` DAG-traversal table — killed the FINAL REPORT of all three
         re-run ablations (2026-09-05) after they had otherwise completed, exit 0, ~19h of GPU

    Each was fixed individually and the next one appeared anyway, so this scans the source instead
    of trusting review. A grep-style test is the right shape here: the failure mode is a *textual*
    pattern that is invisible until a scoreless node reaches it at runtime, which on BC5CDR happens
    only after mining exhausts, hours in.
    """

    PATTERN = re.compile(
        r"\{[^{}]*\.get\((['\"])score\1\s*,\s*[-\d.]+\s*\)[^{}]*:[^{}]*[fd]\}"
    )
    FILES = (
        "tests/pipeline/run.py",
        "agent/pipeline_status.py",
        "agent/run_health.py",
        "agent/run_graphics.py",
        "agent/nodes/rollback.py",
        "agent/nodes/evaluate.py",
        "agent/nodes/iterate.py",
        "agent/nodes/curate.py",
    )

    def test_no_source_file_formats_a_defaulted_score(self):
        root = Path(__file__).resolve().parents[2]
        offenders = []
        for name in self.FILES:
            path = root / name
            if not path.is_file():
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
                stripped = line.strip()
                # Comments are skipped: the fixes above quote the broken pattern in their own
                # explanatory comments, and flagging those would make the test unfixable.
                if stripped.startswith("#"):
                    continue
                for match in self.PATTERN.finditer(line):
                    offenders.append(f"{name}:{lineno}: {match.group(0)}")
        assert not offenders, (
            "a numeric format spec is applied to a defaulted `score` lookup, which raises "
            "TypeError on a skipped (scoreless) DAG node. Use "
            "`f'{v:.4f}' if isinstance(v, (int, float)) else '<n/a>'` instead:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_pattern_the_scan_looks_for_really_does_raise(self):
        """Guard the guard: if this stops raising, the scan above is testing nothing."""
        node = {"score": None}
        with pytest.raises(TypeError, match="unsupported format string"):
            f"{node.get('score', 0):8.4f}"


class TestDagSummaryWriterIsNoneSafe:
    """The second bug: the partial-report writer crashed on the same node and hid the first.

    `f"{node.get('score', 0):.4f}"` looks safe and is not — `dict.get` does not substitute the
    default when the key exists holding None. Because it ran inside the PARTIAL-REPORT path, its
    own TypeError replaced the real traceback, so the logs blamed a formatting error in the error
    reporter instead of naming the scoreless node that reached rollback.
    """

    def test_get_with_a_default_does_not_rescue_an_explicit_none(self):
        node = {"iteration": 6, "score": None}
        assert node.get("score", 0) is None
        with pytest.raises(TypeError, match="unsupported format string"):
            f"{node.get('score', 0):.4f}"

    def test_the_none_safe_formatting_the_writer_now_uses(self):
        for node, expected in (
            ({"iteration": 6, "score": None}, "  n/a "),
            ({"iteration": 6, "score": 0.4105}, "0.4105"),
        ):
            score = node.get("score")
            assert (f"{score:.4f}" if isinstance(score, (int, float)) else "  n/a ") == expected
