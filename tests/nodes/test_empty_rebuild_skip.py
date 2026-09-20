"""A rebuild that adds no rows must not cost a train+evaluate cycle.

WHY
    Curate's own log already said it: "the curriculum is unchanged at 3,900 row(s), so training this
    iteration would repeat the previous one exactly." It trained anyway. That happened twice in
    calendar run 39294409 and eight times in run 38566712, which spent 7h42m on them.

    The wasted GPU time is the smaller half. Training here is not deterministic — nothing seeds it
    and every iteration re-quantizes — so the repeat came back with a DIFFERENT score for an
    unchanged experiment, and that score entered the trajectory, the rollback decision and the
    attribution table as though it measured something. `08-23` §1.2 has three fits of identical data
    and identical hyperparameters scoring 0.0037, 0.0019 and 0.4598.
"""
from __future__ import annotations

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("EXA_API_KEY", "x")

from agent.graph import _route_after_curate, graph_topology_descriptor
from agent.nodes.curate import _record_skipped_iteration
from agent.run_health import _dag_rows, format_health_summary
from agent.state import SKIPPED_NO_ROWS


class TestRouting:
    def test_an_empty_rebuild_routes_back_to_iterate(self):
        assert _route_after_curate({"next_action": "iterate"}) == "iterate"

    def test_a_productive_rebuild_still_routes_to_train(self):
        assert _route_after_curate({}) == "train"
        assert _route_after_curate({"next_action": None}) == "train"

    def test_termination_still_wins(self):
        state = {"next_action": "iterate", "_graph_steps": 10**9}
        assert _route_after_curate(state) == "terminate"

    def test_the_declared_topology_carries_the_new_edge(self):
        """The descriptor is hashed into the checkpoint's topology fingerprint, so an edge added to
        the compiled graph and not to the descriptor makes every resume compare against a graph
        that no longer exists."""
        curate = graph_topology_descriptor("cold_start")["conditional_edges"]["curate"]
        assert curate["iterate"] == "iterate"
        assert curate["train"] == "train"


class TestTheSkippedIterationIsRecorded:
    @staticmethod
    def _state():
        state = {"iteration": 4, "dag": [], "dataset_version": 3,
                 "current_dataset_path": "/d.jsonl", "last_hypothesis": "mine more rows",
                 "selected_model": None}
        _record_skipped_iteration(state, model_id="m", plan={"strategy": "mine_new_real"},
                                  curriculum=3900)
        return state

    def test_it_lands_in_the_dag(self):
        """The orchestrator reads the DAG as its own history. An attempt that left no trace is one
        it will propose again verbatim."""
        assert len(self._state()["dag"]) == 1

    def test_it_is_marked(self):
        assert self._state()["dag"][0]["status"] == SKIPPED_NO_ROWS

    def test_it_carries_no_score(self):
        """There was no evaluation. Carrying the previous score forward would put a fabricated
        measurement into the series that rollback and stagnation read."""
        assert self._state()["dag"][0]["score"] is None

    def test_it_is_not_marked_as_rolled_back(self):
        """Nothing was trained, so there is no checkpoint that lost to anything."""
        assert self._state()["dag"][0]["pruned"] is False

    def test_it_keeps_the_plan_that_produced_nothing(self):
        node = self._state()["dag"][0]
        assert node["pi"]["D"]["plan"]["strategy"] == "mine_new_real"


class TestReporting:
    @staticmethod
    def _rows():
        state = {
            "escalation_history": [], "selected_model": None, "run_health": {"history": []},
            "dag": [
                {"iteration": 1, "score": 0.50, "model": "m", "tier": 1, "pruned": False,
                 "intervention": "data_rebuild",
                 "pi": {"D": {"plan": {"strategy": "mine_new_real"}}}},
                {"iteration": 2, "score": None, "model": "m", "tier": 1, "pruned": False,
                 "status": SKIPPED_NO_ROWS, "intervention": "data_rebuild",
                 "pi": {"D": {"plan": {"strategy": "mine_new_real"}}}},
            ],
        }
        return state, _dag_rows(state)

    def test_the_skipped_row_is_flagged(self):
        _, rows = self._rows()
        assert rows[1]["skipped"] is True
        assert rows[0]["skipped"] is False

    def test_the_table_says_skipped_not_rolled_back(self):
        state, _ = self._rows()
        body = "\n".join(format_health_summary(state))
        assert "skipped (added no rows)" in body

    def test_a_skipped_row_contributes_no_score_change(self):
        from agent.run_health import _attribute

        _, rows = self._rows()
        _, contributions = _attribute(rows)
        assert contributions["mine_new_real"]["gain"] == 0.0
