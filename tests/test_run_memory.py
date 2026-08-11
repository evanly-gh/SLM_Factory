# tests/test_run_memory.py
"""
Run memory replaces the raw data-curation.md dump in the orchestrator prompt.

The trajectory it must reproduce is slm-clinc150-cse-38155022: two early improvements, then
seventeen consecutive failures dominated by data_rebuild/synthesize, which the old dump
presented as seventeen indistinguishable rows.
"""
from agent.run_memory import MAX_DETAILED_FAILURES, build_run_memory


def _node(iteration, score, intervention="data_rebuild", strategy="synthesize", hypothesis=""):
    return {
        "iteration": iteration,
        "score": score,
        "intervention": intervention,
        "hypothesis": hypothesis,
        "pi": {"D": {"version": 2, "plan": {"strategy": strategy} if strategy else None}},
        "evaluation_state": {"test_report": {}},
    }


def _clinc_like_dag():
    """Two improvements, then a long failing tail — the real run's shape."""
    dag = [
        _node(1, 0.8261, "data_rebuild", "acquire", "first trained model"),
        _node(2, 0.8734, "hyperparameter", None, "raise lora_rank 16 -> 32"),
    ]
    dag += [_node(i, 0.60 + i * 0.001, "data_rebuild", "synthesize", f"attempt {i}")
            for i in range(3, 18)]
    dag += [_node(i, 0.55, "data_rebuild", "acquire", f"attempt {i}") for i in range(18, 20)]
    return dag


class TestOutcomesAreDistinguishable:
    """The central defect: a new best and a rollback used to render identically."""

    def test_improvements_are_listed_as_kept(self):
        memory = build_run_memory({"dag": _clinc_like_dag()})
        assert "WHAT WORKED" in memory
        assert "iter 1" in memory and "iter 2" in memory

    def test_most_recent_iteration_is_marked_rolled_back(self):
        memory = build_run_memory({"dag": _clinc_like_dag()})
        assert "ROLLED BACK" in memory

    def test_most_recent_improvement_is_marked_kept(self):
        memory = build_run_memory({"dag": [_node(1, 0.5), _node(2, 0.9)]})
        assert "KEPT (new best)" in memory

    def test_deltas_are_shown(self):
        memory = build_run_memory({"dag": [_node(1, 0.8261), _node(2, 0.8734)]})
        assert "+0.0473" in memory


class TestFailuresSinceLastImprovement:
    def test_counts_every_failure_since_the_last_kept_attempt(self):
        memory = build_run_memory({"dag": _clinc_like_dag()})
        assert "17 attempts, none kept" in memory

    def test_aggregates_by_intervention_and_sub_strategy(self):
        memory = build_run_memory({"dag": _clinc_like_dag()})
        assert "data_rebuild/synthesize  x15" in memory
        assert "data_rebuild/acquire  x2" in memory

    def test_calls_out_the_dominant_failing_strategy(self):
        memory = build_run_memory({"dag": _clinc_like_dag()})
        assert "has been tried 15x since the last improvement" in memory
        assert "DIFFERENT intervention type" in memory

    def test_says_so_when_the_last_attempt_improved(self):
        memory = build_run_memory({"dag": [_node(1, 0.5), _node(2, 0.9)]})
        assert "none — the last attempt improved" in memory

    def test_older_failures_are_counted_not_narrated(self):
        memory = build_run_memory({"dag": _clinc_like_dag()})
        assert f"the {MAX_DETAILED_FAILURES} most recent, in full" in memory
        assert "older failure(s) counted in the totals above" in memory


class TestHypothesesAreCompleteNeverTruncated:
    """The whole point of the rewrite: full reasoning, in the log and in the memory."""

    LONG = (
        "Hard-bucket accuracy is 0.683 (n=142) with confusion pairs "
        "change_ai_name<->change_user_name (7), change_language->translate (4), "
        "cancel->freeze_account (3), account_blocked->extraction_failed (3), and "
        "w2->income_tax (2), which are near-synonym intents whose surface forms overlap."
    )

    def test_kept_improvement_keeps_its_full_reasoning(self):
        memory = build_run_memory({"dag": [_node(1, 0.9, hypothesis=self.LONG)]})
        assert self.LONG in memory

    def test_recent_failure_keeps_its_full_reasoning(self):
        dag = [_node(1, 0.9), _node(2, 0.5, hypothesis=self.LONG)]
        memory = build_run_memory({"dag": dag})
        assert self.LONG in memory

    def test_nothing_ends_mid_word_with_an_ellipsis(self):
        memory = build_run_memory({"dag": _clinc_like_dag()})
        assert "..." not in memory.replace("---", "")


class TestLabellingIsHonest:
    """
    `pi.D.plan` persists across iterations, so a hyperparameter node still carries the plan
    from the last rebuild. Rendering that as `hyperparameter/acquire` would credit a data
    strategy to an iteration that never touched the data — the B237 false-memory defect again.
    """

    def test_hyperparameter_node_carrying_a_stale_plan_is_not_given_a_sub_strategy(self):
        node = _node(2, 0.8734, intervention="hyperparameter", strategy="acquire")
        memory = build_run_memory({"dag": [node]})
        assert "hyperparameter/acquire" not in memory
        assert "hyperparameter" in memory

    def test_data_rebuild_keeps_its_sub_strategy(self):
        node = _node(3, 0.72, intervention="data_rebuild", strategy="synthesize")
        assert "data_rebuild/synthesize" in build_run_memory({"dag": [node]})


class TestSurgicalSpend:
    def test_reports_an_exhausted_pair(self):
        state = {
            "dag": [_node(1, 0.8)],
            "surgical_pair_history": {
                "change_ai_name->change_user_name": {
                    "count_when_targeted": 7, "iteration": 4, "rows_generated": 40,
                }
            },
            "test_report": {"confusion_pairs": [
                {"gold": "change_ai_name", "predicted": "change_user_name", "count": 7}
            ]},
        }
        memory = build_run_memory(state)
        assert "EXHAUSTED, do not target again" in memory

    def test_reports_an_improving_pair(self):
        state = {
            "dag": [_node(1, 0.8)],
            "surgical_pair_history": {
                "change_language->translate": {
                    "count_when_targeted": 4, "iteration": 5, "rows_generated": 20,
                }
            },
            "test_report": {"confusion_pairs": [
                {"gold": "change_language", "predicted": "translate", "count": 2}
            ]},
        }
        assert "improving" in build_run_memory(state)

    def test_reports_a_resolved_pair(self):
        state = {
            "dag": [_node(1, 0.8)],
            "surgical_pair_history": {
                "a->b": {"count_when_targeted": 5, "iteration": 3, "rows_generated": 15}
            },
            "test_report": {"confusion_pairs": []},
        }
        assert "RESOLVED" in build_run_memory(state)

    def test_section_absent_when_no_surgical_history(self):
        assert "SURGICAL SPEND" not in build_run_memory({"dag": [_node(1, 0.8)]})


class TestRobustness:
    def test_empty_dag_returns_empty_string(self):
        assert build_run_memory({}) == ""
        assert build_run_memory({"dag": []}) == ""

    def test_survives_malformed_nodes(self):
        dag = [None, {"iteration": 1}, _node(2, 0.7), "junk"]
        memory = build_run_memory({"dag": dag})
        assert "MOST RECENT ITERATION" in memory

    def test_rolled_back_attempts_are_still_present(self):
        """rollback marks nodes pruned; memory must still show them."""
        dag = [_node(1, 0.9), {**_node(2, 0.4), "pruned": True}]
        memory = build_run_memory({"dag": dag})
        assert "iter 2" in memory or "iteration 2" in memory

    def test_difficulty_and_confusions_render_for_latest(self):
        node = _node(1, 0.7)
        node["evaluation_state"]["test_report"] = {
            "by_difficulty": {"easy": {"accuracy": 0.918, "n": 214},
                              "hard": {"accuracy": 0.197, "n": 142}},
            "confusion_pairs": [{"gold": "a", "predicted": "b", "count": 7}],
        }
        memory = build_run_memory({"dag": [node]})
        assert "easy=0.918(n=214)" in memory
        assert "a->b (7)" in memory
