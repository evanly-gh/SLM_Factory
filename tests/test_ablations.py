"""The ablation switches in `agent/ablations.py`.

Two properties matter more than any individual assertion here:

  1. WITH NO FLAGS SET, NOTHING CHANGES. Every ablation is reached through a function that reads
     an env var defaulting to "0", so a normal run must be indistinguishable from one built before
     this module existed. `test_escalate_new.py` covers the same ground from the escalate side.
  2. A MISCONFIGURED ABLATION FAILS LOUDLY. The failure mode that would waste a multi-day L40S
     job is an ablation that quietly degrades into the baseline it is being compared against, so
     the reset raises rather than carrying the dataset forward when its snapshot is missing.
"""
import json
import os
from unittest.mock import patch

import pytest

from agent.ablations import (
    RESET_STRATEGY,
    AblationStateError,
    active_ablations,
    capture_seed_snapshot,
    reset_curriculum_to_seed,
    reset_data_on_escalation,
    synthesis_disallowed,
)
from agent.run_health import MINING_RETIRED_KEY

SEED_ROWS = [{"text": f"seed {i}", "label": "a", "_provenance": "train_anchor"} for i in range(6)]
GOLD_POOL = [{"text": f"pool {i}", "label": "a", "_provenance": "train_anchor"} for i in range(10)]


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A run's CWD. `ARTIFACTS_DIR` is relative, exactly as curate and eval_setup use it."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts", exist_ok=True)
    return tmp_path


@pytest.fixture
def reset_on(monkeypatch):
    monkeypatch.setenv("SLM_ABLATION_RESET_DATA_ON_ESCALATION", "1")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _seeded_state(run_dir):
    """State as it stands right after the initial_gold build, with dataset_v1 on disk."""
    path = os.path.join("artifacts", "dataset_v1.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in SEED_ROWS:
            handle.write(json.dumps({**row, "_dataset_version": 1}) + "\n")
    return {
        "train_examples": list(GOLD_POOL),
        "current_dataset_path": path,
        "dataset_version": 1,
        "source_progress": {"bc5cdr": {"consumed": 10, "asked_for": 10, "exhausted": False}},
        "last_curation": {"total_examples": len(SEED_ROWS), "strategy": "initial_gold",
                          "n_gold": len(SEED_ROWS), "rows_added": len(GOLD_POOL)},
        "data_source_usage": [{"iteration": 0, "dataset_version": "v1",
                               "strategy": "initial_gold", "sources": [{"source": "bc5cdr",
                                                                        "rows": 6}]}],
        "iteration": 0,
    }


class TestFlagsDefaultOff:
    def test_both_flags_are_off_with_no_env(self, monkeypatch):
        monkeypatch.delenv("SLM_ABLATION_RESET_DATA_ON_ESCALATION", raising=False)
        monkeypatch.delenv("SLM_SYNTH_DISALLOW", raising=False)
        assert reset_data_on_escalation() is False
        assert synthesis_disallowed() is False
        assert active_ablations() == []

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_spellings_all_enable(self, monkeypatch, value):
        monkeypatch.setenv("SLM_SYNTH_DISALLOW", value)
        assert synthesis_disallowed() is True

    @pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
    def test_falsy_spellings_all_disable(self, monkeypatch, value):
        monkeypatch.setenv("SLM_SYNTH_DISALLOW", value)
        assert synthesis_disallowed() is False

    def test_active_ablations_names_each_live_flag(self, monkeypatch):
        monkeypatch.setenv("SLM_ABLATION_RESET_DATA_ON_ESCALATION", "1")
        monkeypatch.setenv("SLM_SYNTH_DISALLOW", "1")
        described = active_ablations()
        assert len(described) == 2
        assert any("SLM_ABLATION_RESET_DATA_ON_ESCALATION=1" in line for line in described)
        assert any("SLM_SYNTH_DISALLOW=1" in line for line in described)


class TestSeedSnapshot:
    def test_no_snapshot_is_taken_when_the_flag_is_off(self, run_dir, monkeypatch):
        monkeypatch.delenv("SLM_ABLATION_RESET_DATA_ON_ESCALATION", raising=False)
        state = _seeded_state(run_dir)
        capture_seed_snapshot(state, state["current_dataset_path"], log=lambda _: None)
        assert state.get("seed_dataset_path") is None
        assert not os.path.exists(os.path.join("artifacts", "seed_train_examples.jsonl"))

    def test_snapshot_records_paths_progress_and_composition(self, run_dir, reset_on):
        state = _seeded_state(run_dir)
        capture_seed_snapshot(state, state["current_dataset_path"], log=lambda _: None)
        assert state["seed_dataset_path"] == os.path.join("artifacts", "dataset_v1.jsonl")
        assert _read_jsonl(state["seed_train_examples_path"]) == GOLD_POOL
        assert state["seed_source_progress"] == state["source_progress"]
        assert state["seed_last_curation"]["strategy"] == "initial_gold"

    def test_the_snapshot_is_a_copy_not_a_reference(self, run_dir, reset_on):
        """Mutating live state afterwards must not rewrite history."""
        state = _seeded_state(run_dir)
        capture_seed_snapshot(state, state["current_dataset_path"], log=lambda _: None)
        state["source_progress"]["bc5cdr"]["consumed"] = 999
        state["last_curation"]["strategy"] = "surgical_synthesis"
        assert state["seed_source_progress"]["bc5cdr"]["consumed"] == 10
        assert state["seed_last_curation"]["strategy"] == "initial_gold"

    def test_a_second_call_does_not_overwrite_the_seed(self, run_dir, reset_on):
        """A requeue resuming past the first curate must not re-snapshot a grown curriculum."""
        state = _seeded_state(run_dir)
        capture_seed_snapshot(state, state["current_dataset_path"], log=lambda _: None)
        state["train_examples"] = GOLD_POOL + [{"text": "mined", "_provenance": "mined_real"}]
        capture_seed_snapshot(state, "artifacts/dataset_v7.jsonl", log=lambda _: None)
        assert state["seed_dataset_path"] == os.path.join("artifacts", "dataset_v1.jsonl")
        assert _read_jsonl(state["seed_train_examples_path"]) == GOLD_POOL


class TestResetToSeed:
    def _grown_state(self, run_dir):
        """State as it stands at the end of a tier: curriculum grown, bookkeeping spent."""
        state = _seeded_state(run_dir)
        capture_seed_snapshot(state, state["current_dataset_path"], log=lambda _: None)
        grown = os.path.join("artifacts", "dataset_v4.jsonl")
        with open(grown, "w", encoding="utf-8") as handle:
            for row in SEED_ROWS + [{"text": "synth", "_provenance": "synthetic"}]:
                handle.write(json.dumps(row) + "\n")
        state.update({
            "current_dataset_path": grown,
            "dataset_version": 4,
            "train_examples": GOLD_POOL + [{"text": "mined", "_provenance": "mined_real"}],
            "source_progress": {"bc5cdr": {"consumed": 10, "asked_for": 10, "exhausted": True}},
            "surgical_category_history": {"missing_entity": 3},
            "failed_discovery_rounds": 2,
            "last_curation": {"total_examples": 7, "strategy": "surgical_synthesis"},
            "data_rebuild_plan": {"strategy": "surgical_synthesis"},
            "data_rebuild_plan_identity": "plan-abc",
            "run_health": {"empty_rebuilds": 1, "empty_mining": 2, "empty_synthesis": 1,
                           "mining_shutouts": 3, "verify_wipeouts": 1, "load_failures": 1,
                           "history": [{"iteration": 1, "strategy": "initial_gold"}]},
            MINING_RETIRED_KEY: "mine_new_real added 0 rows on 2 rounds",
            "iteration": 12,
        })
        return state

    def test_curriculum_returns_to_the_seed_rows(self, run_dir, reset_on):
        state = self._grown_state(run_dir)
        summary = reset_curriculum_to_seed(state, log=lambda _: None)
        rows = _read_jsonl(state["current_dataset_path"])
        assert [row["text"] for row in rows] == [row["text"] for row in SEED_ROWS]
        assert summary["rows"] == len(SEED_ROWS)
        assert summary["previous_rows"] == 7

    def test_the_version_counter_advances_and_nothing_is_clobbered(self, run_dir, reset_on):
        """v1 and v4 must both survive; the reset publishes v5."""
        state = self._grown_state(run_dir)
        reset_curriculum_to_seed(state, log=lambda _: None)
        assert state["dataset_version"] == 5
        assert state["current_dataset_path"] == os.path.join("artifacts", "dataset_v5.jsonl")
        assert len(_read_jsonl(os.path.join("artifacts", "dataset_v1.jsonl"))) == len(SEED_ROWS)
        assert len(_read_jsonl(os.path.join("artifacts", "dataset_v4.jsonl"))) == len(SEED_ROWS) + 1
        assert {row["_dataset_version"] for row in _read_jsonl(state["current_dataset_path"])} == {5}

    def test_the_train_pool_loses_its_mined_rows(self, run_dir, reset_on):
        state = self._grown_state(run_dir)
        reset_curriculum_to_seed(state, log=lambda _: None)
        assert state["train_examples"] == GOLD_POOL
        assert not any(r.get("_provenance") == "mined_real" for r in state["train_examples"])

    def test_all_data_bookkeeping_is_reverted(self, run_dir, reset_on):
        state = self._grown_state(run_dir)
        reset_curriculum_to_seed(state, log=lambda _: None)
        assert state["source_progress"]["bc5cdr"]["exhausted"] is False
        assert state["surgical_category_history"] == {}
        assert state["failed_discovery_rounds"] == 0
        assert state["data_rebuild_plan"] is None
        assert state["data_rebuild_plan_identity"] is None

    def test_mining_is_un_retired_and_available_again(self, run_dir, reset_on):
        from agent.data_rebuild import mining_available_for_state

        state = self._grown_state(run_dir)
        assert mining_available_for_state(state) is False
        reset_curriculum_to_seed(state, log=lambda _: None)
        assert not state[MINING_RETIRED_KEY]
        assert mining_available_for_state(state) is True

    def test_empty_route_counters_clear_but_the_audit_trail_survives(self, run_dir, reset_on):
        state = self._grown_state(run_dir)
        reset_curriculum_to_seed(state, log=lambda _: None)
        health = state["run_health"]
        assert health["empty_rebuilds"] == 0
        assert health["empty_mining"] == 0
        assert health["empty_synthesis"] == 0
        assert health["mining_shutouts"] == 0
        assert health["verify_wipeouts"] == 0
        # An environment fault and the run's own history are not curriculum state.
        assert health["load_failures"] == 1
        assert health["history"] == [{"iteration": 1, "strategy": "initial_gold"}]

    def test_the_restored_composition_is_honest_about_the_transition(self, run_dir, reset_on):
        """`last_curation` drives the orchestrator prompt; None would read as zero rows."""
        state = self._grown_state(run_dir)
        reset_curriculum_to_seed(state, log=lambda _: None)
        curation = state["last_curation"]
        assert curation["total_examples"] == len(SEED_ROWS)
        assert curation["strategy"] == RESET_STRATEGY
        assert curation["previous_rows"] == 7
        assert curation["rows_added"] == 0
        assert curation["failed_discovery_rounds"] == 0

    def test_the_reset_is_recorded_in_the_provenance_ledger(self, run_dir, reset_on):
        state = self._grown_state(run_dir)
        reset_curriculum_to_seed(state, log=lambda _: None)
        entry = state["data_source_usage"][-1]
        assert entry["strategy"] == RESET_STRATEGY
        assert entry["dataset_version"] == "v5"
        # Empty on purpose: re-listing the seed's sources would double-count their rows.
        assert entry["sources"] == []

    def test_per_source_row_totals_are_not_double_counted(self, run_dir, reset_on):
        from data.provenance import aggregate_data_sources

        state = self._grown_state(run_dir)
        reset_curriculum_to_seed(state, log=lambda _: None)
        reset_curriculum_to_seed(state, log=lambda _: None)
        totals = {row["source"]: row["total_rows"]
                  for row in aggregate_data_sources(state["data_source_usage"])}
        assert totals == {"bc5cdr": 6}

    def test_repeated_resets_are_idempotent_in_content(self, run_dir, reset_on):
        """Three tier promotions must each land on the same rows."""
        state = self._grown_state(run_dir)
        landed = []
        for _ in range(3):
            reset_curriculum_to_seed(state, log=lambda _: None)
            landed.append([row["text"] for row in _read_jsonl(state["current_dataset_path"])])
        assert landed[0] == landed[1] == landed[2] == [row["text"] for row in SEED_ROWS]
        assert state["dataset_version"] == 7

    def test_a_missing_seed_snapshot_raises_rather_than_carrying_forward(self, run_dir, reset_on):
        """The failure mode this guards: silently running the baseline under an ablation's name."""
        state = self._grown_state(run_dir)
        state["seed_dataset_path"] = None
        with pytest.raises(AblationStateError, match="no seed curriculum snapshot"):
            reset_curriculum_to_seed(state, log=lambda _: None)

    def test_a_half_written_snapshot_raises(self, run_dir, reset_on):
        state = self._grown_state(run_dir)
        os.remove(state["seed_train_examples_path"])
        with pytest.raises(AblationStateError, match="seed train pool does not"):
            reset_curriculum_to_seed(state, log=lambda _: None)


class TestEscalateHonoursTheFlag:
    """The reset has to happen on the PROMOTION, before the new tier's first measurement."""

    def _escalate(self, run_dir, state_extras):
        from config.android_pool import ANDROID_POOL, HardwareConstraints

        lowest = min(m.tier for m in ANDROID_POOL)
        start = next(m for m in ANDROID_POOL if m.tier == lowest)
        next_tier = sorted({m.tier for m in ANDROID_POOL if m.tier > lowest})[0]
        state = {
            "selected_model": start,
            "scores": [0.70, 0.71, 0.71],
            "best_score": 0.71,
            "hardware_constraints": HardwareConstraints(
                storage_mb=10000, memory_mb=10000, latency_ttft_ms=5000,
            ),
            "hw_gating_enabled": False,
            "task": "clinc150",
            "task_plan": {"task": "clinc150", "task_name": "test"},
            "model_baselines": [],
            "dag": [],
            "consecutive_no_improvement": 0,
            "last_eval": None,
            "last_hypothesis": "",
            "llm_iterate_decision": None,
        }
        state.update(state_extras)
        with patch("agent.nodes.escalate._llm_choose_model") as mock_llm:
            mock_llm.return_value = next(m for m in ANDROID_POOL if m.tier == next_tier)
            from agent.nodes.escalate import escalate_node

            return escalate_node(state)

    def test_flag_off_carries_the_dataset_forward(self, run_dir, monkeypatch):
        monkeypatch.delenv("SLM_ABLATION_RESET_DATA_ON_ESCALATION", raising=False)
        grown = TestResetToSeed()._grown_state(run_dir)
        out = self._escalate(run_dir, grown)
        assert out["current_dataset_path"] == os.path.join("artifacts", "dataset_v4.jsonl")
        assert out["dataset_version"] == 4
        assert out["surgical_category_history"] == {"missing_entity": 3}

    def test_flag_on_resets_before_the_new_tier_trains(self, run_dir, reset_on):
        grown = TestResetToSeed()._grown_state(run_dir)
        out = self._escalate(run_dir, grown)
        assert out["dataset_version"] == 5
        assert [r["text"] for r in _read_jsonl(out["current_dataset_path"])] == [
            r["text"] for r in SEED_ROWS
        ]
        assert out["train_examples"] == GOLD_POOL
        assert out["surgical_category_history"] == {}
        # Unchanged by the ablation: a new tier's first iteration is still a plain retrain, which
        # is what makes the tier's opening score attributable to the model alone.
        assert out["last_intervention"] == "hyperparameter"
        assert out["next_action"] == "curate"


class TestSynthesisDisallowed:
    def test_disallow_beats_a_teacher_that_cleared_the_gate(self, monkeypatch):
        from agent.teacher_fitness import synthesis_allowed

        state = {"teacher_fitness": {"status": "measured", "score": 0.95,
                                     "synthesis_allowed": True}}
        monkeypatch.delenv("SLM_SYNTH_DISALLOW", raising=False)
        assert synthesis_allowed(state) is True
        monkeypatch.setenv("SLM_SYNTH_DISALLOW", "1")
        assert synthesis_allowed(state) is False

    def test_disallow_beats_the_bypass_override(self, monkeypatch):
        """Every curated launcher sets BYPASS; it must not hand synthesis back."""
        import importlib

        import agent.teacher_fitness as tf

        monkeypatch.setenv("SLM_TEACHER_SYNTH_BYPASS", "1")
        monkeypatch.setenv("SLM_SYNTH_DISALLOW", "1")
        tf = importlib.reload(tf)
        try:
            assert tf.BYPASS is True
            assert tf.synthesis_allowed({"teacher_fitness": {"synthesis_allowed": False}}) is False
            assert tf.synthesis_allowed({}) is False
        finally:
            monkeypatch.delenv("SLM_TEACHER_SYNTH_BYPASS", raising=False)
            monkeypatch.delenv("SLM_SYNTH_DISALLOW", raising=False)
            importlib.reload(tf)

    def test_the_verdict_records_an_operator_refusal_not_a_failed_gate(self, monkeypatch):
        from types import SimpleNamespace

        from agent.teacher_fitness import _apply_operator_overrides

        monkeypatch.setenv("SLM_SYNTH_DISALLOW", "1")
        verdict = {"status": "measured", "score": 0.95, "synthesis_allowed": True}
        lines = []
        out = _apply_operator_overrides(verdict, SimpleNamespace(name="ner_bc5cdr"),
                                       log=lines.append)
        assert out["synthesis_allowed"] is False
        assert out["disallowed_by_operator"] is True
        assert any("SLM_SYNTH_DISALLOW=1" in line for line in lines)
        assert any("WOULD have been allowed to generate" in line for line in lines)

    def test_data_rebuild_is_refused_when_mining_is_also_gone(self, monkeypatch):
        """With both routes dead the orchestrator must be pushed to hyperparameters."""
        from agent.data_rebuild import DataInterventionUnavailable, normalize_data_rebuild_plan

        monkeypatch.setenv("SLM_SYNTH_DISALLOW", "1")
        with pytest.raises(DataInterventionUnavailable):
            normalize_data_rebuild_plan(
                {"strategy": "surgical_synthesis"},
                task="ner_bc5cdr",
                mining_available=False,
                synthesis_allowed=False,
            )

    def test_a_synthesis_plan_is_rewritten_to_mining_while_mining_lives(self, monkeypatch):
        from agent.data_rebuild import normalize_data_rebuild_plan

        monkeypatch.setenv("SLM_SYNTH_DISALLOW", "1")
        plan = normalize_data_rebuild_plan(
            {"strategy": "surgical_synthesis"},
            task="ner_bc5cdr",
            mining_available=True,
            synthesis_allowed=False,
        )
        assert plan["strategy"] == "mine_new_real"


class TestDataScarcityAblation:
    """Ablation 4: does synthetic data earn its keep when real data is SCARCE?

    The suite's runs all trained on thousands of real rows and synthesis contributed between
    +0.008 and +0.050 on four of the five 09-2026 tasks. This arm starves a task to 100 real rows
    to find out whether abundance was the reason. Two flags, because capping alone does nothing:
    `mine_new_real` reads deeper into the same corpus, so a capped run would mine its way back to
    thousands and measure nothing at all.
    """

    def test_the_cap_is_absent_by_default_and_read_when_set(self, monkeypatch):
        from agent.ablations import train_cap_override

        monkeypatch.delenv("SLM_ABLATION_TRAIN_CAP", raising=False)
        assert train_cap_override() is None
        monkeypatch.setenv("SLM_ABLATION_TRAIN_CAP", "100")
        assert train_cap_override() == 100

    def test_a_malformed_cap_raises_rather_than_falling_back(self, monkeypatch):
        """Silently using the task's own cap would run the BASELINE under an ablation's job name.

        That is the single outcome this module's docstring calls worse than not running the
        experiment, because the result looks valid and is not.
        """
        import pytest

        from agent.ablations import train_cap_override

        for bad in ("oops", "0", "-5"):
            monkeypatch.setenv("SLM_ABLATION_TRAIN_CAP", bad)
            with pytest.raises(ValueError):
                train_cap_override()

    def test_mining_is_refused_for_the_whole_run_when_the_flag_is_set(self, monkeypatch):
        """Checked ahead of run health and source bookkeeping, so a starved arm stays starved."""
        from agent.ablations import mining_disallowed
        from agent.data_rebuild import mining_available_for_state

        # A state that would otherwise offer mining: nothing retired, a source with rows left.
        state = {"source_progress": {"src": {"consumed": 10, "total": 5000}},
                 "failed_discovery_rounds": 0}
        monkeypatch.delenv("SLM_ABLATION_DISALLOW_MINING", raising=False)
        assert mining_disallowed() is False
        assert mining_available_for_state(state) is True

        monkeypatch.setenv("SLM_ABLATION_DISALLOW_MINING", "1")
        assert mining_disallowed() is True
        assert mining_available_for_state(state) is False

    def test_both_switches_are_reported_in_the_run_header(self, monkeypatch):
        """An unreported ablation is one nobody can tell was on when they read the result later."""
        from agent.ablations import active_ablations

        monkeypatch.setenv("SLM_ABLATION_TRAIN_CAP", "100")
        monkeypatch.setenv("SLM_ABLATION_DISALLOW_MINING", "1")
        blob = " ".join(active_ablations())
        assert "SLM_ABLATION_TRAIN_CAP=100" in blob
        assert "SLM_ABLATION_DISALLOW_MINING=1" in blob

    def test_both_switches_are_resume_sensitive(self, monkeypatch):
        """A requeue that dropped either would finish a scarcity run on a non-scarce curriculum."""
        from agent.checkpoint import runtime_config_fingerprint

        monkeypatch.delenv("SLM_ABLATION_TRAIN_CAP", raising=False)
        monkeypatch.delenv("SLM_ABLATION_DISALLOW_MINING", raising=False)
        plain = runtime_config_fingerprint("main")
        monkeypatch.setenv("SLM_ABLATION_TRAIN_CAP", "100")
        capped = runtime_config_fingerprint("main")
        assert capped != plain, "the row cap must change the resume fingerprint"
        monkeypatch.setenv("SLM_ABLATION_DISALLOW_MINING", "1")
        assert runtime_config_fingerprint("main") not in (plain, capped), (
            "the mining refusal must change the fingerprint independently of the cap"
        )
