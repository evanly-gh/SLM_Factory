"""Tests for model selection strategies in agent/nodes/cold_start/model_selection/."""
import os

# interpolation._probe_model lazily imports agent.nodes.evaluate, which imports
# config.config and hard-requires these at module load. Set them here so this file
# passes standalone rather than only when some other test module happens to be
# collected first and set them.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

from unittest.mock import patch, MagicMock  # noqa: E402
import pytest  # noqa: E402

from config.android_pool import CapabilityMeasurement, ModelSpec, HardwareConstraints
from eval.harness import EvalResult


def _model(model_id, size_mb, tier=1, quant=None):
    return ModelSpec(
        model_id=model_id,
        size_mb=size_mb,
        tier=tier,
        capability_measurements=(
            CapabilityMeasurement(
                metric="GSM8K", value=0.6, artifact=model_id,
                mode=None, protocol="test", source="https://example.test",
            ),
            CapabilityMeasurement(
                metric="MMLU", value=0.5, artifact=model_id,
                mode=None, protocol="test", source="https://example.test",
            ),
        ),
        quant=quant,
    )


def _hw():
    return HardwareConstraints(
        storage_mb=5000, memory_mb=5000, latency_ttft_ms=3000,
    )


def _state(feasible, **overrides):
    from data.eval_set import EvalSet
    eval_set = MagicMock(spec=EvalSet)
    s = {
        "description": "test task",
        "feasible_models": feasible,
        "stop_threshold": 0.80,
        "initial_stop_threshold": 0.80,
        "task_type": "classification",
        "task_plan": {"task_type": "classification", "task_name": "test"},
        "hardware_constraints": _hw(),
        "eval_set": eval_set,
        "current_dataset_path": "/fake/dataset.jsonl",
        "scores": [],
        "dag": [],
        "iteration": 0,
        "dataset_version": 0,
        "best_score": 0.0,
        "lifetime_best_score": 0.0,
        "best_weights_ref": None,
        "last_eval": None,
        "last_curation": None,
        "last_intervention": "data_rebuild",
        "last_hypothesis": "",
        "llm_iterate_decision": None,
        "consecutive_no_improvement": 0,
        "downward_probe_done": False,
        "_largest_first_phase": None,
    }
    s.update(overrides)
    return s


# ── smallest_first ──────────────────────────────────────────────────────

class TestSmallestFirst:
    def test_selects_smallest(self):
        from agent.nodes.cold_start.model_selection.smallest_first import smallest_first_node
        models = [_model("big", 2000), _model("med", 1000), _model("small", 300)]
        state = _state(models)
        result = smallest_first_node(state)
        assert result["selected_model"].model_id == "small"

    def test_single_model(self):
        from agent.nodes.cold_start.model_selection.smallest_first import smallest_first_node
        models = [_model("only", 500)]
        state = _state(models)
        result = smallest_first_node(state)
        assert result["selected_model"].model_id == "only"

    def test_empty_raises(self):
        from agent.nodes.cold_start.model_selection.smallest_first import smallest_first_node
        with pytest.raises(RuntimeError, match="feasible_models is empty"):
            smallest_first_node(_state([]))

    def test_force_model_override(self):
        from agent.nodes.cold_start.model_selection.smallest_first import smallest_first_node
        models = [_model("big", 2000), _model("small", 300)]
        state = _state(models)
        with patch.dict(os.environ, {"SLM_FORCE_MODEL": "big"}):
            result = smallest_first_node(state)
        assert result["selected_model"].model_id == "big"


# ── largest_first ───────────────────────────────────────────────────────

class TestLargestFirst:
    def test_selects_largest_in_probe_phase(self):
        from agent.nodes.cold_start.model_selection.largest_first import largest_first_node
        models = [_model("big", 2000), _model("small", 300)]
        state = _state(models)
        result = largest_first_node(state)
        assert result["selected_model"].model_id == "big"
        assert result["_largest_first_phase"] == "probe"

    def test_probe_success_switches_to_smallest(self):
        from agent.nodes.cold_start.model_selection.largest_first import check_probe_result
        models = [_model("big", 2000), _model("small", 300)]
        state = _state(models, scores=[0.85], best_score=0.85,
                       selected_model=models[0], _largest_first_phase="probe")
        result = check_probe_result(state)
        assert result["selected_model"].model_id == "small"
        assert result["_largest_first_phase"] == "escalate"
        assert result["scores"] == []

    def test_probe_failure_stays(self):
        from agent.nodes.cold_start.model_selection.largest_first import check_probe_result
        models = [_model("big", 2000)]
        state = _state(models, scores=[0.60], best_score=0.60,
                       selected_model=models[0], _largest_first_phase="probe")
        result = check_probe_result(state)
        assert result["_largest_first_phase"] == "done"

    def test_empty_raises(self):
        from agent.nodes.cold_start.model_selection.largest_first import largest_first_node
        with pytest.raises(RuntimeError, match="feasible_models is empty"):
            largest_first_node(_state([]))


# ── interpolation ───────────────────────────────────────────────────────

class TestInterpolation:
    def test_curve_fit_aggregates_duplicate_deployment_footprints(self):
        from agent.nodes.cold_start.model_selection import interpolation

        fitter = getattr(interpolation, "_fit_scaling_curve", None)
        assert callable(fitter), "_fit_scaling_curve is not implemented"

        fit = fitter([(1_000, 0.4), (1_000, 0.6), (2_000, 0.8)])

        assert fit is not None
        slope, intercept = fit
        assert slope * __import__("math").log(1_000) + intercept == pytest.approx(0.5)
        assert slope * __import__("math").log(2_000) + intercept == pytest.approx(0.8)

    def test_curve_fit_rejects_single_unique_deployment_footprint(self):
        from agent.nodes.cold_start.model_selection import interpolation

        fitter = getattr(interpolation, "_fit_scaling_curve", None)
        assert callable(fitter), "_fit_scaling_curve is not implemented"
        assert fitter([(1_000, 0.4), (1_000, 0.6)]) is None

    def test_quantized_probe_scores_exact_deployment_artifact(self, tmp_path):
        from agent.nodes.cold_start.model_selection.interpolation import _probe_model

        model = _model("same", 900, tier=1, quant="Q8_0")
        state = _state([model])
        state["current_dataset_path"] = "/data.jsonl"
        result = EvalResult(
            f1=0.77,
            per_class={},
            pos_score=0.77,
            neg_score=0.77,
            boundary_score=0.77,
            failures=[],
        )
        with (
            patch(
                "agent.nodes.cold_start.model_selection.interpolation.slm_train",
                return_value=MagicMock(weights_ref="/probe/weights"),
            ),
            patch(
                "agent.nodes.evaluate._build_gguf_for_eval",
                return_value="/probe/model-q8_0.gguf",
            ) as build,
            patch(
                "agent.nodes.cold_start.model_selection.interpolation.run_eval",
                return_value=result,
            ) as run_eval,
        ):
            score = _probe_model(model, state, str(tmp_path))

        assert score == 0.77
        build.assert_called_once_with(
            "/probe/weights",
            "same",
            "Q8_0",
            "same [Q8_0]",
        )
        assert run_eval.call_args.kwargs["quant"] == "Q8_0"
        assert run_eval.call_args.kwargs["gguf_path"] == "/probe/model-q8_0.gguf"

    @patch("agent.nodes.cold_start.model_selection.interpolation._probe_model")
    def test_selects_closest_to_ram_target(self, mock_probe):
        from agent.nodes.cold_start.model_selection.interpolation import interpolation_node
        models = [
            _model("large", 2000),
            _model("medium", 1000),
            _model("small", 500),
        ]
        mock_probe.side_effect = [0.90, 0.85, 0.75]
        state = _state(models)
        state["hardware_constraints"] = HardwareConstraints(
            storage_mb=5000, memory_mb=1500, latency_ttft_ms=3000,
        )
        result = interpolation_node(state)
        assert result["selected_model"] is not None

    @patch("agent.nodes.cold_start.model_selection.interpolation._probe_model")
    def test_falls_back_to_largest_when_none_qualify(self, mock_probe):
        from agent.nodes.cold_start.model_selection.interpolation import interpolation_node
        models = [_model("large", 2000), _model("small", 300)]
        mock_probe.side_effect = [0.30, 0.20]
        state = _state(models)
        result = interpolation_node(state)
        assert result["selected_model"].model_id == "large"

    def test_empty_raises(self):
        from agent.nodes.cold_start.model_selection.interpolation import interpolation_node
        with pytest.raises(RuntimeError, match="feasible_models is empty"):
            interpolation_node(_state([]))


# ── orchestrator_choice ─────────────────────────────────────────────────

class TestOrchestratorChoice:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"}, clear=False)
    @patch("anthropic.Anthropic")
    def test_selects_llm_choice(self, mock_cls):
        import anthropic as _anth
        from agent.nodes.cold_start.model_selection.orchestrator_choice import orchestrator_choice_node
        models = [_model("big", 2000), _model("small", 300)]
        state = _state(models)

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        block = _anth.types.TextBlock(text='{"model_id": "small", "reason": "simple task"}', type="text")
        mock_client.messages.create.return_value = MagicMock(content=[block])

        result = orchestrator_choice_node(state)
        assert result["selected_model"].model_id == "small"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"}, clear=False)
    @patch("anthropic.Anthropic")
    def test_falls_back_on_failure(self, mock_cls):
        from agent.nodes.cold_start.model_selection.orchestrator_choice import orchestrator_choice_node
        models = [_model("big", 2000), _model("small", 300)]
        state = _state(models)

        mock_cls.side_effect = Exception("API failure")

        result = orchestrator_choice_node(state)
        assert result["selected_model"].model_id == "small"

    def test_empty_raises(self):
        from agent.nodes.cold_start.model_selection.orchestrator_choice import orchestrator_choice_node
        with pytest.raises(RuntimeError, match="feasible_models is empty"):
            orchestrator_choice_node(_state([]))


# ── registry ────────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_strategies_registered(self):
        from agent.nodes.cold_start.model_selection import STRATEGIES
        assert set(STRATEGIES.keys()) == {
            "smallest_first", "largest_first", "interpolation", "orchestrator_choice",
        }

    def test_get_unknown_raises(self):
        from agent.nodes.cold_start.model_selection import get_model_selection_node
        with pytest.raises(ValueError, match="Unknown MODEL_SELECTION_STRATEGY"):
            get_model_selection_node("nonexistent")

    def test_get_valid_returns_callable(self):
        from agent.nodes.cold_start.model_selection import get_model_selection_node
        for name in ("smallest_first", "largest_first", "interpolation", "orchestrator_choice"):
            node = get_model_selection_node(name)
            assert callable(node)

    @pytest.mark.parametrize(
        "strategy",
        ("smallest_first", "largest_first", "interpolation", "orchestrator_choice"),
    )
    def test_force_model_selects_exact_quant_variant(self, strategy):
        from agent.nodes.cold_start.model_selection import get_model_selection_node

        siblings = [
            _model("same", 1_800, tier=2, quant=None),
            _model("same", 900, tier=1, quant="Q8_0"),
            _model("same", 500, tier=0, quant="Q4_K_M"),
        ]
        wanted = siblings[1]
        with patch.dict(os.environ, {"SLM_FORCE_MODEL": "same@Q8_0"}):
            result = get_model_selection_node(strategy)(_state(siblings))

        assert result["selected_model"] is wanted

    def test_ambiguous_bare_force_model_uses_lowest_resource_variant(self):
        from agent.nodes.cold_start.model_selection.smallest_first import smallest_first_node

        siblings = [
            _model("same", 1_800, tier=2, quant=None),
            _model("same", 900, tier=1, quant="Q8_0"),
            _model("same", 500, tier=0, quant="Q4_K_M"),
        ]
        with patch.dict(os.environ, {"SLM_FORCE_MODEL": "same"}):
            result = smallest_first_node(_state(siblings))

        assert result["selected_model"].quant == "Q4_K_M"


# ── iterate integration with largest_first ──────────────────────────────

class TestIterateLargestFirstIntegration:
    @patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip"))
    def test_probe_stagnation_terminates(self, _mock):
        from agent.nodes.iterate import iterate_node, STAGNATION_WINDOW
        m = _model("big", 2000, tier=2)
        # A full flat window below threshold → stagnation in the largest_first probe phase
        # means the task is infeasible → terminate. Uses the configured window length.
        flat = [0.71] * STAGNATION_WINDOW
        state = {
            "selected_model": m,
            "scores": flat,
            "best_score": 0.71,
            "iteration": 3,
            "turn_budget": 1000,
            "stop_threshold": 0.90,
            "initial_stop_threshold": 0.90,
            "task_type": "classification",
            "last_eval": None,
            "hw_gating_enabled": False,
            "downward_probe_done": False,
            "consecutive_no_improvement": 0,
            "_largest_first_phase": "probe",
            "hardware_constraints": _hw(),
        }
        out = iterate_node(state)
        assert out["next_action"] == "terminate"
        assert out["_largest_first_phase"] == "done"

    @patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip"))
    @patch("agent.nodes.cold_start.model_selection.largest_first.check_probe_result")
    def test_probe_success_routes_to_curate(self, mock_check, _mock_llm):
        from agent.nodes.iterate import iterate_node
        m = _model("big", 2000, tier=2)

        def fake_check(state):
            state["next_action"] = "curate"
            state["_largest_first_phase"] = "escalate"
            state["selected_model"] = _model("small", 300, tier=0)
            return state

        mock_check.side_effect = fake_check

        state = {
            "selected_model": m,
            "scores": [0.92],
            "best_score": 0.92,
            "iteration": 1,
            "turn_budget": 1000,
            "stop_threshold": 0.90,
            "initial_stop_threshold": 0.90,
            "task_type": "classification",
            "last_eval": None,
            "hw_gating_enabled": False,
            "downward_probe_done": False,
            "_largest_first_phase": "probe",
            "hardware_constraints": _hw(),
        }
        out = iterate_node(state)
        assert out["next_action"] == "curate"
        assert out["selected_model"].model_id == "small"
