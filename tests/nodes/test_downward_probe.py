import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from config.android_pool import CapabilityMeasurement, ModelSpec, HardwareConstraints
from eval.harness import EvalResult

os.environ.setdefault("ANTHROPIC_API_KEY", "test-no-network")
os.environ.setdefault("EXA_API_KEY", "test-no-network")


def _model(tier, model_id="test/M", quant=None):
    return ModelSpec(
        model_id=model_id, size_mb=1000, tier=tier,
        capability_measurements=(
            CapabilityMeasurement(
                metric="MMLU", value=0.5, artifact=model_id,
                mode=None, protocol="test", source="https://example.test",
            ),
        ),
        quant=quant,
    )


def _state(current):
    return {
        "selected_model": current,
        "task": "clinc150",
        "task_plan": {"task": "clinc150"},
        "stop_threshold": 0.90,
        "best_score": 0.95,
        "best_weights_ref": "/big/ckpt",
        "current_dataset_path": "/data.jsonl",
        "eval_set": MagicMock(),
        "hardware_constraints": HardwareConstraints(
            storage_mb=10000, memory_mb=10000, latency_ttft_ms=5000,
        ),
    }


def _fixed_probe_h():
    from agent.nodes.downward_probe import DOWNWARD_PROBE_H

    return dict(DOWNWARD_PROBE_H)


def _valid_hyperparameter_decision():
    """An orchestrator decision the validator accepts.

    Used by the one case here whose below-threshold half actually consults the orchestrator. The
    converged cases keep a raising stub, because for them the LLM must never be called at all: a
    threshold-clearing score is deterministic routing and spends no API call.
    """
    return {
        "intervention": "hyperparameter",
        "hypothesis": "the hard bucket is under-fit, so the adapter needs more capacity",
        "hyperparams": {
            "lora_rank": 16,
            "alpha_ratio": 2,
            "weight_decay": 0.01,
            "learning_rate": 0.0001,
            "nr_epochs": 4,
        },
    }


def test_agent_state_declares_downward_probe_history():
    from agent.state import AgentState

    assert "downward_probe_history" in AgentState.__annotations__
    assert "downward_probe_pending" in AgentState.__annotations__


def test_quantized_downward_probe_scores_exact_deployment_artifact():
    from agent.nodes.downward_probe import _train_and_eval

    current = _model(tier=2, model_id="test/Current")
    candidate = _model(tier=1, model_id="test/Candidate", quant="Q4_K_M")
    result = EvalResult(
        f1=0.88,
        per_class={},
        failures=[],
    )
    with (
        patch(
            "agent.nodes.downward_probe.run_training_atomically",
            return_value=MagicMock(weights_ref="/probe/weights"),
        ),
        patch(
            "agent.nodes.evaluate._build_quant_artifact_for_eval",
            return_value="/probe/model-q4_k_m.gguf",
        ) as build,
        patch("agent.nodes.downward_probe.run_eval", return_value=result) as run_eval,
    ):
        weights_ref, scored = _train_and_eval(candidate, _state(current), "/data.jsonl")

    assert weights_ref == "/probe/weights"
    assert scored is result
    build.assert_called_once_with(
        "/probe/weights",
        "test/Candidate",
        "Q4_K_M",
        "test/Candidate [Q4_K_M]",
    )
    assert run_eval.call_args.kwargs["quant"] == "Q4_K_M"
    assert run_eval.call_args.kwargs["quant_artifact"] == "/probe/model-q4_k_m.gguf"


def test_downward_training_uses_stable_atomic_run_artifact(tmp_path, monkeypatch):
    from agent.nodes import downward_probe as probe_module
    from training.lora_trainer import TrainingOutput

    current = _model(tier=2, model_id="test/Current")
    candidate = _model(tier=1, model_id="test/Candidate")
    state = _state(current)
    artifacts = tmp_path / "run-artifacts"
    monkeypatch.setattr(probe_module, "ARTIFACTS_DIR", str(artifacts))
    calls = []

    def fake_train(*, output_dir, **_kwargs):
        calls.append({"output_dir": output_dir, **_kwargs})
        checkpoint = Path(output_dir) / "final_checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        return TrainingOutput(str(checkpoint), None)

    result = EvalResult(0.91, {}, [])
    monkeypatch.setattr(probe_module, "slm_train", fake_train)
    monkeypatch.setattr(probe_module, "run_eval", lambda *_args, **_kwargs: result)

    first_weights, _ = probe_module._train_and_eval(
        candidate, state, "/data.jsonl"
    )
    second_weights, _ = probe_module._train_and_eval(
        candidate, state, "/data.jsonl"
    )

    expected = (
        artifacts
        / "downward_probe"
        / "test_Current__bf16"
        / "test_Candidate__bf16"
        / "final_checkpoint"
    )
    assert first_weights == second_weights == str(expected)
    assert (expected / ".slm_artifact_manifest.json").is_file()
    assert (expected / ".slm_complete").is_file()
    assert len(calls) == 1
    assert calls[0]["lora_rank"] == 16
    assert calls[0]["lora_alpha"] == 32
    assert calls[0]["lora_dropout"] == 0.0
    assert calls[0]["weight_decay"] == 0.01
    assert calls[0]["micro_batch_size"] == 8
    assert calls[0]["gradient_accumulation_steps"] == 1
    assert calls[0]["effective_batch_size"] == 8
    assert not list(expected.parent.parent.glob("*.partial-*"))


def test_tier0_model_skips_probe():
    from agent.nodes.downward_probe import downward_probe_node
    out = downward_probe_node(_state(_model(tier=0)))
    assert out["next_action"] == "terminate"
    assert out["selected_model"].tier == 0  # unchanged
    assert out["downward_probe_done"] is True


@patch("agent.nodes.downward_probe._train_and_eval")
@patch("agent.nodes.downward_probe._llm_choose_model")
@patch("agent.nodes.downward_probe.filter_pool")
def test_adopts_smaller_model_when_it_clears_threshold(mock_fp, mock_choose, mock_te):
    from agent.nodes.downward_probe import downward_probe_node

    current = _model(tier=2, model_id="test/Big", quant="Q8_0")
    smaller = _model(tier=1, model_id="test/Small")
    mock_fp.return_value = [smaller]
    mock_choose.return_value = smaller
    mock_te.return_value = ("/small/ckpt", EvalResult(
        f1=0.92, per_class={}, failures=[]))
    state = _state(current)
    state.update({
        "iteration": 4,
        "scores": [0.81, 0.88, 0.95],
        "dag": [
            {
                "iteration": 3,
                "model_id": current.model_id,
                "score": 0.95,
            }
        ],
    })

    out = downward_probe_node(state)

    assert out["selected_model"].model_id == "test/Small"
    assert out["best_weights_ref"] == "/small/ckpt"
    assert out["best_score"] == 0.92
    assert out["next_action"] == "terminate"
    assert out["converged_model_ref"]["selector"] == smaller.selector
    assert mock_choose.call_args.kwargs["direction"] == "down"
    history = out["downward_probe_history"]
    assert history["origin"] == {
        "selector": current.selector,
        "model_id": current.model_id,
        "quant": "Q8_0",
        "tier": 2,
        "score": 0.95,
        "weights_ref": "/big/ckpt",
        "iterations": 4,
        "scores": [0.81, 0.88, 0.95],
        "dag": state["dag"],
    }
    assert history["attempts"] == [
        {
            "selector": smaller.selector,
            "model_id": smaller.model_id,
            "quant": None,
            "tier": 1,
            "score": 0.92,
            "weights_ref": "/small/ckpt",
            "result": "adopted",
            "adopted": True,
            "error": None,
                "H": _fixed_probe_h(),
        }
    ]


@patch("agent.nodes.downward_probe._train_and_eval")
@patch("agent.nodes.downward_probe._llm_choose_model")
@patch("agent.nodes.downward_probe.filter_pool")
def test_keeps_current_when_smaller_fails_threshold(mock_fp, mock_choose, mock_te):
    from agent.nodes.downward_probe import downward_probe_node
    smaller = _model(tier=1, model_id="test/Small")
    mock_fp.return_value = [smaller]
    mock_choose.return_value = smaller
    mock_te.return_value = ("/small/ckpt", EvalResult(
        f1=0.70, per_class={}, failures=[]))
    current = _model(tier=2, model_id="test/Big")
    out = downward_probe_node(_state(current))
    assert out["selected_model"].model_id == "test/Big"  # unchanged
    assert out["best_weights_ref"] == "/big/ckpt"        # unchanged
    assert out["next_action"] == "terminate"
    assert out["downward_probe_history"]["attempts"] == [
        {
            "selector": smaller.selector,
            "model_id": smaller.model_id,
            "quant": None,
            "tier": 1,
            "score": 0.70,
            "weights_ref": "/small/ckpt",
            "result": "rejected",
            "adopted": False,
            "error": None,
                "H": _fixed_probe_h(),
        }
    ]


@patch("agent.nodes.downward_probe.filter_pool")
def test_no_lower_tier_candidates_skips(mock_fp):
    from agent.nodes.downward_probe import downward_probe_node
    mock_fp.return_value = []  # nothing feasible
    out = downward_probe_node(_state(_model(tier=2)))
    assert out["selected_model"].tier == 2  # unchanged
    assert out["next_action"] == "terminate"


def test_tiers_already_explored_reads_model_baselines():
    from agent.nodes.downward_probe import tiers_already_explored

    tier1 = _model(tier=1, model_id="test/Lower")
    tier2 = _model(tier=2, model_id="test/Current")
    state = {
        "model_baselines": [
            {"selector": tier1.selector, "baseline_f1": 0.5, "best_finetuned_f1": 0.69},
        ],
    }
    assert tiers_already_explored(state, [tier1, tier2]) == {1}


def test_tiers_already_explored_ignores_selectors_no_longer_feasible():
    from agent.nodes.downward_probe import tiers_already_explored

    tier2 = _model(tier=2, model_id="test/Current")
    state = {"model_baselines": [{"selector": "test/Gone@Q4_K_M"}]}
    assert tiers_already_explored(state, [tier2]) == set()


@patch("agent.nodes.downward_probe._train_and_eval")
@patch("agent.nodes.downward_probe._llm_choose_model")
@patch("agent.nodes.downward_probe.filter_pool")
def test_downward_probe_skips_tier_already_trained_by_main_ladder(
    mock_fp, mock_choose, mock_train
):
    """The math run's real bug: tier 2 was the run's OWN starting tier (59 real
    iterations, best F1 0.6925) before escalating to tier 3. On convergence at tier 3,
    the downward probe re-selected tier 2 anyway — `downward_tiers_tried` only tracks
    tiers the probe itself already tried, not tiers the main ladder already covered —
    and retrained it with a single arbitrary fixed config, scoring 0.54: strictly worse
    than, and no more informative than, the 0.6925 already on record for that exact
    model. Probing a tier with a known best score below threshold can never change the
    outcome, so it must be skipped, not re-run.
    """
    from agent.nodes.downward_probe import downward_probe_node

    tier1 = _model(tier=1, model_id="test/Lower")
    tier2 = _model(tier=2, model_id="test/Current")
    mock_fp.return_value = [tier1, tier2]
    state = _state(tier2)
    state["model_baselines"] = [
        {"selector": tier1.selector, "baseline_f1": 0.5, "best_finetuned_f1": 0.6925},
    ]

    out = downward_probe_node(state)

    assert out["next_action"] == "terminate"
    mock_choose.assert_not_called()
    mock_train.assert_not_called()


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "interpolation"})
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
def test_iterate_routes_to_downward_probe_on_success(_mock):
    # downward_probe only fires for strategies that don't start at the bottom of the
    # feasible set (interpolation / orchestrator_choice) — Q3/B?. With smallest_first it
    # is correctly skipped (already smallest) and the run terminates instead.
    from agent.nodes.iterate import iterate_node
    m = _model(tier=2)
    state = {
        "selected_model": m, "scores": [0.95], "best_score": 0.95, "iteration": 3,
        "turn_budget": 1000, "stop_threshold": 0.90, "initial_stop_threshold": 0.90,
        "task": "clinc150", "last_eval": None, "hw_gating_enabled": False,
        "downward_probe_done": False,
    }
    out = iterate_node(state)
    assert out["next_action"] == "downward_probe"
    _mock.assert_not_called()


@patch.dict("os.environ", {"SLM_MODEL_SELECTION_STRATEGY": "smallest_first"})
@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
def test_iterate_skips_downward_probe_at_the_lowest_tier(_mock):
    """Policy 2026-08-05: regression is structural, not strategy-gated.

    At tier 0 there is nothing below, so a converged run terminates. Under `smallest_first` the
    run starts at tier 0, which is why it normally never regresses — but that is a consequence of
    where it started, not a rule about the strategy.
    """
    from agent.nodes.iterate import iterate_node
    m = _model(tier=0)
    state = {
        "selected_model": m, "scores": [0.95], "best_score": 0.95, "iteration": 3,
        "turn_budget": 1000, "stop_threshold": 0.90, "initial_stop_threshold": 0.90,
        "task": "clinc150", "last_eval": None, "hw_gating_enabled": False,
        "downward_probe_done": False,
    }
    out = iterate_node(state)
    assert out["next_action"] == "terminate"
    _mock.assert_not_called()


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
def test_iterate_probes_down_after_escalation_even_under_smallest_first(_mock):
    """A smallest_first run that ESCALATED must be able to come back down again."""
    from agent.nodes.iterate import iterate_node
    m = _model(tier=2)
    state = {
        "selected_model": m, "scores": [0.95], "best_score": 0.95, "iteration": 3,
        "turn_budget": 1000, "stop_threshold": 0.90, "initial_stop_threshold": 0.90,
        "task": "clinc150", "last_eval": None, "hw_gating_enabled": False,
        "downward_probe_done": False,
    }
    out = iterate_node(state)
    assert out["next_action"] == "downward_probe"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
def test_iterate_terminates_after_probe_done(_mock):
    from agent.nodes.iterate import iterate_node
    m = _model(tier=2)
    state = {
        "selected_model": m, "scores": [0.95], "best_score": 0.95, "iteration": 3,
        "turn_budget": 1000, "stop_threshold": 0.90, "initial_stop_threshold": 0.90,
        "task": "clinc150", "last_eval": None, "hw_gating_enabled": False,
        "downward_probe_done": True,  # already probed
    }
    out = iterate_node(state)
    assert out["next_action"] == "terminate"
    _mock.assert_not_called()


@pytest.mark.parametrize(
    "strategy",
    ("interpolation", "orchestrator_choice"),
)
def test_iterate_routes_downward_only_after_threshold_with_untried_lower_tier(
    monkeypatch,
    strategy,
):
    """The threshold is what gates the downward probe, not the availability of a lower tier.

    The below-threshold half of this case DOES consult the orchestrator, so it is stubbed with a
    valid decision: an unusable one raises rather than routing anywhere at all (B316), which would
    prove nothing about where a below-threshold turn goes.
    """
    from agent.nodes.iterate import iterate_node

    monkeypatch.setenv("SLM_MODEL_SELECTION_STRATEGY", strategy)
    current = _model(tier=2, model_id="test/Current")
    lower = _model(tier=1, model_id="test/Lower")
    state = {
        "selected_model": current,
        "feasible_models": [lower, current],
        "downward_tiers_tried": [],
        "scores": [0.90],
        "best_score": 0.90,
        "iteration": 3,
        "turn_budget": 1000,
        "stop_threshold": 0.90,
        "initial_stop_threshold": 0.90,
        "task": "clinc150",
        "last_eval": None,
        "hw_gating_enabled": False,
        "downward_probe_done": False,
    }
    with patch(
        "agent.nodes.iterate._llm_iterate",
        return_value=_valid_hyperparameter_decision(),
    ) as llm:
        at_threshold = iterate_node(dict(state))
        below_threshold = iterate_node(
            {
                **state,
                "scores": [0.89],
                "best_score": 0.89,
            }
        )

    assert at_threshold["next_action"] == "downward_probe"
    assert below_threshold["next_action"] != "downward_probe"
    # Exactly once: the converged turn routes deterministically and spends no API call.
    assert llm.call_count == 1


@pytest.mark.parametrize(
    ("feasible_models", "tried"),
    (
        ([_model(tier=2, model_id="test/Current")], []),
        (
            [
                _model(tier=1, model_id="test/Lower"),
                _model(tier=2, model_id="test/Current"),
            ],
            [1],
        ),
    ),
)
@pytest.mark.parametrize(
    "strategy",
    ("interpolation", "orchestrator_choice"),
)
def test_iterate_skips_downward_when_no_lower_untried_tier(
    monkeypatch,
    feasible_models,
    tried,
    strategy,
):
    from agent.nodes.iterate import iterate_node

    monkeypatch.setenv("SLM_MODEL_SELECTION_STRATEGY", strategy)
    current = next(model for model in feasible_models if model.tier == 2)
    state = {
        "selected_model": current,
        "feasible_models": feasible_models,
        "downward_tiers_tried": tried,
        "scores": [0.95],
        "best_score": 0.95,
        "iteration": 3,
        "turn_budget": 1000,
        "stop_threshold": 0.90,
        "initial_stop_threshold": 0.90,
        "task": "clinc150",
        "last_eval": None,
        "hw_gating_enabled": False,
        "downward_probe_done": False,
    }
    with patch(
        "agent.nodes.iterate._llm_iterate",
        side_effect=Exception("skip llm"),
    ) as llm:
        out = iterate_node(state)

    assert out["next_action"] == "terminate"
    llm.assert_not_called()


@pytest.mark.parametrize(
    "strategy",
    ("interpolation", "orchestrator_choice"),
)
def test_iterate_skips_downward_for_tier_already_trained_by_main_ladder(
    monkeypatch,
    strategy,
):
    """Same bug, exercised at iterate_node's routing gate rather than inside the probe
    itself: the gate must not even ROUTE to downward_probe when the only lower tier is
    one the main ladder already trained (recorded in model_baselines), even though
    `downward_tiers_tried` — probe-only history — is empty.
    """
    from agent.nodes.iterate import iterate_node

    monkeypatch.setenv("SLM_MODEL_SELECTION_STRATEGY", strategy)
    lower = _model(tier=1, model_id="test/Lower")
    current = _model(tier=2, model_id="test/Current")
    state = {
        "selected_model": current,
        "feasible_models": [lower, current],
        "downward_tiers_tried": [],
        "model_baselines": [
            {"selector": lower.selector, "baseline_f1": 0.5, "best_finetuned_f1": 0.69},
        ],
        "scores": [0.95],
        "best_score": 0.95,
        "iteration": 3,
        "turn_budget": 1000,
        "stop_threshold": 0.90,
        "initial_stop_threshold": 0.90,
        "task": "clinc150",
        "last_eval": None,
        "hw_gating_enabled": False,
        "downward_probe_done": False,
    }
    with patch(
        "agent.nodes.iterate._llm_iterate",
        side_effect=Exception("skip llm"),
    ) as llm:
        out = iterate_node(state)

    assert out["next_action"] == "terminate"
    llm.assert_not_called()


def test_downward_probe_tracks_and_adopts_multiple_successful_tiers():
    from agent.nodes.downward_probe import downward_probe_node

    current = _model(tier=3, model_id="test/Tier3")
    candidates = [
        _model(tier=0, model_id="test/Tier0"),
        _model(tier=1, model_id="test/Tier1"),
        _model(tier=2, model_id="test/Tier2"),
        current,
    ]
    scores_by_tier = {
        2: 0.94,
        1: 0.93,
        0: 0.92,
    }

    def choose_candidate(**kwargs):
        return kwargs["candidates"][0]

    def train_and_eval(candidate, *_args):
        score = scores_by_tier[candidate.tier]
        return (
            f"/tier-{candidate.tier}/ckpt",
            EvalResult(
                f1=score,
                per_class={},
                failures=[],
            ),
        )

    with (
        patch("agent.nodes.downward_probe.filter_pool", return_value=candidates),
        patch(
            "agent.nodes.downward_probe._llm_choose_model",
            side_effect=choose_candidate,
        ),
        patch(
            "agent.nodes.downward_probe._train_and_eval",
            side_effect=train_and_eval,
        ) as train_eval,
    ):
        state = _state(current)
        state.update({
            "iteration": 5,
            "scores": [0.80, 0.90, 0.95],
            "dag": [{"iteration": 5, "model_id": current.model_id, "score": 0.95}],
        })
        out = downward_probe_node(state)

    assert [call.args[0].tier for call in train_eval.call_args_list] == [2, 1, 0]
    assert out["downward_tiers_tried"] == [0, 1, 2]
    assert out["selected_model"].tier == 0
    assert out["best_weights_ref"] == "/tier-0/ckpt"
    assert out["converged_model_ref"]["selector"] == out["selected_model"].selector
    assert out["next_action"] == "terminate"
    attempts = out["downward_probe_history"]["attempts"]
    assert [attempt["selector"] for attempt in attempts] == [
        candidates[2].selector,
        candidates[1].selector,
        candidates[0].selector,
    ]
    assert [attempt["result"] for attempt in attempts] == [
        "adopted",
        "adopted",
        "adopted",
    ]
    assert [attempt["weights_ref"] for attempt in attempts] == [
        "/tier-2/ckpt",
        "/tier-1/ckpt",
        "/tier-0/ckpt",
    ]


def test_downward_probe_records_failed_attempt_before_exiting():
    from agent.nodes.downward_probe import downward_probe_node

    current = _model(tier=2, model_id="test/Current")
    lower = _model(tier=1, model_id="test/Lower", quant="Q4_K_M")
    with (
        patch(
            "agent.nodes.downward_probe.filter_pool",
            return_value=[lower, current],
        ),
        patch(
            "agent.nodes.downward_probe._llm_choose_model",
            return_value=lower,
        ),
        patch(
            "agent.nodes.downward_probe._train_and_eval",
            side_effect=RuntimeError("llama-quantize missing"),
        ),
    ):
        out = downward_probe_node(_state(current))

    assert out["downward_probe_history"]["attempts"] == [
        {
            "selector": lower.selector,
            "model_id": lower.model_id,
            "quant": "Q4_K_M",
            "tier": 1,
            "score": None,
            "weights_ref": None,
            "result": "error",
            "adopted": False,
            "error": "RuntimeError: llama-quantize missing",
                "H": _fixed_probe_h(),
        }
    ]


def test_downward_model_chooser_fatal_error_preserves_converged_result(
    capsys,
):
    from agent.llm_errors import FatalLLMError
    from agent.nodes.downward_probe import downward_probe_node

    current = _model(tier=2, model_id="test/Current", quant="Q8_0")
    lower = _model(tier=1, model_id="test/Lower", quant="Q4_K_M")
    state = _state(current)
    original_weights = state["best_weights_ref"]
    original_score = state["best_score"]
    with (
        patch(
            "agent.nodes.downward_probe.filter_pool",
            return_value=[lower, current],
        ),
        patch(
            "agent.nodes.downward_probe._llm_choose_model",
            side_effect=FatalLLMError("provider permission denied"),
        ),
        patch("agent.nodes.downward_probe._train_and_eval") as train_eval,
    ):
        out = downward_probe_node(state)

    train_eval.assert_not_called()
    assert out["selected_model"] is current
    assert out["best_weights_ref"] == original_weights
    assert out["best_score"] == original_score
    assert out["downward_probe_done"] is True
    assert out["next_action"] == "terminate"
    assert out["downward_probe_history"]["attempts"] == []
    assert out["downward_probe_history"]["termination"] == {
        "stage": "model_chooser",
        "result": "skipped_error",
        "target_tier": 1,
        "candidate_selectors": [lower.selector],
        "reason": "FatalLLMError: provider permission denied",
    }
    rendered = capsys.readouterr().out
    assert "optional downward re-exploration skipped" in rendered
    assert "provider permission denied" in rendered


def test_downward_probe_step_checkpoints_choice_before_training():
    from agent.nodes.downward_probe import downward_probe_step_node

    current = _model(tier=2, model_id="test/Current")
    lower = _model(tier=1, model_id="test/Lower")
    with (
        patch(
            "agent.nodes.downward_probe.filter_pool",
            return_value=[lower, current],
        ),
        patch(
            "agent.nodes.downward_probe._llm_choose_model",
            return_value=lower,
        ),
        patch("agent.nodes.downward_probe._train_and_eval") as train_eval,
    ):
        out = downward_probe_step_node(_state(current))

    train_eval.assert_not_called()
    assert out["downward_probe_pending"] == {
        "selector": lower.selector,
        "target_tier": 1,
        "dataset_path": "/data.jsonl",
        "H": _fixed_probe_h(),
    }
    assert out["downward_probe_history"]["fixed_H"] == _fixed_probe_h()
    assert out["next_action"] == "downward_probe"
    assert out["downward_probe_done"] is False


def test_downward_probe_pending_retry_skips_the_model_choice_api():
    from agent.nodes.downward_probe import downward_probe_step_node

    current = _model(tier=2, model_id="test/Current")
    lower = _model(tier=1, model_id="test/Lower")
    state = _state(current)
    state["downward_probe_pending"] = {
        "selector": lower.selector,
        "target_tier": 1,
        "dataset_path": "/data.jsonl",
    }
    result = EvalResult(
        f1=0.92,
        per_class={},
        failures=[],
    )
    with (
        patch(
            "agent.nodes.downward_probe.filter_pool",
            return_value=[lower, current],
        ),
        patch("agent.nodes.downward_probe._llm_choose_model") as choose,
        patch(
            "agent.nodes.downward_probe._train_and_eval",
            return_value=("/lower/ckpt", result),
        ) as train_eval,
    ):
        out = downward_probe_step_node(state)

    choose.assert_not_called()
    train_eval.assert_called_once_with(
        lower,
        state,
        "/data.jsonl",
        _fixed_probe_h(),
    )
    assert out["downward_probe_pending"] is None
    assert out["downward_probe_history"]["attempts"][0]["adopted"] is True
    assert out["downward_probe_history"]["attempts"][0]["H"] == _fixed_probe_h()
    assert out["selected_model"] is lower


def test_downward_probe_is_unconditional_when_a_lower_tier_is_untried():
    """Policy 2026-08-05: no orchestrator gate — an untried lower tier is always probed.

    The gate used to spend an API call to sometimes decline the one thing the run exists to
    determine (the smallest model that clears the goal). Only structural conditions stop it.
    """
    from agent.nodes.downward_probe import downward_probe_node

    current = _model(tier=2, model_id="test/Current")
    lower = _model(tier=1, model_id="test/Lower")
    with (
        patch(
            "agent.nodes.downward_probe.filter_pool",
            return_value=[lower, current],
        ),
        patch(
            "agent.nodes.downward_probe._llm_choose_model", return_value=lower
        ) as choose,
        patch(
            "agent.nodes.downward_probe._train_and_eval",
            return_value=("/lower/ckpt", EvalResult(f1=0.80, per_class={}, failures=[])),
        ) as train_eval,
    ):
        downward_probe_node(_state(current))

    # It proceeded to pick and train a lower-tier candidate instead of declining.
    choose.assert_called_once()
    train_eval.assert_called_once()
