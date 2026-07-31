# tests/nodes/test_evaluate_node.py
import os
import pytest
from unittest.mock import patch, MagicMock
from config.android_pool import CapabilityMeasurement, ModelSpec, HardwareConstraints
from training.lora_trainer import TrainingOutput
from eval.harness import EvalResult
from agent.data_rebuild import normalize_data_rebuild_plan

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")


def _make_model(quant=None):
    return ModelSpec(
        model_id="test/Model-1B",
        size_mb=700,
        tier=1,
        capability_measurements=(
            CapabilityMeasurement(
                metric="MMLU", value=0.5, artifact="test/Model-1B",
                mode=None, protocol="test", source="https://example.test",
            ),
        ),
        quant=quant,
    )


def _make_state(quant=None):
    from data.eval_set import EvalSet
    eval_set = MagicMock(spec=EvalSet)
    eval_set.all = [{"text": f"ex{i}"} for i in range(10)]
    return {
        "task_type": "classification",
        "selected_model": _make_model(quant=quant),
        "eval_set": eval_set,
        "iteration": 2,
        "best_score": 0.0,
        "scores": [],
        "dag": [],
        "dataset_version": 1,
        "current_dataset_path": "/data.jsonl",
        "hardware_constraints": HardwareConstraints(
            storage_mb=5000, memory_mb=5000, latency_ttft_ms=3000,
        ),
        "model_baselines": [],
        "last_curation": {},
        "last_hypothesis": "",
        "_pending_weights_refs": {"main": "/ckpt"},
        "_pending_training_outputs": {"main": TrainingOutput(weights_ref="/ckpt", gguf_path=None)},
        "_pending_configs": {"main": {
            "label": "main",
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "weight_decay": 0.1,
            "learning_rate": 2e-4,
            "nr_epochs": 3,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "batch_size": 2,
        }},
        "consecutive_no_improvement": 0,
    }


def _mock_eval_result(f1=0.85):
    return EvalResult(f1=f1, per_class={}, pos_score=0.9, neg_score=0.8, boundary_score=0.85, failures=[])


def test_gguf_build_delegates_to_disposable_worker_when_enabled(monkeypatch):
    monkeypatch.setenv("SLM_CUDA_ISOLATION", "1")
    monkeypatch.delenv("SLM_CUDA_WORKER", raising=False)
    with patch("training.cuda_isolation.run_isolated", return_value="/gguf/model.gguf") as worker:
        from agent.nodes.evaluate import _build_gguf_for_eval

        result = _build_gguf_for_eval("/ckpt", "test/Model-1B", "Q4_K_M", "label")

    assert result == "/gguf/model.gguf"
    worker.assert_called_once_with(
        "build_gguf",
        {
            "weights_ref": "/ckpt",
            "model_id": "test/Model-1B",
            "quant": "Q4_K_M",
            "mlabel": "label",
        },
    )


def test_gguf_build_failure_does_not_silently_score_bf16(monkeypatch):
    monkeypatch.delenv("SLM_CUDA_ISOLATION", raising=False)
    with patch("agent.nodes.evaluate._build_or_reuse_gguf", return_value=None):
        from agent.nodes.evaluate import _build_gguf_for_eval

        with pytest.raises(RuntimeError, match="quantized GGUF"):
            _build_gguf_for_eval("/ckpt", "test/Model-1B", "Q4_K_M", "label")


def test_gguf_build_reuses_exact_cached_weights_and_quant():
    from agent.nodes.evaluate import _build_or_reuse_gguf

    with (
        patch(
            "agent.nodes.evaluate.validated_gguf_cache_hit",
            return_value=True,
        ),
        patch("agent.nodes.evaluate.merge_for_quantization") as merge,
        patch("agent.nodes.evaluate.quantize_from_model_spec") as quantize,
    ):
        result = _build_or_reuse_gguf(
            "/same/weights",
            "test/Model-1B",
            "Q8_0",
            "test/Model-1B [Q8_0]",
        )

    assert result.endswith("/model-q8_0.gguf")
    merge.assert_not_called()
    quantize.assert_not_called()


def test_base_model_gguf_converts_pinned_snapshot_without_merge_or_deletion(
    monkeypatch,
    tmp_path,
):
    from agent.nodes.evaluate import _build_or_reuse_gguf

    monkeypatch.chdir(tmp_path)
    snapshot = "/shared/hf/hub/models--Qwen--Qwen3.5-2B/snapshots/abc123"
    with (
        patch(
            "agent.nodes.evaluate.validated_gguf_cache_hit",
            return_value=False,
        ),
        patch(
            "agent.nodes.evaluate.resolve_hf_snapshot",
            return_value=snapshot,
        ) as resolve,
        patch("agent.nodes.evaluate.merge_for_quantization") as merge,
        patch(
            "agent.nodes.evaluate.quantize_from_model_spec",
            return_value="/gguf/base-q4_k_m.gguf",
        ) as quantize,
        patch(
            "agent.nodes.evaluate.validate_and_record_gguf",
        ) as validate,
        patch("agent.nodes.evaluate.shutil.rmtree") as rmtree,
    ):
        result = _build_or_reuse_gguf(
            "Qwen/Qwen3.5-2B",
            "Qwen/Qwen3.5-2B",
            "Q4_K_M",
            "Qwen/Qwen3.5-2B [Q4_K_M]",
        )

    assert result == "/gguf/base-q4_k_m.gguf"
    resolve.assert_called_once_with("Qwen/Qwen3.5-2B")
    quantize.assert_called_once()
    assert quantize.call_args.args[0] == snapshot
    merge.assert_not_called()
    validate.assert_called_once_with("/gguf/base-q4_k_m.gguf")
    rmtree.assert_not_called()


def test_unvalidated_missing_layer_cache_is_invalidated_and_rebuilt(
    monkeypatch,
    tmp_path,
):
    import hashlib

    from agent.nodes.evaluate import _build_or_reuse_gguf

    monkeypatch.chdir(tmp_path)
    model_id = "Qwen/Qwen3.5-2B"
    quant = "Q4_K_M"
    wkey = hashlib.sha1(f"{model_id}|{quant}".encode()).hexdigest()[:12]
    cached = (
        tmp_path
        / "artifacts"
        / "gguf"
        / "Qwen_Qwen3.5-2B"
        / wkey
        / "model-q4_k_m.gguf"
    )
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"320 tensors; missing blk.24.attn_norm.weight")

    def rebuild(_source, _output_dir, _quant):
        cached.write_bytes(b"335 tensors; complete")
        return str(cached.relative_to(tmp_path))

    with (
        patch(
            "agent.nodes.evaluate.resolve_hf_snapshot",
            return_value="/shared/snapshot/abc123",
        ),
        patch(
            "agent.nodes.evaluate.quantize_from_model_spec",
            side_effect=rebuild,
        ) as quantize,
        patch(
            "agent.nodes.evaluate.validate_and_record_gguf",
        ) as validate,
    ):
        result = _build_or_reuse_gguf(
            model_id,
            model_id,
            quant,
            "Qwen/Qwen3.5-2B [Q4_K_M]",
        )

    quantize.assert_called_once()
    validate.assert_called_once_with(result)
    assert cached.read_bytes() == b"335 tensors; complete"


def test_qwen35_gguf_build_merges_multimodal_adapter_before_exact_quant():
    from agent.nodes.evaluate import _build_or_reuse_gguf

    with (
        patch(
            "agent.nodes.evaluate.validated_gguf_cache_hit",
            return_value=False,
        ),
        patch(
            "agent.nodes.evaluate.merge_for_quantization",
            return_value="/merged/qwen35",
        ) as merge,
        patch(
            "agent.nodes.evaluate.quantize_from_model_spec",
            return_value="/gguf/qwen35-q4_k_m.gguf",
        ) as quantize,
        patch(
            "agent.nodes.evaluate.validate_and_record_gguf",
        ) as validate,
    ):
        result = _build_or_reuse_gguf(
            "/adapters/qwen35",
            "Qwen/Qwen3.5-0.8B",
            "Q4_K_M",
            "Qwen/Qwen3.5-0.8B [Q4_K_M]",
        )

    assert result == "/gguf/qwen35-q4_k_m.gguf"
    assert merge.call_args.args[0] == "/adapters/qwen35"
    quantize.assert_called_once()
    assert quantize.call_args.args[0] == "/merged/qwen35"
    assert quantize.call_args.args[2] == "Q4_K_M"
    validate.assert_called_once_with("/gguf/qwen35-q4_k_m.gguf")


@patch("agent.nodes.evaluate.config.QUANT_ACCURACY_EVAL", True)
@patch("agent.nodes.evaluate.apply_iteration_policy", return_value={"band": "good", "intervention": "hyperparameter"})
@patch("agent.nodes.evaluate._build_gguf_for_eval")
@patch("agent.nodes.evaluate.run_eval")
@patch("agent.nodes.evaluate.CurationLog")
@patch("agent.nodes.evaluate.theoretical_hardware_profile", return_value={"size_mb": 700, "tier": 1})
@patch("config.android_pool.check_hardware_constraints", return_value={"storage": {"pass": True}, "memory": {"pass": True}, "latency": {"pass": True}, "power": {"pass": True}})
def test_quantized_zero_shot_baseline_uses_selected_deployment_quant(
    mock_hw,
    mock_profile,
    mock_log,
    mock_eval,
    mock_build,
    mock_policy,
):
    from agent.nodes.evaluate import evaluate_node

    state = _make_state(quant="Q4_K_M")
    state["iteration"] = 1
    mock_log.return_value.write_iteration = MagicMock()
    mock_build.side_effect = ["/gguf/base-q4.gguf", "/gguf/adapter-q4.gguf"]
    mock_eval.side_effect = [_mock_eval_result(0.90), _mock_eval_result(0.80)]

    out = evaluate_node(state)

    assert mock_build.call_args_list[0].args == (
        "test/Model-1B",
        "test/Model-1B",
        "Q4_K_M",
        "test/Model-1B [Q4_K_M]",
    )
    baseline_call = mock_eval.call_args_list[0]
    assert baseline_call.kwargs["quant"] == "Q4_K_M"
    assert baseline_call.kwargs["gguf_path"] == "/gguf/base-q4.gguf"
    assert out["model_baselines"][0]["selector"] == "test/Model-1B@Q4_K_M"
    assert out["best_weights_ref"] == "test/Model-1B"
    assert out["dag"][-1]["pi"]["H"]["lora_rank"] is None
    assert out["dag"][-1]["trained_configs"] == [{
        "label": "main",
        "score": 0.80,
        "H": {
            key: value
            for key, value in state["_pending_configs"]["main"].items()
            if key != "label"
        },
    }]


@patch("agent.nodes.evaluate.run_eval")
def test_generation_baseline_reraises_local_judge_infrastructure_error(mock_eval):
    from agent.nodes.evaluate import evaluate_node
    from eval.judge_client import JudgeInfrastructureError

    state = _make_state(quant=None)
    state["task_type"] = "generation"
    state["iteration"] = 1
    mock_eval.side_effect = [
        JudgeInfrastructureError("local judge unavailable"),
        AssertionError("trained evaluation must not run after baseline judge failure"),
    ]

    with pytest.raises(JudgeInfrastructureError, match="local judge unavailable"):
        evaluate_node(state)

    assert mock_eval.call_count == 1


@patch("agent.nodes.evaluate._build_gguf_for_eval")
@patch("agent.nodes.evaluate.run_eval")
def test_quantized_baseline_reraises_quantization_infrastructure_error(
    mock_eval,
    mock_build,
):
    from agent.nodes.evaluate import evaluate_node
    from training.quantize import QuantizationInfrastructureError

    state = _make_state(quant="Q4_K_M")
    state["iteration"] = 1
    mock_build.side_effect = QuantizationInfrastructureError(
        "missing tensor blk.24.attn_norm.weight"
    )

    with pytest.raises(
        QuantizationInfrastructureError,
        match=r"blk\.24\.attn_norm\.weight",
    ):
        evaluate_node(state)

    mock_eval.assert_not_called()


@patch("agent.nodes.evaluate.apply_iteration_policy", return_value={"band": "good", "intervention": "hyperparameter"})
@patch("agent.nodes.evaluate.run_eval")
@patch("agent.nodes.evaluate.CurationLog")
@patch("agent.nodes.evaluate.theoretical_hardware_profile", return_value={"size_mb": 700, "tier": 1})
@patch("config.android_pool.check_hardware_constraints", return_value={"storage": {"pass": True}, "memory": {"pass": True}, "latency": {"pass": True}, "power": {"pass": True}})
def test_evaluate_node_uses_bf16_path_when_quant_none(mock_hw, mock_profile, mock_log, mock_eval, mock_policy):
    mock_eval.return_value = _mock_eval_result()
    mock_log.return_value.write_iteration = MagicMock()
    from agent.nodes.evaluate import evaluate_node
    state = _make_state(quant=None)
    plan = normalize_data_rebuild_plan(
        {
            "strategy": "resample",
            "target_rows": 64,
        },
        task_type="classification",
        hypothesis="rebalance aggregate classes",
    )
    state["data_rebuild_plan"] = plan
    state["last_curation"] = {
        "total_examples": 12,
        "n_gold": 5,
        "n_hard": 3,
        "n_hard_source": 2,
        "n_hard_generated": 3,
        "label_dist": {"a": 6, "b": 6},
        "data_rebuild_plan": plan,
        "rebuild_config": {"target_rows": 64, "seed": 17},
        "strategy_composition": [{
            "strategy": "resample",
            "rows": 12,
        }],
        "plan_yield": {"status": "novel", "novel_rows": 12},
    }
    out = evaluate_node(state)
    # run_eval called without gguf_path (or None)
    mock_eval.assert_called_once()
    call_kwargs = mock_eval.call_args[1]
    assert call_kwargs.get("gguf_path") is None
    assert call_kwargs.get("quant") is None
    log_kwargs = mock_log.return_value.write_iteration.call_args.kwargs
    assert log_kwargs["total_examples"] == 12
    assert log_kwargs["n_hard_source"] == 2
    assert log_kwargs["n_hard_generated"] == 3
    assert out["dag"][-1]["pi"]["H"] == {
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.1,
        "learning_rate": 2e-4,
        "nr_epochs": 3,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 8,
        "batch_size": 2,
    }
    assert out["dag"][-1]["pi"]["S"]["loss_masking"] == "assistant_only"
    assert out["dag"][-1]["pi"]["S"]["loss_contract_version"] >= 2
    dataset_identity = out["dag"][-1]["pi"]["D"]
    assert dataset_identity["path"] == "/data.jsonl"
    assert dataset_identity["version"] == 1
    assert dataset_identity["plan"] == plan
    assert dataset_identity["config"] == {"target_rows": 64, "seed": 17}
    assert dataset_identity["composition"] == state["last_curation"]
    assert out["dag"][-1]["evaluation_state"]["last_eval"]["f1"] == 0.85
    assert "test_report" in out["dag"][-1]["evaluation_state"]


@patch("agent.nodes.evaluate.apply_iteration_policy", return_value={"band": "good", "intervention": "hyperparameter"})
@patch("agent.nodes.evaluate.run_eval")
@patch("agent.nodes.evaluate.CurationLog")
@patch("agent.nodes.evaluate.theoretical_hardware_profile", return_value={"size_mb": 700, "tier": 1})
@patch("config.android_pool.check_hardware_constraints", return_value={"storage": {"pass": True}, "memory": {"pass": True}, "latency": {"pass": True}, "power": {"pass": True}})
def test_evaluate_normalizes_legacy_checkpoint_config_into_complete_dag_h(
    mock_hw,
    mock_profile,
    mock_log,
    mock_eval,
    mock_policy,
):
    mock_eval.return_value = _mock_eval_result()
    mock_log.return_value.write_iteration = MagicMock()
    from agent.nodes.evaluate import evaluate_node

    state = _make_state(quant=None)
    state["_pending_configs"] = {
        "main": {
            "label": "legacy",
            "lora_rank": 8,
            "learning_rate": 2e-4,
            "nr_epochs": 3,
            "batch_size": 4,
        }
    }

    out = evaluate_node(state)

    assert out["dag"][-1]["pi"]["H"] == {
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "weight_decay": 0.01,
        "learning_rate": 2e-4,
        "nr_epochs": 3,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 4,
        "batch_size": 4,
    }


@patch("agent.nodes.evaluate.config.HW_ONDEVICE_BACKEND", "smolchat")
@patch("agent.nodes.evaluate.apply_iteration_policy", return_value={"band": "good", "intervention": "hyperparameter"})
@patch("agent.nodes.evaluate.quantize_from_model_spec", return_value="/gguf/model.gguf")
@patch("agent.nodes.evaluate.validate_and_record_gguf")
@patch("agent.nodes.evaluate.merge_for_quantization", return_value="/merged/checkpoint")
@patch("agent.nodes.evaluate.run_eval")
@patch("agent.nodes.evaluate.CurationLog")
@patch("agent.nodes.evaluate.theoretical_hardware_profile", return_value={"size_mb": 700, "tier": 1})
@patch("config.android_pool.check_hardware_constraints", return_value={"storage": {"pass": True}, "memory": {"pass": True}, "latency": {"pass": True}, "power": {"pass": True}})
def test_evaluate_node_quantizes_and_uses_gguf_path_when_quant_set(
    mock_hw,
    mock_profile,
    mock_log,
    mock_eval,
    mock_merge,
    mock_validate,
    mock_quantize,
    mock_policy,
):
    mock_eval.return_value = _mock_eval_result()
    mock_log.return_value.write_iteration = MagicMock()
    from agent.nodes.evaluate import evaluate_node
    state = _make_state(quant="Q4_K_M")
    evaluate_node(state)
    # merge was called with weights_ref
    mock_merge.assert_called_once()
    assert mock_merge.call_args[0][0] == "/ckpt"
    merge_call_args = mock_merge.call_args[0]
    # B160/B161: GGUF cached by a hash of (weights_ref, quant) — not a per-iteration path,
    # and quant-aware so a Q8_0 request never reuses a Q4_K_M file.
    import hashlib
    wkey = hashlib.sha1(b"/ckpt|Q4_K_M").hexdigest()[:12]
    assert f"artifacts/merged/test_Model-1B/{wkey}" in merge_call_args[1].replace("\\", "/")
    # quantize was called with correct quant string
    mock_quantize.assert_called_once()
    assert mock_quantize.call_args[0][2] == "Q4_K_M"
    mock_validate.assert_called_once_with("/gguf/model.gguf")
    # run_eval called with gguf_path
    mock_eval.assert_called_once()
    call_kwargs = mock_eval.call_args[1]
    assert call_kwargs.get("gguf_path") == "/gguf/model.gguf"
    assert call_kwargs.get("quant") == "Q4_K_M"
