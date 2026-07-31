import json
from dataclasses import asdict
from pathlib import Path

import pytest

from config.android_pool import ANDROID_POOL, HardwareConstraints
from data.eval_set import EvalSet
from eval.harness import EvalResult

from agent.checkpoint import (
    CheckpointArtifactError,
    CheckpointCompatibilityError,
    CheckpointCorruptError,
    checkpoint_compatibility,
    load_checkpoint,
    runtime_config_fingerprint,
    runtime_config_snapshot,
    save_checkpoint,
    write_artifact_manifest,
)
from agent.state_codec import StateCodecError, decode_state, encode_state
from agent.data_rebuild import normalize_data_rebuild_plan

# Opaque plan-identity string: identity is no longer computed by the pipeline (the
# redesign dropped plan dedup), but the state field still round-trips through the codec.
_PLAN_ID = "plan-hash-test"


def _state(tmp_path: Path) -> dict:
    from agent.nodes.downward_probe import DOWNWARD_PROBE_H

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"hello","label":"ok"}\n', encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "adapter_config.json").write_text("{}", encoding="utf-8")
    (weights / "adapter_model.safetensors").write_bytes(b"weights-v1")
    write_artifact_manifest(weights, artifact_type="training_checkpoint")
    model = ANDROID_POOL[0]
    eval_set = EvalSet(
        pos=[{"text": "p", "label": "yes"}],
        neg=[{"text": "n", "label": "no"}],
        boundary=[],
        task_type="classification",
        multi_label=False,
        schema={"label": "string"},
        multilingual=False,
    )
    result = EvalResult(
        f1=0.75,
        per_class={"yes": 0.8},
        pos_score=0.8,
        neg_score=0.7,
        boundary_score=0.0,
        failures=[{"text": "n"}],
        execution_diagnostics=[{"status": "ok"}],
    )
    rebuild_plan = normalize_data_rebuild_plan(
        {
            "strategy": "resample",
            "target_rows": 64,
        },
        task_type="classification",
        hypothesis="preserve the winning rows while resampling",
    )
    curation = {
        "total_examples": 1,
        "data_rebuild_plan": rebuild_plan,
        "data_rebuild_plan_identity": _PLAN_ID,
        "rebuild_config": {"target_rows": 64, "seed": 19},
        "strategy_composition": [{
            "strategy": "resample",
            "rows": 1,
        }],
        "plan_yield": {"status": "novel", "novel_rows": 1},
    }
    return {
        "description": "round trip",
        "hardware_constraints": HardwareConstraints(
            storage_mb=8192,
            memory_mb=4096,
            latency_ttft_ms=2000,
            power_watts=6.5,
            target_chip="snapdragon_8gen3",
            min_tok_s=6.0,
        ),
        "selected_model": model,
        "feasible_models": [model, ANDROID_POOL[1]],
        "eval_set": eval_set,
        "last_eval": result,
        "scores": [0.5, 0.75],
        "current_dataset_path": str(dataset),
        "dataset_version": 3,
        "last_curation": curation,
        "data_rebuild_plan": rebuild_plan,
        "data_rebuild_plan_identity": _PLAN_ID,
        "source_acquire_rounds_used": 2,
        "curation_log_path": str(tmp_path / "data-curation.md"),
        "test_report": {
            "overall": 0.75,
            "confusion_pairs": [
                {"gold": "yes", "predicted": "no", "count": 1},
            ],
        },
        "best_weights_ref": str(weights),
        "_pending_weights_refs": {"candidate": str(weights)},
        "_pending_configs": {
            "candidate": {
                "lora_rank": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "weight_decay": 0.1,
                "learning_rate": 0.0002,
                "nr_epochs": 6,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "effective_batch_size": 16,
                "batch_size": 2,
            }
        },
        "dag": [{
            "pi": {
                "D": {
                    "version": 3,
                    "path": str(dataset),
                    "plan": rebuild_plan,
                    "plan_identity": _PLAN_ID,
                    "config": curation["rebuild_config"],
                    "composition": curation,
                },
                "H": {
                    "lora_rank": 8,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "weight_decay": 0.1,
                    "learning_rate": 0.0002,
                    "nr_epochs": 6,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 8,
                    "effective_batch_size": 16,
                    "batch_size": 2,
                },
            },
            "evaluation_state": {
                "last_eval": asdict(result),
                "test_report": {
                    "overall": 0.75,
                    "confusion_pairs": [
                        {"gold": "yes", "predicted": "no", "count": 1},
                    ],
                },
            },
        }],
        "downward_probe_history": {
            "origin": None,
            "fixed_H": dict(DOWNWARD_PROBE_H),
            "attempts": [{
                "selector": model.selector,
                "H": dict(DOWNWARD_PROBE_H),
                "result": "rejected",
            }],
        },
        "_pending_training_outputs": {"candidate": object()},
    }


def test_state_roundtrip_rehydrates_domain_objects_without_runtime_objects(tmp_path):
    state = _state(tmp_path)

    encoded = encode_state(state)
    decoded = decode_state(json.loads(json.dumps(encoded)))

    assert encoded["selected_model"] == state["selected_model"].selector
    assert encoded["feasible_models"] == [
        model.selector for model in state["feasible_models"]
    ]
    assert "_pending_training_outputs" not in encoded
    assert decoded["selected_model"] is state["selected_model"]
    assert decoded["feasible_models"] == state["feasible_models"]
    assert decoded["hardware_constraints"] == state["hardware_constraints"]
    assert decoded["eval_set"] == state["eval_set"]
    assert decoded["last_eval"] == state["last_eval"]
    assert decoded["_pending_configs"] == state["_pending_configs"]
    assert decoded["dag"] == state["dag"]
    assert decoded["data_rebuild_plan"] == state["data_rebuild_plan"]
    assert (
        decoded["data_rebuild_plan_identity"]
        == state["data_rebuild_plan_identity"]
    )
    assert decoded["last_curation"] == state["last_curation"]
    assert decoded["test_report"] == state["test_report"]
    assert decoded["source_acquire_rounds_used"] == 2
    assert decoded["curation_log_path"] == state["curation_log_path"]
    assert (
        decoded["downward_probe_history"]
        == state["downward_probe_history"]
    )
    assert decoded["_pending_training_outputs"] is None


def test_state_codec_rejects_unknown_runtime_objects():
    with pytest.raises(StateCodecError, match="runtime object"):
        encode_state({"trainer": object()})


def test_decode_requires_exact_model_selector(tmp_path):
    encoded = encode_state(_state(tmp_path))
    encoded["selected_model"] = encoded["selected_model"].split("@", 1)[0]

    with pytest.raises(StateCodecError, match="exact model selector"):
        decode_state(encoded)


def test_checkpoint_detects_pool_and_artifact_drift(tmp_path):
    state = _state(tmp_path)
    path = tmp_path / "checkpoint.json"
    compatibility = checkpoint_compatibility(
        mode="cold_start",
        pool_fingerprint="pool-a",
        topology_fingerprint="topology-a",
        config_fingerprint="config-a",
    )
    saved = save_checkpoint(
        path,
        state,
        thread_id="stable-thread",
        compatibility=compatibility,
        graph_steps=7,
        last_node="train",
        next_nodes=("evaluate",),
        cumulative_wall_time_s=12.5,
    )
    assert saved["progress"]["pending_weights_refs"] == state[
        "_pending_weights_refs"
    ]
    assert saved["progress"]["pending_configs"] == state["_pending_configs"]

    with pytest.raises(CheckpointCompatibilityError, match="pool fingerprint"):
        load_checkpoint(
            path,
            expected_compatibility={
                **compatibility,
                "pool_fingerprint": "pool-b",
            },
        )

    Path(state["current_dataset_path"]).write_text(
        '{"text":"drifted"}\n', encoding="utf-8"
    )
    with pytest.raises(CheckpointArtifactError, match="dataset"):
        load_checkpoint(path, expected_compatibility=compatibility)


def test_mined_train_examples_survive_checkpoint_roundtrip(tmp_path):
    state = _state(tmp_path)
    mined = {
        "text": "durable newly mined row",
        "label": "ok",
        "_source": "hf:source/train",
        "_source_record": {
            "kind": "hf",
            "id": "source",
            "split": "train",
            "role": "curriculum",
        },
        "_provenance": "mined_real",
        "_strategy_origin": "mine_new_real_source",
    }
    state["train_examples"] = [
        {"text": "original row", "label": "ok"},
        mined,
    ]
    compatibility = checkpoint_compatibility(
        mode="cold_start",
        pool_fingerprint="p",
        topology_fingerprint="t",
        config_fingerprint="c",
    )
    path = tmp_path / "checkpoint.json"

    save_checkpoint(
        path,
        state,
        thread_id="stable",
        compatibility=compatibility,
        graph_steps=2,
        last_node="curate",
        next_nodes=("train",),
        cumulative_wall_time_s=2.0,
    )
    restored = load_checkpoint(
        path,
        expected_compatibility=compatibility,
    )

    assert restored["state"]["train_examples"] == [
        {"text": "original row", "label": "ok"},
        mined,
    ]


def test_checkpoint_tracks_prior_datasets_needed_for_rollback(tmp_path):
    state = _state(tmp_path)
    prior = tmp_path / "dataset-v2.jsonl"
    prior.write_text('{"text":"prior","label":"ok"}\n', encoding="utf-8")
    prior_node = dict(state["dag"][0])
    prior_node["pi"] = {
        **state["dag"][0]["pi"],
        "D": {
            **state["dag"][0]["pi"]["D"],
            "version": 2,
            "path": str(prior),
        },
    }
    state["dag"] = [prior_node, state["dag"][0]]
    path = tmp_path / "checkpoint.json"
    compatibility = checkpoint_compatibility(
        mode="cold_start",
        pool_fingerprint="p",
        topology_fingerprint="t",
        config_fingerprint="c",
    )

    saved = save_checkpoint(
        path,
        state,
        thread_id="stable",
        compatibility=compatibility,
        graph_steps=2,
        last_node="evaluate",
        next_nodes=("iterate",),
        cumulative_wall_time_s=2.0,
    )

    assert {
        artifact["role"] for artifact in saved["artifacts"]
        if artifact["kind"] == "file"
    } >= {"dataset", "dag_dataset:v2", "dag_dataset:v3"}

    prior.write_text('{"text":"drifted","label":"ok"}\n', encoding="utf-8")
    with pytest.raises(CheckpointArtifactError, match="dag_dataset:v2"):
        load_checkpoint(path, expected_compatibility=compatibility)


def test_checkpoint_atomic_replace_preserves_previous_file_on_encode_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"old":true}\n', encoding="utf-8")

    def explode(_state):
        raise StateCodecError("boom")

    monkeypatch.setattr("agent.checkpoint.encode_state", explode)
    with pytest.raises(StateCodecError, match="boom"):
        save_checkpoint(
            path,
            {},
            thread_id="thread",
            compatibility=checkpoint_compatibility(
                mode="cold_start",
                pool_fingerprint="p",
                topology_fingerprint="t",
                config_fingerprint="c",
            ),
            graph_steps=0,
            last_node=None,
            next_nodes=("task_analysis",),
            cumulative_wall_time_s=0.0,
        )

    assert path.read_text(encoding="utf-8") == '{"old":true}\n'
    assert not list(tmp_path.glob("checkpoint.json.tmp.*"))


def test_corrupt_checkpoint_has_specific_error(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"schema_version":', encoding="utf-8")

    with pytest.raises(CheckpointCorruptError, match="valid JSON"):
        load_checkpoint(path)


def test_same_size_checkpoint_weight_corruption_is_detected(tmp_path):
    state = _state(tmp_path)
    path = tmp_path / "checkpoint.json"
    compatibility = checkpoint_compatibility(
        mode="cold_start",
        pool_fingerprint="p",
        topology_fingerprint="t",
        config_fingerprint="c",
    )
    save_checkpoint(
        path,
        state,
        thread_id="stable",
        compatibility=compatibility,
        graph_steps=1,
        last_node="train",
        next_nodes=("evaluate",),
        cumulative_wall_time_s=1.0,
    )
    weights = Path(state["best_weights_ref"]) / "adapter_model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(b"x" * len(original))

    with pytest.raises(CheckpointArtifactError, match="content hash"):
        load_checkpoint(path, expected_compatibility=compatibility)


def test_incomplete_final_checkpoint_cannot_be_published(tmp_path):
    checkpoint = tmp_path / "final_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")

    with pytest.raises(CheckpointArtifactError, match="adapter_config"):
        write_artifact_manifest(
            checkpoint,
            artifact_type="training_checkpoint",
        )


def test_checkpoint_capture_hashes_manifest_not_model_bytes(tmp_path, monkeypatch):
    state = _state(tmp_path)
    calls = []
    from agent import checkpoint as checkpoint_module

    original = checkpoint_module._sha256_file

    def track(path):
        calls.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", track)
    save_checkpoint(
        tmp_path / "checkpoint.json",
        state,
        thread_id="stable",
        compatibility=checkpoint_compatibility(
            mode="cold_start",
            pool_fingerprint="p",
            topology_fingerprint="t",
            config_fingerprint="c",
        ),
        graph_steps=1,
        last_node="train",
        next_nodes=("evaluate",),
        cumulative_wall_time_s=1.0,
    )

    assert "adapter_model.safetensors" not in calls
    assert ".slm_artifact_manifest.json" in calls


def test_resume_hashes_duplicate_checkpoint_path_only_once(tmp_path, monkeypatch):
    state = _state(tmp_path)
    compatibility = checkpoint_compatibility(
        mode="cold_start",
        pool_fingerprint="p",
        topology_fingerprint="t",
        config_fingerprint="c",
    )
    path = tmp_path / "checkpoint.json"
    save_checkpoint(
        path,
        state,
        thread_id="stable",
        compatibility=compatibility,
        graph_steps=1,
        last_node="train",
        next_nodes=("evaluate",),
        cumulative_wall_time_s=1.0,
    )
    calls = []
    from agent import checkpoint as checkpoint_module

    original = checkpoint_module._sha256_file

    def track(candidate):
        if Path(candidate).name == "adapter_model.safetensors":
            calls.append(str(candidate))
        return original(candidate)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", track)
    load_checkpoint(path, expected_compatibility=compatibility)

    assert len(calls) == 1


def test_resume_config_fingerprint_covers_state_affecting_runtime_settings(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-no-network")
    monkeypatch.setenv("EXA_API_KEY", "test-no-network")
    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "4096")
    monkeypatch.setenv("SLM_EVAL_MAX_NEW_TOKENS_NER", "512")
    monkeypatch.setenv("SLM_EVAL_MAX_NEW_TOKENS_APPS", "1024")
    monkeypatch.setenv("SLM_APPS_PROBLEM_TIMEOUT_S", "5")
    monkeypatch.setenv("SLM_STAGNATION_WINDOW", "41")
    monkeypatch.setenv("SLM_MAX_STALL_EVALS", "42")
    monkeypatch.setenv("SLM_REQUIRE_SYNTH", "1")
    monkeypatch.setenv("SLM_AGENT_FIRST_DATASET_DISCOVERY", "1")
    snapshot = runtime_config_snapshot("cold_start")

    assert snapshot["SLM_MAX_SEQ_LENGTH"] == "4096"
    assert snapshot["SLM_EVAL_MAX_NEW_TOKENS_NER"] == "512"
    assert snapshot["SLM_EVAL_MAX_NEW_TOKENS_APPS"] == "1024"
    assert snapshot["SLM_APPS_PROBLEM_TIMEOUT_S"] == "5"
    assert snapshot["SLM_STAGNATION_WINDOW"] == "41"
    assert snapshot["SLM_MAX_STALL_EVALS"] == "42"
    assert snapshot["SLM_REQUIRE_SYNTH"] == "1"
    assert snapshot["SLM_AGENT_FIRST_DATASET_DISCOVERY"] == "1"
    for key in (
        "CHEAP_MODE",
        "SYNTH_MODEL",
        "JUDGE_MODEL",
        "JUDGE_CONCURRENCY",
        "JUDGE_REQUEST_TIMEOUT_S",
        "LOCAL_DATASET_DIR",
    ):
        assert key in snapshot
    assert snapshot["LORA_SEARCH_SPACE_VERSION"] >= 1
    assert snapshot["LORA_MAX_EFFECTIVE_BATCH_SIZE"] == 64
    assert snapshot["SFT_LOSS_CONTRACT_VERSION"] >= 2
    assert snapshot["DATA_REBUILD_SCHEMA_VERSION"] == 1
    assert snapshot["DATA_REBUILD_MAX_PAID_ROUNDS_PER_RUN"] == 9
    assert snapshot["ACQUISITION_BUDGET_SCHEMA_VERSION"] == 1

    before = runtime_config_fingerprint("cold_start")
    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "8192")
    after = runtime_config_fingerprint("cold_start")
    assert before != after
