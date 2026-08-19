from pathlib import Path

import pytest

from agent.nodes import train as train_module
from config.android_pool import ANDROID_POOL
from training.lora_trainer import TrainingOutput


def _state(dataset: Path) -> dict:
    return {
        "selected_model": ANDROID_POOL[0],
        "current_dataset_path": str(dataset),
        "iteration": 0,
        "task": "clinc150",
        "dag": [],
        "last_intervention": "data_rebuild",
        "llm_iterate_decision": None,
        "dataset_version": 0,
    }


def _final_dir(artifacts: Path) -> Path:
    selector = ANDROID_POOL[0].selector.replace("/", "_").replace("@", "__")
    return artifacts / "training" / selector / "iter1-d0"


def test_train_directory_is_published_only_after_training_completes(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"x","label":"y"}\n', encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(train_module, "ARTIFACTS_DIR", str(artifacts))

    observed = {}

    def fake_train(*, output_dir, **_kwargs):
        observed["output_dir"] = output_dir
        observed["kwargs"] = _kwargs
        checkpoint = Path(output_dir) / "final_checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        return TrainingOutput(weights_ref=str(checkpoint), gguf_path=None)

    monkeypatch.setattr(train_module, "slm_train", fake_train)

    result = train_module.train_node(_state(dataset))

    assert ".partial-" in observed["output_dir"]
    final_dir = _final_dir(artifacts)
    assert final_dir.is_dir()
    assert (final_dir / "final_checkpoint" / ".slm_complete").is_file()
    assert (
        final_dir / "final_checkpoint" / ".slm_artifact_manifest.json"
    ).is_file()
    assert result["_pending_weights_refs"]
    weights_ref = next(iter(result["_pending_weights_refs"].values()))
    assert weights_ref == str(final_dir / "final_checkpoint")
    assert observed["kwargs"]["lora_rank"] == 16
    assert observed["kwargs"]["lora_alpha"] == 32
    assert observed["kwargs"]["lora_dropout"] == 0.0
    assert observed["kwargs"]["weight_decay"] == 0.01
    assert observed["kwargs"]["micro_batch_size"] == 8
    assert observed["kwargs"]["gradient_accumulation_steps"] == 1
    assert observed["kwargs"]["effective_batch_size"] == 8
    assert not list(final_dir.parent.glob("iter1-d0.partial-*"))


def test_failed_training_leaves_no_resumable_iteration_directory(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"x","label":"y"}\n', encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(train_module, "ARTIFACTS_DIR", str(artifacts))

    def fail_train(*, output_dir, **_kwargs):
        Path(output_dir).mkdir(parents=True)
        (Path(output_dir) / "partial.bin").write_bytes(b"incomplete")
        raise RuntimeError("simulated train crash")

    monkeypatch.setattr(train_module, "slm_train", fail_train)

    with pytest.raises(RuntimeError, match="train crash"):
        train_module.train_node(_state(dataset))

    final_dir = _final_dir(artifacts)
    assert not final_dir.exists()
    assert not list(final_dir.parent.glob("iter1-d0.partial-*"))


def test_retry_reuses_atomically_published_training_before_graph_commit(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"x","label":"y"}\n', encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(train_module, "ARTIFACTS_DIR", str(artifacts))
    calls = []

    def fake_train(*, output_dir, **_kwargs):
        calls.append(output_dir)
        checkpoint = Path(output_dir) / "final_checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        return TrainingOutput(str(checkpoint), None)

    monkeypatch.setattr(train_module, "slm_train", fake_train)

    first = train_module.train_node(_state(dataset))
    second = train_module.train_node(_state(dataset))

    assert len(calls) == 1
    assert first["_pending_weights_refs"] == second["_pending_weights_refs"]
