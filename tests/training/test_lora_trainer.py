# tests/training/test_lora_trainer.py
import json
import gc
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from training.lora_trainer import (
    TrainingConfig,
    TrainingOutput,
    _ensure_model_cached,
    _run_unsloth_training,
    is_multimodal_model,
    merge_for_quantization,
    text_tokenizer,
)


class _TokenizerDouble:
    """Stands in for a Qwen tokenizer, rendering real ChatML.

    It used to render a made-up `<user>..</user><assistant>` shape. That was fine while nothing
    compared the rendered text against the inference prompt, but `_assert_train_serve_prefix_alignment`
    (B290) does exactly that, and a Qwen tokenizer that emits non-ChatML is a configuration that cannot
    occur — the double was asserting against an impossible world. Emitting the hybrid-Qwen3 form (with
    the pre-filled empty think block, matching `_qwen_no_think_prompt`) keeps these tests exercising
    their real subjects instead of tripping the skew guard.
    """

    chat_template = "qwen-template"
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 3
    eos_token = "<eos>"

    def __init__(self):
        self.template_calls = []
        self.save_pretrained = MagicMock()

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.template_calls.append({
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        })
        prompt = (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        if add_generation_prompt:
            return prompt
        return f"{prompt}{messages[-1]['content']}<|im_end|>\n"

    def __call__(self, text, *, truncation, add_special_tokens):
        return {"input_ids": [ord(char) + 10 for char in text]}


def _write_valid_snapshot_tokenizer(snapshot):
    (snapshot / "tokenizer.model").write_bytes(b"sentencepiece-protobuf")


def test_training_output_is_namedtuple():
    out = TrainingOutput(weights_ref="/some/path", gguf_path=None)
    assert out.weights_ref == "/some/path"
    assert out.gguf_path is None

def test_training_output_fields():
    out = TrainingOutput(weights_ref="/a", gguf_path="/b")
    assert out[0] == "/a"
    assert out[1] == "/b"


def test_model_prefetch_uses_complete_local_snapshot_without_network(
    tmp_path,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    _write_valid_snapshot_tokenizer(snapshot)
    snapshot_download = MagicMock(return_value=str(snapshot))
    model_info = MagicMock(return_value=SimpleNamespace(sha="abc123"))
    hub = SimpleNamespace(
        model_info=model_info,
        snapshot_download=snapshot_download,
    )

    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        _ensure_model_cached("Qwen/Qwen3.5-4B", retries=1)

    model_info.assert_not_called()
    snapshot_download.assert_called_once()
    assert snapshot_download.call_args.kwargs["local_files_only"] is True


def test_model_prefetch_incomplete_local_snapshot_falls_through_online(
    tmp_path,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    _write_valid_snapshot_tokenizer(snapshot)

    def download(*args, **kwargs):
        if not kwargs.get("local_files_only", False):
            (snapshot / "model-00002-of-00002.safetensors").write_bytes(
                b"second"
            )
        return str(snapshot)

    snapshot_download = MagicMock(side_effect=download)
    model_info = MagicMock(return_value=SimpleNamespace(sha="abc123"))
    hub = SimpleNamespace(
        model_info=model_info,
        snapshot_download=snapshot_download,
    )

    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        _ensure_model_cached("Qwen/Qwen3.5-4B", retries=1)

    assert snapshot_download.call_count == 2
    assert snapshot_download.call_args_list[0].kwargs[
        "local_files_only"
    ] is True
    assert snapshot_download.call_args_list[1].kwargs["revision"] == "abc123"
    model_info.assert_called_once_with("Qwen/Qwen3.5-4B")


def test_model_prefetch_pins_revision_and_resumes_partial_shards(
    tmp_path,
    monkeypatch,
):
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    _write_valid_snapshot_tokenizer(snapshot)

    downloads = MagicMock()

    def resume_download(*args, **kwargs):
        if downloads.call_count == 2:
            (snapshot / "model-00002-of-00002.safetensors").write_bytes(
                b"second"
            )
        return str(snapshot)

    downloads.side_effect = resume_download
    hub = SimpleNamespace(
        model_info=MagicMock(return_value=SimpleNamespace(sha="abc123")),
        snapshot_download=downloads,
    )
    hub_utils = SimpleNamespace(
        GatedRepoError=type("GatedRepoError", (Exception,), {}),
        RepositoryNotFoundError=type(
            "RepositoryNotFoundError",
            (Exception,),
            {},
        ),
    )
    monkeypatch.setenv("SLM_HF_DOWNLOAD_BACKOFF_SECONDS", "0")

    with patch.dict(
        sys.modules,
        {
            "huggingface_hub": hub,
            "huggingface_hub.utils": hub_utils,
        },
    ):
        _ensure_model_cached("Qwen/Qwen3.5-4B", retries=2)

    assert hub.model_info.call_count == 1
    assert downloads.call_count == 2
    assert downloads.call_args_list[0].kwargs["local_files_only"] is True
    online_call = downloads.call_args_list[1]
    assert online_call.kwargs["revision"] == "abc123"
    assert online_call.kwargs["max_workers"] == 1
    assert online_call.kwargs["local_files_only"] is False


def test_model_prefetch_uses_configured_low_worker_count(
    tmp_path,
    monkeypatch,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    _write_valid_snapshot_tokenizer(snapshot)
    def download(*args, **kwargs):
        if kwargs.get("local_files_only", False):
            raise FileNotFoundError("not cached")
        return str(snapshot)

    snapshot_download = MagicMock(side_effect=download)
    hub = SimpleNamespace(
        model_info=MagicMock(return_value=SimpleNamespace(sha="abc123")),
        snapshot_download=snapshot_download,
    )
    hub_utils = SimpleNamespace(
        GatedRepoError=type("GatedRepoError", (Exception,), {}),
        RepositoryNotFoundError=type(
            "RepositoryNotFoundError",
            (Exception,),
            {},
        ),
    )
    monkeypatch.setenv("SLM_HF_DOWNLOAD_WORKERS", "2")

    with patch.dict(
        sys.modules,
        {
            "huggingface_hub": hub,
            "huggingface_hub.utils": hub_utils,
        },
    ):
        _ensure_model_cached("Qwen/Qwen3.5-4B", retries=1)

    assert snapshot_download.call_count == 2
    assert snapshot_download.call_args_list[0].kwargs[
        "local_files_only"
    ] is True
    assert snapshot_download.call_args_list[1].kwargs["max_workers"] == 2


def test_model_prefetch_raises_clear_error_after_incomplete_retries(
    tmp_path,
    monkeypatch,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"")
    hub = SimpleNamespace(
        model_info=MagicMock(return_value=SimpleNamespace(sha="abc123")),
        snapshot_download=MagicMock(return_value=str(snapshot)),
    )
    hub_utils = SimpleNamespace(
        GatedRepoError=type("GatedRepoError", (Exception,), {}),
        RepositoryNotFoundError=type(
            "RepositoryNotFoundError",
            (Exception,),
            {},
        ),
    )
    monkeypatch.setenv("SLM_HF_DOWNLOAD_BACKOFF_SECONDS", "0")

    with (
        patch.dict(
            sys.modules,
            {
                "huggingface_hub": hub,
                "huggingface_hub.utils": hub_utils,
            },
        ),
        pytest.raises(
            RuntimeError,
            match=(
                r"infrastructure.*Qwen/Qwen3\.5-4B.*"
                r"model-00002-of-00002\.safetensors"
            ),
        ),
    ):
        _ensure_model_cached("Qwen/Qwen3.5-4B", retries=2)

    assert hub.snapshot_download.call_count == 3
    assert hub.snapshot_download.call_args_list[0].kwargs[
        "local_files_only"
    ] is True
    for call in hub.snapshot_download.call_args_list[1:]:
        assert call.kwargs["revision"] == "abc123"
        assert call.kwargs["local_files_only"] is False


def test_model_prefetch_local_checkpoint_is_noop(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with patch.dict(sys.modules, {"huggingface_hub": None}):
        assert _ensure_model_cached(str(checkpoint)) is None


def test_training_max_sequence_length_honors_an_explicit_override(monkeypatch):
    """An explicit SLM_MAX_SEQ_LENGTH wins over the task's own ceiling on both sides.

    Training and inference must agree: a row that fits training but not eval would be scored on a
    truncated prompt. Both read `task_max_seq_length`, so the override cannot apply to one only.
    """
    from training import lora_trainer

    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "1024")

    assert lora_trainer._configured_max_seq_length("gsm8k") == 1024


def test_training_rejects_target_truncation_before_sft():
    from training import lora_trainer

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": list(range(len(text)))}

    with pytest.raises(
        ValueError,
        match=r"training row 0.*12 tokens.*context 10.*target-critical",
    ):
        lora_trainer._validate_training_sequence_lengths(
            [{"text": "prompttarget"}],
            Tokenizer(),
            10,
        )


def test_qwen35_pool_entry_is_multimodal_but_unknown_model_is_not():
    assert is_multimodal_model("Qwen/Qwen3.5-0.8B") is True
    assert is_multimodal_model("Qwen/Qwen3.5-unknown") is False


def test_text_tokenizer_unwraps_processor_text_tokenizer():
    class Qwen35Processor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    inner = MagicMock()
    inner.apply_chat_template = MagicMock()
    processor = Qwen35Processor(inner)

    assert text_tokenizer(processor) is inner


def test_qwen35_text_only_training_uses_vision_loader_and_freezes_vision(tmp_path):
    dataset_path = tmp_path / "tiny.jsonl"
    dataset_path.write_text(
        "\n".join(
            (
                json.dumps({"text": "happy", "label": "joy"}),
                json.dumps({"text": "sad", "label": "sadness"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    class Qwen35Processor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    tokenizer = _TokenizerDouble()
    processor = Qwen35Processor(tokenizer)
    model = MagicMock()
    language_loader = MagicMock()
    vision_loader = MagicMock()
    vision_loader.from_pretrained.return_value = (model, processor)
    vision_loader.get_peft_model.return_value = model
    trainer = MagicMock()

    unsloth_mock = MagicMock(
        FastLanguageModel=language_loader,
        FastVisionModel=vision_loader,
    )
    trl_mock = MagicMock()
    trl_mock.SFTConfig.side_effect = lambda **kwargs: kwargs
    trl_mock.SFTTrainer.return_value = trainer
    datasets_mock = MagicMock()
    datasets_mock.Dataset.from_list.side_effect = lambda rows: rows
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.cuda.is_bf16_supported.return_value = False

    config = TrainingConfig(
        base_model="Qwen/Qwen3.5-0.8B",
        nr_epochs=1,
        learning_rate=2e-4,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        weight_decay=0.1,
        micro_batch_size=1,
        gradient_accumulation_steps=8,
        task="clinc150",
    )
    output_dir = tmp_path / "output"
    with (
        patch("training.lora_trainer._ensure_model_cached"),
        patch("agent.logging_setup.quiet_ml_logging"),
        patch.dict(
            sys.modules,
            {
                "datasets": datasets_mock,
                "torch": torch_mock,
                "trl": trl_mock,
                "unsloth": unsloth_mock,
            },
        ),
    ):
        checkpoint = _run_unsloth_training(
            str(dataset_path),
            config,
            str(output_dir),
            task=config.task,
        )

    assert checkpoint == str(output_dir / "final_checkpoint")
    language_loader.from_pretrained.assert_not_called()
    # Derived, not hardcoded: the context ceiling is per task, and training must request
    # exactly what eval will use or a row could fit one side and be truncated on the other.
    from training.slm_helpers import task_max_seq_length

    vision_loader.from_pretrained.assert_called_once_with(
        model_name="Qwen/Qwen3.5-0.8B",
        max_seq_length=task_max_seq_length("clinc150"),
        load_in_4bit=True,
        trust_remote_code=True,
    )
    peft_kwargs = vision_loader.get_peft_model.call_args.kwargs
    assert peft_kwargs["finetune_vision_layers"] is False
    assert peft_kwargs["finetune_language_layers"] is True
    assert peft_kwargs["r"] == 8
    assert peft_kwargs["lora_alpha"] == 32
    assert peft_kwargs["lora_dropout"] == 0.05
    sft_kwargs = trl_mock.SFTConfig.call_args.kwargs
    assert sft_kwargs["per_device_train_batch_size"] == 1
    assert sft_kwargs["gradient_accumulation_steps"] == 8
    assert sft_kwargs["weight_decay"] == 0.1
    assert sft_kwargs["completion_only_loss"] is True
    assert trl_mock.SFTTrainer.call_args.kwargs["processing_class"] is tokenizer
    trainer_kwargs = trl_mock.SFTTrainer.call_args.kwargs
    collated = trainer_kwargs["data_collator"](
        trainer_kwargs["train_dataset"]
    )
    labels = collated["labels"][0].tolist()
    assert -100 in labels
    assert any(label != -100 for label in labels)
    model.save_pretrained.assert_called_once_with(checkpoint)
    tokenizer.save_pretrained.assert_called_once_with(checkpoint)


def test_text_training_wires_expanded_peft_and_sft_kwargs(tmp_path):
    dataset_path = tmp_path / "tiny.jsonl"
    dataset_path.write_text(
        json.dumps({"text": "happy", "label": "joy"}) + "\n",
        encoding="utf-8",
    )
    tokenizer = _TokenizerDouble()
    model = MagicMock()
    language_loader = MagicMock()
    language_loader.from_pretrained.return_value = (model, tokenizer)
    language_loader.get_peft_model.return_value = model
    trainer = MagicMock()
    trl_mock = MagicMock()
    trl_mock.SFTConfig.side_effect = lambda **kwargs: kwargs
    trl_mock.SFTTrainer.return_value = trainer
    datasets_mock = MagicMock()
    datasets_mock.Dataset.from_list.side_effect = lambda rows: rows
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.cuda.is_bf16_supported.return_value = False
    config = TrainingConfig(
        base_model="text/model",
        nr_epochs=6,
        learning_rate=5e-4,
        lora_rank=16,
        lora_alpha=64,
        lora_dropout=0.1,
        weight_decay=0.05,
        micro_batch_size=2,
        gradient_accumulation_steps=4,
        task="clinc150",
    )

    with (
        patch("training.lora_trainer._ensure_model_cached"),
        patch("training.lora_trainer.is_multimodal_model", return_value=False),
        patch("agent.logging_setup.quiet_ml_logging"),
        patch.dict(
            sys.modules,
            {
                "datasets": datasets_mock,
                "torch": torch_mock,
                "trl": trl_mock,
                "unsloth": MagicMock(FastLanguageModel=language_loader),
            },
        ),
    ):
        _run_unsloth_training(
            str(dataset_path),
            config,
            str(tmp_path / "output"),
            task=config.task,
        )

    peft_kwargs = language_loader.get_peft_model.call_args.kwargs
    assert peft_kwargs["r"] == 16
    assert peft_kwargs["lora_alpha"] == 64
    assert peft_kwargs["lora_dropout"] == 0.1
    sft_kwargs = trl_mock.SFTConfig.call_args.kwargs
    assert sft_kwargs["num_train_epochs"] == 6
    assert sft_kwargs["learning_rate"] == 5e-4
    assert sft_kwargs["per_device_train_batch_size"] == 2
    assert sft_kwargs["gradient_accumulation_steps"] == 4
    assert sft_kwargs["weight_decay"] == 0.05
    assert sft_kwargs["completion_only_loss"] is True
    trainer_kwargs = trl_mock.SFTTrainer.call_args.kwargs
    collated = trainer_kwargs["data_collator"](
        trainer_kwargs["train_dataset"]
    )
    labels = collated["labels"][0].tolist()
    assert -100 in labels
    assert any(label != -100 for label in labels)


def test_early_stop_failure_reloads_fresh_model_and_full_training_rows(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "rows.jsonl"
    rows = [
        {"text": f"example {index}", "label": "yes" if index % 2 else "no"}
        for index in range(20)
    ]
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLM_EARLY_STOPPING", "1")
    monkeypatch.setenv("SLM_MIN_FOR_VAL", "2")

    tokenizer_one = _TokenizerDouble()
    tokenizer_two = _TokenizerDouble()
    raw_model_one = MagicMock(name="raw_model_one")
    raw_model_two = MagicMock(name="raw_model_two")
    peft_model_one = MagicMock(name="peft_model_one")
    peft_model_two = MagicMock(name="peft_model_two")
    events = []
    cleanup_probes = []
    transient_ref = lambda: None

    class TracebackTensor:
        pass

    def fail_with_traceback_tensor():
        nonlocal transient_ref
        traceback_tensor = TracebackTensor()
        transient_ref = weakref.ref(traceback_tensor)
        raise RuntimeError("checkpoint path failed")

    load_results = iter([
        (raw_model_one, tokenizer_one),
        (raw_model_two, tokenizer_two),
    ])

    def load_model(**_kwargs):
        result = next(load_results)
        if result[0] is raw_model_two:
            events.append("reload")
            assert events[:2] == ["gc", "empty_cache"]
            assert sys.exc_info()[0] is None
            assert transient_ref() is None
        return result

    loader = MagicMock()
    loader.from_pretrained.side_effect = load_model
    loader.get_peft_model.side_effect = [
        peft_model_one,
        peft_model_two,
    ]
    first_trainer = MagicMock(name="first_trainer")
    first_trainer.train.side_effect = fail_with_traceback_tensor
    second_trainer = MagicMock(name="second_trainer")
    trl_mock = MagicMock()
    trl_mock.SFTConfig.side_effect = lambda **kwargs: kwargs
    trl_mock.SFTTrainer.side_effect = [first_trainer, second_trainer]
    datasets_mock = MagicMock()
    datasets_mock.Dataset.from_list.side_effect = lambda values: list(values)
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = True
    torch_mock.cuda.is_bf16_supported.return_value = False
    torch_mock.cuda.empty_cache.side_effect = lambda: events.append(
        "empty_cache"
    )
    transformers_mock = MagicMock()
    config = TrainingConfig(
        base_model="text/model",
        nr_epochs=4,
        learning_rate=2e-4,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        weight_decay=0.01,
        micro_batch_size=2,
        gradient_accumulation_steps=4,
        task="clinc150",
    )
    output_dir = tmp_path / "output"

    original_collect = gc.collect

    def collect_and_probe():
        cleanup_probes.append((
            sys.exc_info()[0],
            transient_ref() is None,
        ))
        events.append("gc")
        return original_collect()

    with (
        patch("training.lora_trainer._ensure_model_cached"),
        patch("training.lora_trainer.is_multimodal_model", return_value=False),
        patch("agent.logging_setup.quiet_ml_logging"),
        patch("gc.collect", side_effect=collect_and_probe),
        patch.dict(
            sys.modules,
            {
                "datasets": datasets_mock,
                "torch": torch_mock,
                "transformers": transformers_mock,
                "trl": trl_mock,
                "unsloth": MagicMock(FastLanguageModel=loader),
            },
        ),
    ):
        checkpoint = _run_unsloth_training(
            str(dataset_path),
            config,
            str(output_dir),
            task=config.task,
        )

    assert checkpoint == str(output_dir / "final_checkpoint")
    assert loader.from_pretrained.call_count == 2
    assert loader.get_peft_model.call_count == 2
    first_kwargs, second_kwargs = [
        call.kwargs for call in trl_mock.SFTTrainer.call_args_list
    ]
    assert first_kwargs["model"] is peft_model_one
    assert second_kwargs["model"] is peft_model_two
    assert len(first_kwargs["train_dataset"]) == 12
    assert len(first_kwargs["eval_dataset"]) == 8
    assert len(second_kwargs["train_dataset"]) == 20
    assert "eval_dataset" not in second_kwargs
    first_args, second_args = [
        call.kwargs for call in trl_mock.SFTConfig.call_args_list
    ]
    for key in (
        "num_train_epochs",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "weight_decay",
        "completion_only_loss",
    ):
        assert second_args[key] == first_args[key]
    peft_model_one.save_pretrained.assert_not_called()
    peft_model_two.save_pretrained.assert_called_once_with(checkpoint)
    tokenizer_two.save_pretrained.assert_called_once_with(checkpoint)
    assert events[:3] == ["gc", "empty_cache", "reload"]
    assert cleanup_probes[0] == (None, True)


def _record_merge_write(mock_model):
    """Make a mocked save_pretrained_merged actually emit a weight file.

    merge_for_quantization now rejects an empty merge directory outright (B219), so a mock that
    writes nothing would look like the unsloth_zoo silent no-op it is designed to catch.
    """
    mock_model.save_pretrained_merged.side_effect = (
        lambda dest, _tokenizer, **_kwargs: (
            Path(dest).mkdir(parents=True, exist_ok=True),
            (Path(dest) / "model.safetensors").write_bytes(b"merged"),
        )
    )
    return mock_model


def test_merge_for_quantization_loads_qwen35_base_locally_then_adapter(
    tmp_path,
):
    language_loader = MagicMock()
    vision_loader = MagicMock()
    adapter_model = _record_merge_write(MagicMock())
    tokenizer = MagicMock()
    unsloth_mock = MagicMock(
        FastLanguageModel=language_loader,
        FastVisionModel=vision_loader,
    )
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-2B"}),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"lora")
    (checkpoint / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE"}}),
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshots" / "15852e"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights-with-mtp")
    (snapshot / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE"}}),
        encoding="utf-8",
    )
    snapshot_download = MagicMock(return_value=str(snapshot))
    hub_mock = SimpleNamespace(snapshot_download=snapshot_download)
    output_dir = tmp_path / "quant"
    staged_adapter = None

    def load_local_adapter(**kwargs):
        nonlocal staged_adapter
        staged_adapter = kwargs["model_name"]
        staged_config = json.loads(
            (tmp_path / staged_adapter / "adapter_config.json").read_text(
                encoding="utf-8",
            )
        )
        assert staged_config["base_model_name_or_path"] == str(
            snapshot.resolve()
        )
        assert (
            tmp_path / staged_adapter / "adapter_model.safetensors"
        ).read_bytes() == b"lora"
        return adapter_model, tokenizer

    vision_loader.from_pretrained.side_effect = load_local_adapter

    with (
        patch.dict(
            sys.modules,
            {
                "huggingface_hub": hub_mock,
                "unsloth": unsloth_mock,
            },
        ),
        patch(
            "training.lora_trainer.verify_hf_model_snapshot",
        ) as verify_snapshot,
    ):
        result = merge_for_quantization(str(checkpoint), str(output_dir))

    assert result == str(output_dir / "merged")
    snapshot_download.assert_called_once_with(
        "Qwen/Qwen3.5-2B",
        local_files_only=True,
        ignore_patterns=[
            "*.gguf",
            "original/*",
            "*.pth",
            "consolidated*",
        ],
    )
    vision_loader.from_pretrained.assert_called_once_with(
        model_name=staged_adapter,
        max_seq_length=4096,
        load_in_4bit=False,
        local_files_only=True,
        trust_remote_code=True,
    )
    adapter_model.save_pretrained_merged.assert_called_once_with(
        str(output_dir / "merged"),
        tokenizer,
        save_method="merged_16bit",
    )
    verify_snapshot.assert_called_once_with(str(output_dir / "merged"))
    assert not (tmp_path / staged_adapter).exists()
    language_loader.from_pretrained.assert_not_called()


def test_merge_for_quantization_keeps_full_local_checkpoint_path(tmp_path):
    language_loader = MagicMock()
    model = _record_merge_write(MagicMock())
    tokenizer = MagicMock()
    language_loader.from_pretrained.return_value = (model, tokenizer)
    unsloth_mock = MagicMock(
        FastLanguageModel=language_loader,
        FastVisionModel=MagicMock(),
    )
    checkpoint = tmp_path / "full-checkpoint"
    checkpoint.mkdir()
    output_dir = tmp_path / "quant"

    with (
        patch.dict(sys.modules, {"unsloth": unsloth_mock}),
        patch(
            "training.lora_trainer.resolve_cached_hf_snapshot",
        ) as resolve_snapshot,
        patch(
            "training.lora_trainer.verify_hf_model_snapshot",
        ),
    ):
        result = merge_for_quantization(str(checkpoint), str(output_dir))

    assert result == str(output_dir / "merged")
    resolve_snapshot.assert_not_called()
    language_loader.from_pretrained.assert_called_once_with(
        model_name=str(checkpoint),
        max_seq_length=4096,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    model.save_pretrained_merged.assert_called_once()


# QLoRA trains on a 4-bit base, and Unsloth silently rewrites the adapter's
# base_model_name_or_path to its own pre-quantized mirror (e.g. Qwen/Qwen3-0.6B becomes
# unsloth/qwen3-0.6b-unsloth-bnb-4bit). Merging *that* with save_method="merged_16bit" makes
# unsloth_zoo emit only a UserWarning and write nothing at all, so the caller must be able to
# pin the merge to the canonical 16-bit base it actually selected.

def _merge_fixture(tmp_path, recorded_base):
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": recorded_base}), encoding="utf-8"
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"lora")
    (checkpoint / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE"}}), encoding="utf-8"
    )
    snapshot = tmp_path / "snapshots" / "sixteenbit"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE"}}), encoding="utf-8"
    )
    return checkpoint, snapshot


def test_merge_for_quantization_pins_merge_to_explicit_16bit_base(tmp_path):
    checkpoint, snapshot = _merge_fixture(
        tmp_path, "unsloth/qwen3-0.6b-unsloth-bnb-4bit"
    )
    model, tokenizer = _record_merge_write(MagicMock()), MagicMock()
    language_loader = MagicMock()
    staged_base = None

    def load_local_adapter(**kwargs):
        nonlocal staged_base
        staged_base = json.loads(
            (Path(kwargs["model_name"]) / "adapter_config.json").read_text(
                encoding="utf-8"
            )
        )["base_model_name_or_path"]
        return model, tokenizer

    language_loader.from_pretrained.side_effect = load_local_adapter
    unsloth_mock = MagicMock(
        FastLanguageModel=language_loader, FastVisionModel=MagicMock()
    )
    output_dir = tmp_path / "quant"

    with (
        patch.dict(sys.modules, {"unsloth": unsloth_mock}),
        patch(
            "training.lora_trainer.resolve_cached_hf_snapshot",
            return_value=str(snapshot),
        ) as resolve_snapshot,
        patch("training.lora_trainer.verify_hf_model_snapshot"),
    ):
        merge_for_quantization(
            str(checkpoint), str(output_dir), base_model_id="Qwen/Qwen3-0.6B"
        )

    # The canonical 16-bit id is resolved, NOT the 4-bit mirror the adapter recorded.
    resolve_snapshot.assert_called_once_with("Qwen/Qwen3-0.6B")
    assert staged_base == str(snapshot)
    model.save_pretrained_merged.assert_called_once_with(
        str(output_dir / "merged"), tokenizer, save_method="merged_16bit"
    )


def test_merge_for_quantization_without_explicit_base_uses_adapter_record(tmp_path):
    checkpoint, snapshot = _merge_fixture(tmp_path, "Qwen/Qwen3-0.6B")
    model, tokenizer = _record_merge_write(MagicMock()), MagicMock()
    language_loader = MagicMock(
        from_pretrained=MagicMock(return_value=(model, tokenizer))
    )
    unsloth_mock = MagicMock(
        FastLanguageModel=language_loader, FastVisionModel=MagicMock()
    )

    with (
        patch.dict(sys.modules, {"unsloth": unsloth_mock}),
        patch(
            "training.lora_trainer.resolve_cached_hf_snapshot",
            return_value=str(snapshot),
        ) as resolve_snapshot,
        patch("training.lora_trainer.verify_hf_model_snapshot"),
    ):
        merge_for_quantization(str(checkpoint), str(tmp_path / "quant"))

    resolve_snapshot.assert_called_once_with("Qwen/Qwen3-0.6B")


def test_merge_for_quantization_reports_silent_empty_merge_actionably(tmp_path):
    """An empty merge dir must name the 4-bit-base cause, not just 'snapshot is incomplete'."""
    checkpoint, snapshot = _merge_fixture(
        tmp_path, "unsloth/qwen3-0.6b-unsloth-bnb-4bit"
    )
    model, tokenizer = MagicMock(), MagicMock()
    # save_pretrained_merged is a no-op, exactly as unsloth_zoo behaves on a 4-bit base.
    language_loader = MagicMock(
        from_pretrained=MagicMock(return_value=(model, tokenizer))
    )
    unsloth_mock = MagicMock(
        FastLanguageModel=language_loader, FastVisionModel=MagicMock()
    )

    with (
        patch.dict(sys.modules, {"unsloth": unsloth_mock}),
        patch(
            "training.lora_trainer.resolve_cached_hf_snapshot",
            return_value=str(snapshot),
        ),
        pytest.raises(RuntimeError, match="wrote no files"),
    ):
        merge_for_quantization(str(checkpoint), str(tmp_path / "quant"))
