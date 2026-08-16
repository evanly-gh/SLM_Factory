# tests/training/test_slm_helpers.py
import os
import sys
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from unittest.mock import patch, MagicMock
from training.slm_helpers import infer_batch_gguf
from training.lora_trainer import TrainingOutput


class _FakeCudaOOM(RuntimeError):
    pass


class _FakeCuda:
    OutOfMemoryError = _FakeCudaOOM

    def __init__(self, *, cleanup_events=None, cleanup_probe=None):
        self.empty_cache_calls = 0
        self.cleanup_events = cleanup_events
        self.cleanup_probe = cleanup_probe
        self.exception_types_at_cleanup = []
        self.cleanup_probe_results = []

    def empty_cache(self):
        self.empty_cache_calls += 1
        self.exception_types_at_cleanup.append(sys.exc_info()[0])
        if self.cleanup_events is not None:
            self.cleanup_events.append("empty_cache")
        if self.cleanup_probe is not None:
            self.cleanup_probe_results.append(self.cleanup_probe())


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class _FakeTorch:
    OutOfMemoryError = _FakeCudaOOM

    def __init__(self, *, cleanup_events=None, cleanup_probe=None):
        self.cuda = _FakeCuda(
            cleanup_events=cleanup_events,
            cleanup_probe=cleanup_probe,
        )

    @staticmethod
    def no_grad():
        return _FakeNoGrad()


class _FakeIds(list):
    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)


class _FakeEncoding(dict):
    def to(self, device):
        self.device = device
        return self


class _FakeTokenizer:
    chat_template = "explicit-chat-template"
    eos_token = "<eos>"
    eos_token_id = 99

    def __init__(self, token_sequences):
        self.token_sequences = token_sequences
        self.padding_side = "right"
        self.pad_token = None
        self.pad_token_id = None
        self.template_calls = []
        self.tokenize_calls = []
        self.decode_calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return messages[0]["content"]

    def __call__(
        self,
        texts,
        *,
        return_tensors,
        padding=False,
        truncation=False,
    ):
        is_batch = isinstance(texts, list)
        values = texts if is_batch else [texts]
        rows = [list(self.token_sequences[value]) for value in values]
        width = max(len(row) for row in rows)
        if padding:
            rows = [
                [self.pad_token_id] * (width - len(row)) + row
                for row in rows
            ]
        self.tokenize_calls.append(
            {
                "texts": texts,
                "return_tensors": return_tensors,
                "padding": padding,
                "truncation": truncation,
            }
        )
        return _FakeEncoding(
            input_ids=_FakeIds(rows),
            attention_mask=_FakeIds(
                [[int(token != self.pad_token_id) for token in row] for row in rows]
            ),
        )

    def decode(self, tokens, *, skip_special_tokens):
        values = list(tokens)
        self.decode_calls.append(
            {"tokens": values, "skip_special_tokens": skip_special_tokens}
        )
        return ",".join(str(token) for token in values)


class _FakeModel:
    def __init__(self, *, oom_above=None):
        self.device = "cpu"
        self.generation_config = SimpleNamespace(max_length=40960)
        self.oom_above = oom_above
        self.generate_calls = []

    def generate(self, *, input_ids, attention_mask, **kwargs):
        rows = [list(row) for row in input_ids]
        self.generate_calls.append(
            {
                "input_ids": rows,
                "attention_mask": [list(row) for row in attention_mask],
                **kwargs,
            }
        )
        if self.oom_above is not None and len(rows) > self.oom_above:
            raise _FakeCudaOOM("CUDA out of memory in fake model")
        return [row + [1000 + row[-1]] for row in rows]


@contextmanager
def _local_cached_inference(model, tokenizer, *, fake_torch=None):
    fake_torch = fake_torch or _FakeTorch()
    # The context ceiling is per task type, and the cache is keyed on it, so a fixture that
    # assumed one global value silently missed the cache and tried a real model load. Seed every
    # ceiling the production table can produce so the fixture stays task-agnostic.
    from training.slm_helpers import _DEFAULT_MAX_SEQ_LENGTH, _TASK_MAX_SEQ_LENGTH

    override = os.environ.get("SLM_MAX_SEQ_LENGTH")
    lengths = (
        {int(override)}
        if override is not None
        else set(_TASK_MAX_SEQ_LENGTH.values()) | {_DEFAULT_MAX_SEQ_LENGTH}
    )
    cache = {
        ("/weights", "model-id", length): (model, tokenizer) for length in sorted(lengths)
    }
    cache_key = next(iter(cache))
    with (
        patch("training.cuda_isolation.isolation_enabled", return_value=False),
        patch("training.slm_helpers._inference_cache", cache),
        patch("training.slm_helpers._cache_order", list(cache)),
        patch.dict(sys.modules, {"torch": fake_torch}),
        patch("agent.timing.record_timing_event") as timing,
    ):
        yield fake_torch, timing


def test_infer_batch_gguf_raises_import_error_when_llama_cpp_missing():
    with patch.dict("sys.modules", {"llama_cpp": None}):
        with pytest.raises(ImportError, match="llama-cpp-python"):
            infer_batch_gguf(["hello"], "/fake/model.gguf")


def test_infer_batch_gguf_returns_list_of_strings():
    class NonThinkingChatLlama:
        def __init__(self):
            self.calls = []

        def create_chat_completion(
            self,
            messages,
            *,
            max_tokens,
            temperature,
            chat_template_kwargs=None,
        ):
            self.calls.append(chat_template_kwargs)
            return {"choices": [{"message": {"content": "spam"}}]}

    mock_llama_instance = NonThinkingChatLlama()
    mock_llama_cls = MagicMock(return_value=mock_llama_instance)

    with patch("training.slm_helpers._gguf_cache", {}), \
         patch("training.slm_helpers._gguf_cache_order", []), \
         patch.dict(os.environ, {"SLM_MAX_SEQ_LENGTH": "4096"}):
        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_cls)}):
            # Clear module-level cache to force load
            import training.slm_helpers as sh
            sh._gguf_cache.clear()
            sh._gguf_cache_order.clear()
            result = infer_batch_gguf(
                ["hello", "world"],
                "/fake/model.gguf",
                base_model="Qwen/Qwen3-1.7B",
            )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(s, str) for s in result)
    assert mock_llama_instance.calls == [
        {"enable_thinking": False},
        {"enable_thinking": False},
    ]
    assert mock_llama_cls.call_args.kwargs["n_ctx"] == 4096


def test_infer_batch_gguf_uses_documented_qwen_no_think_prompt_when_kwargs_unsupported():
    class LegacyLlama:
        def __init__(self):
            self.prompts = []

        def create_chat_completion(self, messages, *, max_tokens, temperature):
            raise AssertionError("legacy chat path must not be used without mode kwargs")

        def __call__(self, prompt, *, max_tokens, temperature, echo, stop):
            self.prompts.append(prompt)
            return {"choices": [{"text": "spam"}]}

    instance = LegacyLlama()
    with (
        patch("training.slm_helpers._gguf_cache", {}),
        patch("training.slm_helpers._gguf_cache_order", []),
        patch.dict(
            "sys.modules",
            {"llama_cpp": MagicMock(Llama=MagicMock(return_value=instance))},
        ),
    ):
        result = infer_batch_gguf(
            ["hello"],
            "/fake/legacy.gguf",
            base_model="Qwen/Qwen3.5-4B",
        )

    assert result == ["spam"]
    assert instance.prompts == [
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ]


def test_non_thinking_only_qwen_gguf_matches_plain_hf_generation_prefix():
    class LegacyLlama:
        def __init__(self):
            self.prompts = []

        def create_chat_completion(self, messages, *, max_tokens, temperature):
            raise AssertionError("installed legacy signature must use explicit framing")

        def __call__(self, prompt, *, max_tokens, temperature, echo, stop):
            self.prompts.append(prompt)
            return {"choices": [{"text": "answer"}]}

    instance = LegacyLlama()
    with (
        patch("training.slm_helpers._gguf_cache", {}),
        patch("training.slm_helpers._gguf_cache_order", []),
        patch.dict(
            "sys.modules",
            {"llama_cpp": MagicMock(Llama=MagicMock(return_value=instance))},
        ),
    ):
        result = infer_batch_gguf(
            ["hello"],
            "/fake/instruct.gguf",
            base_model="Qwen/Qwen3-4B-Instruct-2507",
        )

    assert result == ["answer"]
    assert instance.prompts == [
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n"
    ]
    assert "<think>" not in instance.prompts[0]


def test_infer_batch_gguf_fails_clearly_when_mode_cannot_be_controlled():
    class LegacyNonQwenLlama:
        def create_chat_completion(self, messages, *, max_tokens, temperature):
            return {"choices": [{"message": {"content": "unsafe"}}]}

    with (
        patch("training.slm_helpers._gguf_cache", {}),
        patch("training.slm_helpers._gguf_cache_order", []),
        patch.dict(
            "sys.modules",
            {
                "llama_cpp": MagicMock(
                    Llama=MagicMock(return_value=LegacyNonQwenLlama())
                )
            },
        ),
    ):
        with pytest.raises(RuntimeError, match="cannot enforce non-thinking"):
            infer_batch_gguf(
                ["hello"],
                "/fake/non-qwen.gguf",
                base_model="other/model",
            )


def test_hf_inference_chat_template_explicitly_disables_thinking():
    from training.slm_helpers import infer

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = [[1, 2, 3]]
    tokenizer = MagicMock()
    tokenizer.chat_template = "qwen-template"
    tokenizer.eos_token = "<eos>"
    tokenizer.eos_token_id = 99
    tokenizer.pad_token = None
    tokenizer.pad_token_id = None
    tokenizer.padding_side = "right"
    tokenizer.apply_chat_template.return_value = "rendered"
    encoded = MagicMock()
    encoded.to.return_value = encoded
    encoded.__getitem__.return_value.shape = (1, 2)
    tokenizer.return_value = encoded
    tokenizer.decode.return_value = "answer"
    loader = MagicMock()
    loader.from_pretrained.return_value = (model, tokenizer)
    torch_mock = MagicMock()
    torch_mock.no_grad.return_value.__enter__.return_value = None
    unsloth_mock = MagicMock(FastLanguageModel=loader)

    with (
        patch("training.cuda_isolation.isolation_enabled", return_value=False),
        patch("training.slm_helpers._inference_cache", {}),
        patch("training.slm_helpers._cache_order", []),
        patch("training.slm_helpers._is_adapter_only_checkpoint", return_value=False),
        patch("training.lora_trainer._ensure_model_cached"),
        patch.dict("sys.modules", {"torch": torch_mock, "unsloth": unsloth_mock}),
    ):
        assert infer("hello", "/weights", "Qwen/Qwen3-1.7B") == "answer"

    kwargs = tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs["enable_thinking"] is False
    assert kwargs["add_generation_prompt"] is True


def test_qwen35_hf_inference_uses_vision_loader_and_inner_text_tokenizer(
    monkeypatch,
):
    from training.slm_helpers import infer

    class Qwen35Processor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = [[1, 2, 3]]
    tokenizer = MagicMock()
    tokenizer.chat_template = "qwen-template"
    tokenizer.eos_token = "<eos>"
    tokenizer.eos_token_id = 99
    tokenizer.pad_token = None
    tokenizer.pad_token_id = None
    tokenizer.padding_side = "right"
    tokenizer.apply_chat_template.return_value = "rendered"
    encoded = MagicMock()
    encoded.to.return_value = encoded
    encoded.__getitem__.return_value.shape = (1, 2)
    tokenizer.return_value = encoded
    tokenizer.decode.return_value = "answer"
    processor = Qwen35Processor(tokenizer)
    language_loader = MagicMock()
    vision_loader = MagicMock()
    vision_loader.from_pretrained.return_value = (model, processor)
    torch_mock = MagicMock()
    torch_mock.no_grad.return_value.__enter__.return_value = None
    unsloth_mock = MagicMock(
        FastLanguageModel=language_loader,
        FastVisionModel=vision_loader,
    )
    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "1024")

    with (
        patch("training.cuda_isolation.isolation_enabled", return_value=False),
        patch("training.slm_helpers._inference_cache", {}),
        patch("training.slm_helpers._cache_order", []),
        patch(
            "training.slm_helpers._is_adapter_only_checkpoint",
            return_value=False,
        ),
        patch("training.lora_trainer._ensure_model_cached") as prefetch,
        patch.dict(sys.modules, {"torch": torch_mock, "unsloth": unsloth_mock}),
    ):
        result = infer(
            "hello",
            "/weights",
            "Qwen/Qwen3.5-0.8B",
        )

    assert result == "answer"
    assert [call.args[0] for call in prefetch.call_args_list] == [
        "Qwen/Qwen3.5-0.8B",
        "/weights",
    ]
    language_loader.from_pretrained.assert_not_called()
    vision_loader.from_pretrained.assert_called_once_with(
        model_name="/weights",
        max_seq_length=1024,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    vision_loader.for_inference.assert_called_once_with(model)
    tokenizer.assert_called_once_with(
        "rendered",
        return_tensors="pt",
        truncation=False,
    )
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.pad_token_id == 99


@pytest.mark.parametrize(
    "adapter_weight_name",
    ["adapter_model.safetensors", "adapter_model.bin"],
)
def test_adapter_weight_files_are_still_adapter_only(
    tmp_path,
    adapter_weight_name,
):
    from training.slm_helpers import _is_adapter_only_checkpoint

    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / adapter_weight_name).write_bytes(b"adapter")

    assert _is_adapter_only_checkpoint(str(checkpoint)) is True


@pytest.mark.parametrize(
    "full_weight_name",
    ["model.safetensors", "model-00001-of-00002.safetensors", "pytorch_model.bin"],
)
def test_full_model_weight_files_are_not_adapter_only(tmp_path, full_weight_name):
    from training.slm_helpers import _is_adapter_only_checkpoint

    checkpoint = tmp_path / "full"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / full_weight_name).write_bytes(b"full")

    assert _is_adapter_only_checkpoint(str(checkpoint)) is False


def test_inference_prefetches_base_and_weights_before_language_loader():
    from training.slm_helpers import _load_inference_model

    events = []
    model = MagicMock()
    model.generation_config = SimpleNamespace(max_length=40960)
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    loader = MagicMock()

    def load_model(**kwargs):
        events.append(("load", kwargs["model_name"]))
        return model, tokenizer

    loader.from_pretrained.side_effect = load_model
    unsloth_mock = MagicMock(FastLanguageModel=loader)

    def prefetch(model_id):
        events.append(("prefetch", model_id))

    with (
        patch("training.slm_helpers._inference_cache", {}),
        patch("training.slm_helpers._cache_order", []),
        patch(
            "training.lora_trainer._ensure_model_cached",
            side_effect=prefetch,
        ),
        patch(
            "training.lora_trainer.is_multimodal_model",
            return_value=False,
        ),
        patch("agent.logging_setup.quiet_ml_logging"),
        patch.dict(sys.modules, {"unsloth": unsloth_mock}),
    ):
        _load_inference_model(
            "Org/model-weights",
            "Org/base-model",
            512,
        )

    assert events == [
        ("prefetch", "Org/base-model"),
        ("prefetch", "Org/model-weights"),
        ("load", "Org/model-weights"),
    ]


def test_adapter_only_loader_loads_base_then_applies_adapter(tmp_path):
    """The adapter must be attached with PeftModel, not model.load_adapter (B254).

    `load_adapter` builds the LoRA modules on an Unsloth-patched model but leaves every
    lora_B tensor at its zero init, which makes the adapter an identity and silently scores
    the base model.
    """
    from training.slm_helpers import _load_inference_model

    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")

    base = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    loader = MagicMock()
    loader.from_pretrained.return_value = (base, tokenizer)
    unsloth_mock = MagicMock(FastLanguageModel=loader)

    # A PEFT-wrapped model whose adapter weights actually loaded.
    peft_model = MagicMock()
    peft_model.generation_config = SimpleNamespace(max_length=40960)
    peft_model.named_parameters.return_value = iter(
        [("base.layers.0.q_proj.lora_B.weight", torch.tensor([[0.25]]))]
    )
    peft_mock = MagicMock()
    peft_mock.PeftModel.from_pretrained.return_value = peft_model

    with (
        patch("training.slm_helpers._inference_cache", {}),
        patch("training.slm_helpers._cache_order", []),
        patch("training.lora_trainer._ensure_model_cached"),
        patch("training.lora_trainer.is_multimodal_model", return_value=False),
        patch("agent.logging_setup.quiet_ml_logging"),
        patch.dict(sys.modules, {"unsloth": unsloth_mock, "peft": peft_mock}),
    ):
        loaded_model, loaded_tokenizer = _load_inference_model(
            str(checkpoint),
            "base-model",
            512,
        )

    assert loaded_model is peft_model
    assert loaded_tokenizer is tokenizer
    loader.from_pretrained.assert_called_once_with(
        model_name="base-model",
        max_seq_length=512,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    peft_mock.PeftModel.from_pretrained.assert_called_once_with(base, str(checkpoint))
    base.load_adapter.assert_not_called()
    loader.for_inference.assert_called_once_with(peft_model)


def test_adapter_only_loader_rejects_an_inert_adapter(tmp_path):
    """An all-zero lora_B must abort the eval instead of scoring the base model."""
    from training.slm_helpers import _load_inference_model

    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    loader = MagicMock()
    loader.from_pretrained.return_value = (MagicMock(), tokenizer)
    unsloth_mock = MagicMock(FastLanguageModel=loader)

    peft_model = MagicMock()
    peft_model.named_parameters.return_value = iter(
        [("base.layers.0.q_proj.lora_B.weight", torch.zeros(2, 2))]
    )
    peft_mock = MagicMock()
    peft_mock.PeftModel.from_pretrained.return_value = peft_model

    with (
        patch("training.slm_helpers._inference_cache", {}),
        patch("training.slm_helpers._cache_order", []),
        patch("training.lora_trainer._ensure_model_cached"),
        patch("training.lora_trainer.is_multimodal_model", return_value=False),
        patch("agent.logging_setup.quiet_ml_logging"),
        patch.dict(sys.modules, {"unsloth": unsloth_mock, "peft": peft_mock}),
        pytest.raises(RuntimeError, match="ZERO"),
    ):
        _load_inference_model(str(checkpoint), "base-model", 512)


def test_train_returns_training_output():
    from training.slm_helpers import train
    with patch("training.slm_helpers.run_lora_training") as mock_train:
        mock_train.return_value = TrainingOutput(weights_ref="/ckpt", gguf_path=None)
        result = train("/data.jsonl", "model-id", 1, 2e-4, 8, 8)
    assert isinstance(result, TrainingOutput)
    assert result.weights_ref == "/ckpt"
    assert result.gguf_path is None
    config = mock_train.call_args.args[1]
    assert config.micro_batch_size == 8
    assert config.batch_size == 8
    assert config.gradient_accumulation_steps == 1
    assert config.effective_batch_size == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.0
    assert config.weight_decay == 0.01


def test_train_delegates_to_disposable_worker_when_enabled(monkeypatch):
    from training.slm_helpers import train

    expected = TrainingOutput(weights_ref="/isolated", gguf_path=None)
    monkeypatch.setenv("SLM_CUDA_ISOLATION", "1")
    monkeypatch.delenv("SLM_CUDA_WORKER", raising=False)
    with patch("training.cuda_isolation.run_isolated", return_value=expected) as worker, \
         patch("training.slm_helpers.clear_inference_cache") as clear:
        result = train(
            "/data.jsonl",
            "model-id",
            3,
            2e-4,
            lora_rank=16,
            output_dir="/out",
            task_type="classification",
            lora_alpha=64,
            lora_dropout=0.1,
            weight_decay=0.05,
            micro_batch_size=2,
            gradient_accumulation_steps=4,
        )

    assert result == expected
    clear.assert_called()
    worker.assert_called_once_with(
        "train",
        {
            "dataset_path": "/data.jsonl",
            "base_model": "model-id",
            "nr_epochs": 3,
            "learning_rate": 2e-4,
            "lora_rank": 16,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "weight_decay": 0.05,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "output_dir": "/out",
            "task_type": "classification",
        },
    )


def test_infer_delegates_to_disposable_worker_when_enabled(monkeypatch):
    from training.slm_helpers import infer

    monkeypatch.setenv("SLM_CUDA_ISOLATION", "1")
    monkeypatch.delenv("SLM_CUDA_WORKER", raising=False)
    with patch("training.cuda_isolation.run_isolated", return_value="prediction") as worker:
        result = infer("prompt", "/weights", "model-id", 77)

    assert result == "prediction"
    worker.assert_called_once_with(
        "infer",
        {
            "prompt": "prompt",
            "weights_ref": "/weights",
            "base_model": "model-id",
            "max_new_tokens": 77,
        },
    )


def test_infer_batch_generates_multiple_prompts_once_and_slices_each_continuation():
    from training.slm_helpers import infer_batch

    tokenizer = _FakeTokenizer({"short": [1], "long": [2, 3, 4]})
    model = _FakeModel()

    with _local_cached_inference(model, tokenizer):
        result = infer_batch(
            ["short", "long"],
            "/weights",
            "model-id",
            max_new_tokens=17,
            task_type="classification",
        )

    assert result == ["1001", "1004"]
    assert len(model.generate_calls) == 1
    assert model.generate_calls[0]["input_ids"] == [[99, 99, 1], [2, 3, 4]]
    assert model.generate_calls[0]["max_new_tokens"] == 17
    assert model.generate_calls[0]["do_sample"] is False
    assert [call["tokens"] for call in tokenizer.decode_calls] == [[1001], [1004]]
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.pad_token_id == 99
    assert all(call["enable_thinking"] is False for call in tokenizer.template_calls)
    assert all(call["add_generation_prompt"] is True for call in tokenizer.template_calls)


def test_infer_batch_rejects_mixed_overlength_prompts_before_padded_generation(
    monkeypatch,
):
    from training.slm_helpers import infer_batch

    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "60")
    tokenizer = _FakeTokenizer(
        {
            "short": [1, 2],
            "too long": list(range(11)),
        }
    )
    model = _FakeModel()

    with _local_cached_inference(model, tokenizer):
        with pytest.raises(
            ValueError,
            match=(
                r"prompt index 1.*11 tokens.*input budget 10"
                r".*max sequence length 60"
                r".*SLM_MAX_SEQ_LENGTH"
            ),
        ):
            infer_batch(
                ["short", "too long"],
                "/weights",
                "model-id",
                task_type="classification",
            )

    assert model.generate_calls == []
    assert tokenizer.decode_calls == []
    assert tokenizer.tokenize_calls[0]["truncation"] is False


def test_code_inference_reserves_1024_tokens_inside_4096_context(monkeypatch):
    from training.slm_helpers import infer_batch

    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "4096")
    tokenizer = _FakeTokenizer({"too long": [1] * 3073})
    model = _FakeModel()

    with _local_cached_inference(model, tokenizer):
        with pytest.raises(
            ValueError,
            match=r"3073 tokens.*input budget 3072.*1024 output tokens",
        ):
            infer_batch(
                ["too long"],
                "/weights",
                "model-id",
                max_new_tokens=1024,
                task_type="code_generation",
            )

    assert model.generate_calls == []


@pytest.mark.parametrize(
    "task_type",
    ["NER", "math_reasoning", "generation", "code_generation"],
)
def test_default_long_task_prompt_budget_fits_or_fails_actionably(
    monkeypatch,
    task_type,
):
    from eval.harness import eval_output_token_reserve
    from training.slm_helpers import infer_batch

    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "4096")
    reserve = eval_output_token_reserve(
        task_type,
        max_seq_length=4096,
    )
    input_budget = 4096 - reserve
    assert input_budget > 0
    tokenizer = _FakeTokenizer({
        "fits": [1] * input_budget,
        "too long": [1] * (input_budget + 1),
    })
    model = _FakeModel()

    with _local_cached_inference(model, tokenizer):
        assert len(
            infer_batch(
                ["fits"],
                "/weights",
                "model-id",
                max_new_tokens=reserve,
                task_type=task_type,
            )
        ) == 1
        with pytest.raises(
            ValueError,
            match=(
                rf"prompt index 0.*{input_budget + 1} tokens"
                rf".*input budget {input_budget}"
                rf".*{reserve} output tokens"
                r".*SLM_MAX_SEQ_LENGTH"
            ),
        ):
            infer_batch(
                ["too long"],
                "/weights",
                "model-id",
                max_new_tokens=reserve,
                task_type=task_type,
            )


def test_infer_and_single_item_batch_have_identical_prompt_and_output():
    from training.slm_helpers import infer, infer_batch

    tokenizer = _FakeTokenizer({"same prompt": [7, 8]})
    model = _FakeModel()

    with _local_cached_inference(model, tokenizer):
        single = infer("same prompt", "/weights", "model-id", max_new_tokens=9)
        batched = infer_batch(
            ["same prompt"],
            "/weights",
            "model-id",
            max_new_tokens=9,
            task_type="classification",
        )

    assert single == batched[0] == "1008"
    assert tokenizer.template_calls[0] == tokenizer.template_calls[1]


@pytest.mark.parametrize(
    ("task_type", "prompt_count", "expected_batch_sizes"),
    [
        ("classification", 33, [32, 1]),
        ("NER", 17, [16, 1]),
        ("generation", 17, [16, 1]),
        ("math_reasoning", 17, [16, 1]),
        ("code_generation", 17, [16, 1]),
    ],
)
def test_infer_batch_uses_task_aware_defaults(
    monkeypatch, task_type, prompt_count, expected_batch_sizes
):
    from training.slm_helpers import infer_batch

    monkeypatch.delenv("SLM_EVAL_BATCH_SIZE", raising=False)
    prompts = [f"p{index}" for index in range(prompt_count)]
    tokenizer = _FakeTokenizer(
        {prompt: [index + 1] for index, prompt in enumerate(prompts)}
    )
    model = _FakeModel()

    with _local_cached_inference(model, tokenizer):
        result = infer_batch(
            prompts,
            "/weights",
            "model-id",
            task_type=task_type,
        )

    assert len(result) == prompt_count
    assert [len(call["input_ids"]) for call in model.generate_calls] == (
        expected_batch_sizes
    )


def test_infer_batch_environment_override_controls_batch_size(monkeypatch):
    from training.slm_helpers import infer_batch

    monkeypatch.setenv("SLM_EVAL_BATCH_SIZE", "3")
    prompts = [f"p{index}" for index in range(5)]
    tokenizer = _FakeTokenizer(
        {prompt: [index + 1] for index, prompt in enumerate(prompts)}
    )
    model = _FakeModel()

    with _local_cached_inference(model, tokenizer):
        infer_batch(
            prompts,
            "/weights",
            "model-id",
            task_type="classification",
        )

    assert [len(call["input_ids"]) for call in model.generate_calls] == [3, 2]


@pytest.mark.parametrize("invalid_batch_size", ["not-an-int", "0"])
def test_invalid_batch_size_records_failed_timing_event(
    monkeypatch,
    invalid_batch_size,
):
    from training.slm_helpers import infer_batch

    monkeypatch.setenv("SLM_EVAL_BATCH_SIZE", invalid_batch_size)
    tokenizer = _FakeTokenizer({"prompt": [1]})
    model = _FakeModel()

    with _local_cached_inference(model, tokenizer) as (_, timing):
        with pytest.raises(ValueError, match="SLM_EVAL_BATCH_SIZE"):
            infer_batch(
                ["prompt"],
                "/weights",
                "model-id",
                task_type="classification",
            )

    event = timing.call_args.args[0]
    assert event.kind == "inference"
    assert event.name == "infer_batch"
    assert event.status == "error"
    assert event.metadata["initial_batch_size"] is None
    assert event.metadata["error_type"] == "ValueError"
    assert event.metadata["latency_ms"] >= 0
    assert model.generate_calls == []


def test_infer_batch_halves_after_oom_preserves_order_and_records_timing(monkeypatch):
    from training.slm_helpers import infer_batch

    monkeypatch.setenv("SLM_EVAL_BATCH_SIZE", "4")
    prompts = [f"p{index}" for index in range(5)]
    tokenizer = _FakeTokenizer(
        {prompt: [index + 1] for index, prompt in enumerate(prompts)}
    )
    model = _FakeModel(oom_above=2)
    cleanup_events = []
    fake_torch = _FakeTorch(cleanup_events=cleanup_events)

    with patch(
        "gc.collect",
        side_effect=lambda: cleanup_events.append("gc"),
    ):
        with _local_cached_inference(
            model,
            tokenizer,
            fake_torch=fake_torch,
        ) as (fake_torch, timing):
            result = infer_batch(
                prompts,
                "/weights",
                "model-id",
                task_type="classification",
            )

    assert result == ["1001", "1002", "1003", "1004", "1005"]
    assert [len(call["input_ids"]) for call in model.generate_calls] == [4, 2, 2, 1]
    assert fake_torch.cuda.empty_cache_calls == 1
    assert cleanup_events == ["gc", "empty_cache"]
    assert fake_torch.cuda.exception_types_at_cleanup == [None]
    event = timing.call_args.args[0]
    assert event.kind == "inference"
    assert event.name == "infer_batch"
    assert event.status == "success"
    assert event.metadata["initial_batch_size"] == 4
    assert event.metadata["attempted_batch_sizes"] == [4, 2, 2, 1]
    assert event.metadata["completed_batch_sizes"] == [2, 2, 1]
    assert event.metadata["oom_retries"] == 1
    assert event.metadata["latency_ms"] >= 0


def test_qwen35_oom_releases_generate_traceback_before_cuda_cleanup(monkeypatch):
    from training.slm_helpers import infer_batch

    class TransientTensor:
        pass

    class Qwen35OOMOnceModel(_FakeModel):
        def __init__(self):
            super().__init__()
            self.transient_ref = lambda: None
            self.raise_once = True

        def generate(self, *, input_ids, attention_mask, **kwargs):
            rows = [list(row) for row in input_ids]
            self.generate_calls.append(
                {
                    "input_ids": rows,
                    "attention_mask": [list(row) for row in attention_mask],
                    **kwargs,
                }
            )
            if self.raise_once:
                self.raise_once = False
                transient_tensor = TransientTensor()
                self.transient_ref = weakref.ref(transient_tensor)
                raise _FakeCudaOOM("Qwen3.5 CUDA out of memory")
            return [row + [1000 + row[-1]] for row in rows]

    monkeypatch.setenv("SLM_EVAL_BATCH_SIZE", "2")
    tokenizer = _FakeTokenizer({"p0": [1], "p1": [2]})
    model = Qwen35OOMOnceModel()
    fake_torch = _FakeTorch(
        cleanup_probe=lambda: model.transient_ref() is None,
    )

    with _local_cached_inference(
        model,
        tokenizer,
        fake_torch=fake_torch,
    ):
        result = infer_batch(
            ["p0", "p1"],
            "/weights",
            "model-id",
            task_type="classification",
        )

    assert result == ["1001", "1002"]
    assert [len(call["input_ids"]) for call in model.generate_calls] == [2, 1, 1]
    assert fake_torch.cuda.cleanup_probe_results == [True]
    assert fake_torch.cuda.exception_types_at_cleanup == [None]


def test_infer_batch_raises_diagnostic_when_batch_one_ooms(monkeypatch):
    from training.slm_helpers import infer_batch

    monkeypatch.setenv("SLM_EVAL_BATCH_SIZE", "1")
    tokenizer = _FakeTokenizer({"prompt": [1]})
    model = _FakeModel(oom_above=0)

    with _local_cached_inference(model, tokenizer) as (fake_torch, timing):
        with pytest.raises(
            RuntimeError,
            match=r"CUDA OOM.*batch_size=1.*task_type=classification.*model-id",
        ):
            infer_batch(
                ["prompt"],
                "/weights",
                "model-id",
                task_type="classification",
            )

    assert fake_torch.cuda.empty_cache_calls == 1
    event = timing.call_args.args[0]
    assert event.status == "error"
    assert event.metadata["attempted_batch_sizes"] == [1]


def test_infer_batch_empty_input_does_not_load_or_delegate(monkeypatch):
    from training.slm_helpers import infer_batch

    monkeypatch.setenv("SLM_CUDA_ISOLATION", "1")
    monkeypatch.delenv("SLM_CUDA_WORKER", raising=False)
    with patch("training.cuda_isolation.run_isolated") as worker:
        assert infer_batch(
            [],
            "/weights",
            "model-id",
            task_type="classification",
        ) == []

    worker.assert_not_called()


def test_infer_batch_delegates_whole_batch_to_disposable_worker(monkeypatch):
    from training.slm_helpers import infer_batch

    monkeypatch.setenv("SLM_CUDA_ISOLATION", "1")
    monkeypatch.delenv("SLM_CUDA_WORKER", raising=False)
    with patch(
        "training.cuda_isolation.run_isolated",
        return_value=["first", "second"],
    ) as worker:
        result = infer_batch(
            ["p1", "p2"],
            "/weights",
            "model-id",
            max_new_tokens=77,
            max_workers=3,
            task_type="NER",
        )

    assert result == ["first", "second"]
    worker.assert_called_once_with(
        "infer_batch",
        {
            "prompts": ["p1", "p2"],
            "weights_ref": "/weights",
            "base_model": "model-id",
            "max_new_tokens": 77,
            "max_workers": 3,
            "task_type": "NER",
        },
    )
