import sys
from unittest.mock import patch, MagicMock

import pytest


class _TokenizerDouble:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 3
    eos_token = "<eos>"

    def __init__(self, *, chat_template=None):
        self.chat_template = chat_template
        self.template_calls = []
        self.save_pretrained = MagicMock()

    def __call__(self, text, *, truncation, add_special_tokens):
        return {"input_ids": [ord(char) + 10 for char in text]}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        # A real tokenizer with no chat_template raises rather than inventing a format. The double used
        # to render regardless, which let `test_qwen_training_fails_if_chat_template_is_missing` reach
        # the renderer at all — so once B290's alignment guard started rendering, the double answered
        # for a tokenizer that in reality could not.
        if self.chat_template is None:
            raise ValueError("cannot use apply_chat_template because this tokenizer has no template")
        self.template_calls.append({
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        })
        # Real ChatML, matching `_qwen_no_think_prompt` for hybrid Qwen3: the invented
        # `<user>..</user>` shape is not something a Qwen tokenizer can emit, and the alignment guard
        # correctly rejects it.
        prompt = (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        if add_generation_prompt:
            return prompt
        return f"{prompt}{messages[-1]['content']}<|im_end|>\n"


def _make_config(task, base_model="dummy"):
    from training.lora_trainer import TrainingConfig
    return TrainingConfig(
        base_model=base_model, nr_epochs=1, learning_rate=2e-4,
        batch_size=8, lora_rank=8, task=task,
    )


def _run_training_and_capture(ds_path, task, base_model="dummy"):
    """
    Run _run_unsloth_training with heavy mocking and capture the list of dicts
    passed to Dataset.from_list.  Returns the captured list.
    """
    trained_texts = []

    mock_flm = MagicMock()
    mock_model = MagicMock()
    mock_model.chat_template = None
    mock_tokenizer = _TokenizerDouble()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)

    mock_ds = MagicMock()
    mock_ds.from_list.side_effect = lambda exs: (trained_texts.extend(exs), MagicMock())[1]

    unsloth_mock = MagicMock()
    unsloth_mock.FastLanguageModel = mock_flm

    with patch.dict(sys.modules, {
        "unsloth": unsloth_mock,
        "transformers": MagicMock(),
        "trl": MagicMock(),
        "torch": MagicMock(),
        "datasets": MagicMock(Dataset=mock_ds),
        "eval.scorers.classification": MagicMock(CLASSIFY_PROMPT="Classify: {text}"),
    }):
        # Re-import so that patched sys.modules are picked up fresh.
        if "training.lora_trainer" in sys.modules:
            del sys.modules["training.lora_trainer"]
        from training.lora_trainer import _run_unsloth_training
        with patch("training.lora_trainer._ensure_model_cached"):
            _run_unsloth_training(
                str(ds_path),
                _make_config(task, base_model=base_model),
                str(ds_path.parent),
                task=task,
            )

    return trained_texts


def test_qwen_training_fails_if_chat_template_is_missing(tmp_path):
    import json

    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(json.dumps({"text": "hello", "label": "greeting"}) + "\n")

    with pytest.raises(RuntimeError, match="cannot enforce non-thinking"):
        _run_training_and_capture(
            ds_path,
            "clinc150",
            base_model="Qwen/Qwen3-1.7B",
        )


def test_math_examples_include_answer(tmp_path):
    """A math task's training text must contain the answer."""
    import json

    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(json.dumps({"prompt": "1+1=?", "answer": "2"}) + "\n")

    trained_texts = _run_training_and_capture(ds_path, "gsm8k")

    assert trained_texts, "Dataset.from_list was never called — format routing broken"
    assert any("2" in str(t) for t in trained_texts), (
        f"gsm8k training text should contain the answer '2'; got: {trained_texts}"
    )


def test_training_chat_template_explicitly_disables_thinking(tmp_path):
    import json

    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(json.dumps({"text": "hello", "label": "greeting"}) + "\n")
    mock_model = MagicMock()
    mock_tokenizer = _TokenizerDouble(chat_template="qwen-template")
    mock_flm = MagicMock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
    mock_ds = MagicMock()
    unsloth_mock = MagicMock(FastLanguageModel=mock_flm)

    with patch.dict(sys.modules, {
        "unsloth": unsloth_mock,
        "transformers": MagicMock(),
        "trl": MagicMock(),
        "torch": MagicMock(),
        "datasets": MagicMock(Dataset=mock_ds),
    }):
        if "training.lora_trainer" in sys.modules:
            del sys.modules["training.lora_trainer"]
        from training.lora_trainer import _run_unsloth_training

        with patch("training.lora_trainer._ensure_model_cached"):
            _run_unsloth_training(
                str(ds_path),
                _make_config("clinc150"),
                str(tmp_path),
                task="clinc150",
            )

    assert [call["add_generation_prompt"] for call in mock_tokenizer.template_calls] == [
        True,
        False,
    ]
    assert all(
        call["enable_thinking"] is False
        for call in mock_tokenizer.template_calls
    )
