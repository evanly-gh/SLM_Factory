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
        self.template_calls.append({
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        })
        prompt = f"<user>{messages[0]['content']}</user><assistant>"
        if add_generation_prompt:
            return prompt
        return f"{prompt}{messages[-1]['content']}</assistant>"


def _make_config(task_type, base_model="dummy"):
    from training.lora_trainer import TrainingConfig
    return TrainingConfig(
        base_model=base_model, nr_epochs=1, learning_rate=2e-4,
        batch_size=8, lora_rank=8, task_type=task_type,
    )


def _run_training_and_capture(ds_path, task_type, base_model="dummy"):
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
                _make_config(task_type, base_model=base_model),
                str(ds_path.parent),
                task_type=task_type,
            )

    return trained_texts


def test_qwen_training_fails_if_chat_template_is_missing(tmp_path):
    import json

    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(json.dumps({"text": "hello", "label": "greeting"}) + "\n")

    with pytest.raises(RuntimeError, match="cannot enforce non-thinking"):
        _run_training_and_capture(
            ds_path,
            "classification",
            base_model="Qwen/Qwen3-1.7B",
        )


def test_math_examples_include_answer(tmp_path):
    """math_reasoning training text must contain the answer."""
    import json

    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(json.dumps({"prompt": "1+1=?", "answer": "2"}) + "\n")

    trained_texts = _run_training_and_capture(ds_path, "math_reasoning")

    assert trained_texts, "Dataset.from_list was never called — format routing broken"
    assert any("2" in str(t) for t in trained_texts), (
        f"math_reasoning training text should contain the answer '2'; got: {trained_texts}"
    )


def test_code_task_uses_generation_branch(tmp_path):
    """code_generation training text must contain the answer (not just the prompt)."""
    import json

    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(
        json.dumps({"prompt": "def add(a,b):", "answer": "return a+b"}) + "\n"
    )

    trained_texts = _run_training_and_capture(ds_path, "code_generation")

    assert trained_texts, "Dataset.from_list was never called — format routing broken"
    assert any("return a+b" in str(t) for t in trained_texts), (
        f"code_generation training text should contain the answer 'return a+b'; got: {trained_texts}"
    )


def test_apps_training_reuses_eval_code_prompt_with_starter_interface(
    tmp_path,
    monkeypatch,
):
    import json

    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "4096")
    row = {
        "text": "Add two integers.",
        "answer": (
            "class Solution:\n"
            "    def add(self, a, b):\n"
            "        return a + b"
        ),
        "starter_code": (
            "class Solution:\n"
            "    def add(self, a, b):\n"
            "        pass"
        ),
        "input_output": {
            "fn_name": "add",
            "inputs": ["[2, 3]"],
            "outputs": ["5"],
        },
        "execution_mode": "call_based",
        "fn_name": "add",
        "entry_point": "add",
    }
    ds_path = tmp_path / "apps.jsonl"
    ds_path.write_text(json.dumps(row) + "\n")

    trained_texts = _run_training_and_capture(ds_path, "code_generation")
    rendered = trained_texts[0]["text"]

    from eval.scorers.generation import build_code_prompt

    assert build_code_prompt(row) in rendered
    assert row["answer"] in rendered
    assert "Required entry point: add" in rendered
    assert row["starter_code"] in rendered


def test_code_cot_is_trained_as_executable_python_comments():
    from training.lora_trainer import build_code_training_target

    target = build_code_training_target(
        {
            "answer": "def add(a, b):\n    return a + b",
            "cot_reasoning": "Use direct addition.\nReturn the computed value.",
        }
    )

    assert target.startswith(
        "# Use direct addition.\n# Return the computed value.\n"
    )
    compile(target, "<code_target>", "exec")


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
                _make_config("classification"),
                str(tmp_path),
                task_type="classification",
            )

    assert [call["add_generation_prompt"] for call in mock_tokenizer.template_calls] == [
        True,
        False,
    ]
    assert all(
        call["enable_thinking"] is False
        for call in mock_tokenizer.template_calls
    )
