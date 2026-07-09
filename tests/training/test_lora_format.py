import sys
from unittest.mock import patch, MagicMock


def _make_config(task_type):
    from training.lora_trainer import TrainingConfig
    return TrainingConfig(
        base_model="dummy", nr_epochs=1, learning_rate=2e-4,
        batch_size=8, lora_rank=8, task_type=task_type,
    )


def _run_training_and_capture(ds_path, task_type):
    """
    Run _run_unsloth_training with heavy mocking and capture the list of dicts
    passed to Dataset.from_list.  Returns the captured list.
    """
    trained_texts = []

    mock_flm = MagicMock()
    mock_model = MagicMock()
    mock_model.chat_template = None
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
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
        _run_unsloth_training(
            str(ds_path),
            _make_config(task_type),
            str(ds_path.parent),
            task_type=task_type,
        )

    return trained_texts


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
