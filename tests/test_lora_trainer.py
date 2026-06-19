# tests/test_lora_trainer.py
import os, json, tempfile
from unittest.mock import patch, MagicMock
from training.lora_trainer import run_lora_training, TrainingConfig

def _fake_dataset(path):
    examples = [
        {"text": f"FREE PRIZE {i}", "label": "spam"} for i in range(20)
    ] + [
        {"text": f"Hey how are you {i}", "label": "ham"} for i in range(20)
    ]
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

def test_training_config_validation():
    config = TrainingConfig(
        base_model="Qwen/Qwen3-0.6B",
        nr_epochs=3,
        learning_rate=2e-4,
        batch_size=8,
        lora_rank=8,
    )
    assert config.lora_rank in {4, 8, 16, 32, 64}
    assert 0 < config.learning_rate < 1

def test_run_lora_training_returns_weights_ref():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "train.jsonl")
        _fake_dataset(dataset_path)
        config = TrainingConfig(
            base_model="Qwen/Qwen3-0.6B",
            nr_epochs=1,
            learning_rate=2e-4,
            batch_size=4,
            lora_rank=4,
        )
        with patch("training.lora_trainer._run_unsloth_training") as mock_train:
            mock_train.return_value = os.path.join(tmpdir, "checkpoint")
            os.makedirs(os.path.join(tmpdir, "checkpoint"))
            weights_ref = run_lora_training(dataset_path, config, output_dir=tmpdir)
        assert isinstance(weights_ref, str)
        assert len(weights_ref) > 0
