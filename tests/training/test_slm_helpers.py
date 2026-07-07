# tests/training/test_slm_helpers.py
import pytest
from unittest.mock import patch, MagicMock
from training.slm_helpers import infer_batch_gguf
from training.lora_trainer import TrainingOutput


def test_infer_batch_gguf_raises_import_error_when_llama_cpp_missing():
    with patch.dict("sys.modules", {"llama_cpp": None}):
        with pytest.raises(ImportError, match="llama-cpp-python"):
            infer_batch_gguf(["hello"], "/fake/model.gguf")


def test_infer_batch_gguf_returns_list_of_strings():
    mock_llama_instance = MagicMock()
    mock_llama_instance.return_value = {"choices": [{"text": "spam"}]}
    mock_llama_cls = MagicMock(return_value=mock_llama_instance)

    with patch("training.slm_helpers._gguf_cache", {}), \
         patch("training.slm_helpers._gguf_cache_order", []):
        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_cls)}):
            # Clear module-level cache to force load
            import training.slm_helpers as sh
            sh._gguf_cache.clear()
            sh._gguf_cache_order.clear()
            result = infer_batch_gguf(["hello", "world"], "/fake/model.gguf")

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(s, str) for s in result)


def test_train_returns_training_output():
    from training.slm_helpers import train
    with patch("training.slm_helpers.run_lora_training") as mock_train:
        mock_train.return_value = TrainingOutput(weights_ref="/ckpt", gguf_path=None)
        result = train("/data.jsonl", "model-id", 1, 2e-4, 8, 8)
    assert isinstance(result, TrainingOutput)
    assert result.weights_ref == "/ckpt"
    assert result.gguf_path is None
