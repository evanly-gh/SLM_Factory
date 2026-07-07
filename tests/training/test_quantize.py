# tests/training/test_quantize.py
import pytest
from unittest.mock import patch
from training.quantize import quantize_from_model_spec, QuantizationResult


def _mock_result(success=True, gguf_path="/out/model-q4_k_m.gguf", error=None, method="q4_k_m"):
    return QuantizationResult(
        gguf_path=gguf_path,
        original_size_mb=1000.0,
        quantized_size_mb=500.0,
        compression_ratio=2.0,
        method=method,
        success=success,
        error=error,
    )


@patch("training.quantize.quantize_checkpoint")
def test_q4_k_m_maps_to_correct_method(mock_qc):
    mock_qc.return_value = _mock_result(gguf_path="/out/model-q4_k_m.gguf")
    result = quantize_from_model_spec("/checkpoint", "/out", "Q4_K_M")
    mock_qc.assert_called_once_with("/checkpoint", "/out", "q4_k_m")
    assert result == "/out/model-q4_k_m.gguf"


@patch("training.quantize.quantize_checkpoint")
def test_q8_0_maps_to_correct_method(mock_qc):
    mock_qc.return_value = _mock_result(gguf_path="/out/model-q8_0.gguf", method="q8_0")
    result = quantize_from_model_spec("/checkpoint", "/out", "Q8_0")
    mock_qc.assert_called_once_with("/checkpoint", "/out", "q8_0")
    assert result == "/out/model-q8_0.gguf"


def test_unknown_quant_raises_value_error():
    with pytest.raises(ValueError, match="Unknown quant"):
        quantize_from_model_spec("/checkpoint", "/out", "INT8")


@patch("training.quantize.quantize_checkpoint")
def test_failed_quantization_raises_runtime_error(mock_qc):
    mock_qc.return_value = _mock_result(success=False, gguf_path=None, error="llama-quantize not found")
    with pytest.raises(RuntimeError, match="Quantization failed"):
        quantize_from_model_spec("/checkpoint", "/out", "Q4_K_M")


@patch("training.quantize.quantize_checkpoint")
def test_f16_fallback_raises_runtime_error(mock_qc):
    mock_qc.return_value = QuantizationResult(
        gguf_path="/out/model-f16.gguf",
        original_size_mb=1000.0,
        quantized_size_mb=2000.0,
        compression_ratio=0.5,
        method="f16",
        success=True,
        error="llama-quantize not found; produced f16 GGUF only",
    )
    with pytest.raises(RuntimeError, match="Quantization failed"):
        quantize_from_model_spec("/checkpoint", "/out", "Q4_K_M")
