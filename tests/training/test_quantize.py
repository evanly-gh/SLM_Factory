# tests/training/test_quantize.py
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import training.quantize as quantize_module
from training.quantize import QuantizationResult, quantize_from_model_spec


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


def test_resolve_hf_snapshot_pins_model_revision_when_available():
    model_info = MagicMock(return_value=SimpleNamespace(sha="abc123"))
    snapshot_download = MagicMock(return_value="/cache/snapshots/abc123")
    hub = SimpleNamespace(
        model_info=model_info,
        snapshot_download=snapshot_download,
    )

    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        result = quantize_module.resolve_hf_snapshot("Qwen/Qwen3.5-2B")

    assert result == "/cache/snapshots/abc123"
    model_info.assert_called_once_with("Qwen/Qwen3.5-2B")
    snapshot_download.assert_called_once()
    assert snapshot_download.call_args.kwargs["revision"] == "abc123"


def test_validation_failure_does_not_write_cache_sidecar(tmp_path):
    gguf_path = tmp_path / "model-q4_k_m.gguf"
    gguf_path.write_bytes(b"incomplete-qwen35-gguf")
    llama = MagicMock()
    llama.Llama.side_effect = ValueError(
        "missing tensor blk.24.attn_norm.weight"
    )

    with patch.dict(sys.modules, {"llama_cpp": llama}):
        with pytest.raises(
            RuntimeError, match=r"blk\.24\.attn_norm\.weight"
        ):
            quantize_module.validate_and_record_gguf(str(gguf_path))

    assert not (tmp_path / "model-q4_k_m.gguf.validation.json").exists()


def test_validated_cache_hit_requires_matching_size_and_hash(tmp_path):
    gguf_path = tmp_path / "model-q4_k_m.gguf"
    gguf_path.write_bytes(b"complete-gguf")
    # The completion call must return real text: validation now generation-tests the artifact, and a
    # bare MagicMock is neither a passing nor a failing model, just an unrepresentative one.
    llama_instance = MagicMock(
        return_value={"choices": [{"text": " Hello, how can I help?"}]}
    )
    llama_instance.close = MagicMock()
    llama = SimpleNamespace(
        __version__="0.3.test",
        Llama=MagicMock(return_value=llama_instance),
    )

    with patch.dict(sys.modules, {"llama_cpp": llama}):
        quantize_module.validate_and_record_gguf(str(gguf_path))

    sidecar_path = quantize_module.gguf_validation_sidecar_path(str(gguf_path))
    sidecar = json.loads(open(sidecar_path, encoding="utf-8").read())
    assert sidecar["file_size"] == len(b"complete-gguf")
    assert len(sidecar["sha256"]) == 64
    assert sidecar["tool_versions"]["llama_cpp_python"] == "0.3.test"
    assert quantize_module.validated_gguf_cache_hit(str(gguf_path)) is True

    gguf_path.write_bytes(b"tampered-gguf")
    assert quantize_module.validated_gguf_cache_hit(str(gguf_path)) is False
