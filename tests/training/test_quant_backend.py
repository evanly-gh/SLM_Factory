# tests/training/test_quant_backend.py
"""The routing layer between the two quantization backends.

These tests exist because the GGUF path grew its call sites one at a time — evaluate, the
downward probe, the interpolation probe, the final on-device verification — and B161 is the
record of what one missed site costs. `training/quant_backend.py` is the single place that
choice is now made, so it is the place to pin it.
"""
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import training.quant_backend as quant_backend


def test_default_backend_is_llama_cpp(monkeypatch):
    """Every number this project has published came from llama.cpp. An unset flag must mean that."""
    monkeypatch.delenv("SLM_QUANT_BACKEND", raising=False)
    assert quant_backend.resolve_backend() == "llama_cpp"


def test_environment_selects_the_backend(monkeypatch):
    monkeypatch.setenv("SLM_QUANT_BACKEND", "mnn")
    assert quant_backend.resolve_backend() == "mnn"


def test_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("SLM_QUANT_BACKEND", "mnn")
    assert quant_backend.resolve_backend("llama_cpp") == "llama_cpp"


def test_unknown_backend_raises_rather_than_falling_back(monkeypatch):
    """A typo must not silently produce llama.cpp numbers under an MNN label."""
    monkeypatch.setenv("SLM_QUANT_BACKEND", "mnn-llm")
    with pytest.raises(ValueError, match="Unknown quantization backend"):
        quant_backend.resolve_backend()


def test_artifact_names_are_distinct_per_backend_and_quant():
    # Checked by exact name rather than by globbing the directory: the same base model is selected
    # at two tiers as two quant variants and their checkpoint paths collide, which is how the Q8_0
    # tier once scored a Q4_K_M file (B161).
    assert quant_backend.artifact_name("Q4_K_M", "llama_cpp") == "model-q4_k_m.gguf"
    assert quant_backend.artifact_name("Q8_0", "llama_cpp") == "model-q8_0.gguf"
    assert quant_backend.artifact_name("Q4_K_M", "mnn") == "model-mnn-q4"
    assert quant_backend.artifact_name("Q8_0", "mnn") == "model-mnn-q8"


def test_backends_cache_into_separate_artifact_trees():
    """`gguf` is unchanged so existing warm caches still hit; MNN cannot collide with them."""
    assert quant_backend.artifacts_subdir("llama_cpp") == "gguf"
    assert quant_backend.artifacts_subdir("mnn") == "mnn"


def test_quantize_routes_to_llama_cpp(monkeypatch):
    monkeypatch.delenv("SLM_QUANT_BACKEND", raising=False)
    with (
        patch("training.quantize.quantize_from_model_spec", return_value="/out/m.gguf") as gguf,
        patch("training.quantize_mnn.export_from_model_spec") as mnn,
    ):
        result = quant_backend.quantize_from_model_spec("/ckpt", "/out", "Q4_K_M")

    assert result == "/out/m.gguf"
    gguf.assert_called_once_with("/ckpt", "/out", "Q4_K_M")
    mnn.assert_not_called()


def test_quantize_routes_to_mnn(monkeypatch):
    monkeypatch.setenv("SLM_QUANT_BACKEND", "mnn")
    with (
        patch("training.quantize.quantize_from_model_spec") as gguf,
        patch(
            "training.quantize_mnn.export_from_model_spec",
            return_value="/out/model-mnn-q4",
        ) as mnn,
    ):
        result = quant_backend.quantize_from_model_spec("/ckpt", "/out", "Q4_K_M")

    assert result == "/out/model-mnn-q4"
    mnn.assert_called_once_with("/ckpt", "/out", "Q4_K_M")
    gguf.assert_not_called()


def test_validation_routes_to_mnn_and_carries_the_quant(monkeypatch):
    """MNN validation needs the quant name; GGUF validation does not and must not be given it.

    The reason is asymmetric and worth pinning: a GGUF's quantization is in its file format, so a
    Q4_K_M file cannot be mistaken for a Q8_0 one. An MNN export at 4 bits and one at 8 bits are
    the same five filenames, so the only way to catch a mislabelled artifact is to re-read the
    exporter's own record of `--quant_bit`, which needs the expected value passed in.
    """
    monkeypatch.setenv("SLM_QUANT_BACKEND", "mnn")
    with patch("training.quantize_mnn.validate_and_record_mnn", return_value={}) as validate:
        quant_backend.validate_and_record("/out/model-mnn-q4", base_model="m", quant="Q4_K_M")
    validate.assert_called_once_with("/out/model-mnn-q4", base_model="m", quant="Q4_K_M")

    monkeypatch.setenv("SLM_QUANT_BACKEND", "llama_cpp")
    with patch("training.quantize.validate_and_record_gguf", return_value={}) as validate:
        quant_backend.validate_and_record("/out/m.gguf", base_model="m", quant="Q4_K_M")
    validate.assert_called_once_with("/out/m.gguf", base_model="m")


def test_artifact_existence_is_a_file_for_gguf_and_a_directory_for_mnn(tmp_path):
    gguf = tmp_path / "model-q4_k_m.gguf"
    gguf.write_bytes(b"x")
    mnn_dir = tmp_path / "model-mnn-q4"
    mnn_dir.mkdir()

    assert quant_backend.artifact_exists(str(gguf), "llama_cpp") is True
    assert quant_backend.artifact_exists(str(mnn_dir), "llama_cpp") is False
    assert quant_backend.artifact_exists(str(mnn_dir), "mnn") is True
    assert quant_backend.artifact_exists(str(gguf), "mnn") is False


def test_size_is_measured_comparably_across_backends(tmp_path):
    """Both must report the bytes that SHIP, or a size chart silently favours one backend.

    The MNN export directory also holds a tokenizer and two JSON records; counting those against a
    GGUF, whose tokenizer lives inside the single file, would overstate MNN.
    """
    gguf = tmp_path / "model-q4_k_m.gguf"
    gguf.write_bytes(b"0" * (2 * 1024 * 1024))
    assert quant_backend.artifact_size_mb(str(gguf), "llama_cpp") == pytest.approx(2.0)

    mnn_dir = tmp_path / "model-mnn-q4"
    mnn_dir.mkdir()
    (mnn_dir / "llm.mnn").write_bytes(b"0" * (1024 * 1024))
    (mnn_dir / "llm.mnn.weight").write_bytes(b"0" * (1024 * 1024))
    (mnn_dir / "tokenizer.mtok").write_bytes(b"0" * (5 * 1024 * 1024))
    assert quant_backend.artifact_size_mb(str(mnn_dir), "mnn") == pytest.approx(2.0)


def test_backend_label_is_readable_in_logs():
    assert quant_backend.backend_label("llama_cpp") == "llama.cpp/GGUF"
    assert quant_backend.backend_label("mnn") == "MNN"


def test_config_rejects_an_unknown_backend_at_import(monkeypatch):
    """The flag is validated at config import so a bad value fails before any training starts."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setenv("SLM_QUANT_BACKEND", "tensorrt")
    monkeypatch.delitem(sys.modules, "config.config", raising=False)
    with pytest.raises(RuntimeError, match="not a known quantization backend"):
        import config.config  # noqa: F401
    monkeypatch.delitem(sys.modules, "config.config", raising=False)
