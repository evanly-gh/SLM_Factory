# training/quantize.py
"""
INT4 quantization and hardware profiling for Android-deployable models.

Phase 1: theoretical estimates from ANDROID_POOL benchmarks.
Phase 2: actual W4A16 quantization via llama.cpp + measured hardware profiles.

Design doc §6.5: 'Phase 2 replaces theoretical estimates with actual quantization
and measured values via Qualcomm AI Hub or physical device.'
"""
import os
import shutil
import subprocess
from dataclasses import dataclass
from config.android_pool import ANDROID_POOL, ModelSpec


@dataclass
class QuantizationResult:
    """Result of quantizing a fine-tuned checkpoint to INT4."""
    gguf_path: str | None
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    method: str
    success: bool
    error: str | None = None


def theoretical_hardware_profile(model_id: str) -> dict:
    """Return theoretical hardware estimates for a model from the Android pool.
    Phase 1 uses benchmark-derived estimates, not measurements.
    """
    spec = next((m for m in ANDROID_POOL if m.model_id == model_id), None)
    if spec is None:
        return {
            "model_id": model_id,
            "size_mb": None,
            "tier": None,
            "tok_s_snapdragon_778g": None,
            "peak_memory_mb": None,
            "phase": "theoretical",
            "note": "Model not in Android pool",
        }
    return {
        "model_id": spec.model_id,
        "size_mb": spec.size_mb,
        "tier": spec.tier,
        "tok_s_snapdragon_660": spec.tok_s_snapdragon_660,
        "tok_s_snapdragon_778g": spec.tok_s_snapdragon_778g,
        "tok_s_snapdragon_8gen3": spec.tok_s_snapdragon_8gen3,
        "peak_memory_mb": spec.peak_memory_mb,
        "phase": "theoretical",
    }


def _dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


def _file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def quantize_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    method: str = "q4_k_m",
) -> QuantizationResult:
    """Quantize a HuggingFace checkpoint to INT4 GGUF via llama.cpp.

    Steps:
    1. Convert HF checkpoint to GGUF using convert_hf_to_gguf.py
    2. Quantize the f16 GGUF to Q4_K_M using llama-quantize
    3. Report actual file sizes

    Falls back gracefully if llama.cpp tools are not installed.
    """
    os.makedirs(output_dir, exist_ok=True)
    original_size = _dir_size_mb(checkpoint_path) if os.path.isdir(checkpoint_path) else _file_size_mb(checkpoint_path)

    f16_gguf = os.path.join(output_dir, "model-f16.gguf")
    q4_gguf = os.path.join(output_dir, f"model-{method}.gguf")

    # Step 1: Convert HF → GGUF
    convert_script = (
        shutil.which("convert_hf_to_gguf")
        or shutil.which("convert_hf_to_gguf.py")
        or shutil.which("convert-hf-to-gguf.py")
    )
    if not convert_script:
        return QuantizationResult(
            gguf_path=None, original_size_mb=original_size, quantized_size_mb=0,
            compression_ratio=0, method=method, success=False,
            error="convert_hf_to_gguf not found. Clone llama.cpp and add it to PATH.",
        )

    try:
        subprocess.run(
            [convert_script, checkpoint_path, "--outfile", f16_gguf, "--outtype", "f16"],
            shell=False, check=True, capture_output=True, text=True, timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return QuantizationResult(
            gguf_path=None, original_size_mb=original_size, quantized_size_mb=0,
            compression_ratio=0, method=method, success=False,
            error=f"GGUF conversion failed: {e}. Install llama.cpp tools.",
        )

    # Step 2: Quantize f16 → Q4_K_M
    quantize_bin = shutil.which("llama-quantize")
    if not quantize_bin:
        quantize_bin = shutil.which("quantize")
    if not quantize_bin:
        return QuantizationResult(
            gguf_path=f16_gguf, original_size_mb=original_size,
            quantized_size_mb=_file_size_mb(f16_gguf),
            compression_ratio=original_size / max(_file_size_mb(f16_gguf), 0.1),
            method="f16", success=True,
            error="llama-quantize not found; produced f16 GGUF only",
        )

    try:
        subprocess.run(
            [quantize_bin, f16_gguf, q4_gguf, method.upper()],
            check=True, capture_output=True, text=True, timeout=600,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return QuantizationResult(
            gguf_path=f16_gguf, original_size_mb=original_size,
            quantized_size_mb=_file_size_mb(f16_gguf),
            compression_ratio=original_size / max(_file_size_mb(f16_gguf), 0.1),
            method="f16", success=True,
            error=f"Quantization to {method} failed ({e}); produced f16 GGUF only",
        )

    q4_size = _file_size_mb(q4_gguf)
    # Clean up f16 intermediate
    if os.path.exists(f16_gguf) and os.path.exists(q4_gguf):
        os.remove(f16_gguf)

    return QuantizationResult(
        gguf_path=q4_gguf, original_size_mb=original_size,
        quantized_size_mb=q4_size,
        compression_ratio=original_size / max(q4_size, 0.1),
        method=method, success=True,
    )


_QUANT_METHOD_MAP: dict[str, str] = {
    "Q4_K_M": "q4_k_m",
    "Q8_0": "q8_0",
}


def quantize_from_model_spec(checkpoint_path: str, output_dir: str, quant: str) -> str:
    """
    Quantize a merged HF checkpoint to GGUF using the quant string from ModelSpec.

    Args:
        checkpoint_path: Path to a merged full-precision HF checkpoint directory.
        output_dir: Directory to write the GGUF file into.
        quant: ModelSpec.quant value — "Q4_K_M" or "Q8_0".

    Returns:
        Absolute path to the produced GGUF file.

    Raises:
        ValueError: if quant is not a recognized value.
        RuntimeError: if llama.cpp tools are not installed or quantization fails.
    """
    if quant not in _QUANT_METHOD_MAP:
        raise ValueError(
            f"Unknown quant {quant!r}. Valid values: {list(_QUANT_METHOD_MAP)}"
        )
    method = _QUANT_METHOD_MAP[quant]
    result = quantize_checkpoint(checkpoint_path, output_dir, method)
    if not result.success or result.gguf_path is None or result.method != method:
        raise RuntimeError(
            f"Quantization failed for {checkpoint_path!r} → {quant} ({method}): "
            f"{result.error or f'produced {result.method!r} instead of {method!r}'}. "
            f"Ensure llama.cpp tools (llama-quantize, convert_hf_to_gguf) are installed."
        )
    return result.gguf_path
