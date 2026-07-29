# training/quantize.py
"""
INT4 quantization and hardware profiling for Android-deployable models.

Phase 1: theoretical estimates from ANDROID_POOL benchmarks.
Phase 2: actual W4A16 quantization via llama.cpp + measured hardware profiles.

Design doc §6.5: 'Phase 2 replaces theoretical estimates with actual quantization
and measured values via Qualcomm AI Hub or physical device.'
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from config.android_pool import (
    ANDROID_POOL,
    measured_size_mb,
    resolve_model_selector,
)

# Chip whose recorded measurements this profile reports. Host-side callers have no
# target device, so the default is the local host key rather than a phone's — a
# measurement is only ever valid for the silicon it was taken on.
DEFAULT_PROFILE_CHIP = os.environ.get("SLM_PROFILE_CHIP", "host_cpu")

# convert_hf_to_gguf / llama-quantize wall-clock ceiling. 600s is fine for a merged
# checkpoint on local scratch, but converting a large BASE snapshot straight out of the
# shared HF cache on Lustre reads ~9GB over the network filesystem and blows past it
# (observed on Qwen3.5-4B). Override with SLM_QUANT_TIMEOUT_S for big/remote sources.
_QUANT_SUBPROCESS_TIMEOUT_S = int(os.environ.get("SLM_QUANT_TIMEOUT_S", "600"))


class QuantizationInfrastructureError(RuntimeError):
    """The required quantized artifact could not be built or validated."""


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


_HF_SNAPSHOT_IGNORE_PATTERNS = [
    "*.gguf",
    "original/*",
    "*.pth",
    "consolidated*",
]
_GGUF_VALIDATION_SCHEMA_VERSION = 1
_GGUF_VALIDATION_SUFFIX = ".validation.json"


def resolve_hf_snapshot(model_id: str) -> str:
    """Download a base model and return its immutable local HF snapshot path."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import model_info, snapshot_download

    revision = None
    try:
        revision = getattr(model_info(model_id), "sha", None)
    except Exception:  # noqa: BLE001 - snapshot_download still supports offline cache
        pass
    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        ignore_patterns=_HF_SNAPSHOT_IGNORE_PATTERNS,
    )


def gguf_validation_sidecar_path(gguf_path: str) -> str:
    """Return the atomic validation-record path for a GGUF artifact."""
    return f"{gguf_path}{_GGUF_VALIDATION_SUFFIX}"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def validate_and_record_gguf(gguf_path: str) -> dict:
    """Load every tensor with llama.cpp, then atomically record file identity."""
    if not os.path.isfile(gguf_path):
        raise RuntimeError(f"GGUF validation failed: file not found: {gguf_path}")
    initial_size = os.path.getsize(gguf_path)
    if initial_size <= 0:
        raise RuntimeError(f"GGUF validation failed: empty file: {gguf_path}")

    try:
        import llama_cpp
    except (ImportError, TypeError) as exc:
        raise RuntimeError(
            "GGUF validation requires llama-cpp-python to perform a real model load"
        ) from exc

    model = None
    try:
        model = llama_cpp.Llama(
            model_path=os.path.abspath(gguf_path),
            n_ctx=128,
            n_batch=16,
            n_gpu_layers=0,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - preserve llama.cpp loader detail
        raise RuntimeError(f"GGUF model-load validation failed: {exc}") from exc
    finally:
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                close()

    final_size = os.path.getsize(gguf_path)
    if final_size != initial_size:
        raise RuntimeError(
            f"GGUF changed during validation: {initial_size} -> {final_size} bytes"
        )
    record = {
        "schema_version": _GGUF_VALIDATION_SCHEMA_VERSION,
        "file_size": final_size,
        "sha256": _sha256_file(gguf_path),
        "tool_versions": {
            "llama_cpp_python": str(
                getattr(llama_cpp, "__version__", "unknown")
            ),
            "python": platform.python_version(),
        },
    }
    _atomic_write_json(gguf_validation_sidecar_path(gguf_path), record)
    return record


def validated_gguf_cache_hit(gguf_path: str) -> bool:
    """Return whether the GGUF exactly matches a successful load-validation record."""
    sidecar_path = gguf_validation_sidecar_path(gguf_path)
    if not os.path.isfile(gguf_path) or not os.path.isfile(sidecar_path):
        return False
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            record = json.load(handle)
        return (
            record.get("schema_version") == _GGUF_VALIDATION_SCHEMA_VERSION
            and record.get("file_size") == os.path.getsize(gguf_path)
            and isinstance(record.get("tool_versions"), dict)
            and bool(record["tool_versions"])
            and record.get("sha256") == _sha256_file(gguf_path)
        )
    except (OSError, TypeError, ValueError):
        return False


def invalidate_gguf_cache(gguf_path: str) -> None:
    """Remove only a derived GGUF and its validation record."""
    for path in (gguf_path, gguf_validation_sidecar_path(gguf_path)):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def theoretical_hardware_profile(model_id: str) -> dict:
    """Return this model's known deployment facts from the Android pool.

    Name kept for call-site compatibility, but nothing here is "theoretical" any more:
    it used to emit modelled tok/s for three chips plus a modelled peak RAM. Those were
    fabricated and have been removed pool-wide. What remains is on-disk weight size
    (real arithmetic over the weight files, and a measured value when
    config/measured_metrics.json has one) plus any recorded runtime measurement.
    """
    spec = resolve_model_selector(ANDROID_POOL, model_id)
    if spec is None:
        return {
            "model_id": model_id,
            "size_mb": None,
            "tier": None,
            "measured": None,
            "note": "Model not in Android pool",
        }
    return {
        "selector": spec.selector,
        "model_id": spec.model_id,
        "size_mb": spec.size_mb,
        "size_source": (
            "measured"
            if measured_size_mb(spec.model_id, spec.quant) is not None
            else "bytes-per-parameter arithmetic"
        ),
        "tier": spec.tier,
        # Runtime metrics are present only where a real measurement was recorded.
        "measured": spec.measured(DEFAULT_PROFILE_CHIP),
        "measured_chip": DEFAULT_PROFILE_CHIP,
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
            shell=False, check=True, capture_output=True, text=True,
            timeout=_QUANT_SUBPROCESS_TIMEOUT_S,
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
            check=True, capture_output=True, text=True,
            timeout=_QUANT_SUBPROCESS_TIMEOUT_S,
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
