# training/quant_backend.py
"""
The single place the pipeline decides WHICH on-device runtime a quantized artifact is built for.

Two backends exist (see `config.QUANT_BACKEND` for the full rationale): llama.cpp's GGUF, which
is the default and the format every published number so far was measured in, and MNN, which is
what the Qwen-family mobile stack and MNN-Chat actually run. One run uses one of them.

Everything backend-specific about an artifact — how it is built, how it is validated, how a warm
cache is recognised, what "its size" means, and which engine scores it — is routed through the
functions here, so a caller can hold a quantized artifact without knowing its format. That is the
whole reason this module exists rather than an `if backend == "mnn"` at each of the six call
sites; the GGUF path acquired those call sites one at a time, and B161 is what a missed one costs.

WHY IT READS THE ENVIRONMENT DIRECTLY
    `config.config` requires ANTHROPIC_API_KEY at import time, and this module is imported by
    `training/` and by disposable CUDA workers that have no business holding an API key. The
    launcher exports SLM_QUANT_BACKEND before anything imports either module, exactly as it
    already does for SLM_CHEAP and SLM_SYNTH_API_MODE, so both readers see one value.
"""
import os

LLAMA_CPP = "llama_cpp"
MNN = "mnn"
BACKENDS = (LLAMA_CPP, MNN)

_ENV_VAR = "SLM_QUANT_BACKEND"


def resolve_backend(backend: str | None = None) -> str:
    """The backend for this run: an explicit argument, else SLM_QUANT_BACKEND, else llama.cpp.

    Raises on an unknown name rather than falling back. A typo that silently selected the default
    would report MNN-labelled results measured through llama.cpp, which is the one failure mode
    this flag must never have.
    """
    value = (backend or os.environ.get(_ENV_VAR) or LLAMA_CPP).strip().lower()
    if value not in BACKENDS:
        raise ValueError(
            f"Unknown quantization backend {value!r}. Valid values: {list(BACKENDS)}. "
            f"Set it with {_ENV_VAR} or `python tests/pipeline/run.py --quant-backend ...`."
        )
    return value


def backend_label(backend: str | None = None) -> str:
    """Human name for logs and reports."""
    return {LLAMA_CPP: "llama.cpp/GGUF", MNN: "MNN"}[resolve_backend(backend)]


def artifacts_subdir(backend: str | None = None) -> str:
    """Where this backend's artifacts are cached under `artifacts/`.

    `gguf` is unchanged so every warm cache from an existing run is still a hit; MNN gets its own
    tree because its artifacts are directories and would otherwise collide with a GGUF's name.
    """
    return {LLAMA_CPP: "gguf", MNN: "mnn"}[resolve_backend(backend)]


def artifact_name(quant: str, backend: str | None = None) -> str:
    """The exact artifact name expected for a (quant, backend) pair.

    Checked by name rather than by globbing for "any artifact in the directory": the same base
    model is selected at two tiers as two quant variants, the checkpoint path collides across
    tiers because the iteration counter resets on escalation, and globbing therefore made the
    Q8_0 tier silently reuse the Q4_K_M file (B161).
    """
    if resolve_backend(backend) == MNN:
        from training.quantize_mnn import artifact_name as mnn_artifact_name

        return mnn_artifact_name(quant)
    method = {"Q4_K_M": "q4_k_m", "Q8_0": "q8_0"}.get(quant, str(quant).lower())
    return f"model-{method}.gguf"


def quantize_from_model_spec(
    checkpoint_path: str,
    output_dir: str,
    quant: str,
    backend: str | None = None,
) -> str:
    """Build the quantized artifact for `quant` on `backend` and return its path.

    A file for llama.cpp, a directory for MNN. Either way the return value is what `run_eval`
    scores and what the on-device backends push to a phone.
    """
    if resolve_backend(backend) == MNN:
        from training.quantize_mnn import export_from_model_spec

        return export_from_model_spec(checkpoint_path, output_dir, quant)
    from training.quantize import quantize_from_model_spec as quantize_gguf

    return quantize_gguf(checkpoint_path, output_dir, quant)


def validate_and_record(
    artifact_path: str,
    base_model: str | None = None,
    quant: str | None = None,
    backend: str | None = None,
) -> dict:
    """Load the artifact in its real runtime, smoke-test it, and write the cache sidecar."""
    if resolve_backend(backend) == MNN:
        from training.quantize_mnn import validate_and_record_mnn

        return validate_and_record_mnn(artifact_path, base_model=base_model, quant=quant)
    from training.quantize import validate_and_record_gguf

    return validate_and_record_gguf(artifact_path, base_model=base_model)


def validated_cache_hit(artifact_path: str, backend: str | None = None) -> bool:
    """Whether the artifact matches a validation record written after a real load."""
    if resolve_backend(backend) == MNN:
        from training.quantize_mnn import validated_mnn_cache_hit

        return validated_mnn_cache_hit(artifact_path)
    from training.quantize import validated_gguf_cache_hit

    return validated_gguf_cache_hit(artifact_path)


def invalidate_cache(artifact_path: str, backend: str | None = None) -> None:
    """Remove a derived artifact and its validation record, and nothing else."""
    if resolve_backend(backend) == MNN:
        from training.quantize_mnn import invalidate_mnn_cache

        invalidate_mnn_cache(artifact_path)
        return
    from training.quantize import invalidate_gguf_cache

    invalidate_gguf_cache(artifact_path)


def artifact_exists(artifact_path: str, backend: str | None = None) -> bool:
    """Whether something is already there — a file for GGUF, a directory for MNN."""
    if resolve_backend(backend) == MNN:
        return os.path.isdir(artifact_path)
    return os.path.isfile(artifact_path)


def artifact_size_mb(artifact_path: str, backend: str | None = None) -> float:
    """On-disk size of the SHIPPED weights, comparably measured across both backends.

    For GGUF that is the one file. For MNN it is `llm.mnn` + `llm.mnn.weight` and deliberately not
    the whole export directory, which also holds the tokenizer and the exporter's own JSON records
    — counting those would overstate MNN against a GGUF whose tokenizer is inside the file.
    """
    if resolve_backend(backend) == MNN:
        from training.quantize_mnn import weight_size_mb

        return weight_size_mb(artifact_path)
    from training.quantize import _file_size_mb

    return _file_size_mb(artifact_path)
