# training/on_device_eval.py
"""
DEPRECATED — moved to hardware_eval/on_device_eval.py.

On-device hardware measurement was consolidated into a single module,
hardware_eval/on_device_eval.py, which now owns the theoretical / llama_cpp /
adb_llama / smolchat backends and the canonical HardwareEvalResult type.

This shim re-exports the new API and maps the old function names so any external
caller keeps working. Prefer importing from hardware_eval.on_device_eval directly.

Old → new mapping:
    measure_llama_cpp(gguf_path, ...)     -> hardware_eval measure_llama_cpp(gguf_path, model, constraints, ...)
    measure_adb(gguf_path, serial, ...)   -> hardware_eval measure_adb_llama(gguf_path, model, constraints, serial=...)
    measure_qualcomm_hub(...)             -> still NotImplemented
"""
from hardware_eval.on_device_eval import (  # noqa: F401
    HardwareEvalResult,
    measure_llama_cpp,
    measure_adb_llama,
    run_on_device_eval,
)


def measure_qualcomm_hub(model_name: str, chip: str):
    """Remote profiling via Qualcomm AI Hub API. Not implemented."""
    raise NotImplementedError(
        "Qualcomm AI Hub profiling not implemented. "
        "Use hardware_eval.on_device_eval.run_on_device_eval with backend "
        "'llama_cpp', 'adb_llama', or 'smolchat'."
    )
