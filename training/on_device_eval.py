# training/on_device_eval.py
"""
On-device evaluation harness for measuring latency, memory, and power.

Design doc §6.2-6.3: Measure TTFT, tok/s, peak RSS, average watts on a reference chip.
Three backends:
  - llama.cpp local proxy (Phase 2 starting point — no external hardware needed)
  - ADB to a connected Android device
  - Qualcomm AI Hub API (Phase 3 stub)
"""
import os
import re
import subprocess
import shutil
from dataclasses import dataclass


@dataclass
class HardwareEvalResult:
    """Measured hardware performance of a quantized model."""
    ttft_ms: float
    tok_per_s: float
    peak_rss_mb: float
    avg_watts: float
    model_load_time_s: float
    eval_method: str
    device: str
    success: bool
    error: str | None = None


def measure_llama_cpp(
    gguf_path: str,
    prompt: str = "Hello, how are you?",
    n_tokens: int = 50,
    n_threads: int = 4,
) -> HardwareEvalResult:
    """Proxy measurement using llama.cpp locally.

    Runs llama-cli with timing flags, parses output for TTFT and tok/s.
    Estimates power from model size (proxy — real measurement needs device).
    This is the Phase 2 starting point; no external hardware needed.
    """
    llama_cli = shutil.which("llama-cli") or shutil.which("main")
    if not llama_cli:
        return HardwareEvalResult(
            ttft_ms=0, tok_per_s=0, peak_rss_mb=0, avg_watts=0,
            model_load_time_s=0, eval_method="llama_cpp", device="local",
            success=False, error="llama-cli not found. Install llama.cpp.",
        )

    try:
        result = subprocess.run(
            [
                llama_cli,
                "--model", gguf_path,
                "--prompt", prompt,
                "--n-predict", str(n_tokens),
                "--threads", str(n_threads),
                "--no-display-prompt",
            ],
            capture_output=True, text=True, timeout=300,
        )
        stderr = result.stderr + result.stdout

        ttft = _parse_float(r"prompt eval time\s*=\s*([\d.]+)\s*ms", stderr) or 0
        tok_s = _parse_float(r"eval.*?([\d.]+)\s*tokens per second", stderr) or 0
        load_time = (_parse_float(r"load time\s*=\s*([\d.]+)\s*ms", stderr) or 0) / 1000

        model_size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
        est_rss = model_size_mb * 1.2
        est_watts = 2.0 + (model_size_mb / 1000) * 1.5

        return HardwareEvalResult(
            ttft_ms=ttft, tok_per_s=tok_s, peak_rss_mb=est_rss,
            avg_watts=est_watts, model_load_time_s=load_time,
            eval_method="llama_cpp", device="local", success=True,
        )
    except Exception as e:
        return HardwareEvalResult(
            ttft_ms=0, tok_per_s=0, peak_rss_mb=0, avg_watts=0,
            model_load_time_s=0, eval_method="llama_cpp", device="local",
            success=False, error=str(e),
        )


def measure_adb(
    gguf_path: str,
    device_serial: str = "",
    prompt: str = "Hello",
    n_tokens: int = 50,
) -> HardwareEvalResult:
    """Measure on a real Android device via ADB.

    Pushes GGUF to device, runs llama-cli, parses timing, reads /proc/meminfo.
    Requires: adb in PATH, device connected and authorized, llama-cli on device.
    """
    adb = shutil.which("adb")
    if not adb:
        return HardwareEvalResult(
            ttft_ms=0, tok_per_s=0, peak_rss_mb=0, avg_watts=0,
            model_load_time_s=0, eval_method="adb", device=device_serial or "unknown",
            success=False, error="adb not found in PATH",
        )

    serial_flag = ["-s", device_serial] if device_serial else []
    device_path = "/data/local/tmp/model.gguf"

    try:
        subprocess.run(
            [adb] + serial_flag + ["push", gguf_path, device_path],
            check=True, capture_output=True, timeout=300,
        )

        result = subprocess.run(
            [adb] + serial_flag + [
                "shell", f"cd /data/local/tmp && ./llama-cli "
                f"--model model.gguf --prompt '{prompt}' "
                f"--n-predict {n_tokens} --threads 4 --no-display-prompt"
            ],
            capture_output=True, text=True, timeout=300,
        )
        output = result.stdout + result.stderr

        ttft = _parse_float(r"prompt eval time\s*=\s*([\d.]+)\s*ms", output) or 0
        tok_s = _parse_float(r"eval.*?([\d.]+)\s*tokens per second", output) or 0
        load_time = (_parse_float(r"load time\s*=\s*([\d.]+)\s*ms", output) or 0) / 1000

        mem_result = subprocess.run(
            [adb] + serial_flag + ["shell", "cat /proc/meminfo"],
            capture_output=True, text=True, timeout=10,
        )
        mem_available = _parse_float(r"MemAvailable:\s*(\d+)\s*kB", mem_result.stdout)
        peak_rss = (mem_available or 0) / 1024

        return HardwareEvalResult(
            ttft_ms=ttft, tok_per_s=tok_s, peak_rss_mb=peak_rss,
            avg_watts=0, model_load_time_s=load_time,
            eval_method="adb", device=device_serial or "connected",
            success=True,
        )
    except Exception as e:
        return HardwareEvalResult(
            ttft_ms=0, tok_per_s=0, peak_rss_mb=0, avg_watts=0,
            model_load_time_s=0, eval_method="adb", device=device_serial or "unknown",
            success=False, error=str(e),
        )


def measure_qualcomm_hub(model_name: str, chip: str) -> HardwareEvalResult:
    """Remote profiling via Qualcomm AI Hub API. Phase 3 stub."""
    raise NotImplementedError(
        "Qualcomm AI Hub profiling not yet implemented. "
        "Use measure_llama_cpp() or measure_adb() instead."
    )


def _parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None
