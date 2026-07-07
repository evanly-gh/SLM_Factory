# hardware_eval/on_device_eval.py
"""
Stub for Stage 2 on-device hardware evaluation.

Phase 1: returns placeholder passing values and logs what would happen.
Phase 2 (future): ADB shell commands to deploy GGUF, run llama-bench,
parse latency/power from dumpsys output.
"""
import logging
from dataclasses import dataclass
from config.android_pool import ModelSpec, HardwareConstraints

logger = logging.getLogger(__name__)


@dataclass
class HardwareEvalResult:
    model_id: str
    success: bool
    ttft_ms: float | None        # time-to-first-token, ms
    tok_per_s: float | None      # sustained decode throughput
    avg_watts: float | None      # average power during inference
    peak_memory_mb: int | None   # measured peak RAM
    error: str | None = None


def run_on_device_eval(
    model: ModelSpec,
    constraints: HardwareConstraints,
) -> HardwareEvalResult:
    """
    Phase 1 stub: log what would run and return a synthetic pass result.

    Real implementation would:
      1. Push GGUF to device via `adb push`
      2. Run `adb shell llama-bench -m <model> -n 128 -p 64`
      3. Parse stdout for tok/s and ttft_ms
      4. Run `adb shell dumpsys thermalservice` for thermal status
      5. Parse `adb shell cat /sys/class/power_supply/battery/current_now` for watts
    """
    logger.info(
        "[hardware_eval][STUB] Would run on-device eval for %s on chip=%s "
        "(storage=%dMB, memory=%dMB, ttft_limit=%dms, power_limit=%.1fW)",
        model.model_id, constraints.target_chip,
        constraints.storage_mb, constraints.memory_mb,
        constraints.latency_ttft_ms, constraints.power_watts,
    )
    logger.info(
        "[hardware_eval][STUB] Phase 2 will deploy via ADB, run llama-bench, "
        "parse dumpsys thermalservice. Returning synthetic pass for now."
    )

    # Synthetic values derived from ModelSpec theoretical estimates.
    tok_s = model.tok_s_for_chip(constraints.target_chip)
    ttft_ms = (1.0 / max(tok_s, 0.1)) * 1000

    return HardwareEvalResult(
        model_id=model.model_id,
        success=True,
        ttft_ms=round(ttft_ms, 1),
        tok_per_s=round(tok_s, 1),
        avg_watts=None,    # Phase 2: parsed from battery current_now
        peak_memory_mb=model.peak_memory_mb,
        error=None,
    )
