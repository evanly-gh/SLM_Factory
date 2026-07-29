"""Final cost/timing artifact assembly shared by normal and early exits."""
from __future__ import annotations

import datetime as _datetime
import json
import os
from pathlib import Path


def _atomic_json(path: Path, payload: dict) -> None:
    """Replace one JSON artifact atomically without exposing partial content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _api_timing_summary(cost: dict) -> dict:
    by_provider = {
        provider: {
            "calls": summary["calls"],
            "failures": summary["failures"],
            "latency_ms": summary["latency_ms"],
        }
        for provider, summary in cost.get("by_provider", {}).items()
    }
    by_stage = {
        stage: {
            "calls": summary["calls"],
            "failures": summary["failures"],
            "latency_ms": summary["latency_ms"],
        }
        for stage, summary in cost.get("by_stage", {}).items()
    }
    anthropic_by_stage: dict[str, dict] = {}
    for summary in cost.get("by_provider_model_stage", []):
        if summary.get("provider") != "anthropic":
            continue
        stage = str(summary.get("stage") or "unknown")
        target = anthropic_by_stage.setdefault(
            stage, {"calls": 0, "failures": 0, "latency_ms": 0.0}
        )
        target["calls"] += int(summary.get("calls") or 0)
        target["failures"] += int(summary.get("failures") or 0)
        target["latency_ms"] += float(summary.get("latency_ms") or 0.0)
    for target in anthropic_by_stage.values():
        target["latency_ms"] = round(target["latency_ms"], 3)

    anthropic = cost.get("by_provider", {}).get("anthropic", {})
    return {
        "total_latency_ms": cost.get("total_latency_ms", 0.0),
        "anthropic_api_latency_ms": anthropic.get("latency_ms", 0.0),
        "anthropic_by_stage": dict(sorted(anthropic_by_stage.items())),
        "by_provider": by_provider,
        "by_stage": by_stage,
    }


def write_observability_artifacts(
    run_dir: str | os.PathLike,
    cost_ledger,
    timing_ledger,
    *,
    run_status: str,
    exit_code: int | None = None,
    reason: str | None = None,
) -> tuple[dict, dict]:
    """Write ``cost.json`` and ``timings.json`` for every termination path."""
    destination = Path(run_dir).expanduser().resolve()
    cost = cost_ledger.snapshot()
    timings = timing_ledger.snapshot()
    timings["api_calls"] = _api_timing_summary(cost)
    timings["run_exit"] = {
        "status": run_status,
        "exit_code": exit_code,
        "reason": reason,
        "recorded_at": (
            _datetime.datetime.now(_datetime.timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        ),
    }
    _atomic_json(destination / "cost.json", cost)
    _atomic_json(destination / "timings.json", timings)
    return cost, timings
