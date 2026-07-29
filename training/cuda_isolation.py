"""Disposable subprocess boundary for GPU-heavy pipeline operations.

The parent process exchanges only trusted local pickle payloads/results with the worker.
Model tensors never cross this boundary; worker exit destroys its complete CUDA context.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_OBSERVABILITY_PATH_VARS = (
    "SLM_COST_EVENT_PATH",
    "SLM_TIMING_EVENT_PATH",
)


class CudaWorkerError(RuntimeError):
    """A disposable CUDA worker failed, including its remote traceback."""

    def __init__(self, operation: str, exit_code: int, response: dict | None = None):
        response = response or {}
        error_type = response.get("error_type", "WorkerProcessError")
        error = response.get("error", "worker produced no response")
        remote_traceback = response.get("traceback", "")
        message = (
            f"CUDA worker '{operation}' failed (exit={exit_code}, {error_type}: {error})"
        )
        if remote_traceback:
            message += f"\nRemote traceback:\n{remote_traceback}"
        super().__init__(message)
        self.operation = operation
        self.exit_code = exit_code
        self.remote_error_type = str(error_type)
        self.remote_traceback = remote_traceback


def isolation_enabled() -> bool:
    """True only in the parent pipeline when disposable workers are enabled."""
    return (
        os.environ.get("SLM_CUDA_ISOLATION", "0") == "1"
        and os.environ.get("SLM_CUDA_WORKER") != "1"
    )


def _read_response(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            response = pickle.load(f)
        return response if isinstance(response, dict) else None
    except Exception:
        return None


def _normalize_observability_env(env: dict[str, str]) -> dict[str, str]:
    """Export absolute shared paths, failing if pipeline-required paths vanished."""
    required = env.get("SLM_OBSERVABILITY_REQUIRED") == "1"
    for variable in _OBSERVABILITY_PATH_VARS:
        value = env.get(variable)
        if value:
            env[variable] = os.path.abspath(os.path.expanduser(value))
        elif required:
            raise RuntimeError(
                f"{variable} is required before launching a disposable CUDA worker"
            )
    return env


def run_isolated(operation: str, payload: dict):
    """Run one operation in a fresh Python process and return its pickled result."""
    operation_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="slm-cuda-") as tmp:
        request_path = os.path.join(tmp, "request.pkl")
        response_path = os.path.join(tmp, "response.pkl")
        with open(request_path, "wb") as f:
            pickle.dump({"operation": operation, "payload": payload}, f)

        env = _normalize_observability_env(dict(os.environ))
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-m", "training.cuda_worker", request_path, response_path],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    print(line, end="", flush=True)
            exit_code = proc.wait()
        except BaseException:
            # Never orphan a CUDA-owning child if the parent is interrupted while
            # streaming logs or handling cancellation.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            from agent.timing import TimingEvent, record_timing_event
            record_timing_event(TimingEvent(
                kind="worker_op",
                name=operation,
                duration_ms=(time.perf_counter() - operation_started) * 1000,
                status="error",
                metadata={"interrupted": True},
            ))
            raise
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
        response = _read_response(response_path)
        if exit_code != 0 or not response or not response.get("ok"):
            from agent.timing import TimingEvent, record_timing_event
            record_timing_event(TimingEvent(
                kind="worker_op",
                name=operation,
                duration_ms=(time.perf_counter() - operation_started) * 1000,
                status="error",
                metadata={"exit_code": exit_code},
            ))
            raise CudaWorkerError(operation, exit_code, response)
        from agent.timing import TimingEvent, record_timing_event
        record_timing_event(TimingEvent(
            kind="worker_op",
            name=operation,
            duration_ms=(time.perf_counter() - operation_started) * 1000,
            status="success",
            metadata={"exit_code": exit_code},
        ))
        return response["result"]


def merge_and_quantize(
    checkpoint_path: str,
    merged_output_dir: str,
    gguf_output_dir: str,
    quant: str,
) -> str:
    """Merge an adapter and quantize it, outside the parent process when enabled."""
    payload = {
        "checkpoint_path": checkpoint_path,
        "merged_output_dir": merged_output_dir,
        "gguf_output_dir": gguf_output_dir,
        "quant": quant,
    }
    if isolation_enabled():
        return run_isolated("merge_quantize", payload)

    from training.lora_trainer import merge_for_quantization
    from training.quantize import quantize_from_model_spec

    merged = merge_for_quantization(checkpoint_path, merged_output_dir)
    return quantize_from_model_spec(merged, gguf_output_dir, quant)
