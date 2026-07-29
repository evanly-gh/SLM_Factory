"""One-shot worker for GPU-heavy operations.

Invoked only through ``training.cuda_isolation.run_isolated``.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import traceback


def _write_response(path: str, response: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(response, f)
    os.replace(tmp, path)


def _cuda_stats() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "device": torch.cuda.current_device(),
            "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
            "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        }
    except Exception as exc:
        return {"available": False, "telemetry_error": str(exc)[:120]}


def _dispatch(operation: str, payload: dict):
    if operation == "ping":
        return {
            "pid": os.getpid(),
            "cost_event_path": os.environ.get("SLM_COST_EVENT_PATH"),
            "timing_event_path": os.environ.get("SLM_TIMING_EVENT_PATH"),
            **payload,
        }
    if operation == "cuda_probe":
        import torch

        mib = int(payload.get("mib", 256))
        allocation = torch.empty(mib * 2**20, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        return {"pid": os.getpid(), "mib": mib, "stats": _cuda_stats()}
    if operation == "train":
        from training.slm_helpers import train

        return train(**payload)
    if operation == "infer":
        from training.slm_helpers import infer

        return infer(**payload)
    if operation == "infer_batch":
        from training.slm_helpers import infer_batch

        return infer_batch(**payload)
    if operation == "eval":
        from eval.harness import run_eval

        return run_eval(**payload)
    if operation == "build_gguf":
        from agent.nodes.evaluate import _build_or_reuse_gguf

        return _build_or_reuse_gguf(**payload)
    if operation == "merge_quantize":
        from training.cuda_isolation import merge_and_quantize

        return merge_and_quantize(**payload)
    raise ValueError(f"Unsupported CUDA worker operation: {operation!r}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m training.cuda_worker REQUEST.pkl RESPONSE.pkl")
        return 2
    request_path, response_path = argv
    os.environ["SLM_CUDA_WORKER"] = "1"
    # Spawned workers import providers independently; configure their inherited
    # append-only destinations before dispatch can create a judge/API client.
    from agent.cost import install_cost_tracking
    from agent.timing import (
        TimingEvent,
        install_timing_tracking,
        record_timing_event,
    )

    install_cost_tracking()
    install_timing_tracking()

    with open(request_path, "rb") as f:
        request = pickle.load(f)
    operation = request["operation"]
    payload = request.get("payload") or {}
    print(
        f"[cuda-worker] start op={operation} pid={os.getpid()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    operation_started = time.perf_counter()
    try:
        result = _dispatch(operation, payload)
        stats = {"cuda_telemetry": "skipped"} if operation == "ping" else _cuda_stats()
        record_timing_event(TimingEvent(
            kind="worker_dispatch",
            name=operation,
            duration_ms=(time.perf_counter() - operation_started) * 1000,
            status="success",
            metadata={"pid": os.getpid()},
        ))
        print(f"[cuda-worker] finish op={operation} stats={stats}", flush=True)
        _write_response(response_path, {"ok": True, "result": result})
        return 0
    except BaseException as exc:  # noqa: BLE001 - transport every worker failure to parent
        remote_traceback = traceback.format_exc()
        stats = _cuda_stats()
        record_timing_event(TimingEvent(
            kind="worker_dispatch",
            name=operation,
            duration_ms=(time.perf_counter() - operation_started) * 1000,
            status="error",
            metadata={
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            },
        ))
        print(
            f"[cuda-worker] failed op={operation}: {type(exc).__name__}: {exc} stats={stats}",
            flush=True,
        )
        _write_response(
            response_path,
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": remote_traceback,
                "cuda_stats": stats,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
