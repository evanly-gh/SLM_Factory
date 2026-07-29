"""Process-safe timing events for graph nodes and disposable workers."""
from __future__ import annotations

import datetime as _datetime
import fcntl
import functools
import inspect
import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


TIMING_EVENT_PATH_ENV = "SLM_TIMING_EVENT_PATH"
OBSERVABILITY_REQUIRED_ENV = "SLM_OBSERVABILITY_REQUIRED"
_append_lock = threading.Lock()
_fallback_lock = threading.Lock()
_fallback_path: str | None = None


class MissingTimingPathError(RuntimeError):
    """Pipeline instrumentation requires a shared timing event path."""


def _timestamp() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass
class TimingEvent:
    kind: str
    name: str
    duration_ms: float
    status: str = "success"
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=_timestamp)
    pid: int = field(default_factory=os.getpid)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["duration_ms"] = round(float(payload["duration_ms"]), 3)
        return payload


def _absolute_path(path: str | os.PathLike) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _resolve_path(
    path: str | os.PathLike | None = None,
    *,
    required: bool | None = None,
) -> str:
    if path is not None:
        return _absolute_path(path)
    configured = os.environ.get(TIMING_EVENT_PATH_ENV)
    if configured:
        return _absolute_path(configured)
    if required is None:
        required = os.environ.get(OBSERVABILITY_REQUIRED_ENV) == "1"
    if required:
        raise MissingTimingPathError(
            f"{TIMING_EVENT_PATH_ENV} is required for pipeline instrumentation"
        )
    global _fallback_path
    with _fallback_lock:
        if _fallback_path is None:
            _fallback_path = os.path.join(
                tempfile.gettempdir(),
                f"slm-timing-{os.getpid()}-{uuid.uuid4().hex}.jsonl",
            )
        os.environ.setdefault(TIMING_EVENT_PATH_ENV, _fallback_path)
        return _fallback_path


def install_timing_tracking(
    event_path: str | os.PathLike | None = None,
    *,
    required: bool | None = None,
) -> "TimingLedger":
    if required is not None:
        os.environ[OBSERVABILITY_REQUIRED_ENV] = "1" if required else "0"
    resolved = _resolve_path(event_path, required=required)
    os.environ[TIMING_EVENT_PATH_ENV] = resolved
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    Path(resolved).touch(exist_ok=True)
    return TIMINGS


def record_timing_event(
    event: TimingEvent, path: str | os.PathLike | None = None
) -> TimingEvent:
    destination = _resolve_path(path)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"))
    with _append_lock:
        with open(destination, "a", encoding="utf-8") as output:
            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
            try:
                output.write(line + "\n")
                output.flush()
            finally:
                fcntl.flock(output.fileno(), fcntl.LOCK_UN)
    return event


@contextmanager
def timed(
    kind: str,
    name: str,
    *,
    metadata: dict | None = None,
    path: str | os.PathLike | None = None,
):
    """Record elapsed time and failure status around a block."""
    import time

    started = time.perf_counter()
    status = "success"
    details = dict(metadata or {})
    try:
        yield
    except BaseException as exc:
        status = "error"
        details.update(
            {"error_type": type(exc).__name__, "error": str(exc)[:500]}
        )
        raise
    finally:
        record_timing_event(
            TimingEvent(
                kind=kind,
                name=name,
                duration_ms=(time.perf_counter() - started) * 1000,
                status=status,
                metadata=details,
            ),
            path=path,
        )


def instrument_node(
    name: str,
    node,
    *,
    path: str | os.PathLike | None = None,
):
    """Wrap a graph node so success and failure carry its exact name."""
    if inspect.iscoroutinefunction(node):
        @functools.wraps(node)
        async def async_wrapped(*args, **kwargs):
            with timed("graph_node", name, path=path):
                return await node(*args, **kwargs)

        return async_wrapped

    @functools.wraps(node)
    def wrapped(*args, **kwargs):
        with timed("graph_node", name, path=path):
            return node(*args, **kwargs)

    return wrapped


def _empty_summary() -> dict:
    return {
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "duration_ms": 0.0,
        "max_duration_ms": 0.0,
    }


def _add(summary: dict, event: dict) -> None:
    duration = float(event.get("duration_ms") or 0.0)
    summary["calls"] += 1
    if event.get("status") == "success":
        summary["successes"] += 1
    else:
        summary["failures"] += 1
    summary["duration_ms"] += duration
    summary["max_duration_ms"] = max(summary["max_duration_ms"], duration)


def _round(summary: dict) -> dict:
    result = dict(summary)
    result["duration_ms"] = round(result["duration_ms"], 3)
    result["max_duration_ms"] = round(result["max_duration_ms"], 3)
    result["avg_duration_ms"] = round(
        result["duration_ms"] / result["calls"] if result["calls"] else 0.0,
        3,
    )
    return result


class TimingLedger:
    def __init__(self, event_path: str | os.PathLike | None = None):
        self._event_path = (
            _absolute_path(event_path) if event_path is not None else None
        )

    @property
    def event_path(self) -> str:
        return _resolve_path(self._event_path)

    def append(self, event: TimingEvent) -> TimingEvent:
        return record_timing_event(event, path=self.event_path)

    def events(self) -> list[dict]:
        if not os.path.exists(self.event_path):
            return []
        events = []
        with open(self.event_path, encoding="utf-8") as source:
            fcntl.flock(source.fileno(), fcntl.LOCK_SH)
            try:
                for line in source:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        events.append(value)
            finally:
                fcntl.flock(source.fileno(), fcntl.LOCK_UN)
        return events

    def snapshot(self) -> dict:
        events = self.events()
        total = _empty_summary()
        event_starts_ms = []
        event_ends_ms = []
        by_kind: dict[str, dict] = {}
        by_name: dict[str, dict] = {}
        by_kind_and_name: dict[str, dict[str, dict]] = {}
        for event in events:
            kind = str(event.get("kind") or "unknown")
            name = str(event.get("name") or "unknown")
            try:
                timestamp = str(event.get("timestamp") or "").replace(
                    "Z", "+00:00"
                )
                end_ms = _datetime.datetime.fromisoformat(timestamp).timestamp() * 1000
                event_ends_ms.append(end_ms)
                event_starts_ms.append(
                    end_ms - float(event.get("duration_ms") or 0.0)
                )
            except (TypeError, ValueError):
                pass
            _add(total, event)
            _add(by_kind.setdefault(kind, _empty_summary()), event)
            _add(by_name.setdefault(name, _empty_summary()), event)
            _add(
                by_kind_and_name.setdefault(kind, {}).setdefault(
                    name, _empty_summary()
                ),
                event,
            )
        split_summaries = {
            kind: {
                name: _round(summary)
                for name, summary in sorted(names.items())
            }
            for kind, names in sorted(by_kind_and_name.items())
        }
        worker_overhead = {}
        for name, parent in by_kind_and_name.get("worker_op", {}).items():
            child = by_kind_and_name.get("worker_dispatch", {}).get(name)
            if child is None:
                continue
            duration = max(0.0, parent["duration_ms"] - child["duration_ms"])
            calls = min(parent["calls"], child["calls"])
            worker_overhead[name] = {
                "calls": calls,
                "duration_ms": round(duration, 3),
                "avg_duration_ms": round(duration / calls if calls else 0.0, 3),
            }
        return {
            "schema_version": 1,
            "event_path": self.event_path,
            "total_events": total["calls"],
            # Durations are nested (run → graph node → worker), so their sum is
            # useful workload accounting but is not elapsed wall-clock time.
            "summed_event_duration_ms": round(total["duration_ms"], 3),
            "wall_clock_span_ms": round(
                max(event_ends_ms) - min(event_starts_ms)
                if event_ends_ms and event_starts_ms
                else 0.0,
                3,
            ),
            "by_kind": {
                key: _round(value) for key, value in sorted(by_kind.items())
            },
            "by_name": {
                key: _round(value) for key, value in sorted(by_name.items())
            },
            "by_kind_and_name": split_summaries,
            "graph_nodes": split_summaries.get("graph_node", {}),
            "worker_ops": split_summaries.get("worker_op", {}),
            "worker_dispatch": split_summaries.get("worker_dispatch", {}),
            # Parent worker_op includes process startup/import/model-load/transport;
            # child worker_dispatch starts immediately before dispatch. Their
            # difference is the measured disposable-worker overhead.
            "worker_overhead": worker_overhead,
        }


TIMINGS = TimingLedger()
