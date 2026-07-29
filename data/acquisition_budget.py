"""Crash-safe paid-acquisition reservation ledger."""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import uuid
from pathlib import Path
from typing import Any


ACQUISITION_LEDGER_FILENAME = "acquisition-reservations.jsonl"
ACQUISITION_LEDGER_SCHEMA_VERSION = 1


class AcquisitionBudgetError(RuntimeError):
    """The durable paid-acquisition budget cannot be trusted."""


def _utc_now() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def acquisition_ledger_path(
    path: str | os.PathLike | None = None,
) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    run_dir = os.environ.get("SLM_RUN_DIR", "").strip()
    if not run_dir:
        raise AcquisitionBudgetError(
            "SLM_RUN_DIR is required before reserving paid acquisition"
        )
    return Path(run_dir).expanduser().resolve() / ACQUISITION_LEDGER_FILENAME


def _read_events(handle) -> list[dict[str, Any]]:
    handle.seek(0)
    events = []
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AcquisitionBudgetError(
                f"acquisition ledger line {line_number} is corrupt"
            ) from exc
        if (
            not isinstance(event, dict)
            or event.get("schema_version")
            != ACQUISITION_LEDGER_SCHEMA_VERSION
            or event.get("event") not in {"reserved", "completed", "failed"}
            or not isinstance(event.get("reservation_id"), str)
            or not isinstance(event.get("plan_identity"), str)
        ):
            raise AcquisitionBudgetError(
                f"acquisition ledger line {line_number} has invalid schema"
            )
        events.append(event)
    return events


def _snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    reservations: dict[str, dict] = {}
    terminal: dict[str, str] = {}
    for event in events:
        reservation_id = event["reservation_id"]
        if event["event"] == "reserved":
            reservations.setdefault(reservation_id, event)
        else:
            terminal.setdefault(reservation_id, event["event"])
    plans: dict[str, dict[str, int]] = {}
    for reservation_id, event in reservations.items():
        plan = event["plan_identity"]
        summary = plans.setdefault(
            plan,
            {"spent": 0, "completed": 0, "failed": 0, "pending": 0},
        )
        summary["spent"] += 1
        status = terminal.get(reservation_id)
        if status == "completed":
            summary["completed"] += 1
        elif status == "failed":
            summary["failed"] += 1
        else:
            summary["pending"] += 1
    return {
        "run_spent": len(reservations),
        "plans": plans,
        "reservations": reservations,
        "terminal": terminal,
    }


def _append_locked(handle, event: dict[str, Any]) -> None:
    handle.seek(0, os.SEEK_END)
    handle.write(
        json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def acquisition_budget_snapshot(
    path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    ledger = acquisition_ledger_path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            return _snapshot(_read_events(handle))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def reserve_paid_acquisition(
    *,
    plan_identity: str,
    per_plan_limit: int,
    run_limit: int,
    path: str | os.PathLike | None = None,
) -> dict[str, Any] | None:
    """Atomically spend one round before any paid provider call begins."""
    if not isinstance(plan_identity, str) or not plan_identity.strip():
        raise ValueError("plan_identity must be non-empty")
    per_plan_limit = max(0, int(per_plan_limit))
    run_limit = max(0, int(run_limit))
    ledger = acquisition_ledger_path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            snapshot = _snapshot(_read_events(handle))
            plan_spent = (
                snapshot["plans"]
                .get(plan_identity, {})
                .get("spent", 0)
            )
            if (
                snapshot["run_spent"] >= run_limit
                or plan_spent >= per_plan_limit
            ):
                return None
            reservation = {
                "schema_version": ACQUISITION_LEDGER_SCHEMA_VERSION,
                "event": "reserved",
                "reservation_id": uuid.uuid4().hex,
                "plan_identity": plan_identity,
                "plan_round": plan_spent,
                "run_round": snapshot["run_spent"],
                "timestamp": _utc_now(),
                "pid": os.getpid(),
            }
            _append_locked(handle, reservation)
            return reservation
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def reconcile_paid_acquisition(
    reservation: dict[str, Any],
    *,
    status: str,
    detail: str = "",
    path: str | os.PathLike | None = None,
) -> None:
    """Append one completion/failure outcome without refunding its reservation."""
    if status not in {"completed", "failed"}:
        raise ValueError("status must be completed or failed")
    ledger = acquisition_ledger_path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            snapshot = _snapshot(_read_events(handle))
            reservation_id = reservation.get("reservation_id")
            stored = snapshot["reservations"].get(reservation_id)
            if stored is None:
                raise AcquisitionBudgetError(
                    "cannot reconcile an unknown acquisition reservation"
                )
            if reservation_id in snapshot["terminal"]:
                return
            _append_locked(handle, {
                "schema_version": ACQUISITION_LEDGER_SCHEMA_VERSION,
                "event": status,
                "reservation_id": reservation_id,
                "plan_identity": stored["plan_identity"],
                "timestamp": _utc_now(),
                "pid": os.getpid(),
                "detail": str(detail)[:500],
            })
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
