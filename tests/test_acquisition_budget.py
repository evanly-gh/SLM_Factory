import json
import multiprocessing
import os

from agent.data_rebuild import remaining_paid_acquire_rounds
from data.acquisition_budget import (
    ACQUISITION_LEDGER_FILENAME,
    acquisition_budget_snapshot,
    reconcile_paid_acquisition,
    reserve_paid_acquisition,
)


def _reserve_worker(run_dir, queue):
    os.environ["SLM_RUN_DIR"] = run_dir
    reservation = reserve_paid_acquisition(
        plan_identity="parallel-plan",
        per_plan_limit=5,
        run_limit=5,
    )
    queue.put(reservation is not None)


def test_reservation_ledger_is_append_only_and_counts_failure_as_spent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SLM_RUN_DIR", str(tmp_path))

    reservation = reserve_paid_acquisition(
        plan_identity="plan-a",
        per_plan_limit=2,
        run_limit=9,
    )
    reconcile_paid_acquisition(
        reservation,
        status="failed",
        detail="provider timeout",
    )

    ledger = tmp_path / ACQUISITION_LEDGER_FILENAME
    events = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "reserved",
        "failed",
    ]
    snapshot = acquisition_budget_snapshot()
    assert snapshot["run_spent"] == 1
    assert snapshot["plans"]["plan-a"]["spent"] == 1
    assert snapshot["plans"]["plan-a"]["failed"] == 1


def test_reservations_enforce_plan_and_run_caps_across_resume_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SLM_RUN_DIR", str(tmp_path))
    first = reserve_paid_acquisition(
        plan_identity="plan-a",
        per_plan_limit=1,
        run_limit=2,
    )
    assert first is not None
    assert reserve_paid_acquisition(
        plan_identity="plan-a",
        per_plan_limit=1,
        run_limit=2,
    ) is None
    second = reserve_paid_acquisition(
        plan_identity="plan-b",
        per_plan_limit=1,
        run_limit=2,
    )
    assert second is not None
    assert reserve_paid_acquisition(
        plan_identity="plan-c",
        per_plan_limit=1,
        run_limit=2,
    ) is None

    # JSON checkpoint state may lag a reservation if the process died. The
    # append-only ledger remains authoritative on resume.
    assert remaining_paid_acquire_rounds(
        {"source_acquire_rounds_used": 0}
    ) == 7


def test_reservation_cap_is_process_safe(tmp_path):
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    workers = [
        context.Process(
            target=_reserve_worker,
            args=(str(tmp_path), queue),
        )
        for _ in range(8)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0

    assert sum(queue.get(timeout=2) for _ in workers) == 5
    snapshot = acquisition_budget_snapshot(
        tmp_path / ACQUISITION_LEDGER_FILENAME
    )
    assert snapshot["run_spent"] == 5
    assert snapshot["plans"]["parallel-plan"]["spent"] == 5
