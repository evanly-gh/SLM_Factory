#!/usr/bin/env python3
"""Offline real-SQLite kill/reopen/resume smoke.

The train/eval nodes use fresh lightweight Python workers. They deliberately do
no model import or download; their purpose is to exercise disposable-process,
ledger, and checkpoint behavior around a real SIGKILL.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import END, StateGraph

from agent.checkpoint import (
    cumulative_wall_time_from_sqlite,
    inspect_sqlite_state,
    load_checkpoint,
    resume_input_from_sqlite,
    sqlite_checkpointer,
    stream_with_checkpoints,
)
from agent.cost import CostEvent, CostLedger, record_cost_event
from agent.timing import TimingEvent, TimingLedger, record_timing_event


INITIAL_WALL_TIME_S = 7.0
THREAD_ID = "checkpoint-kill-resume-smoke"
COMPATIBILITY = {
    "mode": "smoke",
    "pool_fingerprint": "offline-no-model",
    "topology_fingerprint": "prepare-train-eval-v1",
    "config_fingerprint": "deterministic-v1",
}


class SmokeState(TypedDict):
    value: int
    ledger: list[str]
    _graph_steps: int


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        try:
            output.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            output.flush()
            os.fsync(output.fileno())
        finally:
            fcntl.flock(output.fileno(), fcntl.LOCK_UN)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_execution(run_dir: Path, node: str, *, worker: bool) -> None:
    _append_jsonl(
        run_dir / "executions.jsonl",
        {
            "node": node,
            "pid": os.getpid(),
            "worker": worker,
        },
    )


def _worker(name: str, run_dir: Path) -> int:
    started = time.perf_counter()
    time.sleep(0.03)
    _record_execution(run_dir, name, worker=True)
    record_cost_event(
        CostEvent(
            provider="local",
            model="mock-no-model",
            stage=name,
            pricing_status="not_applicable",
            estimated_usd=0.0,
            metadata={"worker": "lightweight", "model_download": False},
        ),
        path=run_dir / "cost-events.jsonl",
    )
    record_timing_event(
        TimingEvent(
            kind="worker_dispatch",
            name=name,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata={"worker": "lightweight", "model_download": False},
        ),
        path=run_dir / "timing-events.jsonl",
    )
    return 0


def _run_disposable_worker(run_dir: Path, name: str) -> None:
    environment = dict(os.environ)
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(PROJECT_ROOT), current_pythonpath)
        if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
            "--run-dir",
            str(run_dir),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        # A fresh interpreter may spend several seconds importing the project
        # when the full test suite saturates shared storage.
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"lightweight {name} worker failed with exit "
            f"{completed.returncode}: {completed.stderr}"
        )


def _next_state(state: SmokeState, name: str) -> SmokeState:
    return {
        "value": state["value"] + 1,
        "ledger": [*state["ledger"], name],
        "_graph_steps": state["_graph_steps"] + 1,
    }


def _build_graph(run_dir: Path, saver, *, pause_before_eval: bool = False):
    def prepare(state: SmokeState):
        _record_execution(run_dir, "prepare", worker=False)
        return _next_state(state, "prepare")

    def train(state: SmokeState):
        _run_disposable_worker(run_dir, "train")
        return _next_state(state, "train")

    def evaluate(state: SmokeState):
        if pause_before_eval:
            # LangGraph may schedule the next task while the synchronous SQLite
            # saver callback is still observing the prior commit. Keep the
            # pending node side-effect-free until SIGKILL lands.
            time.sleep(5)
        _run_disposable_worker(run_dir, "eval")
        return _next_state(state, "eval")

    builder = StateGraph(SmokeState)
    builder.add_node("prepare", prepare)
    builder.add_node("train", train)
    builder.add_node("eval", evaluate)
    builder.set_entry_point("prepare")
    builder.add_edge("prepare", "train")
    builder.add_edge("train", "eval")
    builder.add_edge("eval", END)
    return builder.compile(checkpointer=saver)


def _config() -> dict:
    return {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": 8,
    }


def _run_phase(run_dir: Path, phase: str) -> int:
    sqlite_path = run_dir / "langgraph.sqlite"
    checkpoint_path = run_dir / "checkpoint.json"
    with sqlite_checkpointer(sqlite_path) as saver:
        graph = _build_graph(
            run_dir,
            saver,
            pause_before_eval=phase == "interrupt",
        )
        config = _config()
        if phase == "interrupt":
            input_state: SmokeState | None = {
                "value": 0,
                "ledger": [],
                "_graph_steps": 0,
            }
            initial_steps = 0
            initial_wall = INITIAL_WALL_TIME_S

            def kill_after_train(snapshot):
                if snapshot.step == 2 and snapshot.last_node == "train":
                    os.kill(os.getpid(), signal.SIGKILL)

            after_commit = kill_after_train
        else:
            stale = load_checkpoint(
                checkpoint_path,
                expected_compatibility=COMPATIBILITY,
            )
            authoritative = inspect_sqlite_state(graph, config)
            if (
                authoritative.step != 2
                or authoritative.next_nodes != ("eval",)
                or authoritative.state.get("ledger") != ["prepare", "train"]
            ):
                raise RuntimeError(
                    "SQLite did not preserve the committed train boundary: "
                    f"{authoritative}"
                )
            input_state = resume_input_from_sqlite(
                authoritative,
                {"value": 99, "ledger": ["wrong"], "_graph_steps": 99},
            )
            initial_steps = authoritative.step
            initial_wall = cumulative_wall_time_from_sqlite(
                stale,
                authoritative,
            )
            after_commit = None

        stream_with_checkpoints(
            graph,
            input_state,
            config=config,
            checkpoint_path=checkpoint_path,
            thread_id=THREAD_ID,
            compatibility=COMPATIBILITY,
            initial_graph_steps=initial_steps,
            initial_wall_time_s=initial_wall,
            after_sqlite_commit=after_commit,
        )
    return 0


def _subprocess_environment(run_dir: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                value
                for value in (
                    str(PROJECT_ROOT),
                    environment.get("PYTHONPATH", ""),
                )
                if value
            ),
            "PYTHONHASHSEED": "0",
            "SLM_COST_EVENT_PATH": str(
                (run_dir / "cost-events.jsonl").resolve()
            ),
            "SLM_TIMING_EVENT_PATH": str(
                (run_dir / "timing-events.jsonl").resolve()
            ),
        }
    )
    return environment


def _launch_phase(run_dir: Path, phase: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--run-dir",
            str(run_dir),
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(run_dir),
        text=True,
        capture_output=True,
        # Importing LangGraph/SQLite can be slow under a saturated full-suite run.
        timeout=30,
        check=False,
    )


def run_smoke(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    interrupted = _launch_phase(run_dir, "interrupt")
    if interrupted.returncode != -signal.SIGKILL:
        raise RuntimeError(
            "interrupt phase did not terminate at the committed train boundary: "
            f"returncode={interrupted.returncode}, stderr={interrupted.stderr!r}"
        )
    executions_before_resume = _read_jsonl(run_dir / "executions.jsonl")
    if [entry["node"] for entry in executions_before_resume] != [
        "prepare",
        "train",
    ]:
        raise RuntimeError(
            f"unexpected pre-resume executions: {executions_before_resume}"
        )

    with sqlite_checkpointer(run_dir / "langgraph.sqlite") as saver:
        graph = _build_graph(run_dir, saver)
        before_resume = inspect_sqlite_state(graph, _config())
    if before_resume.step != 2 or before_resume.next_nodes != ("eval",):
        raise RuntimeError(
            f"unexpected SQLite interruption boundary: {before_resume}"
        )

    resumed = _launch_phase(run_dir, "resume")
    if resumed.returncode != 0:
        raise RuntimeError(
            f"resume phase failed: stdout={resumed.stdout!r}, "
            f"stderr={resumed.stderr!r}"
        )

    with sqlite_checkpointer(run_dir / "langgraph.sqlite") as saver:
        graph = _build_graph(run_dir, saver)
        final_sqlite = inspect_sqlite_state(graph, _config())
    final_json = load_checkpoint(
        run_dir / "checkpoint.json",
        expected_compatibility=COMPATIBILITY,
    )
    executions = _read_jsonl(run_dir / "executions.jsonl")
    executed_nodes = [entry["node"] for entry in executions]
    resumed_nodes = [
        entry["node"]
        for entry in executions[len(executions_before_resume):]
    ]
    cost_stages = [
        event["stage"]
        for event in CostLedger(run_dir / "cost-events.jsonl").events()
    ]
    timing_names = [
        event["name"]
        for event in TimingLedger(run_dir / "timing-events.jsonl").events()
    ]
    cumulative_wall = float(
        final_json["progress"]["cumulative_wall_time_s"]
    )
    summary = {
        "interrupted_returncode": interrupted.returncode,
        "sqlite_step_before_resume": before_resume.step,
        "final_sqlite_step": final_sqlite.step,
        "executed_nodes": executed_nodes,
        "resumed_nodes": resumed_nodes,
        "state_ledger": final_sqlite.state.get("ledger"),
        "cost_stages": cost_stages,
        "timing_names": timing_names,
        "cumulative_wall_time_s": cumulative_wall,
    }
    expected = {
        "executed_nodes": ["prepare", "train", "eval"],
        "resumed_nodes": ["eval"],
        "state_ledger": ["prepare", "train", "eval"],
        "cost_stages": ["train", "eval"],
        "timing_names": ["train", "eval"],
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise RuntimeError(
                f"resume invariant failed for {key}: "
                f"{summary[key]!r} != {value!r}"
            )
    if (
        final_sqlite.step != 3
        or final_sqlite.next_nodes
        or final_json["progress"]["status"] != "completed"
        or int(final_sqlite.state.get("_graph_steps", -1)) != 3
        or cumulative_wall < INITIAL_WALL_TIME_S
    ):
        raise RuntimeError(
            f"final checkpoint state is incomplete: {summary}, "
            f"next={final_sqlite.next_nodes}, "
            f"status={final_json['progress']['status']!r}"
        )
    print(
        "PASS checkpoint kill/resume smoke: "
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise real SQLite process-kill resume behavior offline."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("interrupt", "resume"))
    parser.add_argument("--worker", choices=("train", "eval"))
    args = parser.parse_args()
    if args.worker:
        return _worker(args.worker, args.run_dir.resolve())
    if args.phase:
        return _run_phase(args.run_dir.resolve(), args.phase)
    run_smoke(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
