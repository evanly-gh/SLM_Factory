import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.graph import END, StateGraph

os.environ.setdefault("ANTHROPIC_API_KEY", "test-no-network")
os.environ.setdefault("EXA_API_KEY", "test-no-network")

from agent.checkpoint import (
    checkpoint_compatibility,
    cumulative_wall_time_from_sqlite,
    RecursionBudgetExhausted,
    inspect_sqlite_state,
    load_checkpoint,
    reconcile_checkpoint_from_sqlite,
    remaining_recursion_limit,
    resume_input_from_sqlite,
    save_checkpoint,
    sqlite_checkpointer,
    stream_with_checkpoints,
)
from agent.graph import guard_graph_node


COMPATIBILITY = checkpoint_compatibility(
    mode="cold_start",
    pool_fingerprint="pool",
    topology_fingerprint="topology",
    config_fingerprint="config",
)


class CounterState(TypedDict):
    count: int
    target: int
    _graph_steps: int


def _counter_graph(checkpointer, *, guarded=False):
    def count(state):
        return {"count": state["count"] + 1}

    def route(state):
        return END if state["count"] >= state["target"] else "count"

    builder = StateGraph(CounterState)
    builder.add_node(
        "count",
        guard_graph_node("count", count, max_steps=1500)
        if guarded
        else count,
    )
    builder.set_entry_point("count")
    builder.add_conditional_edges("count", route, {"count": "count", END: END})
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.parametrize(
    ("target", "expected_next"),
    ((2, ("count",)), (1, ())),
)
def test_subprocess_kill_after_sqlite_commit_rebuilds_json_mirror(
    tmp_path, target, expected_next
):
    db_path = tmp_path / "langgraph.sqlite"
    json_path = tmp_path / "checkpoint.json"
    save_checkpoint(
        json_path,
        {"count": 0, "target": target},
        thread_id="stable",
        compatibility=COMPATIBILITY,
        graph_steps=0,
        last_node=None,
        next_nodes=("count",),
        cumulative_wall_time_s=3.0,
    )
    code = r"""
import os
import signal
import sys
import time
from typing import TypedDict
from langgraph.graph import END, StateGraph
from agent.checkpoint import sqlite_checkpointer, stream_with_checkpoints

class State(TypedDict):
    count: int
    target: int

def count(state):
    next_count = state["count"] + 1
    if next_count == 1:
        time.sleep(0.2)
    if next_count == 2:
        time.sleep(10)
    return {"count": next_count}

def route(state):
    return END if state["count"] >= state["target"] else "count"

builder = StateGraph(State)
builder.add_node("count", count)
builder.set_entry_point("count")
builder.add_conditional_edges("count", route, {"count": "count", END: END})
compatibility = {
    "mode": "cold_start",
    "pool_fingerprint": "pool",
    "topology_fingerprint": "topology",
    "config_fingerprint": "config",
}
with sqlite_checkpointer(sys.argv[1]) as saver:
    graph = builder.compile(checkpointer=saver)
    config = {
        "configurable": {"thread_id": "stable"},
        "recursion_limit": 10,
    }
    stream_with_checkpoints(
        graph,
        {"count": 0, "target": int(sys.argv[3])},
        config=config,
        checkpoint_path=sys.argv[2],
        thread_id="stable",
        compatibility=compatibility,
        initial_graph_steps=0,
        initial_wall_time_s=3.0,
        after_sqlite_commit=lambda snapshot: (
            os.kill(os.getpid(), signal.SIGKILL)
            if snapshot.step == 1
            else None
        ),
    )
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(db_path),
            str(json_path),
            str(target),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        check=False,
    )
    assert completed.returncode == -signal.SIGKILL
    stale = load_checkpoint(json_path, expected_compatibility=COMPATIBILITY)
    assert stale["progress"]["graph_steps"] == 0

    with sqlite_checkpointer(db_path) as saver:
        graph = _counter_graph(saver)
        config = {
            "configurable": {"thread_id": "stable"},
            "recursion_limit": 10,
        }
        before_reconcile = inspect_sqlite_state(graph, config)
        recovered_wall = cumulative_wall_time_from_sqlite(
            stale,
            before_reconcile,
        )
        reconciled = reconcile_checkpoint_from_sqlite(
            graph,
            config=config,
            checkpoint_path=json_path,
            thread_id="stable",
            compatibility=COMPATIBILITY,
            cumulative_wall_time_s=recovered_wall,
        )

    assert reconciled.state["count"] == 1
    assert reconciled.step == 1
    assert reconciled.last_node == "count"
    assert reconciled.next_nodes == expected_next
    mirrored = load_checkpoint(json_path, expected_compatibility=COMPATIBILITY)
    assert mirrored["state"]["count"] == 1
    assert mirrored["progress"]["sqlite_step"] == 1
    assert mirrored["progress"]["sqlite_checkpoint_id"]
    assert tuple(mirrored["progress"]["next_nodes"]) == expected_next
    assert mirrored["progress"]["status"] == (
        "completed" if not expected_next else "running"
    )
    assert mirrored["progress"]["cumulative_wall_time_s"] >= 3.15


def test_first_node_failure_uses_sqlite_pending_task_not_json_step(tmp_path):
    class State(TypedDict):
        attempts: int

    should_fail = {"value": True}

    def first(state):
        if should_fail["value"]:
            raise RuntimeError("first node failed")
        return {"attempts": state["attempts"] + 1}

    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.set_entry_point("first")
    builder.add_edge("first", END)
    db_path = tmp_path / "langgraph.sqlite"
    json_path = tmp_path / "checkpoint.json"
    save_checkpoint(
        json_path,
        {"attempts": 0},
        thread_id="stable",
        compatibility=COMPATIBILITY,
        graph_steps=0,
        last_node=None,
        next_nodes=("first",),
        cumulative_wall_time_s=0.0,
    )

    with sqlite_checkpointer(db_path) as saver:
        graph = builder.compile(checkpointer=saver)
        config = {
            "configurable": {"thread_id": "stable"},
            "recursion_limit": 3,
        }
        with pytest.raises(RuntimeError, match="first node failed"):
            stream_with_checkpoints(
                graph,
                {"attempts": 0},
                config=config,
                checkpoint_path=json_path,
                thread_id="stable",
                compatibility=COMPATIBILITY,
                initial_graph_steps=0,
                initial_wall_time_s=0.0,
            )
        authoritative = inspect_sqlite_state(graph, config)
        assert authoritative.exists is True
        assert authoritative.step == 0
        assert authoritative.next_nodes == ("first",)
        assert resume_input_from_sqlite(authoritative, {"attempts": 99}) is None

        should_fail["value"] = False
        result = stream_with_checkpoints(
            graph,
            resume_input_from_sqlite(authoritative, {"attempts": 99}),
            config=config,
            checkpoint_path=json_path,
            thread_id="stable",
            compatibility=COMPATIBILITY,
            initial_graph_steps=0,
            initial_wall_time_s=0.0,
        )

    assert result.state["attempts"] == 1
    assert result.graph_steps == 1


def test_first_node_failure_retries_after_process_reopen(tmp_path):
    db_path = tmp_path / "first.sqlite"
    json_path = tmp_path / "checkpoint.json"
    save_checkpoint(
        json_path,
        {"attempts": 0},
        thread_id="stable",
        compatibility=COMPATIBILITY,
        graph_steps=0,
        last_node=None,
        next_nodes=("first",),
        cumulative_wall_time_s=0.0,
    )
    code = r"""
import sys
from typing import TypedDict
from langgraph.graph import END, StateGraph
from agent.checkpoint import sqlite_checkpointer, stream_with_checkpoints
class State(TypedDict):
    attempts: int
def first(_state):
    raise RuntimeError("first process crash")
builder = StateGraph(State)
builder.add_node("first", first)
builder.set_entry_point("first")
builder.add_edge("first", END)
compatibility = {
    "mode": "cold_start",
    "pool_fingerprint": "pool",
    "topology_fingerprint": "topology",
    "config_fingerprint": "config",
}
with sqlite_checkpointer(sys.argv[1]) as saver:
    graph = builder.compile(checkpointer=saver)
    stream_with_checkpoints(
        graph,
        {"attempts": 0},
        config={"configurable": {"thread_id": "stable"}, "recursion_limit": 2},
        checkpoint_path=sys.argv[2],
        thread_id="stable",
        compatibility=compatibility,
        initial_graph_steps=0,
        initial_wall_time_s=0.0,
    )
"""
    failed = subprocess.run(
        [sys.executable, "-c", code, str(db_path), str(json_path)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert failed.returncode != 0

    class State(TypedDict):
        attempts: int

    builder = StateGraph(State)
    builder.add_node(
        "first",
        lambda state: {"attempts": state["attempts"] + 1},
    )
    builder.set_entry_point("first")
    builder.add_edge("first", END)
    with sqlite_checkpointer(db_path) as saver:
        graph = builder.compile(checkpointer=saver)
        config = {
            "configurable": {"thread_id": "stable"},
            "recursion_limit": 2,
        }
        before = inspect_sqlite_state(graph, config)
        assert before.step == 0
        assert before.next_nodes == ("first",)
        result = stream_with_checkpoints(
            graph,
            resume_input_from_sqlite(before, {"attempts": 99}),
            config=config,
            checkpoint_path=json_path,
            thread_id="stable",
            compatibility=COMPATIBILITY,
            initial_graph_steps=0,
            initial_wall_time_s=0.0,
        )

    assert result.state["attempts"] == 1


def test_json_write_error_after_sqlite_commit_is_reconciled_before_exit(
    tmp_path, monkeypatch
):
    import agent.checkpoint as checkpoint_module

    db_path = tmp_path / "dual.sqlite"
    json_path = tmp_path / "checkpoint.json"
    original_write = checkpoint_module.atomic_write_json
    failures = []

    def fail_once(path, payload):
        progress = payload.get("progress", {}) if isinstance(payload, dict) else {}
        if progress.get("sqlite_step") == 1 and not failures:
            failures.append("failed")
            raise OSError("mirror disk interruption")
        return original_write(path, payload)

    monkeypatch.setattr(checkpoint_module, "atomic_write_json", fail_once)
    with sqlite_checkpointer(db_path) as saver:
        graph = _counter_graph(saver)
        config = {
            "configurable": {"thread_id": "stable"},
            "recursion_limit": 3,
        }
        with pytest.raises(OSError, match="mirror disk interruption"):
            stream_with_checkpoints(
                graph,
                {"count": 0, "target": 1},
                config=config,
                checkpoint_path=json_path,
                thread_id="stable",
                compatibility=COMPATIBILITY,
                initial_graph_steps=0,
                initial_wall_time_s=0.0,
            )

    mirror = load_checkpoint(json_path, expected_compatibility=COMPATIBILITY)
    assert failures == ["failed"]
    assert mirror["state"]["count"] == 1
    assert mirror["progress"]["sqlite_step"] == 1
    assert mirror["progress"]["next_nodes"] == []


def test_langgraph_1500th_node_can_reach_end_without_recursion_error(tmp_path):
    db_path = tmp_path / "langgraph.sqlite"
    json_path = tmp_path / "checkpoint.json"
    with sqlite_checkpointer(db_path) as saver:
        graph = _counter_graph(saver, guarded=True)
        config = {
            "configurable": {"thread_id": "exact"},
            "recursion_limit": remaining_recursion_limit(0, total=1500),
        }
        result = stream_with_checkpoints(
            graph,
            {"count": 0, "target": 1500, "_graph_steps": 0},
            config=config,
            checkpoint_path=json_path,
            thread_id="exact",
            compatibility=COMPATIBILITY,
            initial_graph_steps=0,
            initial_wall_time_s=0.0,
        )
        snapshot = inspect_sqlite_state(graph, config)

    assert result.state["count"] == 1500
    assert result.state["_graph_steps"] == 1500
    assert snapshot.step == 1500
    assert snapshot.next_nodes == ()


def test_langgraph_cannot_execute_node_1501(tmp_path):
    db_path = tmp_path / "langgraph.sqlite"
    json_path = tmp_path / "checkpoint.json"
    with sqlite_checkpointer(db_path) as saver:
        graph = _counter_graph(saver, guarded=True)
        config = {
            "configurable": {"thread_id": "over"},
            "recursion_limit": remaining_recursion_limit(0, total=1500),
        }
        with pytest.raises(RecursionBudgetExhausted):
            stream_with_checkpoints(
                graph,
                {"count": 0, "target": 1501, "_graph_steps": 0},
                config=config,
                checkpoint_path=json_path,
                thread_id="over",
                compatibility=COMPATIBILITY,
                initial_graph_steps=0,
                initial_wall_time_s=0.0,
            )
        snapshot = inspect_sqlite_state(graph, config)

    assert snapshot.step == 1500
    assert snapshot.next_nodes == ("count",)
