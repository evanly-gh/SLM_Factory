import os
from pathlib import Path
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END, StateGraph

os.environ.setdefault("ANTHROPIC_API_KEY", "test-no-network")
os.environ.setdefault("EXA_API_KEY", "test-no-network")

from agent.checkpoint import (
    CheckpointCorruptError,
    SafeCheckpointSerializer,
    sqlite_checkpointer,
)
from agent.graph import build_graph, graph_run_config, graph_topology_descriptor
from config.android_pool import ANDROID_POOL, HardwareConstraints
from data.eval_set import EvalSet
from eval.harness import EvalResult
from training.lora_trainer import TrainingOutput


def test_build_graph_passes_optional_checkpointer_to_langgraph_compile():
    checkpointer = object()
    compiled = MagicMock()
    compiled.with_config.return_value = compiled

    with patch("agent.graph.StateGraph.compile", return_value=compiled) as compile_graph:
        result = build_graph(mode="cold_start", checkpointer=checkpointer)

    assert result is compiled
    compile_graph.assert_called_once_with(checkpointer=checkpointer)


def test_graph_run_config_requires_and_carries_stable_thread_id():
    assert graph_run_config("stable-thread", recursion_limit=1490) == {
        "configurable": {"thread_id": "stable-thread"},
        "recursion_limit": 1490,
    }
    with pytest.raises(ValueError, match="thread_id"):
        graph_run_config("", recursion_limit=1490)


def test_topology_descriptor_is_stable_and_mode_specific():
    cold = graph_topology_descriptor("cold_start")
    production = graph_topology_descriptor("production")

    assert cold == graph_topology_descriptor("cold_start")
    assert cold["entry_point"] == "task_analysis"
    assert production["entry_point"] == "trace_ingest"
    assert "task_analysis" in cold["nodes"]
    assert "trace_ingest" not in cold["nodes"]
    assert "trace_ingest" in production["nodes"]
    assert cold["conditional_edges"]["downward_probe"] == {
        "downward_probe": "downward_probe",
        "terminate": "__end__",
    }
    assert cold["entry_routes"] == {
        "task_analysis": "task_analysis",
        "terminate": "__end__",
    }
    for source in (
        "task_analysis",
        "eval_setup",
        "model_selection",
        "curate",
        "train",
        "rollback",
    ):
        assert "terminate" in cold["conditional_edges"][source]


def test_wall_guard_routes_before_every_long_transition(monkeypatch):
    import agent.graph as graph_module

    monkeypatch.setattr(graph_module, "_wallclock_exceeded", lambda: True)
    state = {"next_action": "train", "scores": []}

    assert graph_module._route_before("task_analysis")(state) == "terminate"
    assert graph_module._route_after_evaluate(state) == "terminate"
    assert graph_module._route_after_iterate(state) == "terminate"
    assert graph_module._route_after_escalate(state) == "terminate"
    assert graph_module._route_after_downward_probe(state) == "terminate"

    monkeypatch.setattr(graph_module, "_wallclock_exceeded", lambda: False)
    exhausted = {"_graph_steps": 1500, "next_action": "train", "scores": []}
    assert graph_module._route_before("train")(exhausted) == "terminate"
    assert graph_module._route_after_iterate(exhausted) == "terminate"


def test_node_wall_guard_skips_long_side_effect_and_routes_cleanly(monkeypatch):
    import agent.graph as graph_module

    monkeypatch.setattr(graph_module, "_wallclock_exceeded", lambda: True)
    calls = []
    guarded = graph_module.guard_graph_node(
        "train",
        lambda state: calls.append("trained") or state,
        max_steps=1500,
    )

    result = guarded({"_graph_steps": 4, "next_action": "train"})

    assert calls == []
    assert result["_graph_steps"] == 5
    assert result["_wallclock_terminated_before"] == "train"
    assert result["next_action"] == "terminate"


@pytest.mark.parametrize("mode", ("cold_start", "production"))
def test_topology_descriptor_matches_compiled_graph(mode):
    descriptor = graph_topology_descriptor(mode)
    compiled = build_graph(mode).get_graph().to_json()
    actual_nodes = {
        node["id"] for node in compiled["nodes"]
        if node["id"] not in {"__start__", "__end__"}
    }
    actual_edges = {
        (
            edge["source"],
            edge["target"],
            bool(edge.get("conditional")),
        )
        for edge in compiled["edges"]
    }
    expected_edges = {
        *(
            ("__start__", target, True)
            for target in descriptor["entry_routes"].values()
        ),
        *(
            (source, target, False)
            for source, target in descriptor["edges"]
        ),
        *(
            (source, target, True)
            for source, routes in descriptor["conditional_edges"].items()
            for target in routes.values()
        ),
    }

    assert actual_nodes == set(descriptor["nodes"])
    assert actual_edges == expected_edges


def test_sqlite_serializer_stores_selectors_and_paths_not_domain_instances():
    serializer = SafeCheckpointSerializer()
    model = ANDROID_POOL[0]
    value = {
        "model": model,
        "hardware": HardwareConstraints(1000, 800, 2000),
        "eval_set": EvalSet(
            pos=[{"text": "p"}],
            neg=[],
            boundary=[],
            task_type="classification",
        ),
        "result": EvalResult(0.5, {}, 0.5, 0.0, 0.0, []),
        "training": TrainingOutput("/weights", "/weights/model.gguf"),
    }

    encoded = serializer.dumps_typed(value)
    blob = encoded[1]
    decoded = serializer.loads_typed(encoded)

    assert model.selector.encode() in blob
    assert model.notes.encode() not in blob
    assert decoded["model"] is model
    assert decoded["hardware"] == value["hardware"]
    assert decoded["eval_set"] == value["eval_set"]
    assert decoded["result"] == value["result"]
    assert decoded["training"] == value["training"]


def test_sqlite_serializer_rejects_arbitrary_runtime_objects():
    with pytest.raises(TypeError, match="runtime object"):
        SafeCheckpointSerializer().dumps_typed({"trainer": object()})


def test_sqlite_checkpointer_context_creates_database(tmp_path):
    path = tmp_path / "langgraph.sqlite"

    with sqlite_checkpointer(path) as saver:
        assert saver is not None

    assert path.is_file()


def test_corrupt_sqlite_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "langgraph.sqlite"
    path.write_bytes(b"not a sqlite database")

    with pytest.raises(CheckpointCorruptError, match="SQLite"):
        with sqlite_checkpointer(path):
            pass


def test_sqlite_stream_none_resumes_pending_conditional_branch(tmp_path):
    class State(TypedDict):
        selected_model: object
        route: str
        visited: list[str]

    attempts = {"crash": True}
    model = ANDROID_POOL[0]

    def decide(state):
        return {"route": "right"}

    def choose(state):
        return state["route"]

    def right(state):
        assert state["selected_model"] is model
        if attempts["crash"]:
            raise RuntimeError("crash inside right")
        return {"visited": list(state["visited"]) + ["right"]}

    builder = StateGraph(State)
    builder.add_node("decide", decide)
    builder.add_node("right", right)
    builder.set_entry_point("decide")
    builder.add_conditional_edges("decide", choose, {"right": "right"})
    builder.add_edge("right", END)
    config = {"configurable": {"thread_id": "stable"}, "recursion_limit": 10}

    with sqlite_checkpointer(tmp_path / "langgraph.sqlite") as saver:
        graph = builder.compile(checkpointer=saver)
        with pytest.raises(RuntimeError, match="crash inside right"):
            list(
                graph.stream(
                    {
                        "selected_model": model,
                        "route": "",
                        "visited": [],
                    },
                    config=config,
                    stream_mode="updates",
                )
            )
        assert graph.get_state(config).next == ("right",)

        attempts["crash"] = False
        updates = list(
            graph.stream(None, config=config, stream_mode="updates")
        )

    assert updates == [{"right": {"visited": ["right"]}}]
