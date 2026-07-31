from pathlib import Path
from unittest.mock import patch

from agent.nodes.iterate import iterate_node
from agent.graph import graph_topology_descriptor


def _state():
    return {
        "selected_model": None,
        "scores": [0.7],
        "best_score": 0.7,
        "iteration": 2,
        "turn_budget": 100,
        "stop_threshold": 0.9,
        "initial_stop_threshold": 0.9,
        "task_type": "classification",
        "dataset_version": 1,
        "current_dataset_path": "/dataset-v1.jsonl",
        "curriculum_size_target": 64,
        "source_acquire_rounds_used": 0,
        "last_eval": None,
        "test_report": None,
        "last_curation": None,
        "last_hypothesis": "",
        "dag": [],
        "hw_gating_enabled": False,
        "consecutive_no_improvement": 0,
    }


@patch(
    "agent.nodes.iterate._llm_iterate",
    return_value={
        "intervention": "data_rebuild",
        "hypothesis": "hard train-only examples are underrepresented",
        "data_rebuild": {
            "strategy": "synthesize",
            "target_rows": 64,
        },
    },
)
def test_data_rebuild_decision_stores_plan_and_routes_to_curate(_decision):
    out = iterate_node(_state())

    assert out["next_action"] == "curate"
    assert out["last_intervention"] == "data_rebuild"
    assert out["data_rebuild_plan"]["strategy"] == "synthesize"


def test_graph_iterate_routes_only_supported_actions():
    routes = graph_topology_descriptor("cold_start")[
        "conditional_edges"
    ]["iterate"]
    assert routes == {
        "train": "train",
        "curate": "curate",
        "escalate": "escalate",
        "downward_probe": "downward_probe",
        "terminate": "__end__",
    }


def test_removed_route_tokens_absent_from_active_code_docs_and_tests():
    root = Path(__file__).resolve().parents[2]
    forbidden = (
        "sur" + "gical",
        "targeted_" + "patterns",
    )
    offenders = []
    for directory in ("agent", "data", "docs", "tests"):
        for path in (root / directory).rglob("*"):
            if path.suffix not in {".py", ".md"}:
                continue
            if "docs/superpowers/plans" in path.as_posix():
                continue
            # Historical bug records intentionally name removed routes.
            if path == root / "docs" / "BUGS.md":
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(
                        f"{path.relative_to(root)} contains {token!r}"
                    )
    assert offenders == []
