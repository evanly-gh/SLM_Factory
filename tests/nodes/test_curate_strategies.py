import json

import agent.nodes.curate as curate
from agent.data_rebuild import normalize_data_rebuild_plan
from agent.nodes.curate import curate_node
from config.android_pool import ANDROID_POOL
from data.eval_set import EvalSet


EVAL_SECRET = "held out evaluation secret 7319"


def _eval_set():
    return EvalSet(
        pos=[{"text": EVAL_SECRET, "label": "a"}],
        neg=[],
        boundary=[],
        task_type="classification",
    )


def _plan(strategy, **overrides):
    raw = {"strategy": strategy, "target_rows": 24, "resample_fraction": 1.0}
    raw.update(overrides)
    return normalize_data_rebuild_plan(
        raw, task_type="classification", hypothesis="hard gap"
    )


def _state(plan, rows):
    return {
        "task_type": "classification",
        "selected_model": ANDROID_POOL[0],
        "last_intervention": "data_rebuild",
        "last_hypothesis": "hard gap",
        "eval_set": _eval_set(),
        "train_examples": rows,
        "test_report": {},
        "current_dataset_path": None,
        "curriculum_size_target": 24,
        "dataset_version": 0,
        "data_source": "fixture",
        "data_sources": [],
        "eval_source_ban": [],
        "source_acquire_rounds_used": 0,
        "data_rebuild_plan": plan,
        "dag": [],
    }


def test_seed_is_entropy_based_not_derived():
    assert curate._entropy_seed() != curate._entropy_seed()


def test_resample_produces_dataset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLM_CHEAP", "1")  # avoid CoT/synth network paths
    rows = [{"text": f"row {i}", "label": "a" if i % 2 else "b"} for i in range(30)]
    plan = _plan("resample")
    out = curate_node(_state(plan, rows))
    path = out["current_dataset_path"]
    assert path and path.endswith("dataset_v1.jsonl")
    written = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert len(written) > 0
    # Eval firewall: the held-out secret never appears in training data.
    assert all(EVAL_SECRET not in r.get("text", "") for r in written)
    assert out["last_curation"]["strategy"] == "resample"
