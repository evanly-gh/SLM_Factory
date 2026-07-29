import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import pytest
from langgraph.graph import END, StateGraph

from agent.checkpoint import (
    CheckpointCompatibilityError,
    RecursionBudgetExhausted,
    atomic_write_jsonl,
    checkpoint_compatibility,
    create_run_manifest,
    durable_resume_available,
    load_run_manifest,
    load_checkpoint,
    remaining_recursion_limit,
    prepare_fresh_run_directory,
    require_sqlite_authority,
    restore_or_initialize_state,
    runtime_config_snapshot,
    save_checkpoint,
    sqlite_checkpointer,
    stream_with_checkpoints,
)
from agent.cost import CostEvent, CostLedger, install_cost_tracking
from agent.timing import TimingEvent, TimingLedger, install_timing_tracking


COMPATIBILITY = checkpoint_compatibility(
    mode="cold_start",
    pool_fingerprint="pool",
    topology_fingerprint="topology",
    config_fingerprint="config",
)


class _CrashAfterCompletedNode:
    def __init__(self):
        self.inputs = []
        self._next = ("right_branch",)

    def stream(self, value, *, stream_mode, config):
        self.inputs.append(value)
        assert stream_mode == "updates"
        assert config["configurable"]["thread_id"] == "stable"
        yield {"decision": {"route": "right", "scores": [0.5]}}
        raise RuntimeError("mid-node crash")

    def get_state(self, config):
        return SimpleNamespace(next=self._next)


class _ResumeRightBranch:
    def __init__(self):
        self.inputs = []

    def stream(self, value, *, stream_mode, config):
        self.inputs.append(value)
        yield {"right_branch": {"route": "done", "scores": [0.5, 0.8]}}

    def get_state(self, config):
        return SimpleNamespace(next=())


def test_mid_node_crash_resumes_with_none_and_preserves_conditional_route(tmp_path):
    path = tmp_path / "checkpoint.json"
    graph = _CrashAfterCompletedNode()

    with pytest.raises(RuntimeError, match="mid-node crash"):
        stream_with_checkpoints(
            graph,
            {"route": "start", "scores": []},
            config={
                "configurable": {"thread_id": "stable"},
                "recursion_limit": 1500,
            },
            checkpoint_path=path,
            thread_id="stable",
            compatibility=COMPATIBILITY,
            initial_graph_steps=0,
            initial_wall_time_s=0.0,
        )

    interrupted = load_checkpoint(path, expected_compatibility=COMPATIBILITY)
    assert interrupted["progress"]["last_node"] == "decision"
    assert interrupted["progress"]["next_nodes"] == ["right_branch"]
    assert interrupted["progress"]["graph_steps"] == 1

    resumed_graph = _ResumeRightBranch()
    result = stream_with_checkpoints(
        resumed_graph,
        None,
        config={
            "configurable": {"thread_id": "stable"},
            "recursion_limit": 1499,
        },
        checkpoint_path=path,
        thread_id="stable",
        compatibility=COMPATIBILITY,
        initial_graph_steps=interrupted["progress"]["graph_steps"],
        initial_wall_time_s=interrupted["progress"]["cumulative_wall_time_s"],
    )

    assert resumed_graph.inputs == [None]
    assert result.last_node == "right_branch"
    assert result.next_nodes == ()
    assert result.graph_steps == 2


def test_pregraph_state_factory_is_skipped_on_resume(tmp_path):
    path = tmp_path / "checkpoint.json"
    from agent.checkpoint import save_checkpoint

    save_checkpoint(
        path,
        {"description": "restored", "scores": [0.4]},
        thread_id="stable",
        compatibility=COMPATIBILITY,
        graph_steps=3,
        last_node="eval_setup",
        next_nodes=("model_selection",),
        cumulative_wall_time_s=40.0,
    )
    calls = []

    state, checkpoint, resumed = restore_or_initialize_state(
        path,
        fresh_state_factory=lambda: calls.append("called") or {"description": "fresh"},
        expected_compatibility=COMPATIBILITY,
    )

    assert resumed is True
    assert calls == []
    assert state["description"] == "restored"
    assert checkpoint["progress"]["next_nodes"] == ["model_selection"]


def test_remaining_recursion_budget_never_resets():
    assert remaining_recursion_limit(0, total=1500) == 1501
    assert remaining_recursion_limit(1499, total=1500) == 2
    assert remaining_recursion_limit(1500, total=1500) == 1
    with pytest.raises(RecursionBudgetExhausted, match="1500"):
        remaining_recursion_limit(1501, total=1500)


def test_existing_cost_and_timing_ledgers_are_appended_on_resume(tmp_path):
    cost_path = tmp_path / "cost.jsonl"
    timing_path = tmp_path / "timing.jsonl"
    cost_path.write_text(
        json.dumps(
            CostEvent(provider="local", model="m", stage="before").to_dict()
        )
        + "\n",
        encoding="utf-8",
    )
    timing_path.write_text(
        json.dumps(TimingEvent(kind="phase", name="before", duration_ms=1).to_dict())
        + "\n",
        encoding="utf-8",
    )

    install_cost_tracking(cost_path, required=True)
    install_timing_tracking(timing_path, required=True)
    CostLedger(cost_path).append(
        CostEvent(provider="local", model="m", stage="after")
    )
    TimingLedger(timing_path).append(
        TimingEvent(kind="phase", name="after", duration_ms=2)
    )

    assert [event["stage"] for event in CostLedger(cost_path).events()] == [
        "before",
        "after",
    ]
    assert [event["name"] for event in TimingLedger(timing_path).events()] == [
        "before",
        "after",
    ]


def test_atomic_dataset_write_never_exposes_partial_content(tmp_path, monkeypatch):
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"old":true}\n', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated publication crash")

    monkeypatch.setattr("agent.checkpoint.os.replace", fail_replace)
    with pytest.raises(OSError, match="publication crash"):
        atomic_write_jsonl(path, [{"new": True}])

    assert path.read_text(encoding="utf-8") == '{"old":true}\n'
    assert not list(tmp_path.glob("dataset.jsonl.tmp.*"))


def test_run_manifest_keeps_stable_thread_and_rejects_task_drift(tmp_path):
    manifest_path = tmp_path / "run-manifest.json"
    created = create_run_manifest(
        manifest_path,
        run_dir=tmp_path,
        description="original task",
        force_model="Qwen/Qwen3-0.6B@Q4_K_M",
        mode="cold_start",
        compatibility=COMPATIBILITY,
    )
    loaded = load_run_manifest(
        manifest_path,
        expected_description="original task",
        expected_force_model="Qwen/Qwen3-0.6B@Q4_K_M",
    )

    assert loaded["thread_id"] == created["thread_id"]
    assert loaded["run_dir"] == str(tmp_path.resolve())
    assert loaded["compatibility"] == COMPATIBILITY
    with pytest.raises(Exception, match="description"):
        load_run_manifest(
            manifest_path,
            expected_description="different task",
            expected_force_model="Qwen/Qwen3-0.6B@Q4_K_M",
        )


def test_run_manifest_persists_human_readable_effective_config(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "4096")
    monkeypatch.setenv("SLM_EVAL_MAX_NEW_TOKENS_APPS", "1024")
    monkeypatch.setenv("SLM_APPS_PROBLEM_TIMEOUT_S", "5")
    manifest_path = tmp_path / "run-manifest.json"

    created = create_run_manifest(
        manifest_path,
        run_dir=tmp_path,
        description="config snapshot",
        force_model="",
        mode="cold_start",
        compatibility=COMPATIBILITY,
    )

    snapshot = created["effective_config"]
    assert created["schema_version"] >= 3
    assert snapshot["SLM_MAX_SEQ_LENGTH"] == "4096"
    assert snapshot["SLM_EVAL_MAX_NEW_TOKENS_APPS"] == "1024"
    assert snapshot["SLM_APPS_PROBLEM_TIMEOUT_S"] == "5"
    assert snapshot["mode"] == "cold_start"
    assert "ANTHROPIC_API_KEY" not in snapshot
    assert "EXA_API_KEY" not in snapshot


def test_runtime_config_ignores_movable_gpu_profile_but_keeps_context(
    monkeypatch,
):
    profile_settings = {
        "SLM_GPU_PROFILE": "auto-2gpu",
        "SLM_GPU_COUNT": "2",
        "SLM_SYNTH_GPU_IDS": "0",
        "SLM_SYNTH_TP": "1",
        "SLM_SYNTH_GPU_UTILIZATION": "0.82",
        "SLM_SYNTH_MAX_NUM_SEQS": "16",
        "SLM_SYNTH_CONCURRENCY": "8",
        "SLM_PIPELINE_GPU_ID": "1",
    }
    for name, value in profile_settings.items():
        monkeypatch.setenv(name, value)
    first = runtime_config_snapshot("cold_start")

    for name, value in profile_settings.items():
        monkeypatch.setenv(name, f"different-{value}")
    second = runtime_config_snapshot("cold_start")

    assert first == second
    assert not profile_settings.keys() & first.keys()
    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "8192")
    assert runtime_config_snapshot("cold_start") != second


def test_run_manifest_reports_effective_config_drift_by_setting(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "4096")
    manifest_path = tmp_path / "run-manifest.json"
    create_run_manifest(
        manifest_path,
        run_dir=tmp_path,
        description="config drift",
        force_model="",
        mode="cold_start",
        compatibility=COMPATIBILITY,
    )
    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "8192")
    expected_config = runtime_config_snapshot("cold_start")

    with pytest.raises(
        CheckpointCompatibilityError,
        match=r"effective config drift.*SLM_MAX_SEQ_LENGTH",
    ):
        load_run_manifest(
            manifest_path,
            expected_description="config drift",
            expected_effective_config=expected_config,
        )


def test_partial_stable_run_directory_is_repaired_safely(tmp_path):
    run_dir = tmp_path / "stable"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "run.log").write_text("partial startup\n", encoding="utf-8")
    (run_dir / "cost-events.jsonl").write_text("", encoding="utf-8")

    prepare_fresh_run_directory(run_dir)

    assert run_dir.is_dir()
    assert (run_dir / "artifacts").is_dir()
    assert not (run_dir / "run.log").exists()


def test_partial_startup_repair_works_after_process_reopen(tmp_path):
    run_dir = tmp_path / "stable-subprocess"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "run.log").write_text("killed during startup\n", encoding="utf-8")
    code = (
        "import sys\n"
        "from agent.checkpoint import prepare_fresh_run_directory\n"
        "prepare_fresh_run_directory(sys.argv[1])\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code, str(run_dir)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )

    assert completed.returncode == 0
    assert (run_dir / "artifacts").is_dir()
    assert not (run_dir / "run.log").exists()


def test_partial_run_with_unknown_content_is_not_deleted(tmp_path):
    run_dir = tmp_path / "stable"
    run_dir.mkdir()
    important = run_dir / "operator-notes.txt"
    important.write_text("keep", encoding="utf-8")

    with pytest.raises(Exception, match="unknown"):
        prepare_fresh_run_directory(run_dir)

    assert important.read_text(encoding="utf-8") == "keep"


def test_shell_resume_requires_manifest_and_checkpoint(tmp_path):
    run_dir = tmp_path / "stable"
    run_dir.mkdir()
    assert durable_resume_available(run_dir) is False
    (run_dir / "run-manifest.json").write_text("{}\n", encoding="utf-8")
    assert durable_resume_available(run_dir) is False
    (run_dir / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    assert durable_resume_available(run_dir) is False
    (run_dir / "run-manifest.json").unlink()
    (run_dir / "checkpoint.json").unlink()
    manifest = create_run_manifest(
        run_dir / "run-manifest.json",
        run_dir=run_dir,
        description="task",
        force_model="",
        mode="cold_start",
        compatibility=COMPATIBILITY,
    )
    save_checkpoint(
        run_dir / "checkpoint.json",
        {"description": "task"},
        thread_id=manifest["thread_id"],
        compatibility=COMPATIBILITY,
        graph_steps=0,
        last_node=None,
        next_nodes=("__pregraph__",),
        cumulative_wall_time_s=0.0,
    )
    assert durable_resume_available(run_dir) is True


def test_durable_resume_probe_needs_no_provider_credentials(tmp_path):
    run_dir = tmp_path / "credential-free-probe"
    run_dir.mkdir()
    manifest = create_run_manifest(
        run_dir / "run-manifest.json",
        run_dir=run_dir,
        description="task",
        force_model="",
        mode="cold_start",
        compatibility=COMPATIBILITY,
        effective_config={
            "mode": "cold_start",
            "SLM_MAX_SEQ_LENGTH": "4096",
        },
    )
    save_checkpoint(
        run_dir / "checkpoint.json",
        {"description": "task", "_graph_steps": 0},
        thread_id=manifest["thread_id"],
        compatibility=COMPATIBILITY,
        graph_steps=0,
        last_node=None,
        next_nodes=("__pregraph__",),
        cumulative_wall_time_s=0.0,
    )
    code = """
import os
import sys
assert "ANTHROPIC_API_KEY" not in os.environ
assert "EXA_API_KEY" not in os.environ
from agent.checkpoint import durable_resume_available
raise SystemExit(0 if durable_resume_available(sys.argv[1]) else 1)
"""

    completed = subprocess.run(
        [sys.executable, "-c", code, str(run_dir)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": str(Path(sys.executable).parent),
            "PYTHONHASHSEED": "0",
        },
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_env_only_runner_snapshot_still_reports_strict_config_drift(
    tmp_path,
):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ANTHROPIC_API_KEY=env-only-anthropic\n"
        "EXA_API_KEY=env-only-exa\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "env-only-run"
    code = """
import os
import sys
from dotenv import load_dotenv

os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("EXA_API_KEY", None)
load_dotenv(sys.argv[1], override=True)

from agent.checkpoint import (
    CheckpointCompatibilityError,
    checkpoint_compatibility,
    create_run_manifest,
    load_run_manifest,
    runtime_config_snapshot,
)

run_dir = sys.argv[2]
snapshot = runtime_config_snapshot("cold_start")
compatibility = checkpoint_compatibility(
    mode="cold_start",
    pool_fingerprint="pool",
    topology_fingerprint="topology",
    config_fingerprint="config",
)
create_run_manifest(
    os.path.join(run_dir, "run-manifest.json"),
    run_dir=run_dir,
    description="env-only",
    force_model="",
    mode="cold_start",
    compatibility=compatibility,
    effective_config=snapshot,
)
os.environ["SLM_MAX_SEQ_LENGTH"] = "8192"
try:
    load_run_manifest(
        os.path.join(run_dir, "run-manifest.json"),
        expected_effective_config=runtime_config_snapshot("cold_start"),
    )
except CheckpointCompatibilityError as error:
    if "SLM_MAX_SEQ_LENGTH" in str(error):
        raise SystemExit(0)
    raise
raise SystemExit("strict drift was not detected")
"""

    completed = subprocess.run(
        [sys.executable, "-c", code, str(env_path), str(run_dir)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": str(Path(sys.executable).parent),
            "PYTHONHASHSEED": "0",
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _durable_checkpoint(run_dir: Path, *, progressed: bool):
    manifest = create_run_manifest(
        run_dir / "run-manifest.json",
        run_dir=run_dir,
        description="task",
        force_model="",
        mode="cold_start",
        compatibility=COMPATIBILITY,
    )
    checkpoint = save_checkpoint(
        run_dir / "checkpoint.json",
        {
            "description": "task",
            "_graph_steps": 1 if progressed else 0,
        },
        thread_id=manifest["thread_id"],
        compatibility=COMPATIBILITY,
        graph_steps=1 if progressed else 0,
        last_node="task_analysis" if progressed else None,
        next_nodes=("eval_setup",) if progressed else ("__pregraph__",),
        cumulative_wall_time_s=1.0,
        sqlite_checkpoint_id="generation-1" if progressed else None,
        sqlite_generation="generation-1" if progressed else None,
        sqlite_step=1 if progressed else None,
    )
    return manifest, checkpoint


def _seed_sqlite_thread(path: Path, thread_id: str):
    class State(TypedDict):
        value: int

    builder = StateGraph(State)
    builder.add_node("done", lambda state: {"value": state["value"] + 1})
    builder.set_entry_point("done")
    builder.add_edge("done", END)
    with sqlite_checkpointer(path) as saver:
        graph = builder.compile(checkpointer=saver)
        list(
            graph.stream(
                {"value": 0},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 2,
                },
                stream_mode="updates",
            )
        )


def test_progressed_json_fails_closed_when_sqlite_is_missing(tmp_path):
    run_dir = tmp_path / "missing-sqlite"
    run_dir.mkdir()
    manifest, checkpoint = _durable_checkpoint(run_dir, progressed=True)

    with pytest.raises(
        CheckpointCompatibilityError,
        match="graph progress.*SQLite.*missing",
    ):
        require_sqlite_authority(
            checkpoint,
            sqlite_path=manifest["sqlite_path"],
            expected_thread_id=manifest["thread_id"],
        )

    assert durable_resume_available(run_dir) is False
    with pytest.raises(CheckpointCompatibilityError, match="SQLite.*missing"):
        prepare_fresh_run_directory(run_dir)
    assert (run_dir / "run-manifest.json").is_file()
    assert (run_dir / "checkpoint.json").is_file()


def test_zero_step_sqlite_generation_still_requires_database(tmp_path):
    run_dir = tmp_path / "generation-zero"
    run_dir.mkdir()
    manifest, _ = _durable_checkpoint(run_dir, progressed=False)
    checkpoint = save_checkpoint(
        run_dir / "checkpoint.json",
        {"description": "task", "_graph_steps": 0},
        thread_id=manifest["thread_id"],
        compatibility=COMPATIBILITY,
        graph_steps=0,
        last_node=None,
        next_nodes=("task_analysis",),
        cumulative_wall_time_s=1.0,
        sqlite_checkpoint_id="ready-generation",
        sqlite_generation="ready-generation",
        sqlite_step=0,
    )

    with pytest.raises(
        CheckpointCompatibilityError,
        match="graph progress.*SQLite.*missing",
    ):
        require_sqlite_authority(
            checkpoint,
            sqlite_path=manifest["sqlite_path"],
            expected_thread_id=manifest["thread_id"],
        )
    assert durable_resume_available(run_dir) is False


def test_zero_step_state_progress_still_requires_database(tmp_path):
    run_dir = tmp_path / "state-progress"
    run_dir.mkdir()
    manifest, _ = _durable_checkpoint(run_dir, progressed=False)
    checkpoint = save_checkpoint(
        run_dir / "checkpoint.json",
        {
            "description": "task",
            "_graph_steps": 0,
            "task_type": "classification",
        },
        thread_id=manifest["thread_id"],
        compatibility=COMPATIBILITY,
        graph_steps=0,
        last_node=None,
        next_nodes=("task_analysis",),
        cumulative_wall_time_s=1.0,
    )

    with pytest.raises(CheckpointCompatibilityError, match="graph progress"):
        require_sqlite_authority(
            checkpoint,
            sqlite_path=manifest["sqlite_path"],
            expected_thread_id=manifest["thread_id"],
        )


def test_progressed_json_fails_closed_when_sqlite_is_empty(tmp_path):
    run_dir = tmp_path / "empty-sqlite"
    run_dir.mkdir()
    manifest, checkpoint = _durable_checkpoint(run_dir, progressed=True)
    with sqlite_checkpointer(manifest["sqlite_path"]):
        pass

    with pytest.raises(
        CheckpointCompatibilityError,
        match="SQLite.*no checkpoints",
    ):
        require_sqlite_authority(
            checkpoint,
            sqlite_path=manifest["sqlite_path"],
            expected_thread_id=manifest["thread_id"],
        )

    assert durable_resume_available(run_dir) is False
    with pytest.raises(CheckpointCompatibilityError, match="no checkpoints"):
        prepare_fresh_run_directory(run_dir)
    assert (run_dir / "checkpoint.json").is_file()


def test_progressed_json_fails_closed_for_wrong_sqlite_thread(tmp_path):
    run_dir = tmp_path / "wrong-thread"
    run_dir.mkdir()
    manifest, checkpoint = _durable_checkpoint(run_dir, progressed=True)
    _seed_sqlite_thread(Path(manifest["sqlite_path"]), "other-thread")

    with pytest.raises(
        CheckpointCompatibilityError,
        match="expected thread.*other-thread",
    ):
        require_sqlite_authority(
            checkpoint,
            sqlite_path=manifest["sqlite_path"],
            expected_thread_id=manifest["thread_id"],
        )

    assert durable_resume_available(run_dir) is False
    with pytest.raises(CheckpointCompatibilityError, match="expected thread"):
        prepare_fresh_run_directory(run_dir)
    assert (run_dir / "checkpoint.json").is_file()


@pytest.mark.parametrize("sqlite_mode", ("missing", "empty", "wrong-thread"))
def test_fresh_pregraph_checkpoint_allows_no_expected_sqlite_state(
    tmp_path,
    sqlite_mode,
):
    run_dir = tmp_path / sqlite_mode
    run_dir.mkdir()
    manifest, checkpoint = _durable_checkpoint(run_dir, progressed=False)
    if sqlite_mode == "empty":
        with sqlite_checkpointer(manifest["sqlite_path"]):
            pass
    elif sqlite_mode == "wrong-thread":
        _seed_sqlite_thread(Path(manifest["sqlite_path"]), "other-thread")

    require_sqlite_authority(
        checkpoint,
        sqlite_path=manifest["sqlite_path"],
        expected_thread_id=manifest["thread_id"],
    )
    assert durable_resume_available(run_dir) is True


def test_pipeline_runner_exposes_resume_environment_and_stream_contract():
    source = (
        Path(__file__).with_name("run.py").read_text(encoding="utf-8")
    )

    assert '"--resume"' in source
    assert "SLM_RESUME" in source
    assert "SLM_RUN_DIR" in source
    assert "sqlite_checkpointer" in source
    assert "stream_with_checkpoints" in source
    assert "resume_input_from_sqlite" in source
    assert "require_sqlite_authority" in source
    assert "_EFFECTIVE_CONFIG = runtime_config_snapshot(_MODE)" in source
    assert "expected_effective_config=_EFFECTIVE_CONFIG" in source
    assert source.index("load_dotenv(") < source.index(
        "_EFFECTIVE_CONFIG = runtime_config_snapshot(_MODE)"
    )
    assert source.index("require_sqlite_authority(") < source.index(
        "research_device(description"
    )
    assert source.index('status="initializing"') < source.index(
        "research_device(description"
    )
    assert source.index("signal.signal") < source.index(
        "research_device(description"
    )
