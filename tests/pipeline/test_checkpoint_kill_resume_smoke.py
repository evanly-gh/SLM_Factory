import json
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "smoke_checkpoint_kill_resume.py"


def test_checkpoint_kill_resume_smoke_uses_real_sqlite_and_lightweight_workers():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "sqlite_checkpointer" in source
    assert "stream_with_checkpoints" in source
    assert "resume_input_from_sqlite" in source
    assert "cumulative_wall_time_from_sqlite" in source
    assert "signal.SIGKILL" in source
    assert "--worker" in source
    assert '"train"' in source
    assert '"eval"' in source
    assert "transformers" not in source
    assert "huggingface" not in source.lower()
    assert "from_pretrained" not in source


def test_checkpoint_kill_resume_smoke_reopens_only_pending_node_and_ledgers(
    tmp_path,
):
    run_dir = tmp_path / "checkpoint-smoke"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run_dir)],
        cwd=ROOT,
        env={
            "PATH": str(Path(sys.executable).parent),
            "PYTHONHASHSEED": "0",
        },
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    pass_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("PASS checkpoint kill/resume smoke: ")
    )
    summary = json.loads(pass_line.split(": ", 1)[1])
    assert summary["interrupted_returncode"] == -signal.SIGKILL
    assert summary["sqlite_step_before_resume"] == 2
    assert summary["final_sqlite_step"] == 3
    assert summary["executed_nodes"] == ["prepare", "train", "eval"]
    assert summary["resumed_nodes"] == ["eval"]
    assert summary["state_ledger"] == ["prepare", "train", "eval"]
    assert summary["cost_stages"] == ["train", "eval"]
    assert summary["timing_names"] == ["train", "eval"]
    assert summary["cumulative_wall_time_s"] >= 7.0

    checkpoint = json.loads(
        (run_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["progress"]["graph_steps"] == 3
    assert checkpoint["progress"]["sqlite_step"] == 3
    assert checkpoint["progress"]["next_nodes"] == []
    assert checkpoint["progress"]["status"] == "completed"
    assert checkpoint["state"]["_graph_steps"] == 3
    assert checkpoint["state"]["ledger"] == ["prepare", "train", "eval"]

    executions = [
        json.loads(line)
        for line in (run_dir / "executions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [entry["node"] for entry in executions] == [
        "prepare",
        "train",
        "eval",
    ]
    worker_entries = [entry for entry in executions if entry["worker"]]
    assert [entry["node"] for entry in worker_entries] == ["train", "eval"]
    assert len({entry["pid"] for entry in worker_entries}) == 2
