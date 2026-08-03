import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.nodes.cold_start import eval_setup
from data.loaders import web_acquire
from data.loaders.dataset_integrity import sha256_file
from scripts import prepare_shared_dataset


def _state():
    return {
        "task_type": "generation",
        "task_plan": {
            "task_type": "generation",
            "benchmark": "SAMSum",
            "multi_label": False,
            "schema": None,
            "multilingual": False,
        },
        "description": "dialogue summarization",
        "curriculum_size_target": 10,
        "eval_size_target": 6,
        "data_sources": [],
        "eval_source_ban": [],
        "feasible_models": [],
    }


def _install_lightweight_dependencies(monkeypatch, tmp_path, acquire):
    monkeypatch.setitem(
        sys.modules,
        "config.config",
        SimpleNamespace(DATA_SIZE_CEILING=100),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.nodes.test_agent",
        SimpleNamespace(label_difficulty=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(web_acquire, "acquire_dataset", acquire)
    monkeypatch.setattr(eval_setup, "ARTIFACTS_DIR", str(tmp_path))


def test_eval_setup_records_only_declared_eval_split_bans(tmp_path, monkeypatch, capsys):
    train_record = {
        "kind": "hf", "id": "knkarthick/samsum", "split": "train", "role": "curriculum"
    }
    test_record = {
        "kind": "hf", "id": "knkarthick/samsum", "split": "test", "role": "eval"
    }

    def acquire(*_args, meta, **_kwargs):
        meta["source"] = "local SAMSum"
        meta["source_records"] = [train_record, test_record]
        meta["eval_ban"] = [test_record]
        return (
            [{"text": "train dialogue", "answer": "train summary", "label": "generation"}],
            [{"text": "test dialogue", "answer": "test summary", "label": "generation"}],
        )

    _install_lightweight_dependencies(monkeypatch, tmp_path, acquire)

    result = eval_setup.eval_setup_node(_state())

    assert result["data_sources"] == [train_record, test_record]
    assert result["eval_source_ban"] == [test_record]
    output = capsys.readouterr().out
    assert "eval source restrictions recorded" in output
    assert "locked" not in output


def test_eval_setup_does_not_invent_eval_bans_from_provenance(tmp_path, monkeypatch):
    records = [
        {"kind": "hf", "id": "source", "split": "train", "role": "curriculum"},
        {"kind": "hf", "id": "source", "split": "test", "role": "eval"},
    ]

    def acquire(*_args, meta, **_kwargs):
        meta["source"] = "source without declared restriction"
        meta["source_records"] = records
        return (
            [{"text": "train", "answer": "a", "label": "generation"}],
            [{"text": "test", "answer": "b", "label": "generation"}],
        )

    _install_lightweight_dependencies(monkeypatch, tmp_path, acquire)

    result = eval_setup.eval_setup_node(_state())

    assert result["data_sources"] == records
    assert result["eval_source_ban"] == []


def test_eval_setup_enforces_normalized_train_test_separation(tmp_path, monkeypatch):
    def acquire(*_args, meta, **_kwargs):
        meta["source"] = "contaminated source"
        meta["source_records"] = []
        meta["eval_ban"] = []
        return (
            [{"text": " Same\n  Dialogue ", "answer": "train", "label": "generation"}],
            [{"text": "same dialogue", "answer": "test", "label": "generation"}],
        )

    _install_lightweight_dependencies(monkeypatch, tmp_path, acquire)

    with pytest.raises(ValueError, match="normalized train/test overlap"):
        eval_setup.eval_setup_node(_state())


def test_eval_restriction_text_does_not_claim_unimplemented_repo_ban():
    root = Path(__file__).parents[2]
    curate_source = (root / "agent" / "nodes" / "curate.py").read_text()
    state_source = (root / "agent" / "state.py").read_text()

    assert "no train data from these eval sources" not in curate_source
    assert "acquisition/rebuild is FORBIDDEN" not in state_source


def test_shared_dataset_uses_explicit_eval_ban_file(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    train = [{"text": "train", "answer": "a", "label": "generation"}]
    test = [{"text": "test", "answer": "b", "label": "generation"}]
    records = [
        {"kind": "hf", "id": "source", "split": "train", "role": "curriculum"},
        {"kind": "hf", "id": "source", "split": "test", "role": "eval"},
    ]
    eval_ban = [records[1]]
    writer = getattr(prepare_shared_dataset, "_write_shared_bundle", None)
    assert callable(writer)
    writer(
        shared,
        train,
        test,
        plan={"task_type": "generation", "benchmark": "SAMSum"},
        difficulty=None,
        meta={
            "source": "local SAMSum",
            "source_records": records,
            "eval_ban": eval_ban,
        },
    )
    monkeypatch.setenv("SLM_SHARED_DATASET_DIR", str(shared))
    monkeypatch.setattr(eval_setup, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    result = eval_setup.eval_setup_node(_state())

    assert result["data_sources"] == records
    assert result["eval_source_ban"] == eval_ban
    assert (shared / "manifest.json").exists()
    assert (shared / "checksums.sha256").exists()


def test_shared_dataset_preparer_persists_eval_ban_metadata():
    root = Path(__file__).parents[2]
    source = (root / "scripts" / "prepare_shared_dataset.py").read_text()

    assert '"eval_ban.json"' in source
    assert '"manifest.json"' in source
    assert '"checksums.sha256"' in source


@pytest.mark.parametrize(
    ("task_type", "row"),
    [
        (
            "code_generation",
            {
                "input_output": {
                    "inputs": ["x\n"],
                    "outputs": ["x\n"],
                },
                "execution_mode": "stdin",
                "runner_compatible": True,
                "label": "code_generation",
            },
        ),
        (
            "generation",
            {"answer": "answer", "label": "generation"},
        ),
    ],
)
def test_shared_preparer_uses_dynamic_800_eval_slices(task_type, row):
    examples = [
        {"text": f"example {index}", **row}
        for index in range(800)
    ]
    plan = {
        "task_type": task_type,
        "multi_label": False,
        "schema": None,
        "multilingual": False,
    }

    eval_set = prepare_shared_dataset._build_requested_eval_set(
        examples,
        plan,
        800,
    )

    assert len(eval_set.all) == 800


def test_shared_dataset_loader_rejects_missing_integrity_files(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "train.jsonl").write_text(
        json.dumps({"text": "train", "answer": "a", "label": "generation"}) + "\n"
    )
    (shared / "test.jsonl").write_text(
        json.dumps({"text": "test", "answer": "b", "label": "generation"}) + "\n"
    )

    with pytest.raises(ValueError, match="manifest.json.*checksums.sha256"):
        eval_setup._load_shared_dataset(str(shared))


def test_shared_dataset_loader_rejects_checksum_tampering(tmp_path):
    writer = getattr(prepare_shared_dataset, "_write_shared_bundle", None)
    assert callable(writer)
    shared = tmp_path / "shared"
    writer(
        shared,
        [{"text": "train", "answer": "a", "label": "generation"}],
        [{"text": "test", "answer": "b", "label": "generation"}],
        plan={"task_type": "generation"},
        difficulty=None,
        meta={"source_records": [], "eval_ban": []},
    )
    with (shared / "train.jsonl").open("a") as handle:
        handle.write(json.dumps({"text": "tampered", "answer": "x"}) + "\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        eval_setup._load_shared_dataset(str(shared))


def test_shared_dataset_loader_validates_row_schema_after_integrity(tmp_path):
    writer = getattr(prepare_shared_dataset, "_write_shared_bundle", None)
    assert callable(writer)
    shared = tmp_path / "shared"
    writer(
        shared,
        [{"text": "train", "answer": "a", "label": "generation"}],
        [{"text": "test", "answer": "b", "label": "generation"}],
        plan={"task_type": "generation"},
        difficulty=None,
        meta={},
    )
    (shared / "test.jsonl").write_text(
        json.dumps({"text": "test", "label": "generation"}) + "\n"
    )
    manifest_path = shared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["integrity"]["files"]["test.jsonl"] = sha256_file(shared / "test.jsonl")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checksum_filenames = [
        line.split("  ", 1)[1]
        for line in (shared / "checksums.sha256").read_text().splitlines()
    ]
    (shared / "checksums.sha256").write_text(
        "".join(
            f"{sha256_file(shared / filename)}  {filename}\n"
            for filename in checksum_filenames
        )
    )

    with pytest.raises(ValueError, match="missing schema fields.*answer"):
        eval_setup._load_shared_dataset(str(shared))


def test_shared_dataset_writer_rejects_schema_and_normalized_overlap(tmp_path):
    writer = getattr(prepare_shared_dataset, "_write_shared_bundle", None)
    assert callable(writer)

    with pytest.raises(ValueError, match="missing schema fields.*answer"):
        writer(
            tmp_path / "bad-schema",
            [{"text": "train", "label": "generation"}],
            [{"text": "test", "answer": "b", "label": "generation"}],
            plan={"task_type": "generation"},
            difficulty=None,
            meta={},
        )

    with pytest.raises(ValueError, match="normalized train/eval overlap"):
        writer(
            tmp_path / "overlap",
            [{"text": " Same\n Dialogue ", "answer": "a", "label": "generation"}],
            [{"text": "same dialogue", "answer": "b", "label": "generation"}],
            plan={"task_type": "generation"},
            difficulty=None,
            meta={},
        )
