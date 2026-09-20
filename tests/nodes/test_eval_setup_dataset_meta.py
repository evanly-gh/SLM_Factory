import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.nodes.cold_start import eval_setup
from data.loaders.dataset_integrity import sha256_file
from scripts import prepare_shared_dataset


def _state():
    return {
        "task": "dialogsum",
        "task_plan": {
            "task": "dialogsum",
            "benchmark": "DialogSum",
            "multi_label": False,
            "schema": None,
            "multilingual": False,
        },
        "description": "dialogue summarization",
        "data_sources": [],
        "eval_source_ban": [],
        "feasible_models": [],
    }


def _install_lightweight_dependencies(monkeypatch, tmp_path):
    # Cold start now asks the orchestrator to write a task brief from the real rows it loaded. That
    # is a live Anthropic call and is covered on its own in tests/test_task_brief.py; here it is
    # stubbed, because these tests are about which dataset SPLITS get banned from the curriculum.
    monkeypatch.setitem(
        sys.modules,
        "agent.task_brief",
        SimpleNamespace(
            build_task_brief=lambda spec, rows, **_kwargs: {
                "summary": spec.title, "output_contract": "stubbed",
                "failure_modes": [], "source": "test-stub",
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.nodes.test_agent",
        SimpleNamespace(label_difficulty=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(eval_setup, "ARTIFACTS_DIR", str(tmp_path))


def _pin_loader(monkeypatch, task, loader):
    """Give `task` a loader that returns fixture rows.

    The loader used to live in `_named_benchmark_loaders`, a dict kept in lockstep by hand with
    `NAMED_BENCHMARK_TASK_TYPES` — two of the five side registries the task specs replaced. It is
    now `TaskSpec.load`, so the loader is pinned by replacing the spec itself.
    """
    import dataclasses

    from tasks import TASKS, get_task

    monkeypatch.setitem(
        TASKS, task, dataclasses.replace(get_task(task), load=loader)
    )


def test_eval_setup_records_only_declared_eval_split_bans(tmp_path, monkeypatch, capsys):
    """Provenance records BOTH official splits; only the held-out one is banned as a source.

    The two lists are built from different keys (`source_records` vs `eval_ban`) precisely so that
    recording where data came from cannot be mistaken for forbidding it — the curriculum is
    supposed to be drawn from the train split of the very dataset the eval set comes from.

    The scenario moved: `acquire_meta` used to be filled by `web_acquire.acquire_dataset` on the
    autonomous path, which is gone. `_load_named_benchmark` now fills it from the task spec, so the
    records name the registry task rather than the hub repo.
    """
    _pin_loader(
        monkeypatch, "dialogsum",
        lambda max_train, max_test, log=print: (
            [{"text": "train dialogue", "answer": "train summary",
              "references": ["train summary"]}],
            [{"text": "test dialogue", "answer": "test summary",
              "references": ["test summary"]}],
        ),
    )
    _install_lightweight_dependencies(monkeypatch, tmp_path)

    result = eval_setup.eval_setup_node(_state())

    train_record = {
        "kind": "hf", "id": "dialogsum", "split": "train", "role": "curriculum"
    }
    test_record = {"kind": "hf", "id": "dialogsum", "split": "test", "role": "eval"}
    assert result["data_sources"] == [train_record, test_record]
    assert result["eval_source_ban"] == [test_record]
    output = capsys.readouterr().out
    assert "eval source restrictions recorded" in output
    assert "locked" not in output


def test_eval_setup_does_not_invent_eval_bans_from_provenance(tmp_path, monkeypatch):
    """A source that declares no eval restriction gets none — the ban is never inferred.

    Inferring it from `source_records` would ban the whole dataset the curriculum is drawn from.
    Asserted on the shared-bundle path because that is the one remaining path whose provenance and
    ban are supplied separately, and so the only one where the two can still disagree.
    """
    records = [
        {"kind": "hf", "id": "source", "split": "train", "role": "curriculum"},
        {"kind": "hf", "id": "source", "split": "test", "role": "eval"},
    ]
    shared = tmp_path / "shared"
    prepare_shared_dataset._write_shared_bundle(
        shared,
        [{"text": "train", "answer": "a", "references": ["a"]}],
        [{"text": "test", "answer": "b", "references": ["b"]}],
        task="dialogsum",
        plan={"task": "dialogsum"},
        difficulty=None,
        meta={"source": "source without declared restriction", "source_records": records},
    )
    monkeypatch.setenv("SLM_SHARED_DATASET_DIR", str(shared))
    _install_lightweight_dependencies(monkeypatch, tmp_path / "artifacts")

    result = eval_setup.eval_setup_node(_state())

    assert result["data_sources"] == records
    assert result["eval_source_ban"] == []


def test_eval_setup_enforces_normalized_train_test_separation(tmp_path, monkeypatch):
    """The Layer-1 firewall is independent of Stage-0, which is the point of having both.

    Stage-0 (`remove_normalized_train_overlap`, inside `_load_named_benchmark`) normally drops the
    offending train row before this check ever sees it, so Stage-0 is disabled here to prove the
    second layer is not merely echoing the first. A loader that returns an overlapping pair — a new
    loader, or a Stage-0 regression — must stop the run rather than train on held-out text.
    """
    _pin_loader(
        monkeypatch, "dialogsum",
        lambda max_train, max_test, log=print: (
            [{"text": " Same\n  Dialogue ", "answer": "train", "references": ["train"]}],
            [{"text": "same dialogue", "answer": "test", "references": ["test"]}],
        ),
    )
    monkeypatch.setattr(
        eval_setup, "remove_normalized_train_overlap", lambda train, test: (train, 0)
    )
    _install_lightweight_dependencies(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="normalized train/test overlap"):
        eval_setup.eval_setup_node(_state())


# Official benchmark splits are not guaranteed disjoint — CLINC150 ships "what's your designation"
# in *both* splits under two different intents — so without Stage-0 the Layer-1 firewall above
# turns a source-data quirk into a hard run-ending raise. Decontaminate first: drop the train row,
# never the official test row.


def test_named_benchmark_path_removes_train_rows_overlapping_official_test(monkeypatch, capsys):
    leaked = {"text": " What's Your  Designation ", "label": "what_is_your_name"}
    kept = {"text": "set an alarm for 6am", "label": "alarm"}
    test_rows = [{"text": "what's your designation", "label": "user_name"}]

    _pin_loader(
        monkeypatch, "clinc150",
        lambda max_train, max_test, log=print: ([leaked, kept], test_rows),
    )

    meta: dict = {}
    train_examples, test_examples = eval_setup._load_named_benchmark(
        "clinc150", {"task": "clinc150"}, meta
    )

    assert train_examples == [kept], "the overlapping train row must be dropped"
    assert test_examples == test_rows, "official test rows must never be modified"
    assert meta["overlap_removed_from_train"] == 1
    assert "removed 1 train row" in capsys.readouterr().out


def test_named_benchmark_path_reports_zero_removal_when_splits_are_clean(monkeypatch):
    train_rows = [{"text": "set an alarm", "label": "alarm"}]
    test_rows = [{"text": "what time is it", "label": "time"}]

    _pin_loader(
        monkeypatch, "clinc150",
        lambda max_train, max_test, log=print: (train_rows, test_rows),
    )

    meta: dict = {}
    train_examples, test_examples = eval_setup._load_named_benchmark(
        "clinc150", {"task": "clinc150"}, meta
    )

    assert train_examples == train_rows
    assert test_examples == test_rows
    assert meta["overlap_removed_from_train"] == 0


def test_named_benchmark_path_refuses_a_mismatched_run_task(monkeypatch):
    """The run's task and SLM_BENCHMARK_TASK must agree. When several tasks shared a channel,
    loading one task's data into another's run was undetectable."""
    _pin_loader(
        monkeypatch, "clinc150",
        lambda max_train, max_test, log=print: ([{"text": "t", "label": "a"}], []),
    )

    with pytest.raises(ValueError, match="does not match the run's task"):
        eval_setup._load_named_benchmark("clinc150", {"task": "routerbench"}, {})


def test_eval_restriction_text_does_not_claim_unimplemented_repo_ban():
    root = Path(__file__).parents[2]
    curate_source = (root / "agent" / "nodes" / "curate.py").read_text()
    state_source = (root / "agent" / "state.py").read_text()

    assert "no train data from these eval sources" not in curate_source
    assert "acquisition/rebuild is FORBIDDEN" not in state_source


def test_shared_dataset_uses_explicit_eval_ban_file(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    train = [{"text": "train", "answer": "a", "references": ["a"]}]
    test = [{"text": "test", "answer": "b", "references": ["b"]}]
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
        task="dialogsum",
        plan={"task": "dialogsum", "benchmark": "DialogSum"},
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
    ("task", "row"),
    [
        ("dialogsum", {"answer": "answer", "references": ["answer"]}),
        ("xlam_bfcl", {"answer": "[]", "label": "function_call"}),
    ],
)
def test_shared_preparer_uses_dynamic_800_eval_slices(task, row):
    """The frozen eval set must be the size the runs that consume it would have built themselves.

    The preparer used to build it through its own `_build_requested_eval_set` wrapper, which took a
    plan and its abstract `task_type`; it now calls `build_eval_set` with the registry task and the
    same `_eval_target` clamp the pipeline uses, so the two cannot size differently.
    """
    from agent.nodes.cold_start.eval_setup import _eval_target
    from data.eval_set import build_eval_set

    examples = [{"text": f"example {index}", **row} for index in range(800)]

    eval_set = build_eval_set(examples, task=task, target=_eval_target(800))

    assert len(eval_set.all) == 800
    assert eval_set.task == task


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
        [{"text": "train", "answer": "a", "references": ["a"]}],
        [{"text": "test", "answer": "b", "references": ["b"]}],
        task="dialogsum",
        plan={"task": "dialogsum"},
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
        [{"text": "train", "answer": "a", "references": ["a"]}],
        [{"text": "test", "answer": "b", "references": ["b"]}],
        task="dialogsum",
        plan={"task": "dialogsum"},
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
            [{"text": "test", "answer": "b", "references": ["b"]}],
            task="dialogsum",
            plan={"task": "dialogsum"},
            difficulty=None,
            meta={},
        )

    # Schema-complete rows, so the overlap check is the thing that fires rather than the schema
    # check firing first and masking it.
    with pytest.raises(ValueError, match="normalized train/eval overlap"):
        writer(
            tmp_path / "overlap",
            [{"text": " Same\n Dialogue ", "answer": "a", "references": ["a"]}],
            [{"text": "same dialogue", "answer": "b", "references": ["b"]}],
            task="dialogsum",
            plan={"task": "dialogsum"},
            difficulty=None,
            meta={},
        )
