import builtins
import json
import sys
from types import SimpleNamespace

import pytest

from data.eval_set import build_eval_set
from data.loaders import web_acquire


@pytest.mark.parametrize(
    ("benchmark", "expected"),
    [
        ("BC5CDR", "bc5cdr"),
        ("Biomedical NER (BC5CDR chemical-disease)", "bc5cdr"),
        ("GSM8K", "gsm8k"),
        ("grade-school math benchmark: GSM8K main", "gsm8k"),
        ("SAMSum", "samsum"),
        ("abstractive dialogue summarization — SAMSum dataset", "samsum"),
        ("RouterBench", "routerbench"),
    ],
)
def test_exact_and_composite_benchmark_aliases(benchmark, expected):
    assert web_acquire._resolve_benchmark_key(benchmark) == expected


def test_stage0_bc5cdr_loader_preserves_ner_schema(monkeypatch):
    calls = []

    def fake_load_dataset(hf_id, config=None, split=None, **_kwargs):
        calls.append((hf_id, config, split))
        if hf_id != "tner/bc5cdr":
            raise AssertionError(hf_id)
        disease = "fever" if str(split).startswith("train") else "rash"
        return [{"tokens": ["Aspirin", "causes", disease], "tags": [1, 0, 2]}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    meta = {}

    train, test = web_acquire.load_benchmark_dataset(
        {"benchmark": "Biomedical NER (BC5CDR)", "task": "ner_bc5cdr"},
        max_train=2,
        max_test=1,
        meta=meta,
    )

    assert train == [
        {
            "text": "Aspirin causes fever",
            "entities": [
                {"text": "Aspirin", "type": "Chemical"},
                {"text": "fever", "type": "Disease"},
            ],
        }
    ]
    assert test[0]["entities"][1]["text"] == "rash"
    assert calls == [
        ("tner/bc5cdr", None, "train[:2]"),
        ("tner/bc5cdr", None, "test[:1]"),
    ]
    assert meta["source_records"][1]["split"] == "test"


def test_stage0_gsm8k_loader_preserves_gold_cot(monkeypatch):
    def fake_load_dataset(hf_id, config=None, split=None, **_kwargs):
        assert (hf_id, config) == ("openai/gsm8k", "main")
        final = "7" if str(split).startswith("train") else "8"
        return [{"question": f"Compute {final}", "answer": f"Reasoning {final}.\n#### {final}"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    train, test = web_acquire.load_benchmark_dataset(
        {"benchmark": "grade-school math (GSM8K)", "task": "gsm8k"},
        max_train=2,
        max_test=1,
    )

    assert train[0] == {
        "text": "Compute 7",
        "answer": "7",
        "cot_reasoning": "Reasoning 7.",
        "label": "math_reasoning",
    }
    assert test[0]["answer"] == "8"


def test_stage0_samsum_loader_uses_accessible_mirror(monkeypatch):
    calls = []

    def fake_load_dataset(hf_id, config=None, split=None, **_kwargs):
        calls.append((hf_id, config, split))
        if hf_id == "samsum":
            raise RuntimeError("legacy source unavailable")
        assert hf_id == "knkarthick/samsum"
        suffix = "train" if str(split).startswith("train") else "test"
        return [{"dialogue": f"A: {suffix}", "summary": f"Summary {suffix}"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    meta = {}

    train, test = web_acquire.load_benchmark_dataset(
        {"benchmark": "SAMSum dialogue summarization", "task": "dialogsum"},
        max_train=2,
        max_test=1,
        meta=meta,
    )

    assert train == [{"text": "A: train", "answer": "Summary train", "label": "generation"}]
    assert test == [{"text": "A: test", "answer": "Summary test", "label": "generation"}]
    assert [call[0] for call in calls] == ["samsum", "knkarthick/samsum", "knkarthick/samsum"]
    assert "knkarthick/samsum" in meta["source"]


def test_stage0_removes_normalized_train_test_overlap(monkeypatch):
    logs = []

    def fake_load_dataset(hf_id, config=None, split=None, **_kwargs):
        if hf_id == "samsum":
            raise RuntimeError("legacy source unavailable")
        if str(split).startswith("train"):
            return [
                {"dialogue": "A: unique", "summary": "Unique."},
                {"dialogue": " A: DUPLICATE\n   text ", "summary": "Duplicate train."},
            ]
        return [{"dialogue": "a: duplicate text", "summary": "Duplicate test."}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    train, test = web_acquire.load_benchmark_dataset(
        {"benchmark": "SAMSum", "task": "dialogsum"},
        max_train=2,
        max_test=1,
        log=logs.append,
    )

    assert [row["text"] for row in train] == ["A: unique"]
    assert [row["text"] for row in test] == ["a: duplicate text"]
    assert any("normalized overlap" in message for message in logs)


def _write_local_bundle(root, name, task, train, test, schema_version=1):
    bundle = root / name
    bundle.mkdir()
    source_id = {
        "bc5cdr": "tner/bc5cdr",
        "gsm8k": "openai/gsm8k",
        "samsum": "knkarthick/samsum",
        "emotion": "dair-ai/emotion",
    }[name]
    config = {"gsm8k": "main"}.get(name)
    manifest = {
        "schema_version": schema_version,
        "name": name,
        # The REGISTRY task this bundle supplies rows for. `_local_manifest_match` compares it to
        # the run's task, so a bundle naming a channel several tasks shared is unreachable.
        "task": task,
        "hf_id": source_id,
        "config": config,
        "source_url": f"https://huggingface.co/datasets/{source_id}",
        "source_splits": {"train": "train", "test": "test"},
        "labels": sorted({
            str(row.get("label"))
            for row in train
            if row.get("label") is not None
        }),
        "counts": {"train": len(train), "test": len(test), "total": len(train) + len(test)},
        "provenance": {
            "official_splits_only": True,
            "records": [
                {"kind": "hf", "id": source_id, "config": config, "split": "train", "role": "curriculum"},
                {"kind": "hf", "id": source_id, "config": config, "split": "test", "role": "eval"},
            ],
        },
        "eval_ban": [
            {"kind": "hf", "id": source_id, "config": config, "split": "test", "role": "eval"}
        ],
        "overlap": {"field": "text", "exact_count": 0, "removed_from_train": 0},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    for split, rows in (("train", train), ("test", test)):
        (bundle / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


@pytest.mark.parametrize(
    ("name", "task", "benchmark", "train_row", "test_row"),
    [
        (
            "bc5cdr",
            "ner_bc5cdr",
            "Biomedical NER (BC5CDR)",
            {"text": "Aspirin helps", "entities": [{"text": "Aspirin", "type": "Chemical"}]},
            {"text": "Fever persists", "entities": [{"text": "Fever", "type": "Disease"}]},
        ),
        (
            "gsm8k",
            "gsm8k",
            "grade-school math GSM8K",
            {"text": "1+1?", "answer": "2", "cot_reasoning": "Add.", "label": "gsm8k"},
            {"text": "2+2?", "answer": "4", "cot_reasoning": "Add.", "label": "gsm8k"},
        ),
        (
            "samsum",
            "dialogsum",
            "SAMSum dialogue summarization",
            {"text": "A: hi", "answer": "A says hi.", "label": "generation"},
            {"text": "B: bye", "answer": "B says bye.", "label": "generation"},
        ),
    ],
)
def test_local_fallback_supports_task_schemas(
    tmp_path, monkeypatch, name, task, benchmark, train_row, test_row
):
    _write_local_bundle(tmp_path, name, task, [train_row], [test_row])
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))
    meta = {}

    train, test = web_acquire.load_local_dataset(
        {"benchmark": benchmark, "task": task},
        task,
        max_train=5,
        max_test=5,
        meta=meta,
    )

    assert train == [train_row]
    assert test == [test_row]
    assert meta["source_records"][0]["id"]
    assert meta["eval_ban"][0]["split"] == "test"


def test_unknown_benchmark_can_use_exact_task_label_match(
    tmp_path,
    monkeypatch,
):
    train_rows = [
        {"text": "joyful text", "label": "joy"},
        {"text": "sad text", "label": "sadness"},
    ]
    test_rows = [{"text": "other joy", "label": "joy"}]
    _write_local_bundle(
        tmp_path,
        "emotion",
        "clinc150",
        train_rows,
        test_rows,
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    loaded = web_acquire.load_local_dataset(
        {
            "benchmark": "custom emotion benchmark",
            "task": "clinc150",
            "labels": ["joy", "sadness"],
        },
        "clinc150",
        max_train=10,
        max_test=10,
    )

    assert loaded == (train_rows, test_rows)


def test_local_fallback_does_not_import_key_bearing_config(tmp_path, monkeypatch):
    train_row = {"text": "train dialogue", "answer": "summary", "label": "generation"}
    test_row = {"text": "test dialogue", "answer": "summary", "label": "generation"}
    _write_local_bundle(tmp_path, "samsum", "dialogsum", [train_row], [test_row])
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "config.config":
            raise AssertionError("local loading must not import config.config")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    train, test = web_acquire.load_local_dataset(
        {"benchmark": "SAMSum", "task": "dialogsum"},
        "dialogsum",
        max_train=5,
        max_test=5,
    )

    assert train == [train_row]
    assert test == [test_row]


def test_local_fallback_fails_closed_on_normalized_overlap(tmp_path, monkeypatch):
    train = {"text": " Duplicated\n  Dialogue ", "answer": "summary", "label": "generation"}
    test = {"text": "duplicated dialogue", "answer": "summary", "label": "generation"}
    _write_local_bundle(tmp_path, "samsum", "dialogsum", [train], [test])
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="normalized train/test text overlap"):
        web_acquire.load_local_dataset(
            {"benchmark": "SAMSum", "task": "dialogsum"},
            "dialogsum",
            max_train=5,
            max_test=5,
        )


def test_local_fallback_rejects_checksum_mismatch(tmp_path, monkeypatch):
    train = {"text": "train", "answer": "summary", "label": "generation"}
    test = {"text": "test", "answer": "summary", "label": "generation"}
    _write_local_bundle(tmp_path, "samsum", "dialogsum", [train], [test])
    bundle = tmp_path / "samsum"
    (bundle / "checksums.sha256").write_text(
        f"{'0' * 64}  train.jsonl\n"
        f"{'0' * 64}  test.jsonl\n"
        f"{'0' * 64}  manifest.json\n"
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="checksum mismatch"):
        web_acquire.load_local_dataset(
            {"benchmark": "SAMSum", "task": "dialogsum"},
            "dialogsum",
            max_train=5,
            max_test=5,
        )


def test_schema_v2_local_bundle_requires_checksum_sidecar(tmp_path, monkeypatch):
    train = {"text": "train", "answer": "summary", "label": "generation"}
    test = {"text": "test", "answer": "summary", "label": "generation"}
    _write_local_bundle(
        tmp_path, "samsum", "dialogsum", [train], [test], schema_version=2
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="schema-v2.*checksums.sha256"):
        web_acquire.load_local_dataset(
            {"benchmark": "SAMSum", "task": "dialogsum"},
            "dialogsum",
            max_train=5,
            max_test=5,
        )


def test_explicit_legacy_schema_v1_loads_without_sidecar_and_logs(tmp_path, monkeypatch):
    train = {"text": "train", "answer": "summary", "label": "generation"}
    test = {"text": "test", "answer": "summary", "label": "generation"}
    _write_local_bundle(
        tmp_path, "samsum", "dialogsum", [train], [test], schema_version=1
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))
    logs = []

    loaded_train, loaded_test = web_acquire.load_local_dataset(
        {"benchmark": "SAMSum", "task": "dialogsum"},
        "dialogsum",
        max_train=5,
        max_test=5,
        log=logs.append,
    )

    assert loaded_train == [train]
    assert loaded_test == [test]
    assert any("legacy schema-v1" in message and "without checksums" in message for message in logs)


def test_discovery_worker_returns_precise_hf_split_provenance(monkeypatch):
    captured = []

    class Queue:
        def put(self, value):
            captured.append(value)

    monkeypatch.setitem(
        sys.modules, "config.config", SimpleNamespace(EXA_API_KEY="mock-exa")
    )
    monkeypatch.setitem(
        sys.modules, "exa_py", SimpleNamespace(Exa=lambda api_key: object())
    )
    monkeypatch.setattr(
        web_acquire, "_exa_find_hf_dataset_ids",
        lambda *_args, **_kwargs: ["org/discovered"],
    )
    monkeypatch.setattr(
        web_acquire, "_peek_hf_dataset",
        lambda *_args, **_kwargs: (
            "cfg", ["train", "validation"], ["prompt", "answer"],
            {"prompt": "p", "answer": "a"}, {},
        ),
    )
    monkeypatch.setattr(
        web_acquire, "_llm_map_dataset",
        lambda *_args, **_kwargs: {
            "train_split": "train",
            "test_split": "validation",
            "question_col": "prompt",
            "answer_col": "answer",
        },
    )
    monkeypatch.setattr(
        web_acquire, "_materialize_from_mapping",
        lambda *_args, **_kwargs: (
            [{"text": "train", "answer": "a", "label": "generation"}],
            [{"text": "test", "answer": "b", "label": "generation"}],
        ),
    )

    web_acquire._discover_worker(
        {"benchmark": "Unknown", "task_name": "task", "labels": []},
        "description",
        "dialogsum",
        5,
        5,
        Queue(),
    )

    result = captured[-1]
    assert result["source_records"] == [
        {
            "kind": "hf", "id": "org/discovered", "config": "cfg",
            "split": "train", "url": "https://huggingface.co/datasets/org/discovered",
            "role": "curriculum",
        },
        {
            "kind": "hf", "id": "org/discovered", "config": "cfg",
            "split": "validation", "url": "https://huggingface.co/datasets/org/discovered",
            "role": "eval",
        },
    ]
    assert result["eval_ban"] == [result["source_records"][1]]


def test_discovery_parent_propagates_precise_metadata(monkeypatch):
    train = [{"text": "train", "answer": "a", "label": "generation"}]
    test = [{"text": "test", "answer": "b", "label": "generation"}]
    records = [
        {"kind": "hf", "id": "org/discovered", "config": "cfg",
         "split": "train", "role": "curriculum"},
        {"kind": "hf", "id": "org/discovered", "config": "cfg",
         "split": "test", "role": "eval"},
    ]

    def fake_worker(_plan, _description, _task, _max_train, _max_test, queue):
        queue.put({
            "train": train,
            "test": test,
            "source": "agentic org/discovered",
            "source_records": records,
            "eval_ban": [records[1]],
            "logs": [],
        })

    def no_fork(_method):
        raise ValueError("force inline")

    monkeypatch.setattr(web_acquire, "_discover_worker", fake_worker)
    monkeypatch.setattr("multiprocessing.get_context", no_fork)
    meta = {}

    result = web_acquire.discover_and_load_hf_dataset(
        {"benchmark": "Unknown"},
        "description",
        "dialogsum",
        5,
        5,
        meta=meta,
    )

    assert result == (train, test)
    assert meta == {
        "source": "agentic org/discovered",
        "source_records": records,
        "eval_ban": [records[1]],
    }


def test_agent_first_flag_falls_back_to_local_before_stage0(monkeypatch):
    order = []
    expected = ([{"text": "train", "answer": "gold"}], [{"text": "test", "answer": "gold"}])

    def discover(*_args, **_kwargs):
        order.append("agentic")
        return None

    def stage0(*_args, **_kwargs):
        order.append("stage0")
        raise AssertionError("Stage-0 must not run when local succeeds")

    def local(*_args, **_kwargs):
        order.append("local")
        return expected

    monkeypatch.setenv("SLM_AGENT_FIRST_DATASET_DISCOVERY", "1")
    monkeypatch.setattr(web_acquire, "discover_and_load_hf_dataset", discover)
    monkeypatch.setattr(web_acquire, "load_benchmark_dataset", stage0)
    monkeypatch.setattr(web_acquire, "load_local_dataset", local)

    result = web_acquire.acquire_dataset(
        {"benchmark": "SAMSum", "task": "dialogsum"},
        description="readiness test",
    )

    assert result == expected
    assert order == ["agentic", "local"]


def test_agent_first_uses_stage0_only_after_local_miss(monkeypatch):
    order = []
    expected = ([{"text": "train", "answer": "gold"}], [{"text": "test", "answer": "gold"}])

    def discover(*_args, **_kwargs):
        order.append("agentic")
        return None

    def local(*_args, **_kwargs):
        order.append("local")
        return None

    def stage0(*_args, **_kwargs):
        order.append("stage0")
        return expected

    monkeypatch.setenv("SLM_AGENT_FIRST_DATASET_DISCOVERY", "1")
    monkeypatch.setattr(web_acquire, "discover_and_load_hf_dataset", discover)
    monkeypatch.setattr(web_acquire, "load_local_dataset", local)
    monkeypatch.setattr(web_acquire, "load_benchmark_dataset", stage0)

    result = web_acquire.acquire_dataset(
        {"benchmark": "SAMSum", "task": "dialogsum"},
        description="readiness test",
    )

    assert result == expected
    assert order == ["agentic", "local", "stage0"]


def test_agent_first_unusable_discovery_falls_back_to_local(monkeypatch):
    """A discovered dataset in which NO row carries the fields the task needs is unusable.

    That is the one whole-source rejection that survives. The three others this test used to
    parametrize over — an internal train/test overlap, a column the mapping does not want, and a
    single-class slice — were removed from `_validate_discovered_splits` on 2026-08-19, because
    each rejected an entire dataset for something that should only have cost it some rows. The
    overlap one in particular was self-inflicted: `_materialize_from_mapping` sliced train and test
    from the front of the SAME underlying split, so the overlap it "found" was always exactly
    `max_test`, which is why every xLAM mirror was rejected.
    """
    order = []
    # Neither train row carries the `answer` that `dialogsum` requires.
    discovered = (
        [{"text": "train row", "label": "generation"}],
        [{"text": "test row", "answer": "a summary", "label": "generation"}],
    )
    local_rows = (
        [{"text": "local train", "answer": "a summary", "label": "generation"}],
        [{"text": "local test", "answer": "another summary", "label": "generation"}],
    )

    def discover(*_args, **_kwargs):
        order.append("agentic")
        return discovered

    def local(*_args, meta=None, **_kwargs):
        order.append("local")
        if meta is not None:
            meta.clear()
            meta["source"] = "local valid SAMSum"
        return local_rows

    def unexpected_stage0(*_args, **_kwargs):
        raise AssertionError("valid local fallback must stop acquisition")

    monkeypatch.setenv("SLM_AGENT_FIRST_DATASET_DISCOVERY", "1")
    monkeypatch.setattr(web_acquire, "discover_and_load_hf_dataset", discover)
    monkeypatch.setattr(web_acquire, "load_local_dataset", local)
    monkeypatch.setattr(web_acquire, "load_benchmark_dataset", unexpected_stage0)
    meta = {"source": "stale"}

    result = web_acquire.acquire_dataset(
        {"benchmark": "SAMSum", "task": "dialogsum"},
        description="readiness test",
        meta=meta,
    )

    assert result == local_rows
    assert order == ["agentic", "local"]
    assert meta == {"source": "local valid SAMSum"}


def test_default_acquisition_uses_local_without_paid_discovery(monkeypatch):
    order = []
    expected = ([{"text": "train", "answer": "gold"}], [{"text": "test", "answer": "gold"}])

    def local(*_args, **_kwargs):
        order.append("local")
        return expected

    def unexpected(*_args, **_kwargs):
        raise AssertionError("remote or paid discovery must not run when local succeeds")

    monkeypatch.delenv("SLM_AGENT_FIRST_DATASET_DISCOVERY", raising=False)
    monkeypatch.setattr(web_acquire, "load_local_dataset", local)
    monkeypatch.setattr(web_acquire, "load_benchmark_dataset", unexpected)
    monkeypatch.setattr(web_acquire, "discover_and_load_hf_dataset", unexpected)

    result = web_acquire.acquire_dataset(
        {"benchmark": "SAMSum", "task": "dialogsum"},
        description="normal run",
    )

    assert result == expected
    assert order == ["local"]


def test_default_acquisition_tries_stage0_before_paid_after_local_miss(monkeypatch):
    order = []
    expected = ([{"text": "train", "answer": "gold"}], [{"text": "test", "answer": "gold"}])

    def local(*_args, **_kwargs):
        order.append("local")
        return None

    def stage0(*_args, **_kwargs):
        order.append("stage0")
        return expected

    def unexpected_discovery(*_args, **_kwargs):
        raise AssertionError("paid discovery must not run after Stage-0 succeeds")

    monkeypatch.delenv("SLM_AGENT_FIRST_DATASET_DISCOVERY", raising=False)
    monkeypatch.setattr(web_acquire, "load_local_dataset", local)
    monkeypatch.setattr(web_acquire, "load_benchmark_dataset", stage0)
    monkeypatch.setattr(web_acquire, "discover_and_load_hf_dataset", unexpected_discovery)

    result = web_acquire.acquire_dataset(
        {"benchmark": "SAMSum", "task": "dialogsum"},
        description="normal run",
    )

    assert result == expected
    assert order == ["local", "stage0"]


# --------------------------------------------------------------------------
# `mine_additional_real_rows` — rung 2 of the mining ladder
# --------------------------------------------------------------------------
#
# Acceptance here is PER ROW, not per source (2026-08-19). A discovered dataset is not thrown away
# for carrying rows we cannot use: unusable rows are dropped and the rest of the source is kept, and
# the source is refused only when nothing survives. See docs/interventions.md section 4.1.

ROUTER_LABELS = {"local", "route"}
DISCOVERED_RECORDS = [
    {"kind": "hf", "id": "fixture/src", "split": "train", "role": "curriculum"},
]


def _mine_from_a_discovered_source(monkeypatch, train):
    """Run the mining entry point with rung 2 returning `train` and nothing else reachable."""
    def discovery(_plan, _description, _task, _max_train, _max_test, log=print, meta=None):
        if meta is not None:
            meta.update({"source": "fixture", "source_records": DISCOVERED_RECORDS})
        return train, [{"text": "held out", "label": "local"}]

    monkeypatch.setattr(web_acquire, "discover_and_load_hf_dataset", discovery)
    logs: list[str] = []
    rows, report = web_acquire.mine_additional_real_rows(
        task_plan={"task_name": "routing", "benchmark": "unknown",
                   "labels": sorted(ROUTER_LABELS)},
        description="route or answer locally",
        task="routerbench",
        existing_rows=[{"text": "seed", "label": "local"}],
        eval_rows=[],
        eval_source_ban=[],
        requested_rows=5,
        query_variant=0,
        label_space=ROUTER_LABELS,
        log=logs.append,
    )
    return rows, report, " ".join(logs)


def test_asking_for_no_rows_is_a_no_op_rather_than_a_crash():
    """The cheapest possible call into mining: nothing is requested, so nothing may be attempted and
    no provider may be touched. Asserted because it is the shortest path through the function, so a
    name that is read before any argument is — a task spec resolved from a local that was never
    bound, say — surfaces here and nowhere cheaper."""
    rows, report = web_acquire.mine_additional_real_rows(
        task_plan={"task_name": "routing", "benchmark": "routerbench"},
        description="route or answer locally",
        task="routerbench",
        existing_rows=[{"text": "seed", "label": "local"}],
        eval_rows=[],
        eval_source_ban=[],
        requested_rows=0,
        query_variant=0,
        label_space=ROUTER_LABELS,
        log=lambda _message: None,
    )
    assert rows == []
    assert report["status"] in ("not_requested", "no_novelty")


def test_a_mined_source_keeps_its_good_rows_and_drops_the_foreign_ones(monkeypatch):
    """RouterBench is a two-class task, and rows labelled `cloud` cannot be scored against a frozen
    eval set that has no such class — so they are dropped, and the drop is logged with a count per
    rejected label because a silent one is indistinguishable from a source that simply had few rows.

    The rest of the source is KEPT. The original guard rejected the whole source instead, which was
    the right instinct and the wrong remedy: B259 saw four hallucinated classes enter a two-class
    RouterBench run and cost ~1,166 quality-control deletions per rebuild for the remainder of it,
    but a dataset whose column mapping got most rows right is still worth most of its rows.
    """
    rows, report, logs = _mine_from_a_discovered_source(monkeypatch, [
        {"text": "novel one", "label": "local"},
        {"text": "novel two", "label": "route"},
        {"text": "novel three", "label": "cloud"},
    ])

    assert [row["text"] for row in rows] == ["novel one", "novel two"]
    assert report["status"] == "novel"
    assert report["rejected_sources"] == 0
    assert "'cloud'x1" in logs, "the log must name the class it dropped, and how often"


def test_a_mined_source_is_refused_only_when_no_row_survives(monkeypatch):
    """"Not one row carries a label of ours" is the honest reading of "this is not our task's data",
    and it is the only condition under which a whole source is still thrown away (B259)."""
    rows, report, logs = _mine_from_a_discovered_source(monkeypatch, [
        {"text": "novel one", "label": "cloud"},
        {"text": "novel two", "label": "on_device"},
    ])

    assert rows == []
    assert report["rejected_sources"] == 1
    assert "REJECTED" in logs
