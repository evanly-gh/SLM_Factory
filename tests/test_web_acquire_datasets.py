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
        ("APPS", "apps"),
        ("APPS introductory Python benchmark", "apps"),
        ("codeparrot/apps", "apps"),
        ("MBPP", "mbpp"),
        ("Python code generation (HumanEval / MBPP)", "mbpp"),
        ("SAMSum", "samsum"),
        ("abstractive dialogue summarization — SAMSum dataset", "samsum"),
        ("UCI SMS Spam Collection", "sms_spam"),
        ("Financial PhraseBank", "fpb"),
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
        {"benchmark": "Biomedical NER (BC5CDR)", "task_type": "NER"},
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
        {"benchmark": "grade-school math (GSM8K)", "task_type": "math_reasoning"},
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


def test_stage0_mbpp_loader_preserves_code_and_test_list(monkeypatch):
    def fake_load_dataset(hf_id, config=None, split=None, **_kwargs):
        assert (hf_id, config) == ("google-research-datasets/mbpp", "sanitized")
        suffix = "train" if str(split).startswith("train") else "test"
        return [
            {
                "task_id": 1,
                "prompt": f"Write {suffix}",
                "code": f"def {suffix}():\n    return 1",
                "test_imports": [],
                "test_list": [f"assert {suffix}() == 1"],
            }
        ]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    train, test = web_acquire.load_benchmark_dataset(
        {"benchmark": "HumanEval / MBPP Python", "task_type": "code_generation"},
        max_train=2,
        max_test=1,
    )

    assert train[0]["answer"] == train[0]["code"]
    assert train[0]["test_list"] == ["assert train() == 1"]
    assert test[0]["text"] == "Write test"


def test_stage0_apps_loader_preserves_both_execution_schemas(monkeypatch):
    calls = []

    def fake_load_dataset(hf_id, config=None, split=None, **kwargs):
        calls.append((hf_id, config, split, kwargs))
        assert hf_id == "json"
        is_train = str(split).startswith("train")
        fn_name = "add" if is_train else None
        return [
            {
                "id": 1 if is_train else 2,
                "question": "Add two integers." if is_train else "Echo input.",
                "solutions": (
                    json.dumps(
                        [
                            "class Solution:\n"
                            "    def add(self, a, b):\n"
                            "        return a + b"
                        ]
                    )
                    if is_train
                    else ""
                ),
                "input_output": json.dumps(
                    {
                        **({"fn_name": fn_name} if fn_name else {}),
                        "inputs": ["[2, 3]"] if is_train else ["hello\n"],
                        "outputs": ["5"] if is_train else ["hello\n"],
                    }
                ),
                "difficulty": "introductory",
                "starter_code": (
                    "class Solution:\n"
                    "    def add(self, a, b):\n"
                    "        pass"
                    if is_train
                    else ""
                ),
                "url": "",
            }
        ]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    meta = {}

    train, test = web_acquire.load_benchmark_dataset(
        {
            "benchmark": "APPS introductory",
            "task_type": "code_generation",
        },
        max_train=2,
        max_test=1,
        meta=meta,
    )

    assert train[0]["execution_mode"] == "call_based"
    assert train[0]["fn_name"] == "add"
    assert train[0]["answer"] == train[0]["solutions"][0]
    assert test[0]["execution_mode"] == "stdin"
    assert test[0]["input_output"]["inputs"] == ["hello\n"]
    assert "answer" not in test[0]
    assert all(call[1] is None for call in calls)
    assert all(call[3]["streaming"] is True for call in calls)
    assert all(
        "21e74ddf8de1a21436da12e3e653065c5213e9d1"
        in next(iter(call[3]["data_files"].values()))
        for call in calls
    )
    assert meta["source_records"][0]["id"] == "codeparrot/apps"
    assert meta["source_records"][0]["config"] == "introductory"
    assert (
        meta["source_records"][0]["revision"]
        == "21e74ddf8de1a21436da12e3e653065c5213e9d1"
    )
    assert meta["eval_ban"][0]["split"] == "test"


def test_stage0_apps_filters_before_800_row_test_cap(monkeypatch):
    from types import SimpleNamespace

    def fake_load_dataset(hf_id, config=None, split=None, **_kwargs):
        assert hf_id == "json"
        if split == "train":
            return [
                {
                    "id": 1,
                    "question": "Train",
                    "solutions": json.dumps(["print(2)"]),
                    "input_output": json.dumps(
                        {"inputs": ["\n"], "outputs": ["2\n"]}
                    ),
                    "difficulty": "introductory",
                    "starter_code": "",
                }
            ]
        return [
            {
                "id": index,
                "question": f"Test {index}",
                "solutions": json.dumps(
                    ["print('bad')" if index < 24 else "print(1)"]
                ),
                "input_output": json.dumps(
                    {"inputs": ["\n"], "outputs": ["1\n"]}
                ),
                "difficulty": "introductory",
                "starter_code": "",
            }
            for index in range(1000)
        ]

    def validate(solution, _row):
        return SimpleNamespace(
            score=0.0 if "bad" in solution else 1.0,
            reason="wrong_output" if "bad" in solution else "passed",
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        "eval.scorers.generation._run_apps_tests",
        validate,
    )

    _, test = web_acquire.load_benchmark_dataset(
        {"benchmark": "APPS introductory", "task_type": "code_generation"},
        max_train=1,
        max_test=800,
    )

    assert len(test) == 800
    assert all(row["runner_compatible"] is True for row in test)
    assert test[0]["text"] == "Test 24"
    assert test[-1]["text"] == "Test 823"


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
        {"benchmark": "SAMSum dialogue summarization", "task_type": "generation"},
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
        {"benchmark": "SAMSum", "task_type": "generation"},
        max_train=2,
        max_test=1,
        log=logs.append,
    )

    assert [row["text"] for row in train] == ["A: unique"]
    assert [row["text"] for row in test] == ["a: duplicate text"]
    assert any("normalized overlap" in message for message in logs)


def _write_local_bundle(root, name, task_type, train, test, schema_version=1):
    bundle = root / name
    bundle.mkdir()
    source_id = {
        "bc5cdr": "tner/bc5cdr",
        "gsm8k": "openai/gsm8k",
        "apps": "codeparrot/apps",
        "mbpp": "google-research-datasets/mbpp",
        "samsum": "knkarthick/samsum",
        "emotion": "dair-ai/emotion",
    }[name]
    config = {
        "gsm8k": "main",
        "apps": "introductory",
        "mbpp": "sanitized",
    }.get(name)
    manifest = {
        "schema_version": schema_version,
        "name": name,
        "task_type": task_type,
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
    ("name", "task_type", "benchmark", "train_row", "test_row"),
    [
        (
            "bc5cdr",
            "NER",
            "Biomedical NER (BC5CDR)",
            {"text": "Aspirin helps", "entities": [{"text": "Aspirin", "type": "Chemical"}]},
            {"text": "Fever persists", "entities": [{"text": "Fever", "type": "Disease"}]},
        ),
        (
            "gsm8k",
            "math_reasoning",
            "grade-school math GSM8K",
            {"text": "1+1?", "answer": "2", "cot_reasoning": "Add.", "label": "math_reasoning"},
            {"text": "2+2?", "answer": "4", "cot_reasoning": "Add.", "label": "math_reasoning"},
        ),
        (
            "mbpp",
            "code_generation",
            "HumanEval / MBPP",
            {
                "text": "Write one",
                "answer": "def one(): return 1",
                "code": "def one(): return 1",
                "test_list": ["assert one() == 1"],
                "test_imports": [],
                "task_id": 1,
                "label": "code_generation",
            },
            {
                "text": "Write two",
                "answer": "def two(): return 2",
                "code": "def two(): return 2",
                "test_list": ["assert two() == 2"],
                "test_imports": [],
                "task_id": 2,
                "label": "code_generation",
            },
        ),
        (
            "apps",
            "code_generation",
            "APPS introductory",
            {
                "text": "Add two integers.",
                "answer": (
                    "class Solution:\n"
                    "    def add(self, a, b):\n"
                    "        return a + b"
                ),
                "code": (
                    "class Solution:\n"
                    "    def add(self, a, b):\n"
                    "        return a + b"
                ),
                "solutions": [
                    "class Solution:\n"
                    "    def add(self, a, b):\n"
                    "        return a + b"
                ],
                "starter_code": (
                    "class Solution:\n"
                    "    def add(self, a, b):\n"
                    "        pass"
                ),
                "difficulty": "introductory",
                "input_output": {
                    "fn_name": "add",
                    "inputs": ["[2, 3]"],
                    "outputs": ["5"],
                },
                "execution_mode": "call_based",
                "fn_name": "add",
                "entry_point": "add",
                "problem_id": 1,
                "label": "code_generation",
            },
            {
                "text": "Echo one line.",
                "starter_code": "",
                "difficulty": "introductory",
                "input_output": {
                    "inputs": ["hello\n"],
                    "outputs": ["hello\n"],
                },
                "execution_mode": "stdin",
                "problem_id": 2,
                "label": "code_generation",
            },
        ),
        (
            "samsum",
            "generation",
            "SAMSum dialogue summarization",
            {"text": "A: hi", "answer": "A says hi.", "label": "generation"},
            {"text": "B: bye", "answer": "B says bye.", "label": "generation"},
        ),
    ],
)
def test_local_fallback_supports_task_schemas(
    tmp_path, monkeypatch, name, task_type, benchmark, train_row, test_row
):
    _write_local_bundle(tmp_path, name, task_type, [train_row], [test_row])
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))
    meta = {}

    train, test = web_acquire.load_local_dataset(
        {"benchmark": benchmark, "task_type": task_type},
        task_type,
        max_train=5,
        max_test=5,
        meta=meta,
    )

    assert train == [train_row]
    assert test == [test_row]
    assert meta["source_records"][0]["id"]
    assert meta["eval_ban"][0]["split"] == "test"


def test_sms_plan_rejects_unrelated_emotion_bundle_then_uses_stage0(
    tmp_path,
    monkeypatch,
):
    _write_local_bundle(
        tmp_path,
        "emotion",
        "classification",
        [
            {"text": "I am joyful", "label": "joy"},
            {"text": "I am sad", "label": "sadness"},
        ],
        [{"text": "I am afraid", "label": "fear"}],
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))
    sms_train = [{"text": "hello friend", "label": "ham"}]
    sms_test = [{"text": "claim prize", "label": "spam"}]
    monkeypatch.setattr(
        "data.loaders.sms_spam.download_sms_spam",
        lambda: (sms_train, sms_test),
    )
    plan = {
        "benchmark": "UCI SMS Spam Collection",
        "task_type": "classification",
        "labels": ["ham", "spam"],
    }

    assert web_acquire.load_local_dataset(
        plan,
        "classification",
        max_train=10,
        max_test=10,
    ) is None
    train, test = web_acquire.acquire_dataset(
        plan,
        benchmark_max_train=10,
        benchmark_max_test=10,
    )

    assert train == sms_train
    assert test == sms_test


def test_fpb_plan_rejects_unrelated_emotion_bundle_then_uses_stage0(
    tmp_path,
    monkeypatch,
):
    _write_local_bundle(
        tmp_path,
        "emotion",
        "classification",
        [
            {"text": "I am joyful", "label": "joy"},
            {"text": "I am sad", "label": "sadness"},
        ],
        [{"text": "I am afraid", "label": "fear"}],
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    def load_dataset(dataset_id, split=None, **_kwargs):
        assert dataset_id == "ChanceFocus/flare-fpb"
        assert split == "train"
        return [
            {"text": "profits rose", "answer": "positive"},
            {"text": "profits fell", "answer": "negative"},
            {"text": "profits held", "answer": "neutral"},
            {"text": "sales rose", "answer": "positive"},
        ]

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    plan = {
        "benchmark": "Financial PhraseBank",
        "task_type": "classification",
        "labels": ["negative", "neutral", "positive"],
    }

    assert web_acquire.load_local_dataset(
        plan,
        "classification",
        max_train=2,
        max_test=1,
    ) is None
    train, test = web_acquire.acquire_dataset(
        plan,
        benchmark_max_train=2,
        benchmark_max_test=1,
    )

    assert len(train) == 2
    assert len(test) == 1
    assert {row["label"] for row in train + test} <= {
        "negative",
        "neutral",
        "positive",
    }


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
        "classification",
        train_rows,
        test_rows,
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    loaded = web_acquire.load_local_dataset(
        {
            "benchmark": "custom emotion benchmark",
            "task_type": "classification",
            "labels": ["joy", "sadness"],
        },
        "classification",
        max_train=10,
        max_test=10,
    )

    assert loaded == (train_rows, test_rows)


def test_local_mbpp_test_metadata_reaches_eval_set_rows(tmp_path, monkeypatch):
    train_row = {
        "text": "Write one",
        "answer": "def one(): return 1",
        "code": "def one(): return 1",
        "test_list": ["assert one() == 1"],
        "test_imports": ["import math"],
        "entry_point": "one",
        "signature": "def one():",
        "task_id": 1,
        "label": "code_generation",
    }
    test_row = {
        **train_row,
        "text": "Write two",
        "answer": "def two(): return 2",
        "code": "def two(): return 2",
        "test_list": ["assert two() == 2"],
        "entry_point": "two",
        "signature": "def two():",
        "task_id": 2,
    }
    _write_local_bundle(
        tmp_path,
        "mbpp",
        "code_generation",
        [train_row],
        [test_row],
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    _, loaded_test = web_acquire.load_local_dataset(
        {"benchmark": "MBPP", "task_type": "code_generation"},
        "code_generation",
        max_train=5,
        max_test=5,
    )
    eval_set = build_eval_set(loaded_test, task_type="code_generation")

    assert eval_set.all[0]["test_list"] == test_row["test_list"]
    assert eval_set.all[0]["test_imports"] == test_row["test_imports"]
    assert eval_set.all[0]["entry_point"] == "two"
    assert eval_set.all[0]["signature"] == "def two():"


def test_local_apps_test_metadata_reaches_eval_set_without_gold(
    tmp_path,
    monkeypatch,
):
    train_row = {
        "text": "Add two integers.",
        "answer": "class Solution:\n    def add(self, a, b): return a + b",
        "solutions": [
            "class Solution:\n    def add(self, a, b): return a + b"
        ],
        "starter_code": "class Solution:\n    def add(self, a, b): pass",
        "difficulty": "introductory",
        "input_output": {
            "fn_name": "add",
            "inputs": ["[2, 3]"],
            "outputs": ["5"],
        },
        "execution_mode": "call_based",
        "fn_name": "add",
        "entry_point": "add",
        "label": "code_generation",
    }
    test_row = {
        "text": "Echo one line.",
        "starter_code": "",
        "difficulty": "introductory",
        "input_output": {
            "inputs": ["hello\n"],
            "outputs": ["hello\n"],
        },
        "execution_mode": "stdin",
        "label": "code_generation",
    }
    _write_local_bundle(
        tmp_path,
        "apps",
        "code_generation",
        [train_row],
        [test_row],
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    loaded_train, loaded_test = web_acquire.load_local_dataset(
        {"benchmark": "APPS introductory", "task_type": "code_generation"},
        "code_generation",
        max_train=5,
        max_test=5,
    )
    eval_set = build_eval_set(loaded_test, task_type="code_generation")

    assert loaded_train == [train_row]
    assert "answer" not in loaded_test[0]
    assert eval_set.all[0]["input_output"] == test_row["input_output"]
    assert eval_set.all[0]["execution_mode"] == "stdin"


def test_local_fallback_does_not_import_key_bearing_config(tmp_path, monkeypatch):
    train_row = {"text": "train dialogue", "answer": "summary", "label": "generation"}
    test_row = {"text": "test dialogue", "answer": "summary", "label": "generation"}
    _write_local_bundle(tmp_path, "samsum", "generation", [train_row], [test_row])
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
        {"benchmark": "SAMSum", "task_type": "generation"},
        "generation",
        max_train=5,
        max_test=5,
    )

    assert train == [train_row]
    assert test == [test_row]


def test_local_fallback_fails_closed_on_normalized_overlap(tmp_path, monkeypatch):
    train = {"text": " Duplicated\n  Dialogue ", "answer": "summary", "label": "generation"}
    test = {"text": "duplicated dialogue", "answer": "summary", "label": "generation"}
    _write_local_bundle(tmp_path, "samsum", "generation", [train], [test])
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="normalized train/test text overlap"):
        web_acquire.load_local_dataset(
            {"benchmark": "SAMSum", "task_type": "generation"},
            "generation",
            max_train=5,
            max_test=5,
        )


def test_local_fallback_rejects_checksum_mismatch(tmp_path, monkeypatch):
    train = {"text": "train", "answer": "summary", "label": "generation"}
    test = {"text": "test", "answer": "summary", "label": "generation"}
    _write_local_bundle(tmp_path, "samsum", "generation", [train], [test])
    bundle = tmp_path / "samsum"
    (bundle / "checksums.sha256").write_text(
        f"{'0' * 64}  train.jsonl\n"
        f"{'0' * 64}  test.jsonl\n"
        f"{'0' * 64}  manifest.json\n"
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="checksum mismatch"):
        web_acquire.load_local_dataset(
            {"benchmark": "SAMSum", "task_type": "generation"},
            "generation",
            max_train=5,
            max_test=5,
        )


def test_schema_v2_local_bundle_requires_checksum_sidecar(tmp_path, monkeypatch):
    train = {"text": "train", "answer": "summary", "label": "generation"}
    test = {"text": "test", "answer": "summary", "label": "generation"}
    _write_local_bundle(
        tmp_path, "samsum", "generation", [train], [test], schema_version=2
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="schema-v2.*checksums.sha256"):
        web_acquire.load_local_dataset(
            {"benchmark": "SAMSum", "task_type": "generation"},
            "generation",
            max_train=5,
            max_test=5,
        )


def test_explicit_legacy_schema_v1_loads_without_sidecar_and_logs(tmp_path, monkeypatch):
    train = {"text": "train", "answer": "summary", "label": "generation"}
    test = {"text": "test", "answer": "summary", "label": "generation"}
    _write_local_bundle(
        tmp_path, "samsum", "generation", [train], [test], schema_version=1
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))
    logs = []

    loaded_train, loaded_test = web_acquire.load_local_dataset(
        {"benchmark": "SAMSum", "task_type": "generation"},
        "generation",
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
        "generation",
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


def test_agent_first_mbpp_routes_through_converter_and_preserves_tests(monkeypatch):
    captured = []
    mapping_calls = []

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
        web_acquire,
        "_exa_find_hf_dataset_ids",
        lambda *_args, **_kwargs: ["google-research-datasets/mbpp"],
    )
    monkeypatch.setattr(
        web_acquire,
        "_peek_hf_dataset",
        lambda *_args, **_kwargs: (
            "sanitized",
            ["train", "test"],
            ["prompt", "code", "test_list", "test_imports"],
            {
                "prompt": "Write one",
                "code": "def one(): return 1",
                "test_list": ["assert one() == 1"],
                "test_imports": [],
            },
            {},
        ),
    )

    def map_dataset(*_args, **_kwargs):
        mapping_calls.append(True)
        return {
            "train_split": "train",
            "test_split": "test",
            "question_col": "prompt",
            "answer_col": "code",
        }

    monkeypatch.setattr(web_acquire, "_llm_map_dataset", map_dataset)
    monkeypatch.setattr(
        web_acquire,
        "_materialize_from_mapping",
        lambda *_args, **_kwargs: (
            [
                {
                    "text": "Write train",
                    "answer": "def train(): return 1",
                    "label": "code_generation",
                }
            ],
            [
                {
                    "text": "Write test",
                    "answer": "def test(): return 1",
                    "label": "code_generation",
                }
            ],
        ),
    )

    def fake_load_dataset(hf_id, config=None, split=None, **_kwargs):
        assert (hf_id, config) == (
            "google-research-datasets/mbpp",
            "sanitized",
        )
        suffix = "train" if str(split).startswith("train") else "test"
        return [
            {
                "task_id": 1 if suffix == "train" else 2,
                "prompt": f"Write {suffix}",
                "code": f"def {suffix}():\n    return 1",
                "test_imports": ["import math"],
                "test_list": [f"assert {suffix}() == 1"],
            }
        ]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    web_acquire._discover_worker(
        {"benchmark": "Unknown", "task_name": "Python", "labels": []},
        "code generation",
        "code_generation",
        5,
        5,
        Queue(),
    )

    result = captured[-1]
    assert mapping_calls == []
    assert result["train"][0]["test_list"] == ["assert train() == 1"]
    assert result["test"][0]["test_imports"] == ["import math"]
    assert result["test"][0]["answer"] == result["test"][0]["code"]
    assert "MBPP" in result["source"]


def test_agent_first_apps_routes_through_converter_and_preserves_schema(
    monkeypatch,
):
    captured = []
    mapping_calls = []

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
        web_acquire,
        "_exa_find_hf_dataset_ids",
        lambda *_args, **_kwargs: ["codeparrot/apps"],
    )
    monkeypatch.setattr(
        web_acquire,
        "_llm_map_dataset",
        lambda *_args, **_kwargs: mapping_calls.append(True),
    )
    expected = (
        [
            {
                "text": "train",
                "answer": "print(1)",
                "solutions": ["print(1)"],
                "starter_code": "",
                "difficulty": "introductory",
                "input_output": {
                    "inputs": ["\n"],
                    "outputs": ["1\n"],
                },
                "execution_mode": "stdin",
                "label": "code_generation",
            }
        ],
        [
            {
                "text": "test",
                "starter_code": "",
                "difficulty": "introductory",
                "input_output": {
                    "inputs": ["x\n"],
                    "outputs": ["x\n"],
                },
                "execution_mode": "stdin",
                "label": "code_generation",
            }
        ],
    )
    metadata = {
        "source": "real APPS introductory",
        "source_records": [
            {
                "kind": "hf",
                "id": "codeparrot/apps",
                "config": "introductory",
                "split": "train",
                "role": "curriculum",
            },
            {
                "kind": "hf",
                "id": "codeparrot/apps",
                "config": "introductory",
                "split": "test",
                "role": "eval",
            },
        ],
    }
    metadata["eval_ban"] = [metadata["source_records"][1]]

    def fake_loader(*_args, meta, **_kwargs):
        meta.update(metadata)
        return expected

    monkeypatch.setattr(web_acquire, "load_benchmark_dataset", fake_loader)

    web_acquire._discover_worker(
        {"benchmark": "Unknown", "task_name": "Python", "labels": []},
        "code generation",
        "code_generation",
        5,
        5,
        Queue(),
    )

    result = captured[-1]
    assert mapping_calls == []
    assert result["train"] == expected[0]
    assert result["test"] == expected[1]
    assert result["source_records"][0]["config"] == "introductory"
    assert result["eval_ban"][0]["split"] == "test"


@pytest.mark.parametrize(
    "row",
    [
        {
            "text": "APPS stdin",
            "input_output": {"inputs": ["1\n"], "outputs": ["1\n"]},
            "execution_mode": "stdin",
            "starter_code": "",
            "difficulty": "introductory",
            "label": "code_generation",
        },
        {
            "text": "MBPP function",
            "answer": "def one(): return 1",
            "test_list": ["assert one() == 1"],
            "test_imports": [],
            "label": "code_generation",
        },
    ],
)
def test_discovery_validation_accepts_apps_or_mbpp_executable_tests(row):
    train = [{**row, "answer": row.get("answer", "print(1)")}]
    test = [{**row, "text": row["text"] + " test"}]

    accepted = web_acquire._validate_discovered_splits(
        (train, test),
        "code_generation",
        source="test",
    )

    assert accepted == (train, test)


def test_discovery_validation_rejects_code_rows_without_executable_tests():
    with pytest.raises(ValueError, match="executable.*input_output.*test_list"):
        web_acquire._validate_discovered_splits(
            (
                [{"text": "train", "answer": "print(1)"}],
                [{"text": "test"}],
            ),
            "code_generation",
            source="test",
        )


def test_agent_first_apps_removes_solution_fingerprint_overlap():
    shared = "print(1)"
    train = [
        {
            "text": "D2",
            "answer": shared,
            "solutions": [shared],
            "input_output": {"inputs": ["\n"], "outputs": ["1\n"]},
            "execution_mode": "stdin",
            "label": "code_generation",
        },
        {
            "text": "unique",
            "answer": "print(2)",
            "solutions": ["print(2)"],
            "input_output": {"inputs": ["\n"], "outputs": ["2\n"]},
            "execution_mode": "stdin",
            "label": "code_generation",
        },
    ]
    test = [
        {
            "text": "D1",
            "solutions": [shared],
            "input_output": {"inputs": ["\n"], "outputs": ["1\n"]},
            "execution_mode": "stdin",
            "label": "code_generation",
        }
    ]

    accepted = web_acquire._accept_discovered_result(
        (train, test),
        "code_generation",
        source="agent-first",
        requested_benchmark="APPS introductory",
    )

    assert accepted == ([train[1]], test)


def test_discovery_parent_propagates_precise_metadata(monkeypatch):
    train = [{"text": "train", "answer": "a", "label": "generation"}]
    test = [{"text": "test", "answer": "b", "label": "generation"}]
    records = [
        {"kind": "hf", "id": "org/discovered", "config": "cfg",
         "split": "train", "role": "curriculum"},
        {"kind": "hf", "id": "org/discovered", "config": "cfg",
         "split": "test", "role": "eval"},
    ]

    def fake_worker(_plan, _description, _task_type, _max_train, _max_test, queue):
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
        "generation",
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
        {"benchmark": "SAMSum", "task_type": "generation"},
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
        {"benchmark": "SAMSum", "task_type": "generation"},
        description="readiness test",
    )

    assert result == expected
    assert order == ["agentic", "local", "stage0"]


@pytest.mark.parametrize(
    "discovered",
    [
        (
            [{"text": "train row", "label": "code_generation"}],
            [
                {
                    "text": "test row",
                    "answer": "def test(): return 1",
                    "test_list": ["assert test() == 1"],
                    "test_imports": [],
                    "label": "code_generation",
                }
            ],
        ),
        (
            [
                {
                    "text": "  Duplicate\nProblem ",
                    "answer": "def train(): return 1",
                    "test_list": ["assert train() == 1"],
                    "test_imports": [],
                    "label": "code_generation",
                }
            ],
            [
                {
                    "text": "duplicate problem",
                    "answer": "def test(): return 1",
                    "test_list": ["assert test() == 1"],
                    "test_imports": [],
                    "label": "code_generation",
                }
            ],
        ),
        (
            [
                {
                    "text": "train row",
                    "answer": "def train(): return 1",
                    "label": "code_generation",
                }
            ],
            [
                {
                    "text": "test row",
                    "answer": "def test(): return 1",
                    "label": "code_generation",
                }
            ],
        ),
    ],
)
def test_agent_first_invalid_discovery_falls_back_to_local(
    monkeypatch,
    discovered,
):
    order = []
    local_rows = (
        [
            {
                "text": "local train",
                "answer": "def local_train(): return 1",
                "test_list": ["assert local_train() == 1"],
                "test_imports": [],
                "label": "code_generation",
            }
        ],
        [
            {
                "text": "local test",
                "answer": "def local_test(): return 1",
                "test_list": ["assert local_test() == 1"],
                "test_imports": [],
                "label": "code_generation",
            }
        ],
    )

    def discover(*_args, **_kwargs):
        order.append("agentic")
        return discovered

    def local(*_args, meta=None, **_kwargs):
        order.append("local")
        if meta is not None:
            meta.clear()
            meta["source"] = "local valid MBPP"
        return local_rows

    def unexpected_stage0(*_args, **_kwargs):
        raise AssertionError("valid local fallback must stop acquisition")

    monkeypatch.setenv("SLM_AGENT_FIRST_DATASET_DISCOVERY", "1")
    monkeypatch.setattr(web_acquire, "discover_and_load_hf_dataset", discover)
    monkeypatch.setattr(web_acquire, "load_local_dataset", local)
    monkeypatch.setattr(web_acquire, "load_benchmark_dataset", unexpected_stage0)
    meta = {"source": "stale"}

    result = web_acquire.acquire_dataset(
        {"benchmark": "MBPP", "task_type": "code_generation"},
        description="readiness test",
        meta=meta,
    )

    assert result == local_rows
    assert order == ["agentic", "local"]
    assert meta == {"source": "local valid MBPP"}


def test_agent_first_apps_rejects_mbpp_schema_and_falls_back_local(
    monkeypatch,
):
    order = []
    discovered_mbpp = (
        [
            {
                "text": "MBPP train",
                "answer": "def one(): return 1",
                "test_list": ["assert one() == 1"],
                "test_imports": [],
                "label": "code_generation",
            }
        ],
        [
            {
                "text": "MBPP test",
                "answer": "def two(): return 2",
                "test_list": ["assert two() == 2"],
                "test_imports": [],
                "label": "code_generation",
            }
        ],
    )
    local_apps = (
        [
            {
                "text": "APPS train",
                "answer": "print(1)",
                "input_output": {"inputs": ["\n"], "outputs": ["1\n"]},
                "execution_mode": "stdin",
                "label": "code_generation",
            }
        ],
        [
            {
                "text": "APPS test",
                "input_output": {"inputs": ["x\n"], "outputs": ["x\n"]},
                "execution_mode": "stdin",
                "label": "code_generation",
            }
        ],
    )

    def discover(*_args, **_kwargs):
        order.append("agentic")
        return discovered_mbpp

    def local(*_args, **_kwargs):
        order.append("local")
        return local_apps

    monkeypatch.setenv("SLM_AGENT_FIRST_DATASET_DISCOVERY", "1")
    monkeypatch.setattr(web_acquire, "discover_and_load_hf_dataset", discover)
    monkeypatch.setattr(web_acquire, "load_local_dataset", local)
    monkeypatch.setattr(
        web_acquire,
        "load_benchmark_dataset",
        lambda *_args, **_kwargs: pytest.fail("local APPS should stop fallback"),
    )

    result = web_acquire.acquire_dataset(
        {"benchmark": "APPS introductory", "task_type": "code_generation"},
        description="APPS code",
    )

    assert result == local_apps
    assert order == ["agentic", "local"]


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
        {"benchmark": "SAMSum", "task_type": "generation"},
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
        {"benchmark": "SAMSum", "task_type": "generation"},
        description="normal run",
    )

    assert result == expected
    assert order == ["local", "stage0"]
