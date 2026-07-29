import hashlib
import json
import re

import pytest

from scripts import download_datasets


def _spec(name):
    return download_datasets.DATASETS_BY_NAME[name]


def test_bc5cdr_converter_builds_exact_entity_spans():
    rows, labels = download_datasets._convert(
        [
            {
                "tokens": ["Aspirin", "can", "cause", "renal", "failure", "."],
                "tags": [1, 0, 0, 2, 3, 0],
            }
        ],
        _spec("bc5cdr"),
    )

    assert rows == [
        {
            "text": "Aspirin can cause renal failure .",
            "entities": [
                {"text": "Aspirin", "type": "Chemical"},
                {"text": "renal failure", "type": "Disease"},
            ],
        }
    ]
    assert labels == ["Chemical", "Disease"]


def test_gsm8k_converter_preserves_gold_reasoning_and_final_answer():
    rows, labels = download_datasets._convert(
        [
            {
                "question": "How many widgets remain?",
                "answer": "Ten minus three is seven.\n#### 7",
            }
        ],
        _spec("gsm8k"),
    )

    assert rows == [
        {
            "text": "How many widgets remain?",
            "answer": "7",
            "cot_reasoning": "Ten minus three is seven.",
            "label": "math_reasoning",
        }
    ]
    assert labels == ["math_reasoning"]


def test_mbpp_converter_preserves_code_and_tests():
    source = {
        "task_id": 602,
        "prompt": "Write a function that returns the first repeated character.",
        "code": "def first_repeated(value):\n    return value[0]",
        "test_imports": ["import math"],
        "test_list": ["assert first_repeated('aa') == 'a'"],
    }

    rows, labels = download_datasets._convert([source], _spec("mbpp"))

    assert rows == [
        {
            "text": source["prompt"],
            "answer": source["code"],
            "code": source["code"],
            "test_imports": source["test_imports"],
            "test_list": source["test_list"],
            "task_id": 602,
            "label": "code_generation",
        }
    ]
    assert labels == ["code_generation"]


def test_apps_train_converter_parses_call_based_solution_and_tests():
    source = {
        "id": 17,
        "question": "Add two integers.",
        "solutions": json.dumps(
            [
                "class Solution:\n"
                "    def add(self, a, b):\n"
                "        return a + b"
            ]
        ),
        "input_output": json.dumps(
            {
                "fn_name": "add",
                "inputs": ["[2, 3]"],
                "outputs": ["5"],
            }
        ),
        "difficulty": "introductory",
        "starter_code": (
            "class Solution:\n"
            "    def add(self, a, b):\n"
            "        pass"
        ),
        "url": "https://example.test/add",
    }

    rows, labels = download_datasets._convert(
        [source],
        _spec("apps"),
        split="train",
    )

    assert rows == [
        {
            "text": source["question"],
            "answer": json.loads(source["solutions"])[0],
            "code": json.loads(source["solutions"])[0],
            "solutions": json.loads(source["solutions"]),
            "starter_code": source["starter_code"],
            "difficulty": "introductory",
            "input_output": json.loads(source["input_output"]),
            "execution_mode": "call_based",
            "runner_compatible": True,
            "fn_name": "add",
            "entry_point": "add",
            "problem_id": 17,
            "url": source["url"],
            "label": "code_generation",
        }
    ]
    assert labels == ["code_generation"]


def test_apps_test_converter_accepts_stdin_tests_without_gold_solution():
    source = {
        "problem_id": 23,
        "question": "Print the sum.",
        "solutions": "",
        "input_output": {
            "inputs": ["2 3\n"],
            "outputs": ["5\n"],
        },
        "difficulty": "introductory",
        "starter_code": "",
        "url": "",
    }

    rows, labels = download_datasets._convert(
        [source],
        _spec("apps"),
        split="test",
    )

    assert rows == [
        {
            "text": source["question"],
            "starter_code": "",
            "difficulty": "introductory",
            "input_output": source["input_output"],
            "execution_mode": "stdin",
            "runner_compatible": True,
            "problem_id": 23,
            "label": "code_generation",
        }
    ]
    assert "answer" not in rows[0]
    assert labels == ["code_generation"]


def test_apps_converter_selects_only_syntactically_usable_gold_solutions():
    source = {
        "problem_id": 24,
        "question": "Print one.",
        "solutions": json.dumps(["def broken(:\n    pass", "print(1)"]),
        "input_output": json.dumps(
            {"inputs": ["\n"], "outputs": ["1\n"]}
        ),
        "difficulty": "introductory",
        "starter_code": "",
    }

    rows, _ = download_datasets._convert(
        [source],
        _spec("apps"),
        split="train",
    )

    assert rows[0]["answer"] == "print(1)"
    assert rows[0]["solutions"] == ["print(1)"]


@pytest.mark.parametrize(
    ("split", "updates"),
    [
        ("train", {"solutions": "[]"}),
        ("train", {"solutions": '["def broken(:\\n    pass"]'}),
        ("train", {"solutions": '["return 1"]'}),
        ("train", {"solutions": '["nonlocal missing"]'}),
        ("train", {"input_output": "{}"}),
        ("test", {"input_output": '{"inputs": [], "outputs": []}'}),
        (
            "test",
            {
                "input_output": (
                    '{"inputs": ["1\\n"], "outputs": ["1\\n", "2\\n"]}'
                )
            },
        ),
        ("test", {"difficulty": "interview"}),
    ],
)
def test_apps_converter_drops_rows_that_are_not_usable_for_split(split, updates):
    source = {
        "problem_id": 1,
        "question": "Echo one.",
        "solutions": '["print(1)"]',
        "input_output": '{"inputs": ["1\\n"], "outputs": ["1\\n"]}',
        "difficulty": "introductory",
        "starter_code": "",
    }
    source.update(updates)

    rows, labels = download_datasets._convert(
        [source],
        _spec("apps"),
        split=split,
    )

    assert rows == []
    assert labels == []


def test_apps_converter_prefers_solution_that_passes_preserved_tests():
    source = {
        "problem_id": 25,
        "question": "Print one.",
        "solutions": json.dumps(["print(0)", "print(1)"]),
        "input_output": json.dumps(
            {"inputs": ["\n"], "outputs": ["1\n"]}
        ),
        "difficulty": "introductory",
        "starter_code": "",
    }
    attempted = []

    def validator(solution, row):
        attempted.append((solution, row["input_output"]))
        return solution == "print(1)"

    rows, _ = download_datasets._convert(
        [source],
        _spec("apps"),
        split="train",
        gold_validator=validator,
    )

    assert rows[0]["answer"] == "print(1)"
    assert rows[0]["solutions"][0] == "print(1)"
    assert [solution for solution, _ in attempted] == ["print(0)", "print(1)"]


def test_apps_test_converter_marks_nonpassing_gold_without_dropping_row():
    source = {
        "problem_id": 26,
        "question": "Print one.",
        "solutions": json.dumps(["print(0)"]),
        "input_output": json.dumps(
            {"inputs": ["\n"], "outputs": ["1\n"]}
        ),
        "difficulty": "introductory",
        "starter_code": "",
    }
    stats = {}

    class Result:
        score = 0.0
        reason = "wrong_output"

    rows, _ = download_datasets._convert(
        [source],
        _spec("apps"),
        split="test",
        gold_validator=lambda *_args: Result(),
        conversion_stats=stats,
    )

    assert len(rows) == 1
    assert "answer" not in rows[0]
    assert rows[0]["solutions"] == ["print(0)"]
    assert rows[0]["gold_validation_status"] == "no_passing_solution"
    assert rows[0]["runner_compatible"] is False
    assert stats["rows_without_passing_gold"]["wrong_output"] == 1


def test_apps_catalog_declares_introductory_official_splits_and_split_schema():
    spec = _spec("apps")

    assert spec["task_type"] == "code_generation"
    assert spec["sources"][0]["id"] == "codeparrot/apps"
    assert spec["sources"][0]["config"] == "introductory"
    assert re.fullmatch(r"[0-9a-f]{40}", spec["sources"][0]["revision"])
    assert spec["sources"][0]["splits"] == {"train": "train", "test": "test"}
    assert spec["sources"][0]["revision"] in spec["sources"][1]["data_files"]["train"]
    assert spec["source_counts"] == {"train": 2639, "test": 1000}
    assert {"input_output", "starter_code", "difficulty", "execution_mode"} <= set(
        spec["row_schema"]["required"]
    )
    assert {"answer", "solutions"} <= set(
        spec["split_row_schema"]["train"]["required"]
    )


def test_samsum_converter_preserves_dialogue_summary_pair():
    rows, labels = download_datasets._convert(
        [{"id": "chat-1", "dialogue": "A: Hi\nB: Hello", "summary": "A greets B."}],
        _spec("samsum"),
    )

    assert rows == [
        {
            "text": "A: Hi\nB: Hello",
            "answer": "A greets B.",
            "label": "generation",
        }
    ]
    assert labels == ["generation"]


def test_manifest_records_schema_counts_provenance_and_eval_ban():
    spec = _spec("gsm8k")
    source = spec["sources"][0]
    manifest = download_datasets._build_manifest(
        spec,
        source,
        train_rows=[{"text": "train", "answer": "1", "label": "math_reasoning"}],
        test_rows=[{"text": "test", "answer": "2", "label": "math_reasoning"}],
        labels=["math_reasoning"],
        removed_train_overlap=0,
        source_revision="abc123",
        content_hashes={"train.jsonl": "a" * 64, "test.jsonl": "b" * 64},
    )

    assert manifest["schema_version"] == 2
    assert manifest["task_type"] == "math_reasoning"
    assert manifest["hf_id"] == "openai/gsm8k"
    assert manifest["config"] == "main"
    assert manifest["source_revision"] == "abc123"
    assert manifest["source_splits"] == {"train": "train", "test": "test"}
    assert manifest["counts"] == {"train": 1, "test": 1, "total": 2}
    assert manifest["provenance"]["official_splits_only"] is True
    assert manifest["provenance"]["records"][0]["split"] == "train"
    assert manifest["eval_ban"][0]["split"] == "test"
    assert manifest["overlap"] == {
        "field": "text",
        "normalization": "nfkc_casefold_whitespace_v1",
        "normalized_count": 0,
        "removed_from_train": 0,
    }
    assert manifest["integrity"]["files"]["train.jsonl"] == "a" * 64


def test_write_bundle_rejects_normalized_train_test_contamination(tmp_path):
    train = {"text": " Same\n  Prompt ", "answer": "gold", "label": "generation"}
    test = {"text": "same prompt", "answer": "gold", "label": "generation"}

    with pytest.raises(ValueError, match="normalized train/test text overlap"):
        download_datasets._write_bundle(
            tmp_path,
            _spec("samsum"),
            _spec("samsum")["sources"][0],
            [train],
            [test],
            labels=["generation"],
        )


def test_write_apps_bundle_rejects_solution_fingerprint_contamination(tmp_path):
    shared_solution = "print(1)"
    train = {
        "text": "hard version",
        "answer": shared_solution,
        "solutions": [shared_solution],
        "starter_code": "",
        "difficulty": "introductory",
        "input_output": {"inputs": ["\n"], "outputs": ["1\n"]},
        "execution_mode": "stdin",
        "label": "code_generation",
    }
    test = {
        "text": "easy version",
        "solutions": [shared_solution],
        "starter_code": "",
        "difficulty": "introductory",
        "input_output": {"inputs": ["\n"], "outputs": ["1\n"]},
        "execution_mode": "stdin",
        "label": "code_generation",
    }

    with pytest.raises(ValueError, match="URL or solution fingerprint overlap"):
        download_datasets._write_bundle(
            tmp_path,
            _spec("apps"),
            _spec("apps")["sources"][1],
            [train],
            [test],
            labels=["code_generation"],
        )


def test_normalized_decontamination_keeps_official_test_rows():
    train = [
        {"text": "unique", "label": "generation"},
        {"text": " Duplicate\n Text ", "label": "generation"},
    ]
    test = [{"text": "duplicate text", "label": "generation"}]

    clean, removed = download_datasets._drop_train_overlap(train, test)

    assert clean == [train[0]]
    assert removed == 1
    assert test == [{"text": "duplicate text", "label": "generation"}]


def test_apps_fingerprint_decontamination_removes_train_side_only():
    shared = "def solve():\n    return 1"
    train = [
        {
            "text": "D2 hard version",
            "solutions": [shared],
            "url": "https://codeforces.com/problem/1/D2",
        },
        {
            "text": "unique",
            "solutions": ["print(2)"],
            "url": "https://example.test/unique",
        },
    ]
    test = [
        {
            "text": "D1 easy version",
            "solutions": ["def solve(): return 1"],
            "url": "https://codeforces.com/problem/1/D1",
        }
    ]

    clean, removed = download_datasets._drop_train_overlap(
        train,
        test,
        spec=_spec("apps"),
    )

    assert clean == [train[1]]
    assert removed == 1
    assert test[0]["text"] == "D1 easy version"


def test_write_bundle_creates_hashed_jsonl_and_manifest(tmp_path):
    train = [{"text": "train dialogue", "answer": "train summary", "label": "generation"}]
    test = [{"text": "test dialogue", "answer": "test summary", "label": "generation"}]

    manifest = download_datasets._write_bundle(
        tmp_path,
        _spec("samsum"),
        _spec("samsum")["sources"][1],
        train,
        test,
        labels=["generation"],
        removed_train_overlap=42,
        source_revision="deadbeef",
    )

    bundle = tmp_path / "samsum"
    assert json.loads((bundle / "train.jsonl").read_text()) == train[0]
    assert json.loads((bundle / "test.jsonl").read_text()) == test[0]
    assert json.loads((bundle / "manifest.json").read_text()) == manifest
    assert manifest["hf_id"] == "knkarthick/samsum"
    assert manifest["overlap"]["removed_from_train"] == 42
    assert manifest["source_revision"] == "deadbeef"
    checksums = {}
    for line in (bundle / "checksums.sha256").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        checksums[filename] = digest
    assert set(checksums) == {"train.jsonl", "test.jsonl", "manifest.json"}
    for filename, expected in checksums.items():
        actual = hashlib.sha256((bundle / filename).read_bytes()).hexdigest()
        assert actual == expected
    assert manifest["integrity"]["files"] == {
        "train.jsonl": checksums["train.jsonl"],
        "test.jsonl": checksums["test.jsonl"],
    }


def test_cached_hf_revision_reads_hub_ref_without_network(tmp_path, monkeypatch):
    ref = tmp_path / "datasets--openai--gsm8k" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("740312add88f781978c0658806c59bc2815b9866\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    assert (
        download_datasets._cached_hf_revision("openai/gsm8k")
        == "740312add88f781978c0658806c59bc2815b9866"
    )


def test_only_cli_downloads_selected_bundles(tmp_path, monkeypatch):
    downloaded = []

    def fake_download(spec, output_dir, load_dataset_fn=None):
        downloaded.append((spec["name"], output_dir))

    monkeypatch.setattr(download_datasets, "_download_dataset", fake_download)
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    assert download_datasets.main(["--only", "gsm8k", "apps", "mbpp"]) == 0
    assert downloaded == [
        ("gsm8k", tmp_path),
        ("apps", tmp_path),
        ("mbpp", tmp_path),
    ]


def test_only_cli_accepts_comma_separated_names_and_rejects_unknown(tmp_path, monkeypatch):
    downloaded = []
    monkeypatch.setattr(
        download_datasets,
        "_download_dataset",
        lambda spec, output_dir, load_dataset_fn=None: downloaded.append(spec["name"]),
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    assert download_datasets.main(["--only", "bc5cdr,samsum"]) == 0
    assert downloaded == ["bc5cdr", "samsum"]
    with pytest.raises(SystemExit):
        download_datasets.main(["--only", "not-a-dataset"])


def test_list_cli_reports_supported_task_aware_catalog(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    assert download_datasets.main(["--list"]) == 0
    listed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    by_name = {item["name"]: item for item in listed}

    assert {
        "emotion",
        "go_emotions",
        "bc5cdr",
        "gsm8k",
        "apps",
        "mbpp",
        "samsum",
    } <= set(by_name)
    assert by_name["bc5cdr"]["task_type"] == "NER"
    assert by_name["gsm8k"]["task_type"] == "math_reasoning"
    assert by_name["apps"]["task_type"] == "code_generation"
    assert by_name["mbpp"]["task_type"] == "code_generation"
    assert by_name["samsum"]["task_type"] == "generation"
    assert by_name["samsum"]["installed"] is False
