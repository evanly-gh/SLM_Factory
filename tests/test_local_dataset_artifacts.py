import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1] / "data" / "local"
EXPECTED = {
    "bc5cdr": "NER",
    "gsm8k": "math_reasoning",
    "apps": "code_generation",
    "mbpp": "code_generation",
    "samsum": "generation",
}
LEGACY = ("emotion", "go_emotions")


def _rows(bundle, split):
    with (bundle / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("name", "task_type"), EXPECTED.items())
def test_local_artifact_manifest_counts_splits_and_hashes(name, task_type):
    bundle = ROOT / name
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    train, test = _rows(bundle, "train"), _rows(bundle, "test")

    assert manifest["schema_version"] == 2
    assert manifest["task_type"] == task_type
    assert manifest["source_splits"] == {"train": "train", "test": "test"}
    assert manifest["provenance"]["official_splits_only"] is True
    assert [(row["split"], row["role"]) for row in manifest["provenance"]["records"]] == [
        ("train", "curriculum"),
        ("test", "eval"),
    ]
    assert [(row["split"], row["role"]) for row in manifest["eval_ban"]] == [
        ("test", "eval")
    ]
    assert manifest["counts"] == {
        "train": len(train),
        "test": len(test),
        "total": len(train) + len(test),
    }
    assert manifest["overlap"]["normalization"] == "nfkc_casefold_whitespace_v1"
    assert manifest["overlap"]["normalized_count"] == 0
    assert not ({_normalized(row["text"]) for row in train}
                & {_normalized(row["text"]) for row in test})
    assert re.fullmatch(r"[0-9a-f]{40,64}", manifest["source_revision"])

    checksum_lines = (bundle / "checksums.sha256").read_text().splitlines()
    checksums = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
    assert set(checksums) == {"train.jsonl", "test.jsonl", "manifest.json"}
    for filename, digest in checksums.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert _sha256(bundle / filename) == digest
    assert manifest["integrity"]["files"] == {
        "train.jsonl": checksums["train.jsonl"],
        "test.jsonl": checksums["test.jsonl"],
    }


def test_bc5cdr_artifact_schema_and_content():
    rows = _rows(ROOT / "bc5cdr", "train") + _rows(ROOT / "bc5cdr", "test")
    for row in rows:
        assert isinstance(row["text"], str) and row["text"].strip()
        assert isinstance(row["entities"], list)
        for entity in row["entities"]:
            assert set(entity) == {"text", "type"}
            assert isinstance(entity["text"], str) and entity["text"]
            assert entity["type"] in {"Chemical", "Disease"}
            assert entity["text"] in row["text"]


def test_gsm8k_artifact_schema_and_content():
    rows = _rows(ROOT / "gsm8k", "train") + _rows(ROOT / "gsm8k", "test")
    for row in rows:
        assert isinstance(row["text"], str) and row["text"].strip()
        assert isinstance(row["answer"], str) and row["answer"].strip()
        assert isinstance(row["cot_reasoning"], str) and row["cot_reasoning"].strip()
        assert row["label"] == "math_reasoning"


def test_mbpp_artifact_preserves_code_tests_and_imports_without_execution():
    rows = _rows(ROOT / "mbpp", "train") + _rows(ROOT / "mbpp", "test")
    for row in rows:
        assert isinstance(row["text"], str) and row["text"].strip()
        assert isinstance(row["code"], str) and row["code"].strip()
        assert row["answer"] == row["code"]
        assert isinstance(row["test_list"], list) and row["test_list"]
        assert all(isinstance(test, str) and test.strip() for test in row["test_list"])
        assert isinstance(row["test_imports"], list)
        assert all(isinstance(statement, str) for statement in row["test_imports"])
        assert isinstance(row["task_id"], int)
        assert row["label"] == "code_generation"


def test_apps_artifact_preserves_executable_schemas_and_split_gold_contract():
    bundle = ROOT / "apps"
    train = _rows(bundle, "train")
    test = _rows(bundle, "test")

    assert train and test
    assert {row["execution_mode"] for row in train + test} == {
        "call_based",
        "stdin",
    }
    for row in train:
        assert isinstance(row["answer"], str) and row["answer"].strip()
        assert isinstance(row["solutions"], list) and row["solutions"]
        assert row["answer"] == row["solutions"][0]
    for row in train + test:
        assert isinstance(row["text"], str) and row["text"].strip()
        assert row["difficulty"] == "introductory"
        assert isinstance(row["starter_code"], str)
        tests = row["input_output"]
        assert isinstance(tests, dict)
        assert isinstance(tests["inputs"], list) and tests["inputs"]
        assert len(tests["inputs"]) == len(tests["outputs"])
        if row["execution_mode"] == "call_based":
            assert row["fn_name"] == tests["fn_name"]
            assert row["entry_point"] == tests["fn_name"]
        else:
            assert "fn_name" not in tests
        assert row["label"] == "code_generation"


def test_apps_artifact_removes_confirmed_d1_d2_solution_overlap():
    def fingerprints(rows):
        return {
            re.sub(r"\s+", "", solution)
            for row in rows
            for solution in row.get("solutions", [])
        }

    bundle = ROOT / "apps"
    manifest = json.loads((bundle / "manifest.json").read_text())
    train = _rows(bundle, "train")
    test = _rows(bundle, "test")

    assert 2389 not in {row.get("problem_id") for row in train}
    assert not (fingerprints(train) & fingerprints(test))
    assert manifest["overlap"]["fingerprint_removed_from_train"] == 1
    assert len(train) > 650
    removed = manifest["filtering"]["removed_unusable"]["train"]
    assert removed == (
        manifest["filtering"]["source_counts"]["train"]
        - len(train)
        - manifest["overlap"]["removed_from_train"]
    )
    reason_counts = manifest["filtering"]["conversion"]["train"][
        "rows_removed_by_reason"
    ]
    assert sum(reason_counts.values()) == removed


def test_apps_official_gold_solutions_pass_both_executor_formats():
    from eval.scorers import generation

    train = _rows(ROOT / "apps", "train")
    for mode in ("call_based", "stdin"):
        candidates = [
            row
            for row in train
            if row["execution_mode"] == mode and row.get("answer")
        ][:5]
        assert candidates, f"APPS artifact has no gold {mode} rows"
        outcomes = [
            generation._run_apps_tests(
                row["answer"],
                row,
                timeout_seconds=2.0,
                max_cases=1,
            )
            for row in candidates
        ]
        assert any(result.score == 1.0 for result in outcomes), [
            result.diagnostic for result in outcomes
        ]


def test_apps_test_gold_is_explicitly_validated_or_unscored():
    bundle = ROOT / "apps"
    manifest = json.loads((bundle / "manifest.json").read_text())
    rows = _rows(bundle, "test")

    for row in rows:
        if row.get("answer"):
            assert row["gold_validation_status"] == "passed"
        else:
            assert row["gold_validation_status"] in {
                "missing",
                "no_passing_solution",
            }
        assert row["runner_compatible"] is (
            row["gold_validation_status"] != "no_passing_solution"
        )
    gold = manifest["filtering"]["conversion"]["test"]["gold"]
    assert gold["passing"] == sum(bool(row.get("answer")) for row in rows)
    assert gold["missing"] + gold["no_passing_solution"] == sum(
        not bool(row.get("answer")) for row in rows
    )
    assert gold["runner_incompatible"] == sum(
        row["runner_compatible"] is False for row in rows
    )


def test_apps_local_loader_skips_explicitly_incompatible_eval_rows():
    from data.loaders.web_acquire import load_local_dataset

    _, test = load_local_dataset(
        {"benchmark": "APPS introductory", "task_type": "code_generation"},
        "code_generation",
        max_train=721,
        max_test=1000,
    )

    assert len(test) == 976
    assert all(row["runner_compatible"] is True for row in test)


def test_apps_default_800_eval_prompts_fit_4096_context_with_output_reserve():
    from data.eval_set import build_eval_set
    from eval.scorers.generation import build_prompts

    rows = _rows(ROOT / "apps", "test")
    eval_set = build_eval_set(
        rows,
        "code_generation",
        target=800,
    )
    prompts = build_prompts(eval_set)
    conservative_counts = [
        len(re.findall(r"\w+|[^\w\s]", prompt)) + 64
        for prompt in prompts
    ]

    assert len(prompts) == 800
    assert max(conservative_counts) <= 4096 - 1024


def test_apps_training_rows_fit_4096_without_target_truncation():
    from eval.scorers.generation import build_code_prompt
    from training.lora_trainer import build_code_training_target

    rows = _rows(ROOT / "apps", "train")
    conservative_counts = [
        len(
            re.findall(
                r"\w+|[^\w\s]",
                build_code_prompt(row)
                + "\n"
                + build_code_training_target(row),
            )
        )
        + 64
        for row in rows
    ]

    assert max(conservative_counts) <= 4096


def test_samsum_artifact_schema_and_content():
    rows = _rows(ROOT / "samsum", "train") + _rows(ROOT / "samsum", "test")
    for row in rows:
        assert isinstance(row["text"], str) and row["text"].strip()
        assert isinstance(row["answer"], str) and row["answer"].strip()
        assert row["label"] == "generation"


@pytest.mark.parametrize("name", LEGACY)
def test_unhashed_legacy_bundles_are_explicit_schema_v1(name):
    bundle = ROOT / name
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert not (bundle / "checksums.sha256").exists()
