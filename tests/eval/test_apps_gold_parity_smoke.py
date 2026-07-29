import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.smoke_apps_gold_parity import (
    DEFAULT_SAMPLES,
    GoldParityError,
    SampleSpec,
    load_deterministic_sample,
    run_gold_parity,
)


ROOT = Path(__file__).parents[2]


def test_default_apps_gold_samples_pass_every_preserved_case_and_report_time():
    messages = []

    results = run_gold_parity(ROOT, output=messages.append)

    assert [(result.split, result.problem_id) for result in results] == [
        ("train", 2361),
        ("test", 4000),
    ]
    assert all(result.cases_total > 0 for result in results)
    assert all(
        result.cases_executed == result.cases_total
        for result in results
    )
    assert all(result.elapsed_s >= 0.0 for result in results)
    report = "\n".join(messages)
    assert "cases=" in report
    assert "time_s=" in report
    assert "PASS APPS gold parity" in report


def test_deterministic_sample_rejects_dataset_identity_drift(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(
        json.dumps({"problem_id": 99}) + "\n",
        encoding="utf-8",
    )
    spec = SampleSpec(
        split="train",
        relative_path=Path("sample.jsonl"),
        line_index=0,
        problem_id=2361,
    )

    with pytest.raises(GoldParityError, match="identity drift"):
        load_deterministic_sample(tmp_path, spec)


def test_apps_gold_parity_fails_closed_on_incompatible_gold(tmp_path):
    row = {
        "problem_id": 7,
        "text": "Print one.",
        "starter_code": "",
        "input_output": {
            "inputs": ["\n"],
            "outputs": ["1\n"],
        },
        "execution_mode": "stdin",
        "gold_validation_status": "passed",
        "answer": "print(0)",
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    spec = SampleSpec(
        split="train",
        relative_path=Path("sample.jsonl"),
        line_index=0,
        problem_id=7,
    )

    with pytest.raises(
        GoldParityError,
        match=r"split=train.*problem_id=7.*wrong_output",
    ):
        run_gold_parity(tmp_path, specs=(spec,), output=lambda _message: None)


def test_default_apps_gold_samples_are_fixed_first_rows():
    assert [
        (
            spec.split,
            spec.relative_path.as_posix(),
            spec.line_index,
            spec.problem_id,
        )
        for spec in DEFAULT_SAMPLES
    ] == [
        ("train", "data/local/apps/train.jsonl", 0, 2361),
        ("test", "data/local/apps/test.jsonl", 0, 4000),
    ]


def test_apps_gold_parity_cli_runs_without_pythonpath():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "smoke_apps_gold_parity.py"),
        ],
        cwd=ROOT,
        env={
            "PATH": str(Path(sys.executable).parent),
            "PYTHONHASHSEED": "0",
        },
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "PASS APPS gold parity:" in completed.stdout
