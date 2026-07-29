#!/usr/bin/env python3
"""Fast offline APPS gold/executor parity smoke.

The fixed first row from each preserved local split is intentional: changing the
sample requires changing its expected problem identity in code and tests.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class GoldParityError(RuntimeError):
    """A fixed APPS gold no longer passes the production executor."""


@dataclass(frozen=True)
class SampleSpec:
    split: str
    relative_path: Path
    line_index: int
    problem_id: int


@dataclass(frozen=True)
class ParityResult:
    split: str
    problem_id: int
    cases_executed: int
    cases_total: int
    elapsed_s: float


DEFAULT_SAMPLES = (
    SampleSpec(
        split="train",
        relative_path=Path("data/local/apps/train.jsonl"),
        line_index=0,
        problem_id=2361,
    ),
    SampleSpec(
        split="test",
        relative_path=Path("data/local/apps/test.jsonl"),
        line_index=0,
        problem_id=4000,
    ),
)


def load_deterministic_sample(root: Path, spec: SampleSpec) -> dict:
    """Load one fixed JSONL row and fail if the local bundle drifted."""
    path = Path(root).resolve() / spec.relative_path
    try:
        with path.open(encoding="utf-8") as source:
            line = next(
                (
                    value
                    for index, value in enumerate(source)
                    if index == spec.line_index
                ),
                None,
            )
    except OSError as exc:
        raise GoldParityError(
            f"cannot read APPS split={spec.split} at {path}: {exc}"
        ) from exc
    if line is None:
        raise GoldParityError(
            f"APPS split={spec.split} has no row {spec.line_index} at {path}"
        )
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise GoldParityError(
            f"APPS split={spec.split} row {spec.line_index} is invalid JSON: {exc}"
        ) from exc
    if not isinstance(row, dict):
        raise GoldParityError(
            f"APPS split={spec.split} row {spec.line_index} is not an object"
        )
    if row.get("problem_id") != spec.problem_id:
        raise GoldParityError(
            f"APPS sample identity drift for split={spec.split}: "
            f"expected problem_id={spec.problem_id}, "
            f"found {row.get('problem_id')!r}"
        )
    return row


def _validate_gold_metadata(spec: SampleSpec, row: dict) -> tuple[str, int]:
    status = row.get("gold_validation_status")
    if status != "passed" or row.get("runner_compatible", True) is False:
        raise GoldParityError(
            f"split={spec.split} problem_id={spec.problem_id} has incompatible "
            f"gold metadata: status={status!r}, "
            f"runner_compatible={row.get('runner_compatible')!r}"
        )
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise GoldParityError(
            f"split={spec.split} problem_id={spec.problem_id} has no gold answer"
        )
    input_output = row.get("input_output")
    inputs = input_output.get("inputs") if isinstance(input_output, dict) else None
    outputs = input_output.get("outputs") if isinstance(input_output, dict) else None
    if (
        not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or not inputs
        or len(inputs) != len(outputs)
    ):
        raise GoldParityError(
            f"split={spec.split} problem_id={spec.problem_id} has invalid "
            "preserved execution cases"
        )
    return answer, len(inputs)


def run_gold_parity(
    root: Path = PROJECT_ROOT,
    *,
    specs: Iterable[SampleSpec] = DEFAULT_SAMPLES,
    output: Callable[[str], object] = print,
) -> list[ParityResult]:
    """Execute every preserved case for each fixed gold through production code."""
    from eval.scorers.generation import _run_apps_tests

    total_started = time.perf_counter()
    results = []
    for spec in tuple(specs):
        row = load_deterministic_sample(root, spec)
        answer, expected_cases = _validate_gold_metadata(spec, row)
        started = time.perf_counter()
        execution = _run_apps_tests(answer, row)
        elapsed_s = time.perf_counter() - started
        if (
            execution.score != 1.0
            or execution.tests_executed != expected_cases
            or execution.tests_total != expected_cases
        ):
            raise GoldParityError(
                f"split={spec.split} problem_id={spec.problem_id} "
                f"reason={execution.reason or 'unknown'} "
                f"cases={execution.tests_executed}/{expected_cases}: "
                f"{execution.diagnostic or 'gold did not pass all cases'}"
            )
        result = ParityResult(
            split=spec.split,
            problem_id=spec.problem_id,
            cases_executed=execution.tests_executed,
            cases_total=execution.tests_total,
            elapsed_s=elapsed_s,
        )
        results.append(result)
        output(
            f"PASS APPS split={result.split} problem_id={result.problem_id} "
            f"cases={result.cases_executed}/{result.cases_total} "
            f"time_s={result.elapsed_s:.3f}"
        )
    output(
        f"PASS APPS gold parity: samples={len(results)} "
        f"cases={sum(result.cases_total for result in results)} "
        f"time_s={time.perf_counter() - total_started:.3f}"
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed local APPS golds through the production executor."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="SLM Factory project root",
    )
    args = parser.parse_args()
    run_gold_parity(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
