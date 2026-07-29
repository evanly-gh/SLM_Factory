"""Schema conversion helpers for the public codeparrot/APPS benchmark."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from collections.abc import Iterable
from collections.abc import Callable


APPS_SOURCE_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"


def _json_value(value: object, expected_type: type):
    if isinstance(value, expected_type):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, expected_type) else None


def valid_apps_input_output(value: object) -> bool:
    """Return whether APPS metadata contains at least one runnable test case."""
    if not isinstance(value, dict):
        return False
    inputs = value.get("inputs")
    outputs = value.get("outputs")
    if (
        not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or not inputs
        or len(inputs) != len(outputs)
    ):
        return False
    fn_name = value.get("fn_name")
    return fn_name is None or (
        isinstance(fn_name, str) and bool(fn_name.strip())
    )


def _solution_fingerprint(solution: object) -> str:
    normalized = "".join(str(solution or "").split())
    return (
        hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if normalized
        else ""
    )


def apps_fingerprint_overlap(
    train_rows: Iterable[dict],
    test_rows: Iterable[dict],
) -> set[str]:
    test_urls = {
        str(row.get("url") or "").strip().rstrip("/")
        for row in test_rows
        if str(row.get("url") or "").strip()
    }
    test_solutions = {
        fingerprint
        for row in test_rows
        for solution in row.get("solutions", [])
        if (fingerprint := _solution_fingerprint(solution))
    }
    overlaps = set()
    for row in train_rows:
        url = str(row.get("url") or "").strip().rstrip("/")
        if url and url in test_urls:
            overlaps.add(f"url:{url}")
        for solution in row.get("solutions", []):
            fingerprint = _solution_fingerprint(solution)
            if fingerprint and fingerprint in test_solutions:
                overlaps.add(f"solution:{fingerprint}")
    return overlaps


def remove_apps_train_fingerprint_overlap(
    train_rows: list[dict],
    test_rows: list[dict],
) -> tuple[list[dict], int]:
    test_urls = {
        str(row.get("url") or "").strip().rstrip("/")
        for row in test_rows
        if str(row.get("url") or "").strip()
    }
    test_solutions = {
        fingerprint
        for row in test_rows
        for solution in row.get("solutions", [])
        if (fingerprint := _solution_fingerprint(solution))
    }

    def overlaps(row: dict) -> bool:
        url = str(row.get("url") or "").strip().rstrip("/")
        if url and url in test_urls:
            return True
        return any(
            _solution_fingerprint(solution) in test_solutions
            for solution in row.get("solutions", [])
        )

    clean = [row for row in train_rows if not overlaps(row)]
    return clean, len(train_rows) - len(clean)


def _usable_python_solution(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    source = value.strip()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(source, "<apps_solution>", "exec")
    except (SyntaxError, ValueError):
        return None
    return source


def convert_apps_rows(
    dataset: Iterable[dict],
    *,
    split: str | None = None,
    limit: int | None = None,
    gold_validator: Callable[[str, dict], bool] | None = None,
    conversion_stats: dict | None = None,
    skip_runner_incompatible: bool = False,
) -> tuple[list[dict], list[str]]:
    """Convert APPS JSON rows while retaining both official execution schemas.

    The official train split is supervision, so rows without a usable Python
    solution are discarded. The official test split is execution-scored and can
    legitimately omit solutions; executable ``input_output`` metadata is enough.
    """
    if split not in (None, "train", "test"):
        raise ValueError(f"unsupported APPS split {split!r}")

    stats = conversion_stats if conversion_stats is not None else {}

    def increment(section: str, key: str | None = None) -> None:
        if key is None:
            stats[section] = int(stats.get(section, 0)) + 1
            return
        values = stats.setdefault(section, {})
        values[key] = int(values.get(key, 0)) + 1

    rows: list[dict] = []
    for example in dataset:
        increment("source_rows_seen")
        if str(example.get("difficulty") or "").strip().lower() != "introductory":
            continue
        increment("introductory_rows_seen")
        question = str(example.get("question") or "").strip()
        input_output = _json_value(example.get("input_output"), dict)
        solutions_value = _json_value(example.get("solutions"), list) or []
        solutions = [
            solution
            for raw_solution in solutions_value
            if (solution := _usable_python_solution(raw_solution)) is not None
        ]
        if not question:
            increment("rows_removed_by_reason", "invalid_question")
            continue
        if not valid_apps_input_output(input_output):
            increment("rows_removed_by_reason", "invalid_tests")
            continue
        if split == "train" and not solutions:
            increment("rows_removed_by_reason", "no_compilable_solution")
            continue

        input_output = dict(input_output)
        fn_name = input_output.get("fn_name")
        if isinstance(fn_name, str):
            fn_name = fn_name.strip()
            input_output["fn_name"] = fn_name
        execution_mode = "call_based" if fn_name else "stdin"
        if not fn_name:
            input_output.pop("fn_name", None)

        row = {
            "text": question,
            "starter_code": str(example.get("starter_code") or "").strip(),
            "difficulty": "introductory",
            "input_output": input_output,
            "execution_mode": execution_mode,
            "runner_compatible": True,
            "label": "code_generation",
        }
        if fn_name:
            row["fn_name"] = fn_name
            row["entry_point"] = fn_name
        selected_solution = solutions[0] if solutions else None
        validation_reasons: Counter = Counter()
        if gold_validator is not None and solutions:
            selected_solution = None
            for solution in solutions:
                result = gold_validator(solution, row)
                passed = (
                    result
                    if isinstance(result, bool)
                    else float(getattr(result, "score", 0.0)) >= 0.5
                )
                reason = (
                    "passed"
                    if passed
                    else str(
                        getattr(
                            result,
                            "reason",
                            "validator_rejected",
                        )
                        or "validator_rejected"
                    )
                )
                increment("gold_solution_attempts", reason)
                validation_reasons[reason] += 1
                if passed:
                    selected_solution = solution
                    break
        if split == "train" and gold_validator is not None:
            if selected_solution is None:
                reason = (
                    validation_reasons.most_common(1)[0][0]
                    if validation_reasons
                    else "no_compilable_solution"
                )
                increment("rows_removed_by_reason", reason)
                continue
            solutions = [
                selected_solution,
                *[
                    solution
                    for solution in solutions
                    if solution != selected_solution
                ],
            ]
            row["gold_validation_status"] = "passed"
        elif split == "test" and gold_validator is not None:
            if selected_solution is None:
                if solutions:
                    row["solutions"] = solutions
                    row["gold_validation_status"] = "no_passing_solution"
                    row["runner_compatible"] = False
                    reason = (
                        validation_reasons.most_common(1)[0][0]
                        if validation_reasons
                        else "validator_rejected"
                    )
                    increment("rows_without_passing_gold", reason)
                    increment("gold", "no_passing_solution")
                    increment("gold", "runner_incompatible")
                else:
                    row["gold_validation_status"] = "missing"
                    increment("gold", "missing")
            else:
                solutions = [
                    selected_solution,
                    *[
                        solution
                        for solution in solutions
                        if solution != selected_solution
                    ],
                ]
                row["gold_validation_status"] = "passed"
                increment("gold", "passing")
        if selected_solution is not None:
            row.update(
                {
                    "answer": selected_solution,
                    "code": selected_solution,
                    "solutions": solutions,
                }
            )

        problem_id = example.get("problem_id", example.get("id"))
        if problem_id is not None:
            row["problem_id"] = problem_id
        url = str(example.get("url") or "").strip()
        if url:
            row["url"] = url
        if (
            skip_runner_incompatible
            and row.get("runner_compatible", True) is False
        ):
            increment("runner_incompatible_skipped")
            continue
        rows.append(row)
        increment("converted_rows")
        if limit is not None and len(rows) >= max(0, int(limit)):
            break

    return rows, ["code_generation"] if rows else []
