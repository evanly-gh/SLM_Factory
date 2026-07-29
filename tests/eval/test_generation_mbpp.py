import os
import subprocess
import sys
from textwrap import dedent
from pathlib import Path

import pytest

from data.eval_set import EvalSet
from eval.scorers import generation


ROOT = Path(__file__).parents[2]


def _row(
    tests=None,
    *,
    imports=None,
    answer="def add(a, b):\n    return a + b",
    **metadata,
):
    return {
        "text": "Write a function that adds two numbers.",
        "answer": answer,
        "test_list": ["assert add(2, 3) == 5"] if tests is None else tests,
        "test_imports": [] if imports is None else imports,
        "label": "code_generation",
        **metadata,
    }


def _score_one(prediction, row=None):
    eval_set = EvalSet(
        pos=[row or _row()],
        neg=[],
        boundary=[],
        task_type="code_generation",
    )
    return generation.score(eval_set, [prediction])


def test_mbpp_correct_code_scores_one():
    result = _score_one("def add(a, b):\n    return a + b")

    assert result["f1"] == 1.0
    assert result["failures"] == []


def test_mbpp_successful_worker_exits_cleanly():
    result = generation._run_mbpp_tests(
        "def add(a, b):\n    return a + b",
        _row(entry_point="add", signature="def add(a, b):"),
    )

    assert result.score == 1.0
    assert result.returncode == 0


def test_mbpp_wrong_but_executable_code_scores_zero_with_diagnostic():
    result = _score_one("def add(a, b):\n    return a - b")

    assert result["f1"] == 0.0
    assert "AssertionError" in result["failures"][0]["execution_diagnostic"]


def test_mbpp_pass_scores_zero():
    assert _score_one("pass")["f1"] == 0.0


@pytest.mark.parametrize(
    ("prediction", "diagnostic"),
    [
        ("def add(:\n    return 5", "SyntaxError"),
        ("def add(a, b):\n    raise RuntimeError('boom')", "RuntimeError"),
    ],
)
def test_mbpp_syntax_and_runtime_failures_score_zero(prediction, diagnostic):
    result = _score_one(prediction)

    assert result["f1"] == 0.0
    assert diagnostic in result["failures"][0]["execution_diagnostic"]


def test_mbpp_infinite_loop_is_stopped_inside_disposable_subprocess():
    script = """
from eval.scorers import generation

row = {
    "text": "loop",
    "answer": "def loop(): return 1",
    "test_imports": [],
    "test_list": ["assert loop() == 1"],
}
result = generation._run_mbpp_tests(
    "def loop():\\n    while True:\\n        pass",
    row,
    timeout_seconds=0.2,
)
assert result.score == 0.0
assert result.timed_out is True
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        # The scorer import can take several seconds under a saturated full-suite run;
        # the inner candidate timeout remains 0.2s and is what this test verifies.
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_mbpp_test_imports_are_available_to_solution_and_tests():
    row = _row(
        ["assert root(81) == 9"],
        imports=["from math import sqrt"],
        answer="def root(value):\n    return sqrt(value)",
    )

    assert _score_one("def root(value):\n    return sqrt(value)", row)["f1"] == 1.0


def test_mbpp_executes_every_test_list_assertion():
    row = _row(
        [
            "assert constant(2, 3) == 5",
            "assert constant(-1, 1) == 0",
        ],
        answer="def constant(a, b):\n    return a + b",
    )

    assert _score_one("def constant(a, b):\n    return 5", row)["f1"] == 0.0


def test_mbpp_empty_test_list_fails_closed():
    result = _score_one("def add(a, b):\n    return a + b", _row([]))

    assert result["f1"] == 0.0
    assert "no executable tests" in result["failures"][0]["execution_diagnostic"].lower()


def test_mbpp_entry_point_metadata_is_enforced():
    row = _row(["assert True"], entry_point="add")

    result = _score_one("def other(a, b):\n    return a + b", row)

    assert result["f1"] == 0.0
    assert "entry point" in result["failures"][0]["execution_diagnostic"].lower()


def test_mbpp_aggregate_score_uses_per_row_tests():
    correct = _row(["assert add(1, 4) == 5"])
    wrong = _row(["assert multiply(3, 4) == 12"], answer="def multiply(a, b): return a * b")
    eval_set = EvalSet(
        pos=[correct],
        neg=[wrong],
        boundary=[],
        task_type="code_generation",
    )

    result = generation.score(
        eval_set,
        [
            "def add(a, b):\n    return a + b",
            "def multiply(a, b):\n    return a + b",
        ],
    )

    assert result["f1"] == 0.5
    assert result["per_class"]["judge_score"] == 0.5
    assert len(result["failures"]) == 1
    assert result["failures"][0]["text"] == wrong["text"]


def test_mbpp_runs_with_sanitized_env_in_a_different_process(monkeypatch):
    monkeypatch.setenv("SLM_TEST_PARENT_SECRET", "must-not-leak")
    parent_pid = os.getpid()
    row = _row(
        [
            f"assert runtime_identity()[0] != {parent_pid}",
            "assert runtime_identity()[1] is False",
        ],
        answer="def runtime_identity(): ...",
    )
    prediction = (
        "def runtime_identity():\n"
        "    import os\n"
        "    return os.getpid(), 'SLM_TEST_PARENT_SECRET' in os.environ"
    )

    assert _score_one(prediction, row)["f1"] == 1.0


def test_mbpp_temporary_working_directory_is_cleaned(tmp_path, monkeypatch):
    from eval.scorers import code_execution

    monkeypatch.setattr(code_execution.tempfile, "tempdir", str(tmp_path))

    assert _score_one("def add(a, b):\n    return a + b")["f1"] == 1.0
    assert list(tmp_path.iterdir()) == []


def test_mbpp_subprocess_start_failure_scores_zero(monkeypatch):
    from eval.scorers import code_execution

    def fail_to_start(*_args, **_kwargs):
        raise OSError("process unavailable")

    monkeypatch.setattr(code_execution.subprocess, "Popen", fail_to_start)

    result = _score_one("def add(a, b):\n    return a + b")

    assert result["f1"] == 0.0
    assert "process unavailable" in result["failures"][0]["execution_diagnostic"]


def test_mbpp_prior_payload_result_token_bypass_scores_zero():
    prediction = dedent(
        """
        import json
        import os
        from pathlib import Path

        payload = json.loads(Path("payload.json").read_text(encoding="utf-8"))
        Path("result.json").write_text(
            json.dumps({"success_token": payload["success_token"]}),
            encoding="utf-8",
        )
        os._exit(0)
        """
    )

    result = _score_one(prediction)

    assert result["f1"] == 0.0
    assert result["failures"]


def test_mbpp_candidate_cannot_spoof_runner_success_fd():
    prediction = dedent(
        """
        import os

        for fd in range(3, 256):
            try:
                os.write(fd, b"MBPP_TESTS_PASSED\\n")
            except OSError:
                pass
        os._exit(0)
        """
    )

    result = _score_one(prediction)

    assert result["f1"] == 0.0
    assert result["failures"]


def test_mbpp_candidate_cannot_inspect_tests_or_expected_payload():
    prediction = dedent(
        """
        import inspect

        def payload_is_hidden():
            frame = inspect.currentframe()
            while frame:
                for value in frame.f_locals.values():
                    if isinstance(value, dict) and (
                        "test_list" in value or "cases" in value
                    ):
                        return False
                frame = frame.f_back
            return True
        """
    )
    row = _row(
        ["assert payload_is_hidden() is True"],
        answer="def payload_is_hidden(): return True",
        entry_point="payload_is_hidden",
        signature="def payload_is_hidden():",
    )

    assert _score_one(prediction, row)["f1"] == 1.0


def test_mbpp_candidate_sees_sanitized_argv_without_success_fd():
    prediction = (
        "import sys\n"
        "def candidate_argv():\n"
        "    return sys.argv\n"
    )
    row = _row(
        ["assert candidate_argv() == ['candidate.py']"],
        answer="def candidate_argv(): return ['candidate.py']",
        entry_point="candidate_argv",
        signature="def candidate_argv():",
    )

    assert _score_one(prediction, row)["f1"] == 1.0


def test_mbpp_candidate_cannot_replace_runner_builtins_to_skip_tests():
    prediction = dedent(
        """
        import builtins

        builtins.exec = lambda *_args, **_kwargs: None

        def add(a, b):
            return a - b
        """
    )

    result = _score_one(prediction)

    assert result["f1"] == 0.0
    assert result["failures"]


def test_code_prompt_preserves_required_signature_without_revealing_solution():
    row = _row(
        answer="def add(a: int, b: int = 0) -> int:\n    return a + b",
        entry_point="add",
        signature="def add(a: int, b: int = 0) -> int:",
    )
    eval_set = EvalSet(pos=[row], neg=[], boundary=[], task_type="code_generation")

    prompt = generation.build_prompts(eval_set)[0]

    assert "def add(a: int, b: int = 0) -> int:" in prompt
    assert "return a + b" not in prompt
    assert "without Markdown fences" in prompt


def test_code_prompt_never_derives_signature_or_helpers_from_hidden_gold():
    row = _row(
        answer=(
            "def hidden_helper(value):\n"
            "    return value\n\n"
            "def secret_entry(value):\n"
            "    return hidden_helper(value)"
        ),
        entry_point="required_entry",
    )
    without_answer = {key: value for key, value in row.items() if key != "answer"}

    prompt_with_gold = generation.build_code_prompt(row)
    prompt_without_gold = generation.build_code_prompt(without_answer)

    assert prompt_with_gold == prompt_without_gold
    assert "hidden_helper" not in prompt_with_gold
    assert "secret_entry" not in prompt_with_gold
    assert "Required entry point: required_entry" in prompt_with_gold


def test_code_extraction_strips_markdown_fences_only_for_code_generation():
    code_set = EvalSet(pos=[_row()], neg=[], boundary=[], task_type="code_generation")
    math_set = EvalSet(
        pos=[{"text": "1+1", "answer": "2"}],
        neg=[],
        boundary=[],
        task_type="math_reasoning",
    )
    raw = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```"

    assert generation.extract_predictions([raw], code_set) == [
        "def add(a, b):\n    return a + b"
    ]
    assert generation.extract_predictions([raw], math_set) == [raw]


def test_pipeline_docs_mark_mbpp_execution_as_trusted_input_only():
    docs = (ROOT / "docs" / "PIPELINE.md").read_text(encoding="utf-8").lower()

    assert "not a hostile-code sandbox" in docs
    assert "trusted-input only" in docs
