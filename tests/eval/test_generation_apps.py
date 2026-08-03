import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from data.eval_set import EvalSet
from eval.scorers import generation


ROOT = Path(__file__).parents[2]


def _apps_row(
    input_output,
    *,
    answer="class Solution:\n    def add(self, a, b):\n        return a + b",
    starter_code="class Solution:\n    def add(self, a, b):\n        pass",
    **metadata,
):
    row = {
        "text": "Add two integers.",
        "answer": answer,
        "solutions": [answer],
        "starter_code": starter_code,
        "difficulty": "introductory",
        "input_output": input_output,
        "execution_mode": (
            "call_based" if input_output.get("fn_name") else "stdin"
        ),
        "label": "code_generation",
        **metadata,
    }
    if input_output.get("fn_name"):
        row["fn_name"] = input_output["fn_name"]
        row["entry_point"] = input_output["fn_name"]
    return row


def _score_one(prediction, row):
    eval_set = EvalSet(
        all=[row],
        task_type="code_generation",
    )
    return generation.score(eval_set, [prediction])


def test_apps_call_based_gold_solution_scores_one():
    row = _apps_row(
        {
            "fn_name": "add",
            "inputs": ["[2, 3]", "[-1, 1]"],
            "outputs": ["5", "0"],
        }
    )

    result = generation._run_apps_tests(row["answer"], row)

    assert result.score == 1.0
    assert result.tests_used == 2
    assert result.tests_total == 2


def test_apps_call_based_accepts_top_level_function():
    row = _apps_row(
        {
            "fn_name": "add",
            "inputs": [[[2, 3], 4]],
            "outputs": [[6, 7]],
        },
        starter_code="def add(values, delta):\n    pass",
    )
    code = "def add(values, delta):\n    return tuple(x + delta for x in values)"

    assert generation._run_apps_tests(code, row).score == 1.0


def test_apps_call_based_provides_official_typing_prelude():
    code = (
        "class Solution:\n"
        "    def total(self, values: List[int]) -> int:\n"
        "        return sum(values)"
    )
    row = _apps_row(
        {
            "fn_name": "total",
            "inputs": ["[[1, 2, 3]]"],
            "outputs": ["6"],
        },
        answer=code,
        starter_code=(
            "class Solution:\n"
            "    def total(self, values: List[int]) -> int:\n"
            "        pass"
        ),
    )

    assert generation._run_apps_tests(code, row).score == 1.0


def test_apps_call_based_supports_official_listnode_cycle_convention():
    code = (
        "class Solution:\n"
        "    def hasCycle(self, head: ListNode) -> bool:\n"
        "        if head is None:\n"
        "            return False\n"
        "        slow, fast = head, head.next\n"
        "        while slow is not fast:\n"
        "            if fast is None or fast.next is None:\n"
        "                return False\n"
        "            slow, fast = slow.next, fast.next.next\n"
        "        return True"
    )
    row = _apps_row(
        {
            "fn_name": "hasCycle",
            "inputs": [
                [[3, 2, 0, -4], 1],
                [[1], -1],
            ],
            "outputs": [[True], [False]],
        },
        answer=code,
        starter_code=(
            "# Definition for singly-linked list.\n"
            "# class ListNode:\n"
            "#     def __init__(self, x):\n"
            "#         self.val = x\n"
            "#         self.next = None\n"
            "class Solution:\n"
            "    def hasCycle(self, head: ListNode) -> bool:\n"
            "        pass"
        ),
    )

    result = generation._run_apps_tests(code, row)

    assert result.score == 1.0
    assert result.tests_executed == 2


def test_apps_call_based_supports_official_two_sum_argument_wrapping():
    code = (
        "class Solution:\n"
        "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
        "        seen = {}\n"
        "        for index, value in enumerate(nums):\n"
        "            if target - value in seen:\n"
        "                return [seen[target - value], index]\n"
        "            seen[value] = index"
    )
    row = _apps_row(
        {
            "fn_name": "twoSum",
            "inputs": [
                [[2, 7, 11, 15], [9]],
                [[3, 2, 4], [6]],
            ],
            "outputs": [[0, 1], [1, 2]],
        },
        answer=code,
        starter_code=(
            "class Solution:\n"
            "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
            "        pass"
        ),
    )

    assert generation._run_apps_tests(code, row).score == 1.0


def test_apps_call_based_restores_json_integer_dictionary_keys():
    code = (
        "class Solution:\n"
        "    def inspect(self, mapping):\n"
        "        return type(next(iter(mapping))).__name__"
    )
    row = _apps_row(
        {
            "fn_name": "inspect",
            "inputs": [[{"1": "value"}]],
            "outputs": ["int"],
        },
        answer=code,
        starter_code=(
            "class Solution:\n"
            "    def inspect(self, mapping):\n"
            "        pass"
        ),
    )

    assert generation._run_apps_tests(code, row).score == 1.0


def test_apps_prelude_exposes_numpy_and_single_thread_runtime_controls(
    monkeypatch,
):
    # Importing NumPy can exceed the production problem budget when the full
    # test suite saturates shared storage; this test targets prelude semantics.
    monkeypatch.setenv("SLM_APPS_PROBLEM_TIMEOUT_S", "30")
    code = (
        "import os\n"
        "class Solution:\n"
        "    def runtime(self):\n"
        "        return np.array([1, 2]), [\n"
        "            os.environ.get('OPENBLAS_NUM_THREADS'),\n"
        "            os.environ.get('OMP_NUM_THREADS'),\n"
        "            os.environ.get('MKL_NUM_THREADS'),\n"
        "        ]"
    )
    row = _apps_row(
        {
            "fn_name": "runtime",
            "inputs": [[]],
            "outputs": [[[1, 2], ["1", "1", "1"]]],
        },
        answer=code,
        starter_code="class Solution:\n    def runtime(self):\n        pass",
    )

    assert generation._run_apps_tests(code, row).score == 1.0


def test_apps_prelude_restores_legacy_fractions_gcd_alias():
    code = (
        "from fractions import gcd\n"
        "class Solution:\n"
        "    def common(self, left, right):\n"
        "        return gcd(left, right)"
    )
    row = _apps_row(
        {
            "fn_name": "common",
            "inputs": ["[8, 12]"],
            "outputs": ["4"],
        },
        answer=code,
        starter_code=(
            "class Solution:\n"
            "    def common(self, left, right):\n"
            "        pass"
        ),
    )

    assert generation._run_apps_tests(code, row).score == 1.0


def test_apps_call_based_accepts_singleton_wrapper_for_structured_actual():
    code = (
        "class Solution:\n"
        "    def pair(self, value):\n"
        "        return (value, value + 1)"
    )
    row = _apps_row(
        {
            "fn_name": "pair",
            "inputs": ["[4]"],
            "outputs": [[[4, 5]]],
        },
        answer=code,
        starter_code="class Solution:\n    def pair(self, value):\n        pass",
    )

    assert generation._run_apps_tests(code, row).score == 1.0


def test_apps_call_based_rejects_scalar_for_expected_list():
    code = (
        "class Solution:\n"
        "    def scalar(self, value):\n"
        "        return value"
    )
    row = _apps_row(
        {
            "fn_name": "scalar",
            "inputs": ["[4]"],
            "outputs": [[4]],
        },
        answer=code,
        starter_code=(
            "class Solution:\n"
            "    def scalar(self, value):\n"
            "        pass"
        ),
    )

    assert generation._run_apps_tests(code, row).score == 0.0


def test_apps_stdin_gold_solution_scores_one():
    code = "a, b = map(int, input().split())\nprint(a + b)"
    row = _apps_row(
        {
            "inputs": ["2 3\n", "10 -4\n"],
            "outputs": ["5\n", "6\n"],
        },
        answer=code,
        starter_code="",
    )

    result = generation._run_apps_tests(code, row)

    assert result.score == 1.0
    assert result.tests_used == 2
    assert result.tests_total == 2


def test_apps_stdin_output_normalization_is_deterministic():
    row = _apps_row(
        {"inputs": ["ignored\n"], "outputs": ["1  2\n3\n"]},
        answer="print('1 2')\nprint('3')",
        starter_code="",
    )

    result = generation._run_apps_tests(
        "print('  1   2  ')\nprint('3   ')\nprint()",
        row,
    )

    assert result.score == 1.0


def test_apps_stdin_tolerates_equivalent_line_tokenization():
    row = _apps_row(
        {"inputs": ["ignored\n"], "outputs": ["1 2\n3 4\n"]},
        answer="print('1')\nprint('2 3')\nprint('4')",
        starter_code="",
    )

    assert generation._run_apps_tests(row["answer"], row).score == 1.0


def test_apps_stdin_ignores_stderr_when_stdout_and_exit_are_correct():
    row = _apps_row(
        {"inputs": ["\n"], "outputs": ["ok\n"]},
        answer="import sys\nprint('debug', file=sys.stderr)\nprint('ok')",
        starter_code="",
    )

    result = generation._run_apps_tests(row["answer"], row)

    assert result.score == 1.0
    assert result.returncode == 0


def test_apps_stdin_joins_official_list_outputs_and_compares_float_tokens():
    row = _apps_row(
        {
            "inputs": ["ignored\n"],
            "outputs": [["value 1.000000", "done 2.0"]],
        },
        answer="print('value 1.0000001')\nprint('done 2.0000001')",
        starter_code="",
    )

    result = generation._run_apps_tests(row["answer"], row)

    assert result.score == 1.0


@pytest.mark.parametrize(
    ("prediction", "diagnostic"),
    [
        (
            "class Solution:\n"
            "    def add(self, a, b):\n"
            "        return a - b",
            "case",
        ),
        (
            "class Solution:\n"
            "    def add(self, a, b):\n"
            "        raise RuntimeError('boom')",
            "RuntimeError",
        ),
        ("class Solution:\n    def add(:\n        pass", "SyntaxError"),
    ],
)
def test_apps_wrong_error_and_syntax_fail_closed(prediction, diagnostic):
    row = _apps_row(
        {"fn_name": "add", "inputs": ["[2, 3]"], "outputs": ["5"]}
    )

    result = generation._run_apps_tests(prediction, row)

    assert result.score == 0.0
    assert result.reason in {
        "wrong_output",
        "candidate_error",
        "compile_or_entrypoint",
    }
    assert diagnostic.lower() in result.diagnostic.lower()


def test_apps_timeout_kills_disposable_process_group():
    script = """
from eval.scorers import generation

row = {
    "text": "loop",
    "input_output": {"inputs": ["1\\n"], "outputs": ["1\\n"]},
    "execution_mode": "stdin",
}
result = generation._run_apps_tests(
    "while True:\\n    pass",
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
        timeout=3,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "input_output",
    [
        {},
        {"inputs": [], "outputs": []},
        {"inputs": ["1\n"], "outputs": []},
        {"inputs": "1\n", "outputs": ["1\n"]},
    ],
)
def test_apps_missing_or_invalid_tests_fail_closed(input_output):
    row = _apps_row(input_output, starter_code="")

    result = generation._run_apps_tests("print(1)", row)

    assert result.score == 0.0
    assert "test" in result.diagnostic.lower()


def test_apps_executes_every_case_even_when_legacy_cap_is_set(monkeypatch):
    monkeypatch.setenv("SLM_APPS_MAX_CASES", "3")
    row = _apps_row(
        {
            "fn_name": "identity",
            "inputs": [f"[{index}]" for index in range(12)],
            "outputs": [str(index) for index in range(12)],
        },
        answer=(
            "class Solution:\n"
            "    def identity(self, value):\n"
            "        return -1 if value == 1 else value"
        ),
        starter_code=(
            "class Solution:\n"
            "    def identity(self, value):\n"
            "        pass"
        ),
    )

    result = generation._run_apps_tests(row["answer"], row)

    assert result.score == 0.0
    assert result.tests_used == 2
    assert result.tests_total == 12


def test_apps_success_records_every_case_executed():
    row = _apps_row(
        {
            "fn_name": "identity",
            "inputs": [f"[{index}]" for index in range(25)],
            "outputs": [str(index) for index in range(25)],
        },
        answer=(
            "class Solution:\n"
            "    def identity(self, value):\n"
            "        return value"
        ),
        starter_code=(
            "class Solution:\n"
            "    def identity(self, value):\n"
            "        pass"
        ),
    )

    result = generation._run_apps_tests(row["answer"], row)

    assert result.score == 1.0
    assert result.tests_executed == 25
    assert result.tests_total == 25


def test_apps_timeout_is_per_case_not_shared_across_suite():
    row = _apps_row(
        {
            "fn_name": "wait_and_echo",
            "inputs": ["[1]", "[2]", "[3]"],
            "outputs": ["1", "2", "3"],
        },
        answer=(
            "import time\n"
            "class Solution:\n"
            "    def wait_and_echo(self, value):\n"
            "        time.sleep(0.12)\n"
            "        return value"
        ),
        starter_code=(
            "class Solution:\n"
            "    def wait_and_echo(self, value):\n"
            "        pass"
        ),
    )

    result = generation._run_apps_tests(
        row["answer"],
        row,
        timeout_seconds=0.2,
    )

    assert result.score == 1.0
    assert result.tests_used == 3


def test_apps_problem_wall_budget_exhaustion_is_distinct_from_case_timeout():
    row = _apps_row(
        {
            "inputs": ["1\n", "2\n", "3\n", "4\n"],
            "outputs": ["1\n", "2\n", "3\n", "4\n"],
        },
        answer=(
            "import time\n"
            "time.sleep(0.2)\n"
            "print(input().strip())"
        ),
        starter_code="",
    )

    result = generation._run_apps_tests(
        row["answer"],
        row,
        timeout_seconds=0.3,
        total_timeout_seconds=0.55,
    )

    assert result.score == 0.0
    assert result.reason == "problem_budget_exhausted"
    assert result.problem_budget_exhausted is True
    assert result.timed_out is False
    assert 0 < result.tests_executed < result.tests_total
    assert result.tests_total == 4
    assert "total wall budget" in result.diagnostic


def test_apps_default_problem_budget_bounds_800_row_pathology_under_90m(
    monkeypatch,
):
    monkeypatch.delenv("SLM_APPS_PROBLEM_TIMEOUT_S", raising=False)

    budget = generation._bounded_apps_problem_timeout(None)

    assert budget == 6.0
    assert budget * 800 < 90 * 60


def test_apps_stdout_comparison_preserves_order_and_duplicate_multiplicity():
    ordered = _apps_row(
        {"inputs": ["\n"], "outputs": ["1 2 1\n"]},
        answer="print('1 1 2')",
        starter_code="",
    )
    missing_duplicate = _apps_row(
        {"inputs": ["\n"], "outputs": ["1 1 2\n"]},
        answer="print('1 2 2')",
        starter_code="",
    )

    assert generation._run_apps_tests(ordered["answer"], ordered).score == 0.0
    assert (
        generation._run_apps_tests(
            missing_duplicate["answer"],
            missing_duplicate,
        ).score
        == 0.0
    )


def test_apps_nonunique_tree_vertices_rule_is_explicitly_scoped():
    alternate = "print(5)\nprint('1 6 8')"
    tree_input = (
        "8\n1 2\n2 3\n3 4\n4 5\n4 6\n3 7\n3 8\n"
    )
    official_row = _apps_row(
        {"inputs": [tree_input], "outputs": ["5\n1 8 6\n"]},
        answer=alternate,
        starter_code="",
        problem_id=4000,
        url="https://codeforces.com/problemset/problem/1294/F",
    )
    generic_row = _apps_row(
        {"inputs": [tree_input], "outputs": ["5\n1 8 6\n"]},
        answer=alternate,
        starter_code="",
        problem_id=9999,
    )

    assert generation._run_apps_tests(alternate, official_row).score == 1.0
    assert generation._run_apps_tests(alternate, generic_row).score == 0.0


@pytest.mark.parametrize("mode", ["call_based", "stdin"])
def test_apps_candidate_cannot_spoof_runner_success_fd(mode):
    attack = (
        "import os\n"
        "for fd in range(3, 256):\n"
        "    try:\n"
        "        os.write(fd, b'APPS_TESTS_PASSED\\n')\n"
        "    except OSError:\n"
        "        pass\n"
        "os._exit(0)\n"
    )
    input_output = (
        {"fn_name": "add", "inputs": ["[2, 3]"], "outputs": ["5"]}
        if mode == "call_based"
        else {"inputs": ["2 3\n"], "outputs": ["5\n"]}
    )
    row = _apps_row(input_output, answer=attack, starter_code="")

    assert generation._run_apps_tests(attack, row).score == 0.0


def test_apps_candidate_cannot_inspect_runner_payload_or_expected_outputs():
    attack = (
        "import inspect\n"
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        frame = inspect.currentframe()\n"
        "        while frame:\n"
        "            for value in frame.f_locals.values():\n"
        "                if isinstance(value, dict) and ('cases' in value or 'outputs' in value):\n"
        "                    return 5\n"
        "            frame = frame.f_back\n"
        "        return 0\n"
    )
    row = _apps_row(
        {"fn_name": "add", "inputs": ["[2, 3]"], "outputs": ["5"]},
        answer=attack,
    )

    assert generation._run_apps_tests(attack, row).score == 0.0


def test_apps_candidate_sees_sanitized_argv_without_result_channel():
    code = (
        "import sys\n"
        "class Solution:\n"
        "    def argv(self):\n"
        "        return sys.argv\n"
    )
    row = _apps_row(
        {"fn_name": "argv", "inputs": ["[]"], "outputs": [["candidate.py"]]},
        answer=code,
        starter_code="class Solution:\n    def argv(self):\n        pass",
    )

    assert generation._run_apps_tests(code, row).score == 1.0


def test_apps_candidate_cannot_forge_payload_or_result_files():
    attack = (
        "from pathlib import Path\n"
        "Path('payload.json').write_text('{}')\n"
        "Path('result.json').write_text("
        "'{\"success\": true, \"score\": 1.0}')\n"
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        return 0\n"
    )
    row = _apps_row(
        {"fn_name": "add", "inputs": ["[2, 3]"], "outputs": ["5"]},
        answer=attack,
    )

    assert generation._run_apps_tests(attack, row).score == 0.0


def test_apps_score_records_case_diagnostics_on_failure():
    row = _apps_row(
        {
            "fn_name": "add",
            "inputs": ["[2, 3]", "[4, 5]"],
            "outputs": ["5", "9"],
        }
    )

    result = _score_one(
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        return 0",
        row,
    )

    failure = result["failures"][0]
    assert failure["execution_tests_used"] == 1
    assert failure["execution_tests_executed"] == 1
    assert failure["execution_tests_total"] == 2
    assert result["execution_diagnostics"][0]["tests_used"] == 1
    assert result["execution_diagnostics"][0]["tests_executed"] == 1
    assert result["execution_diagnostics"][0]["tests_total"] == 2


def test_apps_score_reports_problem_budget_exhaustion_distinctly():
    row = _apps_row(
        {
            "fn_name": "add",
            "inputs": ["[2, 3]", "[4, 5]"],
            "outputs": ["5", "9"],
        }
    )
    exhausted = generation._CodeExecutionResult(
        score=0.0,
        diagnostic="APPS per-problem total wall budget exhausted",
        reason="problem_budget_exhausted",
        problem_budget_exhausted=True,
        tests_used=1,
        tests_total=2,
    )

    with patch(
        "eval.scorers.generation._run_apps_tests",
        return_value=exhausted,
    ):
        result = _score_one(row["answer"], row)

    failure = result["failures"][0]
    assert failure["execution_reason"] == "problem_budget_exhausted"
    assert failure["execution_problem_budget_exhausted"] is True
    assert failure["execution_timed_out"] is False
    assert (
        result["execution_diagnostics"][0]["problem_budget_exhausted"]
        is True
    )


def test_apps_call_prompt_includes_starter_code_and_required_entry_point():
    answer = (
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        return a + b"
    )
    row = _apps_row(
        {"fn_name": "add", "inputs": ["[2, 3]"], "outputs": ["5"]},
        answer=answer,
    )
    eval_set = EvalSet(all=[row], task_type="code_generation")

    prompt = generation.build_prompts(eval_set)[0]

    assert row["starter_code"] in prompt
    assert "Required entry point: add" in prompt
    assert "call-based" in prompt.lower()
    assert "return a + b" not in prompt
    assert "reasoning" in prompt.lower()
    assert "python comments" in prompt.lower()


def test_apps_stdin_prompt_requires_standard_input_and_output():
    row = _apps_row(
        {"inputs": ["2 3\n"], "outputs": ["5\n"]},
        answer="a, b = map(int, input().split())\nprint(a + b)",
        starter_code="",
    )

    prompt = generation.build_code_prompt(row)

    assert "standard input" in prompt.lower()
    assert "standard output" in prompt.lower()
    assert "call-based" not in prompt.lower()
