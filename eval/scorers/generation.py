import ast
import io
import json
import math
import os
import re
import tokenize
from dataclasses import dataclass

from data.eval_set import EvalSet
from eval.judge_client import LocalJudgeClient

GENERATE_PROMPT = "Answer the following question:\n\n{text}"

# Fallback instruction for free-form generation. It suits a question-answering or math task,
# which is what it was written for, and it is actively WRONG for anything else in the family:
# on DialogSum it told the model to "answer" a chat transcript that asks nothing, so the model
# continued the conversation instead of summarizing it (B250). A dataset that is not
# question-answering must therefore state its own instruction via `_instruction` on its rows.
DEFAULT_GENERATION_INSTRUCTION = "Answer the following question:"

# Rows carry the instruction under a leading underscore so it is treated as metadata: it is
# excluded from the JSON schema shown to the synthesis teacher, so generated rows cannot invent
# or reword it.
INSTRUCTION_FIELD = "_instruction"


def resolve_generation_instruction(rows) -> str:
    """The single instruction shared by every row of one dataset.

    Resolved ONCE per dataset rather than per row, and deliberately so: synthetic rows are built
    fresh and do not carry `_instruction`, so a per-row lookup would silently give real and
    synthetic rows different prompts within the same training set. Taking the first instruction
    present makes the whole set consistent.

    This mirrors how classification stays consistent — `build_classify_prompt` derives the label
    list from whichever rows the caller holds, so training and eval agree without either side
    having to be told. Generation now derives its instruction the same way.
    """
    for row in rows or []:
        if isinstance(row, dict):
            instruction = str(row.get(INSTRUCTION_FIELD) or "").strip()
            if instruction:
                return instruction
    return DEFAULT_GENERATION_INSTRUCTION


def build_generation_prompt(text: str, instruction: str) -> str:
    """The ONE generation prompt. Used by the eval harness AND the trainer.

    Before this existed the two built their inputs independently: eval wrapped the text in
    "Answer the following question:", while training passed the bare text with no instruction at
    all. The model was therefore fine-tuned on one input distribution and scored on another —
    the train/serve skew that `build_classify_prompt` had already been introduced to prevent on
    the classification side (B250).
    """
    return f"{instruction}\n\n{text}"

CODE_GENERATE_PROMPT = (
    "Solve the Python programming task below. Return only executable Python code, "
    "without Markdown fences or prose explanation. If you include concise "
    "implementation reasoning, write it only as Python comments so the complete "
    "response remains executable.\n\n"
    "Task:\n{text}{requirements}"
)

_DEFAULT_CODE_TIMEOUT_SECONDS = 3.0
_MAX_CODE_TIMEOUT_SECONDS = 30.0
_DEFAULT_APPS_PROBLEM_TIMEOUT_SECONDS = 6.0
_MAX_APPS_PROBLEM_TIMEOUT_SECONDS = 300.0
_CODE_MEMORY_BYTES = 512 * 1024 * 1024
_CODE_FILE_BYTES = 1024 * 1024


def _function_signature(source: object, entry_point: str = "") -> str:
    """Return a starter-code function/method signature without its body."""
    if not isinstance(source, str) or not source.strip():
        return ""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ""
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    function = next(
        (node for node in functions if entry_point and node.name == entry_point),
        functions[0] if functions else None,
    )
    if function is None:
        return ""

    lines = source.splitlines(keepends=True)

    def offset(position: tuple[int, int]) -> int:
        line, column = position
        return sum(len(value) for value in lines[: line - 1]) + column

    start = (function.lineno, function.col_offset)
    return_end = (
        (function.returns.end_lineno, function.returns.end_col_offset)
        if function.returns is not None
        else None
    )
    saw_open = False
    paren_depth = 0
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.start < start:
                continue
            if token.string == "(":
                saw_open = True
                paren_depth += 1
            elif token.string == ")" and saw_open:
                paren_depth -= 1
            elif (
                token.string == ":"
                and saw_open
                and paren_depth == 0
                and (return_end is None or token.start >= return_end)
            ):
                return source[offset(start):offset(token.end)].strip()
    except (tokenize.TokenError, IndentationError):
        pass

    prefix = "async def" if isinstance(function, ast.AsyncFunctionDef) else "def"
    returns = (
        f" -> {ast.unparse(function.returns)}"
        if function.returns is not None
        else ""
    )
    return (
        f"{prefix} {function.name}({ast.unparse(function.args)})"
        f"{returns}:"
    )


def _required_signature(example: dict) -> str:
    for key in ("signature", "function_signature"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    entry_point = ""
    for key in ("entry_point", "entrypoint", "function_name", "fn_name"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            entry_point = value.strip()
            break
    # Hidden gold answer/code is deliberately excluded.
    return _function_signature(example.get("starter_code"), entry_point)


def _entry_point(example: dict, signature: str = "") -> str:
    for key in ("entry_point", "entrypoint", "function_name", "fn_name"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    match = re.match(
        r"\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(",
        signature,
    )
    return match.group(1) if match else ""


def _code_execution_mode(example: dict) -> str:
    mode = example.get("execution_mode")
    if mode in {"call_based", "stdin"}:
        return mode
    input_output = example.get("input_output")
    if isinstance(input_output, dict):
        fn_name = input_output.get("fn_name")
        return (
            "call_based"
            if isinstance(fn_name, str) and fn_name.strip()
            else "stdin"
        )
    if example.get("test_list"):
        return "call_based"
    return ""


def build_code_prompt(example: dict) -> str:
    """Build one shared train/eval prompt without inspecting hidden gold."""
    signature = _required_signature(example)
    entry_point = _entry_point(example, signature)
    mode = _code_execution_mode(example)
    requirements = []
    if mode == "call_based":
        requirements.append(
            "This is a call-based task. Define the requested callable and return "
            "values directly; do not read from standard input or print the result."
        )
    elif mode == "stdin":
        requirements.append(
            "This is a standard-input task. Read the complete input from standard "
            "input and write only the required answer to standard output."
        )
    if signature:
        requirements.append(f"Required function signature: {signature}")
    if entry_point:
        requirements.append(f"Required entry point: {entry_point}")
    starter_code = example.get("starter_code")
    if isinstance(starter_code, str) and starter_code.strip():
        requirements.append(
            "Starter code (preserve its required interface and complete it):\n"
            f"{starter_code.strip()}"
        )
    suffix = "\n\n" + "\n".join(requirements) if requirements else ""
    return CODE_GENERATE_PROMPT.format(
        text=example.get("text", example.get("prompt", "")),
        requirements=suffix,
    )


def build_prompts(eval_set: EvalSet) -> list[str]:
    if eval_set.task_type != "code_generation":
        instruction = resolve_generation_instruction(eval_set.all)
        return [
            build_generation_prompt(example.get("text", ""), instruction)
            for example in eval_set.all
        ]
    return [build_code_prompt(example) for example in eval_set.all]


_FENCED_CODE_RE = re.compile(
    r"```[ \t]*([^\n`]*)\r?\n(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _extract_code(raw: str) -> str:
    text = raw.strip()
    fenced = _FENCED_CODE_RE.findall(text)
    if not fenced:
        return text
    for language, body in fenced:
        if language.strip().lower() in {"python", "python3", "py"}:
            return body.strip()
    for language, body in fenced:
        if not language.strip():
            return body.strip()
    return fenced[0][1].strip()


# CoT-annotated training targets are `<reasoning>...</reasoning>\n\n<answer>` (see
# training/lora_trainer.py::_training_turn), so a model trained on them emits the reasoning
# inline. Tolerant of whitespace/case and of a missing opening tag, which small models drop.
_REASONING_BLOCK_RE = re.compile(
    r"\s*<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
_ORPHAN_REASONING_CLOSE_RE = re.compile(
    r"^.*?<\s*/\s*reasoning\s*>\s*", flags=re.IGNORECASE | re.DOTALL
)


def split_reasoning(raw: str) -> tuple[str, str]:
    """Split a raw generation into ``(reasoning, answer)``.

    Both halves are kept by the caller: the reasoning is real signal for diagnosing HOW the model
    reached an answer, and discarding it at the source would make that unrecoverable. It simply
    must not reach the judge, which is asked "how good is this summary?" — handing it a
    `<reasoning>` block guarantees a poor score for output that may contain a fine summary.

    Math and code never needed this because their extractors already pull one specific thing (the
    final number, the fenced code block). Judge-scored generation extracts nothing, so the whole
    string was being graded (B251).
    """
    text = raw or ""
    match = _REASONING_BLOCK_RE.search(text)
    if match:
        reasoning = match.group(0)
        answer = (text[: match.start()] + text[match.end():]).strip()
        # A model that emits ONLY reasoning has no answer to judge; keep the text rather than
        # hand the judge an empty string, which would score 0 and hide the real failure.
        return reasoning.strip(), (answer or text.strip())
    orphan = _ORPHAN_REASONING_CLOSE_RE.match(text)
    if orphan:
        answer = text[orphan.end():].strip()
        if answer:
            return orphan.group(0).strip(), answer
    return "", text.strip()


def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]:
    if eval_set.task_type == "code_generation":
        return [_extract_code(raw) for raw in raw_outputs]
    return [split_reasoning(raw)[1] for raw in raw_outputs]


def _final_answer(value: str) -> str:
    """Extract a normalized final numeric answer for math exact match."""
    if value is None:
        return ""
    text = value.strip()
    match = re.search(
        r"(?:####|answer\s*(?:is|:)?)\s*\$?"
        r"(-?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        answer = match.group(1)
    else:
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        if not numbers:
            return text.lower()
        answer = numbers[-1]
    answer = answer.replace(",", "").lstrip("$")
    if re.fullmatch(r"-?\d+\.0+", answer):
        answer = answer.split(".")[0]
    return answer


def _exact_match(gold: str, prediction: str) -> float:
    gold_answer = _final_answer(gold)
    predicted_answer = _final_answer(prediction)
    return (
        1.0
        if gold_answer != "" and gold_answer == predicted_answer
        else 0.0
    )


@dataclass(frozen=True)
class _CodeExecutionResult:
    score: float
    diagnostic: str
    reason: str = ""
    timed_out: bool = False
    problem_budget_exhausted: bool = False
    returncode: int | None = None
    tests_used: int = 0
    tests_total: int = 0

    @property
    def tests_executed(self) -> int:
        return self.tests_used


def _configured_code_timeout() -> float:
    try:
        value = float(
            os.environ.get(
                "SLM_CODE_EVAL_TIMEOUT_S",
                str(_DEFAULT_CODE_TIMEOUT_SECONDS),
            )
        )
    except (TypeError, ValueError):
        value = _DEFAULT_CODE_TIMEOUT_SECONDS
    if not math.isfinite(value):
        value = _DEFAULT_CODE_TIMEOUT_SECONDS
    return min(max(value, 0.1), _MAX_CODE_TIMEOUT_SECONDS)


def _bounded_timeout(timeout_seconds: float | None) -> float:
    if timeout_seconds is None:
        return _configured_code_timeout()
    try:
        value = float(timeout_seconds)
    except (TypeError, ValueError):
        value = _DEFAULT_CODE_TIMEOUT_SECONDS
    if not math.isfinite(value):
        value = _DEFAULT_CODE_TIMEOUT_SECONDS
    return min(max(value, 0.1), _MAX_CODE_TIMEOUT_SECONDS)


def _bounded_apps_problem_timeout(
    timeout_seconds: float | None,
) -> float:
    raw_value = (
        os.environ.get(
            "SLM_APPS_PROBLEM_TIMEOUT_S",
            str(_DEFAULT_APPS_PROBLEM_TIMEOUT_SECONDS),
        )
        if timeout_seconds is None
        else timeout_seconds
    )
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = _DEFAULT_APPS_PROBLEM_TIMEOUT_SECONDS
    if not math.isfinite(value):
        value = _DEFAULT_APPS_PROBLEM_TIMEOUT_SECONDS
    return min(max(value, 0.1), _MAX_APPS_PROBLEM_TIMEOUT_SECONDS)


def _invalid_code_result(
    message: str,
    *,
    reason: str = "invalid_metadata",
    tests_used: int = 0,
    tests_total: int = 0,
) -> _CodeExecutionResult:
    return _CodeExecutionResult(
        score=0.0,
        diagnostic=message,
        reason=reason,
        tests_used=tests_used,
        tests_total=tests_total,
    )


def _test_derived_entry_point(tests: list[str]) -> str:
    for test in tests:
        try:
            tree = ast.parse(test)
        except (SyntaxError, ValueError):
            continue
        call = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
            ),
            None,
        )
        if call is not None:
            return call.func.id
    return ""


def _run_mbpp_tests(
    code: str,
    example: dict,
    *,
    timeout_seconds: float | None = None,
) -> _CodeExecutionResult:
    """Run MBPP tests while candidate code remains in an isolated worker."""
    if not isinstance(code, str) or not code.strip():
        return _invalid_code_result("prediction is empty")
    tests = example.get("test_list")
    if not isinstance(tests, (list, tuple)) or not tests:
        return _invalid_code_result("MBPP row has no executable tests")
    if not all(isinstance(test, str) and test.strip() for test in tests):
        return _invalid_code_result(
            "MBPP row contains invalid test_list metadata"
        )
    imports = example.get("test_imports", [])
    if imports is None:
        imports = []
    if not isinstance(imports, (list, tuple)) or not all(
        isinstance(statement, str) and statement.strip()
        for statement in imports
    ):
        return _invalid_code_result(
            "MBPP row contains invalid test_imports metadata"
        )
    signature = _required_signature(example)
    entry_point = (
        _entry_point(example, signature)
        or _test_derived_entry_point(list(tests))
    )
    if not entry_point:
        return _invalid_code_result(
            "MBPP row has no explicit or test-derived entry point",
            tests_total=len(tests),
        )

    from eval.scorers.code_execution import run_mbpp

    outcome = run_mbpp(
        code,
        tests=list(tests),
        imports=list(imports),
        entry_point=entry_point,
        timeout_seconds=_bounded_timeout(timeout_seconds),
        memory_bytes=_CODE_MEMORY_BYTES,
        file_bytes=_CODE_FILE_BYTES,
    )
    return _CodeExecutionResult(
        score=outcome.score,
        diagnostic=outcome.diagnostic,
        reason=outcome.reason,
        timed_out=outcome.timed_out,
        returncode=outcome.returncode,
        tests_used=outcome.tests_executed,
        tests_total=outcome.tests_total,
    )


def _run_apps_tests(
    code: str,
    example: dict,
    *,
    timeout_seconds: float | None = None,
    total_timeout_seconds: float | None = None,
    max_cases: int | None = None,
) -> _CodeExecutionResult:
    """Run every preserved APPS case with an independent case timeout."""
    del max_cases  # Removed cap; retained only for compatibility with old callers.
    from data.loaders.apps import valid_apps_input_output

    if not isinstance(code, str) or not code.strip():
        return _invalid_code_result("prediction is empty")
    input_output = example.get("input_output")
    if not valid_apps_input_output(input_output):
        return _invalid_code_result(
            "APPS row has no executable tests or contains invalid test metadata"
        )
    tests_total = len(input_output["inputs"])
    mode = _code_execution_mode(example)
    fn_name = input_output.get("fn_name")
    expected_mode = "call_based" if fn_name else "stdin"
    if mode != expected_mode:
        return _invalid_code_result(
            f"APPS execution mode mismatch: metadata requires "
            f"{expected_mode!r}, got {mode!r}",
            tests_total=tests_total,
        )
    if mode == "call_based" and (
        not isinstance(fn_name, str) or not fn_name.strip()
    ):
        return _invalid_code_result(
            "APPS call-based row has no fn_name",
            tests_total=tests_total,
        )
    try:
        json.dumps(input_output)
    except (TypeError, ValueError) as error:
        return _invalid_code_result(
            f"APPS tests are not JSON-serializable: {error}",
            tests_total=tests_total,
        )

    from eval.scorers.code_execution import run_apps

    problem_id = example.get("problem_id")
    url = str(example.get("url") or "").rstrip("/")
    comparison_rule = (
        "codeforces_1294_f_unordered_vertices"
        if problem_id == 4000
        or url.endswith("/problem/1294/F")
        else ""
    )
    outcome = run_apps(
        code,
        input_output=input_output,
        mode=mode,
        fn_name=fn_name.strip() if isinstance(fn_name, str) else "",
        starter_code=str(example.get("starter_code") or ""),
        comparison_rule=comparison_rule,
        # APPS gold/test cases may import NumPy or initialize BLAS on first use.
        # Apply the stable production floor only to the default; explicit caller
        # timeouts (including timeout regression tests) remain authoritative.
        timeout_seconds=(
            max(10.0, _bounded_timeout(None))
            if timeout_seconds is None
            else _bounded_timeout(timeout_seconds)
        ),
        total_timeout_seconds=_bounded_apps_problem_timeout(
            total_timeout_seconds
        ),
        memory_bytes=_CODE_MEMORY_BYTES,
        file_bytes=_CODE_FILE_BYTES,
    )
    return _CodeExecutionResult(
        score=outcome.score,
        diagnostic=outcome.diagnostic,
        reason=outcome.reason,
        timed_out=outcome.timed_out,
        problem_budget_exhausted=outcome.problem_budget_exhausted,
        returncode=outcome.returncode,
        tests_used=outcome.tests_executed,
        tests_total=outcome.tests_total,
    )


def _code_pass_at_1(code: str, example: dict | None = None) -> float:
    example = example or {}
    runner = _run_apps_tests if "input_output" in example else _run_mbpp_tests
    return runner(code, example).score


def score(eval_set: EvalSet, predictions: list[str]) -> dict:
    task_type = eval_set.task_type
    scores = []
    code_results: list[_CodeExecutionResult | None] = []

    if task_type == "math_reasoning":
        for example, prediction in zip(eval_set.all, predictions):
            scores.append(
                _exact_match(example.get("answer", ""), prediction)
            )
            code_results.append(None)
    elif task_type == "code_generation":
        for example, prediction in zip(eval_set.all, predictions):
            runner = (
                _run_apps_tests
                if "input_output" in example
                else _run_mbpp_tests
            )
            result = runner(prediction, example)
            scores.append(result.score)
            code_results.append(result)
    else:
        triples = [
            (
                example.get("text", ""),
                example.get("answer", example.get("label", "")),
                prediction,
            )
            for example, prediction in zip(eval_set.all, predictions)
        ]
        scores.extend(LocalJudgeClient.from_config().score_many(triples))
        code_results.extend([None] * len(triples))

    average = sum(scores) / len(scores) if scores else 0.0
    failures = []
    for example, prediction, value, execution in zip(
        eval_set.all,
        predictions,
        scores,
        code_results,
    ):
        if value >= 0.5:
            continue
        failure = {
            **example,
            "predicted": prediction,
            "judge_score": value,
        }
        if execution is not None:
            failure.update(
                {
                    "execution_diagnostic": execution.diagnostic,
                    "execution_reason": execution.reason,
                    "execution_timed_out": execution.timed_out,
                    "execution_problem_budget_exhausted": (
                        execution.problem_budget_exhausted
                    ),
                    "execution_tests_used": execution.tests_used,
                    "execution_tests_executed": execution.tests_executed,
                    "execution_tests_total": execution.tests_total,
                }
            )
        failures.append(failure)
    execution_diagnostics = [
        {
            "score": execution.score,
            "diagnostic": execution.diagnostic,
            "reason": execution.reason,
            "timed_out": execution.timed_out,
            "problem_budget_exhausted": execution.problem_budget_exhausted,
            "returncode": execution.returncode,
            "tests_used": execution.tests_used,
            "tests_executed": execution.tests_executed,
            "tests_total": execution.tests_total,
        }
        for execution in code_results
        if execution is not None
    ]
    # This scorer serves three task types with three genuinely different measurements, none
    # of which is an F1. `f1` stays as the pipeline's universal comparison scalar (renaming
    # it would break checkpoints and DAG replay), and `metric` says what it really is so a
    # report cannot present a judge mean or an execution pass-rate as an F1.
    metric_name = {
        "math_reasoning": "exact_match",
        "code_generation": "execution_pass@1",
    }.get(task_type, "judge_mean_0_1")
    return {
        "f1": average,
        "metric": metric_name,
        "per_class": {metric_name: average},
        "failures": failures,
        "execution_diagnostics": execution_diagnostics,
    }
