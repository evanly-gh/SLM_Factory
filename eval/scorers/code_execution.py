"""Trusted benchmark controller with isolated candidate workers.

The controller retains tests and expected outputs in the parent process. Candidate
code runs in a child that receives only one input at a time and has no success FD.
This is bounded trusted-benchmark isolation, not a hostile-code seccomp sandbox.

APPS behavior is adapted from codeparrot/apps_metric ``testing_util.py`` revision
9877be0aa476d466e0cf9e252191b3614a13b7cb: its common prelude, tuple/list
normalization, singleton expected wrappers, integer-key restoration, and stdin
token/numeric normalization. Dataset-specific ListNode/Two Sum repairs cover the
pinned introductory rows 4751/4752. We deliberately preserve output token order
and multiplicity instead of copying the reference's global unordered-set fallback.
"""

from __future__ import annotations

import ast
import json
import math
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


_WORKER_SOURCE = r'''
import base64
import builtins
import io
import json
import math
import os
import sys
import traceback

try:
    import resource
except ImportError:
    resource = None

_protocol_in = sys.stdin.buffer
_protocol_out = sys.stdout.buffer
_dumps = json.dumps
_loads = json.loads
_compile = compile
_exec = exec
_callable = callable
_BytesIO = io.BytesIO
_TextIOWrapper = io.TextIOWrapper
_BaseException = BaseException
_SystemExit = SystemExit


class ListNode:
    def __init__(self, value=0, next_node=None):
        self.val = value
        self.next = next_node


_PRELUDE = """
import sys
import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
fractions.gcd = math.gcd
from typing import *
import random
import heapq
from heapq import *
import bisect
from bisect import *
import re
import string
import importlib
class _LazyNumpy:
    _module = None
    def __getattr__(self, name):
        if self._module is None:
            self._module = importlib.import_module("numpy")
        return getattr(self._module, name)
np = _LazyNumpy()
"""


def _set_limit(name, value):
    if resource is None:
        return
    try:
        limit = getattr(resource, name)
        _, hard = resource.getrlimit(limit)
        bounded = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(limit, (bounded, bounded))
    except (AttributeError, OSError, ValueError):
        pass


def _decode(value):
    if not isinstance(value, dict) or "__slm_type__" not in value:
        if isinstance(value, list):
            return [_decode(item) for item in value]
        return value
    kind = value["__slm_type__"]
    items = value.get("items", [])
    if kind == "tuple":
        return tuple(_decode(item) for item in items)
    if kind == "set":
        return set(_decode(item) for item in items)
    if kind == "dict":
        return {
            _decode(pair[0]): _decode(pair[1])
            for pair in items
        }
    if kind == "bytes":
        return base64.b64decode(value["data"])
    if kind == "listnode":
        values = [_decode(item) for item in value.get("values", [])]
        nodes = [ListNode(item) for item in values]
        for index in range(len(nodes) - 1):
            nodes[index].next = nodes[index + 1]
        position = int(value.get("pos", -1))
        if nodes and 0 <= position < len(nodes):
            nodes[-1].next = nodes[position]
        return nodes[0] if nodes else None
    raise TypeError(f"unsupported wire type {kind!r}")


def _encode(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise TypeError("non-finite float result")
        return value
    if hasattr(value, "tolist") and _callable(value.tolist):
        value = value.tolist()
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, tuple):
        return {"__slm_type__": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, set):
        return {
            "__slm_type__": "set",
            "items": [_encode(item) for item in sorted(value, key=repr)],
        }
    if isinstance(value, dict):
        return {
            "__slm_type__": "dict",
            "items": [[_encode(key), _encode(item)] for key, item in value.items()],
        }
    if isinstance(value, bytes):
        return {
            "__slm_type__": "bytes",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, ListNode):
        items = []
        seen = set()
        current = value
        while current is not None and id(current) not in seen and len(items) < 10000:
            seen.add(id(current))
            items.append(_encode(current.val))
            current = current.next
        return items
    raise TypeError(f"unsupported result type {type(value).__name__}")


def _emit(value):
    _protocol_out.write(
        (_dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    )
    _protocol_out.flush()


def _namespace(name):
    namespace = {
        "__builtins__": dict(builtins.__dict__),
        "__name__": name,
        "__file__": "<prediction>",
        "ListNode": ListNode,
    }
    _exec(_PRELUDE, namespace, namespace)
    return namespace


def _load_call_candidate(source, imports_source, fn_name):
    namespace = _namespace("__candidate__")
    prior_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        if imports_source:
            _exec(
                _compile(imports_source, "<test_imports>", "exec"),
                namespace,
                namespace,
            )
        _exec(_compile(source, "<prediction>", "exec"), namespace, namespace)
    finally:
        sys.stdout = prior_stdout
    solution = namespace.get("Solution")
    candidate = (
        getattr(solution(), fn_name, None)
        if isinstance(solution, type)
        else namespace.get(fn_name)
    )
    if not _callable(candidate):
        raise TypeError(
            f"required entry point callable {fn_name!r} was not defined"
        )
    return candidate


def _run_stdin(source, raw_input):
    input_bytes = (
        raw_input.encode("utf-8")
        if isinstance(raw_input, str)
        else bytes(raw_input)
    )
    input_buffer = _BytesIO(input_bytes)
    output_buffer = _BytesIO()
    candidate_stdin = _TextIOWrapper(input_buffer, encoding="utf-8")
    candidate_stdout = _TextIOWrapper(
        output_buffer,
        encoding="utf-8",
        write_through=True,
    )
    prior_stdin, prior_stdout = sys.stdin, sys.stdout
    caught = None
    try:
        sys.stdin, sys.stdout = candidate_stdin, candidate_stdout
        namespace = _namespace("__main__")
        _exec(_compile(source, "<prediction>", "exec"), namespace, namespace)
    except _BaseException as error:
        if not isinstance(error, _SystemExit) or error.code not in (None, 0):
            caught = error
    finally:
        sys.stdin, sys.stdout = prior_stdin, prior_stdout
        try:
            candidate_stdout.flush()
        except _BaseException as error:
            caught = caught or error
    if caught is not None:
        raise caught
    return output_buffer.getvalue().decode("utf-8", errors="replace")


def _main():
    source_path, mode, fn_name, imports_path = sys.argv[1:5]
    memory_bytes, file_bytes, open_files, processes = map(int, sys.argv[5:9])
    _set_limit("RLIMIT_CORE", 0)
    _set_limit("RLIMIT_AS", memory_bytes)
    _set_limit("RLIMIT_FSIZE", file_bytes)
    _set_limit("RLIMIT_NOFILE", open_files)
    _set_limit("RLIMIT_NPROC", processes)

    source = open(source_path, encoding="utf-8").read()
    imports_source = open(imports_path, encoding="utf-8").read()
    # Candidate code sees no worker paths, result descriptor, or controller arguments.
    sys.argv = ["candidate.py"]

    candidate = None
    startup_error = None
    if mode == "call":
        try:
            candidate = _load_call_candidate(source, imports_source, fn_name)
        except _BaseException as error:
            startup_error = f"{type(error).__name__}: {error}"
    elif mode == "stdin":
        try:
            _namespace("__apps_prewarm__")
        except _BaseException as error:
            startup_error = f"{type(error).__name__}: {error}"

    for raw_line in _protocol_in:
        try:
            request = _loads(raw_line.decode("utf-8"))
            if startup_error:
                raise RuntimeError(startup_error)
            operation = request.get("op")
            if operation == "shutdown":
                _emit({"ok": True, "value": True})
                break
            if operation == "check":
                result = True
            elif operation == "call":
                prior_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    result = candidate(
                        *[_decode(item) for item in request.get("args", [])],
                        **{
                            key: _decode(item)
                            for key, item in request.get("kwargs", {}).items()
                        },
                    )
                finally:
                    sys.stdout = prior_stdout
            elif operation == "stdin":
                result = _run_stdin(source, request["input"])
            else:
                raise ValueError(f"unsupported operation {operation!r}")
            _emit({"ok": True, "value": _encode(result)})
        except _BaseException as error:
            _emit(
                {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=8),
                }
            )


_main()
'''


@dataclass(frozen=True)
class ExecutionOutcome:
    score: float
    diagnostic: str
    reason: str = ""
    timed_out: bool = False
    problem_budget_exhausted: bool = False
    returncode: int | None = None
    tests_executed: int = 0
    tests_total: int = 0


class _RemoteCandidateError(RuntimeError):
    pass


class _ProblemBudgetExhausted(RuntimeError):
    pass


def _wire_encode(value):
    import base64

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise TypeError("non-finite float input")
        return value
    if isinstance(value, list):
        return [_wire_encode(item) for item in value]
    if isinstance(value, tuple):
        return {
            "__slm_type__": "tuple",
            "items": [_wire_encode(item) for item in value],
        }
    if isinstance(value, set):
        return {
            "__slm_type__": "set",
            "items": [_wire_encode(item) for item in sorted(value, key=repr)],
        }
    if (
        isinstance(value, dict)
        and value.get("__slm_type__") == "listnode"
    ):
        return {
            "__slm_type__": "listnode",
            "values": [
                _wire_encode(item)
                for item in value.get("values", [])
            ],
            "pos": int(value.get("pos", -1)),
        }
    if isinstance(value, dict):
        return {
            "__slm_type__": "dict",
            "items": [
                [_wire_encode(key), _wire_encode(item)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, bytes):
        return {
            "__slm_type__": "bytes",
            "data": base64.b64encode(value).decode("ascii"),
        }
    raise TypeError(f"unsupported input type {type(value).__name__}")


def _wire_decode(value):
    import base64

    if not isinstance(value, dict) or "__slm_type__" not in value:
        if isinstance(value, list):
            return [_wire_decode(item) for item in value]
        return value
    kind = value["__slm_type__"]
    items = value.get("items", [])
    if kind == "tuple":
        return tuple(_wire_decode(item) for item in items)
    if kind == "set":
        return set(_wire_decode(item) for item in items)
    if kind == "dict":
        return {
            _wire_decode(pair[0]): _wire_decode(pair[1])
            for pair in items
        }
    if kind == "bytes":
        return base64.b64decode(value["data"])
    raise TypeError(f"unsupported result wire type {kind!r}")


def _kill_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        process.wait()
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class _CandidateController:
    def __init__(
        self,
        code: str,
        *,
        mode: str,
        fn_name: str = "",
        imports: list[str] | None = None,
        timeout_seconds: float,
        memory_bytes: int,
        file_bytes: int,
        open_files: int,
        processes: int,
    ):
        self._temporary = tempfile.TemporaryDirectory(prefix="slm-candidate-")
        root = Path(self._temporary.name)
        worker_path = root / "worker.py"
        candidate_path = root / "candidate.py"
        imports_path = root / "imports.py"
        diagnostics_path = root / "diagnostics.txt"
        worker_path.write_text(_WORKER_SOURCE, encoding="utf-8")
        candidate_path.write_text(code, encoding="utf-8")
        imports_path.write_text("\n".join(imports or []), encoding="utf-8")
        self._diagnostics = diagnostics_path.open("w+b")
        self._timeout = timeout_seconds
        environment = {
            "HOME": str(root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "",
            "TMPDIR": str(root),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        kwargs = {
            "cwd": str(root),
            "env": environment,
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": self._diagnostics,
            "close_fds": True,
            "bufsize": 0,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                str(worker_path),
                str(candidate_path),
                mode,
                fn_name,
                str(imports_path),
                str(memory_bytes),
                str(file_bytes),
                str(open_files),
                str(processes),
            ],
            **kwargs,
        )

    def diagnostic(self, limit: int = 32 * 1024) -> str:
        self._diagnostics.flush()
        self._diagnostics.seek(0)
        output = self._diagnostics.read(limit + 1)
        text = output[:limit].decode("utf-8", errors="replace").strip()
        if len(output) > limit:
            text += "\n[diagnostic output truncated]"
        return text

    def request(
        self,
        value: dict,
        *,
        timeout_seconds: float | None = None,
    ) -> object:
        if self.process.poll() is not None:
            raise _RemoteCandidateError(
                f"candidate exited before response (returncode={self.process.returncode})"
            )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        payload = (
            json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise _RemoteCandidateError(
                f"candidate request failed: {type(error).__name__}: {error}"
            ) from error
        ready, _, _ = select.select(
            [self.process.stdout.fileno()],
            [],
            [],
            self._timeout if timeout_seconds is None else timeout_seconds,
        )
        if not ready:
            timeout = (
                self._timeout
                if timeout_seconds is None
                else timeout_seconds
            )
            raise TimeoutError(
                f"candidate case timed out after {timeout:.2f} seconds"
            )
        raw = self.process.stdout.readline()
        if not raw:
            raise _RemoteCandidateError(
                f"candidate exited without response (returncode={self.process.poll()})"
            )
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _RemoteCandidateError(
                "candidate emitted an invalid response frame"
            ) from error
        if not isinstance(response, dict) or response.get("ok") is not True:
            message = (
                response.get("error", "candidate failed")
                if isinstance(response, dict)
                else "candidate returned invalid response"
            )
            raise _RemoteCandidateError(str(message))
        return _wire_decode(response.get("value"))

    def close(self) -> int | None:
        if self.process.poll() is None:
            try:
                self.request({"op": "shutdown"})
                self.process.wait(timeout=1)
            except (
                OSError,
                TimeoutError,
                _RemoteCandidateError,
                subprocess.TimeoutExpired,
            ):
                _kill_worker(self.process)
        else:
            self.process.wait()
        returncode = self.process.returncode
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self._diagnostics.close()
        self._temporary.cleanup()
        return returncode


def _parse_jsonish(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value.strip()


def _restore_json_integer_keys(value):
    if isinstance(value, list):
        return [_restore_json_integer_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_json_integer_keys(item) for item in value)
    if isinstance(value, dict):
        try:
            return {
                int(key): _restore_json_integer_keys(item)
                for key, item in value.items()
            }
        except (TypeError, ValueError):
            return {
                key: _restore_json_integer_keys(item)
                for key, item in value.items()
            }
    return value


def _listnode_parameter_indices(
    starter_code: str,
    fn_name: str,
) -> list[int]:
    if not starter_code or "ListNode" not in starter_code:
        return []
    try:
        tree = ast.parse(starter_code)
    except (SyntaxError, ValueError):
        return []
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == fn_name
        ),
        None,
    )
    if function is None:
        return []
    arguments = list(function.args.args)
    if arguments and arguments[0].arg in {"self", "cls"}:
        arguments = arguments[1:]
    indices = []
    for index, argument in enumerate(arguments):
        annotation = argument.annotation
        if (
            isinstance(annotation, ast.Name)
            and annotation.id == "ListNode"
        ) or (
            isinstance(annotation, ast.Constant)
            and annotation.value == "ListNode"
        ):
            indices.append(index)
    return indices


def _adapt_call_arguments(
    parsed_input,
    *,
    fn_name: str,
    starter_code: str,
) -> tuple[list, bool]:
    arguments = (
        list(parsed_input)
        if isinstance(parsed_input, list)
        else [parsed_input]
    )
    lower_name = fn_name.lower()
    if lower_name in {"twosum", "two_sum"} and len(arguments) == 2:
        if isinstance(arguments[1], list) and len(arguments[1]) == 1:
            arguments[1] = arguments[1][0]

    listnode_indices = _listnode_parameter_indices(
        starter_code,
        fn_name,
    )
    used_listnode = bool(listnode_indices)
    if lower_name == "hascycle" and listnode_indices and len(arguments) == 2:
        values, position = arguments
        arguments = [
            {
                "__slm_type__": "listnode",
                "values": values if isinstance(values, list) else [],
                "pos": position,
            }
        ]
    else:
        for index in listnode_indices:
            if index >= len(arguments):
                continue
            values = arguments[index]
            arguments[index] = {
                "__slm_type__": "listnode",
                "values": values if isinstance(values, list) else [],
                "pos": -1,
            }
    return arguments, used_listnode


def _structured_equal(actual, expected) -> bool:
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
    if type(actual) is not type(expected):
        # APPS treats tuples and lists as equivalent sequence containers.
        if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
            pass
        else:
            return False
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(
            _structured_equal(left, right)
            for left, right in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _structured_equal(actual[key], expected[key])
            for key in actual
        )
    if isinstance(actual, set) and isinstance(expected, set):
        return actual == expected
    return actual == expected


def _official_call_equal(actual, expected) -> bool:
    if _structured_equal(actual, expected):
        return True
    structured_actual = isinstance(
        actual,
        (list, tuple, dict, set),
    )
    return (
        structured_actual
        and isinstance(expected, list)
        and len(expected) == 1
        and _structured_equal(actual, expected[0])
    )


def _normalize_stdout(value) -> list[list[str]]:
    if isinstance(value, list):
        value = "\n".join(str(item) for item in value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return [line.strip().split() for line in lines]


def _stdout_equal(actual, expected) -> bool:
    actual_lines = _normalize_stdout(actual)
    expected_lines = _normalize_stdout(expected)
    if actual_lines == expected_lines:
        return True
    actual_tokens = [
        token
        for line in actual_lines
        for token in line
    ]
    expected_tokens = [
        token
        for line in expected_lines
        for token in line
    ]
    if len(actual_tokens) != len(expected_tokens):
        return False
    for actual_token, expected_token in zip(
        actual_tokens,
        expected_tokens,
    ):
        if actual_token == expected_token:
            continue
        try:
            actual_number = float(actual_token)
            expected_number = float(expected_token)
        except ValueError:
            return False
        if not math.isclose(
            actual_number,
            expected_number,
            rel_tol=1e-5,
            abs_tol=1e-8,
        ):
            return False
    return True


def _codeforces_1294_f_equal(actual, expected, raw_input) -> bool:
    try:
        actual_tokens = [
            int(token)
            for line in _normalize_stdout(actual)
            for token in line
        ]
        expected_tokens = [
            int(token)
            for line in _normalize_stdout(expected)
            for token in line
        ]
        input_tokens = [int(token) for token in str(raw_input).split()]
    except (TypeError, ValueError):
        return False
    if len(actual_tokens) != 4 or not expected_tokens or not input_tokens:
        return False
    score, first, second, third = actual_tokens
    node_count = input_tokens[0]
    if (
        score != expected_tokens[0]
        or len({first, second, third}) != 3
        or not all(1 <= node <= node_count for node in (first, second, third))
        or len(input_tokens) != 1 + 2 * (node_count - 1)
    ):
        return False
    adjacency = [[] for _ in range(node_count + 1)]
    edge_values = input_tokens[1:]
    for index in range(0, len(edge_values), 2):
        left, right = edge_values[index:index + 2]
        adjacency[left].append(right)
        adjacency[right].append(left)

    def distance(start: int, goal: int) -> int:
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            node, depth = queue.popleft()
            if node == goal:
                return depth
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return -1

    covered_edges = (
        distance(first, second)
        + distance(second, third)
        + distance(first, third)
    ) // 2
    return covered_edges == score


def _stdout_equal_with_rule(
    actual,
    expected,
    rule: str,
    raw_input,
) -> bool:
    if _stdout_equal(actual, expected):
        return True
    if rule != "codeforces_1294_f_unordered_vertices":
        return False
    return _codeforces_1294_f_equal(actual, expected, raw_input)


def _stdin_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value) + "\n"
    return str(value) + "\n"


def run_apps(
    code: str,
    *,
    input_output: dict,
    mode: str,
    fn_name: str,
    starter_code: str,
    comparison_rule: str = "",
    timeout_seconds: float,
    total_timeout_seconds: float,
    memory_bytes: int,
    file_bytes: int,
) -> ExecutionOutcome:
    inputs = input_output["inputs"]
    outputs = input_output["outputs"]
    total = len(inputs)
    executed = 0
    controller = None
    timed_out = False
    problem_budget_exhausted = False
    diagnostic = ""
    reason = ""
    returncode = None
    deadline = time.monotonic() + total_timeout_seconds

    def request_with_deadline(value: dict, case_timeout: float):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _ProblemBudgetExhausted(
                "APPS per-problem total wall budget exhausted after "
                f"{total_timeout_seconds:.2f} seconds"
            )
        request_timeout = min(case_timeout, remaining)
        budget_limited = request_timeout < case_timeout
        try:
            return controller.request(
                value,
                timeout_seconds=request_timeout,
            )
        except TimeoutError:
            if budget_limited or time.monotonic() >= deadline:
                raise _ProblemBudgetExhausted(
                    "APPS per-problem total wall budget exhausted after "
                    f"{total_timeout_seconds:.2f} seconds"
                ) from None
            raise

    try:
        controller = _CandidateController(
            code,
            mode="call" if mode == "call_based" else "stdin",
            fn_name=fn_name,
            timeout_seconds=timeout_seconds,
            memory_bytes=memory_bytes,
            file_bytes=file_bytes,
            open_files=64,
            processes=64,
        )
        # Startup/entry-point errors are bounded by the same per-case timeout.
        request_with_deadline(
            {"op": "check"},
            max(10.0, timeout_seconds),
        )
        for index, (raw_input, raw_output) in enumerate(zip(inputs, outputs)):
            executed += 1
            if mode == "call_based":
                parsed = _parse_jsonish(raw_input)
                parsed = _restore_json_integer_keys(parsed)
                arguments, used_listnode = _adapt_call_arguments(
                    parsed,
                    fn_name=fn_name,
                    starter_code=starter_code,
                )
                actual = request_with_deadline(
                    {
                        "op": "call",
                        "args": [_wire_encode(item) for item in arguments],
                        "kwargs": {},
                    },
                    timeout_seconds,
                )
                expected = _parse_jsonish(raw_output)
                expected = _restore_json_integer_keys(expected)
                if (
                    used_listnode
                    and isinstance(expected, list)
                    and len(expected) == 1
                    and not isinstance(
                        actual,
                        (list, tuple, dict, set),
                    )
                ):
                    actual = [actual]
                matches = _official_call_equal(actual, expected)
            else:
                actual = request_with_deadline(
                    {"op": "stdin", "input": _stdin_text(raw_input)},
                    timeout_seconds,
                )
                expected = raw_output
                matches = _stdout_equal_with_rule(
                    actual,
                    expected,
                    comparison_rule,
                    raw_input,
                )
            if not matches:
                reason = "wrong_output"
                diagnostic = (
                    f"APPS case {index} failed: expected {expected!r}, got {actual!r}"
                )
                break
    except _ProblemBudgetExhausted as error:
        problem_budget_exhausted = True
        reason = "problem_budget_exhausted"
        diagnostic = str(error)
    except TimeoutError as error:
        timed_out = True
        reason = "timeout"
        diagnostic = str(error)
    except _RemoteCandidateError as error:
        message = str(error)
        reason = (
            "compile_or_entrypoint"
            if "SyntaxError" in message
            or "required entry point" in message
            or "NameError" in message and executed == 0
            else "candidate_error"
        )
        diagnostic = f"{type(error).__name__}: {error}"
    except (OSError, TypeError, ValueError) as error:
        reason = "infrastructure_error"
        diagnostic = f"{type(error).__name__}: {error}"
    finally:
        if controller is not None:
            candidate_diagnostic = controller.diagnostic()
            if problem_budget_exhausted:
                _kill_worker(controller.process)
            returncode = controller.close()
            if candidate_diagnostic and diagnostic:
                diagnostic = (
                    f"{diagnostic}\n{candidate_diagnostic}"
                    if diagnostic
                    else candidate_diagnostic
                )
    return ExecutionOutcome(
        score=1.0 if not diagnostic and executed == total else 0.0,
        diagnostic=diagnostic,
        reason=reason or ("passed" if executed == total else "unknown"),
        timed_out=timed_out,
        problem_budget_exhausted=problem_budget_exhausted,
        returncode=returncode,
        tests_executed=executed,
        tests_total=total,
    )


def run_mbpp(
    code: str,
    *,
    tests: list[str],
    imports: list[str],
    entry_point: str,
    timeout_seconds: float,
    memory_bytes: int,
    file_bytes: int,
) -> ExecutionOutcome:
    executed = 0
    controller = None
    timed_out = False
    diagnostic = ""
    returncode = None
    try:
        controller = _CandidateController(
            code,
            mode="call",
            fn_name=entry_point,
            imports=imports,
            timeout_seconds=timeout_seconds,
            memory_bytes=memory_bytes,
            file_bytes=file_bytes,
            open_files=64,
            processes=64,
        )
        controller.request(
            {"op": "check"},
            timeout_seconds=max(10.0, timeout_seconds),
        )

        class RemoteCallable:
            def __call__(self, *args, **kwargs):
                return controller.request(
                    {
                        "op": "call",
                        "args": [_wire_encode(item) for item in args],
                        "kwargs": {
                            key: _wire_encode(item)
                            for key, item in kwargs.items()
                        },
                    }
                )

        namespace = {
            "__builtins__": dict(__builtins__)
            if isinstance(__builtins__, dict)
            else dict(__builtins__.__dict__),
            "__name__": "__mbpp_tests__",
        }
        for index, statement in enumerate(imports, start=1):
            exec(
                compile(statement, f"<mbpp_import_{index}>", "exec"),
                namespace,
                namespace,
            )
        namespace[entry_point] = RemoteCallable()
        for index, test in enumerate(tests, start=1):
            executed += 1
            try:
                exec(
                    compile(test, f"<mbpp_test_{index}>", "exec"),
                    namespace,
                    namespace,
                )
            except AssertionError as error:
                diagnostic = f"test {index} failed: AssertionError: {error}"
                break
    except TimeoutError as error:
        timed_out = True
        diagnostic = str(error)
    except BaseException as error:
        diagnostic = f"{type(error).__name__}: {error}"
    finally:
        if controller is not None:
            candidate_diagnostic = controller.diagnostic()
            returncode = controller.close()
            if candidate_diagnostic:
                diagnostic = (
                    f"{diagnostic}\n{candidate_diagnostic}"
                    if diagnostic
                    else candidate_diagnostic
                )
    return ExecutionOutcome(
        score=1.0 if not diagnostic and executed == len(tests) else 0.0,
        diagnostic=diagnostic,
        timed_out=timed_out,
        returncode=returncode,
        tests_executed=executed,
        tests_total=len(tests),
    )
