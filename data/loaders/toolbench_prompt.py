"""ToolBench's own system-prompt construction, reproduced exactly (2026-08-24).

WHY THIS IS A SEPARATE MODULE
    The two halves of this task come from two different places and MUST agree character for
    character:

      * the TRAIN rows arrive with ToolBench's system prompt already baked into
        ``conversations[0]``, written by ToolBench's own inference code in 2023;
      * the EVAL rows are the ToolEval test queries, which ship as a raw ``api_list`` and no prompt
        at all — so the prompt has to be rebuilt here.

    If the rebuild drifts from the original by even a stray space, the model is fine-tuned on one
    input shape and evaluated on another. That is B250/B290 exactly, and on this task it would be
    invisible: both prompts are ~5,000 characters of API schema, so a diff is not something a human
    notices in a log line. Keeping the reconstruction in one small module with the upstream source
    quoted next to it is what makes the agreement checkable —
    ``tests/test_toolbench_prompt.py`` pins it against a verbatim capture of a real training row.

UPSTREAM SOURCES (github.com/OpenBMB/ToolBench @ master, read 2026-08-24)
    ``toolbench/utils.py``                              — ``standardize``, ``change_name``,
                                                          ``process_system_message``
    ``toolbench/inference/Prompts/ReAct_prompts.py``    — ``FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION``
    ``toolbench/inference/Downstream_tasks/rapidapi.py``— ``api_json_to_openai_json``, the
                                                          ``Finish`` function dict, and the
                                                          ``task_description`` tool listing

THE ONE THING THAT LOOKS LIKE A BUG AND IS NOT
    ``process_system_message`` appends ``str(functions)`` — a Python **repr** of a list of dicts,
    with single quotes — not JSON. So the API block in a ToolBench prompt is
    ``[{'name': 'x', ...}]`` and not ``[{"name": "x", ...}]``. Reproducing that is why
    ``data/loaders/toolbench.py`` can read the schemas back out of a training row with
    ``ast.literal_eval`` rather than ``json.loads``. Emitting JSON here would be tidier and would
    silently put every eval row off-distribution.
"""
from __future__ import annotations

import re

# `toolbench/inference/Prompts/ReAct_prompts.py::FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION`, verbatim
# including the trailing space on the "Remember: " line, which is present upstream.
FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = """You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: {task_description}"""

# `toolbench/inference/Prompts/ReAct_prompts.py::FORMAT_INSTRUCTIONS_USER_FUNCTION`.
FORMAT_INSTRUCTIONS_USER_FUNCTION = """
{input_description}
Begin!
"""

# The sentence `process_system_message` rewrites, and what it rewrites it to.
_STEP_SENTENCE = "with a function call to actually excute your step."
_STEP_REPLACEMENT = (
    "with a function call to actually excute your step. Your output should follow this format:\n"
    "Thought:\nAction\nAction Input:\n"
)
_API_PREAMBLE = "\nSpecifically, you have access to the following APIs: "

# `rapidapi_wrapper.__init__`'s task_description header, verbatim.
_TASK_DESCRIPTION_HEADER = (
    "You should use functions to help handle the real time user querys. Remember:\n"
    '1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain '
    "enough information to show to the user,If you can't handle the task, or you find that "
    "function calls always fail(the function is not valid now), use function "
    "Finish->give_up_and_restart.\n"
    "2.Do not use origin tool names, use only subfunctions' names.\n"
    "You have access of the following tools:\n"
)

# The `Finish` pseudo-function every ToolBench prompt declares last. Copied from
# `rapidapi_wrapper.__init__`; it is what makes `Finish` a legal Action and what defines
# `return_type`'s two-value enum, which is the part this task's scorer grades exactly.
FINISH_FUNCTION = {
    "name": "Finish",
    "description": (
        "If you believe that you have obtained a result that can answer the task, please call "
        "this function to provide the final answer. Alternatively, if you recognize that you are "
        "unable to proceed with the task in the current state, call this function to restart. "
        "Remember: you must ALWAYS call this function at the end of your attempt, and the only "
        "part that will be shown to the user is the final answer, so it should contain "
        "sufficient information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "return_type": {
                "type": "string",
                "enum": ["give_answer", "give_up_and_restart"],
            },
            "final_answer": {
                "type": "string",
                "description": (
                    'The final answer you want to give the user. You should have this field if '
                    '"return_type"=="give_answer"'
                ),
            },
        },
        "required": ["return_type"],
    },
}

# `api_json_to_openai_json`'s two constants.
_DESCRIPTION_MAX_LENGTH = 256
_NAME_MAX_LENGTH = 64
_TYPE_MAP = {"NUMBER": "integer", "STRING": "string", "BOOLEAN": "boolean"}

# `standardize`: everything outside CJK / ASCII alphanumerics / underscore becomes an underscore.
_NON_STANDARD = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]")
_RUNS_OF_UNDERSCORE = re.compile(r"(_)\1+")

# `change_name`: parameter names that would shadow a Python/JSON keyword. This is where xLAM's
# much-queried `is_id` argument name comes from, and why a verifier must not "correct" it.
_RESERVED_NAMES = ("from", "class", "return", "false", "true", "id", "and")


def standardize(string: str) -> str:
    """`toolbench/utils.py::standardize`, reproduced including its quirks.

    Lowercases, replaces non-alphanumerics with underscores, collapses runs of underscores, strips
    leading and trailing underscores, and prefixes ``get_`` when the result would start with a
    digit (a leading digit is not a legal Python identifier, which is what the upstream guard is
    for). Returns ``""`` for input that is entirely punctuation, matching upstream's early return.
    """
    result = _NON_STANDARD.sub("_", str(string))
    result = _RUNS_OF_UNDERSCORE.sub("_", result).lower()
    result = result.strip("_")
    if not result:
        return result
    if result[0].isdigit():
        result = "get_" + result
    return result


def change_name(name: str) -> str:
    """`toolbench/utils.py::change_name` — rename a parameter that shadows a keyword."""
    return f"is_{name}" if name in _RESERVED_NAMES else name


def api_function_name(api_name: str, standard_tool_name: str) -> str:
    """The callable name a ToolBench prompt exposes for one API: `<api>_for_<tool>`.

    Truncated to the LAST 64 characters, not the first — that is upstream's ``[-64:]``, and it
    matters because the tool suffix is what disambiguates two same-named APIs. Truncating from the
    front instead would produce collisions the scorer would then grade as tool-selection errors.
    """
    return f"{change_name(standardize(api_name))}_for_{standard_tool_name}"[-_NAME_MAX_LENGTH:]


def api_json_to_openai_json(api_json: dict, standard_tool_name: str) -> dict:
    """`rapidapi_wrapper.api_json_to_openai_json`: one `api_list` entry as a function dict.

    Note that both required and optional parameters land in ``properties``; the two lists
    ``required`` and ``optional`` record which is which. ``optional`` is not an OpenAI schema key —
    it is ToolBench's own addition, and reproducing it is part of matching the prompt.
    """
    name = api_function_name(api_json.get("api_name", ""), standard_tool_name)
    description = f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'
    api_description = str(api_json.get("api_description") or "").strip()
    if api_description:
        truncated = api_description.replace(
            str(api_json.get("api_name") or ""), name
        )[:_DESCRIPTION_MAX_LENGTH]
        description += f'The description of this function is: "{truncated}"'

    properties: dict = {}
    required: list[str] = []
    optional: list[str] = []
    for group, sink in (("required_parameters", required), ("optional_parameters", optional)):
        for parameter in api_json.get(group) or []:
            if not isinstance(parameter, dict):
                continue
            parameter_name = change_name(standardize(parameter.get("name", "")))
            entry = {
                "type": _TYPE_MAP.get(parameter.get("type"), "string"),
                "description": str(parameter.get("description") or "")[:_DESCRIPTION_MAX_LENGTH],
            }
            # Upstream includes `example_value` only when the default is a non-empty string. The
            # length test is on `str(default)`, so a default of `0` DOES produce an example_value
            # while `""` and `None` do not — reproduced rather than simplified.
            default_value = parameter.get("default")
            if len(str(default_value if default_value is not None else "")) != 0:
                entry["example_value"] = default_value
            properties[parameter_name] = entry
            sink.append(parameter_name)

    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "optional": optional,
        },
    }


def build_functions(api_list: list[dict]) -> list[dict]:
    """Every callable a query declares, in ToolBench's order, with `Finish` appended last."""
    functions = [
        api_json_to_openai_json(api, standardize(api.get("tool_name", "")))
        for api in api_list
        if isinstance(api, dict)
    ]
    functions.append(dict(FINISH_FUNCTION))
    return functions


def build_task_description(api_list: list[dict], tool_descriptions: dict[str, str]) -> str:
    """The numbered tool listing, deduplicated by standardized tool name and in first-seen order.

    `tool_descriptions` maps a standardized tool name to that tool's prose description from the
    RapidAPI environment. Upstream truncates to 512 characters, strips newlines, and substitutes
    the literal string ``None`` for an empty description — so a tool we have no description for
    produces exactly what upstream produces for a tool whose description was blank, rather than a
    differently-shaped line.
    """
    ordered: dict[str, str] = {}
    for api in api_list:
        if not isinstance(api, dict):
            continue
        tool = standardize(api.get("tool_name", ""))
        if tool and tool not in ordered:
            ordered[tool] = str(tool_descriptions.get(tool) or "")
    lines = []
    for index, (tool, description) in enumerate(ordered.items(), start=1):
        stripped = description[:512].replace("\n", "").strip() or "None"
        lines.append(f"{index}.{tool}: {stripped}\n")
    return _TASK_DESCRIPTION_HEADER + "".join(lines)


def process_system_message(system_message: str, functions: list[dict]) -> str:
    """`toolbench/utils.py::process_system_message`.

    Upstream asserts the step sentence is present, because the whole function is a targeted
    rewrite of it; the assert is kept as an explicit raise so a future edit to
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION cannot silently produce a prompt with no output-format
    instruction in it.
    """
    if _STEP_SENTENCE not in system_message:
        raise ValueError(
            "the ToolBench system template no longer contains the sentence "
            f"{_STEP_SENTENCE!r} that process_system_message rewrites"
        )
    rewritten = system_message.replace(_STEP_SENTENCE, _STEP_REPLACEMENT)
    # `str(functions)` — a Python repr, not JSON. See the module docstring.
    return rewritten + _API_PREAMBLE + str(functions)


def build_system_prompt(api_list: list[dict], tool_descriptions: dict[str, str]) -> str:
    """The complete ToolBench system message for one query's declared APIs."""
    task_description = build_task_description(api_list, tool_descriptions)
    template = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION.format(task_description=task_description)
    return process_system_message(template, build_functions(api_list))


def build_user_prompt(query: str) -> str:
    """The user turn: the query wrapped in ToolBench's `\\n{query}\\nBegin!\\n`."""
    return FORMAT_INSTRUCTIONS_USER_FUNCTION.format(input_description=str(query or "").strip())
