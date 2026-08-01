"""Function-calling verifier (2026-08-01): BFCL-style AST argument match.

A judge-free, executable-in-spirit scorer for the ``function_call`` task type. Each eval row
carries ``{text, answer, tools}`` where:

- ``text``   — the user request.
- ``answer`` — the GOLD call(s), a JSON string encoding a list of
               ``{"name": str, "arguments": {..}}`` objects.
- ``tools``  — (optional) the available tool signatures; the set of allowed function names is
               derived from it so a hallucinated function name is caught.

Two numbers are produced per row (the two-column format-vs-content split):

- ``format_valid``   — the model emitted parseable JSON of the expected call shape.
- ``content_correct``— every predicted call matches gold: name ∈ allowed set, name == gold,
                       all gold-required args present, values equal gold with light type
                       coercion. This is the comparison scalar (``f1``).

``format_valid`` is reported inside ``per_class`` so the shared ``EvalResult`` is unchanged.
"""
import json
import re

from data.eval_set import EvalSet

FUNCTION_CALL_PROMPT = (
    "You are a function-calling assistant. Given the user request and the available "
    "functions, reply with ONLY a JSON array of the calls to make, where each element is "
    '{{"name": <function name>, "arguments": {{<arg>: <value>, ...}}}}. '
    "Use only the functions listed. Reply with [] if no function applies. "
    "Do not add prose or Markdown.\n\n"
    "Available functions:\n{tools}\n\nUser request: {text}"
)


def _tools_str(example: dict) -> str:
    """Render the row's tool signatures for the prompt (best-effort, JSON if structured)."""
    tools = example.get("tools")
    if tools is None:
        return "(none provided)"
    if isinstance(tools, str):
        return tools
    try:
        return json.dumps(tools, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(tools)


def build_prompts(eval_set: EvalSet) -> list[str]:
    return [
        FUNCTION_CALL_PROMPT.format(tools=_tools_str(ex), text=ex.get("text", ""))
        for ex in eval_set.all
    ]


def _parse_calls(raw: object) -> list[dict] | None:
    """Parse a model/gold string into a list of {name, arguments} calls, or None if it does
    not parse as the expected shape. A lone object is accepted and wrapped in a list."""
    if isinstance(raw, (list, dict)):
        parsed = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            # Tolerate fences / prose: grab the first JSON array or object.
            match = re.search(r"\[.*\]", text, re.DOTALL) or re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group())
            except (ValueError, TypeError):
                return None
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return None
    calls = []
    for item in parsed:
        if not isinstance(item, dict) or "name" not in item:
            return None
        args = item.get("arguments", item.get("args", {}))
        if not isinstance(args, dict):
            return None
        calls.append({"name": str(item["name"]), "arguments": args})
    return calls


def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[list[dict] | None]:
    """None means the output did not parse as calls (format-invalid)."""
    return [_parse_calls(raw) for raw in raw_outputs]


def _allowed_names(example: dict) -> set[str] | None:
    """The set of function names the row permits, or None when unconstrained (no tools)."""
    tools = example.get("tools")
    names: set[str] = set()
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                # OpenAI-style {"function": {"name": ...}} or flat {"name": ...}.
                fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
                name = fn.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
    elif isinstance(tools, dict):
        name = tools.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names or None


def _coerce(value: object) -> object:
    """Light type coercion so 3 == "3" and 1.0 == 1 for scalar argument comparison."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except (ValueError, TypeError):
            return stripped.lower()
    return value


def _args_match(gold_args: dict, pred_args: dict) -> bool:
    """All gold-required args present in the prediction with equal (coerced) values, and no
    extra unexpected args."""
    if set(gold_args) != set(pred_args):
        return False
    for key, gold_value in gold_args.items():
        if isinstance(gold_value, (dict, list)):
            if pred_args.get(key) != gold_value:
                return False
        elif _coerce(pred_args.get(key)) != _coerce(gold_value):
            return False
    return True


def _call_correct(gold_calls: list[dict], pred_calls: list[dict], allowed: set[str] | None) -> bool:
    """Order-sensitive full match of the call list."""
    if len(gold_calls) != len(pred_calls):
        return False
    for gold, pred in zip(gold_calls, pred_calls):
        if allowed is not None and pred["name"] not in allowed:
            return False
        if pred["name"] != gold["name"]:
            return False
        if not _args_match(gold.get("arguments", {}), pred.get("arguments", {})):
            return False
    return True


def score(eval_set: EvalSet, predictions: list[list[dict] | None]) -> dict:
    from eval.metrics import per_slice_scores

    content_scores: list[float] = []
    format_scores: list[float] = []
    failures: list[dict] = []
    for ex, pred in zip(eval_set.all, predictions):
        gold_calls = _parse_calls(ex.get("answer", "")) or []
        format_valid = 1.0 if pred is not None else 0.0
        allowed = _allowed_names(ex)
        content = 1.0 if (pred is not None and _call_correct(gold_calls, pred, allowed)) else 0.0
        format_scores.append(format_valid)
        content_scores.append(content)
        if content < 1.0:
            failures.append({**ex, "predicted": pred, "format_valid": format_valid})

    n = len(content_scores)
    f1 = sum(content_scores) / n if n else 0.0
    format_valid_mean = sum(format_scores) / n if n else 0.0

    predicted_labels = ["correct" if v >= 0.5 else "wrong" for v in content_scores]
    gold_labels = ["correct"] * n
    slices = per_slice_scores(eval_set, predicted_labels, gold_labels=gold_labels)

    return {
        "f1": f1,
        "metric": "ast_arg_match",
        "per_class": {"ast_arg_match": f1, "format_valid": format_valid_mean},
        "slices": slices,
        "failures": failures,
    }
