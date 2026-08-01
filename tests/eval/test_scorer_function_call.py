"""function_call verifier (2026-08-01): BFCL-style AST argument match.

Pins the two-column format-vs-content contract: content_correct is the f1 scalar, and
format_valid rides in per_class. A hallucinated function name, a missing/extra arg, or a
wrong value is content-wrong even when the JSON is well-formed; unparseable JSON is
format-invalid.
"""
import json

from data.eval_set import EvalSet
from eval.scorers.function_call import extract_predictions, score


def _row(text, gold_calls, tools=None):
    return {"text": text, "answer": json.dumps(gold_calls), "tools": tools}


def _eval_set(rows):
    return EvalSet(pos=list(rows), neg=[], boundary=[], task_type="function_call")


TOOLS = [
    {"name": "get_weather", "arguments": {"city": "str", "unit": "str"}},
    {"name": "set_timer", "arguments": {"minutes": "int"}},
]


def _score_raw(rows, raw_outputs):
    es = _eval_set(rows)
    preds = extract_predictions(raw_outputs, es)
    return score(es, preds)


def test_exact_call_scores_one():
    rows = [_row("weather in Paris in celsius",
                 [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}],
                 tools=TOOLS)]
    raw = [json.dumps([{"name": "get_weather",
                        "arguments": {"city": "Paris", "unit": "celsius"}}])]
    result = _score_raw(rows, raw)
    assert result["f1"] == 1.0
    assert result["metric"] == "ast_arg_match"
    assert result["per_class"]["format_valid"] == 1.0


def test_wrong_name_is_content_wrong_but_format_valid():
    rows = [_row("timer for 5",
                 [{"name": "set_timer", "arguments": {"minutes": 5}}], tools=TOOLS)]
    raw = [json.dumps([{"name": "get_weather", "arguments": {"minutes": 5}}])]
    result = _score_raw(rows, raw)
    assert result["f1"] == 0.0
    assert result["per_class"]["format_valid"] == 1.0


def test_hallucinated_function_not_in_allowed_set_fails():
    rows = [_row("do a thing",
                 [{"name": "set_timer", "arguments": {"minutes": 5}}], tools=TOOLS)]
    raw = [json.dumps([{"name": "launch_missiles", "arguments": {"minutes": 5}}])]
    result = _score_raw(rows, raw)
    assert result["f1"] == 0.0


def test_missing_required_arg_fails():
    rows = [_row("weather in Paris in celsius",
                 [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}],
                 tools=TOOLS)]
    raw = [json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}])]
    result = _score_raw(rows, raw)
    assert result["f1"] == 0.0


def test_malformed_json_is_format_invalid():
    rows = [_row("timer for 5",
                 [{"name": "set_timer", "arguments": {"minutes": 5}}], tools=TOOLS)]
    raw = ["set_timer(minutes=5)  # not JSON"]
    result = _score_raw(rows, raw)
    assert result["f1"] == 0.0
    assert result["per_class"]["format_valid"] == 0.0


def test_value_type_coercion_matches_string_and_number():
    rows = [_row("timer for 5",
                 [{"name": "set_timer", "arguments": {"minutes": 5}}], tools=TOOLS)]
    # Model emits "5" as a string; coercion should still match gold int 5.
    raw = [json.dumps([{"name": "set_timer", "arguments": {"minutes": "5"}}])]
    result = _score_raw(rows, raw)
    assert result["f1"] == 1.0


def test_tolerates_prose_wrapping_and_fences():
    rows = [_row("timer for 5",
                 [{"name": "set_timer", "arguments": {"minutes": 5}}], tools=TOOLS)]
    raw = ["Sure! ```json\n[{\"name\": \"set_timer\", \"arguments\": {\"minutes\": 5}}]\n```"]
    result = _score_raw(rows, raw)
    assert result["f1"] == 1.0
    assert result["per_class"]["format_valid"] == 1.0


def test_empty_gold_and_empty_pred_match():
    rows = [_row("hello there", [], tools=TOOLS)]
    raw = ["[]"]
    result = _score_raw(rows, raw)
    assert result["f1"] == 1.0
