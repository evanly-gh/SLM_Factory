# tests/test_cot_answer_split.py
"""
CoT-trained generation models emit `<reasoning>...</reasoning>` before the answer, because that
is the shape of the training target built by `training/lora_trainer.py::_training_turn`.

Math and code were unaffected — their extractors already pull one specific thing (the final
number, the fenced block). Judge-scored generation extracted NOTHING (`raw.strip()`), so the
reasoning block was handed to a judge asked "how good is this summary?", which guarantees a poor
score for output that may contain a perfectly good summary.

The reasoning is separated, not discarded: it is the part that explains WHY an answer is wrong.
See B251.
"""
from eval.harness import _attach_reasoning_to_failures
from eval.scorers.generation import extract_predictions, split_reasoning
from data.eval_set import EvalSet

COT = "<reasoning>\nThe two speakers agree on a time.\n</reasoning>\n\nThey agree to meet at 4pm."


class TestSplit:
    def test_answer_excludes_the_reasoning(self):
        reasoning, answer = split_reasoning(COT)
        assert answer == "They agree to meet at 4pm."
        assert "reasoning" not in answer.lower()

    def test_reasoning_is_retained_not_discarded(self):
        reasoning, _answer = split_reasoning(COT)
        assert "The two speakers agree on a time." in reasoning

    def test_plain_output_without_cot_is_untouched(self):
        reasoning, answer = split_reasoning("They agree to meet at 4pm.")
        assert reasoning == ""
        assert answer == "They agree to meet at 4pm."

    def test_tolerates_whitespace_and_case_variants(self):
        for raw in (
            "< Reasoning >x</ Reasoning >\n\nThe answer.",
            "<REASONING>\nx\n</REASONING>\n\nThe answer.",
            "<reasoning>x</reasoning>The answer.",
        ):
            assert split_reasoning(raw)[1] == "The answer."

    def test_handles_a_dropped_opening_tag(self):
        """Small models frequently emit the closing tag only."""
        assert split_reasoning("thinking out loud</reasoning>\n\nThe answer.")[1] == "The answer."

    def test_reasoning_only_output_keeps_text_rather_than_scoring_an_empty_string(self):
        """An empty prediction scores 0 and hides the real failure mode."""
        _reasoning, answer = split_reasoning("<reasoning>\nI am unsure.\n</reasoning>")
        assert answer != ""

    def test_multiline_reasoning_is_removed_entirely(self):
        raw = "<reasoning>\nline one\nline two\nline three\n</reasoning>\n\nFinal summary."
        assert split_reasoning(raw)[1] == "Final summary."

    def test_empty_and_none_are_safe(self):
        assert split_reasoning("") == ("", "")
        assert split_reasoning(None) == ("", "")


class TestOnlyTheAnswerIsJudged:
    def test_extract_predictions_strips_reasoning_for_generation(self):
        eval_set = EvalSet(all=[{"text": "t", "answer": "a"}], task_type="generation")
        assert extract_predictions([COT], eval_set) == ["They agree to meet at 4pm."]

    def test_code_generation_path_is_unchanged(self):
        eval_set = EvalSet(all=[{"text": "t", "answer": "a"}], task_type="code_generation")
        out = extract_predictions(["```python\nprint(1)\n```"], eval_set)
        assert out == ["print(1)"]


class TestReasoningIsStoredOnFailures:
    def test_failure_records_carry_the_reasoning(self):
        eval_set = EvalSet(all=[{"text": "t", "answer": "a"}], task_type="generation")
        predictions = ["They agree to meet at 4pm."]
        result = {"failures": [{"text": "t", "predicted": predictions[0], "judge_score": 0.1}]}
        _attach_reasoning_to_failures(result, eval_set, [COT], predictions)
        assert "The two speakers agree on a time." in result["failures"][0]["reasoning"]

    def test_no_reasoning_key_when_the_model_emitted_none(self):
        eval_set = EvalSet(all=[{"text": "t", "answer": "a"}], task_type="generation")
        result = {"failures": [{"text": "t", "predicted": "plain", "judge_score": 0.1}]}
        _attach_reasoning_to_failures(result, eval_set, ["plain"], ["plain"])
        assert "reasoning" not in result["failures"][0]

    def test_is_a_noop_for_code_generation(self):
        eval_set = EvalSet(all=[{"text": "t"}], task_type="code_generation")
        result = {"failures": [{"predicted": "print(1)"}]}
        _attach_reasoning_to_failures(result, eval_set, [COT], ["print(1)"])
        assert "reasoning" not in result["failures"][0]

    def test_empty_failures_do_not_crash(self):
        eval_set = EvalSet(all=[{"text": "t"}], task_type="generation")
        _attach_reasoning_to_failures({"failures": []}, eval_set, [COT], ["x"])
        _attach_reasoning_to_failures({}, eval_set, [COT], ["x"])
