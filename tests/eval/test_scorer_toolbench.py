"""ToolEval pass rate: the exact rules, the majority vote, and the weighted aggregation.

WHY THIS FILE EXISTS
    This is the suite's only metric that is not decidable by computation, so the parts that ARE
    decidable have to be pinned hard — otherwise a judging change silently moves the score. Three
    properties matter most:

      * every rule that runs BEFORE the judge (no `Finish`, gave up, over budget) must decide the
        row without a judge call, so judge spend goes only to the question needing judgement;
      * the majority vote must require a strict majority for a pass, because pass rate is a positive
        claim about task completion;
      * the overall number must be the query-count WEIGHTED mean, reproduced against the paper's own
        published table. The unweighted mean of its six subset rates is 77.75% and the number it
        reports is 77.55%, so getting this wrong is a plausible-looking 0.2-point error.
"""
from __future__ import annotations

import pytest

from data.eval_set import EvalSet
from eval.scorers.toolbench import (
    JUDGE_ROUNDS,
    MAX_ACTIONS,
    exact_verdict,
    extract_predictions,
    failure_category,
    majority_verdict,
    parse_solution_path,
    parse_tooleval_verdict,
    score,
)

TOOLS = [{
    "name": "get_weather_for_weather_api",
    "parameters": {"properties": {"city": "string"}, "required": ["city"], "optional": []},
}]


def path(city="Paris", final="It is 18 degrees in Paris.", return_type="give_answer", calls=1):
    steps = "".join(
        f"Thought: step {i} — I will look up the weather.\n"
        f"Action: get_weather_for_weather_api\n"
        f'Action Input: {{"city": "{city}"}}\n'
        for i in range(calls)
    )
    return (
        steps
        + "Thought: I have what I need.\nAction: Finish\n"
        + f'Action Input: {{"return_type": "{return_type}", "final_answer": "{final}"}}'
    )


def row(query="what is the weather in Paris?", subset="G1_instruction"):
    return {"text": f"You are AutoGPT...\n{query}\nBegin!\n", "query": query,
            "answer": "", "tools": TOOLS, "_subset": subset}


class StubJudge:
    """A judge that returns a scripted verdict per round, and counts its calls."""

    def __init__(self, verdicts):
        # `verdicts` is a list per round, or a single value used for every round.
        self.verdicts = verdicts
        self.calls = 0
        self.seen: list[dict] = []

    @classmethod
    def from_config(cls, rubric=None):
        return cls(1.0)

    def score_payloads(self, payloads):
        payloads = list(payloads)
        self.calls += len(payloads)
        self.seen.extend(payloads)
        if isinstance(self.verdicts, (int, float)):
            return [float(self.verdicts)] * len(payloads)
        round_index = payloads[0]["round"] if payloads else 0
        return [float(self.verdicts[round_index])] * len(payloads)


def _score(rows, raw_outputs, judge):
    eval_set = EvalSet(all=[dict(r) for r in rows], task="toolbench")
    predictions = extract_predictions(raw_outputs, eval_set)
    return score(eval_set, predictions, judge=judge)


# --------------------------------------------------------------------------
# Parsing a path
# --------------------------------------------------------------------------


def test_a_multi_step_path_splits_into_its_actions():
    parsed = parse_solution_path(path(calls=2))
    assert parsed["actions"] == [
        "get_weather_for_weather_api", "get_weather_for_weather_api", "Finish",
    ]
    assert parsed["return_type"] == "give_answer"
    assert parsed["final_answer"] == "It is 18 degrees in Paris."
    assert len(parsed["thoughts"]) == 3


def test_prose_with_no_action_is_unparseable_rather_than_an_empty_path():
    """The format/content split (B290). Output the extractor cannot read is not a prediction, and a
    model that emits prose has a prompt problem, not a data problem."""
    assert parse_solution_path("I'm sorry, I can't help with that.") is None
    assert parse_solution_path("") is None
    assert parse_solution_path(None) is None


def test_an_action_input_mentioning_the_word_Action_does_not_end_the_block_early():
    parsed = parse_solution_path(
        "Thought: searching.\nAction: get_weather_for_weather_api\n"
        'Action Input: {"city": "Action City"}\n'
        "Thought: done.\nAction: Finish\n"
        'Action Input: {"return_type": "give_answer", "final_answer": "ok"}'
    )
    assert parsed["actions"] == ["get_weather_for_weather_api", "Finish"]
    assert parsed["steps"][0]["arguments"] == '{"city": "Action City"}'


def test_a_truncated_finish_still_yields_its_return_type():
    """`max_new_tokens` can cut a discursive `final_answer` mid-string. `return_type` is emitted
    first, so it survives — and recovering it is the difference between scoring the row on what it
    decided and throwing the whole path away as unparseable."""
    parsed = parse_solution_path(
        "Thought: done.\nAction: Finish\n"
        'Action Input: {"return_type": "give_answer", "final_answer": "The weather in Paris is'
    )
    assert parsed["return_type"] == "give_answer"
    assert parsed["final_answer"].startswith("The weather in Paris is")


@pytest.mark.parametrize("layout", [
    "Thought: t\nAction: get_x\nAction Input: {}",       # colon, name on the same line (the DATA)
    "Thought: t\nAction get_x\nAction Input: {}",        # no colon, same line
    "Thought: t\nAction:\nget_x\nAction Input: {}",      # colon, name on the NEXT line
    "Thought: t\nAction\nget_x\nAction Input: {}",       # no colon, name on the next line (the PROMPT)
])
def test_every_layout_the_prompt_or_the_data_can_produce_parses(layout):
    """ToolBench's instruction and its data disagree, so all four spellings must parse.

    The prompt shows `Action` alone on its own line with no colon; every gold turn writes
    `Action: <name>` on one line. A model may reasonably do either, or mix them.

    THE REGRESSION THIS PINS (2026-08-25). Making the colon optional as
    `Action\\s*:?[ \\t]*(?P<name>[^\\n]*)` restricted the post-colon whitespace to spaces and tabs,
    which silently dropped the name-on-the-next-line layout — and that is the layout the PROMPT
    shows, so it is the one the teacher uses. Reference-model `format_valid` fell from 1.0000 to
    0.4316 between runs 38820472 and 38832586 and the harness understated itself by 2.3x. Only the
    same-line case was tested, so nothing caught it.
    """
    parsed = parse_solution_path(layout)
    assert parsed is not None, "this layout did not parse at all"
    assert parsed["actions"] == ["get_x"]


def test_the_colon_after_Action_is_optional_because_the_prompt_omits_it():
    """ToolBench's instruction and its data disagree, so both spellings must parse.

    The system prompt asks for `Thought:` / `Action` / `Action Input:` — no colon after `Action` —
    while every gold turn in `toolllama_G123_dfs` reads `Action: <name>`. A model that obeys the
    INSTRUCTION instead of imitating the data was scored `unparseable_path`, which is a measurement
    error attributed to the model.
    """
    with_colon = parse_solution_path("Thought: ok\nAction: get_x_for_y\nAction Input: {}")
    without = parse_solution_path("Thought: ok\nAction get_x_for_y\nAction Input: {}")
    assert with_colon["actions"] == without["actions"] == ["get_x_for_y"]


@pytest.mark.parametrize("degenerate", [
    # All three verbatim from SmolLM2-360M on job 38820307: the format words with no function name.
    "Thought: Action Action Input: Finish: Action Action Input: Finish:",
    "Thought: Action Action Input: Thought: Action Action Input: Thought: Action Action Input:",
    "Thought: Action Action Input: I give up and restart. Finish->give_up_and_restart",
    # `Action` occurring as an ordinary word must not become a call with an empty name.
    "Thought: I will take Action\nAction Input: {}",
])
def test_the_optional_colon_does_not_make_degenerate_output_parseable(degenerate):
    """The other side of the relaxation. Making the colon optional must not turn a model that
    emits the format WORDS with no function name into a valid call — that would convert a real
    format failure into a wrong-content failure and hide it from `format_valid`."""
    assert parse_solution_path(degenerate) is None


def test_a_trailing_End_Action_terminator_is_stripped():
    """ToolBench's zero-shot template teaches an explicit `End Action`, so a model trained on it
    emits one; leaving it attached would make every Action Input invalid JSON."""
    parsed = parse_solution_path(
        "Thought: go.\nAction: get_weather_for_weather_api\n"
        'Action Input: {"city": "Paris"}\nEnd Action'
    )
    assert parsed["steps"][0]["arguments"] == '{"city": "Paris"}'


# --------------------------------------------------------------------------
# The exact rules, applied before any judge call
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "category"), [
    ("no tool applies here", "unparseable_path"),
    ("Thought: looking.\nAction: get_weather_for_weather_api\n"
     'Action Input: {"city": "Paris"}', "no_finish_call"),
    (path(return_type="give_up_and_restart"), "gave_up"),
    (path(final=""), "empty_final_answer"),
    (path(calls=MAX_ACTIONS + 1), "budget_exceeded"),
])
def test_a_rule_decided_row_never_reaches_the_judge(raw, category):
    judge = StubJudge(1.0)
    result = _score([row()], [raw], judge)
    assert judge.calls == 0, f"{category} was sent to the judge"
    assert result["f1"] == 0.0
    assert result["failures"][0]["error_type"] == category


def test_a_well_formed_in_budget_path_IS_sent_to_the_judge():
    """The complement of the rules above: the only rows worth paying for are the ones that produced
    a complete path claiming an answer."""
    judge = StubJudge(1.0)
    result = _score([row()], [path()], judge)
    assert judge.calls == JUDGE_ROUNDS
    assert result["f1"] == 1.0
    assert result["per_class"]["judged_rows"] == 1.0


def test_the_judge_is_asked_about_the_query_and_the_final_answer_only():
    """Not the whole path, and not the prompt. ToolEval's `check_answer_status` takes a query and an
    answer; handing it 5,000 characters of API schema would change what is being judged."""
    judge = StubJudge(1.0)
    _score([row()], [path()], judge)
    payload = judge.seen[0]
    assert payload["query"] == "what is the weather in Paris?"
    assert payload["answer"] == "It is 18 degrees in Paris."
    assert "You are AutoGPT" not in payload["answer"]
    assert set(payload) == {"query", "answer", "round"}


def test_each_round_is_a_distinct_payload_so_the_votes_can_differ():
    """At temperature 0 the rounds would be identical and the cache would collapse them into one
    request, making the majority vote a no-op. The round index is what keeps them separate."""
    judge = StubJudge([1.0, 0.0, 0.0])
    _score([row()], [path()], judge)
    assert sorted(p["round"] for p in judge.seen) == list(range(JUDGE_ROUNDS))


def test_exact_verdict_never_claims_a_row_was_solved():
    """No computable rule can establish that a query WAS solved without executing the APIs — which
    is the entire reason a judge is involved. `exact_verdict` may only reject or defer."""
    assert exact_verdict(row(), parse_solution_path(path())) is None
    assert exact_verdict(row(), None) == "unsolved"


def test_an_undeclared_api_is_reported_but_does_not_bypass_the_judge():
    """Upstream ToolEval judges the ANSWER and knows nothing about which APIs were declared, so
    making this an exact rejection would be this harness inventing a stricter metric.

    It is reported instead — over all rows, including passing ones — because without API execution a
    call to an endpoint that does not exist means the answer was invented, and that number bounds
    how much of the pass rate could be fluent fabrication.
    """
    fabricated = (
        "Thought: I will use a tool.\nAction: get_weather_for_totally_made_up_api\n"
        'Action Input: {"city": "Paris"}\n'
        "Thought: done.\nAction: Finish\n"
        'Action Input: {"return_type": "give_answer", "final_answer": "It is 18 degrees."}'
    )
    judge = StubJudge(1.0)
    result = _score([row()], [fabricated], judge)
    assert judge.calls == JUDGE_ROUNDS
    assert result["f1"] == 1.0, "the judge's verdict must still decide the row"
    assert result["per_class"]["undeclared_api_rate"] == 1.0


# --------------------------------------------------------------------------
# The majority vote
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("votes", "expected"), [
    ([1.0, 1.0, 1.0], "solved"),
    ([1.0, 1.0, 0.0], "solved"),
    # No strict majority for solved: a split panel is not evidence the task was completed.
    ([1.0, 0.0, 0.5], "unsolved"),
    ([1.0, 0.0], "unsolved"),
    ([0.0, 0.0, 1.0], "unsolved"),
    ([0.5, 0.5, 1.0], "unsure"),
    ([0.5, 0.0, 0.0], "unsolved"),
    ([], "unsolved"),
])
def test_the_panel_requires_a_strict_majority_to_pass(votes, expected):
    assert majority_verdict(votes) == expected


def test_an_unsure_verdict_does_not_pass_and_is_reported_separately():
    """ToolEval's own `eval_pass_rate` counts only `Solved` as a pass. Reporting `Unsure` apart from
    `Unsolved` is what makes it visible when the metric is measuring the judge's uncertainty rather
    than the model."""
    result = _score([row()], [path()], StubJudge(0.5))
    assert result["f1"] == 0.0
    assert result["per_class"]["judge_unsure_rate"] == 1.0
    assert result["failures"][0]["error_type"] == "judge_unsure"


@pytest.mark.parametrize(("reply", "expected"), [
    ('{"content": "it answers the query", "answer_status": "Solved"}', 1.0),
    ('{"content": "sorry message", "answer_status": "Unsolved"}', 0.0),
    ('{"answer_status": "Unsure"}', 0.5),
    ("answer_status: solved", 1.0),
    ("The answer is Unsolved.", 0.0),
    # A reply that arrives but names no verdict is Unsure, matching ToolEval's own handling of an
    # assessment it cannot read.
    ("I have no idea what you are asking.", 0.5),
])
def test_a_verdict_is_read_out_of_whatever_shape_the_judge_replied_in(reply, expected):
    assert parse_tooleval_verdict(reply) == expected


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_an_empty_judge_reply_is_infrastructure_failing_and_raises(empty):
    """Scoring it `Unsolved` would turn a dead judge into a plausible zero across the whole eval,
    which is exactly what `TaskSpec.needs_judge` exists to prevent."""
    from eval.judge_client import JudgeInfrastructureError

    with pytest.raises(JudgeInfrastructureError):
        parse_tooleval_verdict(empty)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_the_overall_rate_is_weighted_by_subset_size_not_a_mean_of_rates():
    """The paper's aggregation, on a case where the two differ.

    Three G1 queries of which two pass, one G3 query which fails: the weighted rate is 2/4 = 0.50,
    while the mean of the subset rates would be (0.667 + 0.0)/2 = 0.333. Reporting the latter under
    the former's name is a real risk — it is why the paper's own 77.55% is not the 77.75% its six
    subset numbers average to.
    """
    rows = [row(f"q{i}", "G1_instruction") for i in range(3)] + [row("q3", "G3_instruction")]
    outputs = [path(final="good"), path(final="good"), path(final="bad"), path(final="bad")]

    class Selective(StubJudge):
        def score_payloads(self, payloads):
            payloads = list(payloads)
            self.calls += len(payloads)
            return [1.0 if p["answer"] == "good" else 0.0 for p in payloads]

    result = _score(rows, outputs, Selective(1.0))
    assert result["f1"] == pytest.approx(0.5)
    assert result["per_class"]["G1_instruction"] == pytest.approx(2 / 3)
    assert result["per_class"]["G3_instruction"] == pytest.approx(0.0)
    assert result["per_class"]["tooleval_pass_rate"] == pytest.approx(0.5)


def test_the_paper_headline_number_is_reproduced_by_this_aggregation():
    """arXiv:2512.15943 Table 1/2, recomputed by the rule this scorer implements.

    Its six subset rates over 200/200/200/200/200/100 queries must give the 77.55% it reports. This
    asserts the ARITHMETIC, not the result — it is what pins the aggregation rule to the published
    one, since the paper never states which mean it took.
    """
    subsets = [
        ("G1_instruction", 200, 78.5), ("G1_category", 200, 74.0), ("G1_tool", 200, 79.0),
        ("G2_category", 200, 80.5), ("G2_instruction", 200, 74.5), ("G3_instruction", 100, 80.0),
    ]
    passes = sum(round(n * rate / 100) for _name, n, rate in subsets)
    total = sum(n for _name, n, _rate in subsets)
    assert passes == 853
    assert total == 1100
    assert round(100 * passes / total, 2) == 77.55
    # And the mean of the six rates, which is what a careless implementation would report.
    assert round(sum(r for _n, _c, r in subsets) / 6, 2) == 77.75


def test_an_empty_eval_set_scores_zero_without_dividing_by_zero():
    result = score(EvalSet(all=[], task="toolbench"), [], judge=StubJudge(1.0))
    assert result["f1"] == 0.0
    assert result["format_valid"] == 0.0
    assert result["failures"] == []


def test_format_valid_separates_unreadable_output_from_a_wrong_answer():
    unreadable = _score([row()], ["I cannot help with that."], StubJudge(1.0))
    assert unreadable["format_valid"] == 0.0
    assert unreadable["f1"] == 0.0

    wrong = _score([row()], [path(final="Paris is in France.")], StubJudge(0.0))
    assert wrong["format_valid"] == 1.0
    assert wrong["f1"] == 0.0


def test_a_failure_record_carries_the_path_and_the_votes():
    """Surgical synthesis and the per-difficulty report both read failure records. The votes are
    kept because a 2-1 loss and a 0-3 loss are different facts about the same score."""
    result = _score([row()], [path()], StubJudge([0.0, 1.0, 0.0]))
    failure = result["failures"][0]
    assert failure["text"]
    assert failure["predicted"]["actions"][-1] == "Finish"
    assert failure["judge_votes"] == [0.0, 1.0, 0.0]
    assert failure["verdict"] == "unsolved"


def test_a_row_with_no_subset_label_is_reported_rather_than_dropped():
    """A row from a rebuild or a mined source has no `_subset`. It still counts toward the headline
    rate — silently excluding it would make the denominator disagree with the eval set."""
    unlabelled = row()
    unlabelled.pop("_subset")
    result = _score([unlabelled], [path()], StubJudge(1.0))
    assert result["f1"] == 1.0
    assert result["per_class"]["unlabelled"] == 1.0


def test_failure_category_recomputes_when_no_category_was_stamped():
    from eval.scorers.toolbench import failure_category_of

    assert failure_category_of({"predicted": None, "tools": TOOLS}) == "unparseable_path"
    stamped = {"error_type": "gave_up", "predicted": None}
    assert failure_category_of(stamped) == "gave_up"


def test_the_categories_for_unreadable_and_for_wrong_are_disjoint():
    """They call for different interventions — a prompt/template fix versus more or better data —
    so a taxonomy that gives them the same name tells the orchestrator nothing (B296)."""
    unreadable = failure_category(row(), None)
    wrong = failure_category(row(), parse_solution_path(path()))
    assert unreadable == "unparseable_path"
    assert wrong == "judged_unsolved"
    assert unreadable != wrong
