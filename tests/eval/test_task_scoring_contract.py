"""Every task's prompt -> extract -> score chain round-trips, and reports format apart from content.

WHY THIS FILE EXISTS
    Two separate defects motivate it.

    The first is structural. `build_prompts`, `extract_predictions` and `score` used to branch on
    `eval_set.task_type` inside the scorer modules, so a task got whichever branch its channel
    happened to land in and the `else` was the LLM judge. Anything routed to `generation` without a
    matching branch was silently judged instead of checked. There is no branch left; each task
    names its three callables. This file drives all eight through the whole chain on a row of that
    task's own shape, which is the only way to catch a task whose spec names three functions that
    do not actually compose.

    The second is the format/content split (B290). A low score means two completely different
    things: a CONTENT problem more data can fix, or a FORMAT problem only the prompt or the chat
    template can. The xlam run that forced this was diagnosed entirely from the gap between the
    two numbers — fine-tuned predictions were `<think></think>[{...}]`, scoring 0.0000 while the
    JSON inside them was often right. So every scorer must report `format_valid` alongside its
    content score, an unparseable prediction must LOWER it, and a parseable-but-wrong prediction
    must NOT.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from data.eval_set import EvalSet
from tasks import TASKS, get_task

# Long enough that the classification extractor's short-answer fallback cannot fire, and free of
# any substring of any label in this file — a chatty non-answer, which is what a base model
# produces when it obeys the row's own embedded instruction instead of classifying (B271).
CHATTY = (
    "Certainly! Here is a fuller explanation of the matter you have raised, with several "
    "considerations and caveats that are worth bearing in mind before deciding anything."
)

CALENDAR_TOOL = {
    "name": "calendar.events.insert",
    "parameters": {
        "type": "dict",
        "required": ["summary", "start", "end"],
        "properties": {
            "summary": {"type": "string"},
            "start": {"type": "dict"},
            "end": {"type": "dict"},
        },
    },
}
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _calendar_answer(summary="Dentist", start="2026-03-03T10:00:00", end="2026-03-03T11:00:00"):
    return json.dumps([{
        "name": "calendar.events.insert",
        "arguments": {"summary": summary,
                      "start": {"dateTime": start},
                      "end": {"dateTime": end}},
    }])


def _xlam_answer(city="Paris"):
    return json.dumps([{"name": "get_weather", "arguments": {"city": city}}])


TOOLBENCH_TOOLS = [{
    "name": "get_weather_for_weather_api",
    "parameters": {"properties": {"city": "string"}, "required": ["city"], "optional": []},
}]


def _toolbench_path(city="Paris", final="It is 18 degrees and clear in Paris."):
    """A complete ToolBench solution path: one declared API call, then Finish->give_answer."""
    return (
        "Thought: I should look up the weather for that city.\n"
        f"Action: get_weather_for_weather_api\nAction Input: {{\"city\": \"{city}\"}}\n"
        "Thought: I have what I need and can answer now.\n"
        f'Action: Finish\nAction Input: {{"return_type": "give_answer", "final_answer": "{final}"}}'
    )


# One fixture per task: rows in that task's OWN shape, the raw output a perfect model would emit,
# a raw output the extractor cannot read at all, and one it can read but which is wrong.
#
# Written out per task rather than generated, because the point is to exercise each task's real row
# schema. A generated row would share a shape across tasks and hide exactly the mismatch this file
# is looking for.
FIXTURES: dict[str, dict] = {
    "gsm8k": {
        "rows": [
            {"text": "Janet has 3 apples and buys 15 more. How many?",
             "answer": "She adds them.\n#### 18"},
            {"text": "Tom runs 4 miles a day for 5 days. How far?",
             "answer": "Multiply.\n#### 20"},
        ],
        "perfect": ["Adding gives\n#### 18", "Multiplying gives\n#### 20"],
        "unreadable": ["The result cannot be determined from what was provided.",
                       "No answer is possible here."],
        "wrong_but_readable": ["#### 999", "#### 999"],
    },
    # THREE references per row, which is the whole premise of this task and the reason it was
    # reworked off the LLM judge on 2026-09-06. `perfect` matches the SECOND reference of each
    # row, not the first, so the fixture actually exercises max-over-references: against
    # reference 1 alone these predictions would score poorly.
    #
    # `unreadable` is a transcript TURN LABEL (`#Person1#:`), not an empty string. That is the
    # real format failure for this task — the B250 continuation — and note the reference summaries
    # themselves mention `#Person1#` without a colon, which is legitimate output.
    "dialogsum": {
        "rows": [
            {"text": "#Person1#: Hi. #Person2#: Hello, are we still on for lunch?",
             "answer": "Two people confirm a lunch plan.",
             "references": ["Two people confirm a lunch plan.",
                            "#Person1# and #Person2# agree to meet for lunch.",
                            "A lunch arrangement is confirmed between two colleagues."]},
            {"text": "#Person1#: The report is late. #Person2#: I'll send it tonight.",
             "answer": "One person promises to send a late report tonight.",
             "references": ["One person promises to send a late report tonight.",
                            "#Person2# will send the overdue report this evening.",
                            "The late report is promised for tonight."]},
        ],
        "perfect": ["#Person1# and #Person2# agree to meet for lunch.",
                    "#Person2# will send the overdue report this evening."],
        "unreadable": ["#Person1#: How about you? #Person2#: Nah.",
                       "#Person1#: Sure, tell me more."],
        "wrong_but_readable": ["A recipe for bread.", "An unrelated weather forecast."],
    },
    # The parse string is carried VERBATIM, so `perfect` is the gold. `unreadable` is prose with
    # no bracketed tree; `wrong_but_readable` is a well-formed tree with the wrong intent, which
    # must land in a different failure category from the unparseable one.
    "topv2": {
        "rows": [
            {"text": "set alarm for 6 am",
             "answer": "[IN:CREATE_ALARM set alarm [SL:DATE_TIME for 6 am ] ]",
             "domain": "reminder"},
            {"text": "will it rain tomorrow",
             "answer": "[IN:GET_WEATHER will it rain [SL:DATE_TIME tomorrow ] ]",
             "domain": "weather"},
        ],
        "perfect": ["[IN:CREATE_ALARM set alarm [SL:DATE_TIME for 6 am ] ]",
                    "[IN:GET_WEATHER will it rain [SL:DATE_TIME tomorrow ] ]"],
        "unreadable": ["I am not able to parse that command.",
                       "That request cannot be represented."],
        "wrong_but_readable": ["[IN:CREATE_REMINDER set alarm [SL:TODO for 6 am ] ]"] * 2,
    },
    # Fine-grained types from the 33-class taxonomy. `wrong_but_readable` names a type OUTSIDE the
    # taxonomy on purpose — a label-space error is the dominant and most actionable failure here,
    # and it has to be distinguishable from unparseable output.
    "multiconer": {
        "rows": [
            {"text": "robert gottschalk founded panavision",
             "entities": [{"text": "robert gottschalk", "type": "OtherPER"},
                          {"text": "panavision", "type": "ORG"}]},
            {"text": "aspirin treats gastritis",
             "entities": [{"text": "aspirin", "type": "Medication/Vaccine"},
                          {"text": "gastritis", "type": "Disease"}]},
        ],
        "perfect": [
            json.dumps([{"text": "robert gottschalk", "type": "OtherPER"},
                        {"text": "panavision", "type": "ORG"}]),
            json.dumps([{"text": "aspirin", "type": "Medication/Vaccine"},
                        {"text": "gastritis", "type": "Disease"}]),
        ],
        "unreadable": ["I could not find any named entities in the text.",
                       "There are no entities."],
        "wrong_but_readable": [json.dumps([{"text": "panavision", "type": "PERSON"}])] * 2,
    },
    # The `m2` blocks here are ERRANT-GENERATED, not hand-annotated, and that is deliberate:
    # feeding the real corpus's gold correction back in as the hypothesis scores F0.5 0.8934
    # rather than 1.0, because the reference edits are a human's segmentation while the hypothesis
    # edits are derived by ERRANT's alignment rules. This fixture tests the plumbing, so its
    # reference is machine-generated and the oracle really does reach 1.0.
    #
    # `unreadable` is a genuinely multi-line reply: ERRANT aligns file lines positionally, so an
    # extra line of prose shifts every subsequent sentence and is a FORMAT failure, not a wrong
    # correction.
    "gec_bea19": {
        "rows": [
            {"text": "I has went to the store yesterday .",
             "answer": "I went to the store yesterday .",
             "cefr": "A",
             "m2": "S I has went to the store yesterday .\n"
                   "A 1 2|||U:VERB:TENSE||||||REQUIRED|||-NONE-|||0"},
            {"text": "She are very happy today .",
             "answer": "She is very happy today .",
             "cefr": "B",
             "m2": "S She are very happy today .\n"
                   "A 1 2|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0"},
        ],
        "perfect": ["I went to the store yesterday .", "She is very happy today ."],
        "unreadable": ["I went to the store yesterday .\nHope that helps!",
                       "She is very happy today .\nLet me know if you need more."],
        "wrong_but_readable": ["I has went to the store yesterday .",
                               "She are very happy today ."],
    },
    # Multi-label, so `perfect` is the comma-joined gold. `unreadable` names nothing in the
    # 28-label vocabulary, which for this task is unreadable rather than a considered prediction
    # of "no emotion" — `neutral` is the explicit escape and every gold row has a label.
    #
    # Both rows are Ekman-`joy` and Ekman-`anger` respectively, so the SELECTION metric
    # (Ekman-7 macro-F1) is well defined over them and a perfect answer reaches 1.0.
    "goemotions": {
        "rows": [
            {"text": "thank you so much, this made my day",
             "labels": ["gratitude", "joy"], "label": "gratitude, joy"},
            {"text": "this is the worst decision anyone has ever made",
             "labels": ["anger"], "label": "anger"},
        ],
        "perfect": ["gratitude, joy", "anger"],
        "unreadable": [CHATTY, CHATTY],
        "wrong_but_readable": ["anger", "gratitude, joy"],
    },
    "xlam_bfcl": {
        "rows": [
            {"text": "weather in Paris?", "answer": _xlam_answer("Paris"),
             "tools": [WEATHER_TOOL]},
            {"text": "weather in Oslo?", "answer": _xlam_answer("Oslo"),
             "tools": [WEATHER_TOOL]},
        ],
        "perfect": [_xlam_answer("Paris"), _xlam_answer("Oslo")],
        "unreadable": ["I'm sorry, I can't help with that.", "no function applies here"],
        "wrong_but_readable": [_xlam_answer("Berlin"), _xlam_answer("Berlin")],
    },
    # ToolBench's `wrong_but_readable` is a well-formed, in-budget path ending in
    # Finish->give_answer whose stated answer is about something else, so no exact rule can reject
    # it — it reaches the judge, which is the point. `unreadable` is prose containing no Action at
    # all. The stub judge below passes only an answer matching the row's expected one.
    #
    # These rows carry an `answer` even though real ToolEval eval rows do not: pass rate is
    # reference-free and the test queries ship with no gold path. What is written here is a TRAIN
    # row's shape, which is the union — because this file also drives `build_training_turn`, and a
    # row with no gold cannot become a training turn by design. The scorer ignores `answer`
    # entirely, so its presence changes nothing it asserts.
    "toolbench": {
        "rows": [
            {"text": "You are AutoGPT... \nwhat is the weather in Paris?\nBegin!\n",
             "query": "what is the weather in Paris?",
             "answer": _toolbench_path("Paris", "It is 18 degrees and clear in Paris."),
             "tools": TOOLBENCH_TOOLS,
             "_subset": "G1_instruction",
             "_expected": "It is 18 degrees and clear in Paris."},
            {"text": "You are AutoGPT... \nwhat is the weather in Oslo?\nBegin!\n",
             "query": "what is the weather in Oslo?",
             "answer": _toolbench_path("Oslo", "It is 2 degrees and snowing in Oslo."),
             "tools": TOOLBENCH_TOOLS,
             "_subset": "G2_category",
             "_expected": "It is 2 degrees and snowing in Oslo."},
        ],
        "perfect": [
            _toolbench_path("Paris", "It is 18 degrees and clear in Paris."),
            _toolbench_path("Oslo", "It is 2 degrees and snowing in Oslo."),
        ],
        "unreadable": ["I'm sorry, I can't help with that.", "no tool applies here"],
        "wrong_but_readable": [
            _toolbench_path("Paris", "The capital of France is Paris."),
            _toolbench_path("Oslo", "The capital of France is Paris."),
        ],
    },
    "calendar_json": {
        "rows": [
            {"text": ("Convert the request into a calendar.events.insert call.\n"
                      "Current date and time: 2026-03-01T09:00:00 (Sunday).\n\n"
                      'Add "Dentist" on March 3rd at 10:00 am'),
             "answer": _calendar_answer(), "tools": [CALENDAR_TOOL]},
            {"text": ("Convert the request into a calendar.events.insert call.\n"
                      "Current date and time: 2026-03-01T09:00:00 (Sunday).\n\n"
                      'Add "Standup" on March 4th at 09:00 am'),
             "answer": _calendar_answer("Standup", "2026-03-04T09:00:00",
                                        "2026-03-04T10:00:00"),
             "tools": [CALENDAR_TOOL]},
        ],
        "perfect": [_calendar_answer(),
                    _calendar_answer("Standup", "2026-03-04T09:00:00", "2026-03-04T10:00:00")],
        "unreadable": ["I'll add that to your calendar.", "Sure, done!"],
        "wrong_but_readable": [_calendar_answer("Dentist", "2029-03-03T10:00:00",
                                                "2029-03-03T11:00:00")] * 2,
    },
    "ner_bc5cdr": {
        "rows": [
            {"text": "Aspirin induced gastritis in the patient.",
             "entities": [{"text": "Aspirin", "type": "Chemical"},
                          {"text": "gastritis", "type": "Disease"}]},
            {"text": "Ibuprofen caused nephritis.",
             "entities": [{"text": "Ibuprofen", "type": "Chemical"},
                          {"text": "nephritis", "type": "Disease"}]},
        ],
        "perfect": [
            json.dumps([{"text": "Aspirin", "type": "Chemical"},
                        {"text": "gastritis", "type": "Disease"}]),
            json.dumps([{"text": "Ibuprofen", "type": "Chemical"},
                        {"text": "nephritis", "type": "Disease"}]),
        ],
        "unreadable": ["I could not find any named entities in the text.",
                       "There are no entities."],
        "wrong_but_readable": [json.dumps([{"text": "Warfarin", "type": "Chemical"}])] * 2,
    },
    "routerbench": {
        "rows": [
            {"text": "What is 2 + 2?", "label": "local"},
            {"text": "Summarise this 200-page antitrust filing.", "label": "route"},
            {"text": "Spell 'cat'.", "label": "local"},
            {"text": "Prove the Riemann hypothesis.", "label": "route"},
        ],
        "perfect": ["local", "route", "local", "route"],
        "unreadable": [CHATTY] * 4,
        "wrong_but_readable": ["route", "local", "route", "local"],
    },
    # Two of each class, so `minority_f1` is well defined and the all-majority prediction the
    # `wrong_but_readable` row exercises can actually score 0 rather than being undefined.
    "sms_spam": {
        "rows": [
            {"text": "running 10 min late, order me a coffee", "label": "ham"},
            {"text": "WINNER! Claim your free prize now, txt CLAIM to 81010", "label": "spam"},
            {"text": "can you pick up milk on the way home", "label": "ham"},
            {"text": "URGENT: your account is suspended, click here to verify", "label": "spam"},
        ],
        "perfect": ["ham", "spam", "ham", "spam"],
        "unreadable": [CHATTY] * 4,
        "wrong_but_readable": ["spam", "ham", "spam", "ham"],
    },
    "proactive_listening": {
        "rows": [
            {"text": "A: I need the code... um...", "label": "interrupt"},
            {"text": "A: So anyway, as I was saying,", "label": "wait"},
            {"text": "A: Her number was, hold on,", "label": "interrupt"},
            {"text": "A: And then we drove home.", "label": "wait"},
        ],
        "perfect": ["interrupt", "wait", "interrupt", "wait"],
        "unreadable": [CHATTY] * 4,
        "wrong_but_readable": ["wait", "interrupt", "wait", "interrupt"],
    },
    "clinc150": {
        "rows": [
            {"text": "move money to savings", "label": "transfer"},
            {"text": "how much is in checking", "label": "balance"},
            {"text": "what colour is the sky", "label": "oos"},
            {"text": "send funds to my other account", "label": "transfer"},
        ],
        "perfect": ["transfer", "balance", "oos", "transfer"],
        "unreadable": [CHATTY] * 4,
        "wrong_but_readable": ["balance", "transfer", "balance", "oos"],
    },
}

TASK_IDS = sorted(FIXTURES)


@pytest.fixture(autouse=True)
def _no_live_judge(monkeypatch):
    """The judge is a hosted model. It is replaced with a scorer that reads the gold, so a judged
    task is exercised through the same chain as the rest without a network call.

    Scoring 1.0 only on an exact gold match is deliberately cruder than the real judge; these
    tests assert the CHAIN composes and that format is reported apart from content, never that
    the judge is calibrated.

    Two rubrics need stubbing because two tasks are judged. `dialogsum` reaches its judge through
    `eval.scorers.generation`'s module-level name; `toolbench` constructs one inside `score` from
    `eval.judge_client`, so that is where its stub goes.
    """
    import eval.judge_client as judge_client
    import eval.scorers.generation as generation

    class _Judge:
        @classmethod
        def from_config(cls):
            return cls()

        def score_many(self, triples):
            return [
                1.0 if str(prediction).strip() == str(gold).strip() else 0.0
                for _text, gold, prediction in triples
            ]

    # ToolEval's rubric is asked "does this answer solve this query"; the stub answers it by
    # looking up what the fixture declared the right answer to be for that query.
    expected_by_query = {
        row["query"]: row["_expected"] for row in FIXTURES["toolbench"]["rows"]
    }

    class _ToolEvalJudge:
        @classmethod
        def from_config(cls, rubric=None):
            return cls()

        def score_payloads(self, payloads):
            return [
                1.0
                if str(payload["answer"]).strip()
                == expected_by_query.get(str(payload["query"]), object())
                else 0.0
                for payload in payloads
            ]

    monkeypatch.setattr(generation, "LocalJudgeClient", _Judge)
    monkeypatch.setattr(judge_client, "LocalJudgeClient", _ToolEvalJudge)


def _eval_set(task: str) -> EvalSet:
    return EvalSet(all=[dict(row) for row in FIXTURES[task]["rows"]], task=task)


def _score(task: str, raw_outputs: list[str]) -> dict:
    spec = get_task(task)
    eval_set = _eval_set(task)
    predictions = spec.extract_predictions(raw_outputs, eval_set)
    return spec.score(eval_set, predictions)


def test_every_registered_task_has_a_fixture():
    """A task with no fixture here is a task nothing in this file exercises, which is how a
    silently-broken chain would slip through."""
    assert set(FIXTURES) == set(TASKS)


# --------------------------------------------------------------------------
# The chain composes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASK_IDS)
def test_prompts_are_built_one_per_row_and_contain_the_row(task):
    spec = get_task(task)
    eval_set = _eval_set(task)
    prompts = spec.build_prompts(eval_set)
    assert len(prompts) == len(eval_set.all)
    for prompt, row in zip(prompts, eval_set.all):
        assert isinstance(prompt, str) and prompt.strip()
        assert row["text"] in prompt, f"{task} prompt does not carry the row's own text"


@pytest.mark.parametrize("task", TASK_IDS)
def test_a_perfect_answer_round_trips_to_a_full_score(task):
    """The end-to-end property: the output a correct model emits, fed back through this task's own
    extractor and scorer, scores 1.0. A task whose three callables do not compose fails here."""
    result = _score(task, FIXTURES[task]["perfect"])
    assert result["f1"] == pytest.approx(1.0), f"{task}: {result}"
    assert result["format_valid"] == pytest.approx(1.0)
    assert result["failures"] == []


@pytest.mark.parametrize("task", TASK_IDS)
def test_the_result_reports_both_a_content_score_and_a_format_score(task):
    """`format_valid` at the TOP LEVEL and inside `per_class`. Both, because `EvalResult` reads the
    top level while the per-class block is what reaches the report and the orchestrator."""
    spec = get_task(task)
    result = _score(task, FIXTURES[task]["wrong_but_readable"])

    assert 0.0 <= result["f1"] <= 1.0
    assert 0.0 <= result["format_valid"] <= 1.0
    assert result["per_class"]["format_valid"] == pytest.approx(result["format_valid"])
    assert result["metric"] == spec.metric_name, (
        f"{task} scores under {result['metric']!r} but its spec says {spec.metric_name!r}"
    )


@pytest.mark.parametrize("task", TASK_IDS)
def test_an_unreadable_prediction_lowers_the_format_score(task):
    """Output the extractor cannot read at all is not a prediction, and saying so is the whole
    point of the second number."""
    result = _score(task, FIXTURES[task]["unreadable"])
    assert result["format_valid"] == pytest.approx(0.0), f"{task}: {result}"
    assert result["f1"] == pytest.approx(0.0)


@pytest.mark.parametrize("task", TASK_IDS)
def test_a_readable_but_wrong_prediction_does_not_lower_the_format_score(task):
    """The distinction that makes the pair of numbers diagnostic. A model that answers in the right
    shape and picks the wrong content has a DATA problem; collapsing it into the format number
    would send the loop after the prompt instead."""
    result = _score(task, FIXTURES[task]["wrong_but_readable"])
    assert result["format_valid"] == pytest.approx(1.0), f"{task}: {result}"
    assert result["f1"] < 1.0, f"{task} scored a wrong answer as correct"
    assert result["failures"], f"{task} recorded no failure for a wrong answer"


@pytest.mark.parametrize("task", TASK_IDS)
def test_failures_carry_the_row_and_the_prediction(task):
    """Surgical synthesis and the per-difficulty report both read failure records; one without the
    row's own text cannot be attributed to a difficulty bucket or anchored on."""
    result = _score(task, FIXTURES[task]["wrong_but_readable"])
    for failure in result["failures"]:
        assert failure.get("text"), f"{task} failure record has no text"
        assert "predicted" in failure


# --------------------------------------------------------------------------
# Failure categories name something the scorer measured
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASK_IDS)
def test_every_failure_gets_a_category_from_the_task_taxonomy(task):
    """Every open-ended failure used to be reported as the constant confusion pair
    `gold_verifier -> incorrect`, whose count is the failure count the orchestrator already had —
    and it wrote pages of reasoning about that constant (B296). A category must be a non-empty
    name, and the constant must be gone."""
    spec = get_task(task)
    assert spec.failure_category is not None, f"{task} declares no failure taxonomy"
    for raw_kind in ("unreadable", "wrong_but_readable"):
        result = _score(task, FIXTURES[task][raw_kind])
        for failure in result["failures"]:
            category = spec.failure_category(failure)
            assert isinstance(category, str) and category.strip(), f"{task}/{raw_kind}"
            assert category != "gold_verifier", f"{task} still reports the B296 constant"


@pytest.mark.parametrize("task", TASK_IDS)
def test_unreadable_and_wrong_are_different_categories(task):
    """They call for different interventions — a prompt/template fix versus more data — so a
    taxonomy that gives them the same name tells the orchestrator nothing it can act on."""
    spec = get_task(task)
    unreadable = {
        spec.failure_category(f) for f in _score(task, FIXTURES[task]["unreadable"])["failures"]
    }
    wrong = {
        spec.failure_category(f)
        for f in _score(task, FIXTURES[task]["wrong_but_readable"])["failures"]
    }
    assert unreadable and wrong
    assert not (unreadable & wrong), f"{task} cannot distinguish unreadable from wrong: {unreadable}"


# --------------------------------------------------------------------------
# `None` is not `[]`
# --------------------------------------------------------------------------


def test_ner_distinguishes_unparseable_output_from_an_empty_span_list():
    """They used to be the same value, which made a model emitting prose indistinguishable from one
    correctly reporting no entities. Since most rows DO have entities both scored zero, so the
    format failure was invisible."""
    from eval.scorers.ner import extract_predictions

    eval_set = _eval_set("ner_bc5cdr")
    assert extract_predictions(["[]"], eval_set) == [[]]
    assert extract_predictions(["there are no entities here"], eval_set) == [None]
    assert extract_predictions(['{"not": "a list"}'], eval_set) == [None]


def test_a_legitimate_empty_span_list_is_a_real_prediction():
    """`[]` is format-VALID: the model followed the contract ("Reply with [] if there are no
    entities") and is simply wrong about the content."""
    from eval.scorers.ner import score

    eval_set = _eval_set("ner_bc5cdr")
    result = score(eval_set, [[], []])
    assert result["format_valid"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.0)


def test_function_call_distinguishes_unparseable_output_from_an_empty_call_list():
    from eval.scorers.function_call import extract_predictions

    eval_set = _eval_set("xlam_bfcl")
    assert extract_predictions(["[]"], eval_set) == [[]]
    assert extract_predictions(["I cannot help"], eval_set) == [None]


# --------------------------------------------------------------------------
# The harness reads every choice off the spec
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASK_IDS)
def test_run_eval_takes_the_task_from_the_eval_set_and_returns_both_numbers(task, monkeypatch):
    """`run_eval` no longer takes a task argument: the eval set is the thing that knows which rows
    these are, so the two can no longer be passed inconsistently.

    Inference is stubbed. The assertion is on the WIRING — that the spec's prompt builder,
    extractor, scorer and token reserve are the ones used, and that `format_valid` survives onto
    `EvalResult` rather than being dropped between the scorer and the loop.
    """
    import eval.harness as harness
    import training.cuda_isolation as isolation

    monkeypatch.setattr(isolation, "isolation_enabled", lambda: False)
    spec = get_task(task)
    seen: dict = {}

    def fake_infer(prompts, weights_ref, base_model, **kwargs):
        seen["prompts"] = list(prompts)
        seen["max_new_tokens"] = kwargs.get("max_new_tokens")
        seen["task"] = kwargs.get("task")
        return list(FIXTURES[task]["perfect"])

    monkeypatch.setattr(harness, "infer_batch", fake_infer)
    # dialogsum overlaps judging with the next generation chunk; that path spawns a thread and
    # touches the real judge cache, and it is covered on its own in tests/eval.
    monkeypatch.setenv("SLM_EVAL_JUDGE_OVERLAP_CHUNK", "0")

    result = harness.run_eval(_eval_set(task), "/weights", "Qwen/Qwen3-0.6B")

    assert result.f1 == pytest.approx(1.0)
    assert result.format_valid == pytest.approx(1.0)
    assert result.metric == spec.metric_name
    assert seen["task"] == task
    assert seen["max_new_tokens"] == spec.max_new_tokens
    assert seen["prompts"] == spec.build_prompts(_eval_set(task))


def test_eval_result_reports_format_and_no_longer_carries_execution_diagnostics():
    """`execution_diagnostics` belonged to the code-execution sandbox, deleted with APPS/MBPP on
    2026-08-18. `format_valid` replaced it as the universal second number."""
    from eval.harness import EvalResult

    result = EvalResult(f1=0.5, per_class={}, failures=[])
    assert result.format_valid == 1.0
    assert not hasattr(result, "execution_diagnostics")


@pytest.mark.parametrize("task", TASK_IDS)
def test_the_endpoint_baseline_scores_through_the_same_spec(task, monkeypatch):
    """The reference model's zero-shot number becomes the accuracy goal, so it has to be measured
    by the identical scorer. These were two separate dispatch chains kept in sync by hand."""
    from eval.endpoint_eval import measure_endpoint_baseline

    outputs = iter(FIXTURES[task]["perfect"])

    def generate(_prompt, temperature=0.0, max_tokens=0):
        return next(outputs)

    result = measure_endpoint_baseline(
        _eval_set(task), generate_fn=generate, max_workers=1, log=lambda _m: None,
    )
    assert result is not None
    assert result.f1 == pytest.approx(1.0)
    assert result.metric == get_task(task).metric_name


def test_the_endpoint_baseline_returns_none_when_the_endpoint_is_unreachable(monkeypatch):
    """None, not zero. A zero is indistinguishable from a reference model that cannot do the task,
    and the accuracy goal is derived from this number."""
    import data.synth_client as synth_client
    from eval.endpoint_eval import measure_endpoint_baseline

    monkeypatch.setattr(synth_client, "get_generate_fn", lambda **_kwargs: None)
    logs: list[str] = []
    assert measure_endpoint_baseline(
        _eval_set("clinc150"), log=logs.append,
    ) is None
    assert any("unavailable" in line for line in logs)


# --------------------------------------------------------------------------
# Train/serve prompt parity, per task
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASK_IDS)
def test_the_training_prompt_is_the_prompt_inference_sends(task):
    """B290/B250, at the task level: `build_training_turn` must produce the SAME prompt the eval
    harness builds, or training teaches a prefix inference never supplies.

    The builders import their prompt from the eval scorer rather than reproducing it — when the
    two were written separately the model was fine-tuned on one input shape and scored on another,
    and the NER training copy had quietly dropped the "Reply with [] if there are no entities"
    sentence. This asserts the outcome of that, not the mechanism.
    """
    from tasks._builders import TrainingContext

    spec = get_task(task)
    eval_set = _eval_set(task)
    inference_prompts = spec.build_prompts(eval_set)
    labels = tuple(sorted({str(row["label"]) for row in eval_set.all if "label" in row}))
    instruction = _instruction_for(task, eval_set)
    ctx = TrainingContext(labels=labels, instruction=instruction)

    for row, inference_prompt in zip(eval_set.all, inference_prompts):
        training_prompt, target, *_ = spec.build_training_turn(row, ctx)
        assert training_prompt == inference_prompt, (
            f"{task}: training prompt differs from the inference prompt\n"
            f"  train: {training_prompt[:200]!r}\n"
            f"  serve: {inference_prompt[:200]!r}"
        )
        assert str(target).strip(), f"{task} training turn has an empty target"


def _instruction_for(task: str, eval_set: EvalSet) -> str:
    """The dataset-level instruction the eval prompt builder resolved, so the training context is
    given the same one rather than a second guess at it."""
    if get_task(task).build_prompts.__module__ != "eval.scorers.generation":
        return ""
    from eval.scorers.generation import resolve_generation_instruction

    return resolve_generation_instruction(eval_set.all)


def test_a_function_call_row_with_no_answer_refuses_to_become_a_training_turn():
    """A row whose gold is empty teaches the model to emit nothing. Raising names the row; a silent
    empty target is a curriculum that quietly trains against itself."""
    from tasks._builders import TrainingContext, function_call_turn

    with pytest.raises(ValueError, match="empty 'answer'"):
        function_call_turn({"text": "q", "answer": "  ", "tools": [WEATHER_TOOL]},
                           TrainingContext(labels=(), instruction=""))


def test_a_chain_of_thought_target_still_ends_in_the_answer():
    """Only the ANSWER is scored, so a CoT-trained target must keep the reasoning in a block the
    extractor strips (B251) rather than merged into the answer."""
    from eval.scorers.generation import split_reasoning
    from tasks._builders import TrainingContext, generation_turn

    _prompt, target, *_ = generation_turn(
        {"text": "q", "answer": "#### 18", "cot_reasoning": "add three and fifteen"},
        TrainingContext(labels=(), instruction="Solve:"),
    )
    reasoning, answer = split_reasoning(target)
    assert "add three and fifteen" in reasoning
    assert answer == "#### 18"


def test_eval_sets_cannot_exist_for_an_unregistered_task():
    """Resolving at construction means no downstream consumer has to handle the case."""
    with pytest.raises(ValueError, match="unknown task"):
        EvalSet(all=[{"text": "a"}], task="code_generation")


def test_a_legacy_eval_set_recording_only_a_task_type_refuses_to_load():
    """`function_call` was two tasks and `classification` was three, so guessing would evaluate one
    task with another's configuration. Refusing names the reason and tells the reader to start
    fresh."""
    with pytest.raises(ValueError, match="task_type"):
        EvalSet.from_serialized({"all": [{"text": "a"}], "task_type": "function_call"})


def test_the_eval_set_lost_the_slices_that_had_no_effect():
    """`multi_label`, `schema` and `multilingual` were carried on every eval set and read by
    nobody. They are gone; `all` and `task` are the whole content."""
    eval_set = _eval_set("clinc150")
    for gone in ("multi_label", "schema", "multilingual"):
        assert not hasattr(eval_set, gone), f"EvalSet.{gone} is back"


def test_build_eval_set_round_robins_a_closed_label_space():
    """CLINC150 has 151 classes against an 800-row eval set: without round-robin some classes are
    absent entirely and macro-F1 averages over whichever ones happened to be drawn."""
    from data.eval_set import build_eval_set

    rows = (
        [{"text": f"a{i}", "label": "transfer"} for i in range(50)]
        + [{"text": f"b{i}", "label": "oos"} for i in range(2)]
    )
    built = build_eval_set(rows, task="clinc150", target=6)
    assert len(built.all) == 6
    assert {row["label"] for row in built.all} == {"transfer", "oos"}


def test_build_eval_set_shuffles_a_task_with_no_classes():
    from data.eval_set import build_eval_set

    rows = [{"text": f"t{i}", "answer": str(i)} for i in range(20)]
    built = build_eval_set(rows, task="gsm8k", target=5)
    assert len(built.all) == 5
    assert built.task == "gsm8k"


def test_the_token_reserve_comes_from_the_spec_and_an_override_is_validated():
    """The reserve used to be a dict keyed by task_type with a generous `.get(..., 4096)` default
    on the context side, so an unrecognised type silently received a budget nobody chose."""
    from eval.harness import eval_output_token_reserve

    for name, spec in TASKS.items():
        assert eval_output_token_reserve(name) == spec.max_new_tokens


def test_an_override_that_leaves_no_prompt_budget_is_rejected(monkeypatch):
    from eval.harness import eval_output_token_reserve

    spec = get_task("clinc150")
    monkeypatch.setenv("SLM_EVAL_MAX_NEW_TOKENS", str(spec.max_seq_length))
    with pytest.raises(ValueError, match="no prompt budget"):
        eval_output_token_reserve("clinc150")

    monkeypatch.setenv("SLM_EVAL_MAX_NEW_TOKENS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        eval_output_token_reserve("clinc150")


def test_scorers_are_not_selected_by_counting_labels():
    """`score_minority_f1` used to be chosen implicitly — `>2 classes` chose macro, otherwise
    minority — and the returned `metric` string said `macro_f1` either way. So RouterBench reported
    a minority-class F1 under a macro-F1 label for its whole history."""
    from eval.scorers.classification import score_macro_f1, score_minority_f1

    eval_set = _eval_set("routerbench")
    predictions = FIXTURES["routerbench"]["perfect"]
    assert score_macro_f1(eval_set, predictions)["metric"] == "macro_f1"
    assert score_minority_f1(eval_set, predictions)["metric"] == "minority_f1"
    # The task names one of them, and it is the one the harness uses.
    assert get_task("routerbench").score is score_minority_f1
    assert get_task("clinc150").score is score_macro_f1


def test_a_scorer_handles_an_empty_eval_set_without_dividing_by_zero():
    """Reached when a loader returns nothing; a crash here hides the real cause upstream."""
    for task in TASK_IDS:
        spec = get_task(task)
        empty = EvalSet(all=[], task=task)
        result = spec.score(empty, [])
        assert result["f1"] == 0.0
        assert result["format_valid"] == 0.0


def test_reasoning_is_recorded_on_failures_rather_than_judged():
    """The reasoning explains WHY an answer is wrong, so it is kept on the failure record — but it
    must not reach the judge, which is asked "how good is this summary?" (B251)."""
    from eval.harness import _attach_reasoning_to_failures

    result = {"failures": [{"text": "q", "predicted": "18"}]}
    _attach_reasoning_to_failures(
        result, _eval_set("gsm8k"),
        ["<reasoning>three plus fifteen</reasoning>\n18"], ["18"],
    )
    assert "three plus fifteen" in result["failures"][0]["reasoning"]


def test_every_task_states_whether_it_wants_reasoning_recorded():
    for name, spec in TASKS.items():
        assert isinstance(spec.attach_reasoning, bool), name


def test_the_deleted_scorers_are_gone():
    """The diff scorer and the code-execution sandbox went with APPS/MBPP on 2026-08-18. A leftover
    module is a scorer a task could still name.

    `data.loaders.sms_spam` was on this list until 2026-08-23 and has been deliberately removed
    from it: the task is registered again, so its loader importing is the correct state. What the
    list is guarding is modules with NO owning task — `sms_spam` now has one, and
    `test_every_task_module_is_registered` is what holds it to that.
    """
    import importlib

    for module in ("eval.scorers.code_execution", "eval.scorers.diff", "data.loaders.apps"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


def test_prediction_sampling_prefers_extraction_failures(capsys):
    """Scores alone cannot separate "picked the wrong class" from "answered in a format the
    extractor could not read", so the sample log shows the raw output beside the parsed value and
    puts the diagnostic rows first."""
    from eval.harness import _log_prediction_samples

    eval_set = _eval_set("clinc150")
    raw = ["chatty prose", "transfer", "balance", "oos"]
    predictions = ["__EXTRACTION_FAILED__", "transfer", "balance", "oos"]
    _log_prediction_samples(eval_set, raw, predictions)
    out = capsys.readouterr().out
    assert "EXTRACTION FAILED" in out
    assert "extraction failure(s)" in out


def test_ner_gold_is_shown_even_though_it_lives_in_entities(capsys):
    """B263: the sample display read only `answer`/`label`, so `gold :` printed BLANK on every NER
    row — removing the one human check of gold against prediction from the task where it matters
    most, and making a legitimate 0.0000 baseline indistinguishable from a broken harness."""
    from eval.harness import _log_prediction_samples

    eval_set = _eval_set("ner_bc5cdr")
    _log_prediction_samples(eval_set, ["[]", "[]"], [[], []])
    out = capsys.readouterr().out
    assert "Aspirin" in out and "Chemical" in out


def test_a_spec_cannot_be_built_without_stating_every_decision():
    """The guarantee `tasks/spec.py` exists for, asserted from the outside."""
    from tasks.spec import TaskSpec as Spec

    with pytest.raises(TypeError):
        Spec(name="incomplete")  # type: ignore[call-arg]


def test_a_spec_with_an_impossible_token_budget_is_rejected_at_construction():
    spec = get_task("clinc150")
    fields = {f: getattr(spec, f) for f in spec.__dataclass_fields__}
    fields.update({"name": "probe", "max_new_tokens": 4096, "max_seq_length": 1024})
    with pytest.raises(ValueError, match="no prompt budget"):
        type(spec)(**fields)


def test_a_spec_declaring_label_definitions_without_a_label_space_is_rejected():
    spec = get_task("clinc150")
    fields = {f: getattr(spec, f) for f in spec.__dataclass_fields__}
    fields.update({
        "name": "probe",
        "closed_label_space": False,
        "label_definitions": {"a": "means a"},
    })
    with pytest.raises(ValueError, match="closed label space"):
        type(spec)(**fields)


def test_the_spec_helper_returns_the_frozen_label_vocabulary():
    """`qc_context_labels` is what hands quality control the closed vocabulary, read from the
    FROZEN eval set rather than from whatever rows the run happens to hold."""
    assert get_task("routerbench").qc_context_labels(_eval_set("routerbench")) == {
        "local", "route",
    }
    # No closed space means no vocabulary to police, not an empty one.
    assert get_task("gsm8k").qc_context_labels(_eval_set("gsm8k")) is None
    assert get_task("routerbench").qc_context_labels(
        SimpleNamespace(all=[])
    ) is None
