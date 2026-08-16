"""Six curated benchmark loaders (2026-08-01).

Each loader has a PURE converter that shapes raw HF rows into the task's row schema; the live HF
pull runs on the cluster. These tests exercise the converters on small in-memory samples so the
row shaping is verified here, and confirm CoEdIT emits a git-applicable gold diff.
"""
import os
import shutil

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

import json  # noqa: E402

import pytest  # noqa: E402


# --- HF repo id hygiene --------------------------------------------------------
# huggingface_hub >= 1.0 parses every dataset reference as an `hf://datasets/<id>` URI and
# rejects bare canonical names ("clinc_oos") with HfUriError: a repo id must be
# 'namespace/name'. A namespace-less constant therefore fails at load time on the cluster,
# not at import time, so it survives every unit test until the run crashes.

CURATED_LOADER_REPO_ID_CONSTANTS = {
    "data.loaders.clinc150": ("HF_ID",),
    "data.loaders.routerbench": ("HF_ID",),
    "data.loaders.dialogsum_samsum": ("DIALOGSUM_ID", "SAMSUM_ID"),
    "data.loaders.xlam_bfcl": ("XLAM_ID", "BFCL_ID"),
}


@pytest.mark.parametrize(
    ("module_name", "constant"),
    [
        (module_name, constant)
        for module_name, constants in CURATED_LOADER_REPO_ID_CONSTANTS.items()
        for constant in constants
    ],
)
def test_curated_loader_repo_ids_are_namespaced(module_name, constant):
    import importlib

    repo_id = getattr(importlib.import_module(module_name), constant)
    namespace, sep, name = repo_id.partition("/")
    assert sep and namespace and name, (
        f"{module_name}.{constant} = {repo_id!r} is not a 'namespace/name' repo id; "
        "huggingface_hub >= 1.0 raises HfUriError for bare canonical dataset names"
    )
    assert "/" not in name, f"{module_name}.{constant} = {repo_id!r} has too many path segments"


# --- CLINC150 (classification) -------------------------------------------------

def test_clinc150_shapes_text_label_and_drops_empty():
    from data.loaders.clinc150 import convert_clinc150_rows

    rows = convert_clinc150_rows([
        {"text": "set an alarm for 6am", "label": "alarm"},
        {"text": "  ", "label": "alarm"},          # empty text dropped
        {"text": "asdkjfh", "intent": "oos"},       # intent used when label absent
    ])
    assert rows == [
        {"text": "set an alarm for 6am", "label": "alarm"},
        {"text": "asdkjfh", "label": "oos"},
    ]


# CLINC150's HF splits are grouped by intent (all ~100 rows of intent 61, then the next
# intent, ...). A head slice `train[:3250]` therefore yields only 33 of the 151 intents, and
# `test[:800]` a different 27 — the model would train and be scored on mismatched label
# spaces. The loader must draw across labels instead of taking a prefix.

def test_clinc150_stratified_sample_covers_every_label():
    from data.loaders.clinc150 import stratified_by_label

    rows = [
        {"text": f"utterance {label}-{i}", "label": label}
        for label in ("alarm", "balance", "oos", "translate")
        for i in range(50)
    ]
    picked = stratified_by_label(rows, 8)

    assert len(picked) == 8
    assert {row["label"] for row in picked} == {"alarm", "balance", "oos", "translate"}
    # Round-robin keeps the draw balanced, not front-loaded on one label.
    assert all(
        sum(row["label"] == label for row in picked) == 2
        for label in ("alarm", "balance", "oos", "translate")
    )


def test_clinc150_stratified_sample_is_deterministic_and_bounded():
    from data.loaders.clinc150 import stratified_by_label

    rows = [
        {"text": f"u{label}-{i}", "label": label}
        for label in ("a", "b", "c")
        for i in range(10)
    ]
    # Deterministic across calls: a requeued run must rebuild the identical curriculum.
    assert stratified_by_label(rows, 7) == stratified_by_label(rows, 7)
    # Never invents or drops rows at the boundaries.
    assert stratified_by_label(rows, 0) == []
    assert len(stratified_by_label(rows, 999)) == len(rows)
    assert all(row in rows for row in stratified_by_label(rows, 7))


def test_clinc150_loader_does_not_head_slice_grouped_splits():
    """Guard the root cause: an intent-grouped split must be read whole, then sampled."""
    import inspect

    from data.loaders import clinc150

    source = inspect.getsource(clinc150.load_clinc150)
    assert "[:" not in source.replace("[:limit]", ""), (
        "load_clinc150 must not head-slice the HF split; CLINC150 is grouped by intent so a "
        "prefix covers only a fraction of the 151 labels"
    )
    assert "stratified_by_label" in source


# --- DialogSum / SAMSum (generation) -------------------------------------------

def test_dialogsum_shapes_dialogue_to_answer():
    from data.loaders.dialogsum_samsum import convert_dialogsum_rows

    rows = convert_dialogsum_rows([
        {"dialogue": "A: hi\nB: hey", "summary": "A greets B."},
        {"dialogue": "", "summary": "x"},           # empty dialogue dropped
    ])
    assert len(rows) == 1
    assert rows[0]["text"] == "A: hi\nB: hey"
    assert rows[0]["answer"] == "A greets B."
    assert rows[0]["label"] == "generation"


# --- xLAM / BFCL (function_call) -----------------------------------------------

def test_xlam_normalizes_gold_calls_to_json_string():
    from data.loaders.xlam_bfcl import convert_xlam_rows

    rows = convert_xlam_rows([{
        "query": "what's the weather in Paris?",
        "tools": json.dumps([{"name": "get_weather", "parameters": {"city": "string"}}]),
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
    }])
    assert len(rows) == 1
    assert rows[0]["text"] == "what's the weather in Paris?"
    # answer is a canonical JSON string the function_call scorer can parse.
    parsed = json.loads(rows[0]["answer"])
    assert parsed == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    assert isinstance(rows[0]["tools"], list)


def test_xlam_drops_rows_without_usable_gold():
    from data.loaders.xlam_bfcl import convert_xlam_rows

    rows = convert_xlam_rows([
        {"query": "hi", "answers": "not json"},
        {"query": "", "answers": json.dumps([{"name": "f", "arguments": {}}])},
    ])
    assert rows == []


def test_xlam_gold_is_scored_correct_by_the_function_call_scorer():
    """End-to-end: the loader's gold call, fed back as the prediction, scores 1.0."""
    from data.eval_set import EvalSet
    from data.loaders.xlam_bfcl import convert_xlam_rows
    from eval.scorers.function_call import extract_predictions, score

    rows = convert_xlam_rows([{
        "query": "book a table for 2 at 7pm",
        "tools": [{"name": "book", "parameters": {"people": "int", "time": "string"}}],
        "answers": [{"name": "book", "arguments": {"people": 2, "time": "7pm"}}],
    }])
    es = EvalSet(all=rows, task_type="function_call")
    preds = extract_predictions([rows[0]["answer"]], es)
    result = score(es, preds)
    assert result["f1"] == 1.0
    assert result["metric"] == "ast_arg_match"


# --- CoEdIT (diff) -------------------------------------------------------------

def test_routerbench_derives_local_vs_route_label():
    from data.loaders.routerbench import convert_routerbench_rows

    rows = convert_routerbench_rows([
        {"prompt": "2+2?", "small_model_correct": 1},
        {"prompt": "prove Fermat", "small_model_correct": 0},
        {"prompt": "no signal here"},               # unresolvable → dropped
    ], small_model_key="small_model_correct")
    assert rows == [
        {"text": "2+2?", "label": "local"},
        {"text": "prove Fermat", "label": "route"},
    ]


def test_routerbench_threshold_on_float_score():
    from data.loaders.routerbench import convert_routerbench_rows

    rows = convert_routerbench_rows(
        [{"prompt": "q", "small_model_correct": 0.7}],
        small_model_key="small_model_correct", threshold=0.5)
    assert rows == [{"text": "q", "label": "local"}]


def test_routerbench_default_boundary_is_the_smallest_real_candidate_model():
    """The default used to be `small_model_correct`, a column RouterBench does not have — every
    row would resolve to None and be dropped, yielding a silently empty dataset (B254). The real
    columns are named after the candidate models."""
    from data.loaders.routerbench import _DEFAULT_SMALL_MODEL_KEY, convert_routerbench_rows

    assert _DEFAULT_SMALL_MODEL_KEY == "mistralai/mistral-7b-chat"
    rows = convert_routerbench_rows([
        {"prompt": "easy", "mistralai/mistral-7b-chat": 1.0},
        {"prompt": "hard", "mistralai/mistral-7b-chat": 0.0},
    ])
    assert [r["label"] for r in rows] == ["local", "route"]


def test_routerbench_unpacks_the_python_list_literal_prompt_column():
    """RouterBench's `prompt` column is a STRING containing a Python list literal of the message
    turns. Passing it through str() leaks brackets and quotes into every eval prompt."""
    from data.loaders.routerbench import convert_routerbench_rows

    rows = convert_routerbench_rows([{
        "prompt": "['You are a helpful assistant.', 'What is 2+2?']",
        "mistralai/mistral-7b-chat": 1.0,
    }])
    assert rows == [
        {"text": "You are a helpful assistant.\n\nWhat is 2+2?", "label": "local"},
    ]


def test_routerbench_leaves_a_plain_string_prompt_alone():
    from data.loaders.routerbench import convert_routerbench_rows

    rows = convert_routerbench_rows(
        [{"prompt": "just a prompt", "mistralai/mistral-7b-chat": 1.0}])
    assert rows[0]["text"] == "just a prompt"


# --- BFCL (function_call) ------------------------------------------------------

# One BFCL prompt row in the upstream shape: `question` is a list of turn-lists of chat messages,
# and `function` declares the tools. The gold lives in a SEPARATE possible_answer file, joined on
# `id` — which is what the test below exercises.
_BFCL_PROMPT = {
    "id": "simple_0",
    "question": [[{"role": "user", "content": "Area of a triangle, base 10 height 5."}]],
    "function": [{
        "name": "calculate_triangle_area",
        "description": "Calculate the area of a triangle.",
        "parameters": {
            "type": "dict",
            "properties": {
                "base": {"type": "integer", "description": "Base length."},
                "height": {"type": "integer", "description": "Height."},
                "unit": {"type": "string", "description": "Unit of measurement."},
            },
            "required": ["base", "height"],
        },
    }],
}


def test_bfcl_joins_prompts_to_possible_answers_on_id():
    from data.loaders.xlam_bfcl import convert_bfcl_rows

    rows = convert_bfcl_rows(
        [_BFCL_PROMPT, {**_BFCL_PROMPT, "id": "orphan"}],
        [{"id": "simple_0", "ground_truth": [
            {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]}],
    )
    # The orphan prompt has no gold and must be dropped, not guessed at.
    assert len(rows) == 1
    assert rows[0]["text"] == "Area of a triangle, base 10 height 5."
    assert rows[0]["label"] == "function_call"
    # `answer` is the canonical single-value gold: the FIRST acceptable value per argument. It is
    # used for display and as a training target, never for grading — grading reads `_accept`, so
    # the other acceptable values are not lost.
    assert json.loads(rows[0]["answer"]) == [{
        "name": "calculate_triangle_area",
        "arguments": {"base": 10, "height": 5, "unit": "units"},
    }]
    assert rows[0]["_accept"] == [
        {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]


def test_bfcl_drops_rows_whose_gold_calls_an_undeclared_function():
    """BFCL `simple_363` declares `restaurant_search.find_closest` but its gold calls
    `find_closest`. The scorer rejects calls outside the declared set, so the row is unwinnable
    and would silently cap the ceiling below 1.0."""
    from data.loaders.xlam_bfcl import convert_bfcl_rows

    rows = convert_bfcl_rows(
        [{"id": "x", "question": [[{"role": "user", "content": "find sushi"}]],
          "function": [{"name": "restaurant_search.find_closest", "parameters": {}}]}],
        [{"id": "x", "ground_truth": [{"find_closest": {"location": ["Boston"]}}]}],
    )
    assert rows == []


def test_bfcl_flattens_multi_message_questions_with_role_prefixes():
    from data.loaders.xlam_bfcl import convert_bfcl_rows

    rows = convert_bfcl_rows(
        [{"id": "y", "function": [{"name": "f", "parameters": {}}],
          "question": [[{"role": "system", "content": "Be terse."},
                        {"role": "user", "content": "call f"}]]}],
        [{"id": "y", "ground_truth": [{"f": {}}]}],
    )
    assert rows[0]["text"] == "system: Be terse.\nuser: call f"


def test_function_call_scorer_honours_bfcl_acceptable_value_semantics():
    from data.eval_set import build_eval_set
    from eval.scorers import function_call as fc

    row = {
        "text": "t",
        "answer": json.dumps([{"name": "f", "arguments": {"a": 1}}]),
        "tools": [{"name": "f"}],
        "label": "function_call",
        # `b` may be omitted (""); `a` accepts either 1 or "one"; `c` must be absent ([]).
        "_accept": [{"f": {"a": [1, "one"], "b": ["x", ""], "c": []}}],
    }
    es = build_eval_set([row], task_type="function_call", target=1)

    def scored(pred_json):
        return fc.score(es, fc.extract_predictions([pred_json], es))["f1"]

    assert scored('[{"name":"f","arguments":{"a":1}}]') == 1.0          # omits optional b
    assert scored('[{"name":"f","arguments":{"a":"one","b":"x"}}]') == 1.0  # 2nd acceptable value
    assert scored('[{"name":"f","arguments":{"a":2}}]') == 0.0          # value not acceptable
    assert scored('[{"name":"f","arguments":{"a":1,"c":"nope"}}]') == 0.0  # c must be absent
    assert scored('[{"name":"g","arguments":{"a":1}}]') == 0.0          # undeclared function


def test_function_call_scorer_matches_parallel_calls_order_insensitively():
    from data.eval_set import build_eval_set
    from eval.scorers import function_call as fc

    row = {
        "text": "t", "answer": "[]", "tools": [{"name": "play"}], "label": "function_call",
        "_accept": [{"play": {"artist": ["A"]}}, {"play": {"artist": ["B"]}}],
    }
    es = build_eval_set([row], task_type="function_call", target=1)
    reversed_order = '[{"name":"play","arguments":{"artist":"B"}},' \
                     '{"name":"play","arguments":{"artist":"A"}}]'
    assert fc.score(es, fc.extract_predictions([reversed_order], es))["f1"] == 1.0


def test_function_call_scorer_without_accept_is_unchanged():
    """xLAM rows carry no `_accept` and must still take the original equality path."""
    from data.eval_set import build_eval_set
    from eval.scorers import function_call as fc

    row = {"text": "t", "tools": [{"name": "f"}], "label": "function_call",
           "answer": json.dumps([{"name": "f", "arguments": {"a": 1}}])}
    es = build_eval_set([row], task_type="function_call", target=1)
    assert fc.score(es, fc.extract_predictions(['[{"name":"f","arguments":{"a":1}}]'], es))["f1"] == 1.0
    assert fc.score(es, fc.extract_predictions(['[{"name":"f","arguments":{"a":9}}]'], es))["f1"] == 0.0


# --- BC5CDR NER ----------------------------------------------------------------

def test_bc5cdr_bio_tags_collapse_into_entity_spans():
    from data.loaders.ner_bc5cdr import bio_to_spans

    spans = bio_to_spans(
        ["Naloxone", "reverses", "the", "antihypertensive", "effect", "of", "clonidine"],
        ["B-Chemical", "O", "O", "B-Disease", "I-Disease", "O", "B-Chemical"],
    )
    assert spans == [
        {"text": "Naloxone", "type": "Chemical"},
        {"text": "antihypertensive effect", "type": "Disease"},
        {"text": "clonidine", "type": "Chemical"},
    ]


def test_bc5cdr_resolves_integer_tag_ids_and_drops_length_mismatches():
    from data.loaders.ner_bc5cdr import TNER_TAG_NAMES, convert_tner_rows

    assert TNER_TAG_NAMES[1] == "B-Chemical"
    rows = convert_tner_rows([
        {"tokens": ["aspirin", "helps"], "tags": [1, 0]},
        {"tokens": ["a", "b"], "tags": [0]},  # ragged → dropped
    ])
    assert rows == [{"text": "aspirin helps",
                     "entities": [{"text": "aspirin", "type": "Chemical"}]}]


def test_ner_training_prompt_is_byte_identical_to_the_eval_prompt():
    """The trainer used to hand-copy a version omitting "Reply with [] if there are no
    entities", so the model was tuned on one input shape and scored on another — the same class
    of skew as B250, uncorrected through the whole 44.8h BC5CDR run."""
    from data.eval_set import build_eval_set
    from eval.scorers import ner as ner_scorer
    from training.lora_trainer import _training_turn

    example = {"text": "aspirin helps", "entities": [{"text": "aspirin", "type": "Chemical"}]}
    train_prompt, _target, _marker = _training_turn(example, "NER", [], "")
    eval_prompt = ner_scorer.build_prompts(build_eval_set([example], "NER", target=1))[0]
    assert train_prompt == eval_prompt


# --- Calendar NL -> JSON -------------------------------------------------------

def test_calendar_resolver_handles_the_common_expressions():
    from datetime import datetime

    from data.loaders.calendar_json import resolve_datetime

    ref = datetime(2026, 3, 12, 9, 0, 0)  # a Thursday
    cases = {
        "at 5 pm": "2026-03-12T17:00:00",
        "at 8 am": "2026-03-13T08:00:00",     # already past today → rolls to tomorrow
        "tomorrow at 5pm": "2026-03-13T17:00:00",
        "for tomorrow": "2026-03-13T09:00:00",
        "on Sunday": "2026-03-15T09:00:00",
        "at noon": "2026-03-12T12:00:00",
        "tonight": "2026-03-12T20:00:00",
        "at 8 : 30 am": "2026-03-13T08:30:00",   # TOPv2 splits punctuation apart
        "at 5 p.m .": "2026-03-12T17:00:00",
        "on April 2nd": "2026-04-02T09:00:00",
        "on the 4th of August": "2026-08-04T09:00:00",
    }
    for surface, expected in cases.items():
        got = resolve_datetime(surface, ref)
        assert got is not None, surface
        assert got.strftime("%Y-%m-%dT%H:%M:%S") == expected, surface


def test_calendar_resolver_refuses_what_it_cannot_pin_down():
    """Refusal is the point: this builds GOLD labels, so an expression that is relative to an
    unstated event, a span rather than an instant, or timezone-qualified must drop its row."""
    from datetime import datetime

    from data.loaders.calendar_json import resolve_datetime

    ref = datetime(2026, 3, 12, 9, 0, 0)
    for surface in ("15 minutes before", "an hour before", "this week", "next month",
                    "for the last day", "after 10 am Pacific Time", "every Tuesday",
                    "Thursday at 6 pm her time", "in four days", "before noon on Thursday"):
        assert resolve_datetime(surface, ref) is None, surface


def test_calendar_reference_is_stable_per_key_and_varies_across_keys():
    from data.loaders.calendar_json import reference_for

    assert reference_for("topv2:a") == reference_for("topv2:a")
    assert reference_for("topv2:a") != reference_for("topv2:b")
    assert reference_for("topv2:a").year == 2026


def test_calendar_topv2_conversion_emits_a_resolved_events_insert_call():
    from data.loaders.calendar_json import FUNCTION_NAME, convert_topv2_rows

    rows = convert_topv2_rows([{
        "utterance": "Remind me to pack my lunch for tomorrow.",
        "semantic_parse": "[IN:CREATE_REMINDER Remind [SL:PERSON_REMINDED me ] to "
                          "[SL:TODO pack my lunch ] [SL:DATE_TIME for tomorrow ] . ]",
    }])
    assert len(rows) == 1
    call, = json.loads(rows[0]["answer"])
    assert call["name"] == FUNCTION_NAME
    assert call["arguments"]["summary"] == "pack my lunch"
    start = call["arguments"]["start"]["dateTime"]
    end = call["arguments"]["end"]["dateTime"]
    assert start.endswith("T09:00:00") and end.endswith("T10:00:00")  # 60-minute default
    # The prompt must be self-contained: it states the reference instant the gold resolves against.
    assert "Current date and time:" in rows[0]["text"]
    assert rows[0]["_reference"] in rows[0]["text"]


def test_calendar_topv2_drops_recurring_and_unresolvable_rows():
    from data.loaders.calendar_json import convert_topv2_rows

    assert convert_topv2_rows([{
        "utterance": "remind me to take my meds at 8am daily",
        "semantic_parse": "[IN:CREATE_REMINDER remind [SL:TODO take my meds ] "
                          "[SL:RECURRING_DATE_TIME [IN:GET_RECURRING_DATE_TIME "
                          "[SL:DATE_TIME at 8 am ] [SL:FREQUENCY daily ] ] ] ]",
    }]) == []
    assert convert_topv2_rows([{
        "utterance": "remind me 15 minutes before the exam",
        "semantic_parse": "[IN:CREATE_REMINDER remind [SL:TODO the exam ] "
                          "[SL:DATE_TIME 15 minutes before ] ]",
    }]) == []


def test_calendar_sgd_conversion_uses_the_final_complete_addevent_state():
    from data.loaders.calendar_json import convert_sgd_rows

    rows = convert_sgd_rows([{
        "dialogue_id": "1_00001",
        "services": ["Calendar_1"],
        "turns": [
            {"speaker": "USER", "utterance": "book something", "frames": [
                {"service": "Calendar_1", "state": {
                    "active_intent": "AddEvent",
                    "slot_values": {"event_name": ["dental appointment"]}}}]},
            {"speaker": "USER", "utterance": "confirm", "frames": [
                {"service": "Calendar_1", "state": {
                    "active_intent": "AddEvent",
                    "slot_values": {
                        "event_name": ["dental appointment"],
                        "event_date": ["March 10th"],
                        "event_time": ["12:30 pm"],
                        "event_location": ["689 East Remington Drive"]}}}]},
        ],
    }])
    assert len(rows) == 1
    call, = json.loads(rows[0]["answer"])
    assert call["arguments"]["summary"] == "dental appointment"
    assert call["arguments"]["location"] == "689 East Remington Drive"
    assert call["arguments"]["start"]["dateTime"].endswith("-03-10T12:30:00")


# --- Format-bound training support (the gap that killed both function_call runs) -------------

def test_function_call_training_prompt_is_byte_identical_to_eval():
    """`_training_turn` raised `completion-only SFT does not support task_type='function_call'`
    for every format-bound task, so the pipeline could GRADE function_call/diff but never train
    them. That is why xlam_bfcl and coedit had no run log, and it killed
    slm-xlam-bfcl-cse-38454799 and slm-calendar-json-cse-38455147 after they had already loaded
    data and measured a baseline."""
    from data.eval_set import build_eval_set
    from eval.scorers import function_call as fc
    from training.lora_trainer import _training_turn

    row = {
        "text": "what's the weather in Paris?",
        "answer": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
        "tools": [{"name": "get_weather", "parameters": {"city": "string"}}],
        "label": "function_call",
    }
    user, target, _marker = _training_turn(row, "function_call", [], "")
    assert user == fc.build_prompts(build_eval_set([row], "function_call", target=1))[0]
    assert target == row["answer"]


def test_diff_training_prompt_is_byte_identical_to_eval():
    from data.eval_set import build_eval_set
    from eval.scorers import diff as diff_scorer
    from training.lora_trainer import _training_turn

    row = {"text": "Fix grammar", "src": "He go.\n", "tgt": "He goes.\n",
           "answer": "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-He go.\n+He goes.\n", "label": "diff"}
    user, target, _marker = _training_turn(row, "diff", [], "")
    assert user == diff_scorer.build_prompts(build_eval_set([row], "diff", target=1))[0]
    assert target == row["answer"]


def test_format_bound_training_refuses_an_empty_answer():
    """An empty target would train the model to emit nothing and silently tank format_valid."""
    import pytest

    from training.lora_trainer import _training_turn

    for task_type in ("function_call", "diff"):
        with pytest.raises(ValueError, match="empty 'answer'"):
            _training_turn({"text": "t", "answer": "  "}, task_type, [], "")


def test_unknown_task_type_still_raises():
    import pytest

    from training.lora_trainer import _training_turn

    with pytest.raises(ValueError, match="does not support task_type"):
        _training_turn({"text": "t"}, "not_a_task_type", [], "")
