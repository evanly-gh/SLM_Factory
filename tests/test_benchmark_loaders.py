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
    "data.loaders.medqa": ("HF_ID",),
    "data.loaders.coedit": ("HF_ID",),
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

def test_coedit_splits_instruction_and_builds_diff():
    from data.loaders.coedit import convert_coedit_rows

    rows = convert_coedit_rows([
        {"src": "Fix grammar: He go to school.", "tgt": "He goes to school."},
    ])
    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "Fix grammar"
    assert row["src"].startswith("He go to school")
    assert row["tgt"].startswith("He goes to school")
    assert "@@" in row["answer"] and "+++ b/file.txt" in row["answer"]
    assert row["label"] == "diff"


def test_coedit_drops_noop_edits():
    from data.loaders.coedit import convert_coedit_rows

    rows = convert_coedit_rows([{"src": "Fix grammar: same text.", "tgt": "same text."}])
    assert rows == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_coedit_gold_diff_applies_and_reproduces_target():
    """The loader's difflib gold diff must git-apply to src and reproduce tgt (content=1.0)."""
    from data.eval_set import EvalSet
    from data.loaders.coedit import convert_coedit_rows
    from eval.scorers.diff import extract_predictions, score

    rows = convert_coedit_rows([
        {"src": "Improve clarity: The thing is very big and large.",
         "tgt": "The object is enormous."},
    ])
    es = EvalSet(all=rows, task_type="diff")
    preds = extract_predictions([rows[0]["answer"]], es)
    result = score(es, preds)
    assert result["per_class"]["format_valid"] == 1.0
    assert result["f1"] == 1.0


# --- RouterBench (classification) ----------------------------------------------

def test_routerbench_derives_local_vs_route_label():
    from data.loaders.routerbench import convert_routerbench_rows

    rows = convert_routerbench_rows([
        {"prompt": "2+2?", "small_model_correct": 1},
        {"prompt": "prove Fermat", "small_model_correct": 0},
        {"prompt": "no signal here"},               # unresolvable → dropped
    ])
    assert rows == [
        {"text": "2+2?", "label": "local"},
        {"text": "prove Fermat", "label": "route"},
    ]


def test_routerbench_threshold_on_float_score():
    from data.loaders.routerbench import convert_routerbench_rows

    rows = convert_routerbench_rows(
        [{"prompt": "q", "small_model_correct": 0.7}], threshold=0.5)
    assert rows == [{"text": "q", "label": "local"}]


# --- MedQA (classification) ----------------------------------------------------

def test_medqa_renders_options_and_picks_letter_from_idx():
    from data.loaders.medqa import convert_medqa_rows

    rows = convert_medqa_rows([{
        "question": "Which vitamin is fat-soluble?",
        "options": {"A": "Vitamin C", "B": "Vitamin D", "C": "Vitamin B12", "D": "Folate"},
        "answer_idx": "B",
    }])
    assert len(rows) == 1
    assert rows[0]["label"] == "B"
    assert "A. Vitamin C" in rows[0]["text"]
    assert "Which vitamin is fat-soluble?" in rows[0]["text"]


def test_medqa_resolves_letter_from_answer_text_when_idx_missing():
    from data.loaders.medqa import convert_medqa_rows

    rows = convert_medqa_rows([{
        "question": "Q?",
        "options": ["red", "green", "blue", "yellow"],
        "answer": "blue",
    }])
    assert rows[0]["label"] == "C"
