"""Quality control runs on every task, and is EFFECTIVE on that task's own row shape (B299).

WHY THIS FILE EXISTS
    Quality control used to be one `if task_type == ...` chain in `data/curriculum.py` ending in
    `else: return dataset`. Measured on 2026-08-18, that meant FOUR OF THE EIGHT benchmark tasks
    received no quality control at all, silently:

      * `function_call` (xlam_bfcl, calendar_json) fell into the `else` and was returned untouched;
      * `math_reasoning` (gsm8k) and `generation` (dialogsum) entered their branch but filtered
        length and duplicates on a `"prompt"` key their rows do not carry — so a row of 100,000
        characters survived, and nothing was logged.

    Only `classification` and `NER` were actually filtered, and nothing distinguished "this task
    chose not to deduplicate" from "this task fell through a branch nobody updated".

    So the regression is not "does a step exist". It is: for EACH of the eight tasks, do the steps
    it declared actually remove something when given a row of ITS OWN shape that they are supposed
    to remove — and do they say so. A step that runs and quietly changes nothing is the bug.

    The 100,000-character row is the literal artifact from the measurement, so it is the probe here.
"""
from __future__ import annotations

import json

import pytest

from data.curriculum import apply_quality_controls
from data.quality_controls import QCContext, TRUSTED_LENGTH_PROVENANCE
from tasks import TASKS, get_task

HUGE = "x" * 100_000

CALENDAR_TOOL = {"name": "calendar.events.insert",
                 "parameters": {"properties": {"summary": {}}, "required": ["summary"]}}
WEATHER_TOOL = {"type": "function",
                "function": {"name": "get_weather",
                             "parameters": {"properties": {"city": {}}, "required": ["city"]}}}


def _xlam_row(text, answer=None):
    return {"text": text, "tools": [WEATHER_TOOL],
            "answer": answer if answer is not None
            else json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}])}


def _calendar_row(text, answer=None):
    return {"text": text, "tools": [CALENDAR_TOOL],
            "answer": answer if answer is not None
            else json.dumps([{"name": "calendar.events.insert",
                              "arguments": {"summary": "Dentist"}}])}


def _ner_row(text, entities=None):
    return {"text": text,
            "entities": entities if entities is not None
            else [{"text": "Aspirin", "type": "Chemical"}]}


# A generator of ordinary, healthy rows in each task's own schema. Every helper takes an index so a
# batch is genuinely distinct — otherwise `dedup_surface` would remove the control rows and the
# probe below could not tell which step fired.
ROW_BUILDERS = {
    "gsm8k": lambda i: {"text": f"Ann has {i} pears and buys {i + 2} more. How many?",
                        "answer": f"Add them.\n#### {2 * i + 2}"},
    "dialogsum": lambda i: {"text": f"#Person1#: topic {i}? #Person2#: yes, about item {i}.",
                            "answer": f"Two people discuss topic {i}."},
    "xlam_bfcl": lambda i: _xlam_row(f"what is the weather in city number {i}?"),
    "calendar_json": lambda i: _calendar_row(f"add a dentist visit on the {i}th of March"),
    "ner_bc5cdr": lambda i: _ner_row(f"Compound{i} induced condition{i} in the cohort.",
                                     [{"text": f"Compound{i}", "type": "Chemical"}]),
    "routerbench": lambda i: {"text": f"question number {i} about arithmetic",
                              "label": "local" if i % 2 else "route"},
    "proactive_listening": lambda i: {"text": f"speaker A pauses after clause {i} here",
                                      "label": "wait" if i % 2 else "interrupt"},
    "clinc150": lambda i: {"text": f"utterance number {i} about money",
                           "label": "transfer" if i % 2 else "balance"},
}

# The closed vocabulary each task's frozen eval set would pin, and None where there is no class.
ALLOWED_LABELS = {
    "gsm8k": None,
    "dialogsum": None,
    "xlam_bfcl": None,
    "calendar_json": None,
    "ner_bc5cdr": None,
    "routerbench": {"local", "route"},
    "proactive_listening": {"interrupt", "wait"},
    "clinc150": {"transfer", "balance"},
}

TASK_IDS = sorted(ROW_BUILDERS)


def _healthy(task: str, n: int = 24) -> list[dict]:
    build = ROW_BUILDERS[task]
    return [{**build(i), "_provenance": "train_anchor"} for i in range(n)]


def _run(task: str, rows: list[dict]) -> tuple[list[dict], list[str]]:
    logs: list[str] = []
    kept = apply_quality_controls(
        rows, task, allowed_labels=ALLOWED_LABELS[task], log=logs.append,
    )
    return kept, logs


def test_every_registered_task_has_a_row_builder():
    assert set(ROW_BUILDERS) == set(TASKS)


# --------------------------------------------------------------------------
# The declared steps actually run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASK_IDS)
def test_a_task_declares_its_steps_explicitly(task):
    """`()` is a legal and explicit choice; falling through a branch is not possible because there
    is no branch. Every task in this suite does declare steps, and the tuple is what runs."""
    spec = get_task(task)
    assert isinstance(spec.quality_controls, tuple)
    assert spec.quality_controls, f"{task} declares no quality control at all"
    for step in spec.quality_controls:
        assert callable(step)


@pytest.mark.parametrize("task", TASK_IDS)
def test_healthy_rows_survive_quality_control(task):
    """The control. If QC removes ordinary rows, every "it removed the bad row" assertion below is
    meaningless."""
    rows = _healthy(task)
    kept, _logs = _run(task, rows)
    assert len(kept) == len(rows), f"{task} lost healthy rows: {len(rows)} -> {len(kept)}"


@pytest.mark.parametrize("task", TASK_IDS)
def test_quality_control_delegates_to_the_task_declared_steps(task, monkeypatch):
    """`data.curriculum.apply_quality_controls` must read `TaskSpec.quality_controls`, not decide
    for itself. A probe step appended to the spec has to run."""
    import dataclasses

    import tasks

    ran: list[int] = []

    def probe(rows, ctx):
        ran.append(len(rows))
        return rows

    spec = get_task(task)
    patched = dataclasses.replace(spec, quality_controls=(*spec.quality_controls, probe))
    monkeypatch.setitem(tasks.TASKS, task, patched)

    _run(task, _healthy(task))
    assert ran, f"{task}'s declared steps were not the ones that ran"


# --------------------------------------------------------------------------
# The 100,000-character row (the measured B299 artifact)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASK_IDS)
def test_a_hundred_thousand_character_row_is_removed(task):
    """The literal row that survived. gsm8k and dialogsum filtered on a `"prompt"` key their rows
    have never carried, so both of them kept it; xlam and calendar were never filtered at all.

    All eight tasks now declare `length_outliers(key="text")` — and `key` is stated by the task
    rather than guessed, which is the fix.
    """
    rows = _healthy(task)
    build = ROW_BUILDERS[task]
    monster = {**build(999), "text": HUGE, "_provenance": "mined_real"}
    kept, logs = _run(task, [*rows, monster])

    assert len(kept) == len(rows), f"{task} kept a 100,000-character row"
    assert HUGE not in {row.get("text") for row in kept}
    assert any("length-outlier" in line for line in logs), (
        f"{task} removed the row without saying so — a silent removal is as opaque as none"
    )


@pytest.mark.parametrize("task", TASK_IDS)
def test_the_removal_report_names_the_step_the_count_and_the_reason(task):
    """Quality control is the single biggest consumer of rows in this pipeline — it deleted ~1,500
    of 1,549 synthesized rows in one run — and doing so silently made a curriculum arriving far
    under target look inexplicable (B228)."""
    build = ROW_BUILDERS[task]
    _kept, logs = _run(task, [*_healthy(task),
                              {**build(999), "text": HUGE, "_provenance": "mined_real"}])
    line = next(line for line in logs if "length-outlier" in line)
    assert "removed 1 row(s)" in line
    assert "longer than 3x the median" in line


# --------------------------------------------------------------------------
# A step that cannot measure anything must SAY SO
# --------------------------------------------------------------------------


def test_a_length_step_pointed_at_a_field_the_rows_lack_reports_that_it_skipped():
    """This is B299's exact shape, kept as an executable statement of it.

    A step whose key is absent from every row silently passes the whole dataset. It must announce
    that it could not measure anything, because a silent pass here is indistinguishable from a
    dataset that had no outliers — and that ambiguity is what hid four unfiltered tasks.
    """
    from data.quality_controls import length_outliers

    logs: list[str] = []
    rows = [{"text": "short"}, {"text": HUGE}]
    step = length_outliers(key="prompt")
    kept = step(rows, QCContext(task_name="gsm8k", log=logs.append))

    assert kept == rows, "the step cannot filter on a key no row carries"
    joined = " ".join(logs)
    assert "SKIPPED" in joined
    assert "'prompt'" in joined
    assert "gsm8k" in joined
    assert "quality_controls against its row schema" in joined


def test_no_task_declares_a_length_step_keyed_on_a_field_its_rows_lack():
    """The forward-looking version: every task's `length_outliers` key must be a field that task's
    own rows actually carry, so the SKIPPED path above is never reached in production."""
    for task in TASK_IDS:
        _kept, logs = _run(task, _healthy(task))
        assert not any("SKIPPED" in line for line in logs), f"{task} has a no-op length step"


# --------------------------------------------------------------------------
# The per-task steps, each on its own shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASK_IDS)
def test_a_row_missing_the_field_this_task_grades_is_dropped(task):
    """`require_fields` is named explicitly per task rather than inferred, because the inference was
    wrong: the generation branch accepted a row on `("text", "label")` and then filtered it on
    `"prompt"`, so every row passed the gate and none was measurable."""
    spec = get_task(task)
    target = [field for field in spec.required_fields if field != "text"][0]
    broken = {key: value for key, value in ROW_BUILDERS[task](500).items() if key != target}
    kept, logs = _run(task, [*_healthy(task), broken])

    assert len(kept) == len(_healthy(task)), f"{task} kept a row with no {target!r}"
    assert any("schema" in line for line in logs)


@pytest.mark.parametrize("task", ["xlam_bfcl", "calendar_json"])
def test_a_format_bound_row_whose_gold_is_not_json_is_dropped(task):
    """A row whose own gold does not parse teaches the model to emit something the scorer will mark
    wrong no matter what it predicts. Cheap to check, and it had never run on either task because
    both reached the `else` branch."""
    builder = _xlam_row if task == "xlam_bfcl" else _calendar_row
    rows = _healthy(task)
    kept, logs = _run(task, [*rows, {**builder("a normal length request", answer="not json")}])

    assert len(kept) == len(rows)
    assert any("json-answer" in line for line in logs)


@pytest.mark.parametrize("task", ["xlam_bfcl", "calendar_json"])
def test_a_gold_answer_that_is_already_a_list_is_accepted(task):
    """The loaders store `answer` as a JSON string, but a synthesized row may carry the parsed list.
    Rejecting it would delete correct rows."""
    from data.quality_controls import valid_json_answer

    step = valid_json_answer()
    rows = [{"answer": [{"name": "get_weather", "arguments": {}}]}, {"answer": {"a": 1}}]
    assert step(rows, QCContext(task_name=task)) == rows


@pytest.mark.parametrize("task", ["routerbench", "proactive_listening", "clinc150"])
def test_a_row_labelled_outside_the_frozen_vocabulary_is_dropped(task):
    """A row whose label cannot appear in the eval set can never be scored against it, so it is pure
    training noise. This is where mined sources carrying raw integer class ids get removed
    (B222/B229) and where a hallucinated class dies (B259)."""
    rows = _healthy(task)
    foreign = {**ROW_BUILDERS[task](500), "label": "cloud"}
    kept, logs = _run(task, [*rows, foreign])

    assert "cloud" not in {row["label"] for row in kept}
    assert len(kept) <= len(rows)
    joined = " ".join(logs)
    assert "label-space" in joined and "'cloud'" in joined


def test_the_label_space_step_is_a_no_op_without_a_pinned_vocabulary():
    """An unknown vocabulary must read as "unknown", never as "nothing is allowed"."""
    from data.quality_controls import label_space

    rows = [{"label": "anything"}]
    assert label_space()(rows, QCContext(task_name="routerbench", allowed_labels=None)) == rows


@pytest.mark.parametrize("task", ["routerbench", "clinc150"])
def test_a_class_far_over_the_balance_cap_is_trimmed(task):
    """Declared by these two and deliberately NOT by proactive_listening."""
    rows = [
        *[{**ROW_BUILDERS[task](i), "label": "local" if task == "routerbench" else "transfer"}
          for i in range(40)],
        *[{**ROW_BUILDERS[task](100 + i),
           "label": "route" if task == "routerbench" else "balance"}
          for i in range(2)],
    ]
    kept, logs = _run(task, rows)
    counts = {}
    for row in kept:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    assert max(counts.values()) <= 3 * min(counts.values())
    assert any("label-balance" in line for line in logs)


def test_proactive_listening_is_deliberately_not_label_balanced():
    """`wait` is the overwhelming majority BY CONSTRUCTION — most pauses are not interruption
    points — so a 3:1 cap would train the model on a base rate the eval set does not have. The
    minority-class F1 is what keeps a majority-always model from scoring well.

    Asserted because "this task chose not to balance" and "this task fell through a branch" used to
    look identical, and that ambiguity is the whole of B299.
    """
    rows = [
        *[{"text": f"long pause number {i} in the transcript", "label": "wait"}
          for i in range(40)],
        *[{"text": f"hesitation {i}", "label": "interrupt"} for i in range(2)],
    ]
    kept, _logs = _run("proactive_listening", rows)
    waits = sum(1 for row in kept if row["label"] == "wait")
    assert waits > 3 * 2, "proactive_listening was label-balanced; its base rate is the signal"


@pytest.mark.parametrize("task", ["gsm8k", "xlam_bfcl", "routerbench",
                                  "proactive_listening", "clinc150"])
def test_near_duplicate_rows_are_removed_where_the_task_asked_for_it(task):
    rows = _healthy(task)
    twin = dict(rows[0])
    kept, logs = _run(task, [*rows, twin])
    assert len(kept) == len(rows)
    assert any("surface-dedup" in line for line in logs)


@pytest.mark.parametrize("task", ["dialogsum", "calendar_json"])
def test_two_tasks_deliberately_keep_near_duplicates(task):
    """Both for stated reasons about their own corpora, not about a channel.

    Chat transcripts share a great deal of surface form (greetings, scheduling small talk) and
    calendar utterances are short and highly templated ("remind me to X at Y") — so a Jaccard filter
    removes legitimately distinct rows that differ only in the entity, which is the part the model
    has to learn to extract.
    """
    rows = _healthy(task)
    twin = dict(rows[0])
    kept, _logs = _run(task, [*rows, twin])
    assert len(kept) == len(rows) + 1, f"{task} deduplicated when it declared it would not"


def test_a_repeated_entity_surface_form_is_capped_for_span_extraction():
    """Abstracts repeat the same drug names constantly; without a cap the curriculum teaches a
    handful of surface forms rather than the span task."""
    rows = [
        _ner_row(f"Aspirin was given in cohort {i} of the study.",
                 [{"text": "Aspirin", "type": "Chemical"}])
        for i in range(10)
    ]
    kept, logs = _run("ner_bc5cdr", rows)
    assert len(kept) == 3, "the entity-diversity cap did not fire"
    assert any("entity-diversity" in line for line in logs)


def test_entity_diversity_leaves_a_diverse_batch_alone():
    from data.quality_controls import entity_diversity

    rows = [_ner_row(f"Compound{i} was studied.", [{"text": f"Compound{i}", "type": "Chemical"}])
            for i in range(10)]
    assert entity_diversity(cap=3)(rows, QCContext(task_name="ner_bc5cdr")) == rows


# --------------------------------------------------------------------------
# The trusted median (B260)
# --------------------------------------------------------------------------


def test_the_length_bound_is_anchored_on_the_task_own_real_rows():
    """The filter is RELATIVE, so whatever sets the median sets the cutoff. A verbose teacher would
    otherwise raise the bar and license its own outliers (B260).

    Here the trusted rows are short and the untrusted ones are long: the cutoff must come from the
    short ones, so the long generated rows are removed rather than becoming the new normal.
    """
    trusted = [
        {"text": f"short real row {i}", "answer": "#### 1", "_provenance": "train_anchor"}
        for i in range(10)
    ]
    verbose = [
        {"text": f"rambling teacher row {i} " + " ".join(f"filler{i}_{j}" for j in range(200)),
         "answer": "#### 1", "_provenance": "synthetic"}
        for i in range(10)
    ]
    kept, _logs = _run("gsm8k", [*trusted, *verbose])

    assert len(kept) == 10
    assert all(row["_provenance"] == "train_anchor" for row in kept)


def test_only_real_train_rows_are_trusted_to_set_the_median():
    assert TRUSTED_LENGTH_PROVENANCE == frozenset({"train_anchor", "resample"})


def test_the_median_falls_back_to_the_whole_batch_when_nothing_is_trusted():
    """A rebuild can legitimately contain no real rows. Refusing to filter at all would then be a
    silent pass, which is the failure this module exists to prevent."""
    from data.quality_controls import length_outliers

    rows = [{"text": "short", "_provenance": "synthetic"} for _ in range(10)]
    rows.append({"text": HUGE, "_provenance": "synthetic"})
    kept = length_outliers(key="text")(rows, QCContext(task_name="gsm8k"))
    assert len(kept) == 10


# --------------------------------------------------------------------------
# The plumbing
# --------------------------------------------------------------------------


def test_an_empty_dataset_passes_through_untouched():
    """Reached when a loader returns nothing; a crash here would hide the real cause upstream."""
    assert apply_quality_controls([], "gsm8k") == []


def test_the_report_stays_silent_when_a_step_removes_nothing():
    """A per-step line on every clean pass would bury the removals that matter."""
    ctx = QCContext(task_name="gsm8k", log=lambda _m: pytest.fail("reported a no-op removal"))
    ctx.report("length-outlier", 10, 10, "nothing to say")


def test_steps_run_in_the_order_the_task_declared_them():
    """Order matters: `require_fields` before `valid_json_answer` means the JSON check never sees a
    row with no `answer` at all."""
    import dataclasses

    import tasks

    order: list[str] = []

    def make(tag):
        def step(rows, _ctx):
            order.append(tag)
            return rows
        return step

    spec = get_task("gsm8k")
    monkey = dataclasses.replace(spec, quality_controls=(make("a"), make("b"), make("c")))
    original = tasks.TASKS["gsm8k"]
    tasks.TASKS["gsm8k"] = monkey
    try:
        apply_quality_controls([{"text": "t", "answer": "a"}], "gsm8k")
    finally:
        tasks.TASKS["gsm8k"] = original
    assert order == ["a", "b", "c"]


def test_quality_control_for_an_unknown_task_names_the_registry():
    with pytest.raises(ValueError, match="unknown task"):
        apply_quality_controls([{"text": "t"}], "code_generation")


def test_the_curriculum_shim_and_the_module_agree():
    """`data.curriculum.apply_quality_controls` is the shim curate calls; `data.quality_controls`
    holds the implementation. Two entry points that could disagree is how the old chain drifted."""
    from data.quality_controls import apply_quality_controls as direct

    rows = _healthy("clinc150") + [{**ROW_BUILDERS["clinc150"](500), "text": HUGE}]
    via_shim = apply_quality_controls(rows, "clinc150",
                                      allowed_labels=ALLOWED_LABELS["clinc150"])
    via_module = direct(
        rows, get_task("clinc150").quality_controls,
        task_name="clinc150", allowed_labels=ALLOWED_LABELS["clinc150"],
    )
    assert [row.get("text") for row in via_shim] == [row.get("text") for row in via_module]
