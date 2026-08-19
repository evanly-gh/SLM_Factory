"""The task's label vocabulary is pinned once and never extended (B259).

RouterBench is a two-class task: `local` and `route`. Its runs trained on rows labelled `cloud`,
`on_device`, `router` and `remote` as well. They arrived because acquisition asks an LLM to map a
mined dataset's columns onto our schema, the returned `label_map` was applied with
`lmap.get(str(lab), lab)` so unmapped values passed through VERBATIM, and the mapping was re-requested
per acquire round and is non-deterministic — so each round could mint a new hallucinated class. The
out-of-vocabulary set grew monotonically: `cloud` for iterations 1-9, `+on_device` at 10, `+router`
at 20, `+remote` at 21.

Quality control then deleted ~1,166 rows per rebuild for the rest of the run, and the mined rows that
survived — 34% of the final curriculum — carried invented `local`/`route` labels at a 50/50 rate
against the task's true 30/70 base rate.

THE RULE, which these tests pin: the space comes from the FROZEN EVAL SET, a row outside it is
dropped rather than added to the vocabulary, and an LLM may only map ONTO an existing label — never
introduce one.

Whether a task HAS a closed space is `TaskSpec.closed_label_space`, stated per task. It used to be
inferred from the `classification` channel, which is why span and structured-output tasks — whose
`label` field is a constant tag, not a class — were policed by a guard that had nothing to police.
"""
from types import SimpleNamespace

import pytest

from data.label_space import (
    describe_rejected_labels,
    label_definitions_for,
    label_space_from_eval_set,
    partition_rows_by_label,
    sanitize_label_map,
)

ROUTER = {"local", "route"}
# Exactly what the RouterBench run accumulated, with the real counts from its dataset_v10.
HALLUCINATED = {"cloud": 688, "on_device": 269, "router": 110, "remote": 99}


def _eval_set(labels, task="routerbench"):
    return SimpleNamespace(
        all=[{"text": f"row {i}", "label": label} for i, label in enumerate(labels)],
        task=task,
    )


# --------------------------------------------------------------------------
# The space comes from the frozen eval set
# --------------------------------------------------------------------------


def test_the_space_is_read_from_the_eval_set():
    """The eval set is what the score is computed against, so it is the only defensible authority on
    what counts as a class. A space derived from the training pool would grow as the pool did."""
    assert label_space_from_eval_set(_eval_set(["local", "route", "route"]), "routerbench") == ROUTER


@pytest.mark.parametrize(
    "task", ["ner_bc5cdr", "dialogsum", "xlam_bfcl", "calendar_json", "gsm8k"],
)
def test_a_task_with_no_closed_label_space_has_no_vocabulary_to_police(task):
    """For these, `label` is a constant tag rather than a class to predict. Policing it would drop
    real rows for having the "wrong" value of a field nothing scores."""
    assert label_space_from_eval_set(_eval_set(["anything"], task), task) is None


def test_an_empty_eval_set_yields_no_space_rather_than_an_empty_one():
    """An empty set must read as "unknown", never as "nothing is allowed" — the latter would reject
    the entire curriculum."""
    assert label_space_from_eval_set(_eval_set([]), "routerbench") is None


def test_the_space_is_exactly_the_classes_present_and_no_more():
    """Not the union of the eval set and anything else. A class absent from the eval set cannot be
    scored, so training rows carrying it are unusable by construction."""
    assert label_space_from_eval_set(
        _eval_set(["transfer", "balance", "oos", "transfer"], "clinc150"), "clinc150",
    ) == {"transfer", "balance", "oos"}


# --------------------------------------------------------------------------
# Per-row filtering: a foreign class is dropped, not adopted
# --------------------------------------------------------------------------


def test_out_of_vocabulary_rows_are_dropped_not_added():
    rows = (
        [{"text": "a", "label": "local"}, {"text": "b", "label": "route"}]
        + [{"text": f"h{i}", "label": label}
           for label, count in HALLUCINATED.items() for i in range(count)]
    )
    kept, rejected = partition_rows_by_label(rows, ROUTER)

    assert len(kept) == 2
    assert {row["label"] for row in kept} == ROUTER
    assert rejected == HALLUCINATED


def test_filtering_is_per_row_not_per_source():
    """The original guard was per-SOURCE and passed the whole thing on any overlap, which is exactly
    how the hallucinated classes got in alongside legitimate rows."""
    kept, rejected = partition_rows_by_label(
        [{"text": "good", "label": "local"}, {"text": "bad", "label": "cloud"}], ROUTER,
    )
    assert [row["text"] for row in kept] == ["good"]
    assert rejected == {"cloud": 1}


def test_partitioning_is_a_no_op_without_a_pinned_space():
    rows = [{"text": "a", "label": "whatever"}]
    assert partition_rows_by_label(rows, None) == (rows, {})


def test_rows_without_a_label_are_kept():
    """NER-shaped rows carry no `label` at all; they must not be deleted by a classification guard."""
    rows = [{"text": "a", "entities": []}]
    assert partition_rows_by_label(rows, ROUTER) == (rows, {})


def test_the_rejection_summary_leads_with_the_worst_offender():
    text = describe_rejected_labels(HALLUCINATED)
    assert text.startswith("'cloud'x688")
    assert "'on_device'x269" in text


def test_the_rejection_summary_is_bounded():
    assert "and 15 more" in describe_rejected_labels({f"lab{i}": 1 for i in range(20)}, limit=5)


def test_nothing_rejected_reports_nothing():
    assert describe_rejected_labels({}) == ""


# --------------------------------------------------------------------------
# The LLM cannot introduce a label
# --------------------------------------------------------------------------


def test_a_label_map_targeting_a_hallucinated_class_is_stripped():
    """The exact mapping shape that poisoned RouterBench: complexity tiers onto invented classes.

    Note what is dropped — the ENTRY, not just the row. Honouring the entry is how `cloud` entered a
    two-class task and then reappeared every acquire round.
    """
    clean, dropped = sanitize_label_map(
        {"LOW": "local", "MEDIUM": "cloud", "HIGH": "route", "EXTRA": "on_device"}, ROUTER,
    )
    assert clean == {"LOW": "local", "HIGH": "route"}
    assert dropped == ["cloud", "on_device"]


def test_a_label_map_is_untouched_when_every_target_is_real():
    assert sanitize_label_map({"0": "local", "1": "route"}, ROUTER) == (
        {"0": "local", "1": "route"}, [],
    )


def test_a_label_map_passes_through_when_no_space_is_pinned():
    """Nothing to check against. The strict rule is only safe against an AUTHORITATIVE vocabulary."""
    assert sanitize_label_map({"a": "anything"}, None) == ({"a": "anything"}, [])


def test_an_absent_label_map_is_handled():
    assert sanitize_label_map(None, ROUTER) == ({}, [])
    assert sanitize_label_map({}, ROUTER) == ({}, [])


# --------------------------------------------------------------------------
# Quality control enforces the same closure on the finished curriculum
# --------------------------------------------------------------------------


def test_quality_control_removes_a_row_outside_the_pinned_vocabulary():
    """The backstop. Even if a foreign label reaches the curriculum, the task's declared
    `label_space` step removes it before training and says which labels it removed."""
    from data.curriculum import apply_quality_controls

    rows = [
        {"text": f"legitimate row number {i}", "label": "local" if i % 2 else "route"}
        for i in range(10)
    ] + [{"text": "mined row with an invented class", "label": "cloud"}]
    logs: list[str] = []
    kept = apply_quality_controls(
        rows, "routerbench", allowed_labels=ROUTER, log=logs.append,
    )

    assert "cloud" not in {row["label"] for row in kept}
    joined = " ".join(logs)
    assert "label-space" in joined and "'cloud'x1" in joined


def test_the_closed_space_reaches_quality_control_from_the_frozen_eval_set():
    """`curate` passes `spec.qc_context_labels(eval_set)`, so the vocabulary QC enforces is the same
    one the model is scored against rather than a second derivation of it."""
    from data.eval_set import EvalSet
    from tasks import get_task

    eval_set = EvalSet(
        all=[{"text": "a", "label": "local"}, {"text": "b", "label": "route"}],
        task="routerbench",
    )
    assert get_task("routerbench").qc_context_labels(eval_set) == ROUTER


# --------------------------------------------------------------------------
# What the labels MEAN (B267) — the over-rejection this caused
# --------------------------------------------------------------------------


def test_routerbench_definitions_point_the_teacher_at_difficulty_not_at_the_word():
    """`local` means "a small on-device model answers this correctly", but the teacher read it as
    "a local-information query" and rejected a grade-school math problem with *"the utterance is a
    math problem, not a local query"* — 70% of generated rows discarded for the wrong reason. The
    teacher was not being strict; it was answering a different question than the task asks."""
    definitions = label_definitions_for("routerbench")
    assert set(definitions) == ROUTER
    assert "difficulty" in definitions["route"].lower()
    assert "NOT" in definitions["local"]


def test_definitions_live_on_the_task_spec_rather_than_a_side_table():
    """`_LABEL_DEFINITIONS` and `_BENCHMARK_ALIASES` were two of the five hand-maintained side
    registries the task specs replaced, and each had been added after a bug caused by their being
    out of sync."""
    from tasks import get_task

    assert label_definitions_for("routerbench") == dict(
        get_task("routerbench").label_definitions
    )
    import data.label_space as label_space_module

    for gone in ("_LABEL_DEFINITIONS", "_BENCHMARK_ALIASES"):
        assert not hasattr(label_space_module, gone), f"{gone} is back"


def test_a_task_whose_labels_describe_themselves_needs_no_definitions():
    """CLINC150's labels ARE plain descriptions of the utterance's intent, so naming the label
    already tells the teacher what the class means; a gloss for 151 classes would be prompt noise."""
    assert label_definitions_for("clinc150") == {}


def test_an_unknown_or_missing_task_yields_no_definitions_rather_than_raising():
    """Reached from prompt-building paths that must not be able to fail the run."""
    assert label_definitions_for(None) == {}
    assert label_definitions_for("not_a_task") == {}
