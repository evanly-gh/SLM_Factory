"""The task's label vocabulary is pinned once and never extended (B259).

RouterBench is a two-class task. Its runs trained on rows labelled `cloud`, `on_device`, `router`
and `remote` because an LLM was asked to map a foreign dataset onto our schema, unmapped values were
passed through verbatim, and the source-level guard accepted a source on ANY label overlap. The
out-of-vocabulary set grew by one hallucinated class per acquire round.

These tests pin the three properties that make that impossible: the space comes from the frozen eval
set, a source carrying any foreign label is rejected whole, and an LLM label_map can only ever target
a label that already exists.
"""
from unittest.mock import MagicMock

from data.label_space import (
    describe_rejected_labels,
    label_definitions_for,
    label_space_from_eval_set,
    partition_rows_by_label,
    sanitize_label_map,
)

ROUTER = {"local", "route"}
# Exactly what a RouterBench run accumulated, with the real counts from dataset_v10.
HALLUCINATED = {"cloud": 688, "on_device": 269, "router": 110, "remote": 99}


def _eval_set(labels, task_type="classification"):
    es = MagicMock()
    es.all = [{"text": f"row {i}", "label": lab} for i, lab in enumerate(labels)]
    es.task_type = task_type
    return es


# --------------------------------------------------------------------------
# The space comes from the frozen eval set
# --------------------------------------------------------------------------

def test_space_is_read_from_the_eval_set():
    assert label_space_from_eval_set(
        _eval_set(["local", "route", "route"]), "classification"
    ) == ROUTER


def test_non_classification_tasks_have_no_vocabulary_to_police():
    """`label` is a constant tag for these, not a class."""
    for task_type in ("NER", "generation", "function_call", "diff", "math_reasoning"):
        assert label_space_from_eval_set(
            _eval_set(["function_call"], task_type), task_type
        ) is None


def test_empty_eval_set_yields_no_space_rather_than_an_empty_one():
    """An empty set must read as 'unknown', never as 'nothing is allowed'."""
    assert label_space_from_eval_set(_eval_set([]), "classification") is None


# --------------------------------------------------------------------------
# Per-row filtering
# --------------------------------------------------------------------------

def test_out_of_vocabulary_rows_are_dropped_not_added():
    rows = (
        [{"text": "a", "label": "local"}, {"text": "b", "label": "route"}]
        + [{"text": f"h{i}", "label": lab}
           for lab, n in HALLUCINATED.items() for i in range(n)]
    )
    kept, rejected = partition_rows_by_label(rows, ROUTER)
    assert len(kept) == 2
    assert {r["label"] for r in kept} == ROUTER
    assert rejected == HALLUCINATED


def test_partition_is_a_no_op_without_a_pinned_space():
    rows = [{"text": "a", "label": "whatever"}]
    kept, rejected = partition_rows_by_label(rows, None)
    assert kept == rows and rejected == {}


def test_rows_without_a_label_are_kept():
    """NER-style rows carry no `label`; they must not be filtered by a classification guard."""
    rows = [{"text": "a", "entities": []}]
    kept, rejected = partition_rows_by_label(rows, ROUTER)
    assert kept == rows and rejected == {}


def test_rejection_summary_orders_by_count():
    text = describe_rejected_labels(HALLUCINATED)
    assert text.startswith("'cloud'x688")
    assert "'on_device'x269" in text


def test_rejection_summary_truncates():
    many = {f"lab{i}": 1 for i in range(20)}
    assert "and 15 more" in describe_rejected_labels(many, limit=5)


# --------------------------------------------------------------------------
# The LLM cannot introduce a label
# --------------------------------------------------------------------------

def test_label_map_targeting_a_hallucinated_class_is_stripped():
    """The exact mapping shape that poisoned RouterBench: complexity tiers → invented classes."""
    clean, dropped = sanitize_label_map(
        {"LOW": "local", "MEDIUM": "cloud", "HIGH": "route", "EXTRA": "on_device"},
        ROUTER,
    )
    assert clean == {"LOW": "local", "HIGH": "route"}
    assert dropped == ["cloud", "on_device"]


def test_label_map_is_untouched_when_every_target_is_real():
    clean, dropped = sanitize_label_map({"0": "local", "1": "route"}, ROUTER)
    assert clean == {"0": "local", "1": "route"} and dropped == []


def test_label_map_passes_through_when_no_space_is_pinned():
    clean, dropped = sanitize_label_map({"a": "anything"}, None)
    assert clean == {"a": "anything"} and dropped == []


def test_empty_label_map_is_handled():
    assert sanitize_label_map(None, ROUTER) == ({}, [])
    assert sanitize_label_map({}, ROUTER) == ({}, [])


# --------------------------------------------------------------------------
# Label definitions (the over-rejection fix)
# --------------------------------------------------------------------------

def test_routerbench_definitions_redirect_the_teacher_away_from_the_label_wording():
    definitions = label_definitions_for("routerbench")
    assert set(definitions) == ROUTER
    # The whole point: tell the teacher this is about DIFFICULTY, not about the word "local".
    assert "difficulty" in definitions["route"].lower()
    assert "NOT" in definitions["local"]


def test_tasks_without_definitions_return_empty():
    """CLINC150's labels really are descriptions of the utterance, so none are needed."""
    assert label_definitions_for("clinc150") == {}
    assert label_definitions_for(None) == {}


# --------------------------------------------------------------------------
# Mining enforcement: reject the whole source on ANY foreign label
# --------------------------------------------------------------------------

def _mine(existing_rows, plan_labels, mined_train, label_space=None):
    """Run mining against one fake local source and return (rows, log lines)."""
    from unittest.mock import patch

    from data.loaders import web_acquire

    logs: list[str] = []

    def local(_plan, _task_type, _max_train, _max_test, log, meta):
        meta.update({"source": "fixture", "source_records": [
            {"kind": "hf", "id": "fixture/src", "split": "train", "role": "curriculum"}]})
        return mined_train, [{"text": "held out", "label": mined_train[0]["label"]}]

    with (
        patch("data.loaders.web_acquire.load_local_dataset", side_effect=local),
        patch("data.loaders.web_acquire.load_benchmark_dataset", return_value=None),
        patch("data.loaders.web_acquire.discover_and_load_hf_dataset", return_value=None),
    ):
        rows, _report = web_acquire.mine_additional_real_rows(
            task_plan={
                "task_type": "classification",
                "task_name": "routing",
                "benchmark": "unknown",
                "labels": plan_labels,
            },
            description="route or answer locally",
            task_type="classification",
            existing_rows=existing_rows,
            eval_rows=[],
            eval_source_ban=[],
            requested_rows=5,
            max_paid_rounds=0,
            query_variant=0,
            plan_identity="test-plan",
            label_space=label_space,
            log=logs.append,
        )
    return rows, logs


def test_pinned_space_rejects_a_source_whose_labels_are_only_partly_valid():
    """The exact B259 shape: a source with SOME correct labels used to be admitted wholesale,
    because the guard only required a non-empty overlap."""
    rows, logs = _mine(
        existing_rows=[{"text": "seed", "label": "local"}],
        plan_labels=["local", "route"],
        mined_train=[
            {"text": "novel one", "label": "local"},
            {"text": "novel two", "label": "cloud"},      # hallucinated
            {"text": "novel three", "label": "on_device"},  # hallucinated
        ],
        label_space=ROUTER,
    )
    assert rows == []
    joined = " ".join(logs)
    assert "REJECTED" in joined and "pinned" in joined
    assert "'cloud'" in joined and "'on_device'" in joined


def test_pinned_space_admits_a_source_whose_labels_are_all_valid():
    rows, _logs = _mine(
        existing_rows=[{"text": "seed", "label": "local"}],
        plan_labels=["local", "route"],
        mined_train=[
            {"text": "novel one", "label": "local"},
            {"text": "novel two", "label": "route"},
        ],
        label_space=ROUTER,
    )
    assert {r["text"] for r in rows} == {"novel one", "novel two"}


def test_inferred_vocabulary_does_not_reject_a_class_the_pool_has_not_seen_yet():
    """A space derived only from the rows the run happens to hold is INCOMPLETE.

    Applying the strict subset rule there would reject good sources for classes the pool simply
    had not encountered — so the inferred case keeps the original any-overlap behaviour.
    """
    rows, _logs = _mine(
        existing_rows=[{"text": "seed", "label": "a"}],  # pool has only 'a'
        plan_labels=[],                                   # nothing authoritative
        mined_train=[
            {"text": "novel one", "label": "a"},
            {"text": "novel two", "label": "b"},          # unseen, but legitimate
        ],
    )
    assert {r["text"] for r in rows} == {"novel one", "novel two"}


def test_inferred_vocabulary_still_rejects_a_wholly_disjoint_source():
    """B222's original case must keep working: raw integer class ids from a non-ClassLabel column."""
    rows, logs = _mine(
        existing_rows=[{"text": "seed", "label": "banking_query"}],
        plan_labels=[],
        mined_train=[
            {"text": "novel one", "label": "0"},
            {"text": "novel two", "label": "1"},
        ],
    )
    assert rows == []
    assert "REJECTED" in " ".join(logs)


def test_mining_rejects_a_source_with_partly_valid_labels():
    """Regression for the exact bug: a source whose labels are PARTLY right used to be accepted
    wholesale because the guard only checked for non-empty overlap."""
    from data.loaders import web_acquire

    logs: list[str] = []
    mined, report = web_acquire.mine_additional_real_rows(
        task_plan={"task_type": "classification", "task_name": "routing", "benchmark": None},
        description="route or answer locally",
        task_type="classification",
        existing_rows=[{"text": "seed", "label": "local"}],
        eval_rows=[],
        eval_source_ban=[],
        requested_rows=0,  # no acquisition attempted; we are testing the guard only
        max_paid_rounds=0,
        query_variant=0,
        plan_identity="test-plan",
        label_space=ROUTER,
        log=logs.append,
    )
    assert mined == []
    assert report["status"] in ("not_requested", "no_novelty")
