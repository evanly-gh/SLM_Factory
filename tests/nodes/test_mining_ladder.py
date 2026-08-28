"""The `mine_new_real` ladder, and knowing when it has nothing left to give.

WHY THIS FILE EXISTS
    Mining is a ladder with three rungs and it must stop on the first one that produces rows:

      1. re-read a dataset this run already sourced but has not exhausted — free, no provider call,
         no LLM column mapping, no schema risk;
      2. web research for a dataset we have never used — paid, and only once every known source is
         exhausted;
      3. retirement, after two consecutive discovery rounds that contribute nothing.

    Rung 1 did not exist before 2026-08-19. `acquire` went straight to paid web discovery, which
    then found mirrors of the very corpus sitting in the local cache and rejected them, while
    ~57,000 unused xLAM rows stayed unreachable (B297).

    The distinction the retirement logic protects is subtle and was previously reported
    identically: "mining added nothing this round" and "mining has nothing left to add, ever" are
    different facts. The first is worth retrying; the second means every later `mine_new_real` plan
    burns a full train+eval cycle to add zero rows.
"""
from __future__ import annotations

import pytest

from agent.data_rebuild import (
    MAX_FAILED_DISCOVERY_ROUNDS,
    mining_available_for_state,
    unexhausted_sources,
)


# --------------------------------------------------------------------------
# Which sources still have rows
# --------------------------------------------------------------------------


def test_a_source_with_rows_left_is_unexhausted():
    sources = unexhausted_sources({
        "source_progress": {"tner/bc5cdr": {"consumed": 500, "total": 4560}},
    })
    assert [entry["source"] for entry in sources] == ["tner/bc5cdr"]
    assert sources[0]["consumed"] == 500


def test_a_source_with_an_unknown_total_counts_as_unexhausted():
    """Most loaders take a head slice of a split whose length we never measured. Assuming exhaustion
    from a missing total would give up on the largest corpora in the suite."""
    sources = unexhausted_sources({
        "source_progress": {"Salesforce/xlam-function-calling-60k": {"consumed": 3250}},
    })
    assert len(sources) == 1


def test_an_exhausted_source_is_excluded():
    assert unexhausted_sources({
        "source_progress": {"a": {"exhausted": True}},
    }) == []


def test_exhaustion_is_per_source():
    sources = unexhausted_sources({
        "source_progress": {
            "knkarthick/dialogsum": {"exhausted": True},
            "Samsung/samsum": {"consumed": 100},
        },
    })
    assert [entry["source"] for entry in sources] == ["Samsung/samsum"]


@pytest.mark.parametrize("progress", [None, {}, "not a mapping", [1, 2]])
def test_a_missing_or_malformed_progress_record_is_no_sources_rather_than_a_crash(progress):
    """Reached on a resumed checkpoint written before the field existed."""
    assert unexhausted_sources({"source_progress": progress}) == []


def test_a_non_mapping_source_record_is_skipped():
    assert unexhausted_sources({"source_progress": {"a": "oops", "b": {"consumed": 1}}}) == [
        {"source": "b", "consumed": 1},
    ]


# --------------------------------------------------------------------------
# The transition to "mining cannot help"
# --------------------------------------------------------------------------


def test_mining_is_available_while_any_source_has_rows_left():
    """A source with rows left is free to re-read, so mining is offered regardless of how many
    discovery rounds have failed — those are a different rung."""
    for failed in range(MAX_FAILED_DISCOVERY_ROUNDS + 3):
        assert mining_available_for_state({
            "source_progress": {"a": {"consumed": 10}},
            "failed_discovery_rounds": failed,
        }) is True


def test_mining_is_available_with_every_source_exhausted_but_discovery_untried():
    """A discovery round that has not been tried might find something; refusing before trying is
    how B297's unreachable rows stayed unreachable."""
    assert mining_available_for_state({
        "source_progress": {"a": {"exhausted": True}},
        "failed_discovery_rounds": 0,
    }) is True


def test_mining_is_retired_only_when_both_rungs_are_spent():
    assert mining_available_for_state({
        "source_progress": {"a": {"exhausted": True}},
        "failed_discovery_rounds": MAX_FAILED_DISCOVERY_ROUNDS,
    }) is False


def test_the_last_failed_round_before_the_cap_still_leaves_mining_available():
    """Off-by-one here retires mining a whole round early, silently removing real data from the run."""
    assert mining_available_for_state({
        "source_progress": {"a": {"exhausted": True}},
        "failed_discovery_rounds": MAX_FAILED_DISCOVERY_ROUNDS - 1,
    }) is True


def test_two_failed_discovery_rounds_is_the_cap():
    """One failure is a bad search; two in a row means the hub does not have another corpus
    carrying this task's labels, and continuing to pay is spending money to re-learn that."""
    assert MAX_FAILED_DISCOVERY_ROUNDS == 2


def test_an_empty_state_offers_mining():
    """Cold start: nothing has been consumed, so nothing is exhausted."""
    assert mining_available_for_state({}) is True


# --------------------------------------------------------------------------
# The ladder, end to end
# --------------------------------------------------------------------------


def _state(**overrides):
    from config.android_pool import ANDROID_POOL

    state = {
        "task": "xlam_bfcl",
        "selected_model": ANDROID_POOL[0],
        "train_examples": [],
        "source_progress": {},
        "failed_discovery_rounds": 0,
    }
    state.update(overrides)
    return state


def _stub_spec(load=None, **overrides):
    """The real xlam spec with its LOADER replaced.

    `spec.load` holds a direct reference to the function captured when the spec was built, so
    patching `tasks.xlam_bfcl._load` has no effect — the real loader runs and reads the on-disk
    corpus. Replacing the field is what actually isolates the test from the data.
    """
    import dataclasses

    from tasks import get_task

    if load is not None:
        overrides["load"] = lambda max_train, max_test, log=print: load(max_train, max_test)
    return dataclasses.replace(get_task("xlam_bfcl"), **overrides)


def _mine(state, *, rows=100, loader=None, discovery=None, monkeypatch=None):
    """Run `curate._mine_new_real` with the task's loader and web discovery stubbed."""
    import agent.nodes.curate as curate

    if loader is not None:
        spec = _stub_spec(load=loader)
        monkeypatch.setattr(curate, "_task_spec", lambda _state: spec)
    if discovery is not None:
        monkeypatch.setattr(curate, "_discover_new_source", discovery)
    logs: list[str] = []
    monkeypatch.setattr(curate, "_log", lambda _model_id, message: logs.append(message))
    added, report = curate._mine_new_real(
        state, {"rows": rows}, model_id="probe", eval_set=None,
    )
    return added, report, logs


def test_the_first_rung_re_reads_a_known_source_and_returns_only_the_novel_tail(monkeypatch):
    """Every loader takes a head slice, so asking for a LARGER slice returns a superset whose tail is
    novel by construction — and only that tail is returned.

    Returning the whole slice handed back every row the curriculum already had and left downstream
    deduplication to notice, which is why a request for 600 rows delivered 2,399 on run 38661753.
    """
    asked: list[int] = []

    def loader(max_train, max_test):
        asked.append(max_train)
        return [{"text": f"row {i}", "answer": "[]"} for i in range(max_train)], []

    state = _state(source_progress={
        "Salesforce/xlam-function-calling-60k": {"consumed": 3250},
    })
    added, report, _logs = _mine(state, rows=100, loader=loader, monkeypatch=monkeypatch)

    assert report["stage"] == "known_sources"
    assert asked[0] == 3350, "the slice must extend past what was consumed by exactly the request"
    assert len(added) == 100, "only the rows past the previous high-water mark are new"
    assert {row["text"] for row in added}.isdisjoint(
        {f"row {i}" for i in range(3250)}
    ), "no row the curriculum already consumed may come back"


def test_a_re_read_never_exceeds_the_per_rebuild_ceiling(monkeypatch):
    """A single rebuild adds at most MAX_MINED_ROWS_PER_REBUILD rows, whatever the plan asked for.

    Growth has to stay legible: a rebuild that adds a few hundred rows is an experiment whose effect
    can be read off the next eval, while one that adds three thousand changes the curriculum size, the
    training time and the class balance at once and the score movement cannot be attributed to any of
    them. It also keeps a finite corpus from being drained in a handful of iterations.
    """
    from agent.nodes.curate import MAX_MINED_ROWS_PER_REBUILD

    def loader(max_train, max_test):
        return [{"text": f"row {i}", "answer": "[]"} for i in range(max_train)], []

    state = _state(source_progress={
        "Salesforce/xlam-function-calling-60k": {"consumed": 1000},
    })
    added, _report, _logs = _mine(
        state, rows=MAX_MINED_ROWS_PER_REBUILD * 5, loader=loader, monkeypatch=monkeypatch,
    )
    assert len(added) == MAX_MINED_ROWS_PER_REBUILD


def test_the_source_position_advances_so_the_next_rebuild_resumes_where_this_one_stopped(monkeypatch):
    """Rows past the ceiling are deferred, not skipped.

    This is what makes a source worth returning to: `consumed` records where the read stopped, so the
    next mine_new_real rebuild continues from there. Advancing the pointer past rows that were never
    used would silently skip them for the rest of the run.
    """
    def loader(max_train, max_test):
        return [{"text": f"row {i}", "answer": "[]"} for i in range(max_train)], []

    state = _state(source_progress={
        "Salesforce/xlam-function-calling-60k": {"consumed": 500},
    })
    first, _r, _l = _mine(state, rows=200, loader=loader, monkeypatch=monkeypatch)
    second, _r, _l = _mine(state, rows=200, loader=loader, monkeypatch=monkeypatch)

    assert len(first) == len(second) == 200
    assert {row["text"] for row in first}.isdisjoint({row["text"] for row in second}), (
        "consecutive rebuilds must not hand back the same rows"
    )
    assert state["source_progress"]["Salesforce/xlam-function-calling-60k"]["consumed"] == 900


def test_web_research_is_not_reached_while_a_known_source_still_yields(monkeypatch):
    """Rung 2 costs money. Reaching it while rung 1 works is the B297 waste."""
    discovered = {"called": False}

    def discovery(*_args, **_kwargs):
        discovered["called"] = True
        return [{"text": "paid"}], {}

    def loader(max_train, max_test):
        return [{"text": f"row {i}", "answer": "[]"} for i in range(max_train)], []

    _added, report, _logs = _mine(
        _state(), loader=loader, discovery=discovery, monkeypatch=monkeypatch,
    )
    assert report["stage"] == "known_sources"
    assert discovered["called"] is False


def test_a_source_is_marked_exhausted_only_when_it_returns_fewer_rows_than_asked(monkeypatch):
    """Fewer rows than requested is the ONLY reliable evidence a head slice has hit the end of the
    split. Equal-to-asked means there is probably more, and treating it as the end retires a corpus
    that still has tens of thousands of rows."""
    def short_loader(max_train, max_test):
        return [{"text": f"row {i}", "answer": "[]"} for i in range(5)], []

    state = _state()
    _added, report, logs = _mine(state, loader=short_loader, monkeypatch=monkeypatch)

    assert report["sources_exhausted"] == ["Salesforce/xlam-function-calling-60k"]
    assert state["source_progress"]["Salesforce/xlam-function-calling-60k"]["exhausted"] is True
    assert any("EXHAUSTED" in line for line in logs)


def test_a_full_slice_leaves_the_source_open(monkeypatch):
    def full_loader(max_train, max_test):
        return [{"text": f"row {i}", "answer": "[]"} for i in range(max_train)], []

    state = _state()
    _added, report, _logs = _mine(state, loader=full_loader, monkeypatch=monkeypatch)

    assert report["sources_exhausted"] == []
    assert not state["source_progress"]["Salesforce/xlam-function-calling-60k"].get("exhausted")


def test_a_loader_that_raises_does_not_stop_the_rebuild(monkeypatch):
    """One bad source must not take the whole intervention down with it — and it must say so, or a
    rebuild that produced nothing looks like a mechanism failure rather than a source failure."""
    def broken(max_train, max_test):
        raise RuntimeError("hub is down")

    _added, _report, logs = _mine(
        _state(), loader=broken, discovery=lambda *_a, **_k: ([], {}),
        monkeypatch=monkeypatch,
    )
    assert any("FAILED" in line and "hub is down" in line for line in logs)


def test_the_second_rung_is_reached_once_every_source_is_exhausted(monkeypatch):
    def discovery(*_args, **_kwargs):
        return [{"text": "from a new dataset", "answer": "[]"}], {"status": "ok"}

    state = _state(source_progress={
        "Salesforce/xlam-function-calling-60k": {"exhausted": True},
    })
    added, report, logs = _mine(state, discovery=discovery, monkeypatch=monkeypatch)

    assert report["stage"] == "discovery"
    assert report["discovery_attempted"] is True
    assert [row["text"] for row in added] == ["from a new dataset"]
    assert any("EXHAUSTED" in line for line in logs)


def test_a_successful_discovery_round_resets_the_failure_counter(monkeypatch):
    """Otherwise a run that recovered would still be retired by failures it had already worked past."""
    def discovery(*_args, **_kwargs):
        return [{"text": "found something", "answer": "[]"}], {}

    state = _state(
        source_progress={"Salesforce/xlam-function-calling-60k": {"exhausted": True}},
        failed_discovery_rounds=1,
    )
    _mine(state, discovery=discovery, monkeypatch=monkeypatch)
    assert state["failed_discovery_rounds"] == 0


def test_a_fruitless_discovery_round_increments_the_counter_and_warns(monkeypatch):
    def discovery(*_args, **_kwargs):
        return [], {"status": "no_novelty"}

    state = _state(
        source_progress={"Salesforce/xlam-function-calling-60k": {"exhausted": True}},
    )
    added, report, logs = _mine(state, discovery=discovery, monkeypatch=monkeypatch)

    assert added == []
    assert report["stage"] == "discovery"
    assert state["failed_discovery_rounds"] == 1
    joined = " ".join(logs)
    assert "round 1 of 2" in joined
    assert "retires mine_new_real" in joined


def test_the_second_fruitless_round_announces_retirement(monkeypatch):
    def discovery(*_args, **_kwargs):
        return [], {}

    state = _state(
        source_progress={"Salesforce/xlam-function-calling-60k": {"exhausted": True}},
        failed_discovery_rounds=1,
    )
    _added, _report, logs = _mine(state, discovery=discovery, monkeypatch=monkeypatch)

    assert state["failed_discovery_rounds"] == MAX_FAILED_DISCOVERY_ROUNDS
    assert any("now RETIRED" in line for line in logs)
    assert mining_available_for_state(state) is False


def test_a_retired_ladder_does_not_pay_for_another_discovery_round(monkeypatch):
    """The point of retirement. It must also SAY it is retired, because "mining added nothing" and
    "mining had nothing left to add" are the two facts this file exists to keep apart."""
    called = {"discovery": False}

    def discovery(*_args, **_kwargs):
        called["discovery"] = True
        return [{"text": "x"}], {}

    state = _state(
        source_progress={"Salesforce/xlam-function-calling-60k": {"exhausted": True}},
        failed_discovery_rounds=MAX_FAILED_DISCOVERY_ROUNDS,
    )
    added, report, logs = _mine(state, discovery=discovery, monkeypatch=monkeypatch)

    assert added == []
    assert report["stage"] == "retired"
    assert report["discovery_attempted"] is False
    assert called["discovery"] is False
    assert any("MINING RETIRED" in line for line in logs)


def test_a_task_that_forbids_paid_discovery_says_so_rather_than_calling_out(monkeypatch):
    import agent.nodes.curate as curate

    state = _state(source_progress={"src": {"exhausted": True}})
    logs: list[str] = []
    monkeypatch.setattr(curate, "_log", lambda _m, message: logs.append(message))
    monkeypatch.setattr(
        curate, "_task_spec",
        lambda _state: _spec_without_paid_discovery(),
    )
    rows, report = curate._discover_new_source(
        state, want=100, model_id="probe", eval_set=None,
    )
    assert rows == [] and report == {"status": "not_permitted"}
    assert any("does not permit web discovery" in line for line in logs)


def _spec_without_paid_discovery():
    import dataclasses

    from tasks import get_task

    return dataclasses.replace(get_task("xlam_bfcl"), allow_paid_discovery=False)


def test_discovery_passes_the_pinned_label_space_so_a_foreign_class_cannot_enter(monkeypatch):
    """The closed label space is what stops a repeat of the four hallucinated RouterBench classes
    (B259): any mapped row whose label falls outside the frozen vocabulary is dropped per row."""
    import agent.nodes.curate as curate
    from data.eval_set import EvalSet

    captured: dict = {}

    def fake_mine(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr("data.loaders.web_acquire.mine_additional_real_rows", fake_mine)
    monkeypatch.setattr(curate, "_log", lambda *_a: None)

    eval_set = EvalSet(
        all=[{"text": "a", "label": "local"}, {"text": "b", "label": "route"}],
        task="routerbench",
    )
    state = {
        "task": "routerbench",
        "train_examples": [],
        "task_label_space": {"labels": ["local", "route"]},
        "eval_set": eval_set,
    }
    curate._discover_new_source(state, want=50, model_id="probe", eval_set=eval_set)

    assert captured["label_space"] == {"local", "route"}
    assert captured["task_plan"]["labels"] == ["local", "route"]


# --------------------------------------------------------------------------
# A discovered dataset is an asset, not a one-off delivery (B315)
# --------------------------------------------------------------------------
# Rung 2 costs money. Before 2026-08-19 `_discover_new_source` returned rows and recorded nothing, so a
# corpus found by paid web research was drained of whatever it happened to return in one pass and then
# forgotten — a later rebuild could not read more of it without paying to rediscover it. Registering it
# in `source_progress` converts a one-off lookup into a source rung 1 can return to for free, which is
# the difference between discovery being worth doing once and worth doing at all.


def _discovery(rows, dataset="someone/new-corpus", url="https://hf.co/someone/new-corpus"):
    """A rung-2 discovery that returns `rows` rows and names the dataset it found."""
    def _discover(state, *, want, model_id, eval_set):
        return (
            [{"text": f"discovered {i}", "answer": "[]"} for i in range(rows)],
            {"dataset": dataset, "url": url, "candidate_rows": rows},
        )
    return _discover


def _exhausted_state(**overrides):
    """A state where rung 1 has nothing left, so the ladder falls through to discovery."""
    return _state(
        source_progress={
            "Salesforce/xlam-function-calling-60k": {"consumed": 60_000, "exhausted": True},
        },
        **overrides,
    )


def test_a_discovered_dataset_is_recorded_so_a_later_rebuild_can_read_more_of_it(monkeypatch):
    """The whole point of registering it: rung 1 can return to it for free next time."""
    state = _exhausted_state()
    added, report, _logs = _mine(
        state, rows=200, discovery=_discovery(200), monkeypatch=monkeypatch,
    )

    assert report["stage"] == "discovery"
    assert len(added) == 200
    progress = state["source_progress"]["someone/new-corpus"]
    assert progress["consumed"] == 200, "the position must reflect what this rebuild actually took"
    assert progress["discovered"] is True
    assert progress["url"] == "https://hf.co/someone/new-corpus"
    assert "someone/new-corpus" in state["discovered_sources"]


def test_a_discovery_larger_than_the_ceiling_defers_the_remainder_rather_than_dropping_it(monkeypatch):
    """A generous discovery is capped like any other source, and the position records only the part
    taken — so the rows past the ceiling are deferred to the next rebuild, not skipped.

    Recording the number the PROVIDER returned instead would advance the pointer past rows that were
    never used, which is the same mistake trimming a re-read after the fact makes.
    """
    from agent.nodes.curate import MAX_MINED_ROWS_PER_REBUILD

    surplus = MAX_MINED_ROWS_PER_REBUILD + 500
    state = _exhausted_state()
    added, _report, logs = _mine(
        state, rows=surplus, discovery=_discovery(surplus), monkeypatch=monkeypatch,
    )

    assert len(added) == MAX_MINED_ROWS_PER_REBUILD
    assert state["source_progress"]["someone/new-corpus"]["consumed"] == MAX_MINED_ROWS_PER_REBUILD
    assert any("recording the source" in message for message in logs), (
        "the log must say the remainder was kept, or the next reader assumes it was thrown away"
    )


def test_a_discovery_that_names_no_dataset_says_so_instead_of_recording_a_blank_key(monkeypatch):
    """An unnamed source cannot be re-read, and a blank key in `source_progress` would be worse than
    no key: rung 1 would visit it forever and never find rows."""
    def _nameless(state, *, want, model_id, eval_set):
        return [{"text": "row", "answer": "[]"}], {"candidate_rows": 1}

    state = _exhausted_state()
    added, _report, logs = _mine(state, rows=1, discovery=_nameless, monkeypatch=monkeypatch)

    assert len(added) == 1, "the rows are still usable even though the source cannot be revisited"
    assert state.get("source_progress", {}) .get("") is None
    assert any("did not name a dataset id" in message for message in logs)
