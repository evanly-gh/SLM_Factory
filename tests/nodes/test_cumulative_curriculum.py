"""The curriculum is CUMULATIVE, and no held-out eval row may ever enter it.

WHY THIS FILE EXISTS
    Two invariants, both of which failed silently in production.

    CUMULATIVE. The curriculum used to be rebuilt to a TARGET SIZE every iteration, so something
    had to refill it from the train pool — and with nothing else changed, that filler re-selected
    the identical ~3,235 rows and honestly reported `0 novel` on eight consecutive rebuilds of run
    38566712. The target and the filler are both gone: cold start loads gold rows, every rebuild
    ADDS, and rows leave only via quality control or the eval firewall. Nothing re-draws from a pool
    it has already drawn from.

    A rebuild that adds nothing is now an ERROR rather than one `0 novel` line among twenty. It
    means an intervention was chosen, a plan was built, and the mechanism it named could not do the
    thing it exists to do — so training this iteration would repeat the previous one exactly, and
    the iteration cannot tell us anything.

    THE EVAL FIREWALL. Every layer that can introduce a row — the initial gold load, mining,
    synthesis, the persistent train pool, and the finished dataset — is filtered against the
    normalized text of the frozen eval set. It is applied repeatedly rather than once because each
    layer has its own way of reintroducing a row, and a leak here silently invalidates every number
    the run reports.

    Both invariants are about what may enter the curriculum, so the steps that run between the merge
    and the artifact — CoT annotation, quality control, the anchor pools synthesis draws from — are
    covered here too: they are the only other places a row's content changes on its way to disk.
"""
from __future__ import annotations

import json

import pytest

from config.android_pool import ANDROID_POOL
from data.eval_set import EvalSet
from data.loaders.dataset_integrity import normalize_text

EVAL_TEXT = "Held-Out\nEvaluation Secret"


@pytest.fixture(autouse=True)
def _isolated_artifacts(tmp_path, monkeypatch):
    """`curate_node` writes `artifacts/dataset_v{N}.jsonl` relative to the process cwd."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _small_curriculums_are_allowed(monkeypatch):
    """These fixtures are a handful of rows. They test the firewall and the accumulation rule, not
    curriculum viability, so the 500-row floor is lowered. Patched as an attribute because the
    constant is read from the environment at import time."""
    import agent.nodes.curate as curate

    monkeypatch.setattr(curate, "MIN_CURRICULUM_ROWS", 1)


@pytest.fixture(autouse=True)
def _no_teacher(monkeypatch):
    """No test may reach the synthesis endpoint or the CoT teacher."""
    import data.synth_client as synth_client

    monkeypatch.setattr(synth_client, "wait_until_available", lambda **_k: False)
    monkeypatch.setattr(synth_client, "is_available", lambda **_k: False)
    monkeypatch.setattr(synth_client, "get_generate_fn", lambda **_k: None)


def _eval_set(task="clinc150", rows=None):
    return EvalSet(all=rows or [{"text": EVAL_TEXT, "label": "a"}], task=task)


def _state(task="clinc150", **overrides):
    state = {
        "task": task,
        "selected_model": ANDROID_POOL[0],
        "last_intervention": "data_rebuild",
        "eval_set": _eval_set(task),
        "train_examples": [],
        "current_dataset_path": None,
        "dataset_version": 0,
    }
    state.update(overrides)
    return state


def _rows(n, prefix="row", label="a"):
    return [{"text": f"{prefix} {i}", "label": label} for i in range(n)]


def _saved(state):
    with open(state["current_dataset_path"], encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _texts(rows):
    return {normalize_text(row.get("text", row.get("prompt", ""))) for row in rows}


def _plan(strategy, rows=100, categories=()):
    return {
        "schema_version": 3,
        "strategy": strategy,
        "rows": rows,
        "target_categories": [dict(entry) for entry in categories],
        "pattern_hint": "",
    }


# --------------------------------------------------------------------------
# The first build is the gold rows, and nothing else
# --------------------------------------------------------------------------


def test_the_first_build_is_the_gold_rows_the_loader_returned():
    """No plan and no strategy — there is nothing to add to yet."""
    from agent.nodes.curate import curate_node

    state = _state(train_examples=_rows(6))
    out = curate_node(state)

    saved = _saved(out)
    assert len(saved) == 6
    assert out["last_curation"]["strategy"] == "initial_gold"
    assert out["last_curation"]["rows_added"] == 6
    assert out["last_curation"]["previous_rows"] == 0
    assert all(row["_provenance"] == "train_anchor" for row in saved)


def test_the_first_build_does_not_log_the_zero_added_error(capsys):
    """There is no plan on a first build, so "the mechanism added nothing" cannot apply."""
    from agent.nodes.curate import curate_node

    curate_node(_state(train_examples=_rows(4)))
    assert "added 0 new rows" not in capsys.readouterr().out


def test_a_hyperparameter_iteration_leaves_the_dataset_untouched():
    """The comparison is only clean if the curriculum is held fixed."""
    from agent.nodes.curate import curate_node

    state = _state(train_examples=_rows(4), last_intervention="hyperparameter")
    out = curate_node(state)
    assert out.get("current_dataset_path") is None
    assert "last_curation" not in out


def test_curating_without_a_frozen_eval_set_refuses_rather_than_skipping_the_firewall():
    """Without the eval set there is nothing to filter against, so a rebuild would be
    unfirewalled — the one failure mode that invalidates every number the run reports."""
    from agent.nodes.curate import curate_node

    with pytest.raises(RuntimeError, match="requires a fixed eval_set"):
        curate_node(_state(train_examples=_rows(4), eval_set=None))


def test_a_curriculum_below_the_viability_floor_fails_loudly(monkeypatch):
    """Synth-fill no longer pads to a target, so the size is whatever real data supplies. Below the
    floor the run would burn GPU hours producing a number nobody should trust, and the cause is
    always upstream where it can be fixed."""
    import agent.nodes.curate as curate

    monkeypatch.setattr(curate, "MIN_CURRICULUM_ROWS", 500)
    with pytest.raises(RuntimeError, match="below the 500-row floor"):
        curate.curate_node(_state(train_examples=_rows(4)))


# --------------------------------------------------------------------------
# A rebuild ADDS
# --------------------------------------------------------------------------


def test_a_rebuild_adds_to_what_is_already_there(monkeypatch):
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(5)))
    previous_path = first["current_dataset_path"]

    monkeypatch.setattr(
        "agent.nodes.curate._mine_new_real",
        lambda *_a, **_k: (_rows(3, prefix="mined"), {"stage": "known_sources"}),
    )
    second = curate_node(_state(
        train_examples=_rows(5),
        current_dataset_path=previous_path,
        dataset_version=1,
        data_rebuild_plan=_plan("mine_new_real"),
    ))

    saved = _saved(second)
    assert len(saved) == 8, "the rebuild replaced the curriculum instead of growing it"
    assert second["last_curation"]["rows_added"] == 3
    assert second["last_curation"]["previous_rows"] == 5
    assert _texts(_rows(5)) <= _texts(saved), "existing rows were dropped by a rebuild"


def test_a_rebuild_never_re_draws_from_the_train_pool(monkeypatch):
    """The removed gold FILL. With nothing else changed it re-selected the identical ~3,235 rows and
    reported `0 novel` eight times in one run — so a rebuild whose mechanism produced nothing must
    add nothing, not quietly refill from the pool to hit a target."""
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(40)))
    assert len(_saved(first)) == 40

    monkeypatch.setattr(
        "agent.nodes.curate._mine_new_real", lambda *_a, **_k: ([], {"stage": "retired"}),
    )
    second = curate_node(_state(
        train_examples=_rows(40),
        current_dataset_path=first["current_dataset_path"],
        dataset_version=1,
        data_rebuild_plan=_plan("mine_new_real", rows=2000),
    ))
    assert second["last_curation"]["rows_added"] == 0
    assert len(_saved(second)) == 40, "something refilled the curriculum from the pool"


def test_no_plan_field_can_ask_for_a_target_size():
    """The curriculum has a starting size and it grows; there is nothing for a target to mean."""
    from agent.data_rebuild import normalize_data_rebuild_plan

    assert "target_rows" not in normalize_data_rebuild_plan(
        _plan("surgical_synthesis"), task="clinc150",
    )


def test_a_rebuild_that_adds_nothing_is_logged_as_an_error(monkeypatch, capsys):
    """Not a `0 novel` line among twenty. An intervention was chosen, a plan was built, and the
    mechanism it named could not do the thing it exists to do."""
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(6)))
    monkeypatch.setattr(
        "agent.nodes.curate._mine_new_real", lambda *_a, **_k: ([], {"stage": "retired"}),
    )
    capsys.readouterr()
    curate_node(_state(
        train_examples=_rows(6),
        current_dataset_path=first["current_dataset_path"],
        dataset_version=1,
        data_rebuild_plan=_plan("mine_new_real"),
    ))

    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "added 0 new rows" in out
    assert "mine_new_real" in out
    assert "repeat the previous one exactly" in out


def test_a_rebuild_whose_rows_are_all_duplicates_also_reports_zero_added(monkeypatch, capsys):
    """The distinction that matters: the mechanism RAN and produced rows, and the curriculum still
    did not change. Reporting a row count here would claim progress that did not happen."""
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(6)))
    monkeypatch.setattr(
        "agent.nodes.curate._mine_new_real",
        lambda *_a, **_k: (_rows(6), {"stage": "known_sources"}),
    )
    capsys.readouterr()
    second = curate_node(_state(
        train_examples=_rows(6),
        current_dataset_path=first["current_dataset_path"],
        dataset_version=1,
        data_rebuild_plan=_plan("mine_new_real"),
    ))

    assert second["last_curation"]["rows_added"] == 0
    assert "added 0 new rows" in capsys.readouterr().out


def test_each_rebuild_writes_its_own_versioned_artifact(monkeypatch):
    """Rollback replays a previous dataset, so overwriting one would make the previous iteration
    unreproducible."""
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(5)))
    monkeypatch.setattr(
        "agent.nodes.curate._mine_new_real",
        lambda *_a, **_k: (_rows(2, prefix="mined"), {}),
    )
    second = curate_node(_state(
        train_examples=_rows(5),
        current_dataset_path=first["current_dataset_path"],
        dataset_version=1,
        data_rebuild_plan=_plan("mine_new_real"),
    ))

    assert first["current_dataset_path"].endswith("dataset_v1.jsonl")
    assert second["current_dataset_path"].endswith("dataset_v2.jsonl")
    assert first["current_dataset_path"] != second["current_dataset_path"]
    assert all(row["_dataset_version"] == 2 for row in _saved(second))


def test_mined_rows_join_the_persistent_pool_so_a_later_re_read_counts_them(monkeypatch):
    """Otherwise the next re-read offers them again and the run pays to rediscover its own rows."""
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(5)))
    monkeypatch.setattr(
        "agent.nodes.curate._mine_new_real",
        lambda *_a, **_k: (_rows(3, prefix="mined"), {}),
    )
    second = curate_node(_state(
        train_examples=_rows(5),
        current_dataset_path=first["current_dataset_path"],
        dataset_version=1,
        data_rebuild_plan=_plan("mine_new_real"),
    ))
    assert _texts(_rows(3, prefix="mined")) <= _texts(second["train_examples"])


# --------------------------------------------------------------------------
# `_dedupe_into` is the only place the curriculum grows
# --------------------------------------------------------------------------


def test_dedupe_into_appends_only_novel_normalized_texts():
    from agent.nodes.curate import _dedupe_into

    curriculum = [{"text": "alpha"}, {"text": "beta"}]
    merged, added = _dedupe_into(
        curriculum, [{"text": "  ALPHA  "}, {"text": "gamma"}, {"text": "beta"}],
    )
    assert added == 1
    assert [row["text"] for row in merged] == ["alpha", "beta", "gamma"]


def test_dedupe_into_treats_a_duplicate_inside_the_additions_as_a_duplicate():
    from agent.nodes.curate import _dedupe_into

    merged, added = _dedupe_into([], [{"text": "same"}, {"text": "same"}])
    assert added == 1 and len(merged) == 1


def test_dedupe_into_does_not_mutate_the_curriculum_it_was_given():
    from agent.nodes.curate import _dedupe_into

    curriculum = [{"text": "alpha"}]
    merged, _added = _dedupe_into(curriculum, [{"text": "beta"}])
    assert curriculum == [{"text": "alpha"}]
    assert merged is not curriculum


def test_dedupe_into_keeps_rows_that_carry_no_text():
    """A row with no surface text has no identity to deduplicate on. Dropping it would silently
    delete NER-shaped rows whose content lives elsewhere; counting it is the honest choice."""
    from agent.nodes.curate import _dedupe_into

    merged, added = _dedupe_into([{"text": "alpha"}], [{"entities": []}, {"entities": []}])
    assert added == 2 and len(merged) == 3


def test_dedupe_into_normalizes_whitespace_and_case():
    from agent.nodes.curate import _dedupe_into

    _merged, added = _dedupe_into([{"text": "Held-Out Evaluation Secret"}],
                                  [{"text": "  held-out   evaluation secret "}])
    assert added == 0


# --------------------------------------------------------------------------
# The eval firewall
# --------------------------------------------------------------------------
#
# Covered twice on purpose: at the unit level, where the filter can be called for one layer at a
# time, and end to end through `curate_node`, which is the only place that proves every layer is
# actually wired up. The whole-node half is not redundant — it is the half that exercises the
# per-row REPORTING, which once raised NameError on the first row the firewall blocked, so the
# safety mechanism took the run down at exactly the moment it caught a leak.


@pytest.mark.parametrize("layer", ["initial_gold", "mined", "synthesis", "final"])
def test_no_held_out_eval_row_survives_any_firewall_layer(layer):
    """The invariant, at every checkpoint `curate_node` applies it.

    It is applied repeatedly rather than once because each layer has its own way of reintroducing a
    row — the gold load, mining, synthesis, and the finished dataset — and a leak at any of them
    silently invalidates every number the run reports.
    """
    from agent.nodes.curate import _exclude_eval_rows

    tally: dict[str, int] = {}
    kept, removed = _exclude_eval_rows(
        [*_rows(3), {"text": "  held-out evaluation   secret ", "label": "b"}],
        _eval_set(), tally=tally, layer=layer,
    )
    assert removed == 1
    assert normalize_text(EVAL_TEXT) not in _texts(kept)
    assert tally[layer] == 1


def test_a_clean_layer_records_a_zero_rather_than_nothing():
    """A count of zero is the confirmation a clean build needs; only recording non-zero removals
    makes "the firewall found nothing" indistinguishable from "the firewall did not run"."""
    from agent.nodes.curate import _exclude_eval_rows

    tally: dict[str, int] = {}
    _kept, removed = _exclude_eval_rows(_rows(3), _eval_set(), tally=tally, layer="final")
    assert removed == 0
    assert tally == {"final": 0}


def test_the_comparison_is_on_normalized_text_not_bytes():
    """Whitespace and case differences are exactly how a leaked row escapes a naive comparison."""
    from agent.nodes.curate import _exclude_eval_rows

    for variant in (EVAL_TEXT.upper(), f"  {EVAL_TEXT}  ", EVAL_TEXT.replace("\n", "   ")):
        _kept, removed = _exclude_eval_rows([{"text": variant}], _eval_set())
        assert removed == 1, f"{variant!r} slipped through the firewall"


def test_the_filter_names_a_blocked_row_without_echoing_its_text():
    """The filter must say WHICH rows leaked, or an operator cannot tell a one-row overlap from a
    contaminated source — but the held-out content must never reach the run log, which is why the
    row is reported as a length and a short hash instead of verbatim.

    That reporting line is asserted rather than assumed because it is where the firewall does its
    only non-trivial work outside the comparison itself: it once called `hashlib.sha256` in a module
    that never imported `hashlib`, so the first blocked row raised NameError and the safety
    mechanism killed the run at the moment it caught a leak.
    """
    from agent.nodes.curate import _exclude_eval_rows

    eval_set = EvalSet(all=[{"text": "held out secret", "label": "a"}], task="clinc150")
    logs: list[str] = []
    kept, removed = _exclude_eval_rows(
        [{"text": "held out secret"}, {"text": "safe"}], eval_set,
        tally={}, layer="initial_gold", log=logs.append,
    )
    assert removed == 1
    assert [row["text"] for row in kept] == ["safe"]
    assert any("text_sha8=" in line for line in logs)
    assert not any("held out secret" in line for line in logs)


def test_a_gold_row_matching_a_held_out_eval_row_never_reaches_the_curriculum():
    """The contaminating row differs only in whitespace and case, which is what the normalized
    comparison exists to catch."""
    from agent.nodes.curate import curate_node

    train = _rows(4) + [{"text": "  held-out evaluation   secret ", "label": "b"}]
    out = curate_node(_state(train_examples=train))

    assert normalize_text(EVAL_TEXT) not in _texts(_saved(out))
    assert len(_saved(out)) == 4


def test_a_mined_row_matching_a_held_out_eval_row_is_blocked(monkeypatch):
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(4)))
    monkeypatch.setattr(
        "agent.nodes.curate._mine_new_real",
        lambda *_a, **_k: ([{"text": EVAL_TEXT, "label": "a"},
                            {"text": "clean mined row", "label": "a"}], {}),
    )
    second = curate_node(_state(
        train_examples=_rows(4),
        current_dataset_path=first["current_dataset_path"],
        dataset_version=1,
        data_rebuild_plan=_plan("mine_new_real"),
    ))

    saved = _texts(_saved(second))
    assert normalize_text(EVAL_TEXT) not in saved
    assert normalize_text("clean mined row") in saved


def test_a_synthesized_row_matching_a_held_out_eval_row_is_blocked(monkeypatch):
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(4)))
    monkeypatch.setattr(
        "agent.nodes.curate._surgical_synthesize",
        lambda *_a, **_k: [{"text": EVAL_TEXT, "label": "a", "_provenance": "synthetic"},
                           {"text": "clean synth row", "label": "a",
                            "_provenance": "synthetic"}],
    )
    second = curate_node(_state(
        train_examples=_rows(4),
        current_dataset_path=first["current_dataset_path"],
        dataset_version=1,
        data_rebuild_plan=_plan("surgical_synthesis",
                                categories=[{"category": "wrong_label", "count": 4}]),
    ))

    assert normalize_text(EVAL_TEXT) not in _texts(_saved(second))


def test_the_firewall_is_audited_per_layer_including_the_healthy_zero():
    """A count of zero is the confirmation a clean build needs; only reporting non-zero removals
    makes "the firewall found nothing" indistinguishable from "the firewall did not run"."""
    from agent.nodes.curate import curate_node

    train = _rows(4) + [{"text": EVAL_TEXT, "label": "b"}]
    out = curate_node(_state(train_examples=train))

    firewall = out["last_curation"]["eval_firewall"]
    assert firewall["total"] >= 1
    assert firewall["by_layer"]["initial_gold"] >= 1
    assert "final" in firewall["by_layer"]


def test_each_blocked_row_is_named_with_a_redacted_fingerprint(capsys):
    """An operator has to see WHICH rows leaked, but the eval content must never land in the run
    log — so the text is reported as a length and a short hash."""
    from agent.nodes.curate import curate_node

    curate_node(_state(train_examples=_rows(4) + [{"text": EVAL_TEXT, "label": "b"}]))
    out = capsys.readouterr().out

    assert "[firewall:initial_gold] BLOCKED row" in out
    assert "text_sha8=" in out
    assert "would leak the eval set" in out
    assert EVAL_TEXT not in out
    assert "evaluation secret" not in out.lower()


def test_the_firewall_reports_a_bounded_number_of_blocked_rows(monkeypatch, capsys):
    """A pathological rebuild could otherwise blocklist thousands of rows and bury the log in the
    very noise this reporting exists to replace."""
    import agent.nodes.curate as curate

    monkeypatch.setattr(curate, "_FIREWALL_LOG_LIMIT", 3)
    eval_rows = [{"text": f"secret {i}", "label": "a"} for i in range(10)]
    state = _state(
        eval_set=_eval_set(rows=eval_rows),
        train_examples=_rows(4) + [dict(row) for row in eval_rows],
    )
    curate.curate_node(state)
    out = capsys.readouterr().out

    assert out.count("BLOCKED row") == 3
    assert "and 7 more blocked for the same reason" in out


def test_the_persistent_pool_is_also_firewalled(monkeypatch):
    """Mined rows are merged into `train_examples`, which every later rebuild and checkpoint reads.
    A leak there would survive the rest of the run."""
    from agent.nodes.curate import _merge_persistent_train_rows

    merged = _merge_persistent_train_rows(
        [{"text": "existing"}], [{"text": EVAL_TEXT}, {"text": "novel"}], _eval_set(),
    )
    assert _texts(merged) == {normalize_text("existing"), normalize_text("novel")}


def test_the_persistent_pool_keeps_one_copy_of_each_text():
    from agent.nodes.curate import _merge_persistent_train_rows

    merged = _merge_persistent_train_rows(
        [{"text": "same"}], [{"text": "  SAME  "}], _eval_set(),
    )
    assert len(merged) == 1


def test_an_eval_set_with_no_rows_blocks_nothing_rather_than_everything():
    """An empty set must read as "unknown", never as "nothing is allowed" — the latter would empty
    the curriculum."""
    from agent.nodes.curate import _exclude_eval_rows

    rows = _rows(3)
    kept, removed = _exclude_eval_rows(rows, EvalSet(all=[], task="clinc150"))
    assert kept == rows and removed == 0


def test_the_firewall_compares_prompt_as_well_as_text():
    """Row schemas differ across tasks; a firewall that only reads `text` would miss a row whose
    surface content is stored under `prompt`."""
    from agent.nodes.curate import _exclude_eval_rows

    eval_set = EvalSet(all=[{"prompt": "leaked question", "label": "a"}], task="clinc150")
    kept, removed = _exclude_eval_rows(
        [{"prompt": "leaked question"}, {"text": "safe"}], eval_set,
    )
    assert removed == 1
    assert [row.get("text") for row in kept] == ["safe"]


# --------------------------------------------------------------------------
# CoT annotation
# --------------------------------------------------------------------------


def test_a_cot_task_annotates_its_rows_and_preserves_gold_reasoning():
    """GSM8K ships gold chains-of-thought in its `####` split, so annotation must PRESERVE an
    existing `cot_reasoning` rather than regenerate it: regenerating was both wasteful — one teacher
    call per row — and quality-reducing, replacing a gold chain with a weaker teacher one (B141).

    Driven through `curate._annotate_generation_cot` rather than `data.curriculum.annotate_cot`
    because the two have disagreed before: the call site once passed a `task=` keyword the callee
    never accepted, so every gsm8k rebuild raised TypeError the moment the CoT teacher was
    reachable. A test of the callee alone would not have seen it.
    """
    import agent.nodes.curate as curate

    rows = [
        {"text": "Ann has 3 pears, buys 2. How many?", "answer": "#### 5",
         "cot_reasoning": "gold: three plus two"},
        {"text": "Bob runs 4 miles for 5 days. How far?", "answer": "#### 20"},
    ]
    annotated = curate._annotate_generation_cot(
        rows, {"task": "gsm8k", "task_brief": None}, model_id="probe",
    )

    assert annotated[0]["cot_reasoning"] == "gold: three plus two"
    assert len(annotated) == 2


# --------------------------------------------------------------------------
# What a rebuild records
# --------------------------------------------------------------------------


def test_the_curation_record_names_the_strategy_and_the_rows_it_added(monkeypatch):
    """`pi.D.composition` is what the trajectory table and the post-run report read. A rebuild that
    cannot be attributed to a mechanism is the ambiguity `data_rebuild` alone used to have."""
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(5)))
    monkeypatch.setattr(
        "agent.nodes.curate._mine_new_real",
        lambda *_a, **_k: (_rows(4, prefix="mined"), {"stage": "known_sources"}),
    )
    out = curate_node(_state(
        train_examples=_rows(5),
        current_dataset_path=first["current_dataset_path"],
        dataset_version=1,
        data_rebuild_plan=_plan("mine_new_real", rows=400),
    ))

    record = out["last_curation"]
    assert record["strategy"] == "mine_new_real"
    assert record["rows_added"] == 4
    assert record["n_hard_source"] == 4
    assert record["mining_report"]["stage"] == "known_sources"
    assert record["data_rebuild_plan"]["rows"] == 400


def test_an_invalid_plan_falls_back_rather_than_stopping_the_rebuild(monkeypatch):
    """A rebuild is expensive to reach; a malformed plan should cost a deterministic choice, not the
    iteration. The fallback must also be recorded as a fallback."""
    from agent.nodes.curate import curate_node

    first = curate_node(_state(train_examples=_rows(5)))
    monkeypatch.setattr(
        "agent.nodes.curate._mine_new_real",
        lambda *_a, **_k: (_rows(2, prefix="mined"), {}),
    )
    monkeypatch.setattr(
        "agent.nodes.curate._surgical_synthesize", lambda *_a, **_k: [],
    )
    out = curate_node(_state(
        train_examples=_rows(5),
        current_dataset_path=first["current_dataset_path"],
        dataset_version=1,
        data_rebuild_plan="not a plan at all",
    ))
    assert out["last_curation"]["strategy"] in ("mine_new_real", "surgical_synthesis")
    assert "fallback" in out["last_curation"]["data_rebuild_plan"]["hypothesis"].lower()


def test_the_surgical_history_skips_a_category_that_stopped_responding(monkeypatch):
    """Without this the run keeps pouring rows at something that is not responding, which is the
    loop the CLINC150 run got stuck in (B224)."""
    from agent.nodes.curate import _eligible_categories

    state = _state(surgical_category_history={
        "wrong_label": {"count_when_targeted": 40, "iteration": 2, "rows_generated": 300},
    })
    eligible = _eligible_categories(
        state,
        _plan("surgical_synthesis", categories=[
            {"category": "wrong_label", "count": 41},
            {"category": "extraction_failed", "count": 12},
        ]),
        "probe",
    )
    assert [entry["category"] for entry in eligible] == ["extraction_failed"]


def test_a_category_that_improved_is_targeted_again(monkeypatch):
    from agent.nodes.curate import _eligible_categories

    state = _state(surgical_category_history={
        "wrong_label": {"count_when_targeted": 40, "iteration": 2, "rows_generated": 300},
    })
    eligible = _eligible_categories(
        state, _plan("surgical_synthesis",
                     categories=[{"category": "wrong_label", "count": 25}]), "probe",
    )
    assert [entry["category"] for entry in eligible] == ["wrong_label"]


def test_categories_fall_back_to_the_last_test_report_when_the_plan_names_none():
    from agent.nodes.curate import _eligible_categories

    state = _state(test_report={"confusion_pairs": [
        {"gold": "extraction_failed", "count": 9},
        {"gold": "wrong_label", "count": 30},
    ]})
    eligible = _eligible_categories(state, _plan("surgical_synthesis"), "probe")
    # Largest failure count first, because that is where the points are.
    assert [entry["category"] for entry in eligible] == ["wrong_label", "extraction_failed"]


def test_surgical_synthesis_with_no_evidence_generates_nothing_and_says_why(capsys):
    """There is no failure category to aim at, so there is nothing to generate. Silence here is
    indistinguishable from the B291 fallthrough that returned `[]` for six consecutive rebuilds."""
    from agent.nodes.curate import _surgical_synthesize

    rows = _surgical_synthesize(
        _state(), _plan("surgical_synthesis"), _rows(10), model_id="probe",
        eval_set=_eval_set(),
    )
    assert rows == []


def test_surgical_synthesis_without_anchors_says_so(capsys):
    from agent.nodes.curate import _surgical_synthesize

    assert _surgical_synthesize(
        _state(), _plan("surgical_synthesis"), [], model_id="probe", eval_set=_eval_set(),
    ) == []
    assert "no anchor rows available" in capsys.readouterr().out


def test_an_unreachable_teacher_yields_no_rows_rather_than_hanging_the_rebuild(capsys):
    """`conftest.py` pins the mid-run wait to zero; the decision under test is that an unreachable
    endpoint degrades to zero rows and says so, rather than raising or blocking."""
    from agent.nodes.curate import _surgical_synthesize

    rows = _surgical_synthesize(
        _state(), _plan("surgical_synthesis",
                        categories=[{"category": "wrong_label", "count": 5}]),
        _rows(10), model_id="probe", eval_set=_eval_set(),
    )
    assert rows == []
    assert "unavailable" in capsys.readouterr().out


def test_an_anchor_pool_for_a_closed_label_space_is_that_class_own_rows():
    """The category IS a class here, so anchoring on its own rows conveys its phrasing and length
    distribution."""
    from agent.nodes.curate import _anchor_pool

    train = _rows(3, prefix="a", label="local") + _rows(2, prefix="b", label="route")
    pool = _anchor_pool(_state("routerbench"), "route", train)
    assert {row["label"] for row in pool} == {"route"}


def test_an_anchor_pool_for_an_open_ended_task_is_the_whole_pool():
    """There is no class to anchor on — the category names an error KIND — so the category is
    carried into the prompt as a steer instead."""
    from agent.nodes.curate import _anchor_pool

    train = _rows(5)
    assert _anchor_pool(_state("xlam_bfcl"), "wrong_arguments", train) == train
