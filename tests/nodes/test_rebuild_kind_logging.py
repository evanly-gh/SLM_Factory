"""Every data_rebuild names the KINDS that produced its rows.

"data_rebuild" in a trajectory is ambiguous: a single rebuild routinely runs the plan's strategy,
then resample-fill, then synth-fill, and those are different interventions with different expected
effects. Reading the RouterBench logs it was impossible to tell which had run, and synth-fill rows
carried no `_strategy_origin` at all so they were reported as "unattributed" — which is also why the
run graphics drew a synthetic band of ~1% while most of the synthesis budget was being spent there.
"""
from agent.nodes.curate import (
    REBUILD_KIND_NAMES,
    SYNTHETIC_PROVENANCE_TAGS,
    rebuild_kind_name,
)


def test_every_producer_has_a_canonical_kind_name():
    """The four kinds the orchestrator can cause, plus the universal filler."""
    assert rebuild_kind_name("resample") == "resample-fill"
    assert rebuild_kind_name("mine_new_real_source") == "mine-new-real"
    assert rebuild_kind_name("surgical_synthesize") == "surgical-synth"
    assert rebuild_kind_name("synthesize") == "fill-synth"
    assert rebuild_kind_name("synth_fill") == "synth-fill"


def test_unknown_origin_passes_through_rather_than_being_hidden():
    """An unmapped origin must stay visible; silently renaming it to "unknown" is how the
    synth-fill gap went unnoticed."""
    assert rebuild_kind_name("some_future_strategy") == "some_future_strategy"


def test_kind_names_are_log_safe_tokens():
    for name in REBUILD_KIND_NAMES.values():
        assert name == name.strip() and " " not in name and name


def test_synthetic_provenance_covers_all_three_generated_paths():
    """`n_hard_generated` counted only "synthetic", so synth-fill and generation-family
    positives were invisible in the composition report."""
    assert set(SYNTHETIC_PROVENANCE_TAGS) == {
        "synthetic", "synthetic_fill", "synthetic_positive",
    }


def test_synth_fill_rows_are_attributed():
    """Regression: synth-fill tagged `_provenance` but not `_strategy_origin`, so its rows landed
    in the "unattributed" bucket of the strategy composition."""
    from agent.nodes import curate

    rows = [{"text": "generated", "label": "local", "_source": "synth"}]
    tagged = [
        {
            **row,
            "_provenance": row.get("_provenance") or "synthetic_fill",
            "_strategy_origin": row.get("_strategy_origin") or "synth_fill",
        }
        for row in rows
    ]
    assert tagged[0]["_strategy_origin"] == "synth_fill"
    assert rebuild_kind_name(tagged[0]["_strategy_origin"]) == "synth-fill"
    assert tagged[0]["_provenance"] in curate.SYNTHETIC_PROVENANCE_TAGS
