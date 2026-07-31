import pytest

from agent import data_rebuild as dr


def test_only_three_strategies():
    assert dr.DATA_REBUILD_STRATEGIES == ("resample", "acquire", "synthesize")


def test_normalize_accepts_synthesize_for_math():
    plan = dr.normalize_data_rebuild_plan(
        {"strategy": "synthesize", "synth_rows": 250},
        task_type="math_reasoning",
        hypothesis="weak on hard bucket",
    )
    assert plan["strategy"] == "synthesize"
    assert plan["synth_rows"] == 250  # honored as-is inside the 100–500 band
    assert "primary_strategy" not in plan
    assert "query_variant" not in plan
    assert "support_strategies" not in plan


def test_synth_rows_snapped_into_100_500_band():
    low = dr.normalize_data_rebuild_plan(
        {"strategy": "synthesize", "synth_rows": 40},
        task_type="classification", hypothesis="x",
    )
    assert low["synth_rows"] == 100  # below-range request snaps up to the floor
    high = dr.normalize_data_rebuild_plan(
        {"strategy": "synthesize", "synth_rows": 9999},
        task_type="classification", hypothesis="x",
    )
    assert high["synth_rows"] == 500  # above-range request snaps down to the cap
    unset = dr.normalize_data_rebuild_plan(
        {"strategy": "synthesize"},
        task_type="classification", hypothesis="x",
    )
    assert 100 <= unset["synth_rows"] <= 500  # unset defaults inside the band


def test_normalize_rejects_removed_strategy():
    with pytest.raises(ValueError):
        dr.normalize_data_rebuild_plan(
            {"strategy": "preserve_elite_resample"},
            task_type="classification",
            hypothesis="x",
        )


def test_synthesize_ungated_for_generation():
    # No task gating: synthesize is valid for generation-family too.
    plan = dr.normalize_data_rebuild_plan(
        {"strategy": "synthesize"},
        task_type="generation",
        hypothesis="x",
    )
    assert plan["strategy"] == "synthesize"
    assert plan["synth_rows"] > 0  # auto-filled positive budget


def test_target_rows_clamps_to_ceiling_not_2000():
    plan = dr.normalize_data_rebuild_plan(
        {"strategy": "resample", "target_rows": 5000},
        task_type="classification",
        hypothesis="x",
    )
    assert plan["target_rows"] == 5000


def test_removed_symbols_absent():
    for name in (
        "DataRebuildPlanSpaceExhausted",
        "ensure_untried_data_rebuild_plan",
        "data_rebuild_plan_identity",
        "tried_data_rebuild_plans",
        "eligible_data_rebuild_strategies",
        "resolve_elite_source_path",
        "require_resolvable_elite_source",
        "TARGETED_SYNTH_TASK_TYPES",
        "SAMPLING_STRATEGIES",
        "MAX_SUPPORT_STRATEGIES",
        "QUERY_VARIANTS",
    ):
        assert not hasattr(dr, name), f"{name} should be removed"


def test_fallback_returns_a_valid_strategy():
    state = {
        "task_type": "classification",
        "scores": [0.4],
        "curriculum_size_target": 3000,
        "test_report": {"by_difficulty": {"easy": {"accuracy": 0.3}}},
    }
    plan = dr.fallback_data_rebuild_plan(state, hypothesis="failing easy")
    assert plan["strategy"] in dr.DATA_REBUILD_STRATEGIES
    assert plan["target_rows"] == 3000
