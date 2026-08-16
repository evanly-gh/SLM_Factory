import pytest

from agent import data_rebuild as dr


def test_only_three_strategies():
    assert dr.DATA_REBUILD_STRATEGIES == ("acquire", "synthesize")


# The 2026-07-31 redesign dropped plan-identity *dedup*, but the durable paid-acquisition
# ledger still buckets spend per plan (MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN=3 inside
# MAX_PAID_ACQUIRE_ROUNDS_PER_RUN=9) and rejects an empty identity outright. Plans therefore
# still need a content-addressed budget key — see BUGS B220.

def test_plan_budget_identity_is_stable_and_content_addressed():
    plan = dr.normalize_data_rebuild_plan(
        {"strategy": "acquire", "new_real_rows": 200, "max_acquire_rounds": 2},
        task_type="classification",
        hypothesis="needs more real rows",
    )
    identity = dr.plan_budget_identity(plan)

    assert isinstance(identity, str) and identity.strip()
    # Stable across calls and across an equal copy: the same plan shares one budget bucket.
    assert identity == dr.plan_budget_identity(plan)
    assert identity == dr.plan_budget_identity(dict(plan))
    # Key ordering must not change the identity.
    assert identity == dr.plan_budget_identity(
        dict(reversed(list(plan.items())))
    )


def test_plan_budget_identity_differs_for_materially_different_plans():
    base = dr.normalize_data_rebuild_plan(
        {"strategy": "acquire", "new_real_rows": 200, "max_acquire_rounds": 2},
        task_type="classification", hypothesis="h",
    )
    other = dr.normalize_data_rebuild_plan(
        {"strategy": "acquire", "new_real_rows": 400, "max_acquire_rounds": 2},
        task_type="classification", hypothesis="h",
    )
    assert dr.plan_budget_identity(base) != dr.plan_budget_identity(other)


def test_plan_budget_identity_is_accepted_by_the_acquisition_ledger(tmp_path):
    """The identity curate produces must satisfy reserve_paid_acquisition's guard."""
    from data.acquisition_budget import reserve_paid_acquisition

    plan = dr.normalize_data_rebuild_plan(
        {"strategy": "acquire", "new_real_rows": 200, "max_acquire_rounds": 2},
        task_type="classification", hypothesis="h",
    )
    reservation = reserve_paid_acquisition(
        plan_identity=dr.plan_budget_identity(plan),
        per_plan_limit=2,
        run_limit=9,
        path=tmp_path / "ledger.jsonl",
    )
    assert reservation is not None


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


def test_orchestrator_cannot_override_the_deterministic_target():
    """
    Curriculum size is computed per tier by agent.data_sizing, not chosen by the orchestrator.
    A plan carrying `target_rows` is IGNORED (not an error — the model may emit it from habit),
    and the caller's deterministic value stands. In slm-clinc150-cse-38180646 the plan's 3000
    silently beat the computed 7053 and cut the curriculum nearly in half (B247).
    """
    plan = dr.normalize_data_rebuild_plan(
        {"strategy": "synthesize", "target_rows": 5000},
        task_type="classification",
        hypothesis="x",
        target_rows=7053,
    )
    assert plan["target_rows"] == 7053


def test_target_rows_is_not_an_orchestrator_field():
    assert "target_rows" not in dr._PLAN_FIELDS


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
