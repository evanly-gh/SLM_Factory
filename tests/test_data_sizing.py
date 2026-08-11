"""Deterministic per-tier curriculum sizing (agent/data_sizing.py).

Policy 2026-08-05: the target is a function of measured task novelty (1 − zero-shot baseline) and
model capacity (inverse parameter count), clamped to [floor, ceiling]. It is NOT an LLM decision.
"""
from agent.data_sizing import (
    compute_curriculum_target,
    resize_curriculum_for_tier,
    size_factor_for_params,
)

FLOOR, CEILING = 5000, 25000


def _target(baseline, params):
    value, _ = compute_curriculum_target(
        zero_shot_baseline=baseline, n_params=params, floor=FLOOR, ceiling=CEILING
    )
    return value


def test_bigger_model_gets_a_smaller_target():
    """Explicit requirement: escalating to a larger model must LOWER the data target."""
    small = _target(0.30, 0.6e9)
    large = _target(0.30, 4.0e9)
    assert large < small


def test_more_novel_task_gets_a_bigger_target():
    """A task the model already half-knows needs less data than one it cannot do at all."""
    novel = _target(0.10, 1.0e9)
    familiar = _target(0.85, 1.0e9)
    assert novel > familiar


def test_target_is_always_within_floor_and_ceiling():
    for baseline in (0.0, 0.3152, 0.99, None):
        for params in (0.1e9, 0.6e9, 4e9, 70e9, None):
            value = _target(baseline, params)
            assert FLOOR <= value <= CEILING


def test_size_factor_is_clamped_at_both_ends():
    assert size_factor_for_params(1.0e6) == 2.0     # tiny model, capped
    assert size_factor_for_params(500e9) == 0.5     # huge model, floored
    assert size_factor_for_params(1.0e9) == 1.0     # reference point
    assert size_factor_for_params(None) == 1.0      # unknown -> neutral


def test_missing_baseline_uses_neutral_novelty_not_an_extreme():
    """Before the first eval there is no baseline; the target must not be biased high or low.

    Uses a 0.6B model so the floor does not bind and the ordering is actually observable — at
    1B the "familiar task" target clamps to the floor and ties with the neutral one.
    """
    neutral = _target(None, 0.6e9)
    assert _target(0.85, 0.6e9) < neutral < _target(0.10, 0.6e9)


def test_rationale_shows_the_actual_arithmetic():
    _, rationale = compute_curriculum_target(
        zero_shot_baseline=0.3152, n_params=0.6e9, floor=FLOOR, ceiling=CEILING
    )
    assert "novelty 0.685" in rationale
    assert "size factor 1.67" in rationale
    assert "clamped to [5000, 25000]" in rationale


def test_resize_respects_an_explicit_env_override(monkeypatch):
    monkeypatch.setenv("SLM_CURRICULUM_SIZE", "1234")
    state = {"selected_model": None}
    assert resize_curriculum_for_tier(state, log=lambda _m: None) == 1234
    assert state["curriculum_size_target"] == 1234


def test_resize_uses_the_baseline_recorded_for_the_selected_model(monkeypatch):
    monkeypatch.delenv("SLM_CURRICULUM_SIZE", raising=False)

    class _Model:
        selector = "Qwen/Qwen3-0.6B@Q4_K_M"
        label = selector
        params = 0.6e9

    state = {
        "selected_model": _Model(),
        "model_baselines": [
            {"selector": "Qwen/Qwen3-0.6B@Q4_K_M", "baseline_f1": 0.3152},
            {"selector": "other/model", "baseline_f1": 0.90},
        ],
    }
    target = resize_curriculum_for_tier(state, log=lambda _m: None)

    # Must match the standalone formula for THIS model's baseline, not the other entry's.
    assert target == _target(0.3152, 0.6e9)
    assert state["curriculum_size_target"] == target
