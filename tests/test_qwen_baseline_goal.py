"""Qwen-3.6-baseline accuracy goal (2026-08-01).

The accuracy target a run must beat is the separately-hosted Qwen-3.6 reference model's own
zero-shot score on THIS run's frozen E, floored at 0.8. task_analysis parks the goal PENDING
(E does not exist yet); eval_setup measures it once E is built. These tests pin the floor/cap
math, the measurement path (mocked endpoint), the unreachable-endpoint HARD FAILURE (there is
no fallback — the run raises and stops), and the park/complete handoff.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

import pytest  # noqa: E402

from agent.threshold import (  # noqa: E402
    THRESHOLD_CEILING,
    threshold_from_endpoint_baseline,
)
from data.eval_set import EvalSet  # noqa: E402


# --- floor / cap math ----------------------------------------------------------

def test_measured_above_floor_is_used():
    threshold, reason = threshold_from_endpoint_baseline(0.91, floor=0.8)
    assert threshold == pytest.approx(0.91)
    assert "0.9100" in reason and "0.80" in reason


def test_measured_below_floor_snaps_to_floor():
    threshold, _ = threshold_from_endpoint_baseline(0.42, floor=0.8)
    assert threshold == pytest.approx(0.8)


def test_unreachable_measured_as_zero_snaps_to_floor():
    threshold, _ = threshold_from_endpoint_baseline(0.0, floor=0.8)
    assert threshold == pytest.approx(0.8)


def test_never_targets_a_perfect_score():
    threshold, _ = threshold_from_endpoint_baseline(1.0, floor=0.8)
    assert threshold == pytest.approx(THRESHOLD_CEILING)
    assert threshold < 1.0


def test_non_numeric_measured_falls_back_to_floor():
    threshold, _ = threshold_from_endpoint_baseline(None, floor=0.8)
    assert threshold == pytest.approx(0.8)


# --- measurement path (mocked endpoint) ----------------------------------------

def _classification_eval_set():
    rows = [{"text": "win a free prize now", "label": "spam"}]
    return EvalSet(all=rows, task_type="classification")


def test_measure_endpoint_baseline_scores_with_injected_generate_fn():
    from eval.endpoint_eval import measure_endpoint_baseline
    from eval.harness import EvalResult

    # A reference that always answers "spam" scores the single spam row perfectly.
    def fake_generate(prompt, temperature=0.0, max_tokens=50):
        return "spam"

    result = measure_endpoint_baseline(
        _classification_eval_set(), "classification", generate_fn=fake_generate, log=lambda *a: None
    )
    assert isinstance(result, EvalResult)
    assert result.f1 == pytest.approx(1.0)
    assert result.metric == "macro_f1"


def test_measure_endpoint_baseline_returns_none_when_endpoint_unreachable(monkeypatch):
    import data.synth_client as synth
    from eval.endpoint_eval import measure_endpoint_baseline

    monkeypatch.setattr(synth, "get_generate_fn", lambda *a, **k: None)
    result = measure_endpoint_baseline(
        _classification_eval_set(), "classification", generate_fn=None, log=lambda *a: None
    )
    assert result is None


def test_measure_endpoint_baseline_survives_a_failing_row():
    from eval.endpoint_eval import measure_endpoint_baseline

    def boom(prompt, temperature=0.0, max_tokens=50):
        raise RuntimeError("endpoint blip")

    # One bad row must not abort the measurement; it scores as an empty output.
    result = measure_endpoint_baseline(
        _classification_eval_set(), "classification", generate_fn=boom, log=lambda *a: None
    )
    assert result is not None
    assert 0.0 <= result.f1 <= 1.0


# --- task_analysis parks the goal PENDING --------------------------------------

def test_task_analysis_parks_pending_qwen_baseline(monkeypatch):
    from agent.nodes.cold_start.task_analysis import _calibrate_stop_threshold
    from agent.threshold import UNREACHABLE_PENDING_THRESHOLD

    monkeypatch.delenv("SLM_STOP_THRESHOLD", raising=False)
    state = {"task_plan": {"benchmark": "CLINC150"}}
    _calibrate_stop_threshold(state, "classification")

    cal = state["threshold_calibration"]
    assert cal["source"] == "pending_qwen_baseline"
    assert cal["pending"] is True
    assert cal["floor"] == pytest.approx(0.8)
    assert state["stop_threshold"] == UNREACHABLE_PENDING_THRESHOLD
    assert state["initial_stop_threshold"] == UNREACHABLE_PENDING_THRESHOLD


def test_env_override_still_wins_over_qwen(monkeypatch):
    from agent.nodes.cold_start.task_analysis import _calibrate_stop_threshold

    monkeypatch.setenv("SLM_STOP_THRESHOLD", "0.5")
    state = {"task_plan": {}}
    _calibrate_stop_threshold(state, "classification")
    assert state["stop_threshold"] == pytest.approx(0.5)
    assert state["threshold_calibration"]["source"] == "env_override"


# --- eval_setup completes the goal once E exists -------------------------------

def _pending_state():
    return {
        "task_type": "classification",
        "threshold_calibration": {"source": "pending_qwen_baseline", "pending": True,
                                  "floor": 0.8},
    }


def test_eval_setup_completes_goal_from_measured_baseline(monkeypatch):
    import agent.nodes.cold_start.eval_setup as eval_setup
    from eval.harness import EvalResult

    fake = EvalResult(f1=0.88, per_class={}, failures=[], metric="macro_f1")
    monkeypatch.setattr(eval_setup, "measure_endpoint_baseline",
                        lambda *a, **k: fake, raising=False)

    state = _pending_state()
    eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())

    assert state["threshold_calibration"]["source"] == "qwen_baseline"
    assert state["threshold_calibration"]["pending"] is False
    assert state["threshold_calibration"]["measured_qwen"] == pytest.approx(0.88)
    assert state["stop_threshold"] == pytest.approx(0.88)
    assert state["initial_stop_threshold"] == pytest.approx(0.88)


def test_eval_setup_raises_when_endpoint_unreachable(monkeypatch):
    """The Qwen baseline is the SOLE accuracy target — an unreachable endpoint has no honest
    fallback, so eval_setup raises and breaks the loop instead of degrading to the floor."""
    import agent.nodes.cold_start.eval_setup as eval_setup
    from agent.nodes.cold_start.eval_setup import QwenBaselineUnavailableError

    monkeypatch.setattr(eval_setup, "measure_endpoint_baseline",
                        lambda *a, **k: None, raising=False)
    state = _pending_state()
    with pytest.raises(QwenBaselineUnavailableError):
        eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())


def test_eval_setup_raises_when_measurement_errors(monkeypatch):
    """A measurement exception is also fatal — wrapped as QwenBaselineUnavailableError."""
    import agent.nodes.cold_start.eval_setup as eval_setup
    from agent.nodes.cold_start.eval_setup import QwenBaselineUnavailableError

    def boom(*a, **k):
        raise RuntimeError("endpoint refused connection")

    monkeypatch.setattr(eval_setup, "measure_endpoint_baseline", boom, raising=False)
    state = _pending_state()
    with pytest.raises(QwenBaselineUnavailableError):
        eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())


def test_eval_setup_ignores_non_qwen_calibration(monkeypatch):
    import agent.nodes.cold_start.eval_setup as eval_setup

    state = {"task_type": "classification",
             "threshold_calibration": {"source": "registry", "pending": False}}
    eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())
    # Untouched.
    assert state["threshold_calibration"]["source"] == "registry"
    assert "stop_threshold" not in state
