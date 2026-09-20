"""Teacher-baseline accuracy goal (2026-08-01; five-shot and shared with the gate, 2026-08-29).

The accuracy target a run must beat is the reference teacher's own FIVE-SHOT score on THIS run's
frozen E, floored at 0.8. task_analysis parks the goal PENDING (E does not exist yet); eval_setup
completes it once E is built, from the measurement `_measure_teacher` has ALREADY taken for the
synthesis gate rather than from a second pass of its own.

These tests pin the floor/cap math, the standalone `measure_endpoint_baseline` helper (still used
by `scripts/probe_teacher_fewshot.py`, no longer by the pipeline), the reuse of the single
measurement, what the calibration record has to remember about how it was taken, the
unmeasurable-teacher HARD FAILURE (there is no fallback — the run raises and stops), and the
park/complete handoff.

The `qwen_baseline` / `pending_qwen_baseline` source strings and the `measured_qwen` field keep
their names despite the teacher now being configurable: they are persisted checkpoint schema, and
renaming them would break resume against every checkpoint already on disk. `measured_model`
records which teacher the number actually came from.
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
    return EvalSet(all=rows, task="clinc150")


def test_measure_endpoint_baseline_scores_with_injected_generate_fn():
    from eval.endpoint_eval import measure_endpoint_baseline
    from eval.harness import EvalResult

    # A reference that always answers "spam" scores the single spam row perfectly.
    def fake_generate(prompt, temperature=0.0, max_tokens=50):
        return "spam"

    result = measure_endpoint_baseline(
        _classification_eval_set(), generate_fn=fake_generate, log=lambda *a: None
    )
    assert isinstance(result, EvalResult)
    assert result.f1 == pytest.approx(1.0)
    assert result.metric == "macro_f1"


def test_measure_endpoint_baseline_returns_none_when_endpoint_unreachable(monkeypatch):
    import data.synth_client as synth
    from eval.endpoint_eval import measure_endpoint_baseline

    monkeypatch.setattr(synth, "get_generate_fn", lambda *a, **k: None)
    result = measure_endpoint_baseline(
        _classification_eval_set(), generate_fn=None, log=lambda *a: None
    )
    assert result is None


def test_measure_endpoint_baseline_survives_a_failing_row():
    """ONE bad row scores as an empty output rather than aborting a whole baseline.

    The stub must fail on exactly one row, not all of them: a generate_fn that fails on everything is
    a different situation with a different correct answer, covered below (B313).
    """
    from eval.endpoint_eval import measure_endpoint_baseline

    # A dedicated multi-row set: on the single-row shared fixture "one bad row" IS every row, which is
    # the situation the companion test covers.
    many = EvalSet(
        all=[{"text": f"utterance {i}", "label": "spam"} for i in range(20)], task="clinc150",
    )
    calls = {"n": 0}

    def boom_once(prompt, temperature=0.0, max_tokens=50):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("endpoint blip")
        return "spam"

    result = measure_endpoint_baseline(many, generate_fn=boom_once, log=lambda *a: None)
    assert result is not None
    assert 0.0 <= result.f1 <= 1.0


def test_a_baseline_whose_every_row_failed_is_refused_rather_than_scored_zero():
    """A score built from nothing but failures describes the harness, not the model.

    On run 38661753 the task NAME was passed into the `generate_fn` positional slot, so all 1,000 eval
    rows raised `'str' object is not callable`. The baseline reported 0.0000, and
    `threshold_from_endpoint_baseline` floored the accuracy goal at 0.80 while stating that the teacher
    had scored zero — a claim about a measurement that never happened. Refusing is the only honest
    outcome, because 0.0 from a broken harness and 0.0 from an incapable teacher are indistinguishable
    in the report (B313).
    """
    import pytest

    from eval.endpoint_eval import BaselineGenerationError, measure_endpoint_baseline

    def always_broken(prompt, temperature=0.0, max_tokens=50):
        raise RuntimeError("endpoint died")

    with pytest.raises(BaselineGenerationError) as excinfo:
        measure_endpoint_baseline(
            _classification_eval_set(), generate_fn=always_broken, log=lambda *a: None
        )
    # The message must name the failure rate and the underlying error, or the next reader is left
    # guessing at exactly the point the old behaviour left them guessing.
    assert "endpoint died" in str(excinfo.value)
    assert "100%" in str(excinfo.value)


def test_the_task_name_in_the_generate_fn_slot_is_rejected_immediately():
    """The specific mistake that caused B313, caught before a single call is made.

    `generate_fn` is the second POSITIONAL parameter, so a caller that believes it is passing a task
    lands a string there and every row fails identically. `run_eval` dropped its own task argument in
    the same refactor, which is why four call sites made this mistake at once.
    """
    import pytest

    from eval.endpoint_eval import measure_endpoint_baseline

    with pytest.raises(TypeError) as excinfo:
        measure_endpoint_baseline(_classification_eval_set(), "clinc150", log=lambda *a: None)
    assert "callable generate_fn" in str(excinfo.value)


# --- task_analysis parks the goal PENDING --------------------------------------

def test_task_analysis_parks_pending_qwen_baseline(monkeypatch):
    from agent.nodes.cold_start.task_analysis import _calibrate_stop_threshold
    from agent.threshold import UNREACHABLE_PENDING_THRESHOLD

    monkeypatch.delenv("SLM_STOP_THRESHOLD", raising=False)
    state = {"task_plan": {"benchmark": "CLINC150"}}
    _calibrate_stop_threshold(state, "clinc150")

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
    _calibrate_stop_threshold(state, "clinc150")
    assert state["stop_threshold"] == pytest.approx(0.5)
    assert state["threshold_calibration"]["source"] == "env_override"


# --- eval_setup completes the goal once E exists -------------------------------

def _measured_verdict(**overrides):
    """The shape `agent/teacher_fitness.measure_teacher_fitness` records on state."""
    verdict = {
        "status": "measured",
        "score": 0.88,
        "format_valid": 0.97,
        "metric": "macro_f1",
        "shots": 5,
        "shots_requested": 5,
        "n": 1000,
        "model": "Qwen/Qwen3.6-35B-A3B",
        "endpoint": "http://127.0.0.1:8000/v1",
        "threshold": 0.8,
        "synthesis_allowed": True,
    }
    verdict.update(overrides)
    return verdict


def _pending_state(**overrides):
    state = {
        "task": "clinc150",
        "teacher_fitness": _measured_verdict(),
        "threshold_calibration": {"source": "pending_qwen_baseline", "pending": True,
                                  "floor": 0.8},
    }
    state.update(overrides)
    return state


def test_eval_setup_completes_goal_from_measured_baseline():
    import agent.nodes.cold_start.eval_setup as eval_setup

    state = _pending_state()
    eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())

    assert state["threshold_calibration"]["source"] == "qwen_baseline"
    assert state["threshold_calibration"]["pending"] is False
    assert state["threshold_calibration"]["measured_qwen"] == pytest.approx(0.88)
    assert state["stop_threshold"] == pytest.approx(0.88)
    assert state["initial_stop_threshold"] == pytest.approx(0.88)


def test_the_goal_reuses_the_fitness_measurement_and_takes_no_second_pass(monkeypatch):
    """ONE measurement, two consumers — the synthesis gate and the accuracy goal.

    They used to be measured separately: the gate five-shot on 200 rows, the goal ZERO-shot on all
    1,000, so a run scored its own teacher twice with prompts that disagree by multiples on a
    format-bound task (BC5CDR: 0.1131 against 0.7190) and set the goal from the shape nothing else
    in the pipeline ever sends. Calibration must now read the recorded verdict and call nothing.
    """
    import agent.nodes.cold_start.eval_setup as eval_setup
    import eval.endpoint_eval as endpoint_eval

    def boom(*a, **k):
        raise AssertionError("calibration must not take a second teacher measurement")

    monkeypatch.setattr(endpoint_eval, "measure_endpoint_baseline", boom)
    monkeypatch.setattr(eval_setup, "measure_endpoint_baseline", boom, raising=False)

    state = _pending_state()
    eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())
    assert state["stop_threshold"] == pytest.approx(0.88)


def test_the_goal_records_how_the_teacher_was_measured():
    """The goal is only interpretable with the model, the shot count and the format validity
    beside it: 0.88 five-shot at format_valid 0.97 and 0.88 zero-shot at 0.40 are different
    claims about different things, and only one of them is about capability."""
    import agent.nodes.cold_start.eval_setup as eval_setup

    state = _pending_state()
    eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())

    calibration = state["threshold_calibration"]
    assert calibration["measured_model"] == "Qwen/Qwen3.6-35B-A3B"
    assert calibration["measured_shots"] == 5
    assert calibration["measured_n"] == 1000
    assert calibration["measured_format_valid"] == pytest.approx(0.97)
    assert calibration["measured_metric"] == "macro_f1"


def test_a_weak_but_measured_teacher_is_still_floored_not_rejected():
    """The floor exists so a weak reference cannot set a trivially-low goal. It must still be
    distinguishable from a teacher that genuinely set the goal — hence `floored`."""
    import agent.nodes.cold_start.eval_setup as eval_setup

    state = _pending_state(teacher_fitness=_measured_verdict(score=0.0999,
                                                             synthesis_allowed=False))
    eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())

    assert state["stop_threshold"] == pytest.approx(0.8)
    assert state["threshold_calibration"]["floored"] is True
    assert state["threshold_calibration"]["measured_qwen"] == pytest.approx(0.0999)


def test_eval_setup_raises_when_the_teacher_was_never_measured():
    """The teacher baseline is the SOLE accuracy target — an unmeasurable teacher has no honest
    fallback, so eval_setup raises and breaks the loop instead of degrading to the floor.

    Degrading would be worse here than it looks: the floor is 0.80, so an unreachable endpoint
    would silently produce a plausible-looking goal that no measurement supports.
    """
    import agent.nodes.cold_start.eval_setup as eval_setup
    from agent.nodes.cold_start.eval_setup import QwenBaselineUnavailableError

    for verdict in (
        None,
        {"status": "unmeasured", "reason": "synthesis endpoint unreachable"},
        {"status": "unmeasured", "reason": "generation failed on 1000 of 1000 eval row(s)"},
    ):
        state = _pending_state(teacher_fitness=verdict)
        with pytest.raises(QwenBaselineUnavailableError):
            eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())


def test_eval_setup_ignores_non_qwen_calibration(monkeypatch):
    import agent.nodes.cold_start.eval_setup as eval_setup

    state = {"task": "clinc150",
             "threshold_calibration": {"source": "registry", "pending": False}}
    eval_setup._calibrate_qwen_goal_if_pending(state, _classification_eval_set())
    # Untouched.
    assert state["threshold_calibration"]["source"] == "registry"
    assert "stop_threshold" not in state
