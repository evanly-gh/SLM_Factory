"""Synth-fill is GONE, and the curriculum floor replaces it (2026-08-16).

Synth-fill padded every curriculum up to `target_rows` with generated rows. It was removed because:

  * the size target is itself a heuristic (a function of the zero-shot baseline and parameter count),
    so "reach the target" was never a real requirement;
  * BC5CDR converged at 0.8098 — the project's best result — on a GOLD-ONLY curriculum that ran
    ~7,100 rows BELOW its 8,929-row target;
  * the SFT literature is consistent that a small clean set beats a large noisy one (LIMA;
    AlpaGasus, where 9k filtered rows beat 52k unfiltered);
  * it consumed most of the teacher budget — one traced rebuild spent 749 generations to keep 225
    rows, of which 64 survived quality control.

What replaces it is a VIABILITY FLOOR: the curriculum is whatever real data supplies, and if that is
implausibly small the run fails loudly instead of training on it and reporting a number nobody should
trust. Targeted synthesis (the `synthesize` strategy and surgical confusion-pair rows) is unaffected.
"""
import pytest

import agent.nodes.curate as curate


def test_synth_fill_is_removed():
    """The function is gone, not merely unused — a dead padding path is an invitation to re-enable
    it without re-reading why it was removed."""
    assert not hasattr(curate, "_synth_fill_to_target")


def test_curriculum_floor_accepts_a_viable_dataset():
    rows = [{"text": f"r{i}", "label": "a"} for i in range(curate.MIN_CURRICULUM_ROWS)]
    # Well below target, far above the floor: the normal, expected state post-removal.
    curate._enforce_curriculum_floor(rows, model_id="m", target_rows=9000)


def test_curriculum_floor_rejects_an_implausibly_small_dataset():
    rows = [{"text": f"r{i}", "label": "a"} for i in range(10)]
    with pytest.raises(RuntimeError, match="below the .* floor"):
        curate._enforce_curriculum_floor(rows, model_id="m", target_rows=5000)


def test_curriculum_floor_error_names_the_actual_cause():
    """The failure is always upstream — a loader returning nothing, or QC removing nearly
    everything — so the message has to point there rather than at the floor check."""
    with pytest.raises(RuntimeError) as exc:
        curate._enforce_curriculum_floor([], model_id="m", target_rows=5000)
    message = str(exc.value)
    assert "loader" in message and "[qc]" in message
    assert "SLM_MIN_CURRICULUM_ROWS" in message


def test_curriculum_floor_is_env_tunable(monkeypatch):
    """A deliberately tiny curriculum (a smoke run, a toy task) must remain possible."""
    import importlib

    monkeypatch.setenv("SLM_MIN_CURRICULUM_ROWS", "5")
    reloaded = importlib.reload(curate)
    try:
        assert reloaded.MIN_CURRICULUM_ROWS == 5
        reloaded._enforce_curriculum_floor(
            [{"text": "r", "label": "a"}] * 5, model_id="m", target_rows=100
        )
    finally:
        monkeypatch.delenv("SLM_MIN_CURRICULUM_ROWS", raising=False)
        importlib.reload(curate)


def test_default_floor_is_500():
    assert curate.MIN_CURRICULUM_ROWS == 500
