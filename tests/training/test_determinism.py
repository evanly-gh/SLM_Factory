# tests/training/test_determinism.py
"""The reproducibility contract.

Probe job 40222458 trained one 151-row file twice in a single process with identical arguments and
produced two different adapters, because seeding was in place but kernel reduction order was not.
These tests pin the settings that closed that gap, and the properties a paper depends on: the seed
is explicit, the seed stream is stable across processes, and nothing silently degrades to
"unreproducible" without saying so.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

import training.determinism as det


def test_default_seed_matches_unsloths_own(monkeypatch):
    """Adopting 3407 means switching determinism on does not change any existing run's init."""
    monkeypatch.delenv(det.SEED_ENV, raising=False)
    assert det.configured_seed() == 3407 == det.DEFAULT_SEED


def test_seed_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(det.SEED_ENV, "12345")
    assert det.configured_seed() == 12345


def test_a_malformed_seed_raises_rather_than_guessing(monkeypatch):
    monkeypatch.setenv(det.SEED_ENV, "not-a-number")
    with pytest.raises(ValueError, match=det.SEED_ENV):
        det.configured_seed()


def test_unknown_mode_is_refused_at_startup(monkeypatch):
    """Better to fail submitting than to discover at hour 20 that the mode was a typo."""
    monkeypatch.setenv(det.MODE_ENV, "sort-of")
    with pytest.raises(ValueError, match=det.MODE_ENV):
        det.configured_mode()


def test_cublas_workspace_is_set_but_never_overrides_an_operator(monkeypatch):
    monkeypatch.delenv(det.CUBLAS_ENV, raising=False)
    det.set_cublas_workspace_config()
    assert os.environ[det.CUBLAS_ENV] == det.CUBLAS_DETERMINISTIC

    monkeypatch.setenv(det.CUBLAS_ENV, ":16:8")
    det.set_cublas_workspace_config()
    assert os.environ[det.CUBLAS_ENV] == ":16:8"


def test_enable_determinism_pins_torch_and_reports_what_it_did(monkeypatch):
    monkeypatch.delenv(det.MODE_ENV, raising=False)
    monkeypatch.setenv(det.SEED_ENV, "777")
    torch = MagicMock()
    torch.__version__ = "2.test"
    torch.cuda.is_available.return_value = True
    torch.cuda.is_initialized.return_value = False

    with patch.dict(sys.modules, {"torch": torch}):
        record = det.enable_determinism(log=lambda *a, **k: None)

    assert record["seed"] == 777
    assert record["mode"] == "warn"
    assert record["cublas"] == det.CUBLAS_DETERMINISTIC
    torch.manual_seed.assert_called_once_with(777)
    torch.cuda.manual_seed_all.assert_called_once_with(777)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    # warn_only follows the mode: `warn` must not raise mid-run, `strict` must.
    torch.use_deterministic_algorithms.assert_called_once_with(True, warn_only=True)


def test_strict_mode_asks_torch_to_raise(monkeypatch):
    monkeypatch.setenv(det.MODE_ENV, "strict")
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": torch}):
        det.enable_determinism(log=lambda *a, **k: None)
    torch.use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)


def test_off_mode_says_so_out_loud(monkeypatch):
    """A run that is not reproducible must announce it, or the log implies a guarantee it lacks."""
    monkeypatch.setenv(det.MODE_ENV, "off")
    lines = []
    record = det.enable_determinism(log=lines.append)
    assert record["seed"] is None
    assert any("not reproducible" in line for line in lines)


def test_already_initialised_cuda_is_reported_not_hidden(monkeypatch):
    """CUBLAS_WORKSPACE_CONFIG read after the first handle is created does nothing, silently."""
    monkeypatch.delenv(det.MODE_ENV, raising=False)
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.is_initialized.return_value = True
    with patch.dict(sys.modules, {"torch": torch}):
        record = det.enable_determinism(log=lambda *a, **k: None)
    assert any("already initialised" in note for note in record["notes"])


def test_seed_stream_is_reproducible_diverse_and_namespaced(monkeypatch):
    """The property synthesis needs: same rows across runs, different rows within a run."""
    monkeypatch.setenv(det.SEED_ENV, "3407")
    run1 = det.seed_sequence("synth:teacher")
    first = [run1.next() for _ in range(5)]
    # A second run of the same configuration builds its own sequence from scratch.
    run2 = det.seed_sequence("synth:teacher")
    second = [run2.next() for _ in range(5)]
    assert first == second, "a rerun must regenerate the same rows"
    assert len(set(first)) == 5, "one fixed seed per row would collapse teacher diversity"
    assert run1.drawn == 5
    judge = det.seed_sequence("judge:teacher")
    other = [judge.next() for _ in range(5)]
    assert other != first, "two callers on one run seed must not draw the same stream"


def test_seed_stream_does_not_depend_on_python_hash_randomisation():
    """`hash(str)` is salted per interpreter, so using it would break cross-process reruns."""
    src = open(det.__file__, encoding="utf-8").read()
    assert "zlib.crc32" in src
    assert "abs(hash(" not in src


def test_seed_and_mode_are_in_the_resume_fingerprint():
    """A resumed segment that changed either would publish numbers from two different streams."""
    from agent.checkpoint import _RESUME_ENV_DEFAULTS

    assert _RESUME_ENV_DEFAULTS[det.SEED_ENV] == str(det.DEFAULT_SEED)
    assert _RESUME_ENV_DEFAULTS[det.MODE_ENV] == "warn"
