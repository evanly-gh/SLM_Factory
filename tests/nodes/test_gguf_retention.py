"""A GGUF survives only if its iteration set a new best for the tier.

The `(weights_ref, quant)` cache key is unique per iteration, so the GGUF cache can
never hit — measured 0/138 (NER) and 2/66 (math). Every 2.6 GB file was written, read
once and kept forever, costing 193 GB across three runs. See
docs/superpowers/specs/2026-07-25-checkpoint-artifact-retention-design.md.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

from agent.nodes.evaluate import _reap_gguf  # noqa: E402
from training.quantize import gguf_validation_sidecar_path  # noqa: E402


def _mk_gguf(root, name):
    d = os.path.join(str(root), name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "model-q4_k_m.gguf")
    with open(path, "wb") as fh:
        fh.write(b"gguf-bytes")
    with open(gguf_validation_sidecar_path(path), "w") as fh:
        fh.write("{}")
    return path


def test_new_best_gguf_is_retained_and_others_reaped(tmp_path):
    winner = _mk_gguf(tmp_path, "aaaa")
    loser = _mk_gguf(tmp_path, "bbbb")
    state = {}

    _reap_gguf(state, [winner, loser], keep_path=winner)

    assert os.path.isfile(winner), "the new best must survive"
    assert os.path.isfile(gguf_validation_sidecar_path(winner))
    assert not os.path.exists(loser), "a non-improving GGUF must be reaped"
    assert not os.path.exists(gguf_validation_sidecar_path(loser)), (
        "the sidecar must go too, or a stale validation record could fake a cache hit"
    )
    assert winner in state["retained_gguf_paths"]


def test_no_new_best_reaps_everything_built_this_iteration(tmp_path):
    a = _mk_gguf(tmp_path, "aaaa")
    b = _mk_gguf(tmp_path, "bbbb")
    state = {}

    _reap_gguf(state, [a, b], keep_path=None)

    assert not os.path.exists(a)
    assert not os.path.exists(b)
    assert state["retained_gguf_paths"] == []


def test_previously_retained_gguf_is_never_reaped(tmp_path):
    """Earlier new-bests stay on disk — the policy keeps every score-improving model."""
    old_best = _mk_gguf(tmp_path, "aaaa")
    new_best = _mk_gguf(tmp_path, "bbbb")
    state = {"retained_gguf_paths": [old_best]}

    _reap_gguf(state, [old_best, new_best], keep_path=new_best)

    assert os.path.isfile(old_best), "a prior new-best must not be reaped"
    assert os.path.isfile(new_best)
    assert set(state["retained_gguf_paths"]) == {old_best, new_best}


def test_reaping_removes_the_now_empty_directory(tmp_path):
    loser = _mk_gguf(tmp_path, "bbbb")

    _reap_gguf({}, [loser], keep_path=None)

    assert not os.path.isdir(os.path.dirname(loser))


def test_reap_ignores_none_entries_and_missing_files(tmp_path):
    """gguf_path is None whenever quantized eval is off; that must not raise."""
    state = {}
    _reap_gguf(state, [None, os.path.join(str(tmp_path), "gone", "x.gguf")], keep_path=None)
    assert state["retained_gguf_paths"] == []


def test_retained_paths_are_deduplicated_across_iterations(tmp_path):
    best = _mk_gguf(tmp_path, "aaaa")
    state = {}

    _reap_gguf(state, [best], keep_path=best)
    _reap_gguf(state, [best], keep_path=best)

    assert state["retained_gguf_paths"] == [best]
    assert os.path.isfile(best)
