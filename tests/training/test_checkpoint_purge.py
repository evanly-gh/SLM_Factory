"""Trainer `checkpoint-*` resume state must not survive a successful iteration.

These dirs hold optimizer.pt + a duplicate adapter + a duplicate tokenizer.json —
~1.2 GB per iteration that is dead the moment final_checkpoint is written. They were
purged only on the fallback-retry path, so successful runs leaked 93 GB across three
runs (see docs/superpowers/specs/2026-07-25-checkpoint-artifact-retention-design.md).
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

from training.lora_trainer import _purge_trainer_checkpoints  # noqa: E402


def _mk(root, name, *files):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    for f in files:
        with open(os.path.join(d, f), "wb") as fh:
            fh.write(b"payload")
    return d


def test_purge_removes_checkpoint_dirs_and_keeps_final_checkpoint(tmp_path):
    root = str(tmp_path)
    _mk(root, "checkpoint-20", "optimizer.pt", "adapter_model.safetensors")
    _mk(root, "checkpoint-160", "optimizer.pt", "tokenizer.json")
    final = _mk(root, "final_checkpoint", "adapter_model.safetensors", "adapter_config.json")

    _purge_trainer_checkpoints(root)

    assert not os.path.exists(os.path.join(root, "checkpoint-20"))
    assert not os.path.exists(os.path.join(root, "checkpoint-160"))
    assert os.path.isfile(os.path.join(final, "adapter_model.safetensors"))
    assert os.path.isfile(os.path.join(final, "adapter_config.json"))


def test_purge_is_a_noop_when_no_checkpoints_exist(tmp_path):
    root = str(tmp_path)
    final = _mk(root, "final_checkpoint", "adapter_model.safetensors")

    _purge_trainer_checkpoints(root)  # must not raise

    assert os.path.isfile(os.path.join(final, "adapter_model.safetensors"))


def test_purge_tolerates_missing_output_dir(tmp_path):
    _purge_trainer_checkpoints(str(tmp_path / "does-not-exist"))  # must not raise


def test_purge_leaves_unrelated_siblings_alone(tmp_path):
    """Only `checkpoint-` prefixed directories go. `final_checkpoint` is not a match."""
    root = str(tmp_path)
    _mk(root, "checkpoint-1", "optimizer.pt")
    _mk(root, "merged", "model.safetensors")
    keep_file = os.path.join(root, "checkpoint-notes.txt")
    with open(keep_file, "w") as fh:
        fh.write("a file, not a dir")

    _purge_trainer_checkpoints(root)

    assert not os.path.exists(os.path.join(root, "checkpoint-1"))
    assert os.path.isfile(os.path.join(root, "merged", "model.safetensors"))
    assert os.path.isfile(keep_file), "a plain file must not be removed"
