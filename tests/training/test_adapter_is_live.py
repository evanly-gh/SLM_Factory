"""B254: an attached-but-zeroed LoRA adapter must fail loudly, not score the base model.

`model.load_adapter(path)` on an Unsloth-patched model built the LoRA modules and marked them
active while leaving every lora_B tensor at its zero init. Since LoRA computes B @ A and B is
zero-initialised, the adapter was exactly an identity: eval produced the BASE model's logits
and the run recorded them as a fine-tuned score. Four consecutive tier-2 iterations in run
38303490 returned 0.5414 with failures=392/800 — identical to that tier's own zero-shot
baseline — across three datasets and three hyperparameter configs.
"""
import os

import pytest
import torch

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

from training.slm_helpers import _assert_adapter_is_live


class _FakeModel:
    def __init__(self, named):
        self._named = named

    def named_parameters(self):
        return iter(self._named)


def _lora_pair(b_values):
    return [
        ("base.layers.0.q_proj.lora_A.weight", torch.ones(4, 4)),
        ("base.layers.0.q_proj.lora_B.weight", torch.tensor(b_values, dtype=torch.float32)),
    ]


def test_trained_adapter_passes():
    model = _FakeModel(_lora_pair([[0.0, 0.3], [0.1, 0.0]]))
    _assert_adapter_is_live(model, "/tmp/ckpt")  # must not raise


def test_all_zero_lora_b_is_rejected():
    """The exact production failure: modules attached, weights never populated."""
    model = _FakeModel(_lora_pair([[0.0, 0.0], [0.0, 0.0]]))

    with pytest.raises(RuntimeError) as excinfo:
        _assert_adapter_is_live(model, "/tmp/ckpt")

    message = str(excinfo.value)
    assert "ZERO" in message
    assert "identity" in message
    assert "/tmp/ckpt" in message


def test_missing_adapter_modules_are_rejected():
    model = _FakeModel([("base.layers.0.q_proj.weight", torch.ones(4, 4))])

    with pytest.raises(RuntimeError) as excinfo:
        _assert_adapter_is_live(model, "/tmp/ckpt")

    assert "NO lora_B" in str(excinfo.value)


def test_a_single_nonzero_tensor_is_enough():
    """Some layers legitimately train to near-zero; one live tensor proves the load worked."""
    named = _lora_pair([[0.0, 0.0], [0.0, 0.0]]) + [
        ("base.layers.1.v_proj.lora_B.weight", torch.tensor([[0.0, 0.05]])),
    ]
    _assert_adapter_is_live(_FakeModel(named), "/tmp/ckpt")


def test_inference_loader_uses_peft_not_load_adapter():
    """Guard the fix itself: load_adapter silently no-ops on the Unsloth vision path."""
    source = open("training/slm_helpers.py", encoding="utf-8").read()

    assert "PeftModel.from_pretrained(model, weights_ref)" in source
    assert "model.load_adapter(weights_ref)" not in source
    assert "_assert_adapter_is_live(model, weights_ref)" in source
