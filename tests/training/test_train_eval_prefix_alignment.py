"""The inference prompt must be a strict prefix of the rendered training text (B290).

The invariant, stated once: whatever precedes the answer during training must also be present in the
inference prompt, byte for byte. Break it and the model's first generated tokens are the missing
scaffolding, which then gets scored as its answer.

What made B290 hard to see is that no prompt string in this repo was wrong.
`FastLanguageModel.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")` does not load that repo — it
redirects to `unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit` and returns the mirror's tokenizer,
whose chat template applies the hybrid Qwen3 think-block convention to a thinking-free checkpoint:

    official Qwen/Qwen3-4B-Instruct-2507  -> '<|im_start|>assistant\\nANSWER<|im_end|>\\n'
    unsloth mirror (what training loads)  -> '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nANSWER<|im_end|>\\n'

Meanwhile `merge_for_quantization` pins the OFFICIAL base, so the shipped GGUF is served under the
official template. Fine-tuned predictions came out as `<think></think>[{...}]` and scored 0.0000-0.6120
depending on whether the JSON survived the prefix, against an untrained baseline of 0.8010. The loop
concluded, reasonably from the numbers it had, that fine-tuning never beat zero-shot.

The fix pins the SERVED template for training rather than teaching inference to send Unsloth's, because
the served template is the deployment contract; the alternative hides the skew in our harness and ships
it to the phone.
"""
from types import SimpleNamespace

import pytest

from training.lora_trainer import (
    _assert_train_serve_prefix_alignment,
    _pin_serving_chat_template,
)
from training.slm_helpers import _is_qwen_model_id, _qwen_no_think_prompt

OFFICIAL_4B = "Qwen/Qwen3-4B-Instruct-2507"

# The two renderings at the heart of the bug, as Jinja templates keyed on whether an assistant
# message is present — enough to reproduce the asymmetry without downloading anything.
_THINKING_FREE_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
_THINK_BLOCK_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'assistant' %}<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "{{ m.content }}<|im_end|>\n"
    "{% else %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


class _Tokenizer:
    """Renders `chat_template` with Jinja, like a real tokenizer's apply_chat_template."""

    def __init__(self, chat_template):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        from jinja2 import Template

        return Template(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
        )


# --- the alignment assertion ---------------------------------------------------------------------


def test_the_served_template_passes():
    _assert_train_serve_prefix_alignment(_Tokenizer(_THINKING_FREE_TEMPLATE), OFFICIAL_4B)


def test_the_unsloth_mirror_template_is_rejected():
    """The exact condition that shipped twice: a think block between the prompt and the answer."""
    with pytest.raises(RuntimeError, match="TRAIN/SERVE PREFIX SKEW"):
        _assert_train_serve_prefix_alignment(_Tokenizer(_THINK_BLOCK_TEMPLATE), OFFICIAL_4B)


def test_the_error_names_the_inserted_scaffolding():
    """The message has to point at the think block, or the next reader re-runs my whole diagnosis."""
    with pytest.raises(RuntimeError) as excinfo:
        _assert_train_serve_prefix_alignment(_Tokenizer(_THINK_BLOCK_TEMPLATE), OFFICIAL_4B)
    message = str(excinfo.value)
    assert "<think>" in message
    assert OFFICIAL_4B in message
    assert "Refusing to train" in message


def test_a_hybrid_model_requires_the_think_block():
    """The mirror image: for hybrid Qwen3, inference DOES pre-fill the block, so training must too.
    The thinking-free template is the skew for these models."""
    _assert_train_serve_prefix_alignment(_Tokenizer(_THINK_BLOCK_TEMPLATE), "Qwen/Qwen3-1.7B")
    with pytest.raises(RuntimeError, match="TRAIN/SERVE PREFIX SKEW"):
        _assert_train_serve_prefix_alignment(_Tokenizer(_THINKING_FREE_TEMPLATE), "Qwen/Qwen3-1.7B")


def test_non_qwen_models_are_skipped():
    """`infer_batch_gguf` refuses non-Qwen through the hand-built ChatML path, so there is no prefix
    of ours to compare against; asserting one would fail on templates we never claim to match."""
    _assert_train_serve_prefix_alignment(_Tokenizer(_THINK_BLOCK_TEMPLATE), "google/gemma-3-4b-it")


def test_an_unrenderable_template_does_not_block_training():
    """A template that cannot render is a different failure with its own error path; this guard must
    not convert it into a confusing skew report."""

    class _Broken:
        chat_template = "x"

        def apply_chat_template(self, *a, **k):
            raise ValueError("template is broken")

    _assert_train_serve_prefix_alignment(_Broken(), OFFICIAL_4B)


# --- pinning the served template ------------------------------------------------------------------


def test_a_differing_template_is_replaced_with_the_served_one(monkeypatch):
    loaded = _Tokenizer(_THINK_BLOCK_TEMPLATE)
    served = SimpleNamespace(chat_template=_THINKING_FREE_TEMPLATE)

    import training.lora_trainer as trainer

    monkeypatch.setattr(
        trainer, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: served), raising=False
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: served)),
    )

    result = _pin_serving_chat_template(loaded, OFFICIAL_4B)
    assert result.chat_template == _THINKING_FREE_TEMPLATE
    # And the pinned tokenizer now satisfies the invariant it previously violated.
    _assert_train_serve_prefix_alignment(result, OFFICIAL_4B)


def test_pinning_is_a_noop_when_the_templates_already_agree(monkeypatch):
    loaded = _Tokenizer(_THINKING_FREE_TEMPLATE)
    served = SimpleNamespace(chat_template=_THINKING_FREE_TEMPLATE)
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: served)),
    )
    assert _pin_serving_chat_template(loaded, OFFICIAL_4B).chat_template == _THINKING_FREE_TEMPLATE


def test_an_unreachable_serving_tokenizer_does_not_crash_training(monkeypatch, capsys):
    """Offline is survivable — the alignment assertion is the real gate, and it needs no network."""

    def _raise(*a, **k):
        raise OSError("offline")

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=_raise)),
    )
    loaded = _Tokenizer(_THINK_BLOCK_TEMPLATE)
    assert _pin_serving_chat_template(loaded, OFFICIAL_4B) is loaded
    assert "could not load the serving tokenizer" in capsys.readouterr().out


# --- the inference-side prompt builder is correct as written --------------------------------------


def test_thinking_free_model_gets_a_bare_assistant_prefix():
    assert _qwen_no_think_prompt("P", OFFICIAL_4B).endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"])
def test_hybrid_models_get_the_prefilled_think_block(model_id):
    assert _qwen_no_think_prompt("P", model_id).endswith("<think>\n\n</think>\n\n")


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B", OFFICIAL_4B])
def test_prompt_shape_is_well_formed_chatml(model_id):
    rendered = _qwen_no_think_prompt("PROMPT", model_id)
    assert rendered.startswith("<|im_start|>user\n")
    assert "PROMPT" in rendered
    assert rendered.count("<|im_start|>") == 2
    assert rendered.count("<|im_end|>") == 1


def test_only_qwen_is_routed_through_this_builder():
    assert _is_qwen_model_id(OFFICIAL_4B)
    assert not _is_qwen_model_id("google/gemma-3-4b-it")
    assert not _is_qwen_model_id(None)


# --- the real vendor templates, when they happen to be cached -------------------------------------


@pytest.mark.parametrize("model_id", [OFFICIAL_4B, "Qwen/Qwen3-0.6B"])
def test_real_official_template_satisfies_the_invariant(model_id):
    """Checks the property against the actual vendor template rather than a stand-in. Skipped when the
    tokenizer is not cached, since this asserts something about upstream, not about our code."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - a cold cache is not a failure
        pytest.skip(f"{model_id} tokenizer not cached locally: {exc}")

    _assert_train_serve_prefix_alignment(tokenizer, model_id)


def test_the_unsloth_mirror_really_does_violate_it():
    """Documents the upstream difference as an executable fact, so that if Unsloth ever fixes their
    template this test tells us the workaround can go, instead of it lingering forever."""
    transformers = pytest.importorskip("transformers")
    mirror = "unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(mirror, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{mirror} tokenizer not cached locally: {exc}")

    with pytest.raises(RuntimeError, match="TRAIN/SERVE PREFIX SKEW"):
        _assert_train_serve_prefix_alignment(tokenizer, OFFICIAL_4B)
