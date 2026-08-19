"""The GGUF generation smoke test REPORTS; it must never end a run (B289).

History worth keeping, because the second version of this check was worse than not having it. It began
as a fatal gate on the theory that a GGUF which loads but decodes to garbage was being scored as a real
0.0000. Two things then went wrong:

1. The 0.0000 scores it was written to catch were B290 — a train/serve prefix skew — not corruption. So
   the premise was unfounded.
2. As a fatal check it killed three healthy runs on 2026-08-16 (`38569605` routerbench, `38569606`
   ner_bc5cdr, `38569608` calendar_json), the second of which was already four iterations in. All three
   died because the base `Qwen3-0.6B` Q4_K_M decodes `'////////////////////////////////'` when handed a
   bare `Hello`. That is a small model asked to continue an unformatted string, not a broken artifact —
   and the same runs' fine-tuned 0.6B GGUFs passed the identical check.

Net: three false positives, zero true ones. It now formats its prompt as a chat turn and only warns, so
the eval score stays the arbiter and rollback stays the remedy.
"""
import pytest

from training.quantize import _smoke_test_generation


class _FakeModel:
    """Minimal stand-in for llama_cpp.Llama's __call__ completion interface."""

    def __init__(self, text=None, raises=None):
        self._text = text
        self._raises = raises
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self._raises is not None:
            raise self._raises
        return {"choices": [{"text": self._text}]}


# --- nothing raises ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        "! I'm happy to help you today.",
        " Hi",
        "",
        "   ",
        "</tool_call>",
        "<|im_end|>",
        "////////////////////////////////",  # the exact output that killed three runs
        "!!!...",
    ],
)
def test_no_output_can_end_the_run(output):
    """The property that matters most: whatever comes back, this returns rather than raising."""
    assert _smoke_test_generation(_FakeModel(text=output), "/tmp/m.gguf") == output


def test_a_decode_failure_is_reported_not_raised():
    model = _FakeModel(raises=RuntimeError("llama_decode returned -1"))
    assert _smoke_test_generation(model, "/tmp/m.gguf") is None


# --- what it says --------------------------------------------------------------------------------


def test_degenerate_output_is_reported_as_likely_benign(capsys):
    _smoke_test_generation(_FakeModel(text="////////"), "/path/model-q4_k_m.gguf")
    out = capsys.readouterr().out
    assert "no word characters" in out
    assert "often" in out and "benign" in out, "must not read as an error"
    assert "/path/model-q4_k_m.gguf" in out


def test_healthy_output_is_silent(capsys):
    _smoke_test_generation(_FakeModel(text=" Hello, how can I help?"), "/tmp/m.gguf")
    assert capsys.readouterr().out == ""


def test_markup_wrapped_real_content_is_silent(capsys):
    """Tags AROUND real text are fine — a tool-call wrapper is not degeneracy."""
    _smoke_test_generation(
        _FakeModel(text='<tool_call>{"name": "get_weather"}</tool_call>'), "/tmp/m.gguf"
    )
    assert capsys.readouterr().out == ""


def test_decode_failure_message_points_at_the_eval(capsys):
    _smoke_test_generation(_FakeModel(raises=ValueError("boom")), "/tmp/m.gguf")
    out = capsys.readouterr().out
    assert "could not decode" in out
    assert "near-zero score" in out, "the reader needs to know what actually decides"


# --- how it prompts -----------------------------------------------------------------------------


def test_qwen_is_probed_through_its_chat_template():
    """A raw `Hello` asks the model to continue an unformatted string, which is not how it is ever
    used and is what produced the false positives."""
    model = _FakeModel(text=" Hi there")
    _smoke_test_generation(model, "/tmp/m.gguf", "Qwen/Qwen3-0.6B")
    (prompt, _), = model.calls
    assert prompt.startswith("<|im_start|>user\n")
    assert "Hello" in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")


def test_thinking_free_qwen_gets_its_own_prefix():
    model = _FakeModel(text=" Hi")
    _smoke_test_generation(model, "/tmp/m.gguf", "Qwen/Qwen3-4B-Instruct-2507")
    (prompt, _), = model.calls
    assert prompt.endswith("<|im_start|>assistant\n")


def test_unknown_model_falls_back_to_a_bare_prompt():
    model = _FakeModel(text=" Hi")
    _smoke_test_generation(model, "/tmp/m.gguf", "google/gemma-3-4b-it")
    (prompt, _), = model.calls
    assert prompt == "Hello"


def test_generation_is_greedy_and_cheap():
    """Runs on every build, so it must stay deterministic and short."""
    model = _FakeModel(text=" hello there")
    _smoke_test_generation(model, "/tmp/m.gguf")
    (_, kwargs), = model.calls
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] <= 32
    assert kwargs["echo"] is False


# --- integration with validate_and_record_gguf ---------------------------------------------------


def _fake_llama_module(text):
    import sys

    closed = []

    class _Llama:
        def __init__(self, **kwargs):
            pass

        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": text}]}

        def close(self):
            closed.append(True)

    return type("_M", (), {"Llama": _Llama, "__version__": "0.3.34"}), closed


def test_a_degenerate_artifact_is_still_recorded_as_validated(monkeypatch, tmp_path):
    """The load is the gate, so a model that loads gets its sidecar — otherwise every 0.6B baseline
    would re-quantize on every iteration for no reason."""
    import sys

    import training.quantize as quantize

    gguf = tmp_path / "model-q4_k_m.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    module, closed = _fake_llama_module("////////")
    monkeypatch.setitem(sys.modules, "llama_cpp", module)

    record = quantize.validate_and_record_gguf(str(gguf))
    assert len(record["sha256"]) == 64
    assert quantize.validated_gguf_cache_hit(str(gguf)) is True
    assert closed == [True], "the llama.cpp handle must be closed either way"


def test_the_base_model_is_forwarded_for_prompt_formatting(monkeypatch, tmp_path):
    import sys

    import training.quantize as quantize

    gguf = tmp_path / "model-q4_k_m.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    seen = []

    class _Llama:
        def __init__(self, **kwargs):
            pass

        def __call__(self, prompt, **kwargs):
            seen.append(prompt)
            return {"choices": [{"text": " Hi"}]}

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "llama_cpp", type("_M", (), {"Llama": _Llama, "__version__": "0.3.34"})
    )
    quantize.validate_and_record_gguf(str(gguf), base_model="Qwen/Qwen3-0.6B")
    assert seen and seen[0].startswith("<|im_start|>user\n")


def test_a_file_that_cannot_load_still_fails(monkeypatch, tmp_path):
    """Loosening the smoke test must not loosen the real gate."""
    import sys

    import training.quantize as quantize

    gguf = tmp_path / "model-q4_k_m.gguf"
    gguf.write_bytes(b"truncated")

    class _Llama:
        def __init__(self, **kwargs):
            raise ValueError("missing tensor blk.24.attn_norm.weight")

    monkeypatch.setitem(
        sys.modules, "llama_cpp", type("_M", (), {"Llama": _Llama, "__version__": "0.3.34"})
    )
    with pytest.raises(RuntimeError, match=r"blk\.24\.attn_norm\.weight"):
        quantize.validate_and_record_gguf(str(gguf))
