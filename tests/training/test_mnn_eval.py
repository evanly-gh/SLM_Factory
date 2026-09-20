# tests/training/test_mnn_eval.py
"""Scoring an MNN artifact: the pymnn inference path `eval/harness.py` routes to.

The counterpart of `tests/training/test_gguf_eval_concurrency.py` for the other backend. What is
pinned here is everything that would produce a WRONG NUMBER rather than an error — a stale KV cache
carried between rows, a prompt rendered differently from the GGUF path, a prompt silently truncated
to fit the context — because none of those fail, they just quietly change what was measured.
"""
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import training.slm_helpers as helpers


class FakeMnnLlm:
    """A pymnn `Llm` stand-in that records what was asked of it, in order."""

    def __init__(self, replies=None, tokens_per_prompt=10):
        self._c_obj = object()  # marks this as the Python wrapper shape
        self.config = {}
        self.calls = []
        self.events = []
        self.replies = list(replies or [])
        self.tokens_per_prompt = tokens_per_prompt
        self.loaded = False

    def set_config(self, config):
        assert isinstance(config, dict), "the wrapper takes a dict; the extension takes a string"
        self.config.update(config)
        self.events.append(("set_config", dict(config)))
        return True

    def load(self):
        self.loaded = True
        self.events.append(("load", None))

    def reset(self):
        self.events.append(("reset", None))

    def tokenizer_encode(self, prompt):
        return list(range(self.tokens_per_prompt))

    def response(self, prompt, stream=False):
        self.calls.append(prompt)
        self.events.append(("response", prompt))
        if self.replies:
            return self.replies.pop(0)
        return f"reply-{len(self.calls)}"

    @property
    def context(self):
        return SimpleNamespace(
            prompt_len=self.tokens_per_prompt, gen_seq_len=4,
            prefill_us=1000, decode_us=2000,
        )


@pytest.fixture(autouse=True)
def _clean_mnn_cache():
    helpers._mnn_cache.clear()
    helpers._mnn_cache_order.clear()
    yield
    helpers._mnn_cache.clear()
    helpers._mnn_cache_order.clear()


def _artifact(tmp_path, name="model-mnn-q4", quant_bit=4):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "llm.mnn").write_bytes(b"graph")
    (directory / "llm.mnn.weight").write_bytes(b"weights")
    (directory / "tokenizer.mtok").write_text("tok")
    (directory / "llm_config.json").write_text("{}")
    (directory / "config.json").write_text(json.dumps({"llm_model": "llm.mnn"}))
    # The bit width is read back from here to choose CUDA's memory mode, so a fixture without it
    # would exercise the "unknown width" branch while looking like it exercised the real one.
    (directory / "export_args.json").write_text(json.dumps({
        "quant_bit": quant_bit, "lm_quant_bit": max(8, quant_bit),
    }))
    return str(directory)


# ── Loading ─────────────────────────────────────────────────────────────────────────────────────

def test_load_configures_greedy_decoding_and_the_task_context(tmp_path):
    """MNN's shipped defaults are a chat app's; scoring under them would randomise every eval.

    `sampler_type: mixed` with temperature 0.8 and top_k 40 is what `llmexport.py` writes into
    config.json. Greedy is the MNN spelling of the GGUF path's `temperature=0.0`, and
    `max_all_tokens` has to be raised from MNN's 2048 default because it sizes the KV cache.
    """
    fake = FakeMnnLlm()
    artifact = _artifact(tmp_path)
    with patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(
        create=lambda path, embedding_model=False: fake
    )):
        llm = helpers.load_mnn_llm(artifact, max_seq_length=4096, max_new_tokens=50)

    assert llm is fake
    assert fake.config["sampler_type"] == "greedy"
    # The prompt is rendered by `_serving_prompt_prefix` before it gets here, shared with the GGUF
    # path; letting MNN apply its own template on top is the B290 failure with a new hiding place.
    assert fake.config["use_template"] is False
    assert fake.config["max_all_tokens"] == 4096
    assert fake.config["max_new_tokens"] == 50
    # Each eval row is independent, so carrying a KV cache between them is both wrong and slower.
    assert fake.config["reuse_kv"] is False
    # Config BEFORE load: two of those entries decide what is allocated, not just how it decodes.
    assert [name for name, _ in fake.events] == ["set_config", "load"]


def test_load_refuses_an_incomplete_artifact(tmp_path):
    artifact = tmp_path / "model-mnn-q4"
    artifact.mkdir()
    (artifact / "llm.mnn").write_bytes(b"graph")  # no weights, no tokenizer, no config
    with pytest.raises(RuntimeError, match="missing"):
        helpers.load_mnn_llm(str(artifact))


def test_missing_llm_api_names_the_script_that_builds_it(monkeypatch):
    """pymnn's published wheel has no LLM bindings, so this is the expected first failure."""
    monkeypatch.setitem(sys.modules, "MNN", None)
    monkeypatch.setitem(sys.modules, "MNN.llm", None)
    with pytest.raises(ImportError, match="scripts/setup_mnn_env.sh"):
        helpers._mnn_llm_module()


# ── Generation ──────────────────────────────────────────────────────────────────────────────────

def test_every_row_resets_the_conversation(tmp_path):
    """A pymnn `Llm` is a CHAT object: without reset, row 2 is answered in row 1's context.

    That failure does not raise and does not look wrong in the log — it just makes every score
    after the first row a different measurement.
    """
    fake = FakeMnnLlm(replies=["a", "b", "c"])
    with patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(
        create=lambda path, embedding_model=False: fake
    )):
        outputs = helpers.infer_batch_mnn(
            ["one", "two", "three"], _artifact(tmp_path),
            max_new_tokens=16, base_model="HuggingFaceTB/SmolLM2-360M-Instruct",
            task="clinc150",
        )

    assert outputs == ["a", "b", "c"]
    response_events = [name for name, _ in fake.events if name in {"reset", "response"}]
    assert response_events == ["reset", "response"] * 3


def test_prompts_are_rendered_exactly_as_the_gguf_path_renders_them(tmp_path):
    """The shared rendering is what makes a llama.cpp-vs-MNN score difference interpretable.

    If each backend applied its own chat template, a gap between them could be the quantization,
    the runtime or the template, and nothing would say which.
    """
    fake = FakeMnnLlm()
    base = "HuggingFaceTB/SmolLM2-360M-Instruct"
    with (
        patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(
            create=lambda path, embedding_model=False: fake
        )),
        patch.object(helpers, "_serving_prompt_prefix", return_value="<rendered>") as render,
    ):
        helpers.infer_batch_mnn(["what is the weather"], _artifact(tmp_path), base_model=base)

    render.assert_called_once_with("what is the weather", base)
    assert fake.calls == ["<rendered>"]


def test_a_prompt_that_cannot_fit_the_context_ends_the_eval(tmp_path):
    """Raise, do not truncate, and do not absorb it into the per-row handler.

    MNN drops what does not fit its KV cache, so an over-long prompt yields a plausible answer to a
    question the model never saw. This is a configuration fault that applies to every row — a
    CLINC150 prompt carries all 151 intent names and needs ~1,410 tokens — so scoring the rows as
    empty would report it as a model that answers nothing.
    """
    fake = FakeMnnLlm(tokens_per_prompt=1410)
    with patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(
        create=lambda path, embedding_model=False: fake
    )):
        with pytest.raises(ValueError, match="exceeding input budget"):
            helpers.infer_batch_mnn(
                ["x"], _artifact(tmp_path), max_new_tokens=50, task="clinc150",
            )

    assert fake.calls == [], "nothing may be generated once the budget check has failed"


def test_one_undecodable_row_is_scored_empty_rather_than_losing_the_eval(tmp_path, capsys):
    fake = FakeMnnLlm()

    def explode(llm, prompt, max_new_tokens=512):
        if prompt.endswith("2"):
            raise RuntimeError("MNN session failed mid-decode")
        return "ok"

    with (
        patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(
            create=lambda path, embedding_model=False: fake
        )),
        patch.object(helpers, "_serving_prompt_prefix", side_effect=lambda p, b: p),
        patch.object(helpers, "mnn_generate", side_effect=explode),
    ):
        outputs = helpers.infer_batch_mnn(["row1", "row2", "row3"], _artifact(tmp_path))

    assert outputs == ["ok", "", "ok"]
    assert "row 1 failed to decode" in capsys.readouterr().out


# ── Caching ─────────────────────────────────────────────────────────────────────────────────────

def test_the_model_is_loaded_once_per_artifact_and_context(tmp_path):
    """A reload costs seconds and every iteration scores hundreds of rows."""
    loads = []

    def create(path, embedding_model=False):
        loads.append(path)
        return FakeMnnLlm()

    artifact = _artifact(tmp_path)
    with patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(create=create)):
        helpers.infer_batch_mnn(["a"], artifact, task="clinc150")
        helpers.infer_batch_mnn(["b"], artifact, task="clinc150")

    assert len(loads) == 1


def test_clearing_the_inference_cache_drops_the_mnn_model(tmp_path):
    """Called when the pipeline escalates: the model it moved on from must not stay resident."""
    artifact = _artifact(tmp_path)
    with patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(
        create=lambda path, embedding_model=False: FakeMnnLlm()
    )):
        helpers.infer_batch_mnn(["a"], artifact, task="clinc150")
        assert helpers._mnn_cache

        with patch("torch.cuda.is_available", return_value=False):
            helpers.clear_inference_cache()

    assert helpers._mnn_cache == {}
    assert helpers._mnn_cache_order == []


# ── Which device MNN computes on ─────────────────────────────────────────────────────────────────
#
# The GPU is here for SPEED, and the numbers are the justification: on SmolLM2-360M @ 4-bit over
# clinc150 rows, prefill is 671 tok/s on the CPU against 6,867 on an L40S, and a CLINC150 prompt is
# ~1,385 tokens against a one-label answer — so prefill is ~99% of the work and the GPU is ~6x end
# to end. Accuracy is unchanged, which is the same argument the GGUF path makes for offloading.

def test_auto_selects_cuda_when_a_gpu_is_visible(monkeypatch):
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "auto")
    with patch("torch.cuda.is_available", return_value=True):
        assert helpers.mnn_backend_type() == "cuda"


def test_auto_selects_cpu_without_a_gpu(monkeypatch):
    """A machine with no GPU is not a silent fallback — it is a machine with no GPU."""
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "auto")
    with patch("torch.cuda.is_available", return_value=False):
        assert helpers.mnn_backend_type() == "cpu"


def test_an_explicit_backend_is_honoured(monkeypatch):
    """So that a run pinned to CUDA fails loudly rather than producing CPU numbers 6x slower."""
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cuda")
    with patch("torch.cuda.is_available", return_value=False):
        assert helpers.mnn_backend_type() == "cuda"
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cpu")
    with patch("torch.cuda.is_available", return_value=True):
        assert helpers.mnn_backend_type() == "cpu"


def test_an_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "opencl")
    with pytest.raises(ValueError, match="not a supported MNN backend"):
        helpers.mnn_backend_type()


def test_the_backend_is_read_per_call_not_at_import(monkeypatch):
    """`mnn_backend_matrix` flips it between cells, and an import-time constant broke that.

    Its "cpu" column silently ran on CUDA and came back identical to the GPU column to four
    decimal places, which is the only reason the bug was visible at all.
    """
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cuda")
    assert helpers.mnn_backend_type() == "cuda"
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cpu")
    assert helpers.mnn_backend_type() == "cpu"


def test_a_silent_cpu_fallback_is_refused(tmp_path):
    """MNN reports a fallback ONLY on stdout, and keeps computing — on the wrong device.

    Asked for CUDA with a pymnn whose CUDA backend was not registered, MNN printed
    `Can't Find type=2 backend, use 0 instead` and ran on the CPU at CPU speed, decoding
    `'accept<|endoftext|>...'` instead of `'accept_reservations'` because it had already configured
    itself for a GPU. Four GPU measurements were taken that way before the cause was found. Fatal,
    because a fallback defeats the entire purpose of the GPU path AND corrupts the output.
    """
    with pytest.raises(RuntimeError, match="fell back to the CPU") as error:
        helpers._assert_backend_honoured(
            "cuda", "Can't Find type=2 backend, use 0 instead\n", str(tmp_path)
        )
    message = str(error.value)
    assert "SHARED libMNN.so" in message, "the message must name the build fix"
    assert "scripts/setup_mnn_env.sh" in message
    assert "SLM_MNN_BACKEND_TYPE=cpu" in message, "and the deliberate-CPU escape hatch"


def test_normal_engine_output_is_not_mistaken_for_a_fallback(tmp_path):
    helpers._assert_backend_honoured(
        "cuda", "The device supports: i8sdot:0, fp16:1\nUpdate cache to /tmp/x\n", str(tmp_path)
    )


def test_cuda_memory_mode_is_chosen_by_bit_width(tmp_path, monkeypatch):
    """MNN's CUDA int4 weight-only kernel is sound; its int8 one is not, so the width decides.

    Measured on SmolLM2-360M @ 8-bit over clinc150 rows: `memory=low` on CUDA scored 0.0000 with
    format_valid 0.0000 and `<|endoftext|>` on every row, while `memory=normal` scored exactly what
    the CPU scored (0.2667 both) at 4x the CPU's speed. Dequantizing into device memory costs VRAM,
    not accuracy — the values are still the quantized ones.
    """
    monkeypatch.setattr(helpers, "_MNN_MEMORY", "auto")
    four_bit = _artifact(tmp_path / "q4", quant_bit=4)
    eight_bit = _artifact(tmp_path / "q8", quant_bit=8)
    sixteen_bit = _artifact(tmp_path / "q16", quant_bit=16)

    assert helpers.mnn_memory_mode("cuda", four_bit) == "low"
    assert helpers.mnn_memory_mode("cuda", eight_bit) == "normal"
    assert helpers.mnn_memory_mode("cuda", sixteen_bit) == "normal"
    # The CPU handles every width with the packed weights, which is also what a phone does.
    for artifact in (four_bit, eight_bit, sixteen_bit):
        assert helpers.mnn_memory_mode("cpu", artifact) == "low"


def test_forcing_the_broken_cuda_memory_mode_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "_MNN_MEMORY", "low")
    with pytest.raises(RuntimeError, match="decodes garbage for a 8-bit artifact"):
        helpers.mnn_memory_mode("cuda", _artifact(tmp_path / "q8", quant_bit=8))


def test_an_explicit_memory_mode_is_otherwise_honoured(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "_MNN_MEMORY", "normal")
    assert helpers.mnn_memory_mode("cuda", _artifact(tmp_path / "q4", quant_bit=4)) == "normal"
    monkeypatch.setattr(helpers, "_MNN_MEMORY", "low")
    assert helpers.mnn_memory_mode("cuda", _artifact(tmp_path / "q4b", quant_bit=4)) == "low"


def test_the_measured_bad_cuda_combination_is_refused(tmp_path, monkeypatch):
    """precision=normal + memory=low on CUDA decodes garbage at no speed gain, so it is refused.

    Measured on the same artifact and rows that give `accept_reservations` under precision=low:
    `'ordinaryritz Hviations:`~ ...'`, 0/5 exact, and 0.44 rows/s against 2.86. The int4
    weight-only kernel appears to have no fp32 path.
    """
    monkeypatch.setattr(helpers, "_MNN_PRECISION", "normal")
    monkeypatch.setattr(helpers, "_MNN_MEMORY", "low")
    with pytest.raises(RuntimeError, match="measured to decode GARBAGE"):
        helpers.load_mnn_llm(_artifact(tmp_path), backend_type="cuda")


def test_the_model_cache_is_keyed_on_the_backend(tmp_path, monkeypatch):
    """A cached CPU model must never answer a request that asked for the GPU."""
    loads = []

    def create(path, embedding_model=False):
        loads.append(path)
        return FakeMnnLlm()

    artifact = _artifact(tmp_path)
    with patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(create=create)):
        monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cpu")
        helpers.infer_batch_mnn(["a"], artifact, task="clinc150")
        monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cuda")
        helpers.infer_batch_mnn(["a"], artifact, task="clinc150")

    assert len(loads) == 2, "the second request asked for a different device and must reload"


def test_a_second_cuda_load_in_one_process_is_refused(tmp_path, monkeypatch):
    """MNN's CUDA runtime does not survive it, and it fails by returning EMPTY TEXT.

    Measured over 27 artifacts loaded and freed in one process: the first decoded correctly, the
    second decoded `<|endoftext|>` repeatedly, and every one after returned an empty string —
    scored 0.0000 with nothing raised, at a nonsensical 878 rows/s because generation was doing no
    work. The first version of the backend matrix reported "27/27 passed" on that column.

    Free in the pipeline, which already runs every eval and every artifact build in its own
    disposable CUDA worker; the guard is for everything else.
    """
    monkeypatch.setattr(helpers, "_mnn_cuda_loads", 0)
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cuda")
    artifact = _artifact(tmp_path)

    with patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(
        create=lambda path, embedding_model=False: FakeMnnLlm()
    )):
        helpers.load_mnn_llm(artifact, backend_type="cuda")
        with pytest.raises(RuntimeError, match="already loaded an MNN model on CUDA") as error:
            helpers.load_mnn_llm(artifact, backend_type="cuda")

    message = str(error.value)
    assert "SLM_CUDA_ISOLATION=1" in message, "the message must name the pipeline's own answer"
    assert "SLM_MNN_BACKEND_TYPE=cpu" in message, "and the backend that has no such limit"


def test_the_cpu_backend_may_load_as_many_models_as_it_likes(tmp_path, monkeypatch):
    """The CPU path has no such limit — and the thread cross-check depends on loading a second."""
    monkeypatch.setattr(helpers, "_mnn_cuda_loads", 0)
    artifact = _artifact(tmp_path)

    with patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(
        create=lambda path, embedding_model=False: FakeMnnLlm()
    )):
        helpers.load_mnn_llm(artifact, backend_type="cpu")
        helpers.load_mnn_llm(artifact, backend_type="cpu")
        helpers.load_mnn_llm(artifact, backend_type="cpu")


# ── Concurrency parity with the GGUF path ────────────────────────────────────────────────────────

def test_mnn_eval_is_sequential_like_the_gguf_path(tmp_path, monkeypatch):
    """One row at a time is PARITY, not a shortfall: MAX_GGUF_EVAL_CONCURRENCY defaults to 1 too."""
    assert helpers.MAX_GGUF_EVAL_CONCURRENCY == 1, (
        "if the GGUF default ever rises, this test and the MNN path should be revisited together"
    )
    fake = FakeMnnLlm()
    monkeypatch.delenv("SLM_MNN_EVAL_CONCURRENCY", raising=False)
    with patch.object(helpers, "_mnn_llm_module", return_value=SimpleNamespace(
        create=lambda path, embedding_model=False: fake
    )):
        helpers.infer_batch_mnn(["a", "b"], _artifact(tmp_path), task="clinc150")
    # One reset and one response per row, in order: no interleaving, no second instance.
    assert [name for name, _ in fake.events if name in {"reset", "response"}] == [
        "reset", "response", "reset", "response",
    ]


def test_asking_for_more_concurrency_than_llama_cpp_is_refused(tmp_path, monkeypatch):
    """Refused rather than ignored: a variable that silently does nothing is worse than an error."""
    monkeypatch.setenv("SLM_MNN_EVAL_CONCURRENCY", "4")
    with pytest.raises(ValueError, match="not implemented"):
        helpers.infer_batch_mnn(["a"], _artifact(tmp_path), task="clinc150")


# ── The engine's own chatter ─────────────────────────────────────────────────────────────────────

def test_c_level_output_is_suppressed_inside_the_block(capfd):
    """MNN prints every prompt and response through `printf`, which ignores `sys.stdout`.

    Over an 800-row eval that is tens of thousands of lines between the ones a human is reading.
    Both halves matter: silenced inside, and restored after.
    """
    with helpers._suppress_c_stdout(True):
        os.write(1, b"engine chatter\n")
    os.write(1, b"visible again\n")

    captured = capfd.readouterr().out
    assert "engine chatter" not in captured
    assert "visible again" in captured


def test_suppression_can_be_turned_off_for_debugging(capfd):
    """`SLM_MNN_VERBOSE=1` exists because a failing load's own diagnosis is what you need."""
    with helpers._suppress_c_stdout(False):
        os.write(1, b"engine chatter\n")
    assert "engine chatter" in capfd.readouterr().out


def test_c_level_output_can_be_captured_for_inspection(capfd):
    """The load path CAPTURES rather than discards, because MNN reports a fallback only there."""
    with helpers._capture_c_stdout() as engine:
        os.write(1, b"Can't Find type=2 backend, use 0 instead\n")
    os.write(1, b"restored\n")

    assert "Can't Find type=2" in engine.text
    captured = capfd.readouterr().out
    assert "Can't Find type=2" not in captured, "captured output must not also leak to the log"
    assert "restored" in captured
