# tests/training/test_quantize_mnn.py
"""The MNN export/quantize half of the backend pair.

Deliberately parallel to `tests/training/test_quantize.py`, because the two modules are meant to
be interchangeable behind `training/quant_backend.py`: the same cache protocol (validate, then
record a content hash), the same timeout policy, the same refusal to let a broken toolchain be
mistaken for a bad model. What is tested here and has no GGUF counterpart is the bit-width
verification, for the reason the code gives: MNN's 4-bit and 8-bit exports are the same five
filenames, so nothing but the exporter's own record distinguishes them.
"""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

import training.quantize_mnn as mnn


def _write_artifact(
    directory, quant_bit=4, lm_quant_bit=None, weight=b"weights", graph=b"graph"
):
    """A minimally complete MNN artifact, as llmexport would leave it.

    `export_args.json` carries `lm_quant_bit` because a real export always does, and the cache-hit
    criterion reads it — a fixture without it would test the lenient "unknown setting" branch while
    looking like it tested the real one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "llm.mnn").write_bytes(graph)
    (directory / "llm.mnn.weight").write_bytes(weight)
    (directory / "tokenizer.mtok").write_text("tokenizer")
    (directory / "llm_config.json").write_text(json.dumps({"hidden_size": 576}))
    (directory / "config.json").write_text(json.dumps({"llm_model": "llm.mnn"}))
    (directory / "export_args.json").write_text(json.dumps({
        "quant_bit": quant_bit,
        "lm_quant_bit": max(8, quant_bit) if lm_quant_bit is None else lm_quant_bit,
    }))
    return directory


# ── The selector → bit-width mapping ────────────────────────────────────────────────────────────

def test_pool_quant_names_map_to_mnn_bit_widths():
    """The pool's selectors are shared with the GGUF backend; only the bit width transfers."""
    assert mnn.quant_bit("Q4_K_M") == 4
    assert mnn.quant_bit("Q8_0") == 8


def test_unknown_quant_raises_value_error():
    with pytest.raises(ValueError, match="Unknown quant"):
        mnn.quant_bit("INT3")


def test_artifact_name_encodes_the_bit_width():
    assert mnn.artifact_name("Q4_K_M") == "model-mnn-q4"
    assert mnn.artifact_name("Q8_0") == "model-mnn-q8"


# ── Toolchain discovery ─────────────────────────────────────────────────────────────────────────

def test_missing_toolchain_is_an_infrastructure_error_naming_the_fix(monkeypatch, tmp_path):
    """A missing exporter is never a model's fault, and the message has to say what to run.

    `QuantizationInfrastructureError` specifically: that is the type `evaluate_node` re-raises to
    stop the run, rather than treating as a bad checkpoint to roll back from.
    """
    monkeypatch.setenv("SLM_MNN_ROOT", str(tmp_path / "absent"))
    monkeypatch.delenv("SLM_MNN_LLMEXPORT", raising=False)
    monkeypatch.delenv("SLM_MNN_CONVERT_BIN", raising=False)
    monkeypatch.setenv("SLM_MNN_PYTHON", str(tmp_path / "absent" / "python"))
    with patch("shutil.which", return_value=None):
        with pytest.raises(
            mnn.QuantizationInfrastructureError, match="scripts/setup_mnn_env.sh"
        ) as error:
            mnn.resolve_toolchain()
    message = str(error.value)
    assert "llmexport.py" in message
    assert "MNNConvert" in message


def test_toolchain_honours_explicit_paths(monkeypatch, tmp_path):
    llmexport = tmp_path / "llmexport.py"
    llmexport.write_text("# exporter")
    convert = tmp_path / "MNNConvert"
    convert.write_text("#!/bin/sh\n")
    convert.chmod(0o755)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    monkeypatch.setenv("SLM_MNN_LLMEXPORT", str(llmexport))
    monkeypatch.setenv("SLM_MNN_CONVERT_BIN", str(convert))
    monkeypatch.setenv("SLM_MNN_PYTHON", str(python))

    toolchain = mnn.resolve_toolchain()

    assert toolchain.llmexport == str(llmexport)
    assert toolchain.mnnconvert == str(convert)
    assert toolchain.python == str(python)


def test_default_root_is_a_sibling_of_the_project(monkeypatch):
    """Derived, not hardcoded: it is where the llama.cpp checkout already sits."""
    monkeypatch.delenv("SLM_MNN_ROOT", raising=False)
    assert mnn.default_mnn_root() == os.path.join(
        os.path.dirname(mnn.PROJECT_ROOT), "MNN"
    )


# ── The export command ─────────────────────────────────────────────────────────────────────────

def _fake_toolchain(tmp_path):
    return mnn.MnnToolchain(
        python=str(tmp_path / "python"),
        llmexport=str(tmp_path / "export" / "llmexport.py"),
        mnnconvert=str(tmp_path / "build" / "MNNConvert"),
        root=str(tmp_path),
    )


def test_export_passes_bit_width_block_and_a_real_mnnconvert(tmp_path, monkeypatch):
    """Every one of these four arguments is load-bearing, so the argv is pinned.

    `--mnnconvert` most of all: without it llmexport falls back to the pymnn bindings, which the
    reference harness recorded crashing with a bus error rather than failing cleanly.
    """
    monkeypatch.setattr(mnn, "MNN_QUANT_BLOCK", 64)
    checkpoint = tmp_path / "merged"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"0" * 1024)
    output = tmp_path / "out"

    def fake_run(command, timeout_s, partial_output=None, cwd=None):
        _write_artifact(tmp_path / "out" / "model-mnn-q4")
        fake_run.command = command
        fake_run.cwd = cwd
        return None

    with (
        patch.object(mnn, "resolve_toolchain", return_value=_fake_toolchain(tmp_path)),
        patch.object(mnn, "_run_quant_tool", side_effect=fake_run),
    ):
        result = mnn.export_from_model_spec(str(checkpoint), str(output), "Q4_K_M")

    assert result == str(output / "model-mnn-q4")
    command = fake_run.command
    assert command[2:4] == ["--path", str(checkpoint)]
    assert "--export" in command and command[command.index("--export") + 1] == "mnn"
    assert command[command.index("--quant_bit") + 1] == "4"
    assert command[command.index("--quant_block") + 1] == "64"
    assert command[command.index("--dst_path") + 1] == str(output / "model-mnn-q4")
    assert command[command.index("--mnnconvert") + 1].endswith("MNNConvert")
    # llmexport.py imports its `utils` package relatively, so it only runs from its own directory.
    assert fake_run.cwd == str(tmp_path / "export")


def test_relative_output_paths_are_made_absolute(tmp_path, monkeypatch):
    """The exporter runs from ITS OWN directory, so a relative path would be resolved there.

    This is the fault that killed run 40260162: `evaluate_node` passes
    `artifacts/mnn/<model>/<key>`, relative to the project root it chdir'd into, and the export
    wrote the entire model under `<MNN>/transformers/llm/export/artifacts/mnn/...` and exited 0.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "merged").mkdir()
    (tmp_path / "merged" / "model.safetensors").write_bytes(b"0" * 1024)

    def fake_run(command, timeout_s, partial_output=None, cwd=None):
        _write_artifact(tmp_path / "artifacts" / "mnn" / "key" / "model-mnn-q4")
        fake_run.command = command
        return None

    with (
        patch.object(mnn, "resolve_toolchain", return_value=_fake_toolchain(tmp_path)),
        patch.object(mnn, "_run_quant_tool", side_effect=fake_run),
    ):
        result = mnn.export_from_model_spec(
            "merged", os.path.join("artifacts", "mnn", "key"), "Q4_K_M"
        )

    command = fake_run.command
    assert os.path.isabs(command[command.index("--path") + 1])
    assert os.path.isabs(command[command.index("--dst_path") + 1])
    assert os.path.isabs(result)
    assert result.endswith(os.path.join("artifacts", "mnn", "key", "model-mnn-q4"))


def test_quant_block_is_configurable(tmp_path, monkeypatch):
    """It changes the weights, so it must be settable and it must be recorded."""
    monkeypatch.setattr(mnn, "MNN_QUANT_BLOCK", 128)
    checkpoint = tmp_path / "merged"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"0" * 1024)

    def fake_run(command, timeout_s, partial_output=None, cwd=None):
        _write_artifact(tmp_path / "out" / "model-mnn-q8", quant_bit=8)
        fake_run.command = command
        return None

    with (
        patch.object(mnn, "resolve_toolchain", return_value=_fake_toolchain(tmp_path)),
        patch.object(mnn, "_run_quant_tool", side_effect=fake_run),
    ):
        mnn.export_from_model_spec(str(checkpoint), str(tmp_path / "out"), "Q8_0")

    assert fake_run.command[fake_run.command.index("--quant_block") + 1] == "128"
    assert fake_run.command[fake_run.command.index("--quant_bit") + 1] == "8"


def test_exporter_failure_is_an_infrastructure_error(tmp_path):
    checkpoint = tmp_path / "merged"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"0" * 1024)

    with (
        patch.object(mnn, "resolve_toolchain", return_value=_fake_toolchain(tmp_path)),
        patch.object(
            mnn, "_run_quant_tool", return_value="llmexport.py failed: unsupported architecture"
        ),
    ):
        with pytest.raises(
            mnn.QuantizationInfrastructureError, match="unsupported architecture"
        ):
            mnn.export_from_model_spec(str(checkpoint), str(tmp_path / "out"), "Q4_K_M")


def test_exit_zero_with_an_incomplete_artifact_still_fails(tmp_path):
    """The exporter returning 0 is not evidence that it wrote a model.

    Mirrors the reference harness's completeness check, and it is not hypothetical: a converter
    that dies after the graph but before the weights leaves a directory that looks plausible and
    cannot be loaded.
    """
    checkpoint = tmp_path / "merged"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"0" * 1024)

    def fake_run(command, timeout_s, partial_output=None, cwd=None):
        artifact = tmp_path / "out" / "model-mnn-q4"
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "llm.mnn").write_bytes(b"graph")  # weights and tokenizer never arrived
        return None

    with (
        patch.object(mnn, "resolve_toolchain", return_value=_fake_toolchain(tmp_path)),
        patch.object(mnn, "_run_quant_tool", side_effect=fake_run),
    ):
        with pytest.raises(mnn.QuantizationInfrastructureError, match="incomplete"):
            mnn.export_from_model_spec(str(checkpoint), str(tmp_path / "out"), "Q4_K_M")


def test_a_sentencepiece_tokenizer_spelling_is_also_complete(tmp_path):
    """MNN writes `tokenizer.txt` for sentencepiece and `tokenizer.mtok` for HF fast tokenizers."""
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    os.rename(artifact / "tokenizer.mtok", artifact / "tokenizer.txt")
    assert mnn.missing_files(str(artifact)) == []


def test_export_gets_a_higher_floor_than_a_gguf_quantize(monkeypatch):
    """An MNN export traces to ONNX and rewrites the graph; the GGUF floor was sized on neither.

    Measured: 105s for a merged 360M against 24s for convert+quantize, and 417s for a 135M on a
    cold `.venv_mnn` — which is why this is a floor rather than a steeper per-GB rate. What the
    first export of a run pays is dominated by the torch import, not by the model.
    """
    monkeypatch.delenv("SLM_QUANT_TIMEOUT_S", raising=False)
    monkeypatch.setattr(mnn, "_MNN_EXPORT_TIMEOUT_FLOOR_S", 1800)
    # A small model stays on the floor rather than the GGUF path's 600s.
    assert mnn._export_timeout_s(300.0) == 1800
    # A big one still scales past it.
    assert mnn._export_timeout_s(30_000.0) > 1800


def test_an_explicit_operator_ceiling_is_not_raised_to_the_floor(monkeypatch):
    monkeypatch.setenv("SLM_QUANT_TIMEOUT_S", "900")
    monkeypatch.setattr("training.quantize._QUANT_TIMEOUT_OVERRIDE_S", "900")
    monkeypatch.setattr(mnn, "_MNN_EXPORT_TIMEOUT_FLOOR_S", 1800)
    assert mnn._export_timeout_s(300.0) == 900


# ── Validation and caching ─────────────────────────────────────────────────────────────────────

def test_validation_refuses_a_bit_width_that_does_not_match_the_label(tmp_path):
    """The check with no GGUF counterpart, and the one that makes "4-bit" a verified claim.

    An 8-bit export and a 4-bit export produce identical filenames, so a label mismatch would
    otherwise be reported as one quantization's accuracy under the other's name.
    """
    artifact = _write_artifact(tmp_path / "model-mnn-q4", quant_bit=8)
    with pytest.raises(RuntimeError, match="exported at 8-bit but is labelled Q4_K_M"):
        mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")
    assert not os.path.exists(mnn.mnn_validation_sidecar_path(str(artifact)))


def test_validation_records_a_hash_of_every_file_and_the_cache_hit_checks_it(tmp_path):
    """A weight file swapped under an unchanged graph must not read as a warm cache.

    The GGUF sidecar hashes one file because a GGUF IS one file. An MNN model is only usable if
    graph, weights, tokenizer and both configs agree, so all of them are covered.
    """
    artifact = _write_artifact(tmp_path / "model-mnn-q4", quant_bit=4)
    with (
        patch("training.slm_helpers.load_mnn_llm", return_value=MagicMock()),
        patch("training.slm_helpers.mnn_generate", return_value="Hello! How can I help?"),
    ):
        record = mnn.validate_and_record_mnn(
            str(artifact), base_model="HuggingFaceTB/SmolLM2-135M-Instruct", quant="Q4_K_M"
        )

    assert record["quant_bit"] == 4
    assert record["quant_block"] == mnn.MNN_QUANT_BLOCK
    assert set(record["fingerprint"]["files"]) >= {
        "llm.mnn", "llm.mnn.weight", "tokenizer.mtok", "llm_config.json", "config.json",
    }
    assert all(len(entry["sha256"]) == 64 for entry in record["fingerprint"]["files"].values())
    assert mnn.validated_mnn_cache_hit(str(artifact)) is True

    (artifact / "llm.mnn.weight").write_bytes(b"tampered-weights")
    assert mnn.validated_mnn_cache_hit(str(artifact)) is False


def test_a_degenerate_smoke_test_reports_but_does_not_fail(tmp_path, capsys):
    """B289's lesson, carried over verbatim: the load is the gate, generation only reports.

    As a fatal check this ended three healthy runs, because a small base model answering a trivial
    prompt is not a corrupt artifact. The eval score is the reliable detector.
    """
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    with (
        patch("training.slm_helpers.load_mnn_llm", return_value=MagicMock()),
        patch("training.slm_helpers.mnn_generate", return_value="////////////////"),
    ):
        record = mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    assert "no word characters" in capsys.readouterr().out
    assert record["fingerprint"]["files"]  # still recorded: the load succeeded
    assert mnn.validated_mnn_cache_hit(str(artifact)) is True


# ── The thread-count cross-check ────────────────────────────────────────────────────────────────
#
# In MNN the thread count is a CORRECTNESS setting. Run 40260927 scored
# `Qwen/Qwen3.5-0.8B@Q4_K_M` at 0.0000 with format_valid 0.0000 — all 1,000 CLINC150 rows decoded
# to `%+!!!!!!!!!!` — at `thread_num=14`, while the same artifact at 1/2/4/8/10/12/13/16 answered
# them correctly and identically. The logits are finite with a different argmax, so the only
# detectable symptom is the answer being wrong, and the short smoke test could not see it: the same
# artifact answers a 16-token `Hello` perfectly and fails at ~1,050 tokens.

def _thread_check_mocks(tmp_path, monkeypatch, configured_text, reference_text, threads=14):
    """Patch the two loads and their generations so the cross-check sees a chosen pair of outputs."""
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    monkeypatch.setattr("training.slm_helpers._MNN_THREADS", threads)
    monkeypatch.setattr("training.slm_helpers.MNN_REFERENCE_THREADS", 4)
    monkeypatch.setattr(mnn, "_THREAD_CHECK_ENABLED", True)

    configured = MagicMock(name="configured")
    reference = MagicMock(name="reference")
    prompts: list[str] = []

    def generate(llm, prompt, max_new_tokens=512):
        prompts.append(prompt)
        if llm is reference:
            return reference_text
        # The short smoke test runs on the configured model first and must not be mistaken for the
        # cross-check's own generation.
        return "Hello there!" if prompt == "Hello" or len(prompt) < 100 else configured_text

    context = (
        patch("training.slm_helpers.load_mnn_llm", side_effect=[configured, reference]),
        patch("training.slm_helpers.mnn_generate", side_effect=generate),
    )
    return artifact, prompts, context


def test_a_thread_count_that_disagrees_with_the_reference_is_refused(tmp_path, monkeypatch):
    """Greedy decoding cannot depend on the work split, so a mismatch means corrupted compute.

    Disagreement rather than "looks like garbage" is the condition, and that is measured, not
    stylistic: on the failing artifact 8/10/12/13/16 threads each agree with the 4-thread reference
    token-for-token while 14 differs — yet on synthetic filler the bad setting still produces
    READABLE text, so a degeneracy test passes it.
    """
    artifact, prompts, (patch_load, patch_generate) = _thread_check_mocks(
        tmp_path, monkeypatch, "The label that applies is **scheduling", "scheduling"
    )

    with patch_load, patch_generate:
        with pytest.raises(
            mnn.QuantizationInfrastructureError,
            match="differently at thread_num=14 and at 4",
        ) as error:
            mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    message = str(error.value)
    assert "SLM_MNN_THREADS" in message, "the message must name the knob that fixes it"
    assert "0.0000" in message, "and why refusing beats scoring: the eval would look real"
    assert "SLM_MNN_THREAD_CHECK=0" in message, "a strict check needs a documented escape hatch"
    assert not os.path.exists(mnn.mnn_validation_sidecar_path(str(artifact)))
    # The cross-check's own prompt — the first entry is the short smoke test's `Hello` — has to be
    # long enough to tile a prefill, which is exactly what the smoke test fails to do.
    assert len(prompts[-1]) > 500


def test_outright_garbage_at_the_configured_count_is_also_refused(tmp_path, monkeypatch):
    """The shape the real failure took: `%+!!!!!!!!!!` against a clean label."""
    artifact, _prompts, (patch_load, patch_generate) = _thread_check_mocks(
        tmp_path, monkeypatch, "%+!!!!!!!!!!!!!!!!", "accept_reservations"
    )

    with patch_load, patch_generate:
        with pytest.raises(mnn.QuantizationInfrastructureError, match="corrupting compute"):
            mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")


def test_degenerate_at_both_thread_counts_is_only_a_warning(tmp_path, monkeypatch, capsys):
    """Then there is no threading claim to make — it is a statement about the model (B289)."""
    artifact, _prompts, (patch_load, patch_generate) = _thread_check_mocks(
        tmp_path, monkeypatch, "!!!!!!!!!!!!", "????????????"
    )

    with patch_load, patch_generate:
        record = mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    assert "not the threading" in capsys.readouterr().out
    assert record["fingerprint"]["files"]
    assert mnn.validated_mnn_cache_hit(str(artifact)) is True


def test_the_cross_check_can_be_turned_off(tmp_path, monkeypatch, capsys):
    """Strict by design, so an operator who has judged a mismatch benign needs a way through."""
    artifact, _prompts, (patch_load, patch_generate) = _thread_check_mocks(
        tmp_path, monkeypatch, "%+!!!!!!!!!!!!!!!!", "accept_reservations"
    )
    monkeypatch.setattr(mnn, "_THREAD_CHECK_ENABLED", False)

    with patch_load, patch_generate:
        record = mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    assert "DISABLED" in capsys.readouterr().out
    assert record["validated_threads"] == 14


def test_agreement_across_thread_counts_passes_and_is_recorded(tmp_path, monkeypatch, capsys):
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    monkeypatch.setattr("training.slm_helpers._MNN_THREADS", 8)
    monkeypatch.setattr("training.slm_helpers.MNN_REFERENCE_THREADS", 4)

    with (
        patch("training.slm_helpers.load_mnn_llm", return_value=MagicMock()),
        patch("training.slm_helpers.mnn_generate", return_value="accept_reservations"),
    ):
        record = mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    assert "thread cross-check passed" in capsys.readouterr().out
    # "This artifact loads and decodes" is only a claim about the configuration it was checked at.
    assert record["validated_threads"] == 8


def test_the_gpu_is_cross_checked_against_the_cpu_not_against_a_thread_count(
    tmp_path, monkeypatch, capsys
):
    """On a GPU run the thread count is irrelevant, so the comparison is against the CPU instead.

    The CPU is the device that ships, which makes it the right reference: the question a GPU eval
    has to answer is "does this compute what the phone would", not "does this agree with itself at
    a different thread count".
    """
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cuda")
    monkeypatch.setattr("training.slm_helpers._MNN_THREADS", 14)
    monkeypatch.setattr(mnn, "_GPU_CROSS_CHECK_ENABLED", True)

    with (
        patch("training.slm_helpers.load_mnn_llm", return_value=MagicMock()) as load,
        patch("training.slm_helpers.mnn_generate", return_value="accept_reservations"),
    ):
        record = mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    output = capsys.readouterr().out
    assert "thread cross-check not applicable on the cuda backend" in output
    assert "GPU cross-check passed" in output
    assert load.call_count == 2, "the CPU reference has to be loaded to compare against"
    assert load.call_args.kwargs["backend_type"] == "cpu"
    # The sidecar still says what it was checked under: a load claim is per-configuration.
    assert record["validated_backend"] == "cuda"
    assert record["validated_threads"] == 14


def test_a_gpu_that_decodes_nothing_where_the_cpu_decodes_text_is_refused(tmp_path, monkeypatch):
    """The shape of BOTH MNN/CUDA faults found so far, so it is the invariant worth enforcing.

    An unregistered CUDA backend falling back to the CPU decoded
    `'accept<|endoftext|><|endoftext|>...'`, and MNN's CUDA int8 weight-only kernel scored
    format_valid 0.0000 with `'<|endoftext|>'` on every row — in both cases against a CPU that
    decoded real labels from the same artifact.
    """
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cuda")
    monkeypatch.setattr(mnn, "_GPU_CROSS_CHECK_ENABLED", True)

    on_gpu = MagicMock(name="gpu")
    on_cpu = MagicMock(name="cpu")

    def generate(llm, prompt, max_new_tokens=512):
        if llm is on_cpu:
            return "accept_reservations"
        return "Hello there!" if len(prompt) < 100 else "<|endoftext|><|endoftext|>"

    with (
        patch("training.slm_helpers.load_mnn_llm", side_effect=[on_gpu, on_cpu]),
        patch("training.slm_helpers.mnn_generate", side_effect=generate),
    ):
        with pytest.raises(
            mnn.QuantizationInfrastructureError, match="nothing usable on cuda"
        ) as error:
            mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    message = str(error.value)
    assert "SLM_MNN_BACKEND_TYPE=cpu" in message
    assert "SLM_MNN_GPU_CROSS_CHECK=0" in message
    assert not os.path.exists(mnn.mnn_validation_sidecar_path(str(artifact)))


def test_gpu_and_cpu_disagreeing_on_usable_text_is_not_a_failure(tmp_path, monkeypatch, capsys):
    """Equality is NOT the invariant here, unlike the thread check, and that is measured.

    fp16 accumulation on CUDA against fp32 on the CPU genuinely flips near-ties: the same 4-bit
    artifact scored 0.0600 on CUDA and 0.0317 on the CPU over 100 rows. Enforcing equality between
    devices would end healthy runs.
    """
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    monkeypatch.setenv("SLM_MNN_BACKEND_TYPE", "cuda")
    monkeypatch.setattr(mnn, "_GPU_CROSS_CHECK_ENABLED", True)

    on_gpu = MagicMock(name="gpu")
    on_cpu = MagicMock(name="cpu")

    def generate(llm, prompt, max_new_tokens=512):
        return "cancel_reservation" if llm is on_cpu else "accept_reservations"

    with (
        patch("training.slm_helpers.load_mnn_llm", side_effect=[on_gpu, on_cpu]),
        patch("training.slm_helpers.mnn_generate", side_effect=generate),
    ):
        record = mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    assert "GPU cross-check passed" in capsys.readouterr().out
    assert mnn.validated_mnn_cache_hit(str(artifact)) is True
    assert record["validated_backend"] == "cuda"


def test_the_cross_check_is_skipped_when_there_is_nothing_to_compare(tmp_path, monkeypatch):
    """Configured == reference: a second load would measure the same setting twice."""
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    monkeypatch.setattr("training.slm_helpers._MNN_THREADS", 4)
    monkeypatch.setattr("training.slm_helpers.MNN_REFERENCE_THREADS", 4)

    with (
        patch("training.slm_helpers.load_mnn_llm", return_value=MagicMock()) as load,
        patch("training.slm_helpers.mnn_generate", return_value="ok"),
    ):
        mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    assert load.call_count == 1, "only the artifact's own load, no reference load"


def test_lm_head_is_kept_above_the_body_width(tmp_path, monkeypatch):
    """`Q4_K_M` is mixed-precision — it keeps `output.weight` at Q6_K — so MNN must match it.

    Otherwise the comparison charges MNN for a difference in RECIPE rather than in runtime.
    """
    monkeypatch.setattr(mnn, "MNN_LM_QUANT_BIT", 8)
    checkpoint = tmp_path / "merged"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"0" * 1024)

    def fake_run(command, timeout_s, partial_output=None, cwd=None):
        _write_artifact(tmp_path / "out" / "model-mnn-q4")
        fake_run.command = command
        return None

    with (
        patch.object(mnn, "resolve_toolchain", return_value=_fake_toolchain(tmp_path)),
        patch.object(mnn, "_run_quant_tool", side_effect=fake_run),
    ):
        mnn.export_from_model_spec(str(checkpoint), str(tmp_path / "out"), "Q4_K_M")

    command = fake_run.command
    assert command[command.index("--quant_bit") + 1] == "4"
    assert command[command.index("--lm_quant_bit") + 1] == "8"


def test_lm_head_is_never_narrower_than_the_body(tmp_path, monkeypatch):
    """An 8-bit model with a 4-bit lm_head would be strictly worse than the default it overrides."""
    monkeypatch.setattr(mnn, "MNN_LM_QUANT_BIT", 4)
    checkpoint = tmp_path / "merged"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"0" * 1024)

    def fake_run(command, timeout_s, partial_output=None, cwd=None):
        _write_artifact(tmp_path / "out" / "model-mnn-q8", quant_bit=8)
        fake_run.command = command
        return None

    with (
        patch.object(mnn, "resolve_toolchain", return_value=_fake_toolchain(tmp_path)),
        patch.object(mnn, "_run_quant_tool", side_effect=fake_run),
    ):
        mnn.export_from_model_spec(str(checkpoint), str(tmp_path / "out"), "Q8_0")

    command = fake_run.command
    assert command[command.index("--lm_quant_bit") + 1] == "8"


def test_a_failed_load_writes_no_sidecar(tmp_path):
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    with patch(
        "training.slm_helpers.load_mnn_llm",
        side_effect=RuntimeError("Load module failed: llm.mnn.weight truncated"),
    ):
        with pytest.raises(RuntimeError, match="truncated"):
            mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    assert not os.path.exists(mnn.mnn_validation_sidecar_path(str(artifact)))
    assert mnn.validated_mnn_cache_hit(str(artifact)) is False


def test_changing_an_export_setting_invalidates_the_cache(tmp_path, monkeypatch):
    """The settings are part of the key, because the files cannot say which ones built them.

    `quant_block` and `lm_quant_bit` change the weights. A hit across a change to either would
    score the previous setting's model under the new one's name — B161 with a different label.
    """
    artifact = _write_artifact(tmp_path / "model-mnn-q4", quant_bit=4)
    monkeypatch.setattr(mnn, "MNN_QUANT_BLOCK", 64)
    monkeypatch.setattr(mnn, "MNN_LM_QUANT_BIT", 8)
    with (
        patch("training.slm_helpers.load_mnn_llm", return_value=MagicMock()),
        patch("training.slm_helpers.mnn_generate", return_value="accept_reservations"),
    ):
        mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")
    assert mnn.validated_mnn_cache_hit(str(artifact)) is True

    monkeypatch.setattr(mnn, "MNN_QUANT_BLOCK", 128)
    assert mnn.validated_mnn_cache_hit(str(artifact)) is False

    monkeypatch.setattr(mnn, "MNN_QUANT_BLOCK", 64)
    monkeypatch.setattr(mnn, "MNN_LM_QUANT_BIT", 16)
    assert mnn.validated_mnn_cache_hit(str(artifact)) is False


def test_an_artifact_that_never_recorded_its_lm_head_width_is_not_rejected(tmp_path, monkeypatch):
    """Unknown is not mismatched. An export whose args lack the key predates it being passed."""
    artifact = _write_artifact(tmp_path / "model-mnn-q4", quant_bit=4)
    (artifact / "export_args.json").write_text(json.dumps({"quant_bit": 4}))
    monkeypatch.setattr(mnn, "MNN_LM_QUANT_BIT", 8)
    with (
        patch("training.slm_helpers.load_mnn_llm", return_value=MagicMock()),
        patch("training.slm_helpers.mnn_generate", return_value="accept_reservations"),
    ):
        record = mnn.validate_and_record_mnn(str(artifact), base_model="m", quant="Q4_K_M")

    assert record["lm_quant_bit"] is None
    assert mnn.validated_mnn_cache_hit(str(artifact)) is True


def test_invalidation_removes_the_artifact_and_its_record(tmp_path):
    artifact = _write_artifact(tmp_path / "model-mnn-q4")
    sidecar = mnn.mnn_validation_sidecar_path(str(artifact))
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1}, handle)

    mnn.invalidate_mnn_cache(str(artifact))

    assert not os.path.exists(artifact)
    assert not os.path.exists(sidecar)
    # Idempotent: the reaper calls this for artifacts that may already be gone.
    mnn.invalidate_mnn_cache(str(artifact))


def test_weight_size_counts_only_what_ships(tmp_path):
    artifact = _write_artifact(
        tmp_path / "model-mnn-q4",
        graph=b"0" * (1024 * 1024),
        weight=b"0" * (3 * 1024 * 1024),
    )
    (artifact / "tokenizer.mtok").write_bytes(b"0" * (10 * 1024 * 1024))
    assert mnn.weight_size_mb(str(artifact)) == pytest.approx(4.0)


def test_sidecar_lives_beside_the_artifact_not_inside_it(tmp_path):
    """Otherwise the record would become part of the fingerprint it describes."""
    artifact = tmp_path / "model-mnn-q4"
    assert mnn.mnn_validation_sidecar_path(str(artifact)) == f"{artifact}.validation.json"
