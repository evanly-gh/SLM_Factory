# training/quantize_mnn.py
"""
Weight quantization to MNN — the second on-device runtime this loop can ship to.

`training/quantize.py` is the llama.cpp half of this pair and remains the default. The two are
deliberately shaped the same way (export → quantize → load-validate → cache with an atomic
sidecar) because `agent/nodes/evaluate.py` routes between them on one flag and must not care
which one it got.

WHAT IS DIFFERENT ABOUT MNN, AND WHY THE CODE CANNOT JUST BE THE GGUF CODE
  * The artifact is a DIRECTORY, not a file. MNN splits a model into a graph (`llm.mnn`), its
    weights (`llm.mnn.weight`), a runtime config (`config.json`), the exported architecture
    description (`llm_config.json`) and a tokenizer (`tokenizer.mtok` or `tokenizer.txt`).
    Everything downstream that treats the quantized artifact as a path therefore has to accept a
    directory, which is why `quant_backend.py` exists rather than a second `gguf_path` argument.
  * Quantization is PARAMETRIC, not a named preset. llama.cpp ships `Q4_K_M`/`Q8_0` recipes;
    MNN takes `--quant_bit` (4 or 8 here), `--quant_block` (64 by default — the number of input
    channels that share one scale/zero-point) and a separate `--lm_quant_bit`. The pool's
    selector names are kept and mapped onto those numbers, so `@Q4_K_M` means "the 4-bit build
    for this backend" and a run's model ladder, tiering and reporting are untouched by the
    backend choice.
  * The exporter is a PYTHON program with its own dependency stack (torch + onnx + onnxslim),
    not a compiled CLI. It runs out of a dedicated `.venv_mnn` for exactly the reason
    `scripts/setup_metric_envs.sh` gives for ERRANT and BERTScore: nothing may renegotiate
    `.venv_gpu`'s torch/transformers underneath a training run. `scripts/setup_mnn_env.sh`
    builds both that venv and the MNN toolchain.
  * `llmexport.py` must be launched from its own directory (it imports `utils.*` relatively) and
    must be given a locally built `MNNConvert` binary. Without `--mnnconvert` it falls back to
    the pymnn bindings, which the reference harness recorded crashing with a bus error; a
    missing binary is therefore an infrastructure error here rather than a silent fallback.
"""
import json
import os
import platform
import re
import shutil
import subprocess

from training.quantize import (
    QuantizationInfrastructureError,
    _atomic_write_json,
    _dir_size_mb,
    _file_size_mb,
    _run_quant_tool,
    _sha256_file,
    _subprocess_timeout_s,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The quant_block used for every MNN build unless an operator overrides it. 64 is MNN's own
# default and what the reference harness swept with, so a number produced here is comparable to
# the ones measured on a phone through MNN-Chat. It is recorded in the validation sidecar because
# it changes the weights, and therefore the accuracy, as much as the bit width does.
MNN_QUANT_BLOCK = int(os.environ.get("SLM_MNN_QUANT_BLOCK", "64"))

# The lm_head is kept at 8 bits while the body goes to `quant_bit`, because that is what the
# selector being honoured already means: llama.cpp's `Q4_K_M` is a MIXED-precision recipe that
# quantizes the body to 4-bit and keeps `output.weight` at Q6_K. `llmexport.py` defaults
# `--lm_quant_bit` to `--quant_bit`, so an unqualified 4-bit MNN export is NOT the artifact
# `@Q4_K_M` names, and comparing it against a GGUF would charge MNN for a difference in recipe
# rather than in runtime. The reference harness this backend was ported from also found lm_head
# precision mattered on device (its commit 97865ec, "Fix Qwen3-1.7B/Qwen3.5-2B MNN garbage
# (lm_head precision)").
#
# NOT a fix for the garbage this integration actually hit — that was the thread count, see
# `slm_helpers._MNN_THREADS`. The same 4-bit-lm_head artifact decodes correctly at 8 threads and is
# what the recorded 0.7278 was measured on. Set SLM_MNN_LM_QUANT_BIT=4 to export the plain
# uniform-width model, or 16 to keep the lm_head in fp16.
MNN_LM_QUANT_BIT = int(os.environ.get("SLM_MNN_LM_QUANT_BIT", "8"))

# Selector → BODY bit width (the lm_head is set separately, above). The keys are the pool's
# deployment-variant names, shared with the GGUF backend on purpose (see config.QUANT_BACKEND):
# one run's `@Q4_K_M` is the other's, and only the runtime differs. `Q4_K_M` is a k-quant recipe
# whose per-tensor choices have no exact MNN equivalent, so this mapping is a statement about bit
# width — which is also the only thing the two formats can honestly be compared on.
_QUANT_BIT_MAP: dict[str, int] = {
    "Q4_K_M": 4,
    "Q8_0": 8,
    # Full precision, and deliberately NOT one of the pool's selector names. The pool's
    # third variant is `bf16` (`quant=None`), and for that the loop scores the HF weights through
    # Unsloth on BOTH backends rather than building an artifact — unchanged here. `FP16` exists so
    # that MNN's own full-precision export (`--quant_bit 16`, fp16 weights) is reachable and
    # testable by `hardware_eval/mnn_backend_matrix.py`, which has to cover all three precisions.
    # Nothing in the model pool resolves to this string, so no run can select it by accident.
    "FP16": 16,
}

# The files an MNN export must have produced for the directory to be a usable model. Mirrors the
# reference harness's completeness check, plus the tokenizer being spelled either way: MNN writes
# `tokenizer.mtok` for a HuggingFace fast tokenizer and `tokenizer.txt` for a sentencepiece one,
# so requiring one name specifically would reject half the pool.
_REQUIRED_FILES = ("config.json", "llm.mnn", "llm.mnn.weight", "llm_config.json")
_TOKENIZER_FILES = ("tokenizer.mtok", "tokenizer.txt")

_MNN_VALIDATION_SCHEMA_VERSION = 1
_MNN_VALIDATION_SUFFIX = ".validation.json"

# Wall-clock FLOOR for one export, over and above `quantize._subprocess_timeout_s`'s size scaling.
# The GGUF path's 600s floor was sized on `convert_hf_to_gguf` + `llama-quantize`, which read the
# weights and write them back out. An MNN export does more and takes longer: it traces the model
# through torch to ONNX and then has MNNConvert rewrite and quantize the graph. Measured on this
# cluster — 105s for a merged 360M checkpoint against 24s for the GGUF pair, and 417s for a 135M
# one on a COLD `.venv_mnn` (the torch import and the first NFS read of the exporter dominate a
# small model entirely). The cold case is the reason this is a floor rather than a bigger per-GB
# rate: what the first export of a run pays has almost nothing to do with the model's size.
_MNN_EXPORT_TIMEOUT_FLOOR_S = int(os.environ.get("SLM_MNN_EXPORT_TIMEOUT_FLOOR_S", "1800"))


def _export_timeout_s(source_size_mb: float) -> int:
    """Ceiling for one `llmexport.py` run over a source of ``source_size_mb``.

    `SLM_QUANT_TIMEOUT_S` still means what it means everywhere else — an explicit operator ceiling,
    used verbatim and unscaled — so setting it keeps full control and is NOT raised to the floor.
    """
    scaled = _subprocess_timeout_s(source_size_mb)
    if os.environ.get("SLM_QUANT_TIMEOUT_S"):
        return scaled
    return max(scaled, _MNN_EXPORT_TIMEOUT_FLOOR_S)


def artifact_name(quant: str) -> str:
    """Directory name for this quant's MNN build, e.g. ``model-mnn-q4``.

    Analogous to the GGUF path's `model-<method>.gguf`, and equally load-bearing: the cache key in
    `evaluate._build_or_reuse_quant_artifact` checks for this EXACT name so that two quant tiers of
    the same base model cannot be mistaken for each other (B161).
    """
    return f"model-mnn-q{quant_bit(quant)}"


def quant_bit(quant: str) -> int:
    """MNN bit width for a pool selector's quant name."""
    if quant not in _QUANT_BIT_MAP:
        raise ValueError(
            f"Unknown quant {quant!r} for the MNN backend. Valid values: {list(_QUANT_BIT_MAP)}"
        )
    return _QUANT_BIT_MAP[quant]


class MnnToolchain:
    """Where the three pieces of the MNN export toolchain actually are on this machine."""

    def __init__(self, python: str, llmexport: str, mnnconvert: str, root: str):
        self.python = python
        self.llmexport = llmexport
        self.mnnconvert = mnnconvert
        self.root = root

    def versions(self) -> dict:
        """Tool identity recorded in the validation sidecar, so a cache hit is per-toolchain.

        The MNN commit is what determines the graph and the quantization kernels, so a rebuilt
        toolchain must not be able to reuse an artifact produced by the previous one.
        """
        commit = "unknown"
        try:
            commit = subprocess.run(
                ["git", "-C", self.root, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout.strip()[:12] or "unknown"
        except (OSError, subprocess.SubprocessError):
            pass
        return {
            "mnn_commit": commit,
            "mnn_version": _mnn_version(self.root),
            "python": platform.python_version(),
        }


def _mnn_version(root: str) -> str:
    """MNN's own version triple, read from the header that defines it."""
    header = os.path.join(root, "include", "MNN", "MNNDefine.h")
    parts = {}
    try:
        with open(header, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = re.match(
                    r"#define MNN_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)", line.strip()
                )
                if match:
                    parts[match.group(1)] = match.group(2)
    except OSError:
        return "unknown"
    if len(parts) != 3:
        return "unknown"
    return f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"


def default_mnn_root() -> str:
    """Where MNN is expected to live when `SLM_MNN_ROOT` says nothing.

    A sibling of the project directory, which is where the llama.cpp checkout this pipeline
    already depends on sits. Derived rather than hardcoded so a clone of this repo elsewhere
    resolves its own neighbour.
    """
    explicit = os.environ.get("SLM_MNN_ROOT", "").strip()
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return os.path.join(os.path.dirname(PROJECT_ROOT), "MNN")


def resolve_toolchain() -> MnnToolchain:
    """Locate llmexport.py, MNNConvert and the exporter's interpreter, or say exactly what to run.

    Raises `QuantizationInfrastructureError` — not a bare RuntimeError — because that is the
    exception `evaluate_node` treats as "the toolchain is broken, stop the run" rather than as a
    bad model to roll back from. A missing exporter is never a model's fault.
    """
    root = default_mnn_root()
    llmexport = os.environ.get("SLM_MNN_LLMEXPORT", "").strip() or os.path.join(
        root, "transformers", "llm", "export", "llmexport.py"
    )
    mnnconvert = os.environ.get("SLM_MNN_CONVERT_BIN", "").strip() or os.path.join(
        root, "build", "MNNConvert"
    )
    python = os.environ.get("SLM_MNN_PYTHON", "").strip() or os.path.join(
        PROJECT_ROOT, ".venv_mnn", "bin", "python"
    )
    if not shutil.which(mnnconvert) and not os.path.isfile(mnnconvert):
        found = shutil.which("MNNConvert")
        if found:
            mnnconvert = found

    missing = []
    if not os.path.isfile(llmexport):
        missing.append(f"llmexport.py at {llmexport}")
    if not (os.path.isfile(mnnconvert) and os.access(mnnconvert, os.X_OK)):
        missing.append(f"an executable MNNConvert at {mnnconvert}")
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        missing.append(f"the exporter's python at {python}")
    if missing:
        raise QuantizationInfrastructureError(
            "The MNN quantization backend needs "
            + "; ".join(missing)
            + ". Run `bash scripts/setup_mnn_env.sh` to build the toolchain and its venv, or point "
            "SLM_MNN_ROOT / SLM_MNN_LLMEXPORT / SLM_MNN_CONVERT_BIN / SLM_MNN_PYTHON at an "
            "existing install."
        )
    return MnnToolchain(python=python, llmexport=llmexport, mnnconvert=mnnconvert, root=root)


def mnn_config_path(artifact_dir: str) -> str:
    """The runtime config pymnn is handed to load this artifact."""
    return os.path.join(artifact_dir, "config.json")


def mnn_validation_sidecar_path(artifact_dir: str) -> str:
    """Atomic validation-record path for an MNN artifact directory.

    Beside the directory rather than inside it, so the record cannot become part of what it
    describes — the fingerprint below hashes every file in the directory.
    """
    return f"{artifact_dir.rstrip(os.sep)}{_MNN_VALIDATION_SUFFIX}"


def missing_files(artifact_dir: str) -> list[str]:
    """Which required pieces of an MNN model are absent, for an error message worth reading."""
    if not os.path.isdir(artifact_dir):
        return [f"{artifact_dir} (directory does not exist)"]
    absent = [
        name for name in _REQUIRED_FILES
        if not os.path.isfile(os.path.join(artifact_dir, name))
    ]
    if not any(
        os.path.isfile(os.path.join(artifact_dir, name)) for name in _TOKENIZER_FILES
    ):
        absent.append(" or ".join(_TOKENIZER_FILES))
    return absent


def _artifact_files(artifact_dir: str) -> list[str]:
    """Every file in the artifact, in a stable order, so a fingerprint is reproducible."""
    collected = []
    for dirpath, dirnames, filenames in os.walk(artifact_dir):
        dirnames.sort()
        for name in sorted(filenames):
            collected.append(os.path.join(dirpath, name))
    return collected


def _fingerprint(artifact_dir: str) -> dict:
    """Size + SHA-256 of every file in the artifact, keyed by path relative to the directory.

    The GGUF equivalent hashes one file. An MNN model is only usable if the graph, the weights,
    the tokenizer and both configs agree with each other, so all of them are covered: a weight
    file swapped under an unchanged graph is exactly the kind of half-written artifact the GGUF
    sidecar exists to catch, and hashing only `llm.mnn.weight` would miss it.
    """
    files = {}
    total = 0
    for path in _artifact_files(artifact_dir):
        size = os.path.getsize(path)
        total += size
        files[os.path.relpath(path, artifact_dir)] = {
            "size": size,
            "sha256": _sha256_file(path),
        }
    return {"files": files, "total_size": total}


def weight_size_mb(artifact_dir: str) -> float:
    """On-disk size of what actually ships: the graph plus its weights.

    Deliberately NOT the whole directory. `llmexport.py` also leaves `export_args.json`,
    `llm.mnn.json` and the tokenizer behind, and the reference harness's budget check measured
    `llm.mnn` + `llm.mnn.weight` for the same reason: those two are the model, and comparing a
    directory total against a GGUF file size would overstate MNN by the tokenizer's weight.
    """
    total = 0.0
    for name in ("llm.mnn", "llm.mnn.weight"):
        path = os.path.join(artifact_dir, name)
        if os.path.isfile(path):
            total += _file_size_mb(path)
    return total


def export_from_model_spec(checkpoint_path: str, output_dir: str, quant: str) -> str:
    """Quantize a merged HF checkpoint into an MNN model directory and return its path.

    The MNN counterpart of `quantize.quantize_from_model_spec`, with the same contract: it either
    returns a complete artifact at the requested quantization or raises.

    Args:
        checkpoint_path: A merged full-precision HF checkpoint directory (or an HF snapshot).
        output_dir: Directory to place the MNN build under; the artifact is a subdirectory of it.
        quant: A pool selector's quant value — "Q4_K_M" (4-bit) or "Q8_0" (8-bit).
    """
    bits = quant_bit(quant)
    toolchain = resolve_toolchain()

    # ABSOLUTE PATHS, NOT THE CALLER'S. `llmexport.py` is launched from its own source directory
    # (see `cwd` below), and the pipeline hands relative paths — `evaluate_node` builds
    # `artifacts/mnn/<model>/<key>` relative to the project root it chdir'd into. Passing those
    # through verbatim wrote the whole model into
    # `<MNN>/transformers/llm/export/artifacts/mnn/...`, and the exporter exited 0 having done it,
    # so the only symptom was `exited 0 but the MNN artifact is incomplete` pointing at an empty
    # directory in the repo (run 40260162). The completeness check earned its keep; this is the
    # actual fix.
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    artifact_dir = os.path.join(output_dir, artifact_name(quant))

    # A previous attempt's half-written directory is rubble, not a starting point. The caller's
    # cache check has already accepted any artifact worth reusing by the time we get here.
    shutil.rmtree(artifact_dir, ignore_errors=True)
    os.makedirs(artifact_dir, exist_ok=True)

    source_size_mb = (
        _dir_size_mb(checkpoint_path)
        if os.path.isdir(checkpoint_path)
        else _file_size_mb(checkpoint_path)
    )
    command = [
        toolchain.python,
        toolchain.llmexport,
        "--path", checkpoint_path,
        "--export", "mnn",
        "--quant_bit", str(bits),
        "--quant_block", str(MNN_QUANT_BLOCK),
        # Never below the body's width: asking for an 8-bit model with a 4-bit lm_head would be
        # strictly worse than the default it overrides.
        "--lm_quant_bit", str(max(MNN_LM_QUANT_BIT, bits)),
        "--dst_path", artifact_dir,
        "--mnnconvert", toolchain.mnnconvert,
    ]
    print(
        f"      [quantize-mnn] llmexport.py → {bits}-bit / block {MNN_QUANT_BLOCK} "
        f"/ lm_head {max(MNN_LM_QUANT_BIT, bits)}-bit "
        f"({os.path.basename(checkpoint_path.rstrip(os.sep))} → {artifact_dir})"
    )
    error = _run_quant_tool(
        command,
        _export_timeout_s(source_size_mb),
        partial_output=artifact_dir,
        # llmexport.py imports its own `utils` package and resolves its default tool paths
        # relatively, so it is launched from its source directory — which is exactly why every
        # path above had to be made absolute first.
        cwd=os.path.dirname(toolchain.llmexport),
    )
    if error:
        raise QuantizationInfrastructureError(
            f"MNN export failed for {checkpoint_path!r} → {quant} ({bits}-bit): {error}"
        )

    absent = missing_files(artifact_dir)
    if absent:
        raise QuantizationInfrastructureError(
            f"llmexport.py exited 0 but the MNN artifact at {artifact_dir} is incomplete; "
            f"missing: {absent}"
        )
    return artifact_dir


def _validated_thread_count() -> int:
    """The MNN thread count this process would score with."""
    from training.slm_helpers import _MNN_THREADS

    return _MNN_THREADS


def _validated_backend() -> str:
    """The MNN device this process would score on — "cuda" or "cpu"."""
    from training.slm_helpers import mnn_backend_type

    return mnn_backend_type()


def _recorded_export_arg(artifact_dir: str, name: str):
    """One argument llmexport says it actually ran with, read back from what it wrote."""
    try:
        with open(
            os.path.join(artifact_dir, "export_args.json"), encoding="utf-8"
        ) as handle:
            return json.load(handle).get(name)
    except (OSError, ValueError):
        return None


def _recorded_quant_bit(artifact_dir: str) -> int | None:
    """The bit width llmexport says it actually used, read back from what it wrote.

    This is the check that makes "genuinely quantized" a verified claim rather than an assumption.
    `export_args.json` is the exporter's own record of its arguments, so a run that silently fell
    back to a different width (or to fp16 weights) is caught here instead of being reported as a
    4-bit model whose accuracy happens to look suspiciously good.
    """
    try:
        with open(
            os.path.join(artifact_dir, "export_args.json"), encoding="utf-8"
        ) as handle:
            return int(json.load(handle)["quant_bit"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def validate_and_record_mnn(
    artifact_dir: str,
    base_model: str | None = None,
    quant: str | None = None,
) -> dict:
    """Load the artifact with MNN, smoke-test generation, then record its file identity.

    Same division of labour as the GGUF path (B289): THE LOAD IS THE GATE, and the generation
    probe only reports. A small model given a trivial prompt is not a corrupt model, and three
    healthy runs were once killed by treating it as one; the eval score is the reliable detector
    and rollback is the reliable response.

    Unlike the GGUF path, this ALSO re-reads the exporter's own record of `--quant_bit` and
    refuses a mismatch, because MNN quantization is a flag rather than a file format: a 4-bit
    build and an fp16 build are the same five filenames, and nothing else downstream would notice.
    """
    absent = missing_files(artifact_dir)
    if absent:
        raise RuntimeError(
            f"MNN validation failed: incomplete artifact at {artifact_dir}; missing {absent}"
        )
    if quant is not None:
        expected_bits = quant_bit(quant)
        recorded = _recorded_quant_bit(artifact_dir)
        if recorded is not None and recorded != expected_bits:
            raise RuntimeError(
                f"MNN validation failed: {artifact_dir} was exported at {recorded}-bit but is "
                f"labelled {quant} ({expected_bits}-bit). Scoring it would attribute one "
                f"quantization's accuracy to another."
            )

    from training.slm_helpers import load_mnn_llm, mnn_runtime_versions

    before = _fingerprint(artifact_dir)
    llm = load_mnn_llm(artifact_dir)
    try:
        _smoke_test_generation(llm, artifact_dir, base_model)
        _assert_threaded_compute_is_sound(llm, artifact_dir, base_model)
    finally:
        release = getattr(llm, "reset", None)
        if callable(release):
            try:
                release()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        del llm

    after = _fingerprint(artifact_dir)
    if after != before:
        raise RuntimeError(
            f"MNN artifact changed during validation: {artifact_dir} "
            f"({before['total_size']} → {after['total_size']} bytes)"
        )
    record = {
        "schema_version": _MNN_VALIDATION_SCHEMA_VERSION,
        "quant": quant,
        "quant_bit": _recorded_quant_bit(artifact_dir),
        "quant_block": MNN_QUANT_BLOCK,
        # Recorded because it changes the weights: two artifacts labelled Q4_K_M with different
        # lm_head widths are different models, and the sidecar is the cache key.
        "lm_quant_bit": _recorded_export_arg(artifact_dir, "lm_quant_bit"),
        # The settings the artifact was VALIDATED under. "This loaded and decoded" is only ever a
        # claim about one configuration: MNN's thread count is a correctness setting on the CPU
        # (see `_assert_threaded_compute_is_sound`), and the device decides which kernels ran.
        "validated_threads": _validated_thread_count(),
        "validated_backend": _validated_backend(),
        "weight_size_mb": round(weight_size_mb(artifact_dir), 2),
        "fingerprint": after,
        "tool_versions": {
            **resolve_toolchain().versions(),
            **mnn_runtime_versions(),
        },
    }
    _atomic_write_json(mnn_validation_sidecar_path(artifact_dir), record)
    return record


def looks_degenerate(text: str | None) -> bool:
    """Whether a decoded string carries no content — markup and punctuation runs aside.

    Deliberately permissive: it separates "produced language" from "produced nothing", and does not
    judge quality. Markup comes out first — XML-ish tags like `</tool_call>` and the `<|...|>`
    spelling of `<|im_end|>` — since output made only of those carries no content.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    without_markup = re.sub(r"<\|[^|>]*\|>|</?[A-Za-z_][\w:.-]*/?>", "", stripped)
    return not re.search(r"\w", without_markup)


# Prompt length for the cross-check below. It has to be long enough to tile a prefill — the 16-token
# `Hello` smoke test passes on an artifact that then decodes 1,000 garbage rows — and 512 tokens is
# measured sufficient to SEPARATE the configurations, which is all this check needs. It is not
# trying to reproduce the garbage: on synthetic filler even 1,536 tokens decodes readable text at
# the bad setting. What it detects is the two configurations DISAGREEING, and that shows up at 512.
_THREAD_CHECK_PROMPT_TOKENS = int(os.environ.get("SLM_MNN_THREAD_CHECK_TOKENS", "512"))
# Generated tokens compared between the two configurations. Greedy decoding is deterministic, so a
# handful is enough to separate "wrong compute" from "the same answer" — and keeping it small keeps
# the number of chances for a legitimate near-tie small too.
_THREAD_CHECK_NEW_TOKENS = 8
# The escape hatch. Documented because the invariant below is strict by design and an operator who
# has looked at a mismatch and judged it benign should not have to edit code to proceed.
_THREAD_CHECK_ENABLED = os.environ.get("SLM_MNN_THREAD_CHECK", "1") == "1"
# The GPU's equivalent: does the device decode usable text where the CPU does? Same escape hatch,
# same reason. See `_assert_gpu_agrees_with_cpu`.
_GPU_CROSS_CHECK_ENABLED = os.environ.get("SLM_MNN_GPU_CROSS_CHECK", "1") == "1"


def _long_probe_prompt(llm, base_model: str | None) -> str:
    """A prompt of at least `_THREAD_CHECK_PROMPT_TOKENS` tokens, measured with MNN's own tokenizer.

    Built by repetition rather than taken from a task: this check belongs to the artifact, not to
    whatever is being evaluated, and it has to work for a caller that has no eval set in hand.
    """
    filler = (
        "The assistant answers questions about scheduling, banking, travel and small talk. "
        "It replies with exactly one label and no explanation. "
    )
    encode = getattr(llm, "tokenizer_encode", None)
    body = filler
    if callable(encode):
        for _ in range(12):
            if len(encode(body)) >= _THREAD_CHECK_PROMPT_TOKENS:
                break
            body += filler
    else:
        body = filler * 40
    question = body + "\nWhich label applies? Answer with one label."
    if base_model:
        from training.slm_helpers import _serving_prompt_prefix

        return _serving_prompt_prefix(question, base_model)
    return question


def _assert_gpu_agrees_with_cpu(llm, artifact_dir: str, base_model: str | None) -> None:
    """Refuse a GPU artifact that decodes nothing usable where the CPU decodes real text.

    The GPU cross-check, and the reason the eval can be trusted to a device the phone does not use.
    Both MNN/CUDA failures found so far are exactly this asymmetry:
      * the unregistered-backend fallback decoded `'accept<|endoftext|><|endoftext|>...'` while the
        CPU decoded `'accept_reservations'`;
      * MNN's CUDA int8 weight-only kernel scored format_valid 0.0000 with `'<|endoftext|>'` on
        every row where the CPU scored 0.2667.

    WHY DEGENERACY RATHER THAN EQUALITY, which is the opposite of the thread check's rule. Two
    thread counts compute the same arithmetic in a different order, so their greedy output must
    match token for token. Two DEVICES do not: fp16 accumulation on CUDA against fp32 on the CPU
    genuinely flips near-ties, and it is measured — the same 4-bit artifact scored 0.0600 on CUDA
    and 0.0317 on the CPU over 100 rows. An equality check here would kill healthy runs, so the
    invariant is the weaker one that still catches every failure actually observed: one side
    produces words and the other does not.

    Costs one CPU load plus one prefill per artifact build (the GPU model is already loaded), which
    is a fraction of the export that precedes it. `SLM_MNN_GPU_CROSS_CHECK=0` turns it off.
    """
    from training.slm_helpers import load_mnn_llm, mnn_backend_type, mnn_generate

    if not _GPU_CROSS_CHECK_ENABLED:
        print(
            f"[quantize-mnn]   GPU cross-check DISABLED (SLM_MNN_GPU_CROSS_CHECK=0); scoring on "
            f"{mnn_backend_type()} without comparing against the CPU"
        )
        return
    device = mnn_backend_type()
    prompt = _long_probe_prompt(llm, base_model)
    on_device = mnn_generate(llm, prompt, max_new_tokens=_THREAD_CHECK_NEW_TOKENS)
    reference_llm = load_mnn_llm(
        artifact_dir, max_new_tokens=_THREAD_CHECK_NEW_TOKENS, backend_type="cpu"
    )
    try:
        on_cpu = mnn_generate(
            reference_llm, prompt, max_new_tokens=_THREAD_CHECK_NEW_TOKENS
        )
    finally:
        del reference_llm

    if looks_degenerate(on_device) and not looks_degenerate(on_cpu):
        raise QuantizationInfrastructureError(
            f"MNN decodes nothing usable on {device} where the CPU decodes real text: "
            f"{on_device.strip()[:60]!r} against {on_cpu.strip()[:60]!r} on a "
            f"{_THREAD_CHECK_PROMPT_TOKENS}-token prompt. The artifact is fine and the device is "
            f"not. Both MNN/CUDA faults seen so far look like this — an unregistered backend "
            f"falling back to the CPU while configured as a GPU, and the CUDA int8 weight-only "
            f"kernel, which scored format_valid 0.0000 where the CPU scored 0.2667. Refusing to "
            f"score, because an eval would record it as a model result. Set "
            f"SLM_MNN_BACKEND_TYPE=cpu to evaluate on the CPU, or SLM_MNN_GPU_CROSS_CHECK=0 to "
            f"proceed anyway."
        )
    if looks_degenerate(on_device) and looks_degenerate(on_cpu):
        print(
            f"[quantize-mnn]   ⚠ the long-prompt probe decoded nothing usable on either {device} "
            f"or the CPU ({on_device.strip()[:40]!r}). That is a statement about the model, not "
            f"the device, so the run continues and the eval decides."
        )
        return
    print(
        f"[quantize-mnn]   GPU cross-check passed: {device} decodes usable text on a "
        f"{_THREAD_CHECK_PROMPT_TOKENS}-token prompt"
        + ("" if on_device.strip() == on_cpu.strip() else
           " (differing from the CPU's, which is expected — fp16 accumulation flips near-ties)")
    )


def _assert_threaded_compute_is_sound(llm, artifact_dir: str, base_model: str | None) -> None:
    """Refuse an artifact that decodes one prompt two ways depending on how the work was split.

    THE INVARIANT: greedy decoding is deterministic, so the same prompt must produce the same
    tokens whether MNN splits the work across 4 threads or 14. If it does not, the split is
    corrupting compute, and that is a fact about the runtime rather than a judgement about the
    model.

    WHY IT IS FATAL, unlike the smoke test above. In MNN the thread count is a correctness setting.
    `Qwen/Qwen3.5-0.8B@Q4_K_M` at `thread_num=14` answered all 1,000 CLINC150 rows with
    `%+!!!!!!!!!!` and scored 0.0000 with format_valid 0.0000, while the same artifact at 1, 2, 4,
    8, 10, 12, 13 and 16 threads answered them correctly and identically (run 40260927, then a
    per-process sweep). There is nothing in the output to detect: the logits are finite, plausibly
    scaled, and only the argmax is wrong — so a 0.0000 enters the DAG as a real measurement after
    35 minutes of eval.

    WHY DISAGREEMENT RATHER THAN "LOOKS LIKE GARBAGE". Measured on the failing artifact with a
    512-token probe: 8, 10, 12, 13 and 16 threads each agree with the 4-thread reference
    token-for-token, and only 14 differs — but on synthetic filler the bad setting still produces
    READABLE text, so a degeneracy test passes it. Disagreement separates the configurations
    cleanly where degeneracy does not.

    THE RISK, stated plainly: a different work split changes float reduction order, so a genuine
    near-tie could flip a token and fail this check on a healthy artifact. That is why only
    `_THREAD_CHECK_NEW_TOKENS` tokens are compared, and why `SLM_MNN_THREAD_CHECK=0` exists for an
    operator who has looked at a mismatch and judged it benign. A false positive costs one run and
    names its own cause; a false negative costs a silently wrong number in a results table.

    Skipped when the configured count already IS the reference: there would be nothing to compare.
    """
    from training.slm_helpers import (
        MNN_REFERENCE_THREADS,
        _MNN_THREADS,
        load_mnn_llm,
        mnn_backend_type,
        mnn_generate,
    )

    if mnn_backend_type() != "cpu":
        # The thread count governs MNN's CPU work split, so on a GPU run there is nothing to
        # compare — two thread counts on CUDA exercise a setting neither side uses. The GPU has its
        # own cross-check, against the CPU, which is the device that ships.
        print(
            f"[quantize-mnn]   thread cross-check not applicable on the "
            f"{mnn_backend_type()} backend (it governs the CPU work split only)"
        )
        _assert_gpu_agrees_with_cpu(llm, artifact_dir, base_model)
        return
    if not _THREAD_CHECK_ENABLED:
        print(
            f"[quantize-mnn]   thread cross-check DISABLED (SLM_MNN_THREAD_CHECK=0); scoring at "
            f"thread_num={_MNN_THREADS} without verifying it against {MNN_REFERENCE_THREADS}"
        )
        return
    if _MNN_THREADS == MNN_REFERENCE_THREADS:
        return
    prompt = _long_probe_prompt(llm, base_model)
    configured = mnn_generate(llm, prompt, max_new_tokens=_THREAD_CHECK_NEW_TOKENS)
    # The configured model is ALREADY loaded, which is the only order in which this comparison is
    # valid: MNN's thread pool is process-global and a later load cannot raise the count, only
    # lower it. See `load_mnn_llm`.
    reference_llm = load_mnn_llm(
        artifact_dir,
        max_new_tokens=_THREAD_CHECK_NEW_TOKENS,
        threads=MNN_REFERENCE_THREADS,
    )
    try:
        reference = mnn_generate(
            reference_llm, prompt, max_new_tokens=_THREAD_CHECK_NEW_TOKENS
        )
    finally:
        del reference_llm

    if configured.strip() == reference.strip():
        print(
            f"[quantize-mnn]   thread cross-check passed: {_MNN_THREADS} threads agrees with "
            f"{MNN_REFERENCE_THREADS} on a {_THREAD_CHECK_PROMPT_TOKENS}-token prompt"
        )
        return
    if looks_degenerate(configured) and looks_degenerate(reference):
        # Neither side produced content, so there is nothing to compare and no threading claim to
        # make. This is the B289 case — a statement about the model — and it stays a warning.
        print(
            f"[quantize-mnn]   ⚠ the long-prompt probe decoded nothing usable at either "
            f"{_MNN_THREADS} or {MNN_REFERENCE_THREADS} threads ({configured.strip()[:40]!r}). "
            f"That is a statement about the model, not the threading, so the run continues and the "
            f"eval decides."
        )
        return
    raise QuantizationInfrastructureError(
        f"MNN decodes the same prompt differently at thread_num={_MNN_THREADS} and at "
        f"{MNN_REFERENCE_THREADS}: {configured.strip()[:60]!r} against "
        f"{reference.strip()[:60]!r} on a {_THREAD_CHECK_PROMPT_TOKENS}-token prompt. Greedy "
        f"decoding cannot depend on the work split, so the split is corrupting compute — the "
        f"artifact is fine and the thread count is not (measured on Qwen3.5-0.8B: 14 wrong, "
        f"1/2/4/8/10/12/13/16 right, with 14 scoring 0.0000 on 1,000 rows). Lower SLM_MNN_THREADS. "
        f"Refusing to score rather than recording a real-looking number, because nothing later in "
        f"the run could tell this from a bad model: the logits are finite and only the answer is "
        f"wrong. Set SLM_MNN_THREAD_CHECK=0 to proceed anyway."
    )


def _smoke_test_generation(llm, artifact_dir: str, base_model: str | None) -> str | None:
    """Generate a few tokens and REPORT — never raise — if the result looks degenerate.

    Word-for-word the same policy as `quantize._smoke_test_generation`, including using the
    served prompt rendering rather than a bare string: asking a small instruct model to continue
    `Hello` is not how it is ever used, and judging it on that produced three false positives
    and zero true ones on the GGUF side.
    """
    from training.slm_helpers import mnn_generate

    prompt = "Hello"
    if base_model:
        from training.slm_helpers import _serving_prompt_prefix

        prompt = _serving_prompt_prefix("Hello", base_model)

    try:
        text = mnn_generate(llm, prompt, max_new_tokens=16)
    except Exception as exc:  # noqa: BLE001 - report and let the eval be the judge
        print(
            f"[quantize-mnn]   ⚠ MNN smoke test could not decode ({type(exc).__name__}: {exc}). "
            f"Continuing to eval, which will show a near-zero score if the artifact really is "
            f"bad. Path: {artifact_dir}"
        )
        return None

    if looks_degenerate(text):
        print(
            f"[quantize-mnn]   ⚠ MNN smoke test produced no word characters: {text!r}. This is "
            f"often benign — a small base model given a trivial prompt — so the run continues and "
            f"the eval decides. Path: {artifact_dir}"
        )
    else:
        print(f"[quantize-mnn]   smoke test decoded: {text.strip()[:120]!r}")
    return text


def validated_mnn_cache_hit(artifact_dir: str) -> bool:
    """Whether the artifact exactly matches a successful load-validation record.

    Hash-checked rather than mtime-checked, for the reason the GGUF version gives: a partially
    written artifact and a complete one are indistinguishable by existence alone, and mistaking
    one for a warm cache hit is how a run scores rubble.
    """
    sidecar_path = mnn_validation_sidecar_path(artifact_dir)
    if not os.path.isdir(artifact_dir) or not os.path.isfile(sidecar_path):
        return False
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("schema_version") != _MNN_VALIDATION_SCHEMA_VERSION:
            return False
        if not isinstance(record.get("tool_versions"), dict) or not record["tool_versions"]:
            return False
        # The EXPORT SETTINGS are part of the key, not just the file contents. `quant_block` and
        # `lm_quant_bit` change the weights, and the files on disk cannot say they were built with a
        # different value of either — so a hit here would silently score the previous setting's
        # model under the new one's name, which is B161 with a different label. The recorded MNN
        # commit is covered by `tool_versions` for the same reason.
        if record.get("quant_block") != MNN_QUANT_BLOCK:
            return False
        recorded_lm_bits = record.get("lm_quant_bit")
        expected_bits = record.get("quant_bit")
        if recorded_lm_bits is not None and expected_bits is not None:
            if recorded_lm_bits != max(MNN_LM_QUANT_BIT, expected_bits):
                return False
        return record.get("fingerprint") == _fingerprint(artifact_dir)
    except (OSError, TypeError, ValueError):
        return False


def invalidate_mnn_cache(artifact_dir: str) -> None:
    """Remove a derived MNN artifact and its validation record, and nothing else."""
    shutil.rmtree(artifact_dir, ignore_errors=True)
    try:
        os.remove(mnn_validation_sidecar_path(artifact_dir))
    except FileNotFoundError:
        pass


__all__ = [
    "MNN_QUANT_BLOCK",
    "MnnToolchain",
    "artifact_name",
    "default_mnn_root",
    "export_from_model_spec",
    "invalidate_mnn_cache",
    "missing_files",
    "mnn_config_path",
    "mnn_validation_sidecar_path",
    "quant_bit",
    "resolve_toolchain",
    "validate_and_record_mnn",
    "validated_mnn_cache_hit",
    "weight_size_mb",
]
