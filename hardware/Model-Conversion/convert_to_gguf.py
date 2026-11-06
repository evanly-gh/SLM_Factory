#!/usr/bin/env python3
"""
convert_to_gguf.py — HuggingFace → quantized GGUF pipeline for Android/llama.cpp deployment

Pipeline stages:
  1. Download  — pull model weights + config from HuggingFace Hub (or use a local path)
  2. Convert   — turn the HF checkpoint into a lossless bf16 GGUF via llama.cpp's converter
  3. Quantize  — compress to Q4_K_M / Q8_0 using llama-quantize, from a temporary f16 GGUF
  4. Validate  — load every artifact with llama-cpp-python, smoke-test it, and record a hash sidecar;
                 a load failure is fatal
  5. Report    — file sizes and RAM estimates per quantization level
  6. Deploy    — (optional) push the quantized GGUFs to a connected Android phone via ADB
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — paths and quantization configuration
# ---------------------------------------------------------------------------

# llama.cpp is expected to be cloned here. All conversion and quantization
# tools are built from this source tree, the same one the SmolChat app is
# built from, so the GGUF format matches the runtime that loads it.
LLAMA_CPP_DIR = Path(__file__).resolve().parent.parent / "SmolChat-Android" / "llama.cpp"

# Python script that converts a HuggingFace checkpoint directory into GGUF format.
# Ships with llama.cpp; handles both safetensors and .bin weight files.
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

# Where cmake leaves llama-quantize: single-config generators (make, ninja) write
# build/bin/, multi-config ones (Visual Studio) write build/bin/Release/.
QUANTIZE_BIN_CANDIDATES = [
    LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
    LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize.exe",
    LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",
]

# Quantized levels, smallest first. Q4_K_M is the sweet spot for most Android
# phones; Q8_0 is near-lossless but needs roughly twice the RAM.
QUANT_LEVELS = ["Q4_K_M", "Q8_0"]

# The full-precision deliverable. bf16 is a bit-exact copy of the bf16 weights
# most HF checkpoints ship, where f16 would round the smallest weights.
BASE_LEVEL = "BF16"
BASE_OUTTYPE = "bf16"

# Quantized levels are built from an f16 GGUF rather than the bf16 one, so they
# are byte-for-byte what `convert_hf_to_gguf --outtype f16 | llama-quantize`
# produces: tensors llama-quantize leaves unquantized keep the source's dtype.
# The f16 file is an intermediate and is deleted once every level is built.
INTERMEDIATE_OUTTYPE = "f16"

# Bits-per-parameter used to estimate loaded model RAM.
QUANT_BPP = {
    "BF16":   2.0,
    "Q8_0":   1.0,
    "Q4_K_M": 0.5625,
}

# Maps each quantization level to a human-readable RAM range and target device
# description, used in the validation report to guide deployment decisions.
RAM_RECOMMENDATIONS = {
    "Q4_K_M": ("< 4 GB RAM", "Moto G Power 2021 and similar budget phones"),
    "Q8_0":   ("6 GB+ RAM", "Flagship Android devices"),
}

# Weight files convert_hf_to_gguf.py cannot use, plus pre-built GGUFs and the
# raw Meta/Mistral checkpoints some repos ship alongside the HF weights.
HF_SNAPSHOT_IGNORE_PATTERNS = [
    "*.gguf",
    "original/*",
    "*.pth",
    "consolidated*",
    "*.msgpack",
    "flax_model*",
    "tf_model*",
    "rust_model*",
]

GGUF_VALIDATION_SCHEMA_VERSION = 1
GGUF_VALIDATION_SUFFIX = ".validation.json"

# convert_hf_to_gguf / llama-quantize wall-clock ceiling, scaled by the bytes
# being read: a flat 600s is right for a 360M model and marginal for a 4B one
# on a contended filesystem. SLM_QUANT_TIMEOUT_S, when set, is used verbatim.
QUANT_TIMEOUT_FLOOR_S = 600
QUANT_TIMEOUT_S_PER_GB = int(os.environ.get("SLM_QUANT_TIMEOUT_S_PER_GB", "240"))
QUANT_TIMEOUT_OVERRIDE_S = os.environ.get("SLM_QUANT_TIMEOUT_S")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_active_venv() -> None:
    """Re-exec under $VIRTUAL_ENV's own interpreter if this process isn't already using it.

    Activating a venv only redirects bare `python`/`python3` lookups via PATH.
    If this script was launched with an interpreter path that bypasses PATH
    (shell history, an IDE run config, a cron job, etc.), sys.executable ends up
    pointing at whatever actually ran this file — not the active venv — even
    though $VIRTUAL_ENV is set and the shell prompt shows it active. Every
    subprocess this script spawns (convert_hf_to_gguf.py) inherits sys.executable,
    so that mismatch is what makes convert_hf_to_gguf.py run under a Python whose
    numpy/torch conflict with the venv's, producing "OMP: Error #15".
    """
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return

    if Path(sys.prefix).resolve() == Path(venv).resolve():
        return  # already running under the active venv

    for candidate in (Path("bin") / "python3", Path("bin") / "python", Path("Scripts") / "python.exe"):
        venv_python = Path(venv) / candidate
        if venv_python.exists():
            print(
                f"[INFO] $VIRTUAL_ENV is {venv} but this process is running under "
                f"{sys.executable}; re-executing under {venv_python} to match."
            )
            sys.stdout.flush()
            os.execv(str(venv_python), [str(venv_python), *sys.argv])

    print(
        f"[WARN] $VIRTUAL_ENV is set to {venv} but no python/python3 found there; "
        f"continuing under {sys.executable}.",
        file=sys.stderr,
    )


def run(cmd: list[str], cwd: Path | None = None, capture: bool = False, env: dict | None = None) -> subprocess.CompletedProcess:
    """Execute a shell command, printing it first so the user can see what's running.

    capture=True suppresses stdout/stderr (used when we need to inspect output
    programmatically, e.g. parsing `adb devices`). Otherwise output streams
    live to the terminal so long-running steps like quantization show progress.
    env, if given, replaces the subprocess's environment (callers pass a copy
    of os.environ plus overrides — None here means "inherit unchanged").
    """
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture,
        text=True,
        env=env,
    )


def require(condition: bool, msg: str) -> None:
    """Assert a condition or exit with a clear error message.

    Used as a lightweight guard for preconditions (files exist, commands
    succeed) where continuing would cause a confusing downstream failure.
    """
    if not condition:
        print(f"\n[ERROR] {msg}", file=sys.stderr)
        sys.exit(1)


def file_size_mb(path: Path) -> float:
    """Return the size of a file in megabytes."""
    return path.stat().st_size / (1024 ** 2)


def dir_size_mb(path: Path) -> float:
    """Return the total size of every file under a directory in megabytes."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            total += os.path.getsize(os.path.join(dirpath, name))
    return total / (1024 ** 2)


def subprocess_timeout_s(source_size_mb: float) -> int:
    """Wall-clock ceiling for one llama.cpp subprocess over a source of `source_size_mb`."""
    if QUANT_TIMEOUT_OVERRIDE_S:
        return int(QUANT_TIMEOUT_OVERRIDE_S)
    scaled = (source_size_mb / 1024.0) * QUANT_TIMEOUT_S_PER_GB
    return max(QUANT_TIMEOUT_FLOOR_S, int(scaled))


def run_quant_tool(cmd: list[str], timeout_s: int, partial_output: Path, env: dict | None = None) -> str | None:
    """Run one toolchain subprocess, retrying ONCE at double the ceiling on timeout.

    A timeout is a statement about the filesystem, not the model, so it is
    retried; a non-zero exit means the tool rejected the input and would fail
    the same way again, so it is not. A killed or failed attempt can leave a
    truncated file behind, which is deleted so it is never mistaken for output.

    Returns None on success or an error string.
    """
    name = Path(cmd[1] if cmd[0] == sys.executable else cmd[0]).name

    def discard_partial():
        try:
            partial_output.unlink()
        except FileNotFoundError:
            pass

    for attempt, ceiling in enumerate((timeout_s, timeout_s * 2), start=1):
        print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
        try:
            result = subprocess.run(cmd, text=True, env=env, timeout=ceiling)
        except subprocess.TimeoutExpired:
            discard_partial()
            if attempt == 1:
                print(f"[WARN] {name} exceeded {ceiling}s; retrying once at {ceiling * 2}s "
                      f"(filesystem contention, not a bad model)")
                continue
            return (
                f"{name} timed out twice ({timeout_s}s then {ceiling}s). Raise the ceiling "
                f"with SLM_QUANT_TIMEOUT_S or reduce concurrent jobs on the same disk."
            )
        except FileNotFoundError as exc:
            discard_partial()
            return f"{name} failed: {exc}"
        if result.returncode != 0:
            discard_partial()
            return f"{name} exited with code {result.returncode}"
        return None
    return None


# ---------------------------------------------------------------------------
# Real-load validation + hash-based caching for GGUF outputs
#
# Every produced GGUF is loaded with llama-cpp-python and, only if that load
# succeeds, a sha256 sidecar (<file>.validation.json) is written. A cached file
# is reused only when its sidecar matches its current size and hash; anything
# else is deleted and rebuilt, so a truncated or hand-edited GGUF is never
# silently reused. The load is the gate; the generation smoke test only warns.
# ---------------------------------------------------------------------------

def gguf_validation_sidecar_path(gguf_path: Path) -> Path:
    return gguf_path.with_name(gguf_path.name + GGUF_VALIDATION_SUFFIX)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def require_llama_cpp_python() -> None:
    """Fail before any conversion work if the validator can't run at all."""
    try:
        import llama_cpp  # noqa: F401
    except (ImportError, TypeError) as exc:
        require(False, f"GGUF validation requires llama-cpp-python to perform a real model load "
                       f"(pip install llama-cpp-python==0.3.34): {exc}")


def _qwen_no_think_prompt(prompt: str, base_model: str) -> str:
    """ChatML rendering of a Qwen user turn with thinking disabled.

    Qwen3-4B-Instruct-2507 is thinking-free and its template emits a bare
    assistant prefix; the hybrid Qwen3/Qwen3.5 templates disable thinking by
    pre-filling an empty think block.
    """
    prefix = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if base_model == "Qwen/Qwen3-4B-Instruct-2507":
        return prefix
    return prefix + "<think>\n\n</think>\n\n"


def _smoke_test_generation(model, gguf_path: Path, base_model: str | None = None) -> str | None:
    """Generate a few tokens and report — do NOT raise — if the result looks degenerate.

    A small base model given a trivial prompt often decodes punctuation, which
    is not a corrupt artifact; benchmarks are the real detector. This only
    leaves a legible note in the log.
    """
    prompt = "Hello"
    if base_model and "qwen" in base_model.lower():
        prompt = _qwen_no_think_prompt("Hello", base_model)

    try:
        response = model(prompt, max_tokens=16, temperature=0.0, echo=False)
        text = response["choices"][0]["text"]
    except Exception as exc:  # noqa: BLE001 - report and let the benchmark be the judge
        print(f"[VALIDATE] ⚠ smoke test could not decode ({type(exc).__name__}: {exc}). Path: {gguf_path}")
        return None

    stripped = text.strip()
    without_markup = re.sub(r"<\|[^|>]*\|>|</?[A-Za-z_][\w:.-]*/?>", "", stripped)
    if not stripped or not re.search(r"\w", without_markup):
        print(f"[VALIDATE] ⚠ smoke test produced no word characters: {text!r}. Often benign for a "
              f"small model given a trivial prompt. Path: {gguf_path}")
    return text


def validate_and_record_gguf(gguf_path: Path, base_model: str | None = None) -> dict:
    """Load every tensor with llama.cpp, smoke-test generation, then record file identity.

    Raises RuntimeError if the file is missing, empty, fails to load, or
    changes size during the load.
    """
    if not gguf_path.is_file():
        raise RuntimeError(f"GGUF validation failed: file not found: {gguf_path}")
    initial_size = gguf_path.stat().st_size
    if initial_size <= 0:
        raise RuntimeError(f"GGUF validation failed: empty file: {gguf_path}")

    try:
        import llama_cpp
    except (ImportError, TypeError) as exc:
        raise RuntimeError("GGUF validation requires llama-cpp-python to perform a real model load") from exc

    model = None
    try:
        model = llama_cpp.Llama(
            model_path=str(gguf_path.resolve()), n_ctx=128, n_batch=16, n_gpu_layers=0, verbose=False
        )
    except Exception as exc:  # noqa: BLE001 - preserve llama.cpp loader detail
        raise RuntimeError(f"GGUF model-load validation failed: {exc}") from exc

    try:
        _smoke_test_generation(model, gguf_path, base_model)
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()

    final_size = gguf_path.stat().st_size
    if final_size != initial_size:
        raise RuntimeError(f"GGUF changed during validation: {initial_size} -> {final_size} bytes")

    record = {
        "schema_version": GGUF_VALIDATION_SCHEMA_VERSION,
        "file_size": final_size,
        "sha256": _sha256_file(gguf_path),
        "tool_versions": {
            "llama_cpp_python": str(getattr(llama_cpp, "__version__", "unknown")),
            "python": platform.python_version(),
        },
    }
    _atomic_write_json(gguf_validation_sidecar_path(gguf_path), record)
    return record


def validated_gguf_cache_hit(gguf_path: Path) -> bool:
    """True only if gguf_path exactly matches a successful load-validation record."""
    sidecar_path = gguf_validation_sidecar_path(gguf_path)
    if not gguf_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            record = json.load(handle)
        return (
            record.get("schema_version") == GGUF_VALIDATION_SCHEMA_VERSION
            and record.get("file_size") == gguf_path.stat().st_size
            and isinstance(record.get("tool_versions"), dict)
            and bool(record["tool_versions"])
            and record.get("sha256") == _sha256_file(gguf_path)
        )
    except (OSError, TypeError, ValueError):
        return False


def invalidate_gguf_cache(gguf_path: Path) -> None:
    """Remove only a derived GGUF and its validation record."""
    for path in (gguf_path, gguf_validation_sidecar_path(gguf_path)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def reuse_or_invalidate(gguf_path: Path, step: str) -> bool:
    """Return True if a validated cached GGUF can be reused; otherwise delete it."""
    if validated_gguf_cache_hit(gguf_path):
        print(f"[{step}] {gguf_path.name} already exists and passed validated-cache check, skipping.")
        return True
    if gguf_path.exists():
        print(f"[{step}] {gguf_path.name} exists but is unvalidated or changed - rebuilding.")
    invalidate_gguf_cache(gguf_path)
    return False


def validate_or_exit(gguf_path: Path, base_model: str | None) -> None:
    """Validate a freshly built GGUF; on failure delete it and stop the pipeline."""
    try:
        validate_and_record_gguf(gguf_path, base_model)
    except RuntimeError as exc:
        invalidate_gguf_cache(gguf_path)
        require(False, f"{gguf_path.name} failed validation and was removed: {exc}")
    print(f"[VALIDATE] ✓ {gguf_path.name} loaded and recorded.")


def check_disk_space(path: Path, required_gb: float) -> bool:
    """Warn if the filesystem holding `path` has less than `required_gb` free.

    Returns False when space is low so callers can decide whether to abort.
    Only a warning (not a hard stop) because the estimate is rough.
    """
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024 ** 3)
    if available_gb < required_gb:
        print(f"[WARN] Only {available_gb:.1f} GB free at {path}; need ~{required_gb:.1f} GB")
        return False
    return True


def estimate_params_from_config(model_dir: Path) -> int | None:
    """Approximate total parameter count by reading the model's config.json.

    The formula covers the dominant cost terms (token embeddings + attention
    projections + feed-forward layers). It's intentionally rough — the goal is
    a plausible RAM estimate, not an exact count.
    Returns None if config.json is missing or malformed.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        hidden = cfg.get("hidden_size", 0)
        layers = cfg.get("num_hidden_layers", 0)
        vocab  = cfg.get("vocab_size", 0)
        inter  = cfg.get("intermediate_size", hidden * 4)
        # embedding table + per-layer attention (QKV + O projections) + FFN
        params = vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * inter)
        return max(params, 1)
    except Exception:
        return None


def estimate_ram_gb(params: int | None, quant: str) -> str:
    """Convert a parameter count + quantization level into a human-readable RAM estimate.

    Multiplies params by bits-per-parameter for the given quant, then converts
    to GB. Returns 'unknown' when the param count couldn't be determined.
    """
    if params is None:
        return "unknown"
    bpp = QUANT_BPP.get(quant, 0.5)
    ram_gb = (params * bpp) / (1024 ** 3)
    return f"~{ram_gb:.1f} GB"


def model_prefix(model_arg: str) -> str:
    """Derive a clean, lowercase filename prefix from a HuggingFace model ID or local path.

    Example: "Qwen/Qwen2.5-0.5B-Instruct" → "qwen2.5-0.5b-instruct"
    This prefix is shared by all output files for a given model so they can
    coexist in the same output directory without colliding.
    """
    # Use the final path component whether this is a HF ID ("org/name") or a local path
    name = Path(model_arg).name if Path(model_arg).exists() else model_arg.split("/")[-1]
    return name.lower().replace("_", "-").replace(" ", "-")


# ---------------------------------------------------------------------------
# 1. DOWNLOAD
# ---------------------------------------------------------------------------

def resolve_model(model_arg: str, output_dir: Path) -> Path:
    """Return a local directory containing the model's weights and config.

    If `model_arg` is already a local directory, use it as-is. Otherwise treat
    it as a HuggingFace repo ID and download the snapshot at the repo's current
    commit, skipping weight files convert_hf_to_gguf.py can't use.
    """
    local = Path(model_arg)
    if local.exists() and local.is_dir():
        print(f"[DOWNLOAD] Using local model at {local}")
        return local.resolve()

    print(f"[DOWNLOAD] Fetching {model_arg} from HuggingFace Hub …")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import model_info, snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    # Replace "/" with "__" so the repo ID can be used as a directory name
    model_name_safe = model_arg.replace("/", "__")
    dest = output_dir / model_name_safe

    # Rough disk-space check: assume up to 10 GB of weights plus headroom for the GGUF
    check_disk_space(output_dir, 15.0)

    revision = None
    try:
        revision = getattr(model_info(model_arg), "sha", None)
    except Exception:  # noqa: BLE001 - snapshot_download still works from an offline cache
        pass

    try:
        path = snapshot_download(
            repo_id=model_arg,
            revision=revision,
            local_dir=str(dest),
            ignore_patterns=HF_SNAPSHOT_IGNORE_PATTERNS,
        )
        print(f"[DOWNLOAD] Saved to {path} (revision {revision or 'unknown'})")
        return Path(path).resolve()
    except Exception as e:
        msg = str(e)
        if "404" in msg or "not found" in msg.lower():
            print(f"[ERROR] Model '{model_arg}' not found on HuggingFace. Check the model ID.", file=sys.stderr)
        else:
            print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2. CONVERT
# ---------------------------------------------------------------------------

def detect_weight_format(model_dir: Path) -> str:
    """Determine whether the model uses .safetensors or .bin weights.

    safetensors is the newer, safer HuggingFace format and is preferred.
    .bin files are legacy PyTorch checkpoints. Both are supported by
    convert_hf_to_gguf.py; we just need to know which is present so we
    can report it.
    """
    safetensors = list(model_dir.glob("*.safetensors"))
    bins = list(model_dir.glob("pytorch_model*.bin"))
    if safetensors:
        return "safetensors"
    if bins:
        return "bin"
    require(False, f"No .safetensors or .bin weight files found in {model_dir}")


def convert_hf(model_dir: Path, out_path: Path, outtype: str, source_size_mb: float) -> None:
    """Run convert_hf_to_gguf.py once, writing `out_path` at `outtype`; exit on failure."""
    require(CONVERT_SCRIPT.exists(), f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}")

    # The output is roughly the same size as the source weights
    check_disk_space(out_path.parent, source_size_mb / 1024 * 2 + 2)

    # convert_hf_to_gguf.py imports torch + numpy; if two copies of libomp end up
    # loaded (e.g. via mismatched wheel builds) the process aborts with
    # "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
    # initialized." KMP_DUPLICATE_LIB_OK=TRUE is the known workaround.
    convert_env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    cmd = [sys.executable, str(CONVERT_SCRIPT), str(model_dir), "--outfile", str(out_path), "--outtype", outtype]
    error = run_quant_tool(cmd, subprocess_timeout_s(source_size_mb), partial_output=out_path, env=convert_env)
    require(error is None, f"GGUF conversion to {outtype} failed: {error}")
    require(out_path.exists(), f"Expected {out_path} after conversion but it was not created.")
    print(f"[CONVERT] ✓ {out_path.name}  ({file_size_mb(out_path):.0f} MB)")


def convert_to_base(model_dir: Path, output_dir: Path, prefix: str, base_model: str | None,
                    source_size_mb: float) -> Path:
    """Produce the validated bf16 GGUF deliverable, reusing a validated cached copy."""
    base_path = output_dir / f"{prefix}-{BASE_OUTTYPE}.gguf"
    if reuse_or_invalidate(base_path, "CONVERT"):
        return base_path
    convert_hf(model_dir, base_path, BASE_OUTTYPE, source_size_mb)
    validate_or_exit(base_path, base_model)
    return base_path


# ---------------------------------------------------------------------------
# 3. QUANTIZE
# ---------------------------------------------------------------------------

def find_quantize_bin() -> Path | None:
    return next((p for p in QUANTIZE_BIN_CANDIDATES if p.exists()), None)


def build_quantize() -> Path:
    """Ensure the llama-quantize binary exists, building it from source if needed.

    llama-quantize is a C++ tool that reads a full-precision GGUF and writes
    a smaller, compressed version. It's not distributed as a pre-built binary,
    so we compile it with cmake on first use. Subsequent runs skip the build
    because the binary already exists.
    """
    existing = find_quantize_bin()
    if existing:
        return existing

    print("[QUANTIZE] llama-quantize not found; building llama.cpp …")
    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(exist_ok=True)

    r = run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir)
    require(r.returncode == 0, "cmake configuration failed.")

    # Use all available CPU cores to speed up the build
    cpu_count = os.cpu_count() or 4
    r = run(["cmake", "--build", ".", "--config", "Release", "-j", str(cpu_count)], cwd=build_dir)
    require(r.returncode == 0, "llama.cpp build failed.")
    built = find_quantize_bin()
    require(built is not None, f"Build succeeded but llama-quantize not found in any of {QUANTIZE_BIN_CANDIDATES}.")

    print(f"[QUANTIZE] ✓ Built {built}")
    return built


def quantize_model(model_dir: Path, output_dir: Path, levels: list[str], prefix: str,
                   base_model: str | None, source_size_mb: float) -> dict[str, Path]:
    """Produce one validated GGUF per requested quantization level.

    Levels with a validated cached file are reused. The rest are each built by
    a separate llama-quantize run over one temporary f16 GGUF, which is deleted
    afterwards. Any failure stops the pipeline: a partial set of levels is
    never reported as success.
    Returns a dict mapping level name → output path.
    """
    results: dict[str, Path] = {}
    pending: list[str] = []
    for level in levels:
        out_path = output_dir / f"{prefix}-{level.lower()}.gguf"
        if reuse_or_invalidate(out_path, "QUANTIZE"):
            results[level] = out_path
        else:
            pending.append(level)
    if not pending:
        return results

    quantize_bin = build_quantize()
    intermediate = output_dir / f"{prefix}-{INTERMEDIATE_OUTTYPE}-intermediate.gguf"
    invalidate_gguf_cache(intermediate)
    try:
        convert_hf(model_dir, intermediate, INTERMEDIATE_OUTTYPE, source_size_mb)
        for level in pending:
            out_path = output_dir / f"{prefix}-{level.lower()}.gguf"
            print(f"\n[QUANTIZE] → {level} …")
            error = run_quant_tool(
                [str(quantize_bin), str(intermediate), str(out_path), level],
                subprocess_timeout_s(file_size_mb(intermediate)),
                partial_output=out_path,
            )
            require(error is None, f"Quantization to {level} failed: {error}")
            require(out_path.exists(), f"{out_path.name} not created after quantization.")
            print(f"[QUANTIZE] ✓ {out_path.name}  ({file_size_mb(out_path):.0f} MB)")
            validate_or_exit(out_path, base_model)
            results[level] = out_path
    finally:
        invalidate_gguf_cache(intermediate)

    return results


# ---------------------------------------------------------------------------
# 4. VALIDATE
# ---------------------------------------------------------------------------

def validate_and_report(
    model_name: str,
    original_format: str,
    base_path: Path | None,
    quant_files: dict[str, Path],
    model_dir: Path,
    conversion_time: float,
) -> dict:
    """Verify output files exist, print a size/RAM summary, and build the report dict.

    Reads config.json to estimate parameter count, then uses QUANT_BPP to
    convert that into a per-level RAM estimate. Q4_K_M is always the recommended
    default because it fits comfortably on budget Android phones while still
    producing acceptable output quality.
    """
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    params = estimate_params_from_config(model_dir)
    param_str = f"{params / 1e9:.2f}B" if params else "unknown"
    print(f"Model:      {model_name}  ({param_str} params)")
    print(f"Format:     {original_format} → GGUF")
    print(f"Conversion: {conversion_time:.1f}s\n")

    sizes: dict[str, float] = {}
    ram_estimates: dict[str, str] = {}
    ready = bool(quant_files) or base_path is not None

    if base_path is not None and base_path.exists():
        mb = file_size_mb(base_path)
        sizes[BASE_LEVEL] = round(mb, 1)
        ram = estimate_ram_gb(params, BASE_LEVEL)
        ram_estimates[BASE_LEVEL] = ram
        print(f"  {BASE_LEVEL:<8}        {mb:>8.0f} MB   RAM {ram}")

    for level in QUANT_LEVELS:
        path = quant_files.get(level)
        if path and path.exists():
            mb = file_size_mb(path)
            sizes[level] = round(mb, 1)
            ram = estimate_ram_gb(params, level)
            ram_estimates[level] = ram
            print(f"  {level:<8}        {mb:>8.0f} MB   RAM {ram}")

    # Q4_K_M is hardcoded as the recommendation because it's the smallest format
    # that llama.cpp runs reliably on low-RAM Android devices (< 4 GB).
    recommended = "Q4_K_M"
    print("\nRECOMMENDATION")
    for quant, (ram_range, device_desc) in RAM_RECOMMENDATIONS.items():
        marker = "◀ recommended" if quant == recommended else ""
        if quant in quant_files:
            print(f"  {quant:<8}  {ram_range:<12}  {device_desc}  {marker}")

    print("=" * 60)

    # Collect the actual filenames so the report is self-contained — callers
    # don't need to reconstruct naming logic to find the files.
    output_files = {}
    if base_path is not None and base_path.exists():
        output_files[BASE_LEVEL] = base_path.name
    for level, path in quant_files.items():
        if path.exists():
            output_files[level] = path.name

    report = {
        "model_name": model_name,
        "original_format": original_format,
        "conversion_time_seconds": round(conversion_time, 1),
        "param_count": param_str,
        "output_files": output_files,
        "quantization_sizes_mb": sizes,
        "ram_estimates": ram_estimates,
        "recommended_quantization": recommended,
        "ready_for_deployment": ready,
    }
    return report


# ---------------------------------------------------------------------------
# 5. DEPLOY
# ---------------------------------------------------------------------------

# Common install locations for the Android Debug Bridge (ADB) binary.
# Virtual environments inherit a restricted PATH that often omits the Android
# SDK's platform-tools directory, so we fall back to these known paths.
ADB_SEARCH_PATHS = [
    *(Path(p) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
      for p in (os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT")) if p),
    Path.home() / "Library" / "Android" / "sdk" / "platform-tools" / "adb",
    Path.home() / "Android" / "Sdk" / "platform-tools" / "adb",
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
    Path("/usr/local/bin/adb"),
    Path("/opt/homebrew/bin/adb"),
]


def find_adb() -> Path | None:
    """Locate the adb binary by checking PATH first, then known install locations.

    Returns the full Path to adb, or None if it can't be found anywhere.
    Using the full path in subsequent calls avoids relying on PATH being set
    correctly at runtime (important when running inside a venv).
    """
    adb_in_path = shutil.which("adb")
    if adb_in_path:
        return Path(adb_in_path)
    for candidate in ADB_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    return None


def deploy_via_adb(quant_files: dict[str, Path], preferred: str = "Q4_K_M") -> None:
    """Push EVERY successfully converted/existing GGUF file to a connected
    Android device via ADB - not just `preferred`. `preferred` (Q4_K_M by
    default) remains the validation report's recommended level; it plays no
    special role here beyond being referenced in the manual-instructions
    fallback if ADB/device isn't available at all. Each file is pushed with
    the same single-push logic, just repeated; one file failing to push
    doesn't stop the rest from being attempted.
    """
    print("\n[DEPLOY] Checking ADB connection …")

    adb = find_adb()
    if adb is None:
        searched = "\n    ".join(str(p) for p in ADB_SEARCH_PATHS)
        print(
            "[DEPLOY] ADB not found in PATH or any of these locations:\n"
            f"    {searched}\n"
            "Install Android platform-tools and add the platform-tools directory to PATH:\n"
            "    export PATH=\"$HOME/Library/Android/sdk/platform-tools:$PATH\""
        )
        _print_manual_instructions(quant_files, preferred)
        return

    print(f"[DEPLOY] Using ADB at {adb}")
    # capture=True so we can parse the device list without printing raw adb output
    r = run([str(adb), "devices"], capture=True)
    lines = [l for l in r.stdout.strip().splitlines()[1:] if l.strip() and "offline" not in l]
    if not lines:
        print("[DEPLOY] No Android device connected via ADB.")
        _print_manual_instructions(quant_files, preferred)
        return

    device_line = lines[0]
    print(f"[DEPLOY] Device found: {device_line}")

    if not quant_files:
        print("[DEPLOY] No GGUF files to deploy.")
        return

    for level, gguf_path in quant_files.items():
        remote_path = f"/sdcard/Download/{gguf_path.name}"
        print(f"[DEPLOY] Pushing {level} {gguf_path.name} ({file_size_mb(gguf_path):.0f} MB) → {remote_path}")
        r = run([str(adb), "push", str(gguf_path), remote_path])
        if r.returncode == 0:
            print(f"[DEPLOY] ✓ {level} pushed successfully.")
        else:
            print(f"[DEPLOY] {level} push failed.")

    print(f"\nTo import in SmolChat:")
    print(f"  1. Open SmolChat on your Android device")
    print(f"  2. Tap ☰ → Models → Import model")
    print(f"  3. Navigate to Downloads → select the pushed .gguf file")


def _print_manual_instructions(quant_files: dict[str, Path], preferred: str) -> None:
    """Print step-by-step instructions for manually copying the model to the phone."""
    path = quant_files.get(preferred) or (list(quant_files.values())[0] if quant_files else None)
    print("\nManual deployment instructions:")
    print("  1. Connect your Android phone via USB with file transfer enabled")
    if path:
        print(f"  2. Copy {path} to your phone's Downloads folder")
    print("  3. Open SmolChat → ☰ → Models → Import model → select the .gguf file")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a HuggingFace model to quantized GGUF for Android/llama.cpp"
    )
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    p.add_argument(
        "--quant",
        choices=QUANT_LEVELS + [BASE_LEVEL, "ALL"],
        default="ALL",
        help="Level(s) to produce (default: ALL = BF16, Q4_K_M, Q8_0). BF16 alone stops "
             "after the full-precision conversion and produces no quantized files.",
    )
    p.add_argument("--deploy", action="store_true", help="Push the quantized GGUFs to a connected Android device via ADB")
    return p.parse_args()


def main() -> None:
    """Orchestrate the full download → convert → quantize → validate → deploy pipeline."""
    # Windows encodes redirected output as cp1252, which can't print the progress symbols.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ensure_active_venv()

    args = parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quant == "ALL":
        want_base, quant_levels = True, list(QUANT_LEVELS)
    elif args.quant == BASE_LEVEL:
        want_base, quant_levels = True, []
    else:
        want_base, quant_levels = False, [args.quant]

    require(LLAMA_CPP_DIR.exists(), f"llama.cpp not found at {LLAMA_CPP_DIR}. Clone it first.")
    require_llama_cpp_python()

    # Start the clock here so conversion_time covers the full pipeline
    t_start = time.time()

    # 1. Download / resolve
    model_dir = resolve_model(args.model, output_dir)
    original_format = detect_weight_format(model_dir)
    prefix = model_prefix(args.model)
    print(f"[INFO] Output filename prefix: {prefix}")
    source_size_mb = dir_size_mb(model_dir)
    # The smoke test only formats prompts for Hugging Face ids it recognises.
    base_model = None if Path(args.model).exists() else args.model

    # 2. Full-precision bf16 GGUF
    base_path = convert_to_base(model_dir, output_dir, prefix, base_model, source_size_mb) if want_base else None

    # 3. Quantized levels, each from a temporary f16 GGUF
    quant_files = (
        quantize_model(model_dir, output_dir, quant_levels, prefix, base_model, source_size_mb)
        if quant_levels else {}
    )

    conversion_time = time.time() - t_start

    # 4. Validate outputs and build the report
    model_name = args.model.split("/")[-1] if "/" in args.model and not Path(args.model).exists() else Path(args.model).name
    report = validate_and_report(
        model_name=model_name,
        original_format=original_format,
        base_path=base_path,
        quant_files=quant_files,
        model_dir=model_dir,
        conversion_time=conversion_time,
    )

    report_path = output_dir / "conversion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Saved to {report_path}")

    # 5. Optionally push to phone
    if args.deploy:
        deploy_via_adb(quant_files)
    else:
        print("\nTip: re-run with --deploy to push to a connected Android device via ADB.")

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
