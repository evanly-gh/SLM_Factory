#!/usr/bin/env python3
"""
convert_to_mnn.py — HuggingFace → quantized MNN pipeline for Android/MNN Chat deployment

Mirrors convert_to_gguf.py's interface and behavior, adapted for MNN format.

Pipeline stages:
  1. Download        — pull model weights + config from HuggingFace Hub (or use a local path)
  2. Export+Quantize — llmexport.py converts the HF checkpoint AND quantizes it in one step
                        (unlike GGUF, MNN has no separate full-precision base file; the 16-bit
                        level is its own fp16 export)
  3. Validate         — check the export is complete, was built with exactly the requested
                        recipe, and record a fingerprint sidecar keyed on recipe + MNN version
  4. Report           — file sizes and RAM estimates per quant_bit level
  5. Deploy           — (optional) push the MNN model folders to a connected Android
                        phone via ADB
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# quant_bit levels and the pool selector each one stands for. Q4_K_M has no
# exact MNN equivalent; the mapping is a statement about bit width, which is
# the only thing the two formats can honestly be compared on.
LEVEL_SELECTORS = {"4": "Q4_K_M", "8": "Q8_0", "16": "FP16"}
QUANT_LEVELS = ["4", "8"]

# Full-precision fp16 export (`--quant_bit 16`). Kept out of QUANT_LEVELS
# since it's a much larger, non-quantized artifact.
FP16_QUANT_LEVEL = "16"

# Levels exported by --quant ALL - the project's standard three precisions
# (mirrors convert_to_gguf.py's ALL: BF16, Q4_K_M, Q8_0).
ALL_QUANT_LEVELS = [FP16_QUANT_LEVEL, "4", "8"]

# THE EXPORT RECIPE: block size (input channels sharing one scale/zero-point)
# and whether HQQ searches for those scales instead of taking each block's
# min/max. 4-bit uses HQQ with 32-weight blocks: MNN's default (min/max, block
# 64) was measured losing up to 0.108 macro-F1 against llama.cpp's Q4_K_M on
# the same weights, and HQQ + block 32 closed that to within 0.011 for ~10%
# more file. 8-bit and 16-bit keep MNN's default.
#
# SLM_MNN_QUANT_BLOCK / SLM_MNN_HQQ, when set, override the recipe for EVERY width.
_BLOCK_OVERRIDE = os.environ.get("SLM_MNN_QUANT_BLOCK", "")
_HQQ_OVERRIDE = os.environ.get("SLM_MNN_HQQ", "")
MNN_QUANT_BLOCK = int(_BLOCK_OVERRIDE) if _BLOCK_OVERRIDE else None
MNN_HQQ = (_HQQ_OVERRIDE == "1") if _HQQ_OVERRIDE else None

_DEFAULT_RECIPE = {4: (32, True)}
_FALLBACK_RECIPE = (64, False)

# The lm_head is kept at 8 bits while the body goes to 4: llama.cpp's Q4_K_M
# keeps output.weight at Q6_K, so a uniform 4-bit MNN export would not be the
# artifact `Q4_K_M` names, and low-bit lm_heads were seen producing garbage on
# device. It is never set below the body's width. SLM_MNN_LM_QUANT_BIT=4
# exports the plain uniform-width model, 16 keeps the lm_head in fp16.
MNN_LM_QUANT_BIT = int(os.environ.get("SLM_MNN_LM_QUANT_BIT", "8"))

# Maps each quant_bit level to a human-readable RAM range and target device
# description, used in the validation report to guide deployment decisions.
RAM_RECOMMENDATIONS = {
    "16": ("12 GB+ RAM", "Full-precision fp16 - lossless quality, reference/desktop use"),
    "8": ("6 GB+ RAM", "Flagship Android devices - near-lossless quality"),
    "4": ("< 4 GB RAM", "Budget/mid-range Android devices - the sweet spot for most phones"),
}

# This project's own peak-RSS scaling factor (see check_model_fit.py):
# estimated_rss_mb = combined_file_size_mb * RSS_MULTIPLIER. Applied directly
# to the actual quantized output size rather than reconstructed from a
# param-count + bits-per-parameter table (which GGUF's version uses), since
# the real file size already reflects whatever llmexport actually produced.
RSS_MULTIPLIER = 1.6

# The files an MNN export must have produced for the folder to be a usable
# model. The tokenizer is spelled either way: MNN writes tokenizer.mtok for a
# HuggingFace fast tokenizer and tokenizer.txt for a sentencepiece one.
REQUIRED_FILES = ("config.json", "llm.mnn", "llm.mnn.weight", "llm_config.json")
TOKENIZER_FILES = ("tokenizer.mtok", "tokenizer.txt")

MNN_VALIDATION_SCHEMA_VERSION = 1
MNN_VALIDATION_SUFFIX = ".validation.json"

# Weight files llmexport.py cannot use, plus pre-built GGUFs and the raw
# Meta/Mistral checkpoints some repos ship alongside the HF weights.
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

# Wall-clock ceiling for one export. Scales with the bytes read like the GGUF
# path, but with a much higher floor: an export traces the model through torch
# to ONNX before MNNConvert rewrites it, and a cold first export is dominated
# by the torch import rather than the model's size. SLM_QUANT_TIMEOUT_S, when
# set, is used verbatim and is NOT raised to the floor.
QUANT_TIMEOUT_FLOOR_S = 600
QUANT_TIMEOUT_S_PER_GB = int(os.environ.get("SLM_QUANT_TIMEOUT_S_PER_GB", "240"))
QUANT_TIMEOUT_OVERRIDE_S = os.environ.get("SLM_QUANT_TIMEOUT_S")
MNN_EXPORT_TIMEOUT_FLOOR_S = int(os.environ.get("SLM_MNN_EXPORT_TIMEOUT_FLOOR_S", "1800"))

DEVICE_MODELS_DIR = "/data/local/tmp/mnn_models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, cwd: Path = None, capture: bool = False, env: dict = None) -> subprocess.CompletedProcess:
    """Execute a shell command, printing it first so the user can see what's running.

    capture=True suppresses stdout/stderr (used when we need to inspect output
    programmatically, e.g. parsing `adb devices`). Otherwise output streams
    live to the terminal so the long-running export step shows progress.
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


def combined_size_mb(output_dir: Path):
    """Sum of llm.mnn + llm.mnn.weight - the two files that are the model;
    config/tokenizer files are negligible and roughly constant across levels."""
    total = 0
    found_any = False
    for name in ("llm.mnn", "llm.mnn.weight"):
        p = output_dir / name
        if p.exists():
            total += p.stat().st_size
            found_any = True
    return round(total / (1024 ** 2), 1) if found_any else None


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


def estimate_rss_mb(size_mb) -> str:
    """Convert an on-disk quantized model size into an estimated peak-RSS
    string, using this project's own RSS_MULTIPLIER. Returns 'unknown' if
    size_mb couldn't be determined."""
    if size_mb is None:
        return "unknown"
    return f"~{size_mb * RSS_MULTIPLIER:.0f} MB"


def model_prefix(model_arg: str) -> str:
    """Derive a clean, lowercase filename prefix from a HuggingFace model ID or local path.

    Example: "Qwen/Qwen2.5-0.5B-Instruct" → "qwen2.5-0.5b-instruct"
    Shared by all output folders for a given model so they can coexist in the
    same output directory without colliding.
    """
    name = Path(model_arg).name if Path(model_arg).exists() else model_arg.split("/")[-1]
    return name.lower().replace("_", "-").replace(" ", "-")


def quant_slug(prefix: str, quant_bit: str) -> str:
    return f"{prefix}-mnn-q{quant_bit}"


def export_recipe(bits: int) -> tuple[int, bool]:
    """(quant_block, hqq) for an export at `bits`, honouring any operator override."""
    block, hqq = _DEFAULT_RECIPE.get(bits, _FALLBACK_RECIPE)
    return (
        MNN_QUANT_BLOCK if MNN_QUANT_BLOCK is not None else block,
        MNN_HQQ if MNN_HQQ is not None else hqq,
    )


def lm_quant_bit(bits: int) -> int:
    return max(MNN_LM_QUANT_BIT, bits)


def missing_files(output_dir: Path) -> list[str]:
    """Which required pieces of an MNN model are absent."""
    if not output_dir.is_dir():
        return [f"{output_dir} (directory does not exist)"]
    absent = [name for name in REQUIRED_FILES if not (output_dir / name).is_file()]
    if not any((output_dir / name).is_file() for name in TOKENIZER_FILES):
        absent.append(" or ".join(TOKENIZER_FILES))
    return absent


def is_export_complete(output_dir: Path) -> bool:
    return not missing_files(output_dir)


def export_timeout_s(source_size_mb: float) -> int:
    """Ceiling for one llmexport.py run over a source of `source_size_mb`."""
    if QUANT_TIMEOUT_OVERRIDE_S:
        return int(QUANT_TIMEOUT_OVERRIDE_S)
    scaled = max(QUANT_TIMEOUT_FLOOR_S, int((source_size_mb / 1024.0) * QUANT_TIMEOUT_S_PER_GB))
    return max(scaled, MNN_EXPORT_TIMEOUT_FLOOR_S)


def run_export_tool(cmd: list[str], timeout_s: int, partial_output: Path, cwd: Path) -> str | None:
    """Run llmexport.py, retrying ONCE at double the ceiling on timeout.

    A timeout is a statement about the machine, not the model, so it is
    retried; a non-zero exit means the exporter rejected the input and would
    fail the same way again, so it is not. A killed or failed attempt leaves a
    half-written folder, which is deleted so it is never mistaken for output.

    Returns None on success or an error string.
    """
    def discard_partial():
        shutil.rmtree(partial_output, ignore_errors=True)

    for attempt, ceiling in enumerate((timeout_s, timeout_s * 2), start=1):
        print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
        try:
            result = subprocess.run(cmd, cwd=cwd, text=True, timeout=ceiling)
        except subprocess.TimeoutExpired:
            discard_partial()
            if attempt == 1:
                print(f"[WARN] llmexport.py exceeded {ceiling}s; retrying once at {ceiling * 2}s")
                continue
            return (
                f"llmexport.py timed out twice ({timeout_s}s then {ceiling}s). Raise the ceiling "
                f"with SLM_QUANT_TIMEOUT_S or SLM_MNN_EXPORT_TIMEOUT_FLOOR_S."
            )
        except FileNotFoundError as exc:
            discard_partial()
            return f"llmexport.py failed: {exc}"
        if result.returncode != 0:
            discard_partial()
            return f"llmexport.py exited with code {result.returncode}"
        return None
    return None


# ---------------------------------------------------------------------------
# MNN toolchain
# ---------------------------------------------------------------------------

class MnnToolchain:
    """Where the three pieces of the MNN export toolchain actually are on this machine."""

    def __init__(self, python: Path, llmexport: Path, mnnconvert: Path, root: Path):
        self.python = python
        self.llmexport = llmexport
        self.mnnconvert = mnnconvert
        self.root = root

    def versions(self) -> dict:
        """Tool identity recorded in the validation sidecar, so a cache hit is per-toolchain."""
        commit = "unknown"
        try:
            commit = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout.strip()[:12] or "unknown"
        except (OSError, subprocess.SubprocessError):
            pass
        return {
            "mnn_commit": commit,
            "mnn_version": _mnn_version(self.root),
            "python": platform.python_version(),
        }


def _mnn_version(root: Path) -> str:
    """MNN's own version triple, read from the header that defines it."""
    header = root / "include" / "MNN" / "MNNDefine.h"
    parts = {}
    try:
        with open(header, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = re.match(r"#define MNN_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)", line.strip())
                if match:
                    parts[match.group(1)] = match.group(2)
    except OSError:
        return "unknown"
    if len(parts) != 3:
        return "unknown"
    return f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"


def default_mnn_root() -> Path:
    """SLM_MNN_ROOT, else the git-ignored MNN/ checkout at the root of this repo."""
    explicit = os.environ.get("SLM_MNN_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return PROJECT_ROOT / "MNN"


def _first_existing(candidates: list[Path]) -> Path:
    return next((p for p in candidates if p.is_file()), candidates[0])


def find_toolchain() -> tuple[MnnToolchain | None, str | None]:
    """Locate llmexport.py, MNNConvert and the exporter's interpreter.

    Returns (toolchain, None), or (None, a message saying what's missing).
    llmexport.py must be given a locally built MNNConvert: without
    --mnnconvert it falls back to the pymnn bindings, which crash with a bus
    error, so a missing binary is an error rather than a silent fallback.
    """
    root = default_mnn_root()
    llmexport = Path(
        os.environ.get("SLM_MNN_LLMEXPORT", "").strip()
        or root / "transformers" / "llm" / "export" / "llmexport.py"
    )
    explicit_convert = os.environ.get("SLM_MNN_CONVERT_BIN", "").strip()
    mnnconvert = Path(explicit_convert) if explicit_convert else _first_existing([
        root / "build" / "MNNConvert",
        root / "build" / "MNNConvert.exe",
        root / "build" / "Release" / "MNNConvert.exe",
    ])
    if not mnnconvert.is_file():
        found = shutil.which("MNNConvert")
        if found:
            mnnconvert = Path(found)
    explicit_python = os.environ.get("SLM_MNN_PYTHON", "").strip()
    python = Path(explicit_python) if explicit_python else _first_existing([
        PROJECT_ROOT / ".venv_mnn" / "bin" / "python",
        PROJECT_ROOT / ".venv_mnn" / "Scripts" / "python.exe",
    ])

    missing = []
    if not llmexport.is_file():
        missing.append(f"llmexport.py at {llmexport}")
    if not (mnnconvert.is_file() and os.access(mnnconvert, os.X_OK)):
        missing.append(f"an executable MNNConvert at {mnnconvert}")
    if not (python.is_file() and os.access(python, os.X_OK)):
        missing.append(f"the exporter's python at {python}")
    if missing:
        return None, (
            "The MNN export toolchain needs " + "; ".join(missing)
            + ". Clone and build MNN (with MNNConvert) into MNN/ at the repo root and create "
              ".venv_mnn there for llmexport.py, or point SLM_MNN_ROOT / SLM_MNN_LLMEXPORT / SLM_MNN_CONVERT_BIN / "
              "SLM_MNN_PYTHON at an existing install."
        )
    return MnnToolchain(python=python, llmexport=llmexport, mnnconvert=mnnconvert, root=root), None


# ---------------------------------------------------------------------------
# Validation + fingerprint-based caching for MNN exports
#
# MNN quantization is a flag, not a file format: a 4-bit build and an fp16
# build are the same five filenames. So after every export, the exporter's own
# record of its arguments (export_args.json) is read back and the export is
# refused if it differs from the requested recipe. The sidecar sits beside the
# folder (<folder>.validation.json) and fingerprints every file in it; a cached
# folder is reused only if its fingerprint, recipe and MNN version all match.
# Anything else is deleted and re-exported.
# ---------------------------------------------------------------------------

def mnn_validation_sidecar_path(output_dir: Path) -> Path:
    return output_dir.with_name(output_dir.name + MNN_VALIDATION_SUFFIX)


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


def _fingerprint(output_dir: Path) -> dict:
    """Size + SHA-256 of every file in the export, keyed by relative path.

    The graph, weights, tokenizer and both configs must agree with each other,
    so all of them are covered: a weight file swapped under an unchanged graph
    is exactly the half-written artifact this exists to catch.
    """
    files = {}
    total = 0
    for dirpath, dirnames, filenames in os.walk(output_dir):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            size = path.stat().st_size
            total += size
            files[path.relative_to(output_dir).as_posix()] = {"size": size, "sha256": _sha256_file(path)}
    return {"files": files, "total_size": total}


def _recorded_export_args(output_dir: Path) -> dict:
    """The arguments llmexport.py says it actually ran with, from export_args.json."""
    try:
        with open(output_dir / "export_args.json", encoding="utf-8") as handle:
            recorded = json.load(handle)
        return recorded if isinstance(recorded, dict) else {}
    except (OSError, ValueError):
        return {}


def _expected_min_weight_size(output_dir: Path) -> "int | None":
    """Lower bound on llm.mnn.weight's size from llm_config.json's
    tie_embeddings offsets (the end of the last section llmexport.py records a
    position for). None if not derivable."""
    try:
        with open(output_dir / "llm_config.json", encoding="utf-8") as handle:
            config = json.load(handle)
        tie = config.get("tie_embeddings")
        if not isinstance(tie, dict):
            return None
        offset = tie.get("alpha_offset")
        size = tie.get("alpha_size")
        if isinstance(offset, int) and isinstance(size, int):
            return offset + size
    except (OSError, TypeError, ValueError):
        pass
    return None


def validate_and_record_mnn(output_dir: Path, level: str, toolchain: MnnToolchain) -> dict:
    """Check a fresh export is complete and built with the requested recipe, then record it.

    Raises RuntimeError if files are missing, the weight file is truncated,
    or export_args.json reports a different quant_bit, quant_block, hqq or
    lm_quant_bit than was asked for.
    """
    absent = missing_files(output_dir)
    if absent:
        raise RuntimeError(f"MNN validation failed: incomplete artifact at {output_dir}; missing {absent}")

    weight_path = output_dir / "llm.mnn.weight"
    weight_size = weight_path.stat().st_size
    if weight_size <= 0:
        raise RuntimeError(f"MNN validation failed: empty file: {weight_path}")
    expected_min = _expected_min_weight_size(output_dir)
    if expected_min is not None and weight_size < expected_min:
        raise RuntimeError(
            f"MNN validation failed: {weight_path.name} is {weight_size} bytes, smaller than the "
            f"{expected_min}-byte minimum implied by llm_config.json (looks truncated)"
        )

    bits = int(level)
    block, hqq = export_recipe(bits)
    expected = {"quant_bit": bits, "quant_block": block, "hqq": hqq, "lm_quant_bit": lm_quant_bit(bits)}
    recorded = _recorded_export_args(output_dir)
    for key, want in expected.items():
        got = recorded.get(key)
        if got is None:
            continue
        if (bool(got) if key == "hqq" else int(got)) != want:
            raise RuntimeError(
                f"MNN validation failed: {output_dir.name} was exported with {key}={got} but "
                f"{key}={want} was requested. Scoring it would attribute one recipe's results to another."
            )

    record = {
        "schema_version": MNN_VALIDATION_SCHEMA_VERSION,
        "quant": LEVEL_SELECTORS[level],
        "quant_bit": bits,
        "quant_block": block,
        "hqq": hqq,
        "lm_quant_bit": lm_quant_bit(bits),
        "weight_size_mb": combined_size_mb(output_dir),
        "fingerprint": _fingerprint(output_dir),
        "tool_versions": toolchain.versions(),
    }
    _atomic_write_json(mnn_validation_sidecar_path(output_dir), record)
    return record


def validated_mnn_cache_hit(output_dir: Path, toolchain: MnnToolchain | None) -> bool:
    """Whether the export exactly matches a validation record for the current recipe.

    The export settings are part of the key, not just the file contents: the
    files on disk cannot say they were built with a different block size or
    lm_head width, so a hit on contents alone would reuse the previous
    recipe's model under the new one's name. When the toolchain is available
    its MNN commit/version must match too.
    """
    sidecar_path = mnn_validation_sidecar_path(output_dir)
    if not output_dir.is_dir() or not sidecar_path.is_file():
        return False
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("schema_version") != MNN_VALIDATION_SCHEMA_VERSION:
            return False
        recorded_versions = record.get("tool_versions")
        if not isinstance(recorded_versions, dict) or not recorded_versions:
            return False
        bits = record.get("quant_bit")
        if not isinstance(bits, int):
            return False
        block, hqq = export_recipe(bits)
        if record.get("quant_block") != block or bool(record.get("hqq", False)) != hqq:
            return False
        if record.get("lm_quant_bit") != lm_quant_bit(bits):
            return False
        if toolchain is not None:
            current = toolchain.versions()
            for key in ("mnn_commit", "mnn_version"):
                if recorded_versions.get(key) != current.get(key):
                    return False
        return record.get("fingerprint") == _fingerprint(output_dir)
    except (OSError, TypeError, ValueError):
        return False


def invalidate_mnn_cache(output_dir: Path) -> None:
    """Remove a derived MNN export and its validation record, and nothing else."""
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        mnn_validation_sidecar_path(output_dir).unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# 1. DOWNLOAD
# ---------------------------------------------------------------------------

def resolve_model(model_arg: str, output_dir: Path) -> Path:
    """Return an absolute local directory containing the model's weights and config.

    If `model_arg` is already a local directory, use it as-is. Otherwise treat
    it as a HuggingFace repo ID and download the snapshot at the repo's current
    commit. llmexport.py needs a local folder for --path, not a raw HF ID.
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

    model_name_safe = model_arg.replace("/", "__")
    dest = output_dir / model_name_safe

    # Rough disk-space check: assume up to 10 GB of weights plus headroom for exports
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
# 2. EXPORT + QUANTIZE
# ---------------------------------------------------------------------------

def export_one(toolchain: MnnToolchain, model_dir: Path, level: str, output_dir: Path,
               source_size_mb: float) -> None:
    """Run llmexport.py for a single quant_bit level and validate the result; exit on failure."""
    bits = int(level)
    block, hqq = export_recipe(bits)

    # A previous attempt's half-written folder is rubble, not a starting point.
    invalidate_mnn_cache(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every path is absolute: llmexport.py runs from its own source directory,
    # and a relative --dst_path silently writes the model into the MNN tree.
    cmd = [
        str(toolchain.python), str(toolchain.llmexport),
        "--path", str(model_dir.resolve()),
        "--export", "mnn",
        "--quant_bit", str(bits),
        "--quant_block", str(block),
        "--lm_quant_bit", str(lm_quant_bit(bits)),
        "--dst_path", str(output_dir.resolve()),
        "--mnnconvert", str(toolchain.mnnconvert.resolve()),
    ]
    if hqq:
        cmd.append("--hqq")
    print(f"[EXPORT] {bits}-bit / block {block} / lm_head {lm_quant_bit(bits)}-bit{' / hqq' if hqq else ''}")

    error = run_export_tool(cmd, export_timeout_s(source_size_mb), partial_output=output_dir,
                            cwd=toolchain.llmexport.parent)
    require(error is None, f"MNN export to Q{level} failed: {error}")

    try:
        validate_and_record_mnn(output_dir, level, toolchain)
    except RuntimeError as exc:
        invalidate_mnn_cache(output_dir)
        require(False, f"{output_dir.name} failed validation and was removed: {exc}")


def export_all_levels(model_dir: Path, output_dir: Path, prefix: str, levels: list) -> dict:
    """Produce one validated MNN model folder per requested quant_bit level.

    A level whose folder matches its validation record for the current recipe
    is reused. Anything else - incomplete, unvalidated, built with a different
    recipe or MNN version, or changed since it was validated - is deleted and
    re-exported. Any failure stops the pipeline: a partial set of levels is
    never reported as success.
    Returns a dict mapping level -> output folder Path.
    """
    toolchain, toolchain_problem = find_toolchain()
    if toolchain is None:
        print(f"[WARN] {toolchain_problem}\n[WARN] Only validated exports can be reused, and their MNN "
              f"version can't be checked.")

    source_size_mb = dir_size_mb(model_dir)
    results = {}
    for level in levels:
        slug = quant_slug(prefix, level)
        out_dir = output_dir / slug

        if validated_mnn_cache_hit(out_dir, toolchain):
            print(f"[EXPORT] {slug} already exists and passed validated-cache check, skipping.")
            results[level] = out_dir
            continue
        if out_dir.exists():
            print(f"[EXPORT] {slug} exists but is unvalidated, changed or built with another recipe - re-exporting.")

        require(toolchain is not None, f"Cannot export {slug}: {toolchain_problem}")
        print(f"\n[EXPORT] → Q{level} …")
        export_one(toolchain, model_dir, level, out_dir, source_size_mb)
        size_mb = combined_size_mb(out_dir)
        size_disp = f"{size_mb:.0f} MB" if size_mb is not None else "unknown size"
        print(f"[EXPORT] ✓ {slug}  ({size_disp})")
        results[level] = out_dir

    return results


# ---------------------------------------------------------------------------
# 3. VALIDATE
# ---------------------------------------------------------------------------

def validate_and_report(
    model_name: str,
    original_format: str,
    quant_dirs: dict,
    conversion_time: float,
) -> dict:
    """Verify output folders exist, print a size/RAM summary, and build the report dict.

    Estimates RAM directly from each level's actual combined llm.mnn +
    llm.mnn.weight size via this project's RSS_MULTIPLIER, rather than a
    param-count-based table - MNN's quantized file size already reflects
    exactly what was produced. Q4 is always the recommended default because
    it's the smallest level that runs reliably on budget Android phones
    while still producing acceptable output quality.
    """
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print(f"Model:      {model_name}")
    print(f"Format:     {original_format} → MNN")
    print(f"Conversion: {conversion_time:.1f}s\n")

    sizes = {}
    rss_estimates = {}
    ready = bool(quant_dirs)

    for level in [FP16_QUANT_LEVEL] + QUANT_LEVELS:
        path = quant_dirs.get(level)
        if path and is_export_complete(path):
            mb = combined_size_mb(path)
            sizes[level] = mb
            rss = estimate_rss_mb(mb)
            rss_estimates[level] = rss
            mb_disp = f"{mb:>8.0f} MB" if mb is not None else " unknown"
            label = "F16" if level == FP16_QUANT_LEVEL else f"Q{level}"
            print(f"  {label:<10}      {mb_disp}   RSS {rss}")

    recommended = "4"
    print("\nRECOMMENDATION")
    for level, (ram_range, device_desc) in RAM_RECOMMENDATIONS.items():
        marker = "◀ recommended" if level == recommended else ""
        if level in quant_dirs:
            print(f"  Q{level:<8}  {ram_range:<12}  {device_desc}  {marker}")

    print("=" * 60)

    # Collect the actual folder names so the report is self-contained -
    # callers don't need to reconstruct naming logic to find the files.
    output_files = {level: path.name for level, path in quant_dirs.items() if is_export_complete(path)}

    report = {
        "model_name": model_name,
        "original_format": original_format,
        "conversion_time_seconds": round(conversion_time, 1),
        "output_files": output_files,
        "quantization_sizes_mb": sizes,
        "rss_estimates_mb": rss_estimates,
        "recommended_quant_bit": recommended,
        "ready_for_deployment": ready,
    }
    return report


# ---------------------------------------------------------------------------
# 4. DEPLOY
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


def find_adb():
    """Locate the adb binary by checking PATH first, then known install locations.

    Returns the full Path to adb, or None if it can't be found anywhere.
    """
    adb_in_path = shutil.which("adb")
    if adb_in_path:
        return Path(adb_in_path)
    for candidate in ADB_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    return None


def deploy_via_adb(quant_dirs: dict, preferred: str = "4") -> None:
    """Push EVERY successfully exported MNN model folder to a connected
    Android device via ADB - not just `preferred`. `preferred` (Q4 by
    default) remains the validation report's recommended level; it plays no
    special role here beyond being referenced in the manual-instructions
    fallback if ADB/device isn't available at all. Each level is pushed with
    the same single-push logic, just repeated; one level failing to push
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
        _print_manual_instructions(quant_dirs, preferred)
        return

    print(f"[DEPLOY] Using ADB at {adb}")
    r = run([str(adb), "devices"], capture=True)
    lines = [l for l in r.stdout.strip().splitlines()[1:] if l.strip() and "offline" not in l]
    if not lines:
        print("[DEPLOY] No Android device connected via ADB.")
        _print_manual_instructions(quant_dirs, preferred)
        return

    device_line = lines[0]
    print(f"[DEPLOY] Device found: {device_line}")

    if not quant_dirs:
        print("[DEPLOY] No quantized MNN model folders to deploy.")
        return

    mkdir_result = run([str(adb), "shell", "mkdir", "-p", DEVICE_MODELS_DIR], capture=True)
    if mkdir_result.returncode != 0:
        print(f"[DEPLOY] adb shell mkdir -p {DEVICE_MODELS_DIR} failed: {mkdir_result.stderr.strip()}")
        _print_manual_instructions(quant_dirs, preferred)
        return

    for level, model_dir in quant_dirs.items():
        size_mb = combined_size_mb(model_dir)
        size_disp = f"{size_mb:.0f} MB" if size_mb is not None else "unknown size"
        print(f"[DEPLOY] Pushing Q{level} {model_dir.name} ({size_disp}) → {DEVICE_MODELS_DIR}/")
        r = run([str(adb), "push", str(model_dir), DEVICE_MODELS_DIR + "/"])
        if r.returncode == 0:
            print(f"[DEPLOY] ✓ Q{level} pushed successfully.")
            print(f"  Device model path: {DEVICE_MODELS_DIR}/{model_dir.name}")
        else:
            print(f"[DEPLOY] Q{level} push failed.")

    print("\nUse a pushed path with run_mnn_autobench.py's --model-path, or in MNN Chat directly.")


def _print_manual_instructions(quant_dirs: dict, preferred: str) -> None:
    """Print step-by-step instructions for manually copying the model to the phone."""
    path = quant_dirs.get(preferred) or (list(quant_dirs.values())[0] if quant_dirs else None)
    print("\nManual deployment instructions:")
    print("  1. Connect your Android phone via USB with file transfer enabled")
    if path:
        print(f"  2. adb shell mkdir -p {DEVICE_MODELS_DIR}")
        print(f"  3. adb push {path} {DEVICE_MODELS_DIR}/")
    print("  4. Point MNN Chat (or run_mnn_autobench.py's --model-path) at the pushed folder")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a HuggingFace model to quantized MNN for Android/MNN Chat"
    )
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--quant",
        choices=QUANT_LEVELS + [FP16_QUANT_LEVEL, "FP16", "ALL"],
        default="ALL",
        help="quant_bit level(s) to produce (default: ALL = 16/4/8, i.e. F16/Q4/Q8)",
    )
    p.add_argument("--deploy", action="store_true", help="Push the exported folders to a connected Android device via ADB")
    return p.parse_args()


def main() -> None:
    """Orchestrate the full download → export/quantize → validate → deploy pipeline."""
    # Windows encodes redirected output as cp1252, which can't print the progress symbols.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quant == "FP16":
        args.quant = FP16_QUANT_LEVEL
    quant_levels = ALL_QUANT_LEVELS if args.quant == "ALL" else [args.quant]

    # Start the clock here so conversion_time covers the full pipeline
    t_start = time.time()

    # 1. Download / resolve
    model_dir = resolve_model(args.model, output_dir)
    original_format = "safetensors" if list(model_dir.glob("*.safetensors")) else "bin"
    prefix = model_prefix(args.model)
    print(f"[INFO] Output filename prefix: {prefix}")

    # 2. Export + quantize each requested quant_bit level
    quant_dirs = export_all_levels(model_dir, output_dir, prefix, quant_levels)

    conversion_time = time.time() - t_start

    # 3. Validate outputs and build the report
    model_name = args.model.split("/")[-1] if "/" in args.model and not Path(args.model).exists() else Path(args.model).name
    report = validate_and_report(
        model_name=model_name,
        original_format=original_format,
        quant_dirs=quant_dirs,
        conversion_time=conversion_time,
    )

    report_path = output_dir / "conversion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[REPORT] Saved to {report_path}")

    # 4. Optionally push to phone
    if args.deploy:
        deploy_via_adb(quant_dirs)
    else:
        print("\nTip: re-run with --deploy to push to a connected Android device via ADB.")

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
