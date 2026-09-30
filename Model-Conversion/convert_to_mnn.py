#!/usr/bin/env python3
"""
convert_to_mnn.py — HuggingFace → quantized MNN pipeline for Android/MNN Chat deployment

Mirrors convert_to_gguf.py's interface and behavior, adapted for MNN format.

Pipeline stages:
  1. Download        — pull model weights + config from HuggingFace Hub (or use a local path)
  2. Export+Quantize — llmexport.py converts the HF checkpoint AND quantizes it in one step
                        (unlike GGUF, MNN has no separate full-precision base file - there is
                        no bf16/unquantized MNN export at all)
  3. Validate         — check file sizes and estimate RAM requirements per quant_bit level
  4. Deploy           — (optional) push the chosen MNN model folder to a connected Android
                        phone via ADB
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — paths and quantization configuration
# ---------------------------------------------------------------------------

# MNN is expected to be cloned here. Export tooling and the MNNConvert binary
# both come from this source tree.
MNN_ROOT = Path.home() / "SLM_Factory_Krishna_Personal" / "MNN"

# Script that both converts AND quantizes a HuggingFace checkpoint into MNN
# format in a single pass - MNN has no separate convert-then-quantize split
# the way GGUF/llama.cpp does. Run from its own dedicated .venv (not this
# script's interpreter), since it has its own pinned dependency set.
LLMEXPORT_DIR = MNN_ROOT / "transformers" / "llm" / "export"
LLMEXPORT_SCRIPT = LLMEXPORT_DIR / "llmexport.py"
LLMEXPORT_VENV_PYTHON = LLMEXPORT_DIR / ".venv" / "bin" / "python"

# Required by llmexport.py's --mnnconvert flag - without it, conversion
# crashes with a Bus error via the broken pymnn bindings.
MNNCONVERT_BIN = MNN_ROOT / "build" / "MNNConvert"

# quant_block is fixed rather than swept - it's a block size for the
# quantization scheme, not a quality/size tradeoff axis like quant_bit is.
QUANT_BLOCK = 64

# --quant_bit values MNN actually supports. Unlike GGUF's named presets,
# there is no 5-bit level and no unquantized/bf16 export at all.
QUANT_LEVELS = ["2", "3", "4", "8"]

# Full-precision fp16 export. Verified: llm_config.json reports quant_bit=16
# and the output file size is ~2x the Q8 file, matching genuine fp16 weights
# (not a relabeled 8-bit export). Kept out of QUANT_LEVELS/ALL since it's a
# much larger, non-quantized artifact - only produced when explicitly requested.
FP16_QUANT_LEVEL = "16"

# Levels swept by --quant ALL - the project's standard three-quantization
# set (mirrors convert_to_gguf.py's ALL: F16, Q4_K_M, Q8_0), not the full
# QUANT_LEVELS sweep. Individual levels (2, 3, 4, 8, 16/FP16) remain
# selectable on their own via --quant.
ALL_QUANT_LEVELS = [FP16_QUANT_LEVEL, "4", "8"]

# Confirmed fix for Q4 output instability (repetition/garbage generation):
# keeping the lm_head/tied-embedding layer at 8-bit precision even when the
# rest of the model is quantized to 4-bit, via llmexport.py's --lm_quant_bit,
# matches Alibaba's own official MNN conversion approach. Verified directly
# on Qwen3-1.7B; applied to every Q4 export since this is a general
# precision-sensitivity pattern in low-bit lm_head quantization, not a
# one-off fix for that model. Q8 exports don't need this override - the
# lm_head is already 8-bit there since the whole model is.
Q4_LM_HEAD_QUANT_BIT = 8

# Maps each quant_bit level to a human-readable RAM range and target device
# description, used in the validation report to guide deployment decisions.
RAM_RECOMMENDATIONS = {
    "16": ("12 GB+ RAM", "Full-precision fp16 - lossless quality, reference/desktop use"),
    "8": ("6 GB+ RAM", "Flagship Android devices - near-lossless quality"),
    "4": ("< 4 GB RAM", "Budget/mid-range Android devices - the sweet spot for most phones"),
    "3": ("< 3 GB RAM", "Very constrained devices - noticeable quality tradeoff vs Q4"),
    "2": ("< 2 GB RAM", "Extremely constrained devices - significant quality loss; validate outputs carefully"),
}

# This project's own peak-RSS scaling factor (see check_model_fit.py):
# estimated_rss_mb = combined_file_size_mb * RSS_MULTIPLIER. Applied directly
# to the actual quantized output size rather than reconstructed from a
# param-count + bits-per-parameter table (which GGUF's version uses), since
# the real file size already reflects whatever llmexport actually produced.
RSS_MULTIPLIER = 1.6

# Exactly the files export-completeness is judged on (mirrors
# agent_mnn_quantize.py's is_conversion_complete()) - a successful export
# also produces export_args.json and llm.mnn.json, but those aren't required
# to treat a folder as a reusable, complete conversion.
REQUIRED_FILES = ["config.json", "llm.mnn", "llm.mnn.weight", "llm_config.json", "tokenizer.mtok"]

# The weight file is where a real, concrete corruption was found in practice
# (a 16KB truncated file where ~2GB was expected) - it's the file hashed and
# size-checked by the validation sidecar below.
KEY_WEIGHT_FILENAME = "llm.mnn.weight"

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


def combined_size_mb(output_dir: Path):
    """Sum of llm.mnn + llm.mnn.weight - the two files that actually scale
    with quant_bit; config/tokenizer files are negligible and roughly
    constant across levels."""
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


def is_export_complete(output_dir: Path) -> bool:
    return output_dir.is_dir() and all((output_dir / f).exists() for f in REQUIRED_FILES)


# ---------------------------------------------------------------------------
# Real-load validation + hash-based caching for MNN exports
#
# Mirrors convert_to_gguf.py's validate_and_record_gguf()/validated_gguf_cache_hit()
# pattern, adapted for MNN's multi-file output (a folder of config.json,
# llm.mnn, llm.mnn.weight, llm_config.json, tokenizer.mtok rather than one
# GGUF file). llm.mnn.weight is the file hashed - it's the large binary
# blob where a real corruption (a 16KB file truncated from an expected
# ~2GB) was found, and the other files are small/structural by comparison.
#
# A genuine MNN-engine load test isn't practical from plain Python without
# the full Android/JNI stack, so instead of a real load we (a) hash the
# weight file into a sidecar (<file>.validation.json) and (b) when
# llm_config.json's tie_embeddings offsets make it derivable, check the
# weight file is at least as large as the offset+size of its last recorded
# section - cheap, dependency-free, and enough to catch a truncated file
# like the one that motivated this. On the next run, a cache hit requires
# the sidecar to match the current file's size and hash, so a corrupted or
# hand-edited export is not silently reused. This is best-effort throughout:
# any failure to validate falls back to the plain is_export_complete() check
# rather than blocking the pipeline.
# ---------------------------------------------------------------------------

def mnn_validation_sidecar_path(weight_path: Path) -> Path:
    return weight_path.with_name(weight_path.name + ".validation.json")


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


def _expected_min_weight_size(output_dir: Path) -> "int | None":
    """Best-effort lower bound on llm.mnn.weight's size, derived from
    llm_config.json's tie_embeddings offsets (alpha_offset + alpha_size marks
    the end of the last section llmexport.py records a position for). Returns
    None if llm_config.json is missing, malformed, or doesn't carry this
    field - callers must treat that as "not derivable", not a failure.
    """
    config_path = output_dir / "llm_config.json"
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        tie = config.get("tie_embeddings")
        if not isinstance(tie, dict):
            return None
        offset = tie.get("alpha_offset")
        size = tie.get("alpha_size")
        if isinstance(offset, int) and isinstance(size, int):
            return offset + size
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


def validate_and_record_mnn_export(output_dir: Path) -> "dict | None":
    """Best-effort: hash llm.mnn.weight and record a validation sidecar.

    Returns the validation record on success, or None if the sidecar could
    not be written (a non-fatal problem). Raises only if the weight file
    itself is clearly broken: missing, empty, or smaller than the minimum
    size implied by llm_config.json when that's derivable - exactly the
    class of corruption found in practice (a 16KB file where ~2GB was
    expected). Callers should treat a raise as a warning, not a hard
    failure, so a flaky check never blocks the pipeline.
    """
    weight_path = output_dir / KEY_WEIGHT_FILENAME
    if not weight_path.is_file():
        raise RuntimeError(f"MNN validation failed: file not found: {weight_path}")
    size = weight_path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"MNN validation failed: empty file: {weight_path}")

    expected_min = _expected_min_weight_size(output_dir)
    if expected_min is not None and size < expected_min:
        raise RuntimeError(
            f"MNN validation failed: {weight_path.name} is {size} bytes, smaller than "
            f"the {expected_min}-byte minimum implied by llm_config.json (looks truncated)"
        )

    record = {
        "schema_version": 1,
        "file_size": size,
        "sha256": _sha256_file(weight_path),
        "expected_min_size": expected_min,
    }
    try:
        _atomic_write_json(mnn_validation_sidecar_path(weight_path), record)
    except OSError as exc:
        print(f"[WARN] Could not write validation sidecar for {weight_path.name}: {exc}")
        return None
    return record


def mnn_export_cache_status(output_dir: Path) -> str:
    """Classify an existing export directory for cache-reuse purposes.

    Returns one of:
      "valid"       - complete export, sidecar present, size+hash match:
                       safe to reuse as-is.
      "invalid"     - export dir is incomplete, OR a sidecar exists but does
                       NOT match the current weight file (the file changed
                       or was truncated/corrupted since it was validated):
                       must NOT be reused, caller should reconvert.
      "unvalidated" - complete export but no sidecar to check against (e.g.
                       a pre-existing export from before this validation was
                       added, or the sidecar itself is unreadable): falls
                       back to the old exists()-only trust, since there is
                       nothing to compare against and we must not block the
                       pipeline on a missing sidecar.

    This distinction is what makes corruption detection actually bite: a
    caller must treat "invalid" as "reconvert", not just log a different
    message while still skipping (that was the bug in an earlier version of
    this check - it always skipped on is_export_complete() and only used the
    validation result to pick which message to print).
    """
    if not is_export_complete(output_dir):
        return "invalid"

    weight_path = output_dir / KEY_WEIGHT_FILENAME
    sidecar_path = mnn_validation_sidecar_path(weight_path)
    if not sidecar_path.is_file():
        return "unvalidated"

    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "unvalidated"

    try:
        matches = (
            record.get("schema_version") == 1
            and record.get("file_size") == weight_path.stat().st_size
            and record.get("sha256") == _sha256_file(weight_path)
        )
    except OSError:
        return "unvalidated"

    return "valid" if matches else "invalid"


def validated_mnn_cache_hit(output_dir: Path) -> bool:
    """True only if mnn_export_cache_status() is "valid" - i.e. safe to reuse
    without reconverting. Kept as a convenience wrapper; see
    mnn_export_cache_status() for the "invalid" vs "unvalidated" distinction
    that callers deciding whether to reconvert need.
    """
    return mnn_export_cache_status(output_dir) == "valid"


# ---------------------------------------------------------------------------
# 1. DOWNLOAD
# ---------------------------------------------------------------------------

def resolve_model(model_arg: str, output_dir: Path) -> Path:
    """Return a local directory containing the model's weights and config.

    If `model_arg` is already a local directory, use it as-is. Otherwise treat
    it as a HuggingFace repo ID and download the full snapshot, skipping
    framework-specific weight files we don't need (TF, Flax, Rust) to save
    disk space and download time. llmexport.py needs a local folder for
    --path, not a raw HF ID.
    """
    local = Path(model_arg)
    if local.exists() and local.is_dir():
        print(f"[DOWNLOAD] Using local model at {local}")
        return local

    print(f"[DOWNLOAD] Fetching {model_arg} from HuggingFace Hub …")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    model_name_safe = model_arg.replace("/", "__")
    dest = output_dir / model_name_safe

    # Rough disk-space check: assume up to 10 GB of weights plus headroom for exports
    check_disk_space(output_dir, 15.0)

    try:
        path = snapshot_download(
            repo_id=model_arg,
            local_dir=str(dest),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"[DOWNLOAD] Saved to {path}")
        return Path(path)
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

def find_llmexport_interpreter():
    """Return llmexport.py's colocated venv python, or None if it's missing -
    checked lazily per quant_bit so already-exported levels can still be
    validated/deployed even if the export toolchain isn't set up."""
    if LLMEXPORT_VENV_PYTHON.exists():
        return str(LLMEXPORT_VENV_PYTHON)
    return None


def export_one(interpreter: str, model_dir: Path, quant_bit: str, output_dir: Path,
                awq: bool = False, hqq: bool = False) -> dict:
    """Run llmexport.py for a single quant_bit level. Returns {"ok": True} or
    {"ok": False, "error": str}."""
    require(LLMEXPORT_SCRIPT.exists(), f"llmexport.py not found at {LLMEXPORT_SCRIPT}")
    require(
        MNNCONVERT_BIN.exists(),
        f"MNNConvert binary not found at {MNNCONVERT_BIN} (--mnnconvert is required - "
        "without it, conversion crashes with a Bus error via the broken pymnn bindings)",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        interpreter, str(LLMEXPORT_SCRIPT),
        "--path", str(model_dir),
        "--export", "mnn",
        "--quant_bit", str(quant_bit),
        "--quant_block", str(QUANT_BLOCK),
        "--dst_path", str(output_dir),
        "--mnnconvert", str(MNNCONVERT_BIN),
    ]
    if awq:
        cmd.append("--awq")
    if hqq:
        cmd.append("--hqq")
    if str(quant_bit) == "4":
        # See Q4_LM_HEAD_QUANT_BIT above - confirmed fix for Q4 repetition/
        # garbage output, applied universally to every Q4 export.
        cmd += ["--lm_quant_bit", str(Q4_LM_HEAD_QUANT_BIT)]

    result = run(cmd, cwd=LLMEXPORT_SCRIPT.parent)
    if result.returncode != 0:
        return {"ok": False, "error": f"llmexport.py exited with code {result.returncode}"}

    if not is_export_complete(output_dir):
        missing = [f for f in REQUIRED_FILES if not (output_dir / f).exists()]
        return {"ok": False, "error": f"llmexport.py exited 0 but expected files are missing: {missing}"}

    return {"ok": True}


def export_all_levels(model_dir: Path, output_dir: Path, prefix: str, levels: list,
                       awq: bool = False, hqq: bool = False) -> dict:
    """Produce one quantized MNN model folder per requested quant_bit level.

    Levels that already exist AND pass validation are skipped so the pipeline
    is safe to re-run after a partial failure - same idempotent-rerun
    behavior as convert_to_gguf.py's quantize_model(). A level whose
    directory looks complete but whose weight-file validation sidecar does
    NOT match (corrupted/truncated/hand-edited since it was last validated)
    is treated as invalid and reconverted, not silently reused - that's the
    whole point of the hash check. A level with no sidecar at all (e.g. an
    export produced before this validation existed) falls back to the old
    exists()-only trust rather than forcing a reconvert on stale data.
    A failure at one level is reported and the sweep continues with the
    remaining levels rather than aborting. Returns a dict mapping level ->
    output folder Path for successfully produced levels.
    """
    interpreter = find_llmexport_interpreter()
    results = {}

    for level in levels:
        slug = quant_slug(prefix, level)
        out_dir = output_dir / slug

        status = mnn_export_cache_status(out_dir)
        if status == "valid":
            print(f"[EXPORT] {slug} already exists and passed validated-cache check, skipping.")
            results[level] = out_dir
            continue
        elif status == "unvalidated":
            print(f"[EXPORT] {slug} already exists, skipping.")
            results[level] = out_dir
            continue
        elif out_dir.exists():
            print(f"[WARN] {slug} exists but failed validation (corrupted/truncated/incomplete) - reconverting.")

        if interpreter is None:
            print(f"[WARN] llmexport.py venv not found at {LLMEXPORT_VENV_PYTHON}, skipping Q{level}.")
            continue

        print(f"\n[EXPORT] → Q{level} …")
        r = export_one(interpreter, model_dir, level, out_dir, awq=awq, hqq=hqq)
        if not r["ok"]:
            # Non-fatal: report the failure and continue with remaining levels
            print(f"[WARN] Export to Q{level} failed ({r['error']}), skipping.")
            continue

        size_mb = combined_size_mb(out_dir)
        size_disp = f"{size_mb:.0f} MB" if size_mb is not None else "unknown size"
        print(f"[EXPORT] ✓ {slug}  ({size_disp})")
        try:
            validate_and_record_mnn_export(out_dir)
        except RuntimeError as exc:
            print(f"[WARN] Post-export validation of {slug} raised: {exc}")
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
    Path.home() / "Library" / "Android" / "sdk" / "platform-tools" / "adb",
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
        help="quant_bit level(s) to produce (default: ALL, sweeps 16/4/8 i.e. "
             "F16/Q4/Q8 - the project's standard set; 2/3 remain selectable "
             "individually but are not part of ALL)",
    )
    p.add_argument("--deploy", action="store_true", help="Push the recommended (Q4) level to connected Android via ADB")

    quant_method_group = p.add_mutually_exclusive_group()
    quant_method_group.add_argument("--awq", action="store_true", help="Use AWQ quantization (passed through to llmexport.py)")
    quant_method_group.add_argument("--hqq", action="store_true", help="Use HQQ quantization (passed through to llmexport.py)")

    return p.parse_args()


def main() -> None:
    """Orchestrate the full download → export/quantize → validate → deploy pipeline."""
    args = parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quant == "FP16":
        args.quant = FP16_QUANT_LEVEL
    quant_levels = ALL_QUANT_LEVELS if args.quant == "ALL" else [args.quant]

    require(MNN_ROOT.exists(), f"MNN not found at {MNN_ROOT}. Clone it first.")

    # Start the clock here so conversion_time covers the full pipeline
    t_start = time.time()

    # 1. Download / resolve
    model_dir = resolve_model(args.model, output_dir)
    original_format = "safetensors" if list(model_dir.glob("*.safetensors")) else "bin"
    prefix = model_prefix(args.model)
    print(f"[INFO] Output filename prefix: {prefix}")

    # 2. Export + quantize each requested quant_bit level
    quant_dirs = export_all_levels(model_dir, output_dir, prefix, quant_levels, awq=args.awq, hqq=args.hqq)

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
