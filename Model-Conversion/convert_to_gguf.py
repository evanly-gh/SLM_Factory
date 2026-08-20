#!/usr/bin/env python3
"""
convert_to_gguf.py — HuggingFace → quantized GGUF pipeline for Android/llama.cpp deployment

Pipeline stages:
  1. Download  — pull model weights + config from HuggingFace Hub (or use a local path)
  2. Convert   — turn the HF checkpoint into a full-precision GGUF via llama.cpp's converter
  3. Quantize  — compress to Q4_K_M / Q5_K_M / Q8_0 using llama-quantize
  4. Validate  — check file sizes and estimate RAM requirements per quantization level
  5. Deploy    — (optional) push the chosen GGUF to a connected Android phone via ADB
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — paths and quantization configuration
# ---------------------------------------------------------------------------

# llama.cpp is expected to be cloned here. All conversion and quantization
# tools are built from this source tree.
LLAMA_CPP_DIR = Path(__file__).resolve().parent.parent / "SmolChat-Android" / "llama.cpp"

# Python script that converts a HuggingFace checkpoint directory into GGUF format.
# Ships with llama.cpp; handles both safetensors and .bin weight files.
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

# Compiled binary that compresses a full-precision GGUF into a quantized variant.
# Built from llama.cpp source via cmake (see build_quantize()).
QUANTIZE_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

# Quantization levels produced by default, ordered from smallest to largest.
# Q4_K_M is the sweet spot for most Android phones; Q8_0 is near-lossless but
# requires significantly more RAM.
QUANT_LEVELS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

# Bits-per-parameter used to estimate loaded model RAM.
# f16/bf16 = 16 bits = 2 bytes; quantized formats store fewer bits per weight.
QUANT_BPP = {
    "f16":    2.0,
    "bf16":   2.0,
    "Q8_0":   1.0,
    "Q5_K_M": 0.6875,
    "Q4_K_M": 0.5625,
}

# Maps each quantization level to a human-readable RAM range and target device
# description, used in the validation report to guide deployment decisions.
RAM_RECOMMENDATIONS = {
    "Q4_K_M": ("< 4 GB RAM", "Moto G Power 2021 and similar budget phones"),
    "Q5_K_M": ("4–6 GB RAM", "Mid-range Android devices"),
    "Q8_0":   ("6 GB+ RAM", "Flagship Android devices"),
}

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

    for candidate in ("python3", "python"):
        venv_python = Path(venv) / "bin" / candidate
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


# ---------------------------------------------------------------------------
# Real-load validation + hash-based caching for GGUF outputs
#
# On top of the plain "does the file exist" check, we optionally load each
# produced GGUF with llama-cpp-python and record a sha256 sidecar
# (<file>.validation.json). On the next run, a cache hit requires the sidecar
# to match the current file size and hash, so a corrupted or hand-edited
# GGUF is not silently reused. This is best-effort: if llama-cpp-python isn't
# installed, or validation fails for any reason, callers fall back to the
# original exists()-only check rather than blocking the pipeline.
# ---------------------------------------------------------------------------

def gguf_validation_sidecar_path(gguf_path: Path) -> Path:
    return gguf_path.with_name(gguf_path.name + ".validation.json")


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    import tempfile
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


def validate_and_record_gguf(gguf_path: Path) -> dict | None:
    """Best-effort: load the GGUF with llama-cpp-python and record a hash sidecar.

    Returns the validation record on success, or None if validation could not
    be performed (e.g. llama-cpp-python missing) or failed for a reason that
    should not block the pipeline. Raises only if the file itself is clearly
    broken (missing or empty) or changed size during the load.
    """
    if not gguf_path.is_file():
        raise RuntimeError(f"GGUF validation failed: file not found: {gguf_path}")
    initial_size = gguf_path.stat().st_size
    if initial_size <= 0:
        raise RuntimeError(f"GGUF validation failed: empty file: {gguf_path}")

    try:
        import llama_cpp
    except ImportError:
        print(f"[VALIDATE] llama-cpp-python not installed; skipping load validation for {gguf_path.name}.")
        return None

    model = None
    try:
        model = llama_cpp.Llama(
            model_path=str(gguf_path.resolve()), n_ctx=128, n_batch=16, n_gpu_layers=0, verbose=False
        )
    except Exception as exc:
        print(f"[WARN] GGUF load validation failed for {gguf_path.name}: {exc}")
        return None
    finally:
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                close()

    final_size = gguf_path.stat().st_size
    if final_size != initial_size:
        raise RuntimeError(f"GGUF changed during validation: {initial_size} -> {final_size} bytes")

    import platform
    record = {
        "schema_version": 1,
        "file_size": final_size,
        "sha256": _sha256_file(gguf_path),
        "tool_versions": {
            "llama_cpp_python": str(getattr(llama_cpp, "__version__", "unknown")),
            "python": platform.python_version(),
        },
    }
    try:
        _atomic_write_json(gguf_validation_sidecar_path(gguf_path), record)
    except OSError as exc:
        print(f"[WARN] Could not write validation sidecar for {gguf_path.name}: {exc}")
        return None
    return record


def validated_gguf_cache_hit(gguf_path: Path) -> bool:
    """True if gguf_path exists and its validation sidecar matches (size + sha256).

    Falls back to False (i.e. "not a validated cache hit") on any error reading
    the sidecar, so callers should still fall back to plain exists()-based
    reuse when this returns False but the file is otherwise present.
    """
    sidecar_path = gguf_validation_sidecar_path(gguf_path)
    if not gguf_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            record = json.load(handle)
        return (
            record.get("schema_version") == 1
            and record.get("file_size") == gguf_path.stat().st_size
            and isinstance(record.get("tool_versions"), dict)
            and bool(record["tool_versions"])
            and record.get("sha256") == _sha256_file(gguf_path)
        )
    except (OSError, TypeError, ValueError):
        return False


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
        heads  = cfg.get("num_attention_heads", 1)
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
    it as a HuggingFace repo ID and download the full snapshot, skipping
    framework-specific weight files we don't need (TF, Flax, Rust) to save
    disk space and download time.
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

    # Replace "/" with "__" so the repo ID can be used as a directory name
    model_name_safe = model_arg.replace("/", "__")
    dest = output_dir / model_name_safe

    # Rough disk-space check: assume up to 10 GB of weights plus headroom for the GGUF
    check_disk_space(output_dir, 15.0)

    try:
        path = snapshot_download(
            repo_id=model_arg,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            # Skip weight formats that convert_hf_to_gguf.py can't use
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
# 2. CONVERT
# ---------------------------------------------------------------------------

def detect_weight_format(model_dir: Path) -> str:
    """Determine whether the model uses .safetensors or .bin weights.

    safetensors is the newer, safer HuggingFace format and is preferred.
    .bin files are legacy PyTorch checkpoints. Both are supported by
    convert_hf_to_gguf.py; we just need to know which is present so we
    can report it and glob the right extension for the disk-space check.
    """
    safetensors = list(model_dir.glob("*.safetensors"))
    bins = list(model_dir.glob("pytorch_model*.bin"))
    if safetensors:
        return "safetensors"
    if bins:
        return "bin"
    require(False, f"No .safetensors or .bin weight files found in {model_dir}")


def convert_to_base(model_dir: Path, output_dir: Path, prefix: str, base_dtype: str = "f16") -> Path:
    """Convert a HuggingFace checkpoint to a full-precision GGUF file.

    Calls llama.cpp's convert_hf_to_gguf.py, which reads the model architecture
    from config.json and writes all weights into a single GGUF container.
    `base_dtype` ("f16" or "bf16") is the lossless intermediate — quantized
    variants are produced from it in the next step so we only need to run this
    conversion once.
    Skips conversion if the output file already exists (idempotent reruns).
    """
    require(CONVERT_SCRIPT.exists(), f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}")

    fmt = detect_weight_format(model_dir)
    print(f"[CONVERT] Detected weight format: {fmt}")

    base_path = output_dir / f"{prefix}-{base_dtype}.gguf"
    if base_path.exists():
        if validated_gguf_cache_hit(base_path):
            print(f"[CONVERT] {base_path.name} already exists and passed validated-cache check, skipping conversion.")
        else:
            print(f"[CONVERT] {base_path.name} already exists, skipping conversion.")
        return base_path

    # The base GGUF will be roughly the same size as the source weights
    check_disk_space(output_dir, file_size_mb(next(model_dir.glob(f"*.{fmt if fmt == 'safetensors' else 'bin'}"))) / 1024 * 2 + 2)

    # convert_hf_to_gguf.py imports torch + numpy; if two copies of libomp end up
    # loaded (e.g. via mismatched wheel builds) the process aborts with
    # "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
    # initialized." KMP_DUPLICATE_LIB_OK=TRUE is the same fix already proven in
    # agent_quantize.py's find_convert_interpreter() call site for this exact error.
    convert_env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    result = run(
        [sys.executable, str(CONVERT_SCRIPT), str(model_dir), "--outfile", str(base_path), "--outtype", base_dtype],
        env=convert_env,
    )
    if result.returncode != 0:
        print(f"[ERROR] Conversion failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    require(base_path.exists(), f"Expected {base_path} after conversion but it was not created.")
    print(f"[CONVERT] ✓ {base_path.name}  ({file_size_mb(base_path):.0f} MB)")
    try:
        validate_and_record_gguf(base_path)
    except RuntimeError as exc:
        print(f"[WARN] Post-conversion validation of {base_path.name} raised: {exc}")
    return base_path


# ---------------------------------------------------------------------------
# 3. QUANTIZE
# ---------------------------------------------------------------------------

def build_quantize() -> Path:
    """Ensure the llama-quantize binary exists, building it from source if needed.

    llama-quantize is a C++ tool that reads a full-precision GGUF and writes
    a smaller, compressed version. It's not distributed as a pre-built binary,
    so we compile it with cmake on first use. Subsequent runs skip the build
    because the binary already exists.
    """
    if QUANTIZE_BIN.exists():
        return QUANTIZE_BIN

    print("[QUANTIZE] llama-quantize not found; building llama.cpp …")
    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(exist_ok=True)

    r = run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir)
    require(r.returncode == 0, "cmake configuration failed.")

    # Use all available CPU cores to speed up the build
    cpu_count = os.cpu_count() or 4
    r = run(["cmake", "--build", ".", "--config", "Release", "-j", str(cpu_count)], cwd=build_dir)
    require(r.returncode == 0, "llama.cpp build failed.")
    require(QUANTIZE_BIN.exists(), f"Build succeeded but {QUANTIZE_BIN} not found.")

    print(f"[QUANTIZE] ✓ Built {QUANTIZE_BIN}")
    return QUANTIZE_BIN


def quantize_model(f16_path: Path, output_dir: Path, levels: list[str], prefix: str) -> dict[str, Path]:
    """Produce one compressed GGUF per requested quantization level.

    Each level is a separate llama-quantize invocation reading the same f16
    source file and writing an independent output. Levels that already exist
    are skipped so the pipeline is safe to re-run after a partial failure.
    Returns a dict mapping level name → output path for successfully created files.
    """
    quantize_bin = build_quantize()
    results: dict[str, Path] = {}

    for level in levels:
        out_name = f"{prefix}-{level.lower()}.gguf"
        out_path = output_dir / out_name

        if out_path.exists():
            if validated_gguf_cache_hit(out_path):
                print(f"[QUANTIZE] {out_name} already exists and passed validated-cache check, skipping.")
            else:
                print(f"[QUANTIZE] {out_name} already exists, skipping.")
            results[level] = out_path
            continue

        print(f"\n[QUANTIZE] → {level} …")
        r = run([str(quantize_bin), str(f16_path), str(out_path), level])
        if r.returncode != 0:
            # Non-fatal: report the failure and continue with remaining levels
            print(f"[WARN] Quantization to {level} failed (exit {r.returncode}), skipping.")
            continue

        if out_path.exists():
            print(f"[QUANTIZE] ✓ {out_name}  ({file_size_mb(out_path):.0f} MB)")
            try:
                validate_and_record_gguf(out_path)
            except RuntimeError as exc:
                print(f"[WARN] Post-quantization validation of {out_name} raised: {exc}")
            results[level] = out_path
        else:
            print(f"[WARN] {out_name} not created after quantization.")

    return results


# ---------------------------------------------------------------------------
# 4. VALIDATE
# ---------------------------------------------------------------------------

def validate_and_report(
    model_name: str,
    original_format: str,
    f16_path: Path,
    quant_files: dict[str, Path],
    model_dir: Path,
    conversion_time: float,
    base_dtype: str = "f16",
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
    ready = bool(quant_files)

    if f16_path.exists():
        mb = file_size_mb(f16_path)
        sizes[base_dtype] = round(mb, 1)
        print(f"  {base_dtype} (base):  {mb:>8.0f} MB   RAM {estimate_ram_gb(params, base_dtype)}")

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
    if f16_path.exists():
        output_files[base_dtype] = f16_path.name
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
    Path.home() / "Library" / "Android" / "sdk" / "platform-tools" / "adb",
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
        print("[DEPLOY] No quantized GGUF files to deploy.")
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
        choices=QUANT_LEVELS + ["ALL"],
        default="ALL",
        help="Quantization level(s) to produce (default: ALL)",
    )
    p.add_argument("--deploy", action="store_true", help="Push Q4_K_M to connected Android via ADB")
    p.add_argument("--skip-f16", action="store_true", help="Skip conversion if model-f16.gguf already exists")
    p.add_argument(
        "--base-dtype",
        choices=["f16", "bf16"],
        default="f16",
        help="Unquantized base GGUF dtype to convert to before quantizing (default: f16)",
    )
    return p.parse_args()


def main() -> None:
    """Orchestrate the full download → convert → quantize → validate → deploy pipeline."""
    ensure_active_venv()

    args = parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_levels = QUANT_LEVELS if args.quant == "ALL" else [args.quant]

    require(LLAMA_CPP_DIR.exists(), f"llama.cpp not found at {LLAMA_CPP_DIR}. Clone it first.")

    # Start the clock here so conversion_time covers the full pipeline
    t_start = time.time()

    # 1. Download / resolve
    model_dir = resolve_model(args.model, output_dir)
    original_format = detect_weight_format(model_dir)
    prefix = model_prefix(args.model)
    print(f"[INFO] Output filename prefix: {prefix}")

    # 2. Convert to full-precision GGUF (all quant levels are derived from this)
    f16_path = convert_to_base(model_dir, output_dir, prefix, args.base_dtype)

    # 3. Compress to each requested quantization level
    quant_files = quantize_model(f16_path, output_dir, quant_levels, prefix)

    conversion_time = time.time() - t_start

    # 4. Validate outputs and build the report
    model_name = args.model.split("/")[-1] if "/" in args.model and not Path(args.model).exists() else Path(args.model).name
    report = validate_and_report(
        model_name=model_name,
        original_format=original_format,
        f16_path=f16_path,
        quant_files=quant_files,
        model_dir=model_dir,
        conversion_time=conversion_time,
        base_dtype=args.base_dtype,
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
