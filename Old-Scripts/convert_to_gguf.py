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
LLAMA_CPP_DIR = Path(__file__).resolve().parent / "SmolChat-Android" / "llama.cpp"

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
# f16 = 16 bits = 2 bytes; quantized formats store fewer bits per weight.
QUANT_BPP = {
    "f16":    2.0,
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

def run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    """Execute a shell command, printing it first so the user can see what's running.

    capture=True suppresses stdout/stderr (used when we need to inspect output
    programmatically, e.g. parsing `adb devices`). Otherwise output streams
    live to the terminal so long-running steps like quantization show progress.
    """
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture,
        text=True,
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


def convert_to_f16(model_dir: Path, output_dir: Path, prefix: str) -> Path:
    """Convert a HuggingFace checkpoint to a full-precision (float16) GGUF file.

    Calls llama.cpp's convert_hf_to_gguf.py, which reads the model architecture
    from config.json and writes all weights into a single GGUF container.
    f16 is the lossless intermediate — quantized variants are produced from it
    in the next step so we only need to run this conversion once.
    Skips conversion if the output file already exists (idempotent reruns).
    """
    require(CONVERT_SCRIPT.exists(), f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}")

    fmt = detect_weight_format(model_dir)
    print(f"[CONVERT] Detected weight format: {fmt}")

    f16_path = output_dir / f"{prefix}-f16.gguf"
    if f16_path.exists():
        print(f"[CONVERT] {f16_path.name} already exists, skipping conversion.")
        return f16_path

    # The f16 GGUF will be roughly the same size as the source weights
    check_disk_space(output_dir, file_size_mb(next(model_dir.glob(f"*.{fmt if fmt == 'safetensors' else 'bin'}"))) / 1024 * 2 + 2)

    result = run(
        [sys.executable, str(CONVERT_SCRIPT), str(model_dir), "--outfile", str(f16_path), "--outtype", "f16"],
    )
    if result.returncode != 0:
        print(f"[ERROR] Conversion failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    require(f16_path.exists(), f"Expected {f16_path} after conversion but it was not created.")
    print(f"[CONVERT] ✓ {f16_path.name}  ({file_size_mb(f16_path):.0f} MB)")
    return f16_path


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
        sizes["f16"] = round(mb, 1)
        print(f"  f16 (base):  {mb:>8.0f} MB   RAM {estimate_ram_gb(params, 'f16')}")

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
        output_files["f16"] = f16_path.name
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
    """Push the chosen GGUF file to a connected Android device via ADB.

    Prefers `preferred` (Q4_K_M by default) as the smallest file that still
    runs well on budget phones. Falls back to the first available quant level
    if the preferred one wasn't produced. Prints manual copy instructions if
    ADB isn't found or no device is connected, rather than failing hard.
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

    gguf_path = quant_files.get(preferred)
    if not gguf_path:
        # Preferred level wasn't built; use whatever is available
        available = list(quant_files.keys())
        if not available:
            print("[DEPLOY] No quantized GGUF files to deploy.")
            return
        gguf_path = quant_files[available[0]]
        preferred = available[0]

    remote_path = f"/sdcard/Download/{gguf_path.name}"
    print(f"[DEPLOY] Pushing {gguf_path.name} ({file_size_mb(gguf_path):.0f} MB) → {remote_path}")
    r = run([str(adb), "push", str(gguf_path), remote_path])
    if r.returncode == 0:
        print("[DEPLOY] ✓ Pushed successfully.")
        print(f"\nTo import in SmolChat:")
        print(f"  1. Open SmolChat on your Android device")
        print(f"  2. Tap ☰ → Models → Import model")
        print(f"  3. Navigate to Downloads → {gguf_path.name}")
    else:
        print("[DEPLOY] Push failed.")
        _print_manual_instructions(quant_files, preferred)


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
    return p.parse_args()


def main() -> None:
    """Orchestrate the full download → convert → quantize → validate → deploy pipeline."""
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
    f16_path = convert_to_f16(model_dir, output_dir, prefix)

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
