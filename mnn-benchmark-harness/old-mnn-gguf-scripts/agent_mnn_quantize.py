#!/usr/bin/env python3
"""
agent_mnn_quantize.py — Automated quantization selection for MNN Chat/Android.

Sweeps a HuggingFace model across several MNN quant_bit levels using
llmexport.py -> adb push -> run_mnn_autobench.py, filters the results
against an RSS/file-size budget, then asks Claude (or a deterministic
fallback rule) to recommend which bit-width to actually ship.

Unlike GGUF's named presets (Q4_K_M/Q5_K_M/Q8_0), MNN quantization is
parameterized directly by --quant_bit (commonly 4 or 8) and --quant_block
(commonly 64), so the sweep is driven by a list of bit-widths rather than
named levels.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_AUTOBENCH = SCRIPT_DIR / "run_mnn_autobench.py"

MNN_ROOT = Path.home() / "SLM_Factory_Krishna_Personal/MNN"
LLMEXPORT_DIR = MNN_ROOT / "transformers/llm/export"
LLMEXPORT_SCRIPT = LLMEXPORT_DIR / "llmexport.py"
LLMEXPORT_VENV_PYTHON = LLMEXPORT_DIR / ".venv" / "bin" / "python"
MNNCONVERT_BIN = MNN_ROOT / "build/MNNConvert"

# Local staging area for downloaded HF snapshots and converted MNN folders,
# kept inside this repo rather than the toolchain dirs above so a re-run
# doesn't touch MNN/llmexport's own working tree.
LOCAL_MODELS_DIR = SCRIPT_DIR / "mnn_models"
DEVICE_MODELS_DIR = "/data/local/tmp/mnn_models"

# Exactly the files convert-completeness is judged on (mirrors
# convert_to_gguf.py's caching behavior) - a successful export also produces
# export_args.json and llm.mnn.json, but those aren't required to treat a
# folder as a reusable, complete conversion.
REQUIRED_FILES = ["config.json", "llm.mnn", "llm.mnn.weight", "llm_config.json", "tokenizer.mtok"]

CLAUDE_MODEL = "claude-sonnet-5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_subprocess(cmd: list, env: dict = None, cwd=None) -> int:
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env, cwd=cwd)
    return result.returncode


def model_prefix(model: str) -> str:
    """Mirror convert_to_gguf.py's model_prefix() for the HF-ID case.

    Example: "Qwen/Qwen2.5-0.5B-Instruct" -> "qwen2.5-0.5b-instruct"
    """
    local = Path(model)
    name = local.name if local.exists() and local.is_dir() else model.split("/")[-1]
    return name.lower().replace("_", "-").replace(" ", "-")


def quant_slug(model: str, quant_bit: int) -> str:
    return f"{model_prefix(model)}-mnn-q{quant_bit}"


def is_conversion_complete(output_dir: Path) -> bool:
    return output_dir.is_dir() and all((output_dir / f).exists() for f in REQUIRED_FILES)


def combined_size_mb(output_dir: Path):
    total = 0
    found_any = False
    for name in ("llm.mnn", "llm.mnn.weight"):
        p = output_dir / name
        if p.exists():
            total += p.stat().st_size
            found_any = True
    return round(total / (1024 * 1024), 1) if found_any else None


def _load_run_autobench_module():
    """Load run_mnn_autobench.py by path so its adb-resolution logic
    (PATH first, then the ~/Library/Android/sdk fallback) can be reused
    directly rather than reimplemented here, mirroring agent_quantize.py's
    own precedent of importing convert_to_gguf.py for model_prefix()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_run_mnn_autobench", str(RUN_AUTOBENCH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Download / conversion / push
# ---------------------------------------------------------------------------

def resolve_local_model_dir(model: str) -> dict:
    """Return a local directory containing the model's weights and config,
    downloading it once via huggingface_hub if `model` isn't already a local
    path. Shared across every quant_bit in the sweep."""
    local = Path(model)
    if local.exists() and local.is_dir():
        print(f"[DOWNLOAD] Using local model at {local}")
        return {"ok": True, "local_path": local}

    dest = LOCAL_MODELS_DIR / f"{model_prefix(model)}-hf-raw"
    if dest.exists() and any(dest.iterdir()):
        print(f"[DOWNLOAD] Reusing existing local snapshot: {dest}")
        return {"ok": True, "local_path": dest}

    print(f"[DOWNLOAD] Fetching {model} from HuggingFace Hub...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {"ok": False, "error": "huggingface_hub not installed (pip install huggingface_hub)"}

    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=model,
            local_dir=str(dest),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"[DOWNLOAD] Saved to {path}")
        return {"ok": True, "local_path": Path(path)}
    except Exception as e:
        return {"ok": False, "error": f"download failed: {e}"}


def find_llmexport_interpreter():
    """Return the llmexport.py-colocated venv's python, or None if it's
    missing - checked lazily per quant_bit so already-cached conversions can
    still be pushed/benchmarked even if the export toolchain isn't set up."""
    if LLMEXPORT_VENV_PYTHON.exists():
        return str(LLMEXPORT_VENV_PYTHON)
    return None


def convert_one(interpreter: str, local_model_dir: Path, quant_bit: int, quant_block: int, output_dir: Path) -> dict:
    if not LLMEXPORT_SCRIPT.exists():
        return {"ok": False, "error": f"llmexport.py not found at {LLMEXPORT_SCRIPT}"}
    if not MNNCONVERT_BIN.exists():
        return {"ok": False, "error": f"MNNConvert binary not found at {MNNCONVERT_BIN} (--mnnconvert is required - without it, conversion crashes with a Bus error via the broken pymnn bindings)"}

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        interpreter, str(LLMEXPORT_SCRIPT),
        "--path", str(local_model_dir),
        "--export", "mnn",
        "--quant_bit", str(quant_bit),
        "--quant_block", str(quant_block),
        "--dst_path", str(output_dir),
        "--mnnconvert", str(MNNCONVERT_BIN),
    ]
    rc = run_subprocess(cmd, cwd=str(LLMEXPORT_SCRIPT.parent))
    if rc != 0:
        return {"ok": False, "error": f"llmexport.py exited with code {rc}"}

    if not is_conversion_complete(output_dir):
        missing = [f for f in REQUIRED_FILES if not (output_dir / f).exists()]
        return {"ok": False, "error": f"llmexport.py exited 0 but expected files are missing: {missing}"}

    return {"ok": True}


def push_to_device(adb_bin: str, local_dir: Path, slug: str) -> dict:
    mkdir_result = subprocess.run(
        [adb_bin, "shell", "mkdir", "-p", DEVICE_MODELS_DIR],
        capture_output=True, text=True, timeout=15,
    )
    if mkdir_result.returncode != 0:
        return {"ok": False, "error": f"adb shell mkdir -p {DEVICE_MODELS_DIR} failed: {mkdir_result.stderr.strip()}"}

    print(f"\n[PUSH] {local_dir} -> {DEVICE_MODELS_DIR}/{slug}")
    push_result = subprocess.run(
        [adb_bin, "push", str(local_dir), DEVICE_MODELS_DIR + "/"],
        capture_output=True, text=True, timeout=900,
    )
    if push_result.returncode != 0:
        return {"ok": False, "error": f"adb push failed: {(push_result.stderr or push_result.stdout).strip()}"}

    device_path = f"{DEVICE_MODELS_DIR}/{slug}"
    return {"ok": True, "device_path": device_path}


def benchmark_one(device_model_path: str, questions_path, timeout: int, bench_output_path: Path) -> dict:
    """Run run_mnn_autobench.py against a pushed device folder and parse its
    summary. Returns {"ok": True, ...means...} or {"ok": False, "error": str}."""
    if not RUN_AUTOBENCH.exists():
        return {"ok": False, "error": f"run_mnn_autobench.py not found at {RUN_AUTOBENCH}"}

    cmd = [
        sys.executable, str(RUN_AUTOBENCH),
        "--model-path", device_model_path,
        "--output", str(bench_output_path),
        "--timeout", str(timeout),
    ]
    if questions_path:
        cmd += ["--questions", str(questions_path)]

    rc = run_subprocess(cmd)
    if rc != 0:
        return {"ok": False, "error": f"run_mnn_autobench.py exited with code {rc}"}

    if not bench_output_path.exists():
        return {"ok": False, "error": f"results file not written: {bench_output_path}"}

    try:
        with open(bench_output_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"ok": False, "error": f"failed to read {bench_output_path}: {e}"}

    run_info = data.get("run_info", {})
    completed = run_info.get("completed", 0)
    total = run_info.get("total", 0)
    if completed == 0:
        return {"ok": False, "error": f"0/{total} questions completed successfully"}

    summary = data.get("summary", {})

    def mean_of(key):
        return summary.get(key, {}).get("mean")

    return {
        "ok": True,
        "peak_rss_kb": mean_of("peak_rss_kb"),
        "ttft_ms": mean_of("ttft_ms"),
        "prefill_tps": mean_of("prefill_tps"),
        "decode_tps": mean_of("decode_tps"),
        "n_completed": completed,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Sweep phase
# ---------------------------------------------------------------------------

def sweep(model: str, quant_bits: list, quant_block: int, questions_path, timeout: int,
          budget_rss_mb, budget_file_size_mb, output_stem: str) -> list:
    adb_module = _load_run_autobench_module()
    adb_bin = adb_module.find_adb()  # exits(1) with a clear message if adb isn't found at all

    local_model_dir_cache = {}  # resolved lazily, at most once, shared across bits

    def get_local_model_dir():
        if "result" not in local_model_dir_cache:
            local_model_dir_cache["result"] = resolve_local_model_dir(model)
        return local_model_dir_cache["result"]

    results = []

    for bit in quant_bits:
        label = f"Q{bit}"
        print("\n" + "=" * 60)
        print(f"QUANT LEVEL: {label} (quant_bit={bit}, quant_block={quant_block})")
        print("=" * 60)

        slug = quant_slug(model, bit)
        output_dir = LOCAL_MODELS_DIR / slug

        row = {
            "quant_bit": bit, "quant_block": quant_block, "slug": slug,
            "file_size_mb": None, "peak_rss_mb": None, "ttft_ms": None,
            "prefill_tps": None, "decode_tps": None, "N_completed": 0,
            "fits_rss_budget": False, "fits_file_size_budget": False, "fits_budget": False,
            "status": "failed", "error": None,
        }

        if is_conversion_complete(output_dir):
            print(f"[{label}] Reusing existing conversion at {output_dir}")
        else:
            interpreter = find_llmexport_interpreter()
            if interpreter is None:
                row["error"] = f"llmexport.py venv not found at {LLMEXPORT_VENV_PYTHON}"
                print(f"[{label}] {row['error']} -- skipping")
                results.append(row)
                continue

            dl = get_local_model_dir()
            if not dl["ok"]:
                row["error"] = dl["error"]
                print(f"[{label}] {row['error']} -- skipping")
                results.append(row)
                continue

            conv = convert_one(interpreter, dl["local_path"], bit, quant_block, output_dir)
            if not conv["ok"]:
                row["error"] = conv["error"]
                print(f"[{label}] Conversion failed: {conv['error']} -- skipping")
                results.append(row)
                continue

        file_size_mb = combined_size_mb(output_dir)
        row["file_size_mb"] = file_size_mb

        push = push_to_device(adb_bin, output_dir, slug)
        if not push["ok"]:
            row["error"] = push["error"]
            print(f"[{label}] Push failed: {push['error']} -- skipping")
            results.append(row)
            continue

        bench_output_path = Path(f"{output_stem}_{label.lower()}_bench.json")
        bench = benchmark_one(push["device_path"], questions_path, timeout, bench_output_path)
        if not bench["ok"]:
            row["error"] = bench["error"]
            print(f"[{label}] Benchmark failed: {bench['error']} -- skipping")
            results.append(row)
            continue

        peak_rss_mb = round(bench["peak_rss_kb"] / 1024, 1) if bench["peak_rss_kb"] is not None else None
        ttft_ms = bench["ttft_ms"]
        prefill_tps = round(bench["prefill_tps"], 1) if bench["prefill_tps"] is not None else None
        decode_tps = round(bench["decode_tps"], 1) if bench["decode_tps"] is not None else None

        # A budget axis that wasn't requested is trivially satisfied, so the
        # combined AND reduces to whichever axis (or both) is actually active.
        fits_file_size_budget = (
            True if budget_file_size_mb is None
            else (file_size_mb is not None and file_size_mb <= budget_file_size_mb)
        )
        fits_rss_budget = (
            True if budget_rss_mb is None
            else (peak_rss_mb is not None and peak_rss_mb <= budget_rss_mb)
        )
        fits_budget = fits_file_size_budget and fits_rss_budget

        size_disp = f"{file_size_mb:.0f}MB" if file_size_mb is not None else "N/A"
        rss_disp = f"{peak_rss_mb:.0f}MB" if peak_rss_mb is not None else "N/A"
        ttft_disp = f"{ttft_ms:.0f}ms" if ttft_ms is not None else "N/A"
        prefill_disp = f"{prefill_tps:.0f}" if prefill_tps is not None else "N/A"
        decode_disp = f"{decode_tps:.0f}" if decode_tps is not None else "N/A"
        print(
            f"\n[{label}] RSS={rss_disp} TTFT={ttft_disp} PrefillTPS={prefill_disp} "
            f"DecodeTPS={decode_disp} Size={size_disp} - checking budget..."
        )

        fit_desc = []
        if budget_file_size_mb is not None:
            fit_desc.append(f"size {'fits' if fits_file_size_budget else 'EXCEEDS'} {budget_file_size_mb:.0f}MB")
        if budget_rss_mb is not None:
            fit_desc.append(f"RSS {'fits' if fits_rss_budget else 'EXCEEDS'} {budget_rss_mb:.0f}MB")
        print(f"  -> {' | '.join(fit_desc)}")

        row.update({
            "file_size_mb": file_size_mb,
            "peak_rss_mb": peak_rss_mb,
            "ttft_ms": ttft_ms,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "N_completed": bench["n_completed"],
            "fits_rss_budget": fits_rss_budget,
            "fits_file_size_budget": fits_file_size_budget,
            "fits_budget": fits_budget,
            "status": "success",
            "error": None,
        })
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Reasoning phase
# ---------------------------------------------------------------------------

def _budget_desc(budget_rss_mb, budget_file_size_mb, joiner: str = "and") -> str:
    bits = []
    if budget_file_size_mb is not None:
        bits.append(f"{budget_file_size_mb:.0f}MB file size")
    if budget_rss_mb is not None:
        bits.append(f"{budget_rss_mb:.0f}MB peak RSS")
    return f" {joiner} ".join(bits)


def deterministic_recommendation(sweep_results: list, candidates_within_budget: list,
                                  budget_rss_mb, budget_file_size_mb) -> tuple:
    successful = [r for r in sweep_results if r["status"] == "success"]
    budget_desc = _budget_desc(budget_rss_mb, budget_file_size_mb)

    if candidates_within_budget:
        candidates = [r for r in successful if r["quant_bit"] in candidates_within_budget]
        best = max(candidates, key=lambda r: r["decode_tps"] if r["decode_tps"] is not None else -1)
        reasoning = (
            f"Deterministic fallback (no LLM reasoning): selected Q{best['quant_bit']} because it has the "
            f"highest DecodeTPS ({best['decode_tps']}) among quant_bit levels that satisfy {budget_desc} "
            f"(candidates: {', '.join('Q' + str(b) for b in candidates_within_budget)})."
        )
        return best["quant_bit"], reasoning

    if not successful:
        return None, "Deterministic fallback (no LLM reasoning): no quant_bit level completed benchmarking successfully; no recommendation possible."

    # Rank by proportional overage across whichever budgets are active, so a
    # candidate that's only slightly over stays preferable to one wildly over.
    def overage_score(r):
        score = 0.0
        if budget_file_size_mb is not None and r["file_size_mb"] is not None:
            score += max(0.0, (r["file_size_mb"] / budget_file_size_mb) - 1.0)
        if budget_rss_mb is not None and r["peak_rss_mb"] is not None:
            score += max(0.0, (r["peak_rss_mb"] / budget_rss_mb) - 1.0)
        return score

    closest = min(successful, key=overage_score)

    miss_parts = []
    if budget_file_size_mb is not None and closest["file_size_mb"] is not None:
        if closest["file_size_mb"] > budget_file_size_mb:
            miss_parts.append(f"file size {closest['file_size_mb'] - budget_file_size_mb:.0f}MB over the {budget_file_size_mb:.0f}MB budget")
        else:
            miss_parts.append(f"file size satisfies the {budget_file_size_mb:.0f}MB budget")
    if budget_rss_mb is not None and closest["peak_rss_mb"] is not None:
        if closest["peak_rss_mb"] > budget_rss_mb:
            miss_parts.append(f"peak RSS {closest['peak_rss_mb'] - budget_rss_mb:.0f}MB over the {budget_rss_mb:.0f}MB budget")
        else:
            miss_parts.append(f"peak RSS satisfies the {budget_rss_mb:.0f}MB budget")
    miss_desc = "; ".join(miss_parts) if miss_parts else "a missing metric prevented a direct comparison"

    reasoning = (
        f"Deterministic fallback (no LLM reasoning): no tested quant_bit level satisfied {budget_desc}. "
        f"Recommending Q{closest['quant_bit']} as the closest option ({miss_desc}). Consider a smaller model "
        f"or a higher budget if this is a hard constraint."
    )
    return closest["quant_bit"], reasoning


def extract_recommended_bit(text: str, tested_bits: list):
    lower = text.lower()
    markers = ["i recommend", "recommendation:", "recommend using", "recommend deploying", "final recommendation"]
    for marker in markers:
        idx = lower.find(marker)
        if idx != -1:
            snippet = lower[idx:idx + 200]
            for bit in tested_bits:
                if f"q{bit}" in snippet or f"{bit}-bit" in snippet:
                    return bit

    counts = {bit: lower.count(f"q{bit}") + lower.count(f"{bit}-bit") for bit in tested_bits}
    if any(counts.values()):
        return max(counts, key=counts.get)
    return None


def llm_recommendation(sweep_results: list, budget_rss_mb, budget_file_size_mb, model_name: str) -> tuple:
    import anthropic

    client = anthropic.Anthropic()
    budget_desc = _budget_desc(budget_rss_mb, budget_file_size_mb, joiner="and")

    lines = []
    for r in sweep_results:
        if r["status"] == "success":
            lines.append(
                f"Q{r['quant_bit']} (quant_block={r['quant_block']}): file_size={r['file_size_mb']}MB "
                f"peak_RSS={r['peak_rss_mb']}MB TTFT={r['ttft_ms']}ms PrefillTPS={r['prefill_tps']} "
                f"DecodeTPS={r['decode_tps']} N_completed={r['N_completed']} "
                f"fits_file_size_budget={r['fits_file_size_budget']} fits_rss_budget={r['fits_rss_budget']}"
            )
        else:
            lines.append(f"Q{r['quant_bit']}: FAILED ({r['error']})")

    prompt = f"""We benchmarked the model "{model_name}" at several MNN quant_bit levels on a real Android phone \
via MNN Chat's headless benchmark receiver.
Active budget(s): {budget_desc}.

MNN reports prefill and decode throughput separately (PrefillTPS, DecodeTPS) rather than one combined TPS number.

Results:
{chr(10).join(lines)}

Recommend which quant_bit level to deploy. Reason about the actual tradeoff - for example, a larger/slower-loading
but higher-quality bit-width might be worth it if it still satisfies the active budget(s), rather than always
picking the smallest one that fits. Weigh DecodeTPS (sustained generation speed) and PrefillTPS (prompt processing
speed) as distinct considerations. If none satisfy every active budget, recommend the closest option and explain
by how much it misses each budget and what the practical impact is. Be explicit about which budget(s) - file size,
peak RSS, or both - your final recommendation actually satisfies. State your final recommended quant_bit level
clearly by name (e.g. Q4 or Q8)."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "\n".join(block.text for block in response.content if getattr(block, "type", None) == "text")

    tested_bits = [r["quant_bit"] for r in sweep_results]
    recommended = extract_recommended_bit(text, tested_bits)
    return recommended, text


def get_recommendation(sweep_results: list, candidates_within_budget: list,
                        budget_rss_mb, budget_file_size_mb, model_name: str) -> tuple:
    """Returns (quant_bit, reasoning_text, reasoning_source)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    skip_reason = None

    if not api_key:
        skip_reason = "ANTHROPIC_API_KEY environment variable is not set"
    else:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            skip_reason = "the 'anthropic' package is not installed (run: pip install anthropic)"
        else:
            try:
                bit, text = llm_recommendation(sweep_results, budget_rss_mb, budget_file_size_mb, model_name)
                if bit is None:
                    print("[LLM] Claude responded but no tested quant_bit level could be identified in the reasoning text; falling back to deterministic rule.")
                else:
                    return bit, text, "llm"
            except Exception as e:
                skip_reason = f"Claude API call failed: {e}"

    print(f"[LLM] Skipping LLM reasoning step: {skip_reason}. Falling back to deterministic rule.")
    bit, reasoning = deterministic_recommendation(sweep_results, candidates_within_budget, budget_rss_mb, budget_file_size_mb)
    return bit, reasoning, "deterministic_fallback"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(sweep_results: list, budget_rss_mb, budget_file_size_mb):
    print("\n" + "=" * 100)
    print("SWEEP SUMMARY")
    print("=" * 100)
    header = f"{'Quant':<8}{'Size(MB)':>10}{'RSS(MB)':>10}{'TTFT(ms)':>10}{'PrefillTPS':>12}{'DecodeTPS':>11}{'N':>4}{'FitsSize':>10}{'FitsRSS':>9}{'Status':>10}"
    print(header)
    print("-" * len(header))
    for r in sweep_results:
        def fmt(v, spec=""):
            return format(v, spec) if v is not None else "N/A"

        fits_size_disp = "N/A" if budget_file_size_mb is None else ("YES" if r["fits_file_size_budget"] else "NO")
        fits_rss_disp = "N/A" if budget_rss_mb is None else ("YES" if r["fits_rss_budget"] else "NO")

        row = (
            f"{'Q' + str(r['quant_bit']):<8}"
            f"{fmt(r['file_size_mb'], '.0f'):>10}"
            f"{fmt(r['peak_rss_mb'], '.0f'):>10}"
            f"{fmt(r['ttft_ms'], '.0f'):>10}"
            f"{fmt(r['prefill_tps'], '.1f'):>12}"
            f"{fmt(r['decode_tps'], '.1f'):>11}"
            f"{r['N_completed']:>4}"
            f"{fits_size_disp:>10}"
            f"{fits_rss_disp:>9}"
            f"{r['status']:>10}"
        )
        print(row)

    budget_bits = []
    if budget_file_size_mb is not None:
        budget_bits.append(f"file size <= {budget_file_size_mb:.0f}MB")
    if budget_rss_mb is not None:
        budget_bits.append(f"peak RSS <= {budget_rss_mb:.0f}MB")
    print(f"\nBudget: {' AND '.join(budget_bits)}")


def save_report(output_path: str, model: str, budget_rss_mb, budget_file_size_mb, sweep_results: list,
                 candidates_within_budget: list, quant_bit, reasoning: str, reasoning_source: str):
    report = {
        "model": model,
        "budget_rss_mb": budget_rss_mb,
        "budget_file_size_mb": budget_file_size_mb,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sweep_results": sweep_results,
        "candidates_within_budget": candidates_within_budget,
        "recommendation": {
            "quant_bit": quant_bit,
            "reasoning": reasoning,
            "reasoning_source": reasoning_source,
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OUTPUT] Report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Automated quantization selection agent for MNN Chat/Android")
    p.add_argument("--model", required=True, help="HuggingFace model ID (or local folder containing HF weights)")
    p.add_argument("--budget-rss-mb", type=float, default=None, dest="budget_rss_mb",
                    help="Maximum acceptable Peak RSS in MB (optional; at least one of --budget-rss-mb / --budget-file-size-mb is required)")
    p.add_argument("--budget-file-size-mb", type=float, default=None, dest="budget_file_size_mb",
                    help="Maximum acceptable combined llm.mnn+llm.mnn.weight size in MB (optional; at least one of --budget-rss-mb / --budget-file-size-mb is required)")
    p.add_argument("--quant-bits", default="4,8", help="Comma-separated quant_bit values to sweep")
    p.add_argument("--quant-block", type=int, default=64, help="quant_block value, applied to every level in the sweep")
    p.add_argument("--questions", default=None, help="Path to questions file (default: run_mnn_autobench.py's built-in defaults)")
    p.add_argument("--timeout", type=int, default=120,
                    help="Seconds passed through to run_mnn_autobench.py's --timeout per question. Default is 120 "
                         "(not 60) because real testing has shown MNN needs more than 60s for some questions, "
                         "especially reasoning-mode responses.")
    p.add_argument("--output", default=str(SCRIPT_DIR / "logs" / "mnn_agent_report.json"))
    return p.parse_args()


def main():
    args = parse_args()

    if args.budget_rss_mb is None and args.budget_file_size_mb is None:
        print("[ERROR] At least one of --budget-rss-mb or --budget-file-size-mb is required.")
        sys.exit(1)

    try:
        quant_bits = [int(b.strip()) for b in args.quant_bits.split(",") if b.strip()]
    except ValueError:
        print(f"[ERROR] --quant-bits must be a comma-separated list of integers, got: {args.quant_bits!r}")
        sys.exit(1)
    if not quant_bits:
        print("[ERROR] --quant-bits resolved to an empty list")
        sys.exit(1)

    if args.questions:
        questions_path = os.path.expanduser(args.questions)
        if not os.path.exists(questions_path):
            print(f"[ERROR] Questions file not found: {questions_path}")
            sys.exit(1)
    else:
        questions_path = None

    budget_desc = _budget_desc(args.budget_rss_mb, args.budget_file_size_mb, joiner="AND")

    print("=" * 60)
    print("Agentic MNN Quantization Selector")
    print(f"  Model: {args.model}  Budget: {budget_desc}")
    print(f"  Quant bits: {', '.join('Q' + b for b in args.quant_bits.split(','))}  Quant block: {args.quant_block}  Timeout: {args.timeout}s")
    print("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = str(output_path.with_suffix(""))
    sweep_results = sweep(args.model, quant_bits, args.quant_block, questions_path, args.timeout,
                           args.budget_rss_mb, args.budget_file_size_mb, output_stem)

    candidates_within_budget = [r["quant_bit"] for r in sweep_results if r["status"] == "success" and r["fits_budget"]]

    print("\n" + "=" * 60)
    if candidates_within_budget:
        print(f"[FILTER] Candidates satisfying {budget_desc}: {', '.join('Q' + str(b) for b in candidates_within_budget)}")
    else:
        print(f"[FILTER] NONE of the tested quant_bit levels satisfy {budget_desc}. Proceeding to reasoning phase anyway.")
    print("=" * 60)

    quant_bit, reasoning, reasoning_source = get_recommendation(
        sweep_results, candidates_within_budget, args.budget_rss_mb, args.budget_file_size_mb, args.model
    )

    print_summary_table(sweep_results, args.budget_rss_mb, args.budget_file_size_mb)

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"Quant bit: {('Q' + str(quant_bit)) if quant_bit is not None else 'none'}")
    print(f"Source: {reasoning_source}")
    print(f"Reasoning:\n{reasoning}")

    save_report(args.output, args.model, args.budget_rss_mb, args.budget_file_size_mb, sweep_results,
                candidates_within_budget, quant_bit, reasoning, reasoning_source)


if __name__ == "__main__":
    main()
