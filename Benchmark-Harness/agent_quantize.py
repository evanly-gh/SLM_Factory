#!/usr/bin/env python3
"""
agent_quantize.py — Automated quantization selection for SmolChat/Android.

Sweeps a HuggingFace model across several GGUF quantization levels using the
existing, validated convert_to_gguf.py -> run_autobench.py pipeline, filters
the results against an RSS budget, then asks Claude (or a deterministic
fallback rule) to recommend which quant level to actually ship.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_AUTOBENCH = SCRIPT_DIR / "run_autobench.py"
MODEL_CONVERSION_DIR = SCRIPT_DIR.parent / "Model-Conversion"
CONVERT_SCRIPT = MODEL_CONVERSION_DIR / "convert_to_gguf.py"

KNOWN_QUANT_LEVELS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

CLAUDE_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_subprocess(cmd: list, env: dict = None) -> int:
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def find_convert_interpreter() -> str:
    """Pick the Python interpreter used to run convert_to_gguf.py.

    convert_to_gguf.py depends on a torch/numpy pair that must come from a
    single, internally-consistent build, or importing both triggers
    "OMP: Error #15: ... libomp.dylib already initialized". The .venv
    colocated with convert_to_gguf.py is the one confirmed to have a
    matched pair (it's what the user runs manually); sys.executable (the
    interpreter running agent_quantize.py itself) may resolve to an
    unrelated interpreter such as system Python, which reproduces the
    crash. Prefer, in order: the venv next to convert_to_gguf.py, the
    currently active venv (if any), then fall back to sys.executable.
    """
    colocated_venv = MODEL_CONVERSION_DIR / ".venv" / "bin" / "python"
    if colocated_venv.exists():
        return str(colocated_venv)

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        active_venv_python = Path(virtual_env) / "bin" / "python"
        if active_venv_python.exists():
            return str(active_venv_python)

    return sys.executable


def model_prefix(model: str) -> str:
    """Mirror convert_to_gguf.py's model_prefix() exactly, for the HF-ID case
    (agent_quantize.py only ever passes HuggingFace IDs, never local paths)."""
    name = model.split("/")[-1]
    return name.lower().replace("_", "-").replace(" ", "-")


def find_existing_output_dir(model: str) -> Path:
    """Look for a pre-existing output-* directory that already contains GGUF
    files for this model, regardless of what the directory itself is named.

    convert_to_gguf.py's --output defaults to "./output" but is commonly
    overridden by hand to something readable (e.g. "output-qwen-2.5-0.5b-
    instruct"), which doesn't necessarily match model_prefix()'s generated
    slug ("qwen2.5-0.5b-instruct") character-for-character. Matching on the
    actual filenames inside each candidate directory - which do follow
    model_prefix() exactly, since convert_to_gguf.py generates them - finds
    the right directory regardless of how it was named, so already-converted
    GGUFs are reused instead of re-downloaded/re-converted.
    """
    if not MODEL_CONVERSION_DIR.exists():
        return None
    prefix = model_prefix(model)
    for candidate in sorted(MODEL_CONVERSION_DIR.glob("output-*")):
        if candidate.is_dir() and any(candidate.glob(f"{prefix}-*.gguf")):
            return candidate
    return None


def model_output_dir(model: str) -> Path:
    existing = find_existing_output_dir(model)
    if existing is not None:
        return existing
    return MODEL_CONVERSION_DIR / f"output-{model_prefix(model)}"


# ---------------------------------------------------------------------------
# Sweep phase
# ---------------------------------------------------------------------------

def convert_one(model: str, quant: str, output_dir: Path) -> dict:
    """Run convert_to_gguf.py for a single quant level and read its report.

    Returns {"ok": True, "local_path": Path, "size_mb": float} on success,
    or {"ok": False, "error": str} on any failure.
    """
    if not CONVERT_SCRIPT.exists():
        return {"ok": False, "error": f"convert_to_gguf.py not found at {CONVERT_SCRIPT}"}

    cmd = [
        find_convert_interpreter(), str(CONVERT_SCRIPT),
        "--model", model,
        "--output", str(output_dir),
        "--quant", quant,
        "--deploy",
    ]
    # Scoped to this subprocess only: convert_hf_to_gguf.py / llama-quantize
    # can crash with "OMP: Error #15 ... libomp.dylib already initialized"
    # when the interpreter's numpy/torch pair links a second copy of
    # libomp. This unblocks it without touching agent_quantize.py's own
    # environment or any other subprocess it launches.
    convert_env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    rc = run_subprocess(cmd, env=convert_env)
    if rc != 0:
        return {"ok": False, "error": f"convert_to_gguf.py exited with code {rc}"}

    report_path = output_dir / "conversion_report.json"
    if not report_path.exists():
        return {"ok": False, "error": f"conversion_report.json not found at {report_path}"}

    try:
        with open(report_path) as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"ok": False, "error": f"failed to read conversion_report.json: {e}"}

    filename = report.get("output_files", {}).get(quant)
    if not filename:
        return {"ok": False, "error": f"'{quant}' missing from output_files in conversion_report.json"}

    local_path = output_dir / filename
    if not local_path.exists():
        return {"ok": False, "error": f"expected GGUF file missing on disk: {local_path}"}

    size_mb = report.get("quantization_sizes_mb", {}).get(quant)
    return {"ok": True, "local_path": local_path, "size_mb": size_mb}


def benchmark_one(gguf_path: Path, questions_path, bench_output_path: Path, verify_cold_load: bool = False) -> dict:
    """Run run_autobench.py against a local GGUF file and parse its summary.

    Returns {"ok": True, "ttft_ms": ..., "tps": ..., "memory_kb": ...} on
    success, or {"ok": False, "error": str} on any failure.
    """
    if not RUN_AUTOBENCH.exists():
        return {"ok": False, "error": f"run_autobench.py not found at {RUN_AUTOBENCH}"}

    cmd = [sys.executable, str(RUN_AUTOBENCH), "--model", str(gguf_path), "--output", str(bench_output_path)]
    if questions_path:
        cmd += ["--questions", str(questions_path)]
    if verify_cold_load:
        cmd += ["--reboot-before"]

    rc = run_subprocess(cmd)
    if rc != 0:
        return {"ok": False, "error": f"run_autobench.py exited with code {rc}"}

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
    return {
        "ok": True,
        "ttft_ms": summary.get("ttft_ms", {}).get("mean"),
        "tps": summary.get("tps", {}).get("mean"),
        "memory_kb": summary.get("memory_kb", {}).get("mean"),
        "cold_load_ms": summary.get("first_question_cold_load_ms"),
        "completed": completed,
        "total": total,
    }


def sweep(model: str, quant_levels: list, questions_path, budget_rss_mb, budget_file_size_mb, output_stem: str,
          verify_cold_load: bool = False) -> list:
    output_dir = model_output_dir(model)
    results = []

    for level in quant_levels:
        print("\n" + "=" * 60)
        print(f"QUANT LEVEL: {level}")
        print("=" * 60)

        if level not in KNOWN_QUANT_LEVELS:
            print(f"[{level}] Unrecognized quant level (expected one of {KNOWN_QUANT_LEVELS}) -- skipping")
            results.append({
                "quant": level, "file_size_mb": None, "peak_RSS_mb": None,
                "TTFT_ms": None, "TPS": None,
                "fits_file_size_budget": False, "fits_rss_budget": False, "fits_budget": False,
                "status": "failed", "error": "unrecognized quant level",
            })
            continue

        conv = convert_one(model, level, output_dir)
        if not conv["ok"]:
            print(f"[{level}] Conversion failed: {conv['error']} -- skipping")
            results.append({
                "quant": level, "file_size_mb": None, "peak_RSS_mb": None,
                "TTFT_ms": None, "TPS": None,
                "fits_file_size_budget": False, "fits_rss_budget": False, "fits_budget": False,
                "status": "failed", "error": conv["error"],
            })
            continue

        bench_output_path = Path(f"{output_stem}_{level.lower()}_bench.json")
        bench = benchmark_one(conv["local_path"], questions_path, bench_output_path, verify_cold_load=verify_cold_load)
        if not bench["ok"]:
            print(f"[{level}] Benchmark failed: {bench['error']} -- skipping")
            results.append({
                "quant": level, "file_size_mb": conv["size_mb"], "peak_RSS_mb": None,
                "TTFT_ms": None, "TPS": None,
                "fits_file_size_budget": False, "fits_rss_budget": False, "fits_budget": False,
                "status": "failed", "error": bench["error"],
            })
            continue

        rss_mb = round(bench["memory_kb"] / 1024, 1) if bench["memory_kb"] is not None else None
        ttft_ms = bench["ttft_ms"]
        tps = round(bench["tps"], 3) if bench["tps"] is not None else None
        file_size_mb = conv["size_mb"]

        # A budget axis that wasn't requested is trivially satisfied, so the
        # combined AND reduces to whichever axis (or both) is actually active.
        fits_file_size_budget = (
            True if budget_file_size_mb is None
            else (file_size_mb is not None and file_size_mb <= budget_file_size_mb)
        )
        fits_rss_budget = (
            True if budget_rss_mb is None
            else (rss_mb is not None and rss_mb <= budget_rss_mb)
        )
        fits_budget = fits_file_size_budget and fits_rss_budget

        size_disp = f"{file_size_mb:.0f}MB" if file_size_mb is not None else "N/A"
        rss_disp = f"{rss_mb:.0f}MB" if rss_mb is not None else "N/A"
        ttft_disp = f"{ttft_ms:.0f}ms" if ttft_ms is not None else "N/A"
        tps_disp = f"{tps:.1f}" if tps is not None else "N/A"

        # Only surface ColdLoad when this level was actually rebooted before
        # benchmarking - otherwise it's a warm reload within an
        # already-booted process and would be a misleading number to show
        # alongside genuinely cold-verified levels in the same sweep.
        cold_load_ms = bench.get("cold_load_ms")
        if verify_cold_load and cold_load_ms is not None:
            cold_disp = f"ColdLoad={cold_load_ms:.0f}ms "
        else:
            cold_disp = ""
        print(f"\n[{level}] {cold_disp}Size={size_disp} RSS={rss_disp} TTFT={ttft_disp} TPS={tps_disp} - checking...")

        fit_desc = []
        if budget_file_size_mb is not None:
            fit_desc.append(f"size {'fits' if fits_file_size_budget else 'EXCEEDS'} {budget_file_size_mb:.0f}MB")
        if budget_rss_mb is not None:
            fit_desc.append(f"RSS {'fits' if fits_rss_budget else 'EXCEEDS'} {budget_rss_mb:.0f}MB")
        print(f"  -> {' | '.join(fit_desc)}")

        results.append({
            "quant": level,
            "file_size_mb": file_size_mb,
            "peak_RSS_mb": rss_mb,
            "TTFT_ms": ttft_ms,
            "TPS": tps,
            "fits_file_size_budget": fits_file_size_budget,
            "fits_rss_budget": fits_rss_budget,
            "fits_budget": fits_budget,
            "status": "success",
            "error": None,
        })

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
        candidates = [r for r in successful if r["quant"] in candidates_within_budget]
        best = max(candidates, key=lambda r: r["TPS"] if r["TPS"] is not None else -1)
        reasoning = (
            f"Deterministic fallback (no LLM reasoning): selected {best['quant']} because it has the "
            f"highest TPS ({best['TPS']}) among quant levels that satisfy {budget_desc} "
            f"(candidates: {', '.join(candidates_within_budget)})."
        )
        return best["quant"], reasoning

    if not successful:
        return None, "Deterministic fallback (no LLM reasoning): no quant level completed benchmarking successfully; no recommendation possible."

    # Rank by proportional overage across whichever budgets are active, so a
    # candidate that's only slightly over stays preferable to one wildly over,
    # regardless of which axis (or both) is doing the constraining.
    def overage_score(r):
        score = 0.0
        if budget_file_size_mb is not None and r["file_size_mb"] is not None:
            score += max(0.0, (r["file_size_mb"] / budget_file_size_mb) - 1.0)
        if budget_rss_mb is not None and r["peak_RSS_mb"] is not None:
            score += max(0.0, (r["peak_RSS_mb"] / budget_rss_mb) - 1.0)
        return score

    closest = min(successful, key=overage_score)

    miss_parts = []
    if budget_file_size_mb is not None and closest["file_size_mb"] is not None:
        if closest["file_size_mb"] > budget_file_size_mb:
            miss_parts.append(f"file size {closest['file_size_mb'] - budget_file_size_mb:.0f}MB over the {budget_file_size_mb:.0f}MB budget")
        else:
            miss_parts.append(f"file size satisfies the {budget_file_size_mb:.0f}MB budget")
    if budget_rss_mb is not None and closest["peak_RSS_mb"] is not None:
        if closest["peak_RSS_mb"] > budget_rss_mb:
            miss_parts.append(f"peak RSS {closest['peak_RSS_mb'] - budget_rss_mb:.0f}MB over the {budget_rss_mb:.0f}MB budget")
        else:
            miss_parts.append(f"peak RSS satisfies the {budget_rss_mb:.0f}MB budget")
    miss_desc = "; ".join(miss_parts) if miss_parts else "a missing metric prevented a direct comparison"

    reasoning = (
        f"Deterministic fallback (no LLM reasoning): no tested quant level satisfied {budget_desc}. "
        f"Recommending {closest['quant']} as the closest option ({miss_desc}). Consider a smaller model "
        f"or a higher budget if this is a hard constraint."
    )
    return closest["quant"], reasoning


def extract_recommended_quant(text: str, tested_levels: list):
    lower = text.lower()
    markers = ["i recommend", "recommendation:", "recommend using", "recommend deploying", "final recommendation"]
    for marker in markers:
        idx = lower.find(marker)
        if idx != -1:
            snippet = text[idx:idx + 200]
            for level in tested_levels:
                if level.lower() in snippet.lower():
                    return level

    counts = {level: lower.count(level.lower()) for level in tested_levels}
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
                f"{r['quant']}: file_size={r['file_size_mb']}MB peak_RSS={r['peak_RSS_mb']}MB "
                f"TTFT={r['TTFT_ms']}ms TPS={r['TPS']} fits_file_size_budget={r['fits_file_size_budget']} "
                f"fits_rss_budget={r['fits_rss_budget']}"
            )
        else:
            lines.append(f"{r['quant']}: FAILED ({r['error']})")

    prompt = f"""We benchmarked the model "{model_name}" at several GGUF quantization levels on a real Android phone.
Active budget(s): {budget_desc}.

Results:
{chr(10).join(lines)}

Recommend which quantization level to deploy. Reason about the actual tradeoff - for example, a larger/faster
quant level might be worth it if it still satisfies the active budget(s), rather than always picking the
smallest one that fits. If none satisfy every active budget, recommend the closest option and explain by how
much it misses each budget and what the practical impact is. Be explicit about which budget(s) - file size,
peak RSS, or both - your final recommendation actually satisfies. State your final recommended quant level
clearly by name (e.g. Q4_K_M, Q5_K_M, or Q8_0)."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "\n".join(block.text for block in response.content if getattr(block, "type", None) == "text")

    tested_levels = [r["quant"] for r in sweep_results]
    recommended = extract_recommended_quant(text, tested_levels)
    return recommended, text


def get_recommendation(sweep_results: list, candidates_within_budget: list,
                        budget_rss_mb, budget_file_size_mb, model_name: str) -> tuple:
    """Returns (quant_level, reasoning_text, reasoning_source)."""
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
                quant, text = llm_recommendation(sweep_results, budget_rss_mb, budget_file_size_mb, model_name)
                if quant is None:
                    print("[LLM] Claude responded but no tested quant level could be identified in the reasoning text; falling back to deterministic rule.")
                else:
                    return quant, text, "llm"
            except Exception as e:
                skip_reason = f"Claude API call failed: {e}"

    print(f"[LLM] Skipping LLM reasoning step: {skip_reason}. Falling back to deterministic rule.")
    quant, reasoning = deterministic_recommendation(sweep_results, candidates_within_budget, budget_rss_mb, budget_file_size_mb)
    return quant, reasoning, "deterministic_fallback"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(sweep_results: list, budget_rss_mb, budget_file_size_mb):
    print("\n" + "=" * 90)
    print("SWEEP SUMMARY")
    print("=" * 90)
    header = f"{'Quant':<10}{'Size(MB)':>10}{'RSS(MB)':>10}{'TTFT(ms)':>10}{'TPS':>8}{'FitsSize':>10}{'FitsRSS':>9}{'Status':>10}"
    print(header)
    print("-" * len(header))
    for r in sweep_results:
        def fmt(v, spec=""):
            return format(v, spec) if v is not None else "N/A"

        # "N/A" (not "YES") when a budget axis wasn't requested at all, so
        # the table doesn't imply a check happened that never ran.
        fits_size_disp = "N/A" if budget_file_size_mb is None else ("YES" if r["fits_file_size_budget"] else "NO")
        fits_rss_disp = "N/A" if budget_rss_mb is None else ("YES" if r["fits_rss_budget"] else "NO")

        row = (
            f"{r['quant']:<10}"
            f"{fmt(r['file_size_mb'], '.0f'):>10}"
            f"{fmt(r['peak_RSS_mb'], '.0f'):>10}"
            f"{fmt(r['TTFT_ms'], '.0f'):>10}"
            f"{fmt(r['TPS'], '.1f'):>8}"
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
                 candidates_within_budget: list, quant_level, reasoning: str, reasoning_source: str):
    report = {
        "model": model,
        "budget_rss_mb": budget_rss_mb,
        "budget_file_size_mb": budget_file_size_mb,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sweep_results": sweep_results,
        "candidates_within_budget": candidates_within_budget,
        "recommendation": {
            "quant_level": quant_level,
            "reasoning": reasoning,
            "reasoning_source": reasoning_source,
        },
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OUTPUT] Report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Automated quantization selection agent for SmolChat/Android")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--budget-rss-mb", type=float, default=None, dest="budget_rss_mb",
                    help="Maximum acceptable Peak RSS in MB (optional; at least one of --budget-rss-mb / --budget-file-size-mb is required)")
    p.add_argument("--budget-file-size-mb", type=float, default=None, dest="budget_file_size_mb",
                    help="Maximum acceptable GGUF file size in MB (optional; at least one of --budget-rss-mb / --budget-file-size-mb is required)")
    p.add_argument("--quant-levels", default="Q4_K_M,Q5_K_M,Q8_0", help="Comma-separated quant levels to sweep")
    p.add_argument("--questions", default=None, help="Path to questions file (default: run_autobench.py's built-in defaults)")
    p.add_argument("--output", default="agent_report.json")
    p.add_argument("--verify-cold-load", action="store_true", dest="verify_cold_load",
                    help="Reboot the device before each quant level's benchmark run (via run_autobench.py --reboot-before) for a genuine cold-load read, not just a warm-cache best case")
    return p.parse_args()


def main():
    args = parse_args()

    if args.budget_rss_mb is None and args.budget_file_size_mb is None:
        print("[ERROR] At least one of --budget-rss-mb or --budget-file-size-mb is required.")
        sys.exit(1)

    quant_levels = [q.strip() for q in args.quant_levels.split(",") if q.strip()]
    if not quant_levels:
        print("[ERROR] --quant-levels resolved to an empty list")
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
    print("Agentic Quantization Selector")
    print(f"  Model: {args.model}  Budget: {budget_desc}  Levels: {', '.join(quant_levels)}")
    print(f"  Cold-load verification: {'ON (device reboots before each level)' if args.verify_cold_load else 'OFF (cold_load_ms is warm-cache best case)'}")
    print("=" * 60)

    output_stem = Path(args.output).with_suffix("").name
    sweep_results = sweep(args.model, quant_levels, questions_path, args.budget_rss_mb, args.budget_file_size_mb,
                           output_stem, verify_cold_load=args.verify_cold_load)

    candidates_within_budget = [r["quant"] for r in sweep_results if r["status"] == "success" and r["fits_budget"]]

    print("\n" + "=" * 60)
    if candidates_within_budget:
        print(f"[FILTER] Candidates satisfying {budget_desc}: {', '.join(candidates_within_budget)}")
    else:
        print(f"[FILTER] NONE of the tested quant levels satisfy {budget_desc}. Proceeding to reasoning phase anyway.")
    print("=" * 60)

    quant_level, reasoning, reasoning_source = get_recommendation(
        sweep_results, candidates_within_budget, args.budget_rss_mb, args.budget_file_size_mb, args.model
    )

    print_summary_table(sweep_results, args.budget_rss_mb, args.budget_file_size_mb)

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"Quant level: {quant_level or 'none'}")
    print(f"Source: {reasoning_source}")
    print(f"Reasoning:\n{reasoning}")

    save_report(args.output, args.model, args.budget_rss_mb, args.budget_file_size_mb, sweep_results,
                candidates_within_budget, quant_level, reasoning, reasoning_source)


if __name__ == "__main__":
    main()
