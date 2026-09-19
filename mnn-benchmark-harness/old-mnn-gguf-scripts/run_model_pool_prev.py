#!/usr/bin/env python3
"""
run_model_pool.py — Orchestrates converting, pushing, and benchmarking an
entire pool of models on MNN.

Extends the same proven download -> convert -> push -> benchmark pattern
already used by agent_mnn_quantize.py, but where that script sweeps ONE model
across several quant_bit levels, this script loops over MULTIPLE DIFFERENT
models, each already assigned a SPECIFIC quant level (read from a
model_fit_report.json produced by check_model_fit.py's "fits" list).

Reuses agent_mnn_quantize.py's download/convert/push helpers (and its exact
caching/reuse behavior for already-converted models) and run_mnn_autobench.py's
find_adb() directly, by importing both scripts by path - the same pattern
agent_mnn_quantize.py itself uses to reuse run_mnn_autobench.py's find_adb().
Nothing about the underlying conversion/push/benchmark mechanics is
reimplemented here.

Every variant is processed independently: a failure at any step (download,
convert, push, benchmark) is logged with its specific error and marked
FAILED, but the run continues to the next variant rather than aborting the
whole pool - real testing has shown a model can pass the size-fit check
(check_model_fit.py) and still fail from actual memory pressure once loaded,
so per-variant isolation is essential.

Usage:
    python3 run_model_pool.py --fit-report model_fit_report_v2.json --output model_pool_results.json
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_QUANTIZE_SCRIPT = SCRIPT_DIR / "agent_mnn_quantize.py"
RUN_AUTOBENCH_SCRIPT = SCRIPT_DIR / "run_mnn_autobench.py"
CHECK_MODEL_FIT_SCRIPT = SCRIPT_DIR.parent / "check_model_fit.py"

# How often to print a progress line during the inter-variant rest, so a long
# --rest-seconds wait doesn't make the terminal look frozen.
REST_PROGRESS_INTERVAL_SECONDS = 60

# quant_block isn't swept here (unlike agent_mnn_quantize.py) since each
# variant already has one fixed quant level - this just matches that
# script's own default.
QUANT_BLOCK = 64

# MNN only supports these bit-widths, driven directly by --quant_bit.
QUANT_BIT_MAP = {"Q4_K_M": 4, "Q8_0": 8}
UNSUPPORTED_QUANT_REASON = "bf16 not supported by MNN export"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Quant mapping
# ---------------------------------------------------------------------------

def resolve_quant_bit(quant):
    """Map a pool entry's quant string to an MNN --quant_bit value.

    Returns (quant_bit, skip_reason) - skip_reason is None on success. "null"
    (JSON None) and "bf16" are both explicitly unsupported by MNN export;
    anything else not in QUANT_BIT_MAP is also skipped, with a distinct
    reason, rather than guessed at.
    """
    if quant is None:
        return None, UNSUPPORTED_QUANT_REASON
    quant_key = str(quant).strip().upper()
    if quant_key in ("BF16", "NULL", ""):
        return None, UNSUPPORTED_QUANT_REASON
    if quant_key in QUANT_BIT_MAP:
        return QUANT_BIT_MAP[quant_key], None
    return None, f"unrecognized quant value {quant!r}; no MNN --quant_bit mapping defined"


# ---------------------------------------------------------------------------
# Benchmark (parses run_mnn_autobench.py's actual summary structure)
# ---------------------------------------------------------------------------

def benchmark_one(device_model_path: str, questions_path, timeout: int, bench_output_path: Path,
                   no_think: bool = False, max_tokens: int = None) -> dict:
    """Run run_mnn_autobench.py against a pushed device folder and parse both
    its summary (into exactly the metrics this pool report needs) and its
    full per-question "results" array.

    Returns {"ok": True, "metrics": {...}, "per_question_metrics": [...]} or
    {"ok": False, "error": str}. run_mnn_autobench.py's summary dict has one
    stat_block (mean/std/min/max/n_completed) per SUMMARY_METRICS key - only
    "mean" is needed for "metrics" here, but "results" (each question's own
    text, response, and per-question metrics) is kept in full rather than
    discarded, since it's otherwise unrecoverable once bench_output_path is
    cleaned up.
    """
    cmd = [
        sys.executable, str(RUN_AUTOBENCH_SCRIPT),
        "--model-path", device_model_path,
        "--output", str(bench_output_path),
        "--timeout", str(timeout),
    ]
    if questions_path:
        cmd += ["--questions", str(questions_path)]
    if no_think:
        cmd += ["--no-think"]
    if max_tokens is not None:
        cmd += ["--max-tokens", str(max_tokens)]

    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return {"ok": False, "error": f"run_mnn_autobench.py exited with code {result.returncode}"}

    if not bench_output_path.exists():
        return {"ok": False, "error": f"results file not written: {bench_output_path}"}

    try:
        with open(bench_output_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "error": f"failed to read {bench_output_path}: {exc}"}

    run_info = data.get("run_info", {})
    completed = run_info.get("completed", 0)
    total = run_info.get("total", 0)
    if completed == 0:
        return {"ok": False, "error": f"0/{total} questions completed successfully"}

    summary = data.get("summary", {})

    def mean_of(key):
        return summary.get(key, {}).get("mean")

    metrics = {
        "peak_rss_kb": mean_of("peak_rss_kb"),
        "ttft_ms": mean_of("ttft_ms"),
        "prefill_tps": mean_of("prefill_tps"),
        "decode_tps": mean_of("decode_tps"),
        "power_ma_monsoon": mean_of("power_ma_monsoon"),
        "power_ma": mean_of("power_ma"),
        "thermal_cpu_c": mean_of("thermal_cpu_c"),
        "thermal_skin_c": mean_of("thermal_skin_c"),
        "N_completed": completed,
    }
    return {"ok": True, "metrics": metrics, "per_question_metrics": data.get("results", [])}


# ---------------------------------------------------------------------------
# Pre-flight RAM safety guard (re-checked per variant, right before use)
# ---------------------------------------------------------------------------

def check_current_fit(check_fit, variant: dict, label: str):
    """Re-compare this variant's trusted peak_memory_mb against the phone's
    CURRENT free RAM, queried fresh right before the variant is attempted -
    check_model_fit.py's own fit determination may be stale by the time this
    variant's turn comes up, since free RAM drifts as earlier variants in
    the pool load/unload models. Never raises: a query failure is reported
    as a warning and treated as "unknown", not a fatal error - the variant
    is still attempted either way, per the caller's contract.

    Returns (current_free_ram_mb or None, fits_now or None).
    """
    try:
        current_free_ram_mb = check_fit.get_phone_free_ram_mb()
    except RuntimeError as exc:
        print(f"{label} - [WARN] could not query current free RAM for the pre-flight safety check: {exc}. Proceeding anyway.")
        return None, None

    peak_memory_mb = variant.get("peak_memory_mb")
    if peak_memory_mb is None:
        return current_free_ram_mb, None

    fits_now = peak_memory_mb <= current_free_ram_mb
    if not fits_now:
        print(
            f"{label} - [WARN] peak_memory_mb={peak_memory_mb:.0f}MB now EXCEEDS current free RAM "
            f"({current_free_ram_mb:.0f}MB) - free RAM may have dropped since check_model_fit.py ran. "
            f"Attempting anyway."
        )
    return current_free_ram_mb, fits_now


# ---------------------------------------------------------------------------
# Per-variant processing
# ---------------------------------------------------------------------------

def process_variant(agent, check_fit, adb_bin: str, variant: dict, index: int, total: int,
                     questions_path, timeout: int, no_think: bool = False, max_tokens: int = None) -> dict:
    model_id = variant.get("model_id")
    quant = variant.get("quant")
    label = f"[{index}/{total}] {model_id} [{quant}]"

    entry = dict(variant)  # keep every original field from the pool (includes check_model_fit.py's own "fits")
    entry["status"] = None
    entry["error"] = None
    entry["metrics"] = None
    entry["per_question_metrics"] = None
    entry["free_ram_mb_at_start"] = None
    entry["fits_at_start"] = None
    entry["fits_based_on_measured_rss"] = None

    quant_bit, skip_reason = resolve_quant_bit(quant)
    if skip_reason is not None:
        print(f"{label} - SKIPPED: {skip_reason}")
        entry["status"] = "skipped"
        entry["error"] = skip_reason
        return entry

    # Re-check against CURRENT free RAM right before this variant is
    # attempted (not just check_model_fit.py's earlier, now possibly stale,
    # determination) - a warning only, never blocks the attempt.
    free_ram_mb_at_start, fits_at_start = check_current_fit(check_fit, variant, label)
    entry["free_ram_mb_at_start"] = free_ram_mb_at_start
    entry["fits_at_start"] = fits_at_start

    try:
        slug = agent.quant_slug(model_id, quant_bit)
        output_dir = agent.LOCAL_MODELS_DIR / slug

        if agent.is_conversion_complete(output_dir):
            print(f"{label} - reusing existing conversion at {output_dir}")
        else:
            interpreter = agent.find_llmexport_interpreter()
            if interpreter is None:
                raise RuntimeError(f"llmexport.py venv not found at {agent.LLMEXPORT_VENV_PYTHON}")

            print(f"{label} - downloading (if needed)...")
            dl = agent.resolve_local_model_dir(model_id)
            if not dl["ok"]:
                raise RuntimeError(f"download failed: {dl['error']}")

            print(f"{label} - converting...")
            conv = agent.convert_one(interpreter, dl["local_path"], quant_bit, QUANT_BLOCK, output_dir)
            if not conv["ok"]:
                raise RuntimeError(f"conversion failed: {conv['error']}")

        print(f"{label} - pushing to device...")
        push = agent.push_to_device(adb_bin, output_dir, slug)
        if not push["ok"]:
            raise RuntimeError(f"push failed: {push['error']}")

        print(f"{label} - benchmarking...")
        bench_output_path = agent.LOCAL_MODELS_DIR / f"{slug}_bench.json"
        bench = benchmark_one(push["device_path"], questions_path, timeout, bench_output_path,
                               no_think=no_think, max_tokens=max_tokens)
        if not bench["ok"]:
            raise RuntimeError(f"benchmark failed: {bench['error']}")

    except Exception as exc:
        # Any failure at any step lands here - this variant is marked FAILED
        # and the pool moves on to the next one rather than aborting.
        print(f"{label} - FAILED: {exc}")
        entry["status"] = "failed"
        entry["error"] = str(exc)
        return entry

    metrics = bench["metrics"]
    entry["status"] = "success"
    entry["error"] = None
    entry["metrics"] = metrics
    entry["per_question_metrics"] = bench["per_question_metrics"]

    # The REAL measured peak_rss_kb from this variant's own benchmark run is
    # the final, authoritative fit determination for the output data - both
    # check_model_fit.py's original "fits" and this variant's "fits_at_start"
    # are pre-flight estimates based on a trusted-but-unmeasured
    # peak_memory_mb figure, taken before the model was ever actually loaded.
    peak_rss_kb = metrics.get("peak_rss_kb")
    measured_rss_mb = peak_rss_kb / 1024 if peak_rss_kb is not None else None
    entry["fits_based_on_measured_rss"] = (
        measured_rss_mb <= entry["free_ram_mb_at_start"]
        if measured_rss_mb is not None and entry["free_ram_mb_at_start"] is not None
        else None
    )

    rss_gb = metrics["peak_rss_kb"] / (1024 * 1024) if metrics["peak_rss_kb"] is not None else None
    rss_disp = f"{rss_gb:.1f}GB" if rss_gb is not None else "N/A"
    ttft_disp = f"{metrics['ttft_ms']:.0f}ms" if metrics["ttft_ms"] is not None else "N/A"
    decode_disp = f"{metrics['decode_tps']:.1f}" if metrics["decode_tps"] is not None else "N/A"
    if entry["fits_based_on_measured_rss"] is None:
        fits_disp = "UNKNOWN"
    else:
        fits_disp = "YES" if entry["fits_based_on_measured_rss"] else "NO"
    print(f"{label} - DONE: RSS={rss_disp} TTFT={ttft_disp} DecodeTPS={decode_disp} FitsMeasuredRSS={fits_disp}")

    return entry


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def load_fit_report(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data.get("fits", [])


def print_summary_table(results: list):
    print("\n" + "=" * 110)
    print("MODEL POOL SUMMARY")
    print("=" * 110)
    header = (
        f"{'Model':<30}{'Quant':<10}{'Status':<10}{'RSS(MB)':>10}{'TTFT(ms)':>10}"
        f"{'PrefillTPS':>12}{'DecodeTPS':>11}{'PowerMonsoon(mA)':>18}{'ThermCPU(C)':>13}"
    )
    print(header)
    print("-" * len(header))

    def fmt(v, spec=""):
        return format(v, spec) if isinstance(v, (int, float)) else "N/A"

    for r in results:
        m = r.get("metrics") or {}
        rss_mb = m.get("peak_rss_kb") / 1024 if m.get("peak_rss_kb") is not None else None
        row = (
            f"{str(r.get('model_id')):<30}"
            f"{str(r.get('quant')):<10}"
            f"{r['status']:<10}"
            f"{fmt(rss_mb, '.0f'):>10}"
            f"{fmt(m.get('ttft_ms'), '.0f'):>10}"
            f"{fmt(m.get('prefill_tps'), '.1f'):>12}"
            f"{fmt(m.get('decode_tps'), '.1f'):>11}"
            f"{fmt(m.get('power_ma_monsoon'), '.2f'):>18}"
            f"{fmt(m.get('thermal_cpu_c'), '.1f'):>13}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Inter-variant rest
# ---------------------------------------------------------------------------

def rest_between_variants(rest_seconds: int):
    """Sleep for rest_seconds between two variants, printing a progress line
    every REST_PROGRESS_INTERVAL_SECONDS so a long wait doesn't look frozen."""
    if rest_seconds <= 0:
        return
    elapsed = 0
    print(f"\n[REST] Waiting {rest_seconds}s before next variant for thermal recovery... (elapsed: {elapsed}s/{rest_seconds}s)")
    while elapsed < rest_seconds:
        step = min(REST_PROGRESS_INTERVAL_SECONDS, rest_seconds - elapsed)
        time.sleep(step)
        elapsed += step
        print(f"[REST] Waiting {rest_seconds}s before next variant for thermal recovery... (elapsed: {elapsed}s/{rest_seconds}s)")


def print_post_rest_thermal_note():
    """Automating a real thermal reading here would require firing a full
    throwaway broadcast against some model already on the device (loading a
    model, running inference, and reading THERMAL_TEMP_CPU_C back from
    logcat) - effectively a second cold-load cycle per variant just for a
    sanity-check number. That's real added runtime on top of --rest-seconds,
    so instead we point at a cheap manual check here; the upcoming variant's
    own benchmark will report the genuine on-device THERMAL_TEMP_CPU_C once
    it starts anyway.
    """
    print(
        "[THERMAL CHECK] Rest complete. To confirm the phone actually cooled down before this variant "
        "starts, you can check manually, e.g.: adb shell dumpsys thermalservice | grep -A2 -i cpu"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Convert, push, and benchmark an entire model pool on MNN.")
    p.add_argument(
        "--fit-report", default="model_fit_report_v2.json",
        help="Path to a model_fit_report JSON (from check_model_fit.py); its 'fits' list is the pool to process.",
    )
    p.add_argument(
        "--questions", default=None,
        help="Path to inference question set (default: run_mnn_autobench.py's own built-in defaults).",
    )
    p.add_argument(
        "--timeout", type=int, default=180,
        help="Seconds passed through to run_mnn_autobench.py's --timeout per question (default: 180).",
    )
    p.add_argument(
        "--rest-seconds", type=int, default=None,
        help="Seconds to rest between variants for thermal recovery. Opt-in only: if this flag is "
             "omitted, NO rest happens between variants at all. No rest happens before the first "
             "variant or after the last one even when this is set, and a variant that was SKIPPED "
             "(e.g. unsupported bf16 quant) never triggers a rest, since nothing actually ran on the "
             "phone.",
    )
    p.add_argument(
        "--no-think", action="store_true", dest="no_think",
        help="Passed through as --no-think on every variant's run_mnn_autobench.py call (appends "
             "' /no_think' to every question's prompt - see run_mnn_autobench.py for details). "
             "Default: off (unchanged behavior).",
    )
    p.add_argument(
        "--max-tokens", type=int, default=None, dest="max_tokens",
        help="Passed through as --max-tokens on every variant's run_mnn_autobench.py call. Default: "
             "omitted, so run_mnn_autobench.py uses its own default.",
    )
    p.add_argument("--output", default="model_pool_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    if not AGENT_QUANTIZE_SCRIPT.exists():
        print(f"[ERROR] agent_mnn_quantize.py not found at {AGENT_QUANTIZE_SCRIPT}")
        sys.exit(1)
    if not RUN_AUTOBENCH_SCRIPT.exists():
        print(f"[ERROR] run_mnn_autobench.py not found at {RUN_AUTOBENCH_SCRIPT}")
        sys.exit(1)
    if not CHECK_MODEL_FIT_SCRIPT.exists():
        print(f"[ERROR] check_model_fit.py not found at {CHECK_MODEL_FIT_SCRIPT}")
        sys.exit(1)

    agent = _load_module(AGENT_QUANTIZE_SCRIPT, "_agent_mnn_quantize")
    run_autobench = _load_module(RUN_AUTOBENCH_SCRIPT, "_run_mnn_autobench")
    check_fit = _load_module(CHECK_MODEL_FIT_SCRIPT, "_check_model_fit")

    adb_bin = run_autobench.find_adb()  # exits(1) with a clear message if adb isn't found at all

    try:
        variants = load_fit_report(args.fit_report)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to load fit report '{args.fit_report}': {exc}")
        sys.exit(1)

    if not variants:
        print(f"[ERROR] No variants found in '{args.fit_report}' (expected a non-empty 'fits' list).")
        sys.exit(1)

    questions_path = None
    if args.questions:
        questions_path = os.path.expanduser(args.questions)
        if not os.path.exists(questions_path):
            print(f"[ERROR] Questions file not found: {questions_path}")
            sys.exit(1)

    total = len(variants)
    rest_desc = f"{args.rest_seconds}s between variants" if args.rest_seconds is not None else "OFF (pass --rest-seconds to enable)"
    print("=" * 60)
    print("MNN Model Pool Runner")
    max_tokens_desc = str(args.max_tokens) if args.max_tokens is not None else "default"
    print(f"  Pool: {args.fit_report} ({total} variants)  Timeout: {args.timeout}s  Rest: {rest_desc}  "
          f"No-think: {'ON' if args.no_think else 'OFF'}  MaxTokens: {max_tokens_desc}")
    print("=" * 60)

    results = []
    for i, variant in enumerate(variants, start=1):
        entry = process_variant(agent, check_fit, adb_bin, variant, i, total, questions_path, args.timeout,
                                 no_think=args.no_think, max_tokens=args.max_tokens)
        results.append(entry)

        is_last_variant = i == total
        # A skip means nothing actually ran on the phone (e.g. unsupported
        # bf16 quant), so there's no thermal load to recover from. Resting is
        # opt-in: args.rest_seconds is None unless --rest-seconds was passed
        # explicitly, so omitting the flag means no sleep at all.
        if entry["status"] != "skipped" and not is_last_variant and args.rest_seconds is not None:
            rest_between_variants(args.rest_seconds)
            print_post_rest_thermal_note()

    completed = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")

    report = {
        "run_info": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_variants": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OUTPUT] Results saved: {args.output}")

    print_summary_table(results)

    print("\n" + "=" * 60)
    print(f"DONE - {completed} succeeded, {failed} failed, {skipped} skipped (of {total})")
    print("=" * 60)


if __name__ == "__main__":
    main()
