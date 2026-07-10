#!/usr/bin/env python3
"""
SmolChat Automated Benchmark — CLI wrapper.

Thin command-line front-end over the measurement engine in
hardware_eval/on_device_eval.py. This script owns only CLI concerns:
  - argument parsing and pre-flight device checks
  - resolving --model to a device-side GGUF (local push, or HF→GGUF conversion)
  - loading the question set
  - writing the JSON report and printing the summary table

All metric gathering (broadcast, logcat scrape, per-question stats, mA→W power
conversion) lives in on_device_eval so the pipeline and this CLI share one
implementation. See on_device_eval.measure_smolchat / benchmark_smolchat.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from hardware_eval.on_device_eval import (
    Adb,
    find_adb,
    DEFAULT_QUESTIONS,
    SMOLCHAT_PACKAGE,
    MAIN_ACTIVITY_COMPONENT,
    read_battery_voltage_v,
    benchmark_smolchat,
    summarize_smolchat,
)

CONVERT_SCRIPT = str(Path.home() / "Model-Conversion/convert_to_gguf.py")
CONVERSION_REPORT_CANDIDATES = [
    str(Path.home() / "Model-Conversion/conversion_report.json"),
    str(Path.cwd() / "conversion_report.json"),
]

COLD_BOOT_SETTLE_SECONDS = 25
COLD_LOAD_NOTE = (
    "cold_load_ms reflects a genuinely cold read only if --reboot-before was used for this run. "
    "Without a reboot, the OS file cache may make cold_load_ms appear faster than a true cold read."
)
BATTERY_STATUS_NAMES = {1: "Unknown", 2: "Charging", 3: "Discharging", 4: "Not charging", 5: "Full"}


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def check_device(adb: Adb, device_kind: str):
    if not adb.is_connected():
        print(f"[ERROR] No {device_kind} detected via adb ({adb.flag}).")
        diag = subprocess.run([adb.bin, "devices", "-l"], capture_output=True, text=True, timeout=10)
        print("adb devices -l output:")
        print(diag.stdout.strip() or "(empty)")
        sys.exit(1)
    print(f"[OK] {device_kind.capitalize()} connected via adb {adb.flag}")


def check_smolchat_installed(adb: Adb):
    if not adb.package_installed(SMOLCHAT_PACKAGE):
        print(f"[ERROR] SmolChat ({SMOLCHAT_PACKAGE}) is not installed on the target device.")
        sys.exit(1)
    print(f"[OK] SmolChat ({SMOLCHAT_PACKAGE}) is installed")


def check_battery(adb: Adb) -> dict:
    """Warn (never block) when battery state makes power (mA) readings untrustworthy."""
    import re
    result = adb.run(["shell", "dumpsys", "battery"], timeout=15)
    level_m = re.search(r"level:\s*(\d+)", result.stdout)
    status_m = re.search(r"status:\s*(\d+)", result.stdout)
    level = int(level_m.group(1)) if level_m else None
    status = int(status_m.group(1)) if status_m else None
    status_name = BATTERY_STATUS_NAMES.get(status, "Unknown")
    warning = status in (2, 5) or level == 100
    if warning:
        print(f"[WARN] Battery {level}% ({status_name}). Power (mA) readings are unreliable while "
              "full/charging. Unplug and discharge below ~95% for trustworthy power data.")
    else:
        print(f"[OK] Battery at {level}% ({status_name}) -- power readings should be trustworthy")
    return {"battery_warning": warning, "battery_level_pct": level, "battery_status": status_name}


def reboot_for_cold_load(adb: Adb):
    print("[COLD] Rebooting device for genuine cold-load measurement (~30-60s)...")
    adb.run(["reboot"], timeout=30)
    adb.run(["wait-for-device"], timeout=180)
    time.sleep(COLD_BOOT_SETTLE_SECONDS)


def reset_smolchat(adb: Adb):
    print("[RESET] Force-stopping and restarting SmolChat for a clean process (accurate peak RSS)...")
    adb.run(["shell", "am", "force-stop", SMOLCHAT_PACKAGE], timeout=15)
    adb.run(["shell", "am", "start", "-n", MAIN_ACTIVITY_COMPONENT], timeout=15)
    time.sleep(4)


# ---------------------------------------------------------------------------
# Model resolution / conversion / deploy
# ---------------------------------------------------------------------------

def convert_to_gguf(model_id: str, quant: str) -> str:
    print(f"\n[CONVERSION] Converting {model_id} to GGUF ({quant}) and deploying...")
    if not os.path.exists(CONVERT_SCRIPT):
        print(f"[ERROR] Conversion script not found: {CONVERT_SCRIPT}")
        sys.exit(1)
    result = subprocess.run([sys.executable, CONVERT_SCRIPT, "--model", model_id,
                             "--quant", quant, "--deploy"])
    if result.returncode != 0:
        print(f"[ERROR] Conversion failed for {model_id} (exit {result.returncode})")
        sys.exit(1)
    report_path = next((p for p in CONVERSION_REPORT_CANDIDATES if os.path.exists(p)), None)
    if not report_path:
        print(f"[ERROR] conversion_report.json not found in: {CONVERSION_REPORT_CANDIDATES}")
        sys.exit(1)
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[ERROR] Failed to read conversion report {report_path}: {e}")
        sys.exit(1)
    device_path = report.get("device_path") or report.get("deployed_path")
    if not device_path:
        filename = report.get("filename") or (
            os.path.basename(report["output_path"]) if report.get("output_path") else None)
        if not filename:
            print(f"[ERROR] Could not determine deployed filename from report: {report}")
            sys.exit(1)
        device_path = f"/sdcard/Download/{filename}"
    print(f"[OK] Conversion + deploy complete. Device path: {device_path}")
    return device_path


def push_local_gguf(adb: Adb, local_path: str) -> str:
    local_path = os.path.expanduser(local_path)
    if not os.path.exists(local_path):
        print(f"[ERROR] GGUF file not found: {local_path}")
        sys.exit(1)
    device_path = f"/sdcard/Download/{os.path.basename(local_path)}"
    print(f"\n[DEPLOY] Pushing {local_path} -> {device_path} ...")
    result = adb.run(["push", local_path, device_path], timeout=600)
    if result.returncode != 0:
        print(f"[ERROR] adb push failed: {result.stderr}")
        sys.exit(1)
    print(f"[OK] Push complete. Device path: {device_path}")
    return device_path


def resolve_model(model_arg: str, quant: str, adb: Adb) -> tuple:
    """Return (device_path, display_name)."""
    expanded = os.path.expanduser(model_arg)
    if model_arg.lower().endswith(".gguf") or os.path.isfile(expanded):
        return push_local_gguf(adb, model_arg), os.path.basename(expanded)
    if "/" in model_arg:
        return convert_to_gguf(model_arg, quant), model_arg
    print(f"[ERROR] --model '{model_arg}' is neither a local .gguf nor a HF model ID (no '/').")
    sys.exit(1)


def load_questions(questions_path) -> list:
    if not questions_path:
        print(f"[OK] Using {len(DEFAULT_QUESTIONS)} built-in default questions")
        return list(DEFAULT_QUESTIONS)
    path = os.path.expanduser(questions_path)
    if not os.path.exists(path):
        print(f"[ERROR] Questions file not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8", errors="replace") as f:
        questions = [line.strip() for line in f if line.strip()]
    if not questions:
        print(f"[ERROR] Questions file is empty: {path}")
        sys.exit(1)
    print(f"[OK] Loaded {len(questions)} questions from {path}")
    return questions


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(summary: dict, run_info: dict):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Metric':<12}{'Mean':>12}{'Std':>12}{'Min':>12}{'Max':>12}"
    print(header)
    print("-" * len(header))
    for label, key in [("TTFT_ms", "ttft_ms"), ("TPS", "tps"),
                       ("RSS_KB", "memory_kb"), ("Power_mA", "power_ma")]:
        s = summary[key]
        row = f"{label:<12}" + "".join(
            f"{(s[k] if s[k] is not None else 'N/A'):>12}" for k in ("mean", "std", "min", "max"))
        print(row)
    cold = summary.get("first_question_cold_load_ms")
    print(f"{'ColdLoad_Q1':<12}{(cold if cold is not None else 'N/A'):>12}")
    meas = summary.get("measured", {})
    print(f"\nGating view (measured): TTFT={meas.get('ttft_ms')}ms  TPS={meas.get('tok_per_s')}  "
          f"Power={meas.get('avg_watts')}W  PeakRSS={meas.get('peak_memory_mb')}MB")
    print(f"Thermal states observed: {', '.join(summary['thermal_states_observed']) or 'none'}")
    print(f"{run_info['cold_load_note']} (rebooted_before_run={run_info['rebooted_before_run']})")


def save_results(output_path: str, run_info: dict, summary: dict, results: list):
    report = {"run_info": run_info, "results": results, "summary": summary}
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OUTPUT] Results saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Automated SmolChat GGUF benchmark (CLI over on_device_eval)")
    p.add_argument("--model", required=True, help="HuggingFace model ID or path to a local .gguf file")
    p.add_argument("--device", choices=["phone", "emulator"], default="phone")
    p.add_argument("--serial", default="", help="Explicit adb device serial (overrides --device flag)")
    p.add_argument("--questions", default=None, help="Path to .txt file, one question per line")
    p.add_argument("--output", default="autobench_results.json")
    p.add_argument("--quant", choices=["Q4_K_M", "Q5_K_M", "Q8_0"], default="Q4_K_M")
    p.add_argument("--timeout", type=int, default=60, help="Seconds to wait per question")
    p.add_argument("--reboot-before", action="store_true",
                   help="Reboot the device before benchmarking for a genuine cold-load read")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("SmolChat Automated Benchmark Pipeline")
    print(f"  Model: {args.model}  Device: {args.serial or args.device}  "
          f"Quant: {args.quant}  Timeout: {args.timeout}s")
    print("=" * 60)

    adb_bin = find_adb()
    if not adb_bin:
        print("[ERROR] adb not found on PATH or at ~/Library/Android/sdk/platform-tools/adb")
        sys.exit(1)
    adb = Adb(adb_bin, args.device, args.serial)

    check_device(adb, args.device)
    check_smolchat_installed(adb)

    if args.reboot_before:
        reboot_for_cold_load(adb)

    battery_info = check_battery(adb)
    reset_smolchat(adb)

    model_path, model_name = resolve_model(args.model, args.quant, adb)
    questions = load_questions(args.questions)
    voltage = read_battery_voltage_v(adb)

    start_time = datetime.now(timezone.utc).isoformat()
    results, context_resets = benchmark_smolchat(adb, model_path, questions, args.timeout)
    end_time = datetime.now(timezone.utc).isoformat()

    completed = sum(1 for r in results if r["status"] == "success")
    summary = summarize_smolchat(results, voltage)

    run_info = {
        "model": model_name,
        "model_device_path": model_path,
        "device": args.serial or args.device,
        "quant": args.quant,
        "start_time": start_time,
        "end_time": end_time,
        "total": len(questions),
        "completed": completed,
        "failed": len(results) - completed,
        "context_resets": context_resets,
        "battery_warning": battery_info["battery_warning"],
        "battery_level_pct": battery_info["battery_level_pct"],
        "battery_status": battery_info["battery_status"],
        "rebooted_before_run": args.reboot_before,
        "cold_load_note": COLD_LOAD_NOTE,
    }

    save_results(args.output, run_info, summary, results)
    print_summary_table(summary, run_info)
    print("\n" + "=" * 60)
    print(f"DONE - {completed}/{len(questions)} completed, "
          f"{len(results) - completed} failed, {context_resets} context reset(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
