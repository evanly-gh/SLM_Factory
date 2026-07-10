#!/usr/bin/env python3
"""
SmolChat Automated Benchmark Pipeline.

Converts a model to GGUF (if needed), pushes it to a connected Android
device, fires headless inference runs via broadcast against SmolChat's
HeadlessBenchmarkReceiver, and collects 6 metrics per question with zero
human interaction after launch.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = "io.shubham0204.smollmandroid"
BROADCAST_ACTION = "com.smollmandroid.RUN_PROMPT"
RECEIVER_COMPONENT = f"{PACKAGE}/.headless.HeadlessBenchmarkReceiver"
MAIN_ACTIVITY_COMPONENT = f"{PACKAGE}/.MainActivity"

FALLBACK_ADB = str(Path.home() / "Library/Android/sdk/platform-tools/adb")

CONVERT_SCRIPT = str(Path.home() / "Model-Conversion/convert_to_gguf.py") 
CONVERSION_REPORT_CANDIDATES = [
    str(Path.home() / "Model-Conversion/conversion_report.json"),
    str(Path.cwd() / "conversion_report.json"),
]

LOGCAT_TAG_FILTER = [
    "COLD_LOAD:D", "TTFT:D", "TPS:D", "MEMORY:D",
    "POWER:D", "THERMAL:D", "RUN_DONE:D", "RUN_ERROR:D", "*:S",
]

CONTEXT_SIZE_PHRASE = "context size reached"

COLD_BOOT_SETTLE_SECONDS = 25

COLD_LOAD_NOTE = (
    "cold_load_ms reflects a genuinely cold read only if --reboot-before was used for this run. "
    "Without a reboot, the OS file cache may make cold_load_ms appear faster than a true cold read "
    "from storage, especially for larger model files."
)

DEFAULT_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "What is the largest planet in our solar system?",
    "What year did the first human land on the moon?",
    "What are the three states of matter?",
    "How does a refrigerator keep food cold?",
    "Explain the process of photosynthesis in plants.",
    "Summarize the theory of relativity in simple terms.",
    "Describe the causes and consequences of World War I in a few sentences.",
]


# ---------------------------------------------------------------------------
# ADB resolution / helpers
# ---------------------------------------------------------------------------

def find_adb() -> str:
    on_path = shutil.which("adb")
    if on_path:
        return on_path
    if os.path.exists(FALLBACK_ADB):
        return FALLBACK_ADB
    print("[ERROR] adb not found on PATH or at ~/Library/Android/sdk/platform-tools/adb")
    sys.exit(1)


class Adb:
    def __init__(self, adb_bin: str, device: str):
        self.bin = adb_bin
        self.flag = "-d" if device == "phone" else "-e"
        self.base = [adb_bin, self.flag]

    def run(self, args: list, timeout: int = 30) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(self.base + args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(self.base + args, returncode=1, stdout="", stderr="TIMEOUT")


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_device(adb: Adb, device_kind: str):
    result = adb.run(["get-state"], timeout=10)
    if result.returncode != 0 or result.stdout.strip() != "device":
        print(f"[ERROR] No {device_kind} detected via adb ({adb.flag}).")
        diag = subprocess.run([adb.bin, "devices", "-l"], capture_output=True, text=True, timeout=10)
        print("adb devices -l output:")
        print(diag.stdout.strip() or "(empty)")
        sys.exit(1)
    print(f"[OK] {device_kind.capitalize()} connected via adb {adb.flag}")


def check_smolchat_installed(adb: Adb):
    result = adb.run(["shell", "pm", "list", "packages"], timeout=15)
    if PACKAGE not in result.stdout:
        print(f"[ERROR] SmolChat ({PACKAGE}) is not installed on the target device.")
        sys.exit(1)
    print(f"[OK] SmolChat ({PACKAGE}) is installed")


def print_thermal_reminder():
    print(
        "[NOTE] Thermal state affects TTFT/TPS. For comparable results, ideally "
        "start with the phone rested (not hot from prior use). Continuing anyway."
    )


BATTERY_STATUS_NAMES = {1: "Unknown", 2: "Charging", 3: "Discharging", 4: "Not charging", 5: "Full"}


def check_battery(adb: Adb) -> dict:
    """Warn (but never block) when battery state makes Power (mA) readings untrustworthy.

    BatteryManager reports near-zero/garbage discharge current when the
    battery is full or actively charging, since there's no discharge
    current to measure - only TTFT/TPS/RSS/Thermal stay meaningful then.
    """
    result = adb.run(["shell", "dumpsys", "battery"], timeout=15)
    level_m = re.search(r"level:\s*(\d+)", result.stdout)
    status_m = re.search(r"status:\s*(\d+)", result.stdout)
    level = int(level_m.group(1)) if level_m else None
    status = int(status_m.group(1)) if status_m else None
    status_name = BATTERY_STATUS_NAMES.get(status, "Unknown")

    warning = status == 5 or level == 100 or status == 2
    if warning:
        print(
            f"[WARN] Battery is at {level}% and status is {status_name}. Power (mA) readings from "
            "BatteryManager are known to be unreliable or invalid when the battery is full/charging, "
            "since there is no discharge current to measure. For trustworthy power data, unplug the "
            "phone and let it discharge below ~95% before running this benchmark."
        )
    else:
        print(f"[OK] Battery at {level}% ({status_name}) -- Power readings should be trustworthy")

    return {"battery_warning": warning, "battery_level_pct": level, "battery_status": status_name}


# ---------------------------------------------------------------------------
# Model resolution / conversion / deploy
# ---------------------------------------------------------------------------

def convert_to_gguf(model_id: str, quant: str) -> str:
    print(f"\n[CONVERSION] Converting {model_id} to GGUF ({quant}) and deploying...")
    if not os.path.exists(CONVERT_SCRIPT):
        print(f"[ERROR] Conversion script not found: {CONVERT_SCRIPT}")
        sys.exit(1)

    cmd = [sys.executable, CONVERT_SCRIPT, "--model", model_id, "--quant", quant, "--deploy"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Conversion failed for {model_id} (exit code {result.returncode})")
        sys.exit(1)

    report_path = next((p for p in CONVERSION_REPORT_CANDIDATES if os.path.exists(p)), None)
    if not report_path:
        print(f"[ERROR] conversion_report.json not found in any of: {CONVERSION_REPORT_CANDIDATES}")
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
            os.path.basename(report["output_path"]) if report.get("output_path") else None
        )
        if not filename:
            print(f"[ERROR] Could not determine deployed filename from conversion report: {report}")
            sys.exit(1)
        device_path = f"/sdcard/Download/{filename}"

    print(f"[OK] Conversion + deploy complete. Device path: {device_path}")
    return device_path


def push_local_gguf(adb: Adb, local_path: str) -> str:
    local_path = os.path.expanduser(local_path)
    if not os.path.exists(local_path):
        print(f"[ERROR] GGUF file not found: {local_path}")
        sys.exit(1)

    filename = os.path.basename(local_path)
    device_path = f"/sdcard/Download/{filename}"
    print(f"\n[DEPLOY] Pushing {local_path} -> {device_path} ...")
    result = adb.run(["push", local_path, device_path], timeout=600)
    if result.returncode != 0:
        print(f"[ERROR] adb push failed: {result.stderr}")
        sys.exit(1)
    print(f"[OK] Push complete. Device path: {device_path}")
    return device_path


def resolve_model(model_arg: str, quant: str, adb: Adb) -> tuple:
    """Returns (device_path, display_model_name)."""
    expanded = os.path.expanduser(model_arg)
    is_local_gguf = model_arg.lower().endswith(".gguf") or os.path.isfile(expanded)

    if is_local_gguf:
        device_path = push_local_gguf(adb, model_arg)
        return device_path, os.path.basename(expanded)

    if "/" in model_arg:
        device_path = convert_to_gguf(model_arg, quant)
        return device_path, model_arg

    print(f"[ERROR] --model '{model_arg}' is neither a local .gguf path nor a HuggingFace model ID (no '/').")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

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
# Broadcast / logcat plumbing
# ---------------------------------------------------------------------------

def clear_logcat(adb: Adb):
    adb.run(["logcat", "-c"], timeout=15)


def fire_broadcast(adb: Adb, model_path: str, question: str, run_id: str):
    prompt_arg = question.replace(" ", "_")
    adb.run([
        "shell", "am", "broadcast",
        "-a", BROADCAST_ACTION,
        "-n", RECEIVER_COMPONENT,
        "--es", "model_path", model_path,
        "--es", "prompt", prompt_arg,
        "--es", "run_id", run_id,
    ], timeout=20)


KNOWN_TAGS = ["RUN_DONE", "RUN_ERROR", "COLD_LOAD", "TTFT", "TPS", "MEMORY", "POWER", "THERMAL"]


def parse_run_lines(logcat_text: str, run_id: str) -> dict:
    """Return {TAG: full_line} for every log line belonging to run_id.

    Matches the tag as a standalone token rather than assuming a fixed
    logcat format (brief/time/threadtime all differ in prefix layout).
    """
    id_pattern = re.compile(r"run_id=" + re.escape(run_id) + r"(?:\s|$)")
    lines = {}
    for line in logcat_text.splitlines():
        if not id_pattern.search(line):
            continue
        for tag in KNOWN_TAGS:
            if re.search(r"\b" + tag + r"\b", line):
                lines[tag] = line
                break
    return lines


def extract_value(line: str):
    m = re.search(r"value=(\S+)", line)
    return m.group(1) if m else None


def extract_response(line: str):
    m = re.search(r"response=(.*)$", line)
    return m.group(1).strip() if m else None


def extract_error(line: str):
    m = re.search(r"reason=(\S+)\s+message=(.*)$", line)
    if m:
        return m.group(1), m.group(2).strip()
    return None, line.strip()


def poll_for_result(adb: Adb, run_id: str, timeout: int) -> tuple:
    """Poll logcat every 1s until RUN_DONE/RUN_ERROR for run_id or timeout. Returns (status, lines)."""
    deadline = time.time() + timeout
    last_lines = {}
    while time.time() < deadline:
        result = adb.run(["logcat", "-d", "-s"] + LOGCAT_TAG_FILTER, timeout=20)
        last_lines = parse_run_lines(result.stdout, run_id)
        if "RUN_DONE" in last_lines:
            return "done", last_lines
        if "RUN_ERROR" in last_lines:
            return "error", last_lines
        time.sleep(1)
    return "timeout", last_lines


def restart_smolchat(adb: Adb):
    print("  [RESTART] force-stopping and relaunching SmolChat...")
    adb.run(["shell", "am", "force-stop", PACKAGE], timeout=15)
    adb.run(["shell", "am", "start", "-n", MAIN_ACTIVITY_COMPONENT], timeout=15)
    time.sleep(3)


def reboot_device_for_cold_load(adb: Adb):
    """Reboot the device so cold_load_ms reflects a genuine cold read from
    storage rather than one warmed by the OS file cache from a prior run.

    "device" state from wait-for-device precedes the home screen and system
    services actually being ready, so we sleep an extra settle period on top.
    """
    print("[COLD] Rebooting device for genuine cold-load measurement (this takes ~30-60s)...")
    adb.run(["reboot"], timeout=30)
    adb.run(["wait-for-device"], timeout=180)
    time.sleep(COLD_BOOT_SETTLE_SECONDS)


def reset_smolchat_for_clean_process(adb: Adb):
    """One-time reset at script startup so RSS isn't contaminated by a
    high-water mark left over from a model loaded in a prior, separate
    invocation of this script (see run_benchmark's mid-run context-size
    reset for the unrelated, per-question retry logic)."""
    print("[RESET] Force-stopping and restarting SmolChat for a clean process state (required for accurate Peak RSS)...")
    adb.run(["shell", "am", "force-stop", PACKAGE], timeout=15)
    adb.run(["shell", "am", "start", "-n", MAIN_ACTIVITY_COMPONENT], timeout=15)
    time.sleep(4)


def run_one(adb: Adb, model_path: str, question: str, n: int, timeout: int) -> dict:
    run_id = f"run_{n}_{int(time.time() * 1000)}"
    clear_logcat(adb)
    fire_broadcast(adb, model_path, question, run_id)
    status, lines = poll_for_result(adb, run_id, timeout)
    return {"run_id": run_id, "status": status, "lines": lines}


def build_metrics(lines: dict) -> dict:
    def num(tag, cast):
        v = extract_value(lines[tag]) if tag in lines else None
        if v is None:
            return None
        try:
            return cast(v)
        except ValueError:
            return None

    return {
        "cold_load_ms": num("COLD_LOAD", int),
        "ttft_ms": num("TTFT", int),
        "tps": num("TPS", float),
        "memory_kb": num("MEMORY", int),
        "power_ma": num("POWER", float),
        "thermal": extract_value(lines["THERMAL"]) if "THERMAL" in lines else None,
    }


# ---------------------------------------------------------------------------
# Main per-question loop
# ---------------------------------------------------------------------------

def run_benchmark(adb: Adb, model_path: str, questions: list, timeout: int) -> tuple:
    results = []
    context_resets = 0
    total = len(questions)

    for n, question in enumerate(questions, start=1):
        outcome = run_one(adb, model_path, question, n, timeout)
        status, lines, run_id = outcome["status"], outcome["lines"], outcome["run_id"]
        context_reset_for_this_q = False

        if status == "error":
            reason, message = extract_error(lines.get("RUN_ERROR", ""))
            if message and CONTEXT_SIZE_PHRASE in message.lower():
                print(f"\n[{n}/{total}] \"{question}\"")
                print(f"  CONTEXT WINDOW FULL (reason={reason}, message={message}) -- restarting SmolChat and retrying once")
                restart_smolchat(adb)
                context_resets += 1
                context_reset_for_this_q = True
                outcome = run_one(adb, model_path, question, n, timeout)
                status, lines, run_id = outcome["status"], outcome["lines"], outcome["run_id"]

        entry = {
            "question_number": n,
            "question": question,
            "run_id": run_id,
            "status": None,
            "metrics": None,
            "response": None,
            "error": None,
            "context_reset": context_reset_for_this_q,
        }

        if status == "done":
            metrics = build_metrics(lines)
            response = extract_response(lines["RUN_DONE"])
            entry["status"] = "success"
            entry["metrics"] = metrics
            entry["response"] = response

            print(f"\n[{n}/{total}] \"{question}\"")
            print(
                f"  ColdLoad={metrics['cold_load_ms']}ms TTFT={metrics['ttft_ms']}ms TPS={metrics['tps']} "
                f"RSS={metrics['memory_kb']}KB Power={metrics['power_ma']}mA Thermal={metrics['thermal']}"
            )
            preview = response if response and len(response) <= 160 else (response[:157] + "..." if response else "")
            print(f"  Response: \"{preview}\"")
        elif status == "error":
            reason, message = extract_error(lines.get("RUN_ERROR", ""))
            entry["status"] = "failed"
            entry["error"] = {"reason": reason, "message": message}
            print(f"\n[{n}/{total}] \"{question}\"")
            print(f"  FAILED: reason={reason} message={message}")
        else:  # timeout
            entry["status"] = "failed"
            entry["error"] = {"reason": "timeout", "message": f"No RUN_DONE/RUN_ERROR within {timeout}s"}
            print(f"\n[{n}/{total}] \"{question}\"")
            print(f"  FAILED: timeout after {timeout}s")

        results.append(entry)
        time.sleep(2)

    return results, context_resets


# ---------------------------------------------------------------------------
# Summary / output
# ---------------------------------------------------------------------------

def stat_block(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.pstdev(values), 3),
        "min": min(values),
        "max": max(values),
    }


def compute_summary(results: list) -> dict:
    def vals(key):
        return [
            r["metrics"][key] for r in results
            if r["status"] == "success" and r["metrics"] and r["metrics"].get(key) is not None
        ]

    thermal_states = sorted({
        r["metrics"]["thermal"] for r in results
        if r["status"] == "success" and r["metrics"] and r["metrics"].get("thermal")
    })

    # Every question reloads the model now (context isolation), so only
    # question 1's cold_load_ms is a genuine cold read - question 2+ reload
    # within an already-booted, already-warm-cached process. Deliberately
    # not averaged in with the rest.
    first_q = next((r for r in results if r["question_number"] == 1), None)
    first_cold_load_ms = None
    if first_q and first_q["status"] == "success" and first_q["metrics"]:
        first_cold_load_ms = first_q["metrics"].get("cold_load_ms")

    return {
        "ttft_ms": stat_block(vals("ttft_ms")),
        "tps": stat_block(vals("tps")),
        "memory_kb": stat_block(vals("memory_kb")),
        "power_ma": stat_block(vals("power_ma")),
        "first_question_cold_load_ms": first_cold_load_ms,
        "thermal_states_observed": thermal_states,
        "note": (
            "TPS/TTFT drift across the run reflects real device thermal state "
            "as it heats up under sustained inference, not measurement error."
        ),
        "rss_note": (
            "peak_RSS_KB is only directly comparable across different models if each was "
            "benchmarked in a freshly restarted app process - this script now guarantees "
            "that automatically as of this version."
        ),
    }


def print_summary_table(summary: dict, run_info: dict):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Metric':<10}{'Mean':>12}{'Std':>12}{'Min':>12}{'Max':>12}"
    print(header)
    print("-" * len(header))
    for label, key in [("TTFT_ms", "ttft_ms"), ("TPS", "tps"), ("RSS_KB", "memory_kb"), ("Power_mA", "power_ma")]:
        s = summary[key]
        row = f"{label:<10}" + "".join(
            f"{(s[k] if s[k] is not None else 'N/A'):>12}" for k in ("mean", "std", "min", "max")
        )
        print(row)
    cold_load_q1 = summary.get("first_question_cold_load_ms")
    print(f"{'ColdLoad_Q1':<10}{(cold_load_q1 if cold_load_q1 is not None else 'N/A'):>12}{'N/A':>12}{'N/A':>12}{'N/A':>12}")
    print(f"\nThermal states observed: {', '.join(summary['thermal_states_observed']) or 'none'}")
    print(summary["note"])
    print(summary["rss_note"])
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
    p = argparse.ArgumentParser(description="Automated SmolChat GGUF benchmark pipeline")
    p.add_argument("--model", required=True, help="HuggingFace model ID or path to local .gguf file")
    p.add_argument("--device", choices=["phone", "emulator"], default="phone")
    p.add_argument("--questions", default=None, help="Path to .txt file, one question per line")
    p.add_argument("--output", default="autobench_results.json")
    p.add_argument("--quant", choices=["Q4_K_M", "Q5_K_M", "Q8_0"], default="Q4_K_M")
    p.add_argument("--timeout", type=int, default=60, help="Seconds to wait for RUN_DONE/RUN_ERROR per question")
    p.add_argument("--reboot-before", action="store_true",
                    help="Reboot the device before benchmarking for a genuine cold-load read (adds ~30-60s)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SmolChat Automated Benchmark Pipeline")
    print(f"  Model: {args.model}  Device: {args.device}  Quant: {args.quant}  Timeout: {args.timeout}s")
    print("=" * 60)

    adb_bin = find_adb()
    adb = Adb(adb_bin, args.device)

    check_device(adb, args.device)
    check_smolchat_installed(adb)
    print_thermal_reminder()

    if args.reboot_before:
        reboot_device_for_cold_load(adb)

    battery_info = check_battery(adb)

    reset_smolchat_for_clean_process(adb)

    model_path, model_name = resolve_model(args.model, args.quant, adb)
    questions = load_questions(args.questions)

    start_time = datetime.now(timezone.utc).isoformat()
    results, context_resets = run_benchmark(adb, model_path, questions, args.timeout)
    end_time = datetime.now(timezone.utc).isoformat()

    completed = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - completed

    run_info = {
        "model": model_name,
        "model_device_path": model_path,
        "device": args.device,
        "quant": args.quant,
        "start_time": start_time,
        "end_time": end_time,
        "total": len(questions),
        "completed": completed,
        "failed": failed,
        "context_resets": context_resets,
        "battery_warning": battery_info["battery_warning"],
        "battery_level_pct": battery_info["battery_level_pct"],
        "battery_status": battery_info["battery_status"],
        "rebooted_before_run": args.reboot_before,
        "cold_load_note": COLD_LOAD_NOTE,
    }

    summary = compute_summary(results)
    save_results(args.output, run_info, summary, results)
    print_summary_table(summary, run_info)

    print("\n" + "=" * 60)
    print(f"DONE - {completed}/{len(questions)} completed, {failed} failed, {context_resets} context reset(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
