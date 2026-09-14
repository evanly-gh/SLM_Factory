#!/usr/bin/env python3
"""
MNN Chat Automated Benchmark Pipeline.

Fires headless inference runs via broadcast against MNN Chat's
BenchmarkHeadlessReceiver and collects per-question metrics with zero
human interaction after launch. Mirrors the structure of SmolChat's
run_autobench.py, adapted to MNN Chat's confirmed-working broadcast
interface (single-folder model_path, separate prefill/decode metrics).

This script does NOT handle model conversion or deployment - it assumes
the MNN model folder (config.json/llm.mnn/llm.mnn.weight/etc.) is already
pushed to the device at the path given via --model-path.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = "com.alibaba.mnnllm.android"
BROADCAST_ACTION = "com.mnnllmchat.RUN_PROMPT"
RECEIVER_COMPONENT = f"{PACKAGE}/.benchmark.headless.BenchmarkHeadlessReceiver"

FALLBACK_ADB = str(Path.home() / "Library/Android/sdk/platform-tools/adb")
MONSOON_SCRIPT = Path.home() / "SLM_Factory_Krishna_Personal/Power-Monitor/monsoon_single_reading.py"

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

TIMEOUT_NOTE = (
    "Real testing has shown some models/questions (especially reasoning-mode "
    "responses) need 120-180s to complete, well above the 60s default. If you "
    "see a wave of 'timeout' failures, rerun with a larger --timeout before "
    "concluding the model itself is broken."
)

# Numeric/string metric tags emitted per-line as TAG=value, plus the two
# status tags (RUN_DONE/RUN_ERROR) whose payload follows a "key=value"
# pair(s) rather than being the tag's own value.
NUMERIC_TAGS = [
    "COLD_LOAD_MS", "TTFT_MS", "PREFILL_TIME_US", "DECODE_TIME_US",
    "PROMPT_LEN", "DECODE_LEN", "PEAK_RSS_KB", "POWER_MA",
    "THERMAL_TEMP_CPU_C", "THERMAL_TEMP_SKIN_C",
    "ENERGY_MAS_SAMPLED", "ENERGY_MJ_SAMPLED",
]
STRING_TAGS = ["THERMAL_STATUS"]
STATUS_TAGS = ["RUN_DONE", "RUN_ERROR"]
ALL_TAGS = NUMERIC_TAGS + STRING_TAGS + STATUS_TAGS


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
    def __init__(self, adb_bin: str):
        self.bin = adb_bin
        self.base = [adb_bin]

    def run(self, args: list, timeout: int = 30) -> subprocess.CompletedProcess:
        try:
            # errors="replace": logcat buffers can contain invalid UTF-8
            # (more likely once poll_for_result dumps the full unfiltered
            # buffer), and strict decoding would crash the whole process on
            # the first bad byte.
            return subprocess.run(
                self.base + args, capture_output=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(self.base + args, returncode=1, stdout="", stderr="TIMEOUT")


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_device(adb: Adb):
    result = adb.run(["get-state"], timeout=10)
    if result.returncode != 0 or result.stdout.strip() != "device":
        print("[ERROR] No device detected via adb.")
        diag = subprocess.run([adb.bin, "devices", "-l"], capture_output=True, text=True, timeout=10)
        print("adb devices -l output:")
        print(diag.stdout.strip() or "(empty)")
        sys.exit(1)
    print("[OK] Device connected via adb")


def check_mnnchat_installed(adb: Adb):
    result = adb.run(["shell", "pm", "list", "packages"], timeout=15)
    if PACKAGE not in result.stdout:
        print(f"[ERROR] MNN Chat ({PACKAGE}) is not installed on the target device.")
        sys.exit(1)
    print(f"[OK] MNN Chat ({PACKAGE}) is installed")


def print_thermal_reminder():
    print(
        "[NOTE] Thermal state affects TTFT/PrefillTPS/DecodeTPS. For comparable results, "
        "ideally start with the phone rested (not hot from prior use). Continuing anyway."
    )


BATTERY_STATUS_NAMES = {1: "Unknown", 2: "Charging", 3: "Discharging", 4: "Not charging", 5: "Full"}


def check_battery(adb: Adb) -> dict:
    """Warn (but never block) when battery state makes Power (mA) readings untrustworthy.

    Only status 3 (Discharging) gives a genuine discharge current to
    measure. Status 2 (Charging), 4 (Not charging), and 5 (Full) all break
    Power readings - the other metrics (TTFT/PrefillTPS/DecodeTPS/RSS/
    Thermal) remain meaningful regardless of battery state.
    """
    result = adb.run(["shell", "dumpsys", "battery"], timeout=15)
    level_m = re.search(r"level:\s*(\d+)", result.stdout)
    status_m = re.search(r"status:\s*(\d+)", result.stdout)
    level = int(level_m.group(1)) if level_m else None
    status = int(status_m.group(1)) if status_m else None
    status_name = BATTERY_STATUS_NAMES.get(status, "Unknown")

    warning = status in (2, 4, 5)
    if warning:
        print(
            f"[WARN] Battery status is {status_name} ({status}) at {level}%. Power (mA) readings are "
            "known to be unreliable or invalid unless status is 3 (Discharging), since there is no "
            "real discharge current to measure otherwise. For trustworthy power data, unplug the "
            "phone and let it discharge."
        )
    elif status == 3:
        print(f"[OK] Battery at {level}% (Discharging) -- Power readings should be trustworthy")
    else:
        print(f"[WARN] Battery status could not be confidently determined (status={status}, level={level}%). Power readings may be unreliable.")
        warning = True

    return {"battery_warning": warning, "battery_level_pct": level, "battery_status": status_name}


# ---------------------------------------------------------------------------
# Process reset
# ---------------------------------------------------------------------------

def reset_mnnchat_for_clean_process(adb: Adb):
    """One-time reset at script startup so RSS isn't contaminated by a
    high-water mark left over from a model loaded in a prior, separate
    invocation of this script."""
    print("[RESET] Force-stopping and relaunching MNN Chat for a clean process state (required for accurate Peak RSS)...")
    adb.run(["shell", "am", "force-stop", PACKAGE], timeout=15)
    # Launch via the LAUNCHER category rather than a hardcoded activity name,
    # since only the broadcast receiver's component is confirmed - not the
    # app's main activity.
    adb.run(["shell", "monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1"], timeout=15)
    time.sleep(4)


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


def fire_broadcast(adb: Adb, model_path: str, question: str, run_id: str, max_tokens: int,
                    backend_type: str = None):
    # "adb shell <args...>" re-joins its args into ONE remote command string;
    # an unquoted space inside an extra's value gets split by the on-device
    # shell into extra argv tokens. Building the full command as a single,
    # already shell-quoted string (rather than passing question/model_path
    # as separate list items) sidesteps that regardless of how adb re-joins.
    # max_tokens is sent via --ei (integer extra), matching its int type on
    # the Kotlin side, unlike the string extras above.
    cmd = (
        f"am broadcast -a {BROADCAST_ACTION} -n {RECEIVER_COMPONENT} "
        f"--es model_path {shlex.quote(model_path)} "
        f"--es prompt {shlex.quote(question)} "
        f"--es run_id {shlex.quote(run_id)} "
        f"--ei max_tokens {int(max_tokens)}"
    )
    # Omitted entirely (not sent as an empty/default-valued extra) when
    # backend_type is None, matching BenchmarkHeadlessReceiver.kt's own
    # hasExtra()-gated top_k/top_p/min_p philosophy: intent.getStringExtra()
    # returns null when the extra is absent, which HeadlessBenchmarkRunner
    # logs as "BACKEND_TYPE_OVERRIDE=default" (i.e. leave the model's own
    # shipped config.json backend_type completely untouched). Sending
    # `--es backend_type cpu` explicitly on every call - even though cpu is
    # already every model's shipped default - would still route through the
    # override code path instead of the "no override" path, which is not
    # the same as "existing calls without --backend-type behave exactly as
    # today". Only add the extra when the caller actually asked for one.
    if backend_type is not None:
        cmd += f" --es backend_type {shlex.quote(backend_type)}"
    adb.run(["shell", cmd], timeout=20)


def get_run_id_lines(logcat_text: str, run_id: str) -> list:
    return [line for line in logcat_text.splitlines() if run_id in line]


def tag_of(line: str, run_id: str):
    m = re.search(r"run_id=" + re.escape(run_id) + r"\s+(\w+)", line)
    return m.group(1) if m else None


def parse_run(logcat_text: str, run_id: str) -> tuple:
    """Returns (tag_lines, run_lines): tag_lines maps each recognized TAG to
    its first matching line for run_id; run_lines is every line containing
    run_id, in original order (needed to reconstruct a multi-line response).
    """
    run_lines = get_run_id_lines(logcat_text, run_id)
    tag_lines = {}
    for line in run_lines:
        tag = tag_of(line, run_id)
        if tag and tag in ALL_TAGS and tag not in tag_lines:
            tag_lines[tag] = line
    return tag_lines, run_lines


def poll_for_result(adb: Adb, run_id: str, timeout: int) -> tuple:
    """Poll logcat every 1s until RUN_DONE/RUN_ERROR for run_id or timeout.
    Returns (status, tag_lines, run_lines, raw_log).

    Deliberately unfiltered ("adb logcat -d -b main" with no -s tag list) -
    filtering happens Python-side in parse_run(), which searches raw text
    for the run_id substring directly. "-b main" scopes to the buffer where
    Android app Log.* calls land, avoiding the kernel/radio/perf noise in
    "-b all".

    raw_log is the exact, unfiltered buffer that produced this verdict -
    callers that need a line with no run_id at all (e.g.
    MNN_LLM_ACTUAL_BACKEND, a native-layer log with no run_id, logged once
    per model load) can search it directly instead of re-polling logcat
    separately, which would risk the ring buffer having rotated that line
    out under heavy per-token logging by the time of a second, later read.
    """
    deadline = time.time() + timeout
    last_tag_lines, last_run_lines, last_raw = {}, [], ""
    while time.time() < deadline:
        result = adb.run(["logcat", "-d", "-b", "main"], timeout=30)
        raw = result.stdout or ""
        tag_lines, run_lines = parse_run(raw, run_id)
        last_tag_lines, last_run_lines, last_raw = tag_lines, run_lines, raw
        if "RUN_DONE" in tag_lines:
            return "done", tag_lines, run_lines, raw
        if "RUN_ERROR" in tag_lines:
            return "error", tag_lines, run_lines, raw
        time.sleep(1)
    return "timeout", last_tag_lines, last_run_lines, last_raw


def extract_response(run_lines: list, run_id: str):
    """Reassemble RUN_DONE's response=<text>, which may span multiple
    consecutive log lines (same run_id prefix, no tag) for long responses
    containing newlines."""
    done_idx = None
    initial = None
    for i, line in enumerate(run_lines):
        if tag_of(line, run_id) == "RUN_DONE":
            m = re.search(r"RUN_DONE\s+response=(.*)$", line)
            initial = m.group(1) if m else ""
            done_idx = i
            break
    if done_idx is None:
        return None

    parts = [initial]
    other_tags = set(ALL_TAGS) - {"RUN_DONE"}
    prefix_re = re.compile(r"^.*?run_id=" + re.escape(run_id) + r"\s?")
    for line in run_lines[done_idx + 1:]:
        if tag_of(line, run_id) in other_tags:
            break
        parts.append(prefix_re.sub("", line))
    return "\n".join(parts).strip()


def extract_error(line: str):
    if not line:
        return None, None
    m = re.search(r"reason=(\S+)\s+message=(.*)$", line)
    if m:
        return m.group(1), m.group(2).strip()
    return None, line.strip()


def extract_num(tag_lines: dict, tag: str, cast):
    if tag not in tag_lines:
        return None
    m = re.search(re.escape(tag) + r"=(\S+)", tag_lines[tag])
    if not m:
        return None
    try:
        return cast(m.group(1))
    except ValueError:
        # Covers POWER_MA=unavailable - deliberately becomes None so it's
        # excluded from stats rather than crashing the run.
        return None


def build_metrics(tag_lines: dict) -> dict:
    cold_load_ms = extract_num(tag_lines, "COLD_LOAD_MS", int)
    ttft_ms = extract_num(tag_lines, "TTFT_MS", float)
    prefill_time_us = extract_num(tag_lines, "PREFILL_TIME_US", int)
    decode_time_us = extract_num(tag_lines, "DECODE_TIME_US", int)
    prompt_len = extract_num(tag_lines, "PROMPT_LEN", int)
    decode_len = extract_num(tag_lines, "DECODE_LEN", int)
    peak_rss_kb = extract_num(tag_lines, "PEAK_RSS_KB", int)
    power_ma = extract_num(tag_lines, "POWER_MA", float)
    power_ma_raw = None
    if "POWER_MA" in tag_lines:
        m = re.search(r"POWER_MA=(\S+)", tag_lines["POWER_MA"])
        power_ma_raw = m.group(1) if m else None
    thermal_status = None
    if "THERMAL_STATUS" in tag_lines:
        m = re.search(r"THERMAL_STATUS=(\S+)", tag_lines["THERMAL_STATUS"])
        thermal_status = m.group(1) if m else None
    thermal_cpu_c = extract_num(tag_lines, "THERMAL_TEMP_CPU_C", float)
    thermal_skin_c = extract_num(tag_lines, "THERMAL_TEMP_SKIN_C", float)
    # "unavailable" (PowerSampler produced neither a valid sampled series nor
    # a real per-sample voltage pairing) fails the float() cast the same way
    # POWER_MA's own "unavailable" string does above - extract_num() already
    # catches that and returns None, so no separate handling is needed here.
    energy_mas_sampled = extract_num(tag_lines, "ENERGY_MAS_SAMPLED", float)
    energy_mj_sampled = extract_num(tag_lines, "ENERGY_MJ_SAMPLED", float)

    # MNN reports prefill and decode performance separately (unlike engines
    # that report one combined TPS) - keep them as two distinct metrics.
    prefill_tps = None
    if prompt_len is not None and prefill_time_us:
        prefill_tps = prompt_len / (prefill_time_us / 1_000_000)
    decode_tps = None
    if decode_len is not None and decode_time_us:
        decode_tps = decode_len / (decode_time_us / 1_000_000)

    return {
        "cold_load_ms": cold_load_ms,
        "ttft_ms": ttft_ms,
        "prefill_time_us": prefill_time_us,
        "decode_time_us": decode_time_us,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "prefill_tps": round(prefill_tps, 3) if prefill_tps is not None else None,
        "decode_tps": round(decode_tps, 3) if decode_tps is not None else None,
        "peak_rss_kb": peak_rss_kb,
        "power_ma": power_ma,
        "power_ma_raw": power_ma_raw,
        "energy_mas_sampled": energy_mas_sampled,
        "energy_mj_sampled": energy_mj_sampled,
        "thermal_status": thermal_status,
        "thermal_cpu_c": thermal_cpu_c,
        "thermal_skin_c": thermal_skin_c,
    }


# ---------------------------------------------------------------------------
# Monsoon power measurement (ground truth, independent of BatteryManager)
# ---------------------------------------------------------------------------

def get_monsoon_power(duration_seconds=5) -> dict:
    """Launch monsoon_single_reading.py (physical Monsoon HVPM power meter)
    as a subprocess and return its parsed JSON reading - an independent
    ground-truth cross-check against BatteryManager's on-device power_ma
    estimate. Any failure (missing script, subprocess timeout, bad JSON,
    etc.) is caught and reported back rather than crashing the benchmark run.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(MONSOON_SCRIPT), "--duration-seconds", str(duration_seconds)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        )
        if result.returncode != 0:
            return {
                "power_ma_mean": None,
                "error": f"monsoon_single_reading.py exited with code {result.returncode}: {(result.stderr or '').strip()[:500]}",
            }
        return json.loads(result.stdout)
    except Exception as e:
        return {"power_ma_mean": None, "error": f"{type(e).__name__}: {e}"}


def run_one(adb: Adb, model_path: str, question: str, n: int, timeout: int, no_think: bool = False,
            max_tokens: int = 4096, backend_type: str = None) -> dict:
    run_id = f"run_{n}_{int(time.time() * 1000)}"
    # /no_think is appended only to the text actually sent in the broadcast -
    # the original question (without the suffix) is what gets logged/printed,
    # so progress output and the results file stay readable either way.
    prompt_text = f"{question} /no_think" if no_think else question
    clear_logcat(adb)
    fire_broadcast(adb, model_path, prompt_text, run_id, max_tokens, backend_type=backend_type)
    # Sampled right after firing the broadcast (rather than after polling
    # completes) so the reading window overlaps with the start of inference
    # instead of capturing post-inference idle power.
    monsoon = get_monsoon_power(duration_seconds=5)
    status, tag_lines, run_lines, raw_log = poll_for_result(adb, run_id, timeout)
    return {
        "run_id": run_id, "status": status, "tag_lines": tag_lines, "run_lines": run_lines,
        "monsoon": monsoon, "raw_log": raw_log,
    }


# ---------------------------------------------------------------------------
# Main per-question loop
# ---------------------------------------------------------------------------

def run_benchmark(adb: Adb, model_path: str, questions: list, timeout: int, no_think: bool = False,
                   max_tokens: int = 4096, backend_type: str = None, warmup_runs: int = 0) -> list:
    results = []
    total = len(questions)

    for n, question in enumerate(questions, start=1):
        # Independent of run_fallback_agent_mnn.py's own retry-for-truncation
        # mechanism (which lives entirely in that file, keyed off
        # response_quality.is_garbage() and scoped to that pipeline) - this
        # is a plain "always run N, discard the first N-1, keep only the
        # final one" warmup, with no quality check or garbage-detection logic
        # attached, for reproducing figures that need a steady-state (not
        # cold-cache) measurement. warmup_runs == 0 (the default) skips this
        # block entirely, so the loop falls straight through to the single
        # run_one() call below exactly as it always has.
        if warmup_runs > 0:
            for w in range(1, warmup_runs):
                print(f"[{n}/{total}] warmup attempt {w}/{warmup_runs}")
                run_one(adb, model_path, question, n, timeout, no_think=no_think, max_tokens=max_tokens,
                        backend_type=backend_type)
                time.sleep(2)
            print(f"[{n}/{total}] warmup attempt {warmup_runs}/{warmup_runs} (final - recording)")

        outcome = run_one(adb, model_path, question, n, timeout, no_think=no_think, max_tokens=max_tokens,
                           backend_type=backend_type)
        status, tag_lines, run_lines, run_id, monsoon = (
            outcome["status"], outcome["tag_lines"], outcome["run_lines"], outcome["run_id"], outcome["monsoon"]
        )

        entry = {
            "question_number": n,
            "question": question,
            "run_id": run_id,
            "status": None,
            "metrics": None,
            "response": None,
            "error": None,
        }

        if status == "done":
            metrics = build_metrics(tag_lines)
            metrics["power_ma_monsoon"] = monsoon.get("power_ma_mean")
            response = extract_response(run_lines, run_id)
            entry["status"] = "success"
            entry["metrics"] = metrics
            entry["response"] = response

            def fmt(v, unit="", nd=1):
                return f"{v:.{nd}f}{unit}" if isinstance(v, (int, float)) else "N/A"

            battery_power_disp = metrics["power_ma_raw"] if metrics["power_ma_raw"] is not None else "N/A"
            monsoon_power_disp = fmt(metrics["power_ma_monsoon"], "mA", 2)
            print(f"\n[{n}/{total}] \"{question}\"")
            print(
                f"  ColdLoad={fmt(metrics['cold_load_ms'], 'ms', 0)} TTFT={fmt(metrics['ttft_ms'], 'ms')} "
                f"PrefillTPS={fmt(metrics['prefill_tps'])} DecodeTPS={fmt(metrics['decode_tps'])} "
                f"RSS={fmt(metrics['peak_rss_kb'], 'KB', 0)} "
                f"Power={battery_power_disp}mA (BatteryMgr) / {monsoon_power_disp} (Monsoon) "
                f"ThermalCPU={fmt(metrics['thermal_cpu_c'], '°C')} ThermalSkin={fmt(metrics['thermal_skin_c'], '°C')}"
            )
            preview = response if response and len(response) <= 160 else (response[:157] + "..." if response else "")
            print(f"  Response: \"{preview}\"")
        elif status == "error":
            reason, message = extract_error(tag_lines.get("RUN_ERROR", ""))
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

    return results


# ---------------------------------------------------------------------------
# Summary / output
# ---------------------------------------------------------------------------

def stat_block(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n_completed": 0}
    return {
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.pstdev(values), 3),
        # round() on an int is a no-op, so this is safe for both int- and
        # float-valued metrics - min/max previously went in unrounded, which
        # let full-precision floats (e.g. Monsoon readings) overflow the
        # table's fixed column widths and run into the next column.
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "n_completed": len(values),
    }


SUMMARY_METRICS = [
    "cold_load_ms", "ttft_ms", "prefill_tps", "decode_tps",
    "peak_rss_kb", "power_ma", "power_ma_monsoon", "thermal_cpu_c", "thermal_skin_c",
]


def compute_summary(results: list) -> dict:
    def vals(key):
        return [
            r["metrics"][key] for r in results
            if r["status"] == "success" and r["metrics"] and r["metrics"].get(key) is not None
        ]

    thermal_states = sorted({
        r["metrics"]["thermal_status"] for r in results
        if r["status"] == "success" and r["metrics"] and r["metrics"].get("thermal_status")
    })

    # Only questions where BOTH readings succeeded are comparable - a
    # question where one source failed would otherwise silently drop out of
    # one mean but not the other, making the two means not apples-to-apples.
    power_diffs = [
        abs(r["metrics"]["power_ma"] - r["metrics"]["power_ma_monsoon"])
        for r in results
        if r["status"] == "success" and r["metrics"]
        and r["metrics"].get("power_ma") is not None
        and r["metrics"].get("power_ma_monsoon") is not None
    ]

    summary = {key: stat_block(vals(key)) for key in SUMMARY_METRICS}
    summary["thermal_states_observed"] = thermal_states
    summary["power_comparison"] = {
        "mean_abs_diff_ma": round(statistics.mean(power_diffs), 3) if power_diffs else None,
        "n_completed": len(power_diffs),
    }
    summary["note"] = (
        "power_ma stats exclude questions where POWER_MA was reported as 'unavailable' or missing; "
        "power_ma_monsoon stats exclude questions where the Monsoon reading failed. n_completed is "
        "reported per-metric because failed/timed-out questions - and, for these two power metrics "
        "specifically, unavailable/failed readings - are silently excluded from that metric's mean "
        "otherwise, which can bias results toward easier/luckier questions."
    )
    return summary


def print_summary_table(summary: dict, run_info: dict):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    no_think_disp = "ON (appending /no_think to all prompts)" if run_info.get("no_think_mode") else "OFF"
    print(f"No-think mode: {no_think_disp}")
    # Explicit fixed-precision formatting (not just str() via the field
    # width) so a value's own decimal precision can never overflow its
    # column and run into the next one - the underlying cause of the
    # power_ma_monsoon row garbling together in the terminal.
    def fmt_stat(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"

    header = f"{'Metric':<18}{'Mean':>14}{'Std':>14}{'Min':>14}{'Max':>14}{'N':>6}"
    print(header)
    print("-" * len(header))
    for key in SUMMARY_METRICS:
        s = summary[key]
        row = (
            f"{key:<18}"
            f"{fmt_stat(s['mean']):>14}"
            f"{fmt_stat(s['std']):>14}"
            f"{fmt_stat(s['min']):>14}"
            f"{fmt_stat(s['max']):>14}"
            f"{s['n_completed']:>6}"
        )
        print(row)
    print(f"\nThermal states observed: {', '.join(summary['thermal_states_observed']) or 'none'}")

    pc = summary.get("power_comparison", {})
    if pc.get("mean_abs_diff_ma") is not None:
        print(f"Power comparison (BatteryMgr vs Monsoon): mean |diff| = {pc['mean_abs_diff_ma']:.2f}mA (N={pc['n_completed']})")
    else:
        print(f"Power comparison (BatteryMgr vs Monsoon): N/A (no question had both readings available)")

    print(summary["note"])
    print(f"\n[NOTE] {TIMEOUT_NOTE}")
    if run_info["battery_warning"]:
        print("[WARN] This run's power_ma readings may be unreliable due to battery/charging state - see pre-flight output above.")


def save_results(output_path: str, run_info: dict, summary: dict, results: list):
    report = {"run_info": run_info, "results": results, "summary": summary}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OUTPUT] Results saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Automated MNN Chat benchmark pipeline (headless broadcast)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True,
                    help="Path to the model FOLDER already on the device (containing config.json/llm.mnn/llm.mnn.weight/etc.) - NOT a path to the .mnn file itself. This script does not push or convert models.")
    p.add_argument("--questions", default=None, help="Path to .txt file, one question per line")
    p.add_argument("--output", default=str(Path(__file__).resolve().parent / "logs" / "mnn_autobench_results.json"))
    p.add_argument("--timeout", type=int, default=180,
                    help=f"Seconds to wait for RUN_DONE/RUN_ERROR per question. {TIMEOUT_NOTE}")
    p.add_argument("--no-think", action="store_true", dest="no_think",
                    help="Append ' /no_think' to every question's prompt text before sending it in the broadcast "
                         "(Qwen3 models skip their <think> reasoning block entirely and answer directly - "
                         "confirmed via manual testing to cut decode_len from ~145 to ~12 tokens for the same "
                         "question). Only the text sent to the device is modified; progress output and the "
                         "results file still show the original question. Default: OFF (unchanged behavior).")
    p.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens",
                    help="Max tokens to generate per question, passed through as the max_tokens extra in every "
                         "broadcast (matches BenchmarkHeadlessReceiver's max_tokens extra on the Android side).")
    p.add_argument("--backend-type", choices=["cpu", "vulkan", "opencl"], default="cpu", dest="backend_type",
                    help="Forces a specific MNN backend via the backend_type broadcast extra "
                         "(BenchmarkHeadlessReceiver.EXTRA_BACKEND_TYPE, confirmed read by "
                         "HeadlessBenchmarkService/HeadlessBenchmarkRunner as an in-memory, pre-load override). "
                         "Default 'cpu' matches every shipped model's own config.json backend_type already, so "
                         "omitting this flag is functionally identical to today: the extra is only actually "
                         "included in the broadcast when this resolves to something other than 'cpu', since "
                         "the Kotlin side treats a genuinely absent extra (not merely one valued 'cpu') as "
                         "'leave the model's own shipped config completely untouched' - existing calls stay "
                         "byte-for-byte identical to before this flag existed.")
    p.add_argument("--warmup-runs", type=int, default=0, dest="warmup_runs",
                    help="For each question, run it this many times total and discard the first N-1 results "
                         "entirely (only a 'warmup attempt X/N' progress line each, no metrics/response logged), "
                         "recording only the FINAL run's metrics/response - for steady-state measurements "
                         "(e.g. reproducing a paper figure) rather than cold-cache-per-question ones. "
                         "Independent of run_fallback_agent_mnn.py's own retry-for-truncation mechanism - no "
                         "quality/garbage checking is attached here. Default: 0 (no warmup, unchanged behavior: "
                         "exactly one run per question, same as before this flag existed).")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("MNN Chat Automated Benchmark Pipeline")
    print(f"  Model path: {args.model_path}  Timeout: {args.timeout}s")
    no_think_banner = "ON (appending /no_think to all prompts)" if args.no_think else "OFF"
    print(f"  No-think mode: {no_think_banner}")
    # Only actually sent as a broadcast extra when it differs from "cpu" -
    # see --backend-type's own help text for why "cpu" (the default) must
    # NOT be forwarded as an explicit override to keep existing calls
    # behaviorally identical to before this flag existed.
    broadcast_backend_type = args.backend_type if args.backend_type != "cpu" else None
    print(f"  Backend type: {args.backend_type}" + (" (default, no override sent)" if broadcast_backend_type is None else " (override)"))
    if args.warmup_runs > 0:
        print(f"  Warmup runs: {args.warmup_runs} (discarding first {args.warmup_runs - 1}, recording only the final run per question)")
    print("=" * 70)
    print(f"[NOTE] {TIMEOUT_NOTE}")

    adb_bin = find_adb()
    adb = Adb(adb_bin)

    check_device(adb)
    check_mnnchat_installed(adb)
    battery_info = check_battery(adb)
    print_thermal_reminder()

    reset_mnnchat_for_clean_process(adb)

    questions = load_questions(args.questions)
    device_serial = adb.run(["get-serialno"], timeout=10).stdout.strip()

    start_time = datetime.now(timezone.utc).isoformat()
    results = run_benchmark(adb, args.model_path, questions, args.timeout, no_think=args.no_think,
                             max_tokens=args.max_tokens, backend_type=broadcast_backend_type,
                             warmup_runs=args.warmup_runs)
    end_time = datetime.now(timezone.utc).isoformat()

    completed = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - completed

    run_info = {
        "model_path": args.model_path,
        "device": device_serial or "unknown",
        "start_time": start_time,
        "end_time": end_time,
        "timeout_s": args.timeout,
        "no_think_mode": args.no_think,
        "max_tokens": args.max_tokens,
        "backend_type": args.backend_type,
        "warmup_runs": args.warmup_runs,
        "total": len(questions),
        "completed": completed,
        "failed": failed,
        "battery_warning": battery_info["battery_warning"],
        "battery_level_pct": battery_info["battery_level_pct"],
        "battery_status": battery_info["battery_status"],
    }

    summary = compute_summary(results)
    save_results(args.output, run_info, summary, results)
    print_summary_table(summary, run_info)

    print("\n" + "=" * 70)
    print(f"DONE - {completed}/{len(questions)} completed, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()
