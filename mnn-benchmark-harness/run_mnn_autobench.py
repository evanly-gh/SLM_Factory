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

# Shared measurement protocol (readiness gate, page-cache eviction, Perfetto energy) - the same
# module SmolChat's run_autobench.py uses, so both engines are measured identically.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Benchmark-Harness"))
import bench_common  # noqa: E402

# Set in main(): readiness gate + fixed rest applied before every broadcast (see pre_run()).
_GATE = None
_REST_SECONDS = 0

# HeadlessBenchmarkRunner logs one RESPDEBUG line per streamed chunk with a wall-clock ts=.
RESPDEBUG_RE = re.compile(r"onProgress call #\d+ raw_chunk=(\S+).*?\bts=(\d+)")

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
    # wall-clock markers logged by the app (HeadlessBenchmarkRunner.kt)
    "DISPATCH_EPOCH_MS", "FIRST_TOKEN_EPOCH_MS", "LAST_TOKEN_EPOCH_MS", "STREAM_CHUNKS", "LOAD_START_EPOCH_MS",
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

def check_model_mmap(adb: Adb, model_path: str):
    """Read the model folder's config.json on the device and report its `use_mmap` (MNN default:
    false). false = load() reads the weights into the process's own memory, the same kind of load
    run_autobench.py forces for llama.cpp (no mmap), so cold load means the same thing in both
    engines. true = weights are mapped from an external file in tmp_path, a different mechanism
    that would make cold load non-comparable. Returns True/False, or None if unreadable."""
    out = adb.run(["shell", f"cat {shlex.quote(model_path)}/config.json"], timeout=15).stdout or ""
    try:
        use_mmap = bool(json.loads(out).get("use_mmap", False))
    except ValueError:
        print(f"[WARN] could not read/parse {model_path}/config.json - cannot confirm use_mmap (MNN default: false)")
        return None
    if use_mmap:
        print("[WARN] this model's config.json has use_mmap=true: MNN maps its weights from an external "
              "file, so cold load will NOT be comparable to llama.cpp's no-mmap load. Remove use_mmap "
              "from config.json (default false) and rerun.")
    else:
        print("[OK] model config use_mmap=false (default): weights are read into memory at load(), "
              "same as the llama.cpp benchmark default")
    return use_mmap


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
                    backend_type: str = None, warmup_runs: int = None, trials: int = None):
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
    # Both optional (None means "let the app use its own single-shot
    # defaults": 0 warmups, 1 trial) - see run_sweep(), which is the only
    # caller that ever passes non-None values here. run_one()/run_benchmark()
    # never pass these, so their broadcasts are byte-for-byte identical to
    # before this was added.
    if warmup_runs is not None:
        cmd += f" --ei warmup_runs {int(warmup_runs)}"
    if trials is not None:
        cmd += f" --ei trials {int(trials)}"
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


def wall_clock_metrics(run_lines: list, ttft_ms, prompt_len, decode_len, cold_load_ms, tag_lines: dict = None) -> dict:
    """Paper-definition, wall-clock metrics, so MNN is measured exactly like SmolChat:
    prefill_tps = prompt_len / TTFT, decode_tps = (decode_len - 1) / (t_last_chunk - t_first_chunk),
    TTLT = dispatch -> last chunk. Also returns the epoch-ms window markers used for energy.

    The timestamps come from the app's own tags (DISPATCH_/FIRST_TOKEN_/LAST_TOKEN_EPOCH_MS,
    STREAM_CHUNKS, LOAD_START_EPOCH_MS). Older APKs logged one RESPDEBUG line per chunk instead,
    which is used as a fallback - but the phone's log quota (~300 rows/s per process) drops lines
    from such a burst, including the result tags, so rebuild the app rather than rely on it.
    All None when neither is available."""
    out = dict.fromkeys(("prefill_tps", "decode_tps", "ttlt_ms", "stream_chunks", "dispatch_epoch_ms",
                         "first_token_epoch_ms", "last_token_epoch_ms", "load_start_epoch_ms"))
    if ttft_ms and prompt_len:
        out["prefill_tps"] = prompt_len / (ttft_ms / 1000)
    dispatch = first = last = chunks = load_start = None
    if tag_lines:
        dispatch = extract_num(tag_lines, "DISPATCH_EPOCH_MS", int)
        first = extract_num(tag_lines, "FIRST_TOKEN_EPOCH_MS", int)
        last = extract_num(tag_lines, "LAST_TOKEN_EPOCH_MS", int)
        chunks = extract_num(tag_lines, "STREAM_CHUNKS", int)
        load_start = extract_num(tag_lines, "LOAD_START_EPOCH_MS", int)
    if dispatch is None or first is None or last is None:
        stamps = [int(m.group(2)) for m in (RESPDEBUG_RE.search(l) for l in run_lines or [])
                  if m and m.group(1) != "null"]
        if not stamps or ttft_ms is None or ttft_ms < 0:
            return out
        first, last, chunks = min(stamps), max(stamps), len(stamps)
        dispatch = int(first - ttft_ms)
    if load_start is None and cold_load_ms:
        # Approximation: load() ends right before the first generate(), which is all the code does in
        # between (setKeepHistory/updateThinking/updateMaxNewTokens).
        load_start = int(dispatch - cold_load_ms)
    out.update({
        "stream_chunks": chunks,
        "dispatch_epoch_ms": dispatch,
        "first_token_epoch_ms": first,
        "last_token_epoch_ms": last,
        "ttlt_ms": last - dispatch,
        "load_start_epoch_ms": load_start,
    })
    # The first->last chunk window spans (chunks - 1) steps. Use the streamed chunk count, not decode_len:
    # decode_len counts every sampled token incl. the stop token when the model ends an answer on its
    # own, but that token is never streamed (measured: chunks == decode_len - 1 on every early-stopped
    # answer, chunks == decode_len when the 256-token cap ends it). Using decode_len - 1 overstated
    # decode speed by 20-50% on 4-7 token answers.
    n_steps = (chunks - 1) if chunks else ((decode_len - 1) if decode_len else None)
    if n_steps and n_steps >= 1 and last > first:
        out["decode_tps"] = n_steps / ((last - first) / 1000)
    return out


def build_metrics(tag_lines: dict, run_lines: list = None) -> dict:
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

    native_prefill = round(prefill_tps, 3) if prefill_tps is not None else None
    native_decode = round(decode_tps, 3) if decode_tps is not None else None
    metrics = {
        "cold_load_ms": cold_load_ms,
        "ttft_ms": ttft_ms,
        "prefill_time_us": prefill_time_us,
        "decode_time_us": decode_time_us,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        # Same names/meaning as run_autobench.py (SmolChat), so both engines line up.
        "prompt_tokens": prompt_len,
        "gen_tokens": decode_len,
        "native_prefill_tps": native_prefill,
        "native_decode_tps": native_decode,
        # Without run_lines (legacy callers, e.g. energy_latency_agent.py) these stay the native
        # engine-timer rates, as before; with run_lines they become the wall-clock paper
        # definitions below.
        "prefill_tps": native_prefill,
        "decode_tps": native_decode,
        "cold_start_ms": (cold_load_ms + ttft_ms) if cold_load_ms and ttft_ms and ttft_ms > 0 else None,
        "peak_rss_kb": peak_rss_kb,
        "power_ma": power_ma,
        "power_ma_raw": power_ma_raw,
        "energy_mas_sampled": energy_mas_sampled,
        "energy_mj_sampled": energy_mj_sampled,
        "thermal_status": thermal_status,
        "thermal_cpu_c": thermal_cpu_c,
        "thermal_skin_c": thermal_skin_c,
    }
    if run_lines is not None:
        wall = wall_clock_metrics(run_lines, ttft_ms, prompt_len, decode_len, cold_load_ms, tag_lines)
        for key in ("prefill_tps", "decode_tps"):
            metrics[key] = round(wall[key], 3) if wall[key] is not None else None
        metrics.update({k: v for k, v in wall.items() if k not in ("prefill_tps", "decode_tps")})
        # tokens delivered to the user (SmolChat's count also excludes the stop token)
        if wall.get("stream_chunks") and decode_len is not None and abs(decode_len - wall["stream_chunks"]) <= 1:
            metrics["gen_tokens"] = wall["stream_chunks"]
    return metrics


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


def poll_for_sweep_result(adb: Adb, base_run_id: str, final_sub_id: str, timeout: int) -> tuple:
    """Like poll_for_result(), but for a run_sweep() broadcast that runs
    several generations (warmups + trials) against one loaded session
    inside a single broadcast: succeeds once final_sub_id's own RUN_DONE
    appears (the last trial finished), but also stops early if the whole
    sweep aborts between generations - an unhandled exception inside
    HeadlessBenchmarkRunner.run() logs RUN_ERROR under the sweep's own
    base_run_id, not any one generation's sub run_id, since it isn't tied
    to any single generation. Substring matching (get_run_id_lines) means
    parse_run(raw, base_run_id) would also see every sub-id's lines, but
    tag_of()'s regex requires whitespace immediately after the run_id
    token, which a sub-id like "{base_run_id}_t1" never has right after
    the bare base_run_id - so base_run_id's own tag_lines only ever
    contains genuinely bare-base_run_id lines (SWEEP_CONFIG,
    BACKEND_TYPE_OVERRIDE, MAX_TOKENS, ENABLE_THINKING, SWEEP_STEP, and
    the outer catch block's own RUN_ERROR), never a sub-generation's.
    Returns (status, raw_log) - no tag_lines/run_lines, since callers
    re-derive those per sub-id from raw_log via parse_run() themselves.
    """
    deadline = time.time() + timeout
    last_raw = ""
    while time.time() < deadline:
        result = adb.run(["logcat", "-d", "-b", "main"], timeout=30)
        raw = result.stdout or ""
        last_raw = raw
        final_tags, _ = parse_run(raw, final_sub_id)
        if "RUN_DONE" in final_tags:
            return "done", raw
        if "RUN_ERROR" in final_tags:
            return "error", raw
        base_tags, _ = parse_run(raw, base_run_id)
        if "RUN_ERROR" in base_tags:
            return "error", raw
        time.sleep(1)
    return "timeout", last_raw


def pre_run(adb: Adb) -> dict:
    """Fixed rest, then the readiness gate (cool + CPU caps at baseline), then a device-state
    snapshot recorded with the run. With no gate configured this is just the snapshot."""
    if _REST_SECONDS:
        time.sleep(_REST_SECONDS)
    if _GATE is not None:
        return _GATE.wait()
    return {"passed": None, "waited_s": 0.0, "state": bench_common.device_state(adb)}


def run_sweep(adb: Adb, model_path: str, question: str, n: int, timeout: int, no_think: bool = False,
              max_tokens: int = 4096, backend_type: str = None, warmup_runs: int = 1, trials: int = 1,
              gate: dict = None) -> dict:
    """Fires ONE broadcast that runs `warmup_runs` discarded warmup
    generations followed by `trials` recorded generations, all against a
    SINGLE loaded LlmSession on-device (see HeadlessBenchmarkRunner.run()
    and its warmupRuns/trials parameters) - unlike run_one(), which always
    creates, loads, and tears down a brand-new session for one generation.

    This exists because OpenCL (and any other GPU backend) JIT-compiles
    kernels lazily on the first real forward pass, not at load() time, and
    that compile cost is only amortized across generations sharing one
    live session/runtime. run_trials_benchmark()'s original design fired a
    separate broadcast (and therefore a brand-new session) for the warmup
    AND for every recorded trial, so every one of them - including the
    "warmup" - paid the full cold-compile cost from scratch: warmup never
    actually warmed anything up. run_sweep() reuses one session across the
    whole warmup+trials sequence, matching the paper protocol this harness
    is trying to reproduce ("one warm-up run followed by at least three
    recorded trials").

    Returns a dict: {"warmups": [...], "trials": [...], "monsoon": {...},
    "raw_log": str}, where each list entry is a run_one()-shaped outcome
    dict ({"run_id", "status", "tag_lines", "run_lines"}) for that one
    generation, parsed out of the single shared logcat capture.

    Note on Monsoon: the single physical-power reading below is taken once,
    right after firing the broadcast, so it only really characterizes the
    very start of the sweep (typically the first warmup's cold compile) -
    unlike run_one()'s per-call reading, it is NOT a meaningful per-trial
    ground-truth cross-check here. Each trial's BatteryManager-based
    power_ma is unaffected and still recorded correctly per generation.
    """
    # Gate once per sweep: warmup + trials run back-to-back inside one on-device session.
    gate = gate or pre_run(adb)
    base_run_id = f"run_{n}_{int(time.time() * 1000)}"
    prompt_text = f"{question} /no_think" if no_think else question
    clear_logcat(adb)
    fire_broadcast(adb, model_path, prompt_text, base_run_id, max_tokens,
                   backend_type=backend_type, warmup_runs=warmup_runs, trials=trials)
    # Sampled right after firing the broadcast, same as run_one() - see the
    # Monsoon caveat in this function's docstring.
    monsoon = get_monsoon_power(duration_seconds=5)

    final_sub_id = f"{base_run_id}_t{trials}"
    # Every generation in the sweep could in principle need the full
    # per-generation timeout (the first one always does, for the cold
    # compile; a later one only would if something regressed) - budget for
    # all of them rather than risk cutting off a slow final trial just
    # because the sweep as a whole ran long.
    aggregate_timeout = timeout * (warmup_runs + trials)
    sweep_status, raw_log = poll_for_sweep_result(adb, base_run_id, final_sub_id, aggregate_timeout)

    # If the sweep ended for a reason OTHER than genuinely running out of
    # time (sweep_status != "timeout"), any generation that never got its
    # own RUN_DONE/RUN_ERROR didn't "time out" - the sweep aborted before
    # ever reaching it (most commonly: an unhandled exception inside
    # HeadlessBenchmarkRunner.run(), logged as a bare RUN_ERROR under
    # base_run_id itself, before or between generations - see
    # poll_for_sweep_result()'s own docstring). Label those "aborted"
    # instead of "timeout" and carry over the real reason/message so
    # run_trials_benchmark() can print what actually happened rather than
    # a misleading "timeout after {timeout}s" when the real failure
    # surfaced in a couple of seconds, not {timeout}.
    base_tag_lines, _ = parse_run(raw_log, base_run_id)
    sweep_abort_reason, sweep_abort_message = extract_error(base_tag_lines.get("RUN_ERROR", ""))

    def _generation_status(sub_id: str, tag_lines: dict) -> str:
        if "RUN_DONE" in tag_lines:
            return "done"
        if "RUN_ERROR" in tag_lines:
            return "error"
        if sweep_status != "timeout":
            return "aborted"
        return "timeout"

    warmups = []
    for w in range(1, warmup_runs + 1):
        sub_id = f"{base_run_id}_w{w}"
        tag_lines, run_lines = parse_run(raw_log, sub_id)
        sub_status = _generation_status(sub_id, tag_lines)
        warmups.append({"run_id": sub_id, "status": sub_status, "tag_lines": tag_lines, "run_lines": run_lines})

    trial_outcomes = []
    for t in range(1, trials + 1):
        sub_id = f"{base_run_id}_t{t}"
        tag_lines, run_lines = parse_run(raw_log, sub_id)
        sub_status = _generation_status(sub_id, tag_lines)
        trial_outcomes.append({"run_id": sub_id, "status": sub_status, "tag_lines": tag_lines, "run_lines": run_lines})

    return {
        "base_run_id": base_run_id,
        "sweep_status": sweep_status,
        "sweep_abort_reason": sweep_abort_reason,
        "sweep_abort_message": sweep_abort_message,
        "warmups": warmups,
        "trials": trial_outcomes,
        "monsoon": monsoon,
        "raw_log": raw_log,
        "gate": gate,
        "state_after": bench_common.device_state(adb),
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
            metrics = build_metrics(tag_lines, run_lines)
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


def run_trials_benchmark(adb: Adb, model_path: str, questions: list, timeout: int, no_think: bool = False,
                          max_tokens: int = 4096, backend_type: str = None, trials: int = 1) -> list:
    """Paper-exact protocol, genuinely different from --warmup-runs's own
    "run N, discard N-1, keep only the last" mechanic: for EACH question,
    run once as a discarded warmup, then run `trials` real times, recording
    EVERY one of those trials as its own full result. Each entry carries a
    "trial_number" (1..trials) field (in addition to the usual
    "question_number") so per-question trial statistics can be computed
    afterward (see compute_trial_summary()).

    The warmup and all `trials` recorded generations run inside ONE
    run_sweep() broadcast, sharing a single loaded on-device LlmSession,
    rather than each being its own separate broadcast/session (the original
    implementation). A fresh session per trial paid a full cold OpenCL (or
    any GPU backend) kernel JIT-compile cost on literally every trial,
    warmup included - which meant "warmup" never warmed anything up, and
    every trial number showed the identical cold-compile-dominated timing
    no matter what engine-level fix was tried. See run_sweep()'s own
    docstring for the full explanation.

    Deliberately a fully independent function, not a refactor of
    run_benchmark() into a shared code path - it has its own copy of the
    entry-building/printing logic below, so --warmup-runs's existing,
    already-relied-upon behavior in run_benchmark() cannot be affected by
    anything added here, even indirectly.
    """
    results = []
    total = len(questions)
    warmup_runs = 1  # fixed at 1, matching this protocol's existing "one discarded warmup" contract

    for n, question in enumerate(questions, start=1):
        print(f"\n[{n}/{total}] \"{question}\" - warmup (discarded), then {trials} recorded trial(s) - one shared session")
        sweep = run_sweep(adb, model_path, question, n, timeout, no_think=no_think, max_tokens=max_tokens,
                           backend_type=backend_type, warmup_runs=warmup_runs, trials=trials)
        monsoon = sweep["monsoon"]

        for trial in range(1, trials + 1):
            print(f"[{n}/{total}] trial {trial}/{trials} (recording)")
            outcome = sweep["trials"][trial - 1]
            status, tag_lines, run_lines, run_id = (
                outcome["status"], outcome["tag_lines"], outcome["run_lines"], outcome["run_id"]
            )

            entry = {
                "question_number": n,
                "question": question,
                "trial_number": trial,
                "phase": "steady",
                "run_id": run_id,
                "status": None,
                "metrics": None,
                "response": None,
                "error": None,
                "gate": {k: v for k, v in sweep["gate"].items() if k != "state"},
                "state_before": sweep["gate"].get("state"),
                "state_after": sweep["state_after"],
            }

            if status == "done":
                metrics = build_metrics(tag_lines, run_lines)
                metrics["power_ma_monsoon"] = monsoon.get("power_ma_mean")
                response = extract_response(run_lines, run_id)
                entry["status"] = "success"
                entry["metrics"] = metrics
                entry["response"] = response

                def fmt(v, unit="", nd=1):
                    return f"{v:.{nd}f}{unit}" if isinstance(v, (int, float)) else "N/A"

                print(
                    f"  ColdLoad={fmt(metrics['cold_load_ms'], 'ms', 0)} TTFT={fmt(metrics['ttft_ms'], 'ms')} "
                    f"TTLT={fmt(metrics.get('ttlt_ms'), 'ms', 0)} "
                    f"Prefill={fmt(metrics['prefill_tps'])} (native {fmt(metrics['native_prefill_tps'])}) "
                    f"Decode={fmt(metrics['decode_tps'])} (native {fmt(metrics['native_decode_tps'])}) "
                    f"Tokens={metrics['prompt_len']}+{metrics['decode_len']} (chunks {metrics.get('stream_chunks')}) "
                    f"RSS={fmt(metrics['peak_rss_kb'], 'KB', 0)} "
                    f"ThermalCPU={fmt(metrics['thermal_cpu_c'], '°C')}"
                )
                preview = response if response and len(response) <= 160 else (response[:157] + "..." if response else "")
                print(f"  Response: \"{preview}\"")
            elif status == "error":
                reason, message = extract_error(tag_lines.get("RUN_ERROR", ""))
                entry["status"] = "failed"
                entry["error"] = {"reason": reason, "message": message}
                print(f"  FAILED: reason={reason} message={message}")
            elif status == "aborted":
                # The sweep ended (done or error) before this generation
                # ever got its own RUN_DONE/RUN_ERROR - most likely because
                # an earlier generation in the same sweep hit an unhandled
                # exception, logged under the sweep's own base run_id
                # rather than this generation's sub-id. Surface the real
                # reason/message from that base-level failure instead of a
                # misleading "timeout after {timeout}s", since this branch
                # is typically reached within seconds, not the real timeout.
                reason = sweep.get("sweep_abort_reason") or "sweep_aborted"
                message = sweep.get("sweep_abort_message") or (
                    f"Sweep ended (status={sweep['sweep_status']}) before this generation produced output - "
                    "check logcat for an earlier RUN_ERROR under the base run_id."
                )
                entry["status"] = "failed"
                entry["error"] = {"reason": reason, "message": message}
                print(f"  FAILED: sweep aborted early - reason={reason} message={message}")
            else:  # timeout
                entry["status"] = "failed"
                entry["error"] = {"reason": "timeout", "message": f"No RUN_DONE/RUN_ERROR within {timeout}s"}
                print(f"  FAILED: timeout after {timeout}s")

            results.append(entry)

    return results


def print_run_line(metrics: dict, response: str):
    def f(key, unit="", nd=1):
        v = metrics.get(key)
        return "N/A" if not isinstance(v, (int, float)) else f"{v:.{nd}f}{unit}"
    print(f"  ColdLoad={f('cold_load_ms', 'ms', 0)} TTFT={f('ttft_ms', 'ms')} TTLT={f('ttlt_ms', 'ms', 0)} "
          f"Prefill={f('prefill_tps')} (native {f('native_prefill_tps')}) "
          f"Decode={f('decode_tps')} (native {f('native_decode_tps')}) "
          f"Tokens={metrics.get('prompt_len')}+{metrics.get('decode_len')} (chunks {metrics.get('stream_chunks')}) "
          f"RSS={f('peak_rss_kb', 'KB', 0)} ThermalCPU={f('thermal_cpu_c', 'C')}")
    preview = response if response and len(response) <= 160 else (response[:157] + "..." if response else "")
    print(f"  Response: \"{preview}\"")


def entry_from_outcome(outcome: dict, sweep: dict, n: int, question: str, timeout: int) -> dict:
    """Turn one generation of a run_sweep() into a results entry."""
    status, tag_lines, run_lines, run_id = (
        outcome["status"], outcome["tag_lines"], outcome["run_lines"], outcome["run_id"])
    entry = {
        "question_number": n, "question": question, "run_id": run_id,
        "status": None, "metrics": None, "response": None, "error": None,
        "gate": {k: v for k, v in sweep["gate"].items() if k != "state"},
        "state_before": sweep["gate"].get("state"),
        "state_after": sweep["state_after"],
    }
    if status == "done":
        metrics = build_metrics(tag_lines, run_lines)
        metrics["power_ma_monsoon"] = sweep["monsoon"].get("power_ma_mean")
        entry["status"] = "success"
        entry["metrics"] = metrics
        entry["response"] = extract_response(run_lines, run_id)
    elif status == "aborted":
        entry["status"] = "failed"
        entry["error"] = {"reason": sweep.get("sweep_abort_reason") or "sweep_aborted",
                          "message": sweep.get("sweep_abort_message") or f"Sweep ended (status={sweep['sweep_status']}) "
                                     "before this generation produced output - check logcat for an earlier RUN_ERROR."}
        print(f"  FAILED: sweep aborted early - {entry['error']}")
    elif status == "error":
        reason, message = extract_error(tag_lines.get("RUN_ERROR", ""))
        entry["status"] = "failed"
        entry["error"] = {"reason": reason, "message": message}
        print(f"  FAILED: reason={reason} message={message}")
    else:
        entry["status"] = "failed"
        entry["error"] = {"reason": "timeout", "message": f"No RUN_DONE/RUN_ERROR within {timeout}s"}
        print(f"  FAILED: timeout after {timeout}s")
    return entry


def clear_mnn_kernel_cache(adb: Adb) -> list:
    """Delete MNN's persistent GPU kernel/tuning cache (mnn_cachefile*) so every cold start pays
    kernel compilation/tuning, like a first launch (llama.cpp's OpenCL build recompiles per process).
    MNN writes it to <tmp_path>/mnn_cachefile.bin; without mmap the app leaves tmp_path empty, so
    it becomes a path relative to the app's working directory and may not exist at all - hence the
    search over every plausible location. Returns the removed paths (empty = none existed)."""
    removed = []
    out = adb.run(["shell", f"find /data/local/tmp /sdcard/Android/data/{PACKAGE} -maxdepth 6 "
                            "-name 'mnn_cachefile*' 2>/dev/null"], timeout=60).stdout or ""
    for path in out.split():
        adb.run(["shell", f"rm -f {shlex.quote(path)}"], timeout=15)
        removed.append(path)
    out = adb.run(["shell", f"run-as {PACKAGE} sh -c \"find . -name 'mnn_cachefile*' 2>/dev/null\""],
                  timeout=60).stdout or ""
    for path in out.split():
        adb.run(["shell", f"run-as {PACKAGE} rm -f {shlex.quote(path)}"], timeout=15)
        removed.append(f"[app dir] {path}")
    return removed


def run_protocol_benchmark(adb: Adb, model_path: str, questions: list, runs: int, timeout: int,
                           no_think: bool = False, max_tokens: int = 256, backend_type: str = None) -> list:
    """Per-question protocol. For each question:
      1. rest + readiness gate (once),
      2. force-stop MNN Chat and evict every file of the model folder from the page cache,
      3. one broadcast that creates ONE session, loads the model, and runs the question `runs`
         times back to back (no warm-up, no gate in between): run 1 is cold, runs 2..N are warm.
    Cold load / cold start come from run 1 (MNN only reports COLD_LOAD_MS for a session's first
    generation) and are kept constant for the question's other runs; the LAST successful run is the
    reported one (see bench_common.finalize_question). Every run is kept."""
    results = []
    total = len(questions)
    for n, question in enumerate(questions, start=1):
        for attempt in (1, 2):
            group = _run_question(adb, model_path, question, n, total, runs, timeout, no_think, max_tokens, backend_type)
            if all(e["status"] == "success" for e in group):
                break
            if attempt == 1:
                print("  a run of this question failed (lost log lines?) - retrying the whole question once")
        for e in group:
            e["attempt"] = attempt
        results.extend(group)
        bench_common.finalize_question(group)
    return results


def _run_question(adb: Adb, model_path: str, question: str, n: int, total: int, runs: int, timeout: int,
                  no_think: bool, max_tokens: int, backend_type: str) -> list:
    """One attempt at a question: gate, force-stop, evict, one sweep of `runs` generations."""
    gate = pre_run(adb)
    adb.run(["shell", "am", "force-stop", PACKAGE], timeout=15)
    time.sleep(2)
    eviction = bench_common.evict_page_cache(adb, PACKAGE, model_path)
    print(f"\n[{n}/{total}] \"{question}\" - page cache evicted (model resident after: "
          f"{eviction.get('resident_after_pct')}%)" + (f" [{eviction['error']}]" if eviction.get("error") else ""))
    kernel_cache_removed = None
    if backend_type and backend_type != "cpu":
        kernel_cache_removed = clear_mnn_kernel_cache(adb)
        print("  MNN kernel cache: " + (f"removed {kernel_cache_removed}" if kernel_cache_removed
                                        else "none found (nothing persists between cold starts)"))
    sweep = run_sweep(adb, model_path, question, n, timeout, no_think=no_think, max_tokens=max_tokens,
                      backend_type=backend_type, warmup_runs=0, trials=runs, gate=gate)
    group = []
    for run in range(1, runs + 1):
        phase = "cold" if run == 1 else "warm"
        print(f"[{n}/{total}] run {run}/{runs} ({phase})")
        entry = entry_from_outcome(sweep["trials"][run - 1], sweep, n, question, timeout)
        entry.update({"protocol": bench_common.PROTOCOL, "run_number": run, "phase": phase, "reported": False})
        if run == 1:
            entry["page_cache_eviction"] = eviction
            entry["mnn_kernel_cache_removed"] = kernel_cache_removed
        if entry["metrics"]:
            print_run_line(entry["metrics"], entry["response"])
        group.append(entry)
    return group


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
    "cold_load_ms", "ttft_ms", "ttlt_ms", "prefill_tps", "decode_tps", "native_prefill_tps", "native_decode_tps",
    "prompt_tokens", "gen_tokens", "stream_chunks",
    "peak_rss_kb", "power_ma", "power_ma_monsoon", "thermal_cpu_c", "thermal_skin_c",
    "energy_mas_sampled", "energy_mj_sampled",
    "energy_mj", "energy_net_mj", "avg_power_mw", "energy_mj_per_token", "energy_net_mj_per_token",
]
# Perfetto energy fields only count toward statistics when the run's energy_valid is true.
ENERGY_KEYS = {"energy_mj", "energy_net_mj", "avg_power_mw", "energy_mj_per_token", "energy_net_mj_per_token"}
COLD_METRICS = ["cold_start_ms", "cold_load_ms", "ttft_ms", "cold_start_energy_mj", "cold_start_energy_net_mj"]


def metric_values(results: list, key: str) -> list:
    return [
        r["metrics"][key] for r in results
        if r["status"] == "success" and r["metrics"] and r["metrics"].get(key) is not None
        and (key not in ENERGY_KEYS or r["metrics"].get("energy_valid"))
    ]


def compute_cold_summary(cold_results: list) -> dict:
    return {key: stat_block(metric_values(cold_results, key)) for key in COLD_METRICS}


def compute_summary(results: list) -> dict:
    results = bench_common.summary_entries(results)

    def vals(key):
        return metric_values(results, key)

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
    cold = summary.get("cold_start") or {}
    if cold.get("cold_start_ms", {}).get("n_completed"):
        print("\nCold starts (run 1 of each question, model evicted from page cache):")
        for label, key in [("ColdStart_ms", "cold_start_ms"), ("ColdLoad_ms", "cold_load_ms"), ("ColdTTFT_ms", "ttft_ms")]:
            c_ = cold[key]
            print(f"{label:<18}{fmt_stat(c_['mean']):>14}{fmt_stat(c_['std']):>14}{fmt_stat(c_['min']):>14}"
                  f"{fmt_stat(c_['max']):>14}{c_['n_completed']:>6}")
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


def compute_trial_summary(results: list, questions: list) -> dict:
    """--trials' own new summary section: mean/std (plus min/max/n, via the
    same stat_block() the overall summary uses) across each question's OWN
    kept trials, for every metric in SUMMARY_METRICS - distinct from
    compute_summary(), which still pools ALL recorded results together
    (still meaningful in --trials mode too, just not what this adds).
    Keyed by question_number as a string (JSON object keys must be strings).
    """
    summary_by_question = {}
    for n, question in enumerate(questions, start=1):
        question_results = [r for r in results if r["question_number"] == n]

        def vals(key):
            return metric_values(question_results, key)   # same rules as the headline (energy only if valid)

        summary_by_question[str(n)] = {
            "question": question,
            "n_trials": len(question_results),
            "n_completed": sum(1 for r in question_results if r["status"] == "success"),
            "metrics": {key: stat_block(vals(key)) for key in SUMMARY_METRICS},
        }
    return summary_by_question


def print_trial_summary(trial_summary: dict):
    print("\n" + "=" * 70)
    print("TRIAL SUMMARY (mean/std across N kept trials, per question)")
    print("=" * 70)

    def fmt_stat(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"

    for qnum, qsum in trial_summary.items():
        print(f"\nQ{qnum}: \"{qsum['question']}\"  (n_trials={qsum['n_trials']}, n_completed={qsum['n_completed']})")
        header = f"{'Metric':<18}{'Mean':>14}{'Std':>14}{'Min':>14}{'Max':>14}{'N':>6}"
        print(header)
        print("-" * len(header))
        for key in SUMMARY_METRICS:
            s = qsum["metrics"][key]
            row = (
                f"{key:<18}"
                f"{fmt_stat(s['mean']):>14}"
                f"{fmt_stat(s['std']):>14}"
                f"{fmt_stat(s['min']):>14}"
                f"{fmt_stat(s['max']):>14}"
                f"{s['n_completed']:>6}"
            )
            print(row)


def save_results(output_path: str, run_info: dict, summary: dict, results: list, trial_summary: dict = None):
    report = {"run_info": run_info, "results": results, "summary": summary}
    # Only added when --trials produced one (default None) - existing
    # callers/output shape are completely unaffected when this is omitted.
    if trial_summary is not None:
        report["trial_summary"] = trial_summary
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
    p.add_argument("--max-tokens", type=int, default=256, dest="max_tokens",
                    help="Max tokens to generate per question (a ceiling - generation still stops earlier at the "
                         "model's end token). Default 256 = the same cap SmolChat's harness uses; the old default was 4096.")
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
                    help="[--legacy-protocol only] For each question, run it this many times total and discard the first N-1 results "
                         "entirely (only a 'warmup attempt X/N' progress line each, no metrics/response logged), "
                         "recording only the FINAL run's metrics/response - for steady-state measurements "
                         "(e.g. reproducing a paper figure) rather than cold-cache-per-question ones. "
                         "Independent of run_fallback_agent_mnn.py's own retry-for-truncation mechanism - no "
                         "quality/garbage checking is attached here. Default: 0 (no warmup, unchanged behavior: "
                         "exactly one run per question, same as before this flag existed). Mutually exclusive "
                         "with --trials - genuinely different protocols, not meant to be combined.")
    p.add_argument("--trials", type=int, default=1, dest="trials",
                    help="[--legacy-protocol only] A genuinely different protocol from --warmup-runs's 'discard all but last': for each "
                         "question, run ONCE as a discarded warmup, then run this many SEPARATE, real times, "
                         "recording EVERY one as its own full result (not collapsed into a single kept entry). "
                         "Adds a new 'trial_summary' section to the output JSON: mean/std (and min/max/n) per "
                         "metric, per question, across that question's own N kept trials - separate from the "
                         "existing overall 'summary' section, which still pools every recorded result together "
                         "regardless. Mutually exclusive with --warmup-runs. Default: 1 (no warmup, unchanged "
                         "behavior: exactly one run per question via the existing --warmup-runs code path, same "
                         "as before this flag existed).")
    p.add_argument("--no-process-reset", action="store_true", dest="no_process_reset",
                    help="Skip reset_mnnchat_for_clean_process() (the ONE-TIME force-stop+relaunch that "
                         "otherwise happens once, at script startup, before any question/warmup/trial runs - "
                         "NOT before each individual run). Trades away accurate PEAK_RSS_KB (it can be "
                         "contaminated by a larger model's high-water mark left over from MNN Chat's last use, "
                         "e.g. the normal chat UI or a prior separate script invocation) for skipping that "
                         "~4s force-stop+relaunch+settle delay. Default: OFF (unchanged behavior - the reset "
                         "always runs, same as before this flag existed).")
    bench_common.add_common_args(p)
    return p.parse_args()


def main():
    args = parse_args()

    if args.trials > 1 and args.warmup_runs > 0:
        print("[ERROR] --trials and --warmup-runs are two different, mutually exclusive protocols - use one or the other, not both.")
        sys.exit(1)

    print("=" * 70)
    print("MNN Chat Automated Benchmark Pipeline")
    print(f"  Model path: {args.model_path}  Timeout: {args.timeout}s")
    no_think_banner = "ON (appending /no_think to all prompts)" if args.no_think else "OFF"
    print(f"  No-think mode: {no_think_banner}")
    # Always sent as an explicit broadcast extra now (previously omitted
    # for "cpu" on the assumption that every deployed model's own
    # config.json already defaults to backend_type=cpu, so omitting the
    # override was "behaviorally identical" to sending it). That assumption
    # broke silently: a model folder deployed for GPU testing can have
    # backend_type baked into its own config.json as e.g. "opencl" (this is
    # confirmed true for at least one deployed model here), and in that
    # case the old omit-for-cpu logic meant "--backend-type cpu" never
    # actually forced CPU at all - it silently kept running on whatever
    # backend_type the model's config.json already had. Always forwarding
    # the extra makes this flag's behavior correct regardless of what's
    # baked into any given model folder's config.json - LlmSession.kt's own
    # override check (backendType != null) already handles an explicit
    # "cpu" value correctly, so this was a Python-side over-optimization,
    # not something the Kotlin/native side needed.
    broadcast_backend_type = args.backend_type
    print(f"  Backend type: {args.backend_type} (explicit override sent)")
    if args.warmup_runs > 0:
        print(f"  Warmup runs: {args.warmup_runs} (discarding first {args.warmup_runs - 1}, recording only the final run per question)")
    if args.legacy_protocol:
        print("  Protocol: legacy (--trials / --warmup-runs flow)")
        if args.trials > 1:
            print(f"  Trials: {args.trials} (1 discarded warmup + {args.trials} SEPARATE recorded trials per question)")
    else:
        print(f"  Protocol: per-question - {args.runs_per_question} back-to-back runs in one session after a "
              "force-stop + page-cache eviction; run 1 = cold, last run reported")
    print("=" * 70)
    print(f"[NOTE] {TIMEOUT_NOTE}")

    adb_bin = find_adb()
    adb = Adb(adb_bin)

    check_device(adb)
    check_mnnchat_installed(adb)
    battery_info = check_battery(adb)
    model_use_mmap = check_model_mmap(adb, args.model_path)
    print_thermal_reminder()

    if args.no_process_reset:
        print("[NOTE] --no-process-reset: skipping the one-time force-stop+relaunch. PEAK_RSS_KB may be "
              "contaminated by a high-water mark left over from MNN Chat's prior use.")
    else:
        reset_mnnchat_for_clean_process(adb)

    questions = load_questions(args.questions)
    device_serial = adb.run(["get-serialno"], timeout=10).stdout.strip()

    global _GATE, _REST_SECONDS
    _REST_SECONDS = args.rest_seconds
    initial_state = bench_common.device_state(adb)
    if args.gate_max_temp is not None or args.gate_temp_rise is not None:
        _GATE = bench_common.ReadinessGate(adb, args.gate_max_temp, args.gate_timeout, rise_c=args.gate_temp_rise)
        _GATE.set_baseline(initial_state)
    if args.energy and initial_state["externally_powered"]:
        print(f"[WARN] --energy: phone is externally powered ({', '.join(initial_state['power_sources'])}) - "
              "energy will be recorded but flagged invalid. Unplug and use wireless adb for valid energy.")

    energy_trace, idle_window = None, None
    if args.energy:
        energy_trace = bench_common.EnergyTrace(adb, Path(args.output).resolve().parent / "traces")
        energy_trace.start()
        idle_window = bench_common.measure_idle_window(adb, args.idle_seconds)

    start_time = datetime.now(timezone.utc).isoformat()
    try:
        if args.legacy_protocol:
            if args.trials > 1:
                results = run_trials_benchmark(adb, args.model_path, questions, args.timeout, no_think=args.no_think,
                                                max_tokens=args.max_tokens, backend_type=broadcast_backend_type,
                                                trials=args.trials)
            else:
                results = run_benchmark(adb, args.model_path, questions, args.timeout, no_think=args.no_think,
                                         max_tokens=args.max_tokens, backend_type=broadcast_backend_type,
                                         warmup_runs=args.warmup_runs)
        else:
            results = run_protocol_benchmark(adb, args.model_path, questions, args.runs_per_question, args.timeout,
                                             no_think=args.no_think, max_tokens=args.max_tokens,
                                             backend_type=broadcast_backend_type)
    finally:
        if energy_trace is not None:
            energy_trace.stop()
    end_time = datetime.now(timezone.utc).isoformat()

    idle_power_mw = None
    if energy_trace is not None:
        idle_power_mw = bench_common.attach_energy(results, energy_trace, idle_window)

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
        "trials": args.trials,
        "process_reset": not args.no_process_reset,
        # len(results), not len(questions): with --trials, results holds
        # len(questions) * args.trials entries (numerically identical to
        # len(questions) when args.trials is the default 1).
        "total": len(results),
        "completed": completed,
        "failed": failed,
        "battery_warning": battery_info["battery_warning"],
        "battery_level_pct": battery_info["battery_level_pct"],
        "battery_status": battery_info["battery_status"],
        "use_mmap": model_use_mmap,
        "decode_definition": "stream_chunks-1 over first->last chunk (v2)",
        **bench_common.config_labels("mnn", args.model_path, args.backend_type, results),
        "protocol": "legacy" if args.legacy_protocol else bench_common.PROTOCOL,
        "runs_per_question": None if args.legacy_protocol else args.runs_per_question,
        "gate_max_temp_c": args.gate_max_temp,
        "gate_temp_rise_c": args.gate_temp_rise,
        "gate_baseline_cpu_caps_khz": _GATE.baseline_caps if _GATE else None,
        "rest_seconds": args.rest_seconds,
        "initial_device_state": initial_state,
        "energy_enabled": args.energy,
        "energy_trace": str(energy_trace.local_path) if energy_trace else None,
        "idle_power_mw": idle_power_mw,
        "metric_definitions": (
            "prefill_tps = prompt_len / TTFT and decode_tps = (decode_len - 1) / (t_last_chunk - "
            "t_first_chunk), wall clock (paper definitions, same as run_autobench.py); "
            "native_prefill_tps/native_decode_tps = MNN's own prefill/decode timers."
        ),
    }

    summary = compute_summary(results)
    summary["cold_start"] = compute_cold_summary([r for r in results if r.get("phase") == "cold"])
    summary["energy_aggregate"] = bench_common.aggregate_energy(results) if args.energy else None
    if args.legacy_protocol:
        trial_summary = compute_trial_summary(results, questions) if args.trials > 1 else None
    else:
        # consistency check across each question's warm runs (runs 2..N); the headline uses the last run
        trial_summary = (compute_trial_summary([r for r in results if r.get("phase") == "warm"], questions)
                         if args.runs_per_question > 2 else None)
    save_results(args.output, run_info, summary, results, trial_summary=trial_summary)
    print_summary_table(summary, run_info)
    if trial_summary is not None:
        print_trial_summary(trial_summary)

    print("\n" + "=" * 70)
    print(f"DONE - {completed}/{len(results)} completed, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()
