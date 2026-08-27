#!/usr/bin/env python3
"""
SmolChat Automated Benchmark Pipeline.

Converts a model to GGUF (if needed), pushes it to a connected Android
device, fires headless inference runs via broadcast against SmolChat's
HeadlessBenchmarkReceiver, and collects 6 metrics per question with zero
human interaction after launch.
"""

import argparse
import importlib.util
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

PACKAGE = "io.shubham0204.smollmandroid"
BROADCAST_ACTION = "com.smollmandroid.RUN_PROMPT"
RECEIVER_COMPONENT = f"{PACKAGE}/.headless.HeadlessBenchmarkReceiver"
MAIN_ACTIVITY_COMPONENT = f"{PACKAGE}/.MainActivity"

# The app's own external files dir is exempt from scoped storage - unlike
# /sdcard/Download/, BenchmarkService can read a model pushed here without
# it having been manually imported through SmolChat's UI first (confirmed
# via a real EACCES reading from Download otherwise).
APP_FILES_DIR = f"/sdcard/Android/data/{PACKAGE}/files"

# Confirmed device-specific fallback target (Galaxy S23 / Android 16): even
# though APP_FILES_DIR above is the normal, working location on most devices
# (e.g. our OnePlus 8 Pro), some devices enforce a stricter external-storage
# restriction where the app still gets EACCES reading a file pushed there -
# confirmed genuine, not a code bug, since even "run-as <pkg> cp ..." into
# the same external directory fails with Permission denied on that device.
# The app's own INTERNAL data directory has no such restriction (the app
# owns it outright), so push_to_app_files_dir() falls back to writing here
# via a run-as pipe when the external push's readability check fails.
INTERNAL_CACHE_DIR = f"/data/user/0/{PACKAGE}/files/headless_benchmark_cache"

# Staging area for the known-affected-device path below - NOT app-owned,
# scoped, or external-storage-restricted in any way (it's the standard adb
# staging directory, universally readable/writable by the shell user), so
# it's a safe place to land the file on-device before piping it into
# INTERNAL_CACHE_DIR, without ever touching APP_FILES_DIR at all.
ADB_STAGING_DIR = "/data/local/tmp"

# Devices confirmed to have the external-storage read restriction (dd via
# run-as sometimes passes here even when the app's real read later fails -
# the probe itself is unreliable on this device, not just occasionally
# slow), keyed by (ro.product.model, ro.build.version.sdk). For these,
# push_to_app_files_dir() skips the external push + readability probe
# entirely and goes straight to the internal-storage pipe method, which has
# been reliable every time it's been used. Extend this list as further
# affected devices are confirmed.
KNOWN_AFFECTED_DEVICES = {
    ("SM-S911U", "36"),  # Galaxy S23 (US model), Android 16
}

FALLBACK_ADB = str(Path.home() / "Library/Android/sdk/platform-tools/adb")

MONSOON_SCRIPT = str(Path.home() / "SLM_Factory_Krishna_Personal/Power-Monitor/monsoon_single_reading.py")

CONVERT_SCRIPT = str(Path.home() / "SLM_Factory-SmolChat/Model-Conversion/convert_to_gguf.py")
# Fallback locations only - the real, guaranteed location is computed
# per-call in convert_to_gguf() once the model's output directory is known,
# since we now pass --output explicitly rather than relying on the tool's
# own "./output" default.
CONVERSION_REPORT_CANDIDATES = [
    str(Path.home() / "SLM_Factory_Krishna_Personal/Model-Conversion/conversion_report.json"),
    str(Path.cwd() / "conversion_report.json"),
]

CONTEXT_SIZE_PHRASE = "context size reached"

COLD_BOOT_SETTLE_SECONDS = 110

# How long a given reboot takes to fully settle varies run to run (confirmed
# via 4 clean --reboot-before runs: 2/4 had the first broadcast received
# reliably, 2/4 never received it at all - same code/model, no manual
# interference) - so the FIRST broadcast after a reboot specifically retries
# until BROADCAST_RECEIVER confirms pickup, rather than assuming one attempt
# is enough. Every other question, and every question in a non-reboot run,
# is unaffected by these constants.
#
# COLD_BOOT_SETTLE_SECONDS was raised from 60 to 110: with a 60s settle, the
# retry loop consistently needed all 5 attempts to succeed (~8s poll + 3s
# delay per attempt, so ~50s spent retrying on top of the 60s settle) -
# meaning the actual settle time needed is closer to 60+50=110s, not lower.
# Retries remain as a safety net for run-to-run variance beyond this baseline.
BROADCAST_RECEIPT_POLL_SECONDS = 8
BROADCAST_RECEIPT_MAX_ATTEMPTS = 5
BROADCAST_RECEIPT_RETRY_DELAY_SECONDS = 3

COLD_LOAD_NOTE = (
    "cold_load_ms reflects a genuinely cold read only if --reboot-before was used for this run. "
    "Without a reboot, the OS file cache may make cold_load_ms appear faster than a true cold read "
    "from storage, especially for larger model files."
)

# Set from args.quiet at the top of main(). qprint() (used for routine
# progress/informational output, including the final "DONE" summary line)
# checks this; [ERROR]/[WARN] lines and "Results saved" always use plain
# print() regardless, so --quiet suppresses routine noise without hiding
# real problems or where the output file ended up.
_QUIET = False


def qprint(*args, **kwargs):
    if not _QUIET:
        print(*args, **kwargs)


# Set True the first time warm_up_app_once() actually runs, so it only
# fires once per script invocation regardless of how many times
# push_to_app_files_dir() is called (once per model in a multi-model run).
_APP_WARMED_UP = False


def warm_up_app_once(adb: Adb):
    """One-time workaround for a confirmed Android 16 / Galaxy S23 quirk: a
    freshly-installed app cannot reliably read ANY file pushed to its
    external files directory - even a run-as read attempt right after push
    reports success, but the app's own real process still gets EACCES
    moments later. The only thing that reliably fixes this is manually
    opening the app's UI once through the launcher; this automates that.

    Deliberately launched via the LAUNCHER-category monkey intent, not
    "am start -n <component>" - reset_smolchat_for_clean_process() already
    does that immediately before every push, and the bug still occurs, so
    that alone is confirmed NOT sufficient. Only a genuine launcher-
    initiated open appears to trigger whatever permission grant Android is
    otherwise withholding.

    Cached via _APP_WARMED_UP: runs at most once per script invocation, not
    before every push. Harmless on unaffected devices (e.g. our OnePlus 8
    Pro) even run unconditionally - caching it is purely for efficiency
    there, not correctness.
    """
    global _APP_WARMED_UP
    if _APP_WARMED_UP:
        return

    qprint(f"[WARMUP] Launching {PACKAGE} via the launcher once (Android 16/Galaxy S23 external-storage permission workaround)...")
    adb.run(["shell", "monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1"], timeout=20)
    time.sleep(3)
    adb.run(["shell", "am", "force-stop", PACKAGE], timeout=15)
    _APP_WARMED_UP = True


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
            # encoding/errors instead of text=True: logcat buffers can
            # contain invalid UTF-8 (more likely now that poll_for_result
            # dumps the full unfiltered buffer), and text=True's strict
            # decoding crashes the whole process on the first bad byte.
            # errors="replace" substitutes instead.
            #
            # stdin=DEVNULL: none of our adb calls ever need to read from
            # stdin. Without this, subprocess.run() leaves stdin inherited
            # from this process - fine when run_autobench.py itself has a
            # real TTY, but when it's invoked as a subprocess by another
            # script (capture_output=True, no stdin= set there either),
            # that inherited descriptor is a non-TTY pipe. "adb shell ..."
            # specifically negotiates PTY/stdin duplexing based on whether
            # its own stdin looks like a terminal, and can hang on that
            # negotiation before ever dispatching the command to the
            # device when stdin is an ambiguous inherited pipe - confirmed
            # as the cause of broadcasts never firing under exactly that
            # nested-subprocess condition. An explicit null stdin removes
            # the ambiguity entirely.
            return subprocess.run(
                self.base + args, capture_output=True,
                encoding="utf-8", errors="replace", timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
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
    qprint(f"[OK] {device_kind.capitalize()} connected via adb {adb.flag}")


def check_smolchat_installed(adb: Adb):
    result = adb.run(["shell", "pm", "list", "packages"], timeout=15)
    if PACKAGE not in result.stdout:
        print(f"[ERROR] SmolChat ({PACKAGE}) is not installed on the target device.")
        sys.exit(1)
    qprint(f"[OK] SmolChat ({PACKAGE}) is installed")


def print_thermal_reminder():
    qprint(
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
        qprint(f"[OK] Battery at {level}% ({status_name}) -- Power readings should be trustworthy")

    return {"battery_warning": warning, "battery_level_pct": level, "battery_status": status_name}


# ---------------------------------------------------------------------------
# Model resolution / conversion / deploy
# ---------------------------------------------------------------------------

def _get_device_model_and_sdk(adb: Adb) -> tuple:
    model_result = adb.run(["shell", "getprop", "ro.product.model"], timeout=10)
    sdk_result = adb.run(["shell", "getprop", "ro.build.version.sdk"], timeout=10)
    return (model_result.stdout or "").strip(), (sdk_result.stdout or "").strip()


def _is_known_affected_device(adb: Adb) -> bool:
    """Whether this device is in KNOWN_AFFECTED_DEVICES - confirmed to have
    an external-storage read restriction that the dd-based readability probe
    (_app_can_read()) doesn't reliably catch on its own (sometimes passes
    right after push even though the app's real read later fails)."""
    model, sdk = _get_device_model_and_sdk(adb)
    return (model, sdk) in KNOWN_AFFECTED_DEVICES


def _app_can_read(adb: Adb, device_path: str) -> bool:
    """Whether PACKAGE can actually read device_path - checked via a real
    read attempt ("run-as <pkg> dd if=<path> of=/dev/null bs=1 count=1"),
    not "test -r". Confirmed real false-positive on a Galaxy S23: test -r
    only checks Unix permission bits and reported success even though the
    app's actual FileInputStream still got EACCES - the real restriction
    (likely FUSE/SELinux enforcement specific to external storage) only
    triggers on an actual read() syscall. Reading just 1 byte (count=1)
    keeps this fast and harmless on devices where it already works fine.
    """
    result = adb.run(
        ["shell", "run-as", PACKAGE, "dd", f"if={device_path}", "of=/dev/null", "bs=1", "count=1"],
        timeout=15,
    )
    return result.returncode == 0


def _push_via_internal_pipe(adb: Adb, source_device_path: str, filename: str) -> str:
    """Pipe a file that's already somewhere on-device (source_device_path,
    readable by adb's own shell user) into the app's internal storage,
    which it owns outright and is never subject to the external-storage
    restriction. Piping "cat <source>" into "run-as <pkg> sh -c 'cat >
    ...'" re-writes those bytes as the app's own UID - no re-transfer from
    the host needed. Built as one shell-quoted string (not separate argv
    elements) for the same reason fire_broadcast() is: "adb shell
    <args...>" rejoins separate elements into one remote command anyway, so
    any unescaped metacharacter in a path would corrupt the pipeline -
    shlex.quote() on each piece avoids that regardless of what the
    filename/paths happen to contain. Exits on failure; returns the
    internal path on success (verified actually readable first).
    """
    internal_path = f"{INTERNAL_CACHE_DIR}/{filename}"
    adb.run(["shell", "run-as", PACKAGE, "mkdir", "-p", INTERNAL_CACHE_DIR], timeout=15)
    pipe_cmd = (
        f"cat {shlex.quote(source_device_path)} | run-as {shlex.quote(PACKAGE)} "
        f"sh -c {shlex.quote('cat > ' + internal_path)}"
    )
    pipe_result = adb.run(["shell", pipe_cmd], timeout=600)
    if pipe_result.returncode != 0:
        print(f"[ERROR] Internal-storage push failed: {pipe_result.stderr}")
        sys.exit(1)

    if not _app_can_read(adb, internal_path):
        print(f"[ERROR] Internal-storage push completed but {PACKAGE} still cannot read {internal_path}")
        sys.exit(1)

    return internal_path


def push_to_app_files_dir(adb: Adb, local_path) -> str:
    """Push a local GGUF file to wherever BenchmarkService can actually
    read it from.

    Shared by both model-resolution branches (a local .gguf path, and a
    HuggingFace ID that gets converted first) so this is the one place that
    decides where the app can read a model from - unlike /sdcard/Download/,
    APP_FILES_DIR is exempt from scoped storage and doesn't require the
    model to have been manually imported through SmolChat's UI first
    (confirmed via a real EACCES otherwise).
    """
    local_path = os.path.expanduser(str(local_path))
    if not os.path.exists(local_path):
        print(f"[ERROR] GGUF file not found: {local_path}")
        sys.exit(1)

    warm_up_app_once(adb)

    filename = os.path.basename(local_path)

    # Confirmed device-specific gap (Galaxy S23 / Android 16): on this
    # device, the dd-based readability probe below is itself unreliable -
    # it's sometimes reported success right after push even though the
    # app's real read later still fails, so it can't be trusted to catch
    # every case. Rather than keep relying on a probe known to be flaky on
    # this device, skip the external push + probe entirely and go straight
    # to the internal-storage pipe method, which has been reliable every
    # time it's been used. Staged through ADB_STAGING_DIR (not
    # APP_FILES_DIR) since that's a generic, non-scoped location the shell
    # user can always read/write, letting this avoid APP_FILES_DIR
    # completely rather than just skipping a check against it.
    if _is_known_affected_device(adb):
        staged_path = f"{ADB_STAGING_DIR}/{filename}"
        qprint(f"\n[DEPLOY] Known-affected device detected - staging {local_path} -> {staged_path} ...")
        result = adb.run(["push", local_path, staged_path], timeout=600)
        if result.returncode != 0:
            print(f"[ERROR] adb push failed: {result.stderr}")
            sys.exit(1)
        print(
            f"[WARN] {PACKAGE} - known-affected device (external-storage read restriction confirmed "
            "unreliable to probe) - going straight to the internal-storage pipe method, skipping the "
            "external push and readability check entirely."
        )
        internal_path = _push_via_internal_pipe(adb, staged_path, filename)
        print(f"[OK] Internal-storage push succeeded. Device path: {internal_path}")
        return internal_path

    # Everything below is unchanged for all other devices: external push
    # first, verify readability, and only fall back to internal storage if
    # that verification actually fails.
    device_path = f"{APP_FILES_DIR}/{filename}"
    adb.run(["shell", "mkdir", "-p", APP_FILES_DIR], timeout=15)
    qprint(f"\n[DEPLOY] Pushing {local_path} -> {device_path} ...")
    result = adb.run(["push", local_path, device_path], timeout=600)
    if result.returncode != 0:
        print(f"[ERROR] adb push failed: {result.stderr}")
        sys.exit(1)
    qprint(f"[OK] Push complete. Device path: {device_path}")

    # Confirmed device-specific gap (Galaxy S23 / Android 16): the push
    # above can succeed while BenchmarkService still gets EACCES reading the
    # result - a genuine per-device external-storage restriction, not a
    # code bug (even "run-as <pkg> cp ..." into the same directory fails
    # with Permission denied there). Verify the app can actually read what
    # was just pushed before trusting device_path; unaffected devices (e.g.
    # our OnePlus 8 Pro) pass this check every time and return exactly as
    # before, with no other behavior change.
    if _app_can_read(adb, device_path):
        return device_path

    print(
        f"[WARN] {PACKAGE} cannot read {device_path} despite a successful push "
        "(confirmed device-specific external-storage restriction) - falling back "
        "to the app's internal storage via a run-as pipe..."
    )
    internal_path = _push_via_internal_pipe(adb, device_path, filename)
    print(f"[OK] Internal-storage fallback succeeded. Device path: {internal_path}")
    return internal_path


def _load_convert_module():
    """Load convert_to_gguf.py by file path so its model_prefix() - the only
    deterministic naming logic it exposes - can be called directly rather
    than reimplemented here, so this can't silently drift out of sync if
    that logic ever changes. (The output *directory* has no such logic in
    that script at all - --output is just a caller-supplied string, which
    is exactly why the report went missing: nothing here ever told it
    where to write, so it fell back to "./output".)
    """
    spec = importlib.util.spec_from_file_location("_convert_to_gguf_external", CONVERT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_to_gguf(model_id: str, quant: str, adb: Adb) -> str:
    qprint(f"\n[CONVERSION] Converting {model_id} to GGUF ({quant})...")
    if not os.path.exists(CONVERT_SCRIPT):
        print(f"[ERROR] Conversion script not found: {CONVERT_SCRIPT}")
        sys.exit(1)

    convert_module = _load_convert_module()
    if not hasattr(convert_module, "model_prefix"):
        print(f"[ERROR] {CONVERT_SCRIPT} no longer exposes model_prefix() - cannot compute a deterministic output directory")
        sys.exit(1)
    prefix = convert_module.model_prefix(model_id)

    # Compute (and pass explicitly via --output) the same "output-<prefix>"
    # directory agent_quantize.py already uses, so the report's location is
    # guaranteed rather than left to the tool's own "./output" default.
    output_dir = Path(CONVERT_SCRIPT).parent / f"output-{prefix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --deploy makes the external script push its own copy to
    # /sdcard/Download/ - harmless but redundant now, since we push our own
    # copy to APP_FILES_DIR below (the location BenchmarkService can
    # actually read without the model having been manually imported first).
    cmd = [sys.executable, CONVERT_SCRIPT, "--model", model_id, "--output", str(output_dir), "--quant", quant, "--deploy"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Conversion failed for {model_id} (exit code {result.returncode})")
        sys.exit(1)

    report_candidates = [str(output_dir / "conversion_report.json")] + CONVERSION_REPORT_CANDIDATES
    report_path = next((p for p in report_candidates if os.path.exists(p)), None)
    if not report_path:
        print(f"[ERROR] conversion_report.json not found in any of: {report_candidates}")
        sys.exit(1)

    try:
        with open(report_path) as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[ERROR] Failed to read conversion report {report_path}: {e}")
        sys.exit(1)

    # output_files maps quant level -> filename; the file lives alongside
    # the report itself, in the same output directory.
    filename = report.get("output_files", {}).get(quant)
    if not filename:
        print(f"[ERROR] '{quant}' missing from output_files in conversion report: {report}")
        sys.exit(1)

    local_gguf_path = Path(report_path).parent / filename
    if not local_gguf_path.exists():
        print(f"[ERROR] Converted GGUF not found on disk: {local_gguf_path}")
        sys.exit(1)

    qprint(f"[OK] Conversion complete: {local_gguf_path}")
    return push_to_app_files_dir(adb, local_gguf_path)


def resolve_model(model_arg: str, quant: str, adb: Adb) -> tuple:
    """Returns (device_path, display_model_name)."""
    expanded = os.path.expanduser(model_arg)
    is_local_gguf = model_arg.lower().endswith(".gguf") or os.path.isfile(expanded)

    if is_local_gguf:
        device_path = push_to_app_files_dir(adb, model_arg)
        return device_path, os.path.basename(expanded)

    if "/" in model_arg:
        device_path = convert_to_gguf(model_arg, quant, adb)
        return device_path, model_arg

    print(f"[ERROR] --model '{model_arg}' is neither a local .gguf path nor a HuggingFace model ID (no '/').")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

def load_questions(questions_path) -> list:
    if not questions_path:
        qprint(f"[OK] Using {len(DEFAULT_QUESTIONS)} built-in default questions")
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
    qprint(f"[OK] Loaded {len(questions)} questions from {path}")
    return questions


# ---------------------------------------------------------------------------
# Broadcast / logcat plumbing
# ---------------------------------------------------------------------------

def clear_logcat(adb: Adb):
    adb.run(["logcat", "-c"], timeout=15)


def fire_broadcast(adb: Adb, model_path: str, question: str, run_id: str, max_tokens: int):
    # CONFIRMED BUG (real A/B test: apostrophes in the prompt text produced
    # response=None/near-instant EOS; the exact same question with
    # apostrophes stripped worked fine): passing model_path/prompt/run_id
    # as SEPARATE argv elements here means "adb shell <args...>" re-joins
    # them into ONE remote command string before it reaches the device's
    # shell. The old question.replace(" ", "_") only protected against
    # word-splitting on spaces - it did nothing for apostrophes (or other
    # shell metacharacters: $, `, ", \, ;, &, |, (, )). An unescaped
    # apostrophe in the rejoined remote command opens an unterminated
    # quote, corrupting/truncating everything after it before "am
    # broadcast" ever sees it. Building the full command as a single,
    # already shell-quoted string - the same fix already proven for
    # run_mnn_autobench.py's own fire_broadcast() - sidesteps the rejoining
    # hazard entirely: real spaces and apostrophes both survive intact, so
    # the underscore-encoding hack is no longer needed at all.
    cmd = (
        f"am broadcast -a {BROADCAST_ACTION} -n {RECEIVER_COMPONENT} "
        f"--es model_path {shlex.quote(model_path)} "
        f"--es prompt {shlex.quote(question)} "
        f"--es run_id {shlex.quote(run_id)} "
        f"--ei max_tokens {int(max_tokens)}"
    )
    adb.run(["shell", cmd], timeout=20)


KNOWN_TAGS = ["RUN_DONE", "RUN_ERROR", "BROADCAST_RECEIVER", "COLD_LOAD", "TTFT", "TPS", "MEMORY", "POWER", "THERMAL_TEMP_CPU", "THERMAL_TEMP_SKIN", "THERMAL"]


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
    """Poll logcat every 1s until RUN_DONE/RUN_ERROR for run_id or timeout. Returns (status, lines).

    Deliberately unfiltered ("adb logcat -d" with no -s tag list): confirmed
    via repeated manual testing that server-side tag filtering sometimes
    misses lines that are demonstrably present in the same buffer dump at
    the same moment - a plain dump found them every time the filtered one
    reported a timeout. All filtering happens Python-side in
    parse_run_lines(), which searches raw text for run_id= and tag matches
    and has no dependency on the input being pre-filtered by tag.

    "-b main" reads only the "main" on-device buffer - confirmed that's
    where Android app Log.d() calls (i.e. every one of our custom tags)
    actually land. An earlier "-b all" attempt confirmed the opposite
    problem: it pulled in kernel/perf/radio/events noise (~1MB+, including
    raw kernel boot messages) that's irrelevant to our tags and never
    contains run_id, while still not finding it - so "-b all" wasn't
    the fix. Scoping to "main" cuts the dump back down to just the
    buffer that can possibly contain what we're looking for.
    """
    # TEMP DEBUG (grep "[DEBUG-POLL]" to find/remove all of it): diagnosing
    # repeated --reboot-before-only poll timeouts where manual post-hoc
    # checks confirm RUN_DONE is genuinely in the buffer. Need to see what
    # each poll's own "adb logcat -d" call actually returns in real time -
    # empty, truncated, identical stale content every cycle, or something
    # else - rather than continuing to guess from retrospective checks.
    start_time = time.time()
    iteration = 0
    previous_raw = None

    deadline = time.time() + timeout
    last_lines = {}
    while time.time() < deadline:
        iteration += 1
        result = adb.run(["logcat", "-d", "-b", "main"], timeout=30)
        raw = result.stdout or ""

        elapsed = time.time() - start_time
        same_as_previous = raw == previous_raw
        run_id_present = run_id in raw
        # print(
        #     f"[DEBUG-POLL] iter={iteration} elapsed={elapsed:.1f}s len={len(raw)} "
        #     f"returncode={result.returncode} same_as_prev_dump={same_as_previous}"
        # )
        # print(f"[DEBUG-POLL] run_id={run_id!r} run_id_in_raw_text={run_id_present}")
        # print(f"[DEBUG-POLL] first200={raw[:200]!r}")
        previous_raw = raw

        last_lines = parse_run_lines(raw, run_id)
        if "RUN_DONE" in last_lines:
            return "done", last_lines
        if "RUN_ERROR" in last_lines:
            return "error", last_lines
        time.sleep(1)
    return "timeout", last_lines


def restart_smolchat(adb: Adb):
    qprint("  [RESTART] force-stopping and relaunching SmolChat...")
    adb.run(["shell", "am", "force-stop", PACKAGE], timeout=15)
    adb.run(["shell", "am", "start", "-n", MAIN_ACTIVITY_COMPONENT], timeout=15)
    time.sleep(3)


def reboot_device_for_cold_load(adb: Adb):
    """Reboot the device so cold_load_ms reflects a genuine cold read from
    storage rather than one warmed by the OS file cache from a prior run.

    "device" state from wait-for-device precedes the home screen and system
    services actually being ready, so we sleep an extra settle period on top.
    Confirmed via manual testing: even after that settle period, adb logcat
    itself can still be unattached to the device's logging daemon while the
    device/app are otherwise fully responsive - a real inference completed
    in ~2s but poll_for_result still timed out at 60s because logcat wasn't
    capturing yet. A throwaway "adb logcat -d" here forces that connection
    to establish before the real per-question polling begins.

    Confirmed via poll_for_result's own debug logging (raw dump growing
    ~30-50KB/sec of pure boot-time system noise - WiFi state machine,
    package manager timing, etc.) that this noise can evict our target
    run_id line from the default-sized ring buffer before any poll cycle
    catches it - a genuine buffer overrun, not a timing or parsing bug.
    Resizing the buffer up front gives our line far more room to survive.

    Further confirmed via the static "--------- beginning of perf" marker
    staying first in every dump while total length kept growing: "adb
    logcat -d" combines multiple separate on-device buffers (main, system,
    perf, crash, kernel, ...). A subsequent "-b all" attempt confirmed the
    opposite problem - it pulled in ~1MB+ of kernel/perf/radio/events
    noise (raw kernel boot messages included) that's irrelevant to our
    tags, and still never found run_id. Android app Log.d() calls (i.e.
    every one of our custom tags) land specifically in the "main" buffer,
    so both the resize and poll_for_result's dump now target "-b main"
    only - the smallest scope that can actually contain what we want.
    """
    print("[COLD] Rebooting device for genuine cold-load measurement (this takes ~2 minutes)...")
    adb.run(["reboot"], timeout=30)
    adb.run(["wait-for-device"], timeout=180)
    time.sleep(COLD_BOOT_SETTLE_SECONDS)

    print("[COLD] Resizing on-device logcat main buffer (where our app's Log.d() tags land) to reduce post-reboot ring-buffer eviction risk...")
    resize_result = adb.run(["shell", "logcat", "-G", "16M", "-b", "main"], timeout=20)
    resize_output = (resize_result.stdout or "").strip() or (resize_result.stderr or "").strip() or "(no output)"
    print(f"[COLD] logcat -G 16M -b main -> returncode={resize_result.returncode} output={resize_output!r}")
    if resize_result.returncode != 0:
        print(
            "[COLD] WARNING: logcat -G 16M -b main returned a non-zero exit code - the buffer resize "
            "may NOT have taken effect (this device/OS build might not support '-b main' on -G). "
            "Proceeding anyway, but ring-buffer eviction is more likely for the rest of this run."
        )

    adb.run(["logcat", "-d", "-b", "main"], timeout=20)


def reset_smolchat_for_clean_process(adb: Adb):
    """One-time reset at script startup so RSS isn't contaminated by a
    high-water mark left over from a model loaded in a prior, separate
    invocation of this script (see run_benchmark's mid-run context-size
    reset for the unrelated, per-question retry logic)."""
    qprint("[RESET] Force-stopping and restarting SmolChat for a clean process state (required for accurate Peak RSS)...")
    adb.run(["shell", "am", "force-stop", PACKAGE], timeout=15)
    adb.run(["shell", "am", "start", "-n", MAIN_ACTIVITY_COMPONENT], timeout=15)
    time.sleep(4)


def run_one(adb: Adb, model_path: str, question: str, n: int, timeout: int, max_tokens: int) -> dict:
    run_id = f"run_{n}_{int(time.time() * 1000)}"
    clear_logcat(adb)
    fire_broadcast(adb, model_path, question, run_id, max_tokens)
    status, lines = poll_for_result(adb, run_id, timeout)
    return {"run_id": run_id, "status": status, "lines": lines}


def wait_for_broadcast_receipt(adb: Adb, run_id: str, poll_seconds: int) -> bool:
    """Poll logcat's main buffer briefly for BROADCAST_RECEIVER's own
    'received run_id=...' line - proof the app's receiver actually picked
    up the broadcast, well before full inference completion (which
    poll_for_result checks for separately, on its own timeout)."""
    deadline = time.time() + poll_seconds
    while time.time() < deadline:
        result = adb.run(["logcat", "-d", "-b", "main"], timeout=30)
        lines = parse_run_lines(result.stdout or "", run_id)
        if "BROADCAST_RECEIVER" in lines:
            return True
        time.sleep(1)
    return False


def run_first_question_after_reboot(adb: Adb, model_path: str, question: str, n: int, timeout: int, max_tokens: int) -> dict:
    """Fire the first broadcast after a --reboot-before reboot, retrying
    with the SAME run_id if BROADCAST_RECEIVER never confirms pickup
    within a short window.

    Only used for question 1 of a --reboot-before run - every other
    question, and every question in a non-reboot run, goes through
    run_one() directly and is unaffected by this retry behavior.
    """
    run_id = f"run_{n}_{int(time.time() * 1000)}"
    received = False

    for attempt in range(1, BROADCAST_RECEIPT_MAX_ATTEMPTS + 1):
        clear_logcat(adb)
        fire_broadcast(adb, model_path, question, run_id, max_tokens)
        if wait_for_broadcast_receipt(adb, run_id, BROADCAST_RECEIPT_POLL_SECONDS):
            print(f"[COLD] Broadcast received on attempt {attempt}/{BROADCAST_RECEIPT_MAX_ATTEMPTS}")
            received = True
            break
        if attempt < BROADCAST_RECEIPT_MAX_ATTEMPTS:
            print(
                f"[COLD] Broadcast not received within {BROADCAST_RECEIPT_POLL_SECONDS}s "
                f"(attempt {attempt}/{BROADCAST_RECEIPT_MAX_ATTEMPTS}) - retrying with the same run_id..."
            )
            time.sleep(BROADCAST_RECEIPT_RETRY_DELAY_SECONDS)

    if not received:
        print(
            f"[COLD] ERROR: broadcast was never received after {BROADCAST_RECEIPT_MAX_ATTEMPTS} attempts. "
            "Proceeding to poll for RUN_DONE anyway, but it will likely time out."
        )

    status, lines = poll_for_result(adb, run_id, timeout)
    return {"run_id": run_id, "status": status, "lines": lines}


def get_monsoon_power(duration_seconds=5):
    """One Monsoon HVPM power reading via monsoon_single_reading.py, for
    side-by-side comparison against the existing BatteryManager-based
    power_ma metric. Never raises - any failure returns power_ma_mean=None
    with an error description instead of crashing the calling script.

    Uses sys.executable (not a hardcoded "python3") and encoding="utf-8",
    errors="replace" (not text=True) rather than the originally-proposed
    snippet verbatim - both match Adb.run()'s and convert_to_gguf()'s own
    established conventions elsewhere in this file (the text=True ->
    encoding/errors switch in particular was a confirmed fix for a real
    UnicodeDecodeError crash, not a stylistic choice).
    """
    try:
        result = subprocess.run(
            [sys.executable, MONSOON_SCRIPT, "--duration-seconds", str(duration_seconds)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        )
        return json.loads(result.stdout)
    except Exception as e:
        return {"power_ma_mean": None, "error": str(e)}


def build_metrics(lines: dict) -> dict:
    def num(tag, cast):
        v = extract_value(lines[tag]) if tag in lines else None
        if v is None:
            return None
        try:
            return cast(v)
        except ValueError:
            return None

    monsoon = get_monsoon_power()

    return {
        "cold_load_ms": num("COLD_LOAD", int),
        "ttft_ms": num("TTFT", int),
        "tps": num("TPS", float),
        "memory_kb": num("MEMORY", int),
        "power_ma": num("POWER", float),
        "thermal": extract_value(lines["THERMAL"]) if "THERMAL" in lines else None,
        # "unavailable" fails the float() cast and num() already returns None
        # on ValueError, which is exactly the "unknown reading" value we want.
        "thermal_temp_cpu_c": num("THERMAL_TEMP_CPU", float),
        "thermal_temp_skin_c": num("THERMAL_TEMP_SKIN", float),
        # Independent reading, captured alongside power_ma (BatteryManager)
        # rather than replacing it - both coexist side by side.
        "power_ma_monsoon": monsoon.get("power_ma_mean"),
    }


# ---------------------------------------------------------------------------
# Main per-question loop
# ---------------------------------------------------------------------------

def run_benchmark(adb: Adb, model_path: str, questions: list, timeout: int, max_tokens: int, reboot_before: bool = False) -> tuple:
    results = []
    context_resets = 0
    total = len(questions)

    for n, question in enumerate(questions, start=1):
        if n == 1 and reboot_before:
            outcome = run_first_question_after_reboot(adb, model_path, question, n, timeout, max_tokens)
        else:
            outcome = run_one(adb, model_path, question, n, timeout, max_tokens)
        status, lines, run_id = outcome["status"], outcome["lines"], outcome["run_id"]
        context_reset_for_this_q = False

        if status == "error":
            reason, message = extract_error(lines.get("RUN_ERROR", ""))
            if message and CONTEXT_SIZE_PHRASE in message.lower():
                qprint(f"\n[{n}/{total}] \"{question}\"")
                qprint(f"  CONTEXT WINDOW FULL (reason={reason}, message={message}) -- restarting SmolChat and retrying once")
                restart_smolchat(adb)
                context_resets += 1
                context_reset_for_this_q = True
                outcome = run_one(adb, model_path, question, n, timeout, max_tokens)
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

            temp_parts = []
            if metrics["thermal_temp_cpu_c"] is not None:
                temp_parts.append(f"CPU:{metrics['thermal_temp_cpu_c']}°C")
            if metrics["thermal_temp_skin_c"] is not None:
                temp_parts.append(f"Skin:{metrics['thermal_temp_skin_c']}°C")
            temp_suffix = f" ({' '.join(temp_parts)})" if temp_parts else ""
            qprint(f"\n[{n}/{total}] \"{question}\"")
            qprint(
                f"  ColdLoad={metrics['cold_load_ms']}ms TTFT={metrics['ttft_ms']}ms TPS={metrics['tps']} "
                f"RSS={metrics['memory_kb']}KB Power={metrics['power_ma']}mA Thermal={metrics['thermal']}{temp_suffix}"
            )
            preview = response if response and len(response) <= 160 else (response[:157] + "..." if response else "")
            qprint(f"  Response: \"{preview}\"")
        elif status == "error":
            reason, message = extract_error(lines.get("RUN_ERROR", ""))
            entry["status"] = "failed"
            entry["error"] = {"reason": reason, "message": message}
            qprint(f"\n[{n}/{total}] \"{question}\"")
            qprint(f"  FAILED: reason={reason} message={message}")
        else:  # timeout
            entry["status"] = "failed"
            entry["error"] = {"reason": "timeout", "message": f"No RUN_DONE/RUN_ERROR within {timeout}s"}
            qprint(f"\n[{n}/{total}] \"{question}\"")
            qprint(f"  FAILED: timeout after {timeout}s")

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
        "thermal_temp_cpu_c": stat_block(vals("thermal_temp_cpu_c")),
        "thermal_temp_skin_c": stat_block(vals("thermal_temp_skin_c")),
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
    qprint("\n" + "=" * 60)
    qprint("SUMMARY")
    qprint("=" * 60)
    header = f"{'Metric':<10}{'Mean':>12}{'Std':>12}{'Min':>12}{'Max':>12}"
    qprint(header)
    qprint("-" * len(header))
    for label, key in [("TTFT_ms", "ttft_ms"), ("TPS", "tps"), ("RSS_KB", "memory_kb"), ("Power_mA", "power_ma"), ("ThermalCPU_C", "thermal_temp_cpu_c"), ("ThermalSkin_C", "thermal_temp_skin_c")]:
        s = summary[key]
        row = f"{label:<10}" + "".join(
            f"{(s[k] if s[k] is not None else 'N/A'):>12}" for k in ("mean", "std", "min", "max")
        )
        qprint(row)
    cold_load_q1 = summary.get("first_question_cold_load_ms")
    qprint(f"{'ColdLoad_Q1':<10}{(cold_load_q1 if cold_load_q1 is not None else 'N/A'):>12}{'N/A':>12}{'N/A':>12}{'N/A':>12}")
    qprint(f"\nThermal states observed: {', '.join(summary['thermal_states_observed']) or 'none'}")
    qprint(summary["note"])
    qprint(summary["rss_note"])
    qprint(f"{run_info['cold_load_note']} (rebooted_before_run={run_info['rebooted_before_run']})")


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
    p.add_argument("--timeout", type=int, default=180, help="Seconds to wait for RUN_DONE/RUN_ERROR per question")
    p.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens",
                    help="Max tokens the receiver should generate per response (matches MNN's default)")
    p.add_argument("--reboot-before", action="store_true",
                    help="Reboot the device before benchmarking for a genuine cold-load read (adds ~30-60s)")
    p.add_argument("--quiet", action="store_true",
                    help="Suppress routine per-question and pre-flight progress output. [ERROR]/[WARN] lines "
                         "and the final 'Results saved'/'DONE' confirmation are still always printed.")
    return p.parse_args()


def main():
    args = parse_args()

    global _QUIET
    _QUIET = args.quiet

    qprint("=" * 60)
    qprint("SmolChat Automated Benchmark Pipeline")
    qprint(f"  Model: {args.model}  Device: {args.device}  Quant: {args.quant}  Timeout: {args.timeout}s  MaxTokens: {args.max_tokens}")
    qprint("=" * 60)

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
    results, context_resets = run_benchmark(adb, model_path, questions, args.timeout, args.max_tokens, reboot_before=args.reboot_before)
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

    qprint("\n" + "=" * 60)
    qprint(f"DONE - {completed}/{len(questions)} completed, {failed} failed, {context_resets} context reset(s)")
    qprint("=" * 60)


if __name__ == "__main__":
    main()
