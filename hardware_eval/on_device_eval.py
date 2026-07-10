# hardware_eval/on_device_eval.py
"""
On-device hardware evaluation — the single source of truth for latency, memory,
throughput, and power measurement of Android-deployable SLMs.

This module unifies what used to be three disconnected pieces:
  - hardware_eval/on_device_eval.py  (a synthetic stub the pipeline called)
  - training/on_device_eval.py       (real llama-cli/ADB logic that nothing called)
  - hardware_eval/run_autobench.py   (real SmolChat/logcat metrics, CLI-only)

The metric-gathering logic from run_autobench now lives here as an importable
engine; run_autobench.py is a thin CLI wrapper around it.

Backends (choose via env SLM_HW_BACKEND, default "theoretical"):
  - "theoretical"  : estimates from ModelSpec benchmarks. No hardware. Always the
                     fallback when no GGUF is available (e.g. the pre-training
                     hardware_filter, which screens base models before any GGUF
                     exists). This keeps the pipeline runnable on a bare GPU node.
  - "llama_cpp"    : local llama-cli timing run against a GGUF (no phone needed).
  - "adb_llama"    : push GGUF + run llama-cli on a connected device via ADB.
  - "smolchat"     : broadcast to the SmolChat app's HeadlessBenchmarkReceiver and
                     scrape logcat — the richest path (TTFT/TPS/RSS/power/thermal).

Every measurement function is DEFENSIVE: missing adb / missing binary / missing
device / timeout returns HardwareEvalResult(success=False, error=...) — it never
raises, so a measurement failure can never crash the fine-tuning loop.
"""
from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config knobs (read from env directly so this module never imports config.py,
# which hard-requires ANTHROPIC_API_KEY/EXA_API_KEY at import time). run.py and
# config.py surface the same env vars for central documentation.
# ---------------------------------------------------------------------------

DEFAULT_BACKEND = os.environ.get("SLM_HW_BACKEND", "theoretical")

# Nominal Li-ion battery voltage used to convert BatteryManager current (mA) to
# power (W) when the device's live voltage can't be read from dumpsys.
DEFAULT_BATTERY_VOLTAGE_V = float(os.environ.get("SLM_HW_BATTERY_VOLTAGE_V", "3.85"))

# SmolChat integration constants (must match the installed SmolChat build).
SMOLCHAT_PACKAGE = os.environ.get("SLM_SMOLCHAT_PACKAGE", "io.shubham0204.smollmandroid")
BROADCAST_ACTION = "com.smollmandroid.RUN_PROMPT"
RECEIVER_COMPONENT = f"{SMOLCHAT_PACKAGE}/.headless.HeadlessBenchmarkReceiver"
MAIN_ACTIVITY_COMPONENT = f"{SMOLCHAT_PACKAGE}/.MainActivity"

LOGCAT_TAG_FILTER = [
    "COLD_LOAD:D", "TTFT:D", "TPS:D", "MEMORY:D",
    "POWER:D", "THERMAL:D", "RUN_DONE:D", "RUN_ERROR:D", "*:S",
]
KNOWN_TAGS = ["RUN_DONE", "RUN_ERROR", "COLD_LOAD", "TTFT", "TPS", "MEMORY", "POWER", "THERMAL"]
CONTEXT_SIZE_PHRASE = "context size reached"

DEFAULT_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "What is the largest planet in our solar system?",
    "Explain the process of photosynthesis in plants.",
]


# ---------------------------------------------------------------------------
# Canonical result type
# ---------------------------------------------------------------------------

@dataclass
class HardwareEvalResult:
    """Measured (or estimated) on-device performance of a model variant.

    Field names ttft_ms / tok_per_s / avg_watts / peak_memory_mb are the contract
    consumed by config.android_pool.check_hardware_constraints via to_measured().
    """
    model_id: str
    success: bool
    ttft_ms: float | None = None          # time-to-first-token (prefill latency), ms
    tok_per_s: float | None = None        # sustained decode throughput
    avg_watts: float | None = None        # average power during inference (None = not measured)
    peak_memory_mb: int | None = None     # peak RAM during inference
    cold_load_ms: int | None = None       # cold model-load time (first question only)
    thermal: str | None = None            # device thermal state label, if reported
    eval_method: str = "theoretical"      # theoretical | llama_cpp | adb_llama | smolchat
    device: str | None = None             # device serial/kind the measurement ran on
    error: str | None = None
    raw: dict = field(default_factory=dict)  # backend-specific extras (per-question stats, etc.)

    def to_measured(self) -> dict:
        """Map to the `measured` dict shape check_hardware_constraints expects.

        Only non-None fields are included so the constraint checker falls back to
        its theoretical estimate for anything a given backend didn't measure.
        """
        m: dict = {}
        if self.ttft_ms is not None:
            m["ttft_ms"] = self.ttft_ms
        if self.tok_per_s is not None:
            m["tok_per_s"] = self.tok_per_s
        if self.avg_watts is not None:
            m["avg_watts"] = self.avg_watts
        if self.peak_memory_mb is not None:
            m["peak_memory_mb"] = self.peak_memory_mb
        if self.device is not None:
            m["device"] = self.device
        return m


# ---------------------------------------------------------------------------
# Backend 1 — theoretical (no hardware; ModelSpec-derived estimates)
# ---------------------------------------------------------------------------

def theoretical_profile(model, constraints) -> HardwareEvalResult:
    """Estimate metrics from the ModelSpec's per-chip throughput table.

    Power is deliberately left None (we have no credible parametric estimate),
    which makes the power gate a no-op in theoretical mode rather than a guess.
    """
    tok_s = model.tok_s_for_chip(constraints.target_chip)
    ttft_ms = (1.0 / max(tok_s, 0.1)) * 1000
    return HardwareEvalResult(
        model_id=model.model_id,
        success=True,
        ttft_ms=round(ttft_ms, 1),
        tok_per_s=round(tok_s, 1),
        avg_watts=None,
        peak_memory_mb=model.peak_memory_mb,
        eval_method="theoretical",
        device=constraints.target_chip,
    )


# ---------------------------------------------------------------------------
# Backend 2 — local llama.cpp (GGUF timing on this machine, no phone)
# ---------------------------------------------------------------------------

def _parse_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def measure_llama_cpp(
    gguf_path: str,
    model,
    constraints,
    prompt: str = "Hello, how are you?",
    n_tokens: int = 50,
    n_threads: int = 4,
) -> HardwareEvalResult:
    """Proxy measurement using a local llama-cli run against the GGUF.

    Parses prompt-eval (TTFT) and decode tok/s from llama.cpp timing output.
    Power/peak-RAM are size-proxied (a local desktop run can't measure phone
    power), so avg_watts stays an estimate flagged in raw['power_estimated'].
    """
    model_id = getattr(model, "model_id", gguf_path)
    llama_cli = shutil.which("llama-cli") or shutil.which("main")
    if not llama_cli:
        return HardwareEvalResult(
            model_id=model_id, success=False, eval_method="llama_cpp", device="local",
            error="llama-cli not found on PATH. Build llama.cpp and add it to PATH.",
        )
    if not os.path.exists(gguf_path):
        return HardwareEvalResult(
            model_id=model_id, success=False, eval_method="llama_cpp", device="local",
            error=f"GGUF not found: {gguf_path}",
        )
    try:
        proc = subprocess.run(
            [llama_cli, "--model", gguf_path, "--prompt", prompt,
             "--n-predict", str(n_tokens), "--threads", str(n_threads),
             "--no-display-prompt"],
            capture_output=True, text=True, timeout=300,
        )
        out = proc.stderr + proc.stdout
        ttft = _parse_float(r"prompt eval time\s*=\s*([\d.]+)\s*ms", out)
        tok_s = _parse_float(r"eval.*?([\d.]+)\s*tokens per second", out)
        size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
        est_rss = int(size_mb * 1.2)
        est_watts = round(2.0 + (size_mb / 1000) * 1.5, 2)  # rough size proxy
        return HardwareEvalResult(
            model_id=model_id, success=True,
            ttft_ms=ttft, tok_per_s=tok_s,
            avg_watts=est_watts, peak_memory_mb=est_rss,
            eval_method="llama_cpp", device="local",
            raw={"power_estimated": True, "gguf_size_mb": round(size_mb, 1)},
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return HardwareEvalResult(
            model_id=model_id, success=False, eval_method="llama_cpp", device="local",
            error=f"llama-cli run failed: {e}",
        )


# ---------------------------------------------------------------------------
# ADB helper (shared by adb_llama + smolchat backends)
# ---------------------------------------------------------------------------

def find_adb() -> str | None:
    """Locate adb on PATH or the conventional Android SDK location."""
    on_path = shutil.which("adb")
    if on_path:
        return on_path
    fallback = str(Path.home() / "Library/Android/sdk/platform-tools/adb")
    return fallback if os.path.exists(fallback) else None


class Adb:
    """Thin wrapper over an adb invocation bound to one target device.

    device_kind: "phone" (-d, first USB device) or "emulator" (-e), or an
    explicit serial via serial=... (takes precedence, uses -s).
    """

    def __init__(self, adb_bin: str, device_kind: str = "phone", serial: str = ""):
        self.bin = adb_bin
        if serial:
            self.base = [adb_bin, "-s", serial]
            self.flag = f"-s {serial}"
        else:
            self.flag = "-d" if device_kind == "phone" else "-e"
            self.base = [adb_bin, self.flag]

    def run(self, args: list, timeout: int = 30) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(self.base + args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(self.base + args, returncode=1, stdout="", stderr="TIMEOUT")

    def is_connected(self) -> bool:
        r = self.run(["get-state"], timeout=10)
        return r.returncode == 0 and r.stdout.strip() == "device"

    def package_installed(self, package: str) -> bool:
        r = self.run(["shell", "pm", "list", "packages"], timeout=15)
        return package in r.stdout


def read_battery_voltage_v(adb: Adb) -> float:
    """Read live battery voltage (V) from dumpsys; fall back to the nominal value.

    dumpsys reports voltage in mV. Some devices report unrealistic values while
    charging, so we sanity-bound to [3.0, 4.5] V before trusting it.
    """
    r = adb.run(["shell", "dumpsys", "battery"], timeout=15)
    m = re.search(r"voltage:\s*(\d+)", r.stdout)
    if m:
        v = int(m.group(1)) / 1000.0
        if 3.0 <= v <= 4.5:
            return v
    return DEFAULT_BATTERY_VOLTAGE_V


def ma_to_watts(current_ma: float | None, voltage_v: float) -> float | None:
    """Convert BatteryManager discharge current (mA) to power (W): P = V·I.

    Uses abs() because discharge current is reported negative on some OEMs.
    """
    if current_ma is None:
        return None
    return round(abs(current_ma) / 1000.0 * voltage_v, 3)


# ---------------------------------------------------------------------------
# Backend 3 — adb + on-device llama-cli
# ---------------------------------------------------------------------------

def measure_adb_llama(
    gguf_path: str,
    model,
    constraints,
    serial: str = "",
    prompt: str = "Hello",
    n_tokens: int = 50,
) -> HardwareEvalResult:
    """Push a GGUF to a connected device and time an on-device llama-cli run.

    Requires: adb on PATH, an authorized device, and an llama-cli binary already
    present at /data/local/tmp/llama-cli on the device.
    """
    model_id = getattr(model, "model_id", gguf_path)
    adb_bin = find_adb()
    if not adb_bin:
        return HardwareEvalResult(model_id=model_id, success=False,
                                  eval_method="adb_llama", error="adb not found on PATH")
    if not os.path.exists(gguf_path):
        return HardwareEvalResult(model_id=model_id, success=False,
                                  eval_method="adb_llama", error=f"GGUF not found: {gguf_path}")
    adb = Adb(adb_bin, "phone", serial)
    if not adb.is_connected():
        return HardwareEvalResult(model_id=model_id, success=False,
                                  eval_method="adb_llama", error="no authorized device via adb")
    device_path = "/data/local/tmp/model.gguf"
    try:
        push = adb.run(["push", gguf_path, device_path], timeout=600)
        if push.returncode != 0:
            return HardwareEvalResult(model_id=model_id, success=False,
                                      eval_method="adb_llama", error=f"adb push failed: {push.stderr}")
        run = adb.run(["shell",
                       f"cd /data/local/tmp && ./llama-cli --model model.gguf "
                       f"--prompt '{prompt}' --n-predict {n_tokens} --threads 4 --no-display-prompt"],
                      timeout=300)
        out = run.stdout + run.stderr
        ttft = _parse_float(r"prompt eval time\s*=\s*([\d.]+)\s*ms", out)
        tok_s = _parse_float(r"eval.*?([\d.]+)\s*tokens per second", out)
        voltage = read_battery_voltage_v(adb)
        cur = adb.run(["shell", "cat", "/sys/class/power_supply/battery/current_now"], timeout=10)
        current_ua = _parse_float(r"(-?\d+)", cur.stdout)
        # current_now is µA on most kernels → mA
        watts = ma_to_watts(current_ua / 1000.0 if current_ua is not None else None, voltage)
        return HardwareEvalResult(
            model_id=model_id, success=True, ttft_ms=ttft, tok_per_s=tok_s,
            avg_watts=watts, peak_memory_mb=None,
            eval_method="adb_llama", device=serial or "connected",
            raw={"battery_voltage_v": voltage},
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return HardwareEvalResult(model_id=model_id, success=False,
                                  eval_method="adb_llama", device=serial or "connected",
                                  error=f"adb llama run failed: {e}")


# ---------------------------------------------------------------------------
# Backend 4 — SmolChat headless benchmark (broadcast + logcat scrape)
# The engine below is the metric core lifted out of run_autobench.py.
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


def parse_run_lines(logcat_text: str, run_id: str) -> dict:
    """Return {TAG: full_line} for every log line tagged with this run_id."""
    id_pattern = re.compile(r"run_id=" + re.escape(run_id) + r"(?:\s|$)")
    lines: dict = {}
    for line in logcat_text.splitlines():
        if not id_pattern.search(line):
            continue
        for tag in KNOWN_TAGS:
            if re.search(r"\b" + tag + r"\b", line):
                lines[tag] = line
                break
    return lines


def _extract_value(line: str):
    m = re.search(r"value=(\S+)", line)
    return m.group(1) if m else None


def _extract_response(line: str):
    m = re.search(r"response=(.*)$", line)
    return m.group(1).strip() if m else None


def _extract_error(line: str):
    m = re.search(r"reason=(\S+)\s+message=(.*)$", line)
    if m:
        return m.group(1), m.group(2).strip()
    return None, line.strip()


def poll_for_result(adb: Adb, run_id: str, timeout: int) -> tuple:
    """Poll logcat every 1s until RUN_DONE/RUN_ERROR for run_id or timeout."""
    deadline = time.time() + timeout
    last_lines: dict = {}
    while time.time() < deadline:
        r = adb.run(["logcat", "-d", "-s"] + LOGCAT_TAG_FILTER, timeout=20)
        last_lines = parse_run_lines(r.stdout, run_id)
        if "RUN_DONE" in last_lines:
            return "done", last_lines
        if "RUN_ERROR" in last_lines:
            return "error", last_lines
        time.sleep(1)
    return "timeout", last_lines


def _build_metrics(lines: dict) -> dict:
    def num(tag, cast):
        v = _extract_value(lines[tag]) if tag in lines else None
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
        "thermal": _extract_value(lines["THERMAL"]) if "THERMAL" in lines else None,
    }


def _run_one(adb: Adb, model_path: str, question: str, n: int, timeout: int) -> dict:
    run_id = f"run_{n}_{int(time.time() * 1000)}"
    clear_logcat(adb)
    fire_broadcast(adb, model_path, question, run_id)
    status, lines = poll_for_result(adb, run_id, timeout)
    return {"run_id": run_id, "status": status, "lines": lines}


def restart_smolchat(adb: Adb):
    adb.run(["shell", "am", "force-stop", SMOLCHAT_PACKAGE], timeout=15)
    adb.run(["shell", "am", "start", "-n", MAIN_ACTIVITY_COMPONENT], timeout=15)
    time.sleep(3)


def benchmark_smolchat(adb: Adb, model_path: str, questions: list, timeout: int,
                       log=print) -> tuple:
    """Run every question through SmolChat, one reload each. Returns (results, context_resets).

    results: list of per-question dicts {question, status, metrics, response, error}.
    """
    results = []
    context_resets = 0
    total = len(questions)
    for n, question in enumerate(questions, start=1):
        outcome = _run_one(adb, model_path, question, n, timeout)
        status, lines = outcome["status"], outcome["lines"]
        context_reset = False
        if status == "error":
            _, message = _extract_error(lines.get("RUN_ERROR", ""))
            if message and CONTEXT_SIZE_PHRASE in message.lower():
                log(f"  [{n}/{total}] context full — restarting SmolChat and retrying once")
                restart_smolchat(adb)
                context_resets += 1
                context_reset = True
                outcome = _run_one(adb, model_path, question, n, timeout)
                status, lines = outcome["status"], outcome["lines"]

        entry = {"question_number": n, "question": question, "run_id": outcome["run_id"],
                 "status": None, "metrics": None, "response": None, "error": None,
                 "context_reset": context_reset}
        if status == "done":
            entry["status"] = "success"
            entry["metrics"] = _build_metrics(lines)
            entry["response"] = _extract_response(lines["RUN_DONE"])
            m = entry["metrics"]
            log(f"  [{n}/{total}] TTFT={m['ttft_ms']}ms TPS={m['tps']} "
                f"RSS={m['memory_kb']}KB Power={m['power_ma']}mA Thermal={m['thermal']}")
        elif status == "error":
            reason, message = _extract_error(lines.get("RUN_ERROR", ""))
            entry["status"] = "failed"
            entry["error"] = {"reason": reason, "message": message}
            log(f"  [{n}/{total}] FAILED: {reason} {message}")
        else:
            entry["status"] = "failed"
            entry["error"] = {"reason": "timeout", "message": f"no result within {timeout}s"}
            log(f"  [{n}/{total}] FAILED: timeout after {timeout}s")
        results.append(entry)
        time.sleep(2)
    return results, context_resets


def _stat_block(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.pstdev(values), 3),
        "min": min(values),
        "max": max(values),
    }


def summarize_smolchat(results: list, voltage_v: float) -> dict:
    """Aggregate per-question SmolChat results into stat blocks + a measured view.

    Returns a dict with per-metric stat blocks plus a flat 'measured' summary
    (mean TTFT/TPS, mean power in W, PEAK RSS in MB) suitable for gating.
    """
    def vals(key):
        return [r["metrics"][key] for r in results
                if r["status"] == "success" and r["metrics"] and r["metrics"].get(key) is not None]

    ttft = vals("ttft_ms")
    tps = vals("tps")
    mem_kb = vals("memory_kb")
    power_ma = vals("power_ma")

    mean_power_ma = statistics.mean(power_ma) if power_ma else None
    peak_rss_mb = int(max(mem_kb) / 1024) if mem_kb else None  # peak = worst case across questions

    first_q = next((r for r in results if r["question_number"] == 1), None)
    first_cold = (first_q["metrics"].get("cold_load_ms")
                  if first_q and first_q["status"] == "success" and first_q["metrics"] else None)
    thermal_states = sorted({r["metrics"]["thermal"] for r in results
                             if r["status"] == "success" and r["metrics"] and r["metrics"].get("thermal")})

    return {
        "ttft_ms": _stat_block(ttft),
        "tps": _stat_block(tps),
        "memory_kb": _stat_block(mem_kb),
        "power_ma": _stat_block(power_ma),
        "first_question_cold_load_ms": first_cold,
        "thermal_states_observed": thermal_states,
        "battery_voltage_v": voltage_v,
        "measured": {
            "ttft_ms": round(statistics.mean(ttft), 1) if ttft else None,
            "tok_per_s": round(statistics.mean(tps), 2) if tps else None,
            "avg_watts": ma_to_watts(mean_power_ma, voltage_v),
            "peak_memory_mb": peak_rss_mb,
        },
    }


def measure_smolchat(
    gguf_path: str,
    model,
    constraints,
    questions: list | None = None,
    device_kind: str = "phone",
    serial: str = "",
    timeout: int = 60,
    log=print,
) -> HardwareEvalResult:
    """Push a local GGUF to a device and benchmark it through SmolChat headlessly.

    Defensive: any missing prerequisite (adb, device, SmolChat, receiver) returns
    success=False with a diagnostic error rather than raising.
    """
    model_id = getattr(model, "model_id", gguf_path)
    adb_bin = find_adb()
    if not adb_bin:
        return HardwareEvalResult(model_id=model_id, success=False, eval_method="smolchat",
                                  error="adb not found on PATH")
    if not os.path.exists(gguf_path):
        return HardwareEvalResult(model_id=model_id, success=False, eval_method="smolchat",
                                  error=f"GGUF not found: {gguf_path}")
    adb = Adb(adb_bin, device_kind, serial)
    if not adb.is_connected():
        return HardwareEvalResult(model_id=model_id, success=False, eval_method="smolchat",
                                  error=f"no authorized {device_kind} via adb")
    if not adb.package_installed(SMOLCHAT_PACKAGE):
        return HardwareEvalResult(model_id=model_id, success=False, eval_method="smolchat",
                                  error=f"SmolChat ({SMOLCHAT_PACKAGE}) not installed on device")

    device_path = f"/sdcard/Download/{os.path.basename(gguf_path)}"
    push = adb.run(["push", gguf_path, device_path], timeout=600)
    if push.returncode != 0:
        return HardwareEvalResult(model_id=model_id, success=False, eval_method="smolchat",
                                  error=f"adb push failed: {push.stderr}")

    voltage = read_battery_voltage_v(adb)
    qs = questions or DEFAULT_QUESTIONS
    results, context_resets = benchmark_smolchat(adb, device_path, qs, timeout, log=log)
    summary = summarize_smolchat(results, voltage)
    meas = summary["measured"]

    n_ok = sum(1 for r in results if r["status"] == "success")
    if n_ok == 0:
        return HardwareEvalResult(model_id=model_id, success=False, eval_method="smolchat",
                                  device=serial or device_kind,
                                  error="all SmolChat runs failed (see raw.results)",
                                  raw={"results": results, "summary": summary})
    return HardwareEvalResult(
        model_id=model_id, success=True,
        ttft_ms=meas["ttft_ms"], tok_per_s=meas["tok_per_s"],
        avg_watts=meas["avg_watts"], peak_memory_mb=meas["peak_memory_mb"],
        cold_load_ms=summary["first_question_cold_load_ms"],
        thermal=(summary["thermal_states_observed"] or [None])[-1],
        eval_method="smolchat", device=serial or device_kind,
        raw={"results": results, "summary": summary, "context_resets": context_resets},
    )


# ---------------------------------------------------------------------------
# Dispatcher + mappers
# ---------------------------------------------------------------------------

def run_on_device_eval(model, constraints, *, gguf_path: str | None = None,
                       backend: str | None = None, questions: list | None = None,
                       serial: str = "", timeout: int = 60, log=print) -> HardwareEvalResult:
    """Measure `model` on the target hardware, or estimate it theoretically.

    Backend resolution:
      - explicit `backend` arg wins;
      - else env SLM_HW_BACKEND (DEFAULT_BACKEND);
      - a real backend needs a GGUF — if gguf_path is None we ALWAYS fall back to
        theoretical. This is what keeps the pre-training hardware_filter (which has
        no GGUF yet) on the cheap estimate path and existing tests green.
    """
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "theoretical" or gguf_path is None:
        return theoretical_profile(model, constraints)
    if backend == "llama_cpp":
        return measure_llama_cpp(gguf_path, model, constraints)
    if backend == "adb_llama":
        return measure_adb_llama(gguf_path, model, constraints, serial=serial)
    if backend == "smolchat":
        return measure_smolchat(gguf_path, model, constraints, questions=questions,
                                serial=serial, timeout=timeout, log=log)
    # Unknown backend name — degrade to theoretical rather than crash the loop.
    res = theoretical_profile(model, constraints)
    res.error = f"unknown backend {backend!r}; used theoretical"
    return res


def result_to_dict(result: HardwareEvalResult) -> dict:
    """Full serializable view of a result (for hardware_eval.json)."""
    return asdict(result)
