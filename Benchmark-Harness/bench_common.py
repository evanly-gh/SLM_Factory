#!/usr/bin/env python3
"""
bench_common.py - measurement helpers shared by run_autobench.py (SmolChat / llama.cpp) and
run_mnn_autobench.py (MNN), so both engines are measured the same way on every backend.

  * device_state()   - battery temperature/level/power source, CPU frequency caps, screen and
                        thermal status, read from adb shell (no app involvement).
  * ReadinessGate    - waits before a run until the phone is cool and its CPU frequency caps are
                        back at the session baseline, so thermal/DVFS state is consistent.
  * evict_page_cache - drops a model's files from the OS page cache (page_cache_tool, run as the
                        app's uid via run-as) so the next load is a genuine cold read.
  * EnergyTrace      - one Perfetto session over a whole harness run, recording battery current
                        and voltage; energy for any wall-clock window is integrated afterwards.

Energy is only meaningful when the phone is NOT externally powered (USB/AC/wireless charger):
while powered, battery current follows the charger, not the workload. Use wireless adb with
the cable unplugged. Every energy result carries an explicit validity flag.

Measured on the OnePlus CPH2749 (SM8850): Perfetto's batt.current_ua is really in mA and is
positive while charging; the fuel gauge updates roughly every 5 s, so windows shorter than
~15 s have few independent readings (reported as current_updates).

CLI (runs under the repo .venv, which has the perfetto package):
    python bench_common.py integrate <trace.pftrace> <windows.json>
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_PYTHON = HERE.parent / ".venv" / "bin" / "python"
NDK_CLANG = (Path.home() / "Library/Android/sdk/ndk/27.2.12479018/toolchains/llvm/prebuilt/darwin-x86_64/bin"
             / "aarch64-linux-android28-clang")
TOOL_SRC = HERE / "tools" / "page_cache_tool.c"
TOOL_BIN = HERE / "tools" / "bin" / "page_cache_tool"
DEVICE_TOOL = "/data/local/tmp/page_cache_tool"
APP_PRIVATE_PREFIXES = ("/data/user/", "/data/data/")

# Perfetto battery counters. write_into_file streams to disk, so hours-long sessions don't
# overflow the in-memory buffer.
PERFETTO_CONFIG = """
buffers { size_kb: 16384 fill_policy: DISCARD }
data_sources {
  config {
    name: "android.power"
    android_power_config {
      battery_poll_ms: 100
      battery_counters: BATTERY_COUNTER_CURRENT
      battery_counters: BATTERY_COUNTER_VOLTAGE
      battery_counters: BATTERY_COUNTER_CHARGE
      battery_counters: BATTERY_COUNTER_CAPACITY_PERCENT
    }
  }
}
write_into_file: true
file_write_period_ms: 5000
max_file_size_bytes: 1000000000
duration_ms: 21600000
"""

# On this device positive current means charging (confirmed against the charge counter).
POSITIVE_CURRENT_IS_CHARGING = True
# |current| medians below this are taken to be mA (the gauge reports mA in the µA field).
MA_UNIT_THRESHOLD = 20000


def _sh(adb, cmd: str, timeout: int = 30) -> str:
    return adb.run(["shell", cmd], timeout=timeout).stdout or ""


# ---------------------------------------------------------------------------
# Device state + readiness gate
# ---------------------------------------------------------------------------

def device_state(adb) -> dict:
    """Snapshot of everything that affects performance/energy readings, from adb shell."""
    out = _sh(adb, "dumpsys battery; echo @@CAPS; "
                   "for p in /sys/devices/system/cpu/cpufreq/policy*; do "
                   "echo $(basename $p) $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq); done; "
                   "echo @@POWER; dumpsys power | grep -m1 mWakefulness=; "
                   "echo @@THERMAL; dumpsys thermalservice | grep -m1 'Thermal Status'")
    sections = dict.fromkeys(("BATTERY", "CAPS", "POWER", "THERMAL"), "")
    current = "BATTERY"
    for line in out.splitlines():
        if line.startswith("@@") and line[2:] in sections:
            current = line[2:]
        else:
            sections[current] += line + "\n"
    battery, caps, power, thermal = (sections[k] for k in ("BATTERY", "CAPS", "POWER", "THERMAL"))

    def field(name):
        m = re.search(r"^\s*" + re.escape(name) + r":\s*(\S+)", battery, re.M)
        return m.group(1) if m else None

    sources = {k: field(f"{k.upper()} powered") == "true" for k in ("ac", "usb", "wireless")}
    cur_caps, hw_caps = {}, {}
    for line in caps.split("\n"):
        p = line.split()
        if len(p) == 3 and p[0].startswith("policy") and p[1].isdigit() and p[2].isdigit():
            cur_caps[p[0]], hw_caps[p[0]] = int(p[1]), int(p[2])
    wake = re.search(r"mWakefulness=(\w+)", power)
    tstat = re.search(r"Thermal Status:\s*(\d+)", thermal)
    temp = field("temperature")
    return {
        "epoch_ms": int(time.time() * 1000),
        "battery_temp_c": int(temp) / 10 if temp and temp.lstrip("-").isdigit() else None,
        "battery_level": int(field("level")) if (field("level") or "").isdigit() else None,
        "battery_status": int(field("status")) if (field("status") or "").isdigit() else None,
        "externally_powered": any(sources.values()),
        "power_sources": [k for k, v in sources.items() if v],
        "cpu_caps_khz": cur_caps,
        "cpu_hw_max_khz": hw_caps,
        "cpu_capped": any(cur_caps[k] < hw_caps[k] for k in cur_caps),
        "screen": wake.group(1) if wake else None,
        "thermal_status": int(tstat.group(1)) if tstat else None,
    }


class ReadinessGate:
    """Block until the battery temperature is within its limit and every CPU policy's frequency cap
    is at least its session baseline (set_baseline(), taken after an initial rest).
    Temperature limit = max_temp_c (absolute) and/or baseline temperature + rise_c (relative to the
    phone's own temperature when the configuration started; use this on a phone that idles warm, e.g.
    screen on and charging, where an absolute limit is just room temperature). The baseline,
    not the hardware maximum, is the reference because this phone caps CPU clocks by screen
    state even when cool (screen off 2.0/2.4 GHz, on 2.9/2.9 GHz vs 3.6/4.6 GHz hardware)."""

    def __init__(self, adb, max_temp_c: float | None, timeout_s: int = 900, poll_s: int = 15, log=print,
                 rise_c: float | None = None):
        self.adb, self.max_temp_c, self.timeout_s, self.poll_s, self.log = adb, max_temp_c, timeout_s, poll_s, log
        self.rise_c = rise_c
        self.baseline_caps: dict | None = None
        self.baseline_temp: float | None = None

    def temp_limit(self) -> float | None:
        limits = []
        if self.max_temp_c is not None:
            limits.append(self.max_temp_c)
        if self.rise_c is not None and self.baseline_temp is not None:
            limits.append(self.baseline_temp + self.rise_c)
        return min(limits) if limits else None

    def set_baseline(self, state: dict):
        self.baseline_caps = dict(state["cpu_caps_khz"])
        self.baseline_temp = state["battery_temp_c"]
        self.log(f"[GATE] baseline CPU caps {self.baseline_caps} "
                 f"(hw max {state['cpu_hw_max_khz']}), screen={state['screen']}, "
                 f"battery {state['battery_temp_c']} C")

    def _problems(self, st: dict) -> list:
        issues = []
        limit = self.temp_limit()
        if limit is not None and st["battery_temp_c"] is not None and st["battery_temp_c"] > limit:
            issues.append(f"battery {st['battery_temp_c']} C > {limit:.1f} C")
        for pol, base in (self.baseline_caps or {}).items():
            cur = st["cpu_caps_khz"].get(pol)
            if cur is not None and cur < base:
                issues.append(f"{pol} capped {cur // 1000} MHz < baseline {base // 1000} MHz")
        return issues

    def wait(self) -> dict:
        t0 = time.time()
        last_msg, last_log = None, 0.0
        while True:
            st = device_state(self.adb)
            issues = self._problems(st)
            waited = time.time() - t0
            if not issues:
                return {"passed": True, "waited_s": round(waited, 1), "state": st}
            if waited >= self.timeout_s:
                self.log(f"[GATE] WARNING: not ready after {self.timeout_s}s ({'; '.join(issues)}) - running anyway")
                return {"passed": False, "waited_s": round(waited, 1), "state": st, "issues": issues}
            msg = "; ".join(issues)
            if msg != last_msg or time.time() - last_log >= 60:
                self.log(f"[GATE] waiting ({waited:.0f}s): {msg}")
                last_msg, last_log = msg, time.time()
            time.sleep(self.poll_s)


# ---------------------------------------------------------------------------
# Page-cache eviction for genuine cold starts
# ---------------------------------------------------------------------------

def _ensure_tool(adb) -> None:
    if not TOOL_BIN.exists() or TOOL_BIN.stat().st_mtime < TOOL_SRC.stat().st_mtime:
        TOOL_BIN.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(NDK_CLANG), "-O2", "-o", str(TOOL_BIN), str(TOOL_SRC)], check=True)
    adb.run(["push", str(TOOL_BIN), DEVICE_TOOL], timeout=60)
    _sh(adb, f"chmod 755 {DEVICE_TOOL}")


def evict_page_cache(adb, package: str, path: str) -> dict:
    """Evict `path` (a file, or every file under a directory) from the page cache. The app
    owning the files must already be force-stopped (mapped pages cannot be dropped).
    Returns {"ok", "files": [{path, resident_before, resident_after, bytes}], "error"}."""
    _ensure_tool(adb)
    private = path.startswith(APP_PRIVATE_PREFIXES)
    if private:
        # run-as only works for debuggable apps; the tool is copied into the app's own dir so
        # it runs with the app's uid/SELinux context and can open its private files.
        prep = _sh(adb, f"run-as {package} sh -c 'cp {DEVICE_TOOL} ./page_cache_tool && chmod 700 ./page_cache_tool' 2>&1")
        if "not debuggable" in prep or "unknown package" in prep:
            return {"ok": False, "files": [], "error": prep.strip()}
        run = f"run-as {package} sh -c 'find {path} -type f -exec ./page_cache_tool --evict {{}} +' 2>&1"
    else:
        run = f"find {path} -type f -exec {DEVICE_TOOL} --evict {{}} + 2>&1"
    files, errors = [], []
    for line in _sh(adb, run, timeout=120).splitlines():
        m = re.match(r"(\S+) resident_before=([\d.-]+) resident_after=([\d.-]+) bytes=(\d+)", line.strip())
        if m:
            files.append({"path": m.group(1), "resident_before_pct": float(m.group(2)),
                          "resident_after_pct": float(m.group(3)), "bytes": int(m.group(4))})
        elif line.strip():
            errors.append(line.strip())
    total = sum(f["bytes"] for f in files) or 1
    after = sum(f["resident_after_pct"] * f["bytes"] for f in files) / total
    return {"ok": bool(files) and not errors, "files": files, "resident_after_pct": round(after, 2),
            "error": "; ".join(errors) or None}


# ---------------------------------------------------------------------------
# Energy: Perfetto battery counters over the whole run
# ---------------------------------------------------------------------------

class EnergyTrace:
    """One Perfetto session spanning a harness run. start() before the first generation,
    stop() after the last, then integrate([(key, start_epoch_ms, end_epoch_ms), ...])."""

    def __init__(self, adb, local_dir: Path, log=print):
        self.adb, self.local_dir, self.log = adb, Path(local_dir), log
        self.pid = None
        self.remote = f"/data/misc/perfetto-traces/bench_{int(time.time())}.pftrace"
        self.local_path: Path | None = None

    def start(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pbtx", delete=False) as f:
            f.write(PERFETTO_CONFIG)
            cfg = f.name
        self.adb.run(["push", cfg, "/data/local/tmp/bench_perfetto.pbtx"], timeout=30)
        # Piped via stdin: perfetto (SELinux) may not read a config file in /data/local/tmp.
        out = _sh(self.adb, f"cat /data/local/tmp/bench_perfetto.pbtx | "
                            f"perfetto --txt -c - -o {self.remote} --background-wait 2>&1")
        m = re.search(r"^\s*(\d+)\s*$", out, re.M)
        if not m:
            raise RuntimeError(f"perfetto did not start: {out.strip()[:300]}")
        self.pid = int(m.group(1))
        self.log(f"[ENERGY] Perfetto battery trace started (pid {self.pid})")

    def stop(self) -> Path:
        if self.pid is None:
            raise RuntimeError("EnergyTrace.stop() before start()")
        _sh(self.adb, f"kill -TERM {self.pid}")
        for _ in range(60):
            if "alive" not in _sh(self.adb, f"kill -0 {self.pid} 2>/dev/null && echo alive"):
                break
            time.sleep(1)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.local_path = self.local_dir / Path(self.remote).name
        self.adb.run(["pull", self.remote, str(self.local_path)], timeout=300)
        _sh(self.adb, f"rm -f {self.remote}")
        self.pid = None
        self.log(f"[ENERGY] trace saved: {self.local_path}")
        return self.local_path

    def integrate(self, windows: list) -> dict:
        return integrate_windows(self.local_path, windows)


def integrate_windows(trace_path, windows: list) -> dict:
    """windows: [(key, start_epoch_ms, end_epoch_ms), ...] -> {key: result}. Runs in-process if
    the perfetto package is importable, otherwise via the repo .venv."""
    try:
        import perfetto  # noqa: F401
    except ImportError:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(windows, f)
            wpath = f.name
        out = subprocess.run([str(VENV_PYTHON), __file__, "integrate", str(trace_path), wpath],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out.strip().splitlines()[-1])
    return _integrate_in_process(trace_path, windows)


def _integrate_in_process(trace_path, windows: list) -> dict:
    from perfetto.trace_processor import TraceProcessor

    tp = TraceProcessor(trace=str(trace_path))
    try:
        offset_ns = next(iter(tp.query(
            "select ts - clock_value as o from clock_snapshot where clock_id = 1 limit 1"))).o

        def series(name):
            q = (f"select c.ts as ts, c.value as v from counter c join counter_track t on c.track_id = t.id "
                 f"where t.name = '{name}' order by c.ts")
            return [(r.ts, r.v) for r in tp.query(q)]

        cur, volt = series("batt.current_ua"), series("batt.voltage_uv")
    finally:
        tp.close()
    return _integrate_series(cur, volt, offset_ns, windows)


def _integrate_series(cur: list, volt: list, offset_ns: int, windows: list) -> dict:
    """Pure integration step (no trace access, so it can be unit-tested).
    cur/volt: [(boottime_ns, value)] sorted by time; current in mA or uA (auto-detected), positive =
    charging; voltage in uV. windows: [(key, start_epoch_ms, end_epoch_ms)]; offset_ns converts epoch
    to trace time (trace_ns = epoch_ms * 1e6 + offset_ns). Zero-order hold between samples."""
    if not cur or not volt:
        return {k: {"error": "no battery counters in trace"} for k, *_ in windows}

    ma_scale = 1.0 if statistics.median(abs(v) for _, v in cur) < MA_UNIT_THRESHOLD else 1e-3
    sign = -1.0 if POSITIVE_CURRENT_IS_CHARGING else 1.0

    def held(series_, t):
        """Zero-order hold: value of the latest sample at or before t (first sample if none)."""
        lo, hi = 0, len(series_) - 1
        if t < series_[0][0]:
            return series_[0][1]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if series_[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        return series_[lo][1]

    results = {}
    for key, start_ms, end_ms in windows:
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            results[key] = {"error": "invalid window"}
            continue
        a, b = int(start_ms * 1e6) + offset_ns, int(end_ms * 1e6) + offset_ns
        if a < cur[0][0] or b > cur[-1][0]:
            results[key] = {"error": "window outside trace"}
            continue
        # breakpoints: window edges plus every sample time of either series inside the window
        pts = sorted({a, b, *(t for t, _ in cur if a < t < b), *(t for t, _ in volt if a < t < b)})
        energy_mj = charging_s = 0.0
        for t0, t1 in zip(pts, pts[1:]):
            dt = (t1 - t0) / 1e9
            discharge_ma = sign * held(cur, t0) * ma_scale
            energy_mj += discharge_ma * (held(volt, t0) / 1e6) * dt  # mA * V * s = mJ
            if discharge_ma < 0:
                charging_s += dt
        in_window = [v for t, v in cur if a <= t <= b]
        updates = sum(1 for x, y in zip(in_window, in_window[1:]) if x != y)
        dur = (b - a) / 1e9
        results[key] = {
            "energy_mj": round(energy_mj, 3),
            "avg_power_mw": round(energy_mj / dur, 3),
            "duration_s": round(dur, 3),
            "current_updates": updates,
            "charging_fraction": round(charging_s / dur, 3),
            "current_unit": "mA" if ma_scale == 1.0 else "uA",
        }
    return results


def energy_fields(window: dict | None, idle_power_mw: float | None, tokens: int | None,
                  powered: bool) -> dict:
    """Per-generation energy summary with an explicit validity flag."""
    if not window or "energy_mj" not in window:
        return {"energy_mj": None, "energy_valid": False,
                "energy_invalid_reason": (window or {}).get("error", "no window")}
    reasons = []
    if powered:
        reasons.append("externally powered (charger supplies the load)")
    if window["charging_fraction"] > 0:
        reasons.append("battery was charging during window")
    if window["current_updates"] < 2:
        reasons.append("fewer than 2 fuel-gauge updates in window (too short)")
    net = (window["energy_mj"] - idle_power_mw * window["duration_s"]) if idle_power_mw is not None else None
    return {
        "energy_mj": window["energy_mj"],
        "energy_net_mj": round(net, 3) if net is not None else None,
        "avg_power_mw": window["avg_power_mw"],
        "energy_window_s": window["duration_s"],
        "energy_current_updates": window["current_updates"],
        "energy_mj_per_token": round(window["energy_mj"] / tokens, 3) if tokens else None,
        "energy_net_mj_per_token": round(net / tokens, 3) if (tokens and net is not None) else None,
        "energy_valid": not reasons,
        "energy_invalid_reason": "; ".join(reasons) or None,
        # Short windows are unreliable one at a time but still unbiased in aggregate (see
        # aggregate_energy()); external power or charging makes a window unusable either way.
        "energy_aggregatable": not powered and window["charging_fraction"] == 0,
    }


def aggregate_energy(entries: list) -> dict | None:
    """Energy per token over all usable generations of one configuration: sum of window
    energies / sum of generated tokens. Robust to the ~5 s fuel-gauge update period in a way
    per-generation values for short answers are not. Cold-start runs are excluded."""
    use = [e["metrics"] for e in entries
           if e.get("status") == "success" and e.get("phase") != "cold" and e.get("metrics")
           and e["metrics"].get("energy_aggregatable") and e["metrics"].get("gen_tokens")]
    if not use:
        return None
    tokens = sum(m["gen_tokens"] for m in use)
    total = sum(m["energy_mj"] for m in use)
    nets = [m["energy_net_mj"] for m in use if m.get("energy_net_mj") is not None]
    secs = sum(m["energy_window_s"] for m in use)
    return {
        "generations": len(use),
        "total_tokens": tokens,
        "total_window_s": round(secs, 2),
        "total_current_updates": sum(m["energy_current_updates"] for m in use),
        "energy_mj_per_token": round(total / tokens, 3),
        "energy_net_mj_per_token": round(sum(nets) / tokens, 3) if len(nets) == len(use) else None,
        "avg_power_mw": round(total / secs, 3) if secs else None,
    }


def device_epoch_ms(adb) -> int:
    """The phone's wall clock (the same clock the apps' epoch-ms log markers use)."""
    return int(_sh(adb, "date +%s%N").strip()) // 1_000_000


def measure_idle_window(adb, seconds: int, log=print) -> tuple:
    """Sleep `seconds` with nothing running; returns the (start_ms, end_ms) window, on the
    phone's clock, to integrate as the idle-power baseline."""
    log(f"[ENERGY] measuring {seconds}s idle baseline...")
    start = device_epoch_ms(adb)
    time.sleep(seconds)
    return start, device_epoch_ms(adb)


def attach_energy(entries: list, trace: "EnergyTrace", idle_window: tuple | None, log=print) -> float | None:
    """Integrate energy for every successful entry and merge it into entry["metrics"].

    Generation window: dispatch_epoch_ms -> last_token_epoch_ms (prefill + decode).
    Cold-start window (phase == "cold" only): load_start_epoch_ms -> first_token_epoch_ms.
    Each entry's metrics must carry those epoch fields plus gen_tokens; its state_before /
    state_after decide validity (any external power -> invalid). Returns idle power (mW)."""
    windows = []
    if idle_window:
        windows.append(("idle", *idle_window))
    for i, e in enumerate(entries):
        m = e.get("metrics") or {}
        if e.get("status") != "success":
            continue
        windows.append((f"gen:{i}", m.get("dispatch_epoch_ms"), m.get("last_token_epoch_ms")))
        if e.get("phase") == "cold":
            windows.append((f"cold:{i}", m.get("load_start_epoch_ms"), m.get("first_token_epoch_ms")))
    integrated = trace.integrate(windows)
    idle = integrated.get("idle", {})
    idle_mw = idle.get("avg_power_mw") if "energy_mj" in idle and idle.get("charging_fraction") == 0 else None
    if idle_mw is not None and any((e.get("state_before") or {}).get("externally_powered") for e in entries):
        log(f"[ENERGY] idle baseline {idle_mw} mW ignored: the phone was externally powered, so it is not a real idle draw")
        idle_mw = None
    else:
        log(f"[ENERGY] idle baseline: {idle_mw} mW" + ("" if idle_mw is not None else f" ({idle})"))
    for i, e in enumerate(entries):
        if e.get("status") != "success":
            continue
        powered = any((e.get(k) or {}).get("externally_powered") for k in ("state_before", "state_after"))
        e["metrics"].update(energy_fields(integrated.get(f"gen:{i}"), idle_mw, e["metrics"].get("gen_tokens"), powered))
        if e.get("phase") == "cold":
            cold = energy_fields(integrated.get(f"cold:{i}"), idle_mw, None, powered)
            e["metrics"]["cold_start_energy_mj"] = cold["energy_mj"] if cold["energy_valid"] else None
            e["metrics"]["cold_start_energy_net_mj"] = cold.get("energy_net_mj") if cold["energy_valid"] else None
    return idle_mw


PROTOCOL = "per_question"

_QUANT_RE = re.compile(r"(?:^|[-_.])(q\d+_k_[sml]|q\d+_\d|q\d+|f16|f32|bf16)$", re.I)


def model_and_quant(ref: str, engine: str) -> tuple:
    """Split a model file/folder name into (model name, quantization label).
    smolchat: 'smollm2-135m-q4_k_m.gguf' -> ('smollm2-135m', 'Q4_K_M'); 'gemma-3-270m-it-f16' -> (..., 'F16').
    mnn:      'smollm2-135m-mnn-q4'      -> ('smollm2-135m', 'Q4');     '...-mnn-q16' (fp16) -> 'F16'."""
    name = Path(str(ref).rstrip("/")).name
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    quant = None
    m = _QUANT_RE.search(name)
    if m:
        token = m.group(1).lower()
        name = name[:m.start()]
        quant = "F16" if token in ("q16", "f16", "bf16") else token.upper()
    if engine == "mnn":
        name = re.sub(r"[-_.]mnn$", "", name, flags=re.I)
    return name, quant


def backend_label(engine: str, requested: str | None, entries: list) -> str:
    """cpu | opencl | vulkan. mnn: the requested backend_type. smolchat: what actually registered
    (BACKEND_CHECK), since a GPU request can silently fall back to CPU."""
    if engine == "mnn":
        return requested or "cpu"
    seen = " ".join((e.get("metrics") or {}).get("backend_verified") or "" for e in entries
                    if e.get("status") == "success")
    return "opencl" if "OpenCL" in seen else "vulkan" if "Vulkan" in seen else "cpu"


def config_labels(engine: str, model_ref: str, requested_backend: str | None, entries: list) -> dict:
    """Self-describing labels written into every result file's run_info (read by compare_engines.py)."""
    model, quant = model_and_quant(model_ref, engine)
    return {"engine": engine, "backend": backend_label(engine, requested_backend, entries),
            "model_name": model, "quant_label": quant}


def finalize_question(group: list) -> None:
    """Post-process one question's runs (run 1 = cold, runs 2..N = warm), in place.

    * cold load / cold TTFT / cold start are taken from run 1 and kept constant across the
      question's runs (the model is only cold once); each run's own measured load time is kept
      as metrics["load_ms_measured"].
    * the last successful run is flagged reported=True: its TTFT, TTLT, prefill, decode and
      energy are the question's headline numbers. All runs stay in the results.
    """
    ok = [e for e in group if e.get("status") == "success"]
    cold = next((e for e in group if e.get("phase") == "cold" and e.get("status") == "success"), None)
    if cold:
        c = cold["metrics"]
        for e in ok:
            m = e["metrics"]
            m["load_ms_measured"] = m.get("cold_load_ms")
            m["cold_load_ms"] = c.get("cold_load_ms")
            m["cold_ttft_ms"] = c.get("ttft_ms")
            m["cold_start_ms"] = c.get("cold_start_ms")
            # Memory is taken from run 1 as well: it is the footprint of a fresh process. Reloading the
            # model in the same process (SmolChat reloads on every call) leaves freed memory behind, so
            # later runs' peak RSS creeps up (382 -> 403 -> 508 MB on SmolLM2-135M) without the model
            # getting any bigger.
            for key in ("memory_kb", "peak_rss_kb"):
                if c.get(key) is not None:
                    m[f"{key}_measured"] = m.get(key)
                    m[key] = c[key]
    if ok:
        ok[-1]["reported"] = True


def summary_entries(results: list) -> list:
    """Runs that count toward the headline summary: per-question protocol -> only the reported
    (last successful) run of each question; legacy runs -> every run."""
    if any(r.get("protocol") == PROTOCOL for r in results):
        return [r for r in results if r.get("reported")]
    return list(results)


def add_common_args(p):
    """Measurement-protocol flags shared by both harnesses."""
    g = p.add_argument_group("measurement protocol (bench_common.py)")
    g.add_argument("--runs-per-question", type=int, default=3, dest="runs_per_question",
                   help="Per-question protocol (default). For each question: gate, force-stop the app, evict "
                        "the model from the page cache, then run this many inferences back to back. Run 1 is "
                        "the genuinely cold one (its load time is the question's cold load, kept constant "
                        "for the others); the LAST run is the reported one. All runs are kept in the JSON.")
    g.add_argument("--legacy-protocol", action="store_true", dest="legacy_protocol",
                   help="Use the older flow instead (--trials / --warmup-runs / --reboot-before), where the "
                        "gate runs before every single run and there is no per-question cold start.")
    g.add_argument("--gate-max-temp", type=float, default=None, dest="gate_max_temp",
                   help="Before each question (each run in --legacy-protocol), wait until battery temperature "
                        "<= this (C) and CPU frequency caps are back at the session baseline. Default: gate off.")
    g.add_argument("--gate-temp-rise", type=float, default=None, dest="gate_temp_rise",
                   help="Like --gate-max-temp but relative: wait while the battery is more than this many C above "
                        "its temperature when the run started. Better than an absolute limit on a phone that idles "
                        "warm (screen on, charging). Both may be given; the lower limit applies.")
    g.add_argument("--gate-timeout", type=int, default=900, dest="gate_timeout",
                   help="Max seconds to wait at a gate before running anyway (flagged in results).")
    g.add_argument("--rest-seconds", type=int, default=0, dest="rest_seconds",
                   help="Fixed pause before each gate check (lets the SoC shed heat between questions).")
    g.add_argument("--energy", action="store_true",
                   help="Record battery current/voltage with Perfetto for the whole run and integrate "
                        "energy per generation. Valid only when the phone is unplugged (wireless adb).")
    g.add_argument("--idle-seconds", type=int, default=30, dest="idle_seconds",
                   help="Idle-power baseline measured at the start of an --energy run.")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "integrate":
        windows = json.loads(Path(sys.argv[3]).read_text())
        print(json.dumps(_integrate_in_process(sys.argv[2], windows)))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
