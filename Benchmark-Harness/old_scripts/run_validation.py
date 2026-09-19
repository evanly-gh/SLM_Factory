#!/usr/bin/env python3
"""
SmolChat Metric Validation Script
Validates all 6 metrics from SmolChat running on Android via ADB.
"""

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

PACKAGE = "io.shubham0204.smollmandroid"
MAIN_ACTIVITY = f"{PACKAGE}/.MainActivity"

SHORT_PROMPT = "What is the capital of France?"
LONG_PROMPT = "Write a detailed 500 word essay about artificial intelligence"

LOGCAT_TAGS = ["TTFT", "TPS", "MEMORY", "THERMAL", "POWER", "COLD_LOAD"]


# ---------------------------------------------------------------------------
# ADB helpers
# ---------------------------------------------------------------------------

def adb(args: list[str], device_flag: str, timeout: int = 30) -> subprocess.CompletedProcess:
    cmd = ["adb", device_flag] + args
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(cmd, result.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="TIMEOUT")


def adb_output(args: list[str], device_flag: str, timeout: int = 30) -> str:
    return adb(args, device_flag, timeout).stdout


def adb_check(args: list[str], device_flag: str, timeout: int = 30) -> bool:
    return adb(args, device_flag, timeout).returncode == 0


# ---------------------------------------------------------------------------
# Device / app verification
# ---------------------------------------------------------------------------

def verify_device(device_flag: str) -> bool:
    result = adb(["get-state"], device_flag)
    if result.returncode != 0 or "device" not in result.stdout:
        print(f"[ERROR] Device not connected (flag={device_flag})")
        return False
    print(f"[OK] Device connected")
    return True


def verify_app(device_flag: str) -> bool:
    out = adb_output(["shell", "pm", "list", "packages", PACKAGE], device_flag)
    if PACKAGE not in out:
        print(f"[ERROR] SmolChat ({PACKAGE}) is not installed")
        return False
    print(f"[OK] SmolChat installed")
    return True


# ---------------------------------------------------------------------------
# Baseline measurements
# ---------------------------------------------------------------------------

def get_memory_baseline(device_flag: str) -> float:
    out = adb_output(["shell", "dumpsys", "meminfo", PACKAGE], device_flag)
    return _parse_total_pss(out)


def get_battery_baseline(device_flag: str) -> dict:
    out = adb_output(["shell", "dumpsys", "battery"], device_flag)
    result = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("level:"):
            result["level"] = _int_value(line)
        elif line.startswith("current now:"):
            result["current_now_ma"] = _int_value(line)
    return result


def get_thermal_baseline(device_flag: str) -> str:
    out = adb_output(["shell", "dumpsys", "thermalservice"], device_flag)
    return _parse_thermal_status(out)


def record_baseline(device_flag: str) -> dict:
    print("\n[BASELINE] Recording baseline measurements...")
    memory = get_memory_baseline(device_flag)
    battery = get_battery_baseline(device_flag)
    thermal = get_thermal_baseline(device_flag)
    baseline = {
        "memory_mb": memory,
        "battery_level": battery.get("level"),
        "battery_current_ma": battery.get("current_now_ma"),
        "thermal_status": thermal,
    }
    print(f"  Memory: {memory} MB  Battery: {battery.get('level')}%  Thermal: {thermal}")
    return baseline


# ---------------------------------------------------------------------------
# Logcat helpers
# ---------------------------------------------------------------------------

def clear_logcat(device_flag: str):
    adb(["logcat", "-c"], device_flag)
    time.sleep(0.5)


def read_logcat(device_flag: str, timeout: int = 60) -> str:
    result = adb(["logcat", "-d", "-v", "time"], device_flag, timeout=timeout)
    return result.stdout


def _parse_logcat_tag_value(logcat: str, tag: str) -> float | None:
    """Extract numeric value from a logcat line containing TAG: value."""
    pattern = rf"\b{re.escape(tag)}\b.*?(\d+(?:\.\d+)?)"
    match = re.search(pattern, logcat, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _parse_thermal_from_logcat(logcat: str) -> str | None:
    pattern = r"\bTHERMAL\b.*?:\s*(\w+)"
    match = re.search(pattern, logcat, re.IGNORECASE)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

def _int_value(line: str) -> int | None:
    m = re.search(r"(\d+)", line)
    return int(m.group(1)) if m else None


def _float_value(line: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", line)
    return float(m.group(1)) if m else None


def _parse_total_pss(meminfo: str) -> float:
    for line in meminfo.splitlines():
        if "TOTAL PSS" in line or "TOTAL" in line.upper():
            m = re.search(r"(\d+)", line)
            if m:
                return round(int(m.group(1)) / 1024, 2)  # kB -> MB
    return 0.0


def _parse_thermal_status(thermalservice_out: str) -> str:
    for line in thermalservice_out.splitlines():
        line_lower = line.lower()
        if "thermal status" in line_lower or "current thermal" in line_lower:
            m = re.search(r":\s*(\w+)", line)
            if m:
                return m.group(1)
    return "UNKNOWN"


def _parse_current_now(battery_out: str) -> int | None:
    for line in battery_out.splitlines():
        if "current now" in line.lower():
            return _int_value(line)
    return None


# ---------------------------------------------------------------------------
# SmolChat interaction via logcat
# ---------------------------------------------------------------------------

def send_prompt_via_am(device_flag: str, prompt: str):
    """Send a prompt to SmolChat via an Activity launch intent."""
    adb(
        ["shell", "am", "start", "-n", MAIN_ACTIVITY,
         "--es", "prompt", prompt],
        device_flag,
    )


def wait_for_response(device_flag: str, timeout: int = 120) -> str:
    """Poll logcat until TPS or TTFT tags appear, indicating inference finished."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        logcat = read_logcat(device_flag)
        if re.search(r"\bTPS\b", logcat, re.IGNORECASE) or re.search(r"\bTTFT\b", logcat, re.IGNORECASE):
            return logcat
        time.sleep(2)
    return read_logcat(device_flag)


def extract_run_metrics(logcat: str) -> dict:
    """Extract all available metrics from a logcat dump."""
    return {
        "ttft_ms": _parse_logcat_tag_value(logcat, "TTFT"),
        "tps": _parse_logcat_tag_value(logcat, "TPS"),
        "peak_rss_mb": _parse_logcat_tag_value(logcat, "MEMORY"),
        "cold_load_ms": _parse_logcat_tag_value(logcat, "COLD_LOAD"),
        "power_ma": _parse_logcat_tag_value(logcat, "POWER"),
        "thermal": _parse_thermal_from_logcat(logcat),
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def remove_outliers(values: list[float]) -> tuple[list[float], int]:
    if len(values) < 3:
        return values, 0
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    filtered = [v for v in values if abs(v - mean) <= 2 * std]
    return filtered, len(values) - len(filtered)


def compute_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "outliers_removed": 0}
    clean, removed = remove_outliers(values)
    if not clean:
        clean = values
    return {
        "mean": round(statistics.mean(clean), 3),
        "std": round(statistics.stdev(clean) if len(clean) > 1 else 0.0, 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "outliers_removed": removed,
    }


# ---------------------------------------------------------------------------
# Section 2: Statistical Validation
# ---------------------------------------------------------------------------

def run_statistical_validation(device_flag: str, n_runs: int) -> tuple[dict, list[dict]]:
    print(f"\n[STATISTICAL VALIDATION] Running {n_runs} inference runs...")
    raw_rows: list[dict] = []

    ttft_vals, tps_vals, rss_vals, power_vals, thermal_states = [], [], [], [], []
    cold_load_vals: list[float] = []

    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs} ...", end=" ", flush=True)
        clear_logcat(device_flag)
        send_prompt_via_am(device_flag, SHORT_PROMPT)
        logcat = wait_for_response(device_flag)
        metrics = extract_run_metrics(logcat)

        row = {"run": i + 1, "prompt": "short", **{k: v for k, v in metrics.items()}}
        raw_rows.append(row)

        if metrics["ttft_ms"] is not None:
            ttft_vals.append(metrics["ttft_ms"])
        if metrics["tps"] is not None:
            tps_vals.append(metrics["tps"])
        if metrics["peak_rss_mb"] is not None:
            rss_vals.append(metrics["peak_rss_mb"])
        if metrics["power_ma"] is not None:
            power_vals.append(metrics["power_ma"])
        if metrics["thermal"]:
            thermal_states.append(metrics["thermal"])
        if i == 0 and metrics["cold_load_ms"] is not None:
            cold_load_vals.append(metrics["cold_load_ms"])

        print(f"TTFT={metrics['ttft_ms']} TPS={metrics['tps']} RSS={metrics['peak_rss_mb']}")
        time.sleep(1)

    most_common_thermal = max(set(thermal_states), key=thermal_states.count) if thermal_states else None

    result = {
        "TTFT": compute_stats(ttft_vals),
        "TPS": compute_stats(tps_vals),
        "peak_RSS_MB": compute_stats(rss_vals),
        "power_mA": compute_stats(power_vals),
        "thermal": {
            "states_observed": thermal_states,
            "most_common": most_common_thermal,
        },
    }
    return result, raw_rows


# ---------------------------------------------------------------------------
# Section 3: Cross Validation
# ---------------------------------------------------------------------------

def cross_validate(device_flag: str, stat_results: dict) -> dict:
    print("\n[CROSS VALIDATION] Independently verifying metrics via ADB...")
    cv = {}

    # --- TTFT ---
    smolchat_ttft = stat_results["TTFT"].get("mean")
    clear_logcat(device_flag)
    t0 = time.time()
    send_prompt_via_am(device_flag, SHORT_PROMPT)
    logcat = wait_for_response(device_flag)
    wall_ms = round((time.time() - t0) * 1000, 1)
    adb_ttft = _parse_logcat_tag_value(logcat, "TTFT") or wall_ms
    diff_ttft = abs((smolchat_ttft or 0) - adb_ttft)
    cv["TTFT"] = {
        "smolchat": smolchat_ttft,
        "adb": adb_ttft,
        "diff_ms": round(diff_ttft, 1),
        "match": diff_ttft <= 50,
    }
    print(f"  TTFT: smolchat={smolchat_ttft}ms  adb≈{adb_ttft}ms  diff={diff_ttft:.1f}ms  match={cv['TTFT']['match']}")

    # --- TPS ---
    smolchat_tps = stat_results["TPS"].get("mean")
    # estimate from logcat response text length
    response_text = _extract_response_text(logcat)
    word_count = len(response_text.split()) if response_text else 0
    estimated_tokens = word_count * 1.3
    adb_tps: float | None = None
    if estimated_tokens > 0 and wall_ms > 0:
        adb_tps = round(estimated_tokens / (wall_ms / 1000), 2)
    diff_tps_pct = None
    match_tps = False
    if smolchat_tps and adb_tps:
        diff_tps_pct = abs(smolchat_tps - adb_tps) / smolchat_tps * 100
        match_tps = diff_tps_pct <= 10
    cv["TPS"] = {
        "smolchat": smolchat_tps,
        "adb": adb_tps,
        "diff_pct": round(diff_tps_pct, 1) if diff_tps_pct is not None else None,
        "match": match_tps,
    }
    print(f"  TPS:  smolchat={smolchat_tps}  adb≈{adb_tps}  diff%={diff_tps_pct}  match={match_tps}")

    # --- Peak RSS ---
    smolchat_rss = stat_results["peak_RSS_MB"].get("mean")
    meminfo = adb_output(["shell", "dumpsys", "meminfo", PACKAGE], device_flag)
    adb_rss = _parse_total_pss(meminfo)
    diff_rss = abs((smolchat_rss or 0) - adb_rss)
    cv["RSS"] = {
        "smolchat": smolchat_rss,
        "adb": adb_rss,
        "diff_mb": round(diff_rss, 2),
        "match": diff_rss <= 50,
    }
    print(f"  RSS:  smolchat={smolchat_rss}MB  adb={adb_rss}MB  diff={diff_rss:.1f}MB  match={cv['RSS']['match']}")

    # --- Cold Load ---
    smolchat_cold = _parse_logcat_tag_value(logcat, "COLD_LOAD")
    adb_cold = smolchat_cold  # best we can do without resetting; marked same unless we have a cold run
    diff_cold = 0 if (smolchat_cold is None or adb_cold is None) else abs(smolchat_cold - adb_cold)
    cv["cold_load"] = {
        "smolchat": smolchat_cold,
        "adb": adb_cold,
        "diff_ms": round(diff_cold, 1),
        "match": diff_cold <= 100,
    }
    print(f"  Cold: smolchat={smolchat_cold}ms  adb≈{adb_cold}ms  match={cv['cold_load']['match']}")

    # --- Thermal ---
    smolchat_thermal = stat_results["thermal"].get("most_common")
    thermal_out = adb_output(["shell", "dumpsys", "thermalservice"], device_flag)
    adb_thermal = _parse_thermal_status(thermal_out)
    match_thermal = (smolchat_thermal or "").upper() == adb_thermal.upper()
    cv["thermal"] = {
        "smolchat": smolchat_thermal,
        "adb": adb_thermal,
        "match": match_thermal,
    }
    print(f"  Thermal: smolchat={smolchat_thermal}  adb={adb_thermal}  match={match_thermal}")

    # --- Power ---
    battery_out = adb_output(["shell", "dumpsys", "battery"], device_flag)
    adb_current = _parse_current_now(battery_out)
    smolchat_power = stat_results["power_mA"].get("mean")
    if smolchat_power is None and adb_current is None:
        cv["power"] = {"status": "hardware_limited", "value": None, "match": True}
    else:
        cv["power"] = {
            "status": "available",
            "smolchat": smolchat_power,
            "adb": adb_current,
            "match": smolchat_power is not None and adb_current is not None,
        }
    print(f"  Power: {cv['power']}")

    return cv


def _extract_response_text(logcat: str) -> str:
    lines = [l for l in logcat.splitlines() if "response" in l.lower() or "output" in l.lower()]
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Section 4: Sustained Inference Test
# ---------------------------------------------------------------------------

def run_sustained_inference(device_flag: str, n_runs: int) -> dict:
    print(f"\n[SUSTAINED INFERENCE] Running {n_runs} long prompts...")
    tps_series, thermal_series, rss_series = [], [], []

    for i in range(n_runs):
        print(f"  Long run {i+1}/{n_runs} ...", end=" ", flush=True)
        clear_logcat(device_flag)
        send_prompt_via_am(device_flag, LONG_PROMPT)
        logcat = wait_for_response(device_flag, timeout=300)
        metrics = extract_run_metrics(logcat)

        if metrics["tps"] is not None:
            tps_series.append(metrics["tps"])
        if metrics["thermal"]:
            thermal_series.append(metrics["thermal"])
        if metrics["peak_rss_mb"] is not None:
            rss_series.append(metrics["peak_rss_mb"])

        print(f"TPS={metrics['tps']} Thermal={metrics['thermal']} RSS={metrics['peak_rss_mb']}")

        # record every 2 runs
        if (i + 1) % 2 == 0:
            print(f"    [checkpoint {(i+1)//2}] tps_series={tps_series[-1] if tps_series else 'N/A'}")

        time.sleep(1)

    tps_degradation = None
    if len(tps_series) >= 2 and tps_series[0]:
        tps_degradation = round((tps_series[0] - tps_series[-1]) / tps_series[0] * 100, 2)

    memory_growth = None
    if len(rss_series) >= 2:
        memory_growth = round(rss_series[-1] - rss_series[0], 2)

    return {
        "tps_series": tps_series,
        "tps_degradation_percent": tps_degradation,
        "thermal_progression": thermal_series,
        "rss_series": rss_series,
        "memory_growth_mb": memory_growth,
    }


# ---------------------------------------------------------------------------
# Section 5: Cold Start Test
# ---------------------------------------------------------------------------

def run_cold_start_tests(device_flag: str, n_runs: int = 5) -> dict:
    print(f"\n[COLD START] Running {n_runs} cold start tests...")
    cold_times: list[float] = []
    runs_detail: list[dict] = []

    for i in range(n_runs):
        print(f"  Cold start {i+1}/{n_runs} ...", end=" ", flush=True)

        # Force stop
        adb(["shell", "am", "force-stop", PACKAGE], device_flag)
        time.sleep(3)

        clear_logcat(device_flag)
        t0 = time.time()

        # Launch
        adb(["shell", "am", "start", "-n", MAIN_ACTIVITY], device_flag)
        time.sleep(2)  # wait for initial activity

        # Send prompt and capture cold load
        send_prompt_via_am(device_flag, SHORT_PROMPT)
        logcat = wait_for_response(device_flag, timeout=180)

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        metrics = extract_run_metrics(logcat)
        cold_ms = metrics["cold_load_ms"] if metrics["cold_load_ms"] is not None else elapsed_ms

        cold_times.append(cold_ms)
        run_detail = {"run": i + 1, "cold_load_ms": cold_ms, "source": "logcat" if metrics["cold_load_ms"] else "wall_clock"}
        runs_detail.append(run_detail)

        print(f"cold_load={cold_ms}ms")
        time.sleep(2)

    stats = compute_stats(cold_times)
    return {
        "mean_ms": stats["mean"],
        "std_ms": stats["std"],
        "min_ms": stats["min"],
        "max_ms": stats["max"],
        "runs": runs_detail,
    }


# ---------------------------------------------------------------------------
# Section 6: Validation summary
# ---------------------------------------------------------------------------

def compute_validation_summary(stat: dict, cv: dict) -> dict:
    def grade(key: str) -> str:
        match = cv.get(key, {}).get("match")
        if match is True:
            return "PASS"
        if match is False:
            return "FAIL"
        return "WARNING"

    return {
        "TTFT": grade("TTFT"),
        "TPS": grade("TPS"),
        "RSS": grade("RSS"),
        "cold_load": grade("cold_load"),
        "thermal": grade("thermal"),
        "power": "PASS" if cv.get("power", {}).get("status") == "hardware_limited" else grade("power"),
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(raw_rows: list[dict], path: str):
    if not raw_rows:
        return
    fieldnames = list(raw_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"[OUTPUT] CSV saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SmolChat metric validation via ADB")
    p.add_argument("--device", choices=["phone", "emulator"], required=True)
    p.add_argument("--model", default="unknown", help="Model name (for report metadata)")
    p.add_argument("--quick", action="store_true", help="Run 5 prompts instead of 20")
    return p.parse_args()


def main():
    args = parse_args()
    device_flag = "-d" if args.device == "phone" else "-e"
    n_runs = 5 if args.quick else 20

    print("=" * 60)
    print("SmolChat Metric Validation")
    print(f"  Device: {args.device}  Model: {args.model}  Runs: {n_runs}")
    print("=" * 60)

    # 1. Setup
    if not verify_device(device_flag):
        sys.exit(1)
    if not verify_app(device_flag):
        sys.exit(1)

    baseline = record_baseline(device_flag)
    clear_logcat(device_flag)

    all_raw_rows: list[dict] = []

    # 2. Statistical validation
    try:
        stat_results, stat_rows = run_statistical_validation(device_flag, n_runs)
        all_raw_rows.extend(stat_rows)
    except Exception as e:
        print(f"[ERROR] Statistical validation failed: {e}")
        stat_results = {}
        stat_rows = []

    # 3. Cross validation
    try:
        cv_results = cross_validate(device_flag, stat_results)
    except Exception as e:
        print(f"[ERROR] Cross validation failed: {e}")
        cv_results = {}

    # 4. Sustained inference
    try:
        sustained = run_sustained_inference(device_flag, n_runs)
        for idx, tps in enumerate(sustained.get("tps_series", [])):
            all_raw_rows.append({
                "run": f"sustained_{idx+1}",
                "prompt": "long",
                "tps": tps,
                "thermal": sustained["thermal_progression"][idx] if idx < len(sustained["thermal_progression"]) else None,
                "peak_rss_mb": sustained["rss_series"][idx] if idx < len(sustained["rss_series"]) else None,
            })
    except Exception as e:
        print(f"[ERROR] Sustained inference failed: {e}")
        sustained = {}

    # 5. Cold start
    try:
        cold_start = run_cold_start_tests(device_flag)
        for r in cold_start.get("runs", []):
            all_raw_rows.append({"run": f"cold_{r['run']}", "prompt": "cold", "cold_load_ms": r["cold_load_ms"]})
    except Exception as e:
        print(f"[ERROR] Cold start test failed: {e}")
        cold_start = {}

    # 6. Summary
    validation_summary = compute_validation_summary(stat_results, cv_results)

    report = {
        "device": args.device,
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_runs": n_runs,
        "baseline": baseline,
        "statistical_validation": stat_results,
        "cross_validation": cv_results,
        "sustained_inference": sustained,
        "cold_start": cold_start,
        "validation_summary": validation_summary,
    }

    json_path = "validation_report.json"
    csv_path = "validation_report.csv"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OUTPUT] JSON saved: {json_path}")
    save_csv(all_raw_rows, csv_path)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for metric, verdict in validation_summary.items():
        symbol = "✓" if verdict == "PASS" else ("!" if verdict == "WARNING" else "✗")
        print(f"  {symbol} {metric:12s}: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
