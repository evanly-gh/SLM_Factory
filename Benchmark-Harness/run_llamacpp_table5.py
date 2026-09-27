#!/usr/bin/env python3
"""
llama.cpp Table 5 replication (arXiv 2607.05475): 256-token prefill and
256-token decode throughput (+ optional energy) for Qwen2.5-1.5B/7B and
Llama3.2-1B/3B (Q4_K_M) on the CPU and OpenCL (Adreno) backends, via llama-bench.

Prefill (pp256) and decode (tg256) run as separate llama-bench invocations, each
preceded by a short CPU burst (see cpu_burst()). llama-bench does its own warm-up
before the recorded reps (paper: 1 warm-up + >=3 trials).

Energy (--energy): one Perfetto battery-counter trace over the whole run (see
bench_common.EnergyTrace); each llama-bench invocation's window is bracketed with
the phone's own clock and integrated afterwards:
    avg_power_W  = energy_mJ / window_s / 1000
    mJ_per_token = avg_power_W / tokens_per_s * 1000
so model load and warm-up inside the window only dilute the average power, they
aren't billed as extra tokens. Energy is valid only while the phone is not
externally powered (use wireless adb); each row records whether that held.

Device layout (under /data/local/tmp):
    llama_bench/       CPU build    (build-cpu: GGML_OPENCL=OFF)
    llama_bench_gpu/   OpenCL build (build-gpu: GGML_OPENCL=ON)
    <model>.gguf       the four verified Q4_K_M models

Results: Benchmark-Harness/llamacpp_table5_results/<timestamp>/
    TABLE5_llamacpp.md   human-readable Table 5
    summary.csv          one row per model x backend
    run_meta.json        device/build/params/idle baseline/device states
    raw/                 llama-bench JSON per phase (+ traces/ with --energy)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_common  # noqa: E402

DEV_DIR = "/data/local/tmp"

BACKENDS = {
    # name: (device bin dir, -ngl, expected llama-bench "backends" field)
    "cpu": ("llama_bench", 0, "CPU"),
    "gpu": ("llama_bench_gpu", 99, "OpenCL"),
}
MODELS = {
    # name: expected file size (bytes) - verified against the GGUF tensor table
    "qwen2.5-1.5b-q4_k_m": 986048128,
    "llama-3.2-1b-q4_k_m": 807690432,
    "llama-3.2-3b-q4_k_m": 2019373696,
    "qwen2.5-7b-q4_k_m": 4683073568,
}


class Adb:
    """Minimal adb wrapper with the run(args, timeout) interface bench_common expects."""

    def run(self, args: list, timeout: int = 600) -> subprocess.CompletedProcess:
        return subprocess.run(["adb", *args], capture_output=True, text=True, timeout=timeout)


ADB = Adb()


def sh(cmd: str, timeout: int = 1800) -> str:
    return ADB.run(["shell", cmd], timeout=timeout).stdout


def device_epoch_ms() -> int:
    """The phone's own wall clock, so energy windows line up with the Perfetto trace."""
    return int(sh("date +%s%N").strip()) // 1_000_000


def preflight(args) -> dict:
    devs = ADB.run(["devices"]).stdout.split("\n")[1:]
    if not any(d.strip().endswith("device") for d in devs):
        sys.exit("no adb device attached")
    for b in args.backends:
        bin_dir, _, _ = BACKENDS[b]
        if sh(f"test -x {DEV_DIR}/{bin_dir}/llama-bench && echo ok").strip() != "ok":
            sys.exit(f"missing {DEV_DIR}/{bin_dir}/llama-bench")
    if "gpu" in args.backends:
        listing = sh(f"cd {DEV_DIR}/llama_bench_gpu && LD_LIBRARY_PATH=. ./llama-bench --list-devices 2>/dev/null")
        if "GPUOpenCL" not in listing:
            sys.exit(f"OpenCL device not listed by llama-bench:\n{listing}")
    for m in args.models:
        size = sh(f"stat -c %s {DEV_DIR}/{m}.gguf 2>/dev/null").strip()
        if size != str(MODELS[m]):
            sys.exit(f"{m}.gguf size {size or 'missing'} != expected {MODELS[m]}")
    return {
        "model": sh("getprop ro.product.model").strip(),
        "soc": sh("getprop ro.soc.model").strip() or sh("getprop ro.board.platform").strip(),
        "android": sh("getprop ro.build.version.release").strip(),
    }


def cpu_burst(seconds: int):
    """Short all-core CPU load so the SoC leaves its idle DVFS state before a measurement:
    after idle, OpenCL prefill measured ~28 t/s vs ~100 t/s, and CPU prefill ~89 vs ~233 t/s,
    compared with right after CPU activity."""
    sh(f"cd {DEV_DIR}/llama_bench && LD_LIBRARY_PATH=. timeout {seconds} ./llama-bench "
       f"-m {DEV_DIR}/llama-3.2-1b-q4_k_m.gguf -p 0 -n 4096 -r 100 >/dev/null 2>&1")


def set_controls(on: bool):
    if on:
        sh("cmd connectivity airplane-mode enable")
        sh("input keyevent 223")  # KEYCODE_SLEEP
    else:
        sh("input keyevent 224")  # KEYCODE_WAKEUP
        sh("cmd connectivity airplane-mode disable")


def fmt(mean, std=None, nd=2):
    if mean is None:
        return "n/a"
    return f"{mean:.{nd}f}" if std is None else f"{mean:.{nd}f} ± {std:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=list(BACKENDS))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--n-prompt", type=int, default=256)
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pp-reps", type=int, default=10,
                    help="prefill reps; >3 so a ~1s-per-rep phase spans several fuel-gauge updates")
    ap.add_argument("--tg-reps", type=int, default=8,
                    help="decode reps; decode warm-up is only 1 token, so trial 1 runs under-ramped")
    ap.add_argument("--cooldown", type=int, default=30, help="seconds between runs")
    ap.add_argument("--prime-seconds", type=int, default=5,
                    help="CPU burst length right before each measurement (0 = off); see cpu_burst()")
    ap.add_argument("--energy", action="store_true",
                    help="record energy via Perfetto (valid only while unplugged, i.e. wireless adb)")
    ap.add_argument("--idle-seconds", type=int, default=30, help="idle-power baseline length (with --energy)")
    ap.add_argument("--no-device-controls", dest="controls", action="store_false",
                    help="skip airplane mode + screen off")
    ap.add_argument("--out-dir", type=Path, default=HERE / "llamacpp_table5_results")
    args = ap.parse_args()

    out = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    raw = out / "raw"
    raw.mkdir(parents=True)

    device = preflight(args)
    meta = {"device": device, "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "start": datetime.now().isoformat(), "state_start": bench_common.device_state(ADB)}
    print(f"device: {device}\nresults -> {out}")
    if args.energy and meta["state_start"]["externally_powered"]:
        print("[WARN] --energy while externally powered: energy will be recorded but flagged invalid.")

    trace, windows, rows = None, [], []
    if args.controls:
        set_controls(True)
    try:
        if args.energy:
            trace = bench_common.EnergyTrace(ADB, out / "traces")
            trace.start()
            print(f">> idle baseline ({args.idle_seconds}s)")
            t0 = device_epoch_ms()
            time.sleep(args.idle_seconds)
            windows.append(("idle", t0, device_epoch_ms()))
        for model in args.models:
            for backend in args.backends:
                bin_dir, ngl, expect_backend = BACKENDS[backend]
                row = {"model": model, "backend": backend}
                for phase in ("prefill", "decode"):
                    n_p, n_g, reps = ((args.n_prompt, 0, args.pp_reps) if phase == "prefill"
                                      else (0, args.n_gen, args.tg_reps))
                    tag = f"{model}_{backend}_{phase}"
                    print(f">> {tag} (reps={reps})", flush=True)
                    if args.prime_seconds > 0:
                        cpu_burst(args.prime_seconds)
                    state = bench_common.device_state(ADB)
                    bench = (f"cd {DEV_DIR}/{bin_dir} && LD_LIBRARY_PATH=. ./llama-bench "
                             f"-m {DEV_DIR}/{model}.gguf -p {n_p} -n {n_g} -r {reps} "
                             f"-t {args.threads} -ngl {ngl} -o json 2>/dev/null")
                    t0 = device_epoch_ms() if args.energy else None
                    (raw / f"{tag}.json").write_text(sh(bench))
                    if args.energy:
                        windows.append((tag, t0, device_epoch_ms()))
                    r = json.loads((raw / f"{tag}.json").read_text())[0]
                    if r["backends"] != expect_backend:
                        print(f"   !! backend reported {r['backends']!r}, expected {expect_backend!r}")
                    meta["device"]["llama_build"] = f"{r['build_commit']} (build {r['build_number']})"
                    row.update({
                        f"{phase}_samples_ts": " ".join(f"{x:.1f}" for x in r["samples_ts"]),
                        f"{phase}_tps": r["avg_ts"],
                        f"{phase}_tps_std": r["stddev_ts"],
                        f"{phase}_reps": len(r["samples_ts"]),
                        f"{phase}_battery_temp_c": state["battery_temp_c"],
                        f"{phase}_cpu_caps_khz": state["cpu_caps_khz"],
                        f"{phase}_externally_powered": state["externally_powered"],
                        "llama_backend": r["backends"],
                        "threads": r["n_threads"],
                    })
                    print(f"   {r['avg_ts']:.2f} ± {r['stddev_ts']:.2f} t/s", flush=True)
                    time.sleep(args.cooldown)
                rows.append(row)
                write_summary(out, rows, meta)
    finally:
        if args.controls:
            set_controls(False)
        if trace is not None:
            trace.stop()
        meta["end"] = datetime.now().isoformat()
        meta["state_end"] = bench_common.device_state(ADB)

    if trace is not None:
        add_energy(rows, meta, trace.integrate(windows))
    write_summary(out, rows, meta)
    print(f"done -> {out / 'TABLE5_llamacpp.md'}")


def add_energy(rows: list, meta: dict, integrated: dict):
    idle = integrated.get("idle", {})
    idle_mw = idle.get("avg_power_mw") if "energy_mj" in idle and idle.get("charging_fraction") == 0 else None
    meta["idle_power_mw"] = idle_mw
    for row in rows:
        for phase in ("prefill", "decode"):
            tag = f"{row['model']}_{row['backend']}_{phase}"
            e = bench_common.energy_fields(integrated.get(tag), idle_mw, None, row.get(f"{phase}_externally_powered"))
            tps = row.get(f"{phase}_tps")
            power_w = e["avg_power_mw"] / 1000 if e.get("avg_power_mw") is not None else None
            net_w = ((e["avg_power_mw"] - idle_mw) / 1000) if (power_w is not None and idle_mw is not None) else None
            row.update({
                f"{phase}_power_w": power_w,
                f"{phase}_mj_per_token": power_w / tps * 1000 if (power_w and tps) else None,
                f"{phase}_net_mj_per_token": net_w / tps * 1000 if (net_w is not None and tps) else None,
                f"{phase}_energy_valid": e["energy_valid"],
                f"{phase}_energy_invalid_reason": e["energy_invalid_reason"],
            })


def write_summary(out: Path, rows: list[dict], meta: dict):
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    if not rows:
        return
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "backend"), k))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    p = meta["params"]
    lines = [
        "# llama.cpp Table 5 replication (arXiv 2607.05475)",
        "",
        f"- Device: {meta['device']['model']} ({meta['device']['soc']}), Android {meta['device']['android']}",
        f"- llama.cpp: {meta['device'].get('llama_build', '?')}; Q4_K_M; threads={p['threads']}; "
        f"each run preceded by a {p['prime_seconds']}s CPU burst",
        f"- Prefill: pp{p['n_prompt']} x {p['pp_reps']} reps; decode: tg{p['n_gen']} x {p['tg_reps']} reps "
        "(each after llama-bench's warm-up)",
        f"- Run: {meta['start']} -> {meta.get('end', 'in progress')}",
        "",
    ]
    energy_done = p["energy"] and any("prefill_energy_valid" in r for r in rows)
    if not energy_done:
        lines += [
            "| Model | Backend | Prefill t/s (pp%d) | Decode t/s (tg%d) | Prefill per-rep t/s | Decode per-rep t/s |"
            % (p["n_prompt"], p["n_gen"]),
            "|---|---|--:|--:|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['model']} | {r.get('llama_backend', r['backend'])} "
                f"| {fmt(r.get('prefill_tps'), r.get('prefill_tps_std'))} "
                f"| {fmt(r.get('decode_tps'), r.get('decode_tps_std'))} "
                f"| {r.get('prefill_samples_ts', '')} | {r.get('decode_samples_ts', '')} |"
            )
        (out / "TABLE5_llamacpp.md").write_text("\n".join(lines) + "\n")
        return
    lines += [
        f"- Idle power: {fmt(meta.get('idle_power_mw'), nd=0)} mW",
        "",
        "| Model | Backend | Prefill t/s | Decode t/s | Prefill W | Prefill mJ/tok (net) | Decode W | "
        "Decode mJ/tok (net) | Energy valid |",
        "|---|---|--:|--:|--:|--:|--:|--:|:-:|",
    ]
    for r in rows:
        valid = "yes" if r.get("prefill_energy_valid") and r.get("decode_energy_valid") else "NO"
        lines.append(
            f"| {r['model']} | {r.get('llama_backend', r['backend'])} "
            f"| {fmt(r.get('prefill_tps'), r.get('prefill_tps_std'))} "
            f"| {fmt(r.get('decode_tps'), r.get('decode_tps_std'))} "
            f"| {fmt(r.get('prefill_power_w'))} | {fmt(r.get('prefill_mj_per_token'), nd=1)} "
            f"({fmt(r.get('prefill_net_mj_per_token'), nd=1)}) "
            f"| {fmt(r.get('decode_power_w'))} | {fmt(r.get('decode_mj_per_token'), nd=1)} "
            f"({fmt(r.get('decode_net_mj_per_token'), nd=1)}) | {valid} |"
        )
    lines += [
        "",
        "Energy is integrated from Perfetto battery counters over each llama-bench invocation (includes "
        "model load + warm-up, which only dilutes average power). Rows marked NO were externally powered, "
        "charging, or too short for the ~5 s fuel-gauge update period; see summary.csv for the reason.",
    ]
    (out / "TABLE5_llamacpp.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
