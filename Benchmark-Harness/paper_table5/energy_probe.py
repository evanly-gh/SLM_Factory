#!/usr/bin/env python3
"""
Energy-method probe: find which energy sources a phone exposes, measure the same workloads with all of
them at once, and recommend the most trustworthy one for the Table 5 harness.

  python energy_probe.py discover [--serial S]   what the phone exposes (works over USB)
  python energy_probe.py wireless [--serial S]   switch a USB-connected phone to adb over Wi-Fi
  python energy_probe.py collect  [--serial S]   measure (phone must be UNPLUGGED unless it has rail counters)
  python energy_probe.py analyze  <run dir>      re-run the analysis on a finished collect run

Methods, all recorded during the same windows:
  powercap:<zone>     Qualcomm power-telemetry counters, /sys/class/powercap/*/energy_uj (the paper's
                      PowerBench source; SoC rails; usually needs root)
  rail:<name>         Android Power Stats HAL rails via Perfetto collect_power_rails (no root, vendor-dependent)
  sysfs:current_now   battery current x voltage from /sys/class/power_supply, polled every 100 ms on the phone
  sysfs:current_avg   the gauge's own averaged current, same polling
  sysfs:power_now     battery power, if the driver exposes it
  sysfs:charge_counter coulomb counter delta x mean voltage
  perfetto:current    battery current x voltage from the battery HAL, via Perfetto
  perfetto:charge     battery HAL charge counter delta x mean voltage

Workloads: idle, one busy core, all cores busy, interleaved and repeated. There is no ground truth
without an external meter, so methods are ranked on evidence that can be measured: direct rail counters
first; then repeatability of the same load (coefficient of variation), how often the reading updates,
and agreement of the load-minus-idle power with the other methods.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import bench_common  # noqa: E402
from devenv import Adb, list_devices  # noqa: E402

DEV_DIR = "/data/local/tmp/energy_probe"
TRACE_REMOTE = "/data/misc/perfetto-traces/energy_probe.pftrace"
SUPPLY_FILES = ("current_now", "current_avg", "voltage_now", "voltage_avg", "charge_counter",
                "power_now", "power_avg", "energy_now")
SAMPLED_CURRENT = ("current_now", "current_avg")

PERFETTO_CONFIG = """
buffers { size_kb: 32768 fill_policy: DISCARD }
data_sources {
  config {
    name: "android.power"
    android_power_config {
      battery_poll_ms: 100
      battery_counters: BATTERY_COUNTER_CURRENT
      battery_counters: BATTERY_COUNTER_VOLTAGE
      battery_counters: BATTERY_COUNTER_CHARGE
      collect_power_rails: true
    }
  }
}
write_into_file: true
file_write_period_ms: 5000
max_file_size_bytes: 500000000
duration_ms: 10800000
"""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(adb: Adb) -> dict:
    prop = lambda k: adb.sh(f"getprop {k}").strip()  # noqa: E731
    info = {
        "serial": adb.serial,
        "manufacturer": prop("ro.product.manufacturer"),
        "model": prop("ro.product.model"),
        "soc": prop("ro.soc.model") or prop("ro.board.platform"),
        "android": prop("ro.build.version.release"),
        "sdk": prop("ro.build.version.sdk"),
        "cpus_online": adb.sh("cat /sys/devices/system/cpu/online").strip(),
        "ncpu": int(adb.sh("ls -d /sys/devices/system/cpu/cpu[0-9]* | wc -l").strip() or 8),
    }
    su = adb.sh("su -c id 2>&1").strip()
    info["root"] = {"su_output": su, "rooted": "uid=0" in su}

    zones = []
    listing = adb.sh("for z in /sys/class/powercap/*; do [ -e $z/energy_uj ] || continue; "
                     "echo \"$z|$(cat $z/name 2>/dev/null)|$(cat $z/energy_uj 2>&1)|"
                     "$(cat $z/max_energy_range_uj 2>/dev/null)\"; done")
    for line in listing.splitlines():
        p = line.split("|")
        if len(p) == 4:
            zones.append({"path": p[0], "name": p[1], "shell_readable": p[2].strip().isdigit(),
                          "max_range_uj": int(p[3]) if p[3].strip().isdigit() else None})
    if info["root"]["rooted"]:
        for z in zones:
            if not z["shell_readable"]:
                z["su_readable"] = adb.sh(f"su -c 'cat {z['path']}/energy_uj' 2>&1").strip().isdigit()
    info["powercap"] = {"zones": zones,
                        "usable": any(z["shell_readable"] or z.get("su_readable") for z in zones),
                        "needs_su": bool(zones) and not any(z["shell_readable"] for z in zones)}

    services = [s.strip() for s in adb.sh("dumpsys -l").splitlines() if "power" in s.lower()]
    stats = next((s for s in services if "power.stats" in s.lower()), None)
    info["powerstats"] = {"services": services, "service": stats,
                          "dump_head": adb.sh(f"dumpsys {stats} 2>&1 | head -40") if stats else None}

    supplies = []
    out = adb.sh("for d in /sys/class/power_supply/*; do t=$(cat $d/type 2>/dev/null); "
                 f"for f in {' '.join(SUPPLY_FILES)}; do [ -e $d/$f ] && "
                 "echo \"$d|$t|$f|$(cat $d/$f 2>&1)\"; done; done")
    for line in out.splitlines():
        p = line.split("|")
        if len(p) == 4:
            supplies.append({"dir": p[0], "type": p[1], "file": p[2], "value": p[3].strip(),
                             "readable": bool(re.fullmatch(r"-?\d+", p[3].strip()))})
    bats = sorted({s["dir"] for s in supplies if s["type"] == "Battery"},
                  key=lambda d: (not d.endswith("/battery"), d))
    info["battery_dir"] = bats[0] if bats else None
    info["power_supply"] = supplies
    info["battery_files"] = [s["file"] for s in supplies if s["dir"] == info["battery_dir"] and s["readable"]]
    # Why sysfs may be empty: SELinux often hides /sys/class/power_supply from the shell user.
    info["power_supply_diag"] = adb.sh("ls /sys/class/power_supply/ 2>&1 | head -5; "
                                       "cat /sys/class/power_supply/battery/current_now 2>&1; "
                                       "cat /sys/class/power_supply/battery/charge_counter 2>&1").strip()

    info["perfetto_version"] = adb.sh("perfetto --version 2>&1 | head -1").strip()
    info["has_timeout"] = bool(adb.sh("command -v timeout").strip())
    info["state"] = bench_common.device_state(adb)
    return info


def print_discovery(d: dict):
    print(f"\nDevice: {d['manufacturer']} {d['model']} ({d['soc']}), Android {d['android']} (SDK {d['sdk']}), "
          f"{d['ncpu']} CPUs, serial {d['serial']}")
    print(f"Root (su):             {'YES' if d['root']['rooted'] else 'no'}  [{d['root']['su_output'][:60]}]")
    pc = d["powercap"]
    if pc["zones"]:
        names = ", ".join(z["name"] or Path(z["path"]).name for z in pc["zones"])
        print(f"Powercap zones:        {len(pc['zones'])} ({names}); usable: {pc['usable']}"
              f"{' (via su)' if pc['usable'] and pc['needs_su'] else ''}")
    else:
        print("Powercap zones:        none (paper's PowerBench counters not available)")
    print(f"Power Stats HAL:       {d['powerstats']['service'] or 'not present'}")
    print(f"Battery sysfs:         {d['battery_dir']} -> {', '.join(d['battery_files']) or 'nothing readable'}")
    if not d["battery_files"]:
        diag = " | ".join(line.strip() for line in d.get("power_supply_diag", "").splitlines()[:4])
        print(f"   (shell sees: {diag[:200]}); the Perfetto battery methods still work without it")
    st = d["state"]
    print(f"Externally powered:    {st['externally_powered']} {st['power_sources']}  "
          f"battery {st['battery_level']}%  {st['battery_temp_c']} C  screen={st['screen']}")
    print("Perfetto power rails:  checked during collect (needs the trace)")


# ---------------------------------------------------------------------------
# Wireless adb
# ---------------------------------------------------------------------------

def cmd_wireless(args):
    adb = Adb(args.serial)
    if ":" in adb.serial:
        print(f"{adb.serial} is already a wireless connection.")
        return
    ip = None
    for _ in range(3):
        for iface in ("wlan0", "wlan1", "swlan0"):
            m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", adb.sh(f"ip -f inet addr show {iface}"))
            if m:
                ip = m.group(1)
                break
        if not ip:
            # any private IPv4 on a non-cellular interface
            for line in adb.sh("ip -o -f inet addr show").splitlines():
                m = re.search(r"^\d+:\s+(\S+)\s+inet (\d+\.\d+\.\d+\.\d+)", line)
                if m and not m.group(1).startswith(("lo", "rmnet", "ccmni", "dummy")) and \
                        m.group(2).startswith(("192.168.", "10.", "172.")):
                    ip = m.group(2)
                    break
        if ip:
            break
        adb.sh("input keyevent 224")  # wake: Wi-Fi can be dozing with the screen off
        time.sleep(3)
    if not ip:
        sys.exit("Could not find the phone's Wi-Fi IP. Connect the phone to the same Wi-Fi network as this computer.")
    print(f"Phone Wi-Fi IP: {ip}. Restarting adbd in TCP mode on port {args.port}...")
    adb.run(["tcpip", str(args.port)])
    time.sleep(3)
    target = f"{ip}:{args.port}"
    for _ in range(5):
        out = subprocess.run([adb.bin, "connect", target], capture_output=True, text=True).stdout
        if dict(list_devices(adb.bin)).get(target) == "device":
            break
        time.sleep(2)
    else:
        sys.exit(f"adb connect {target} failed: {out.strip()}\nThe laptop and phone must be able to reach each "
                 "other (corporate/guest Wi-Fi often blocks this; a phone hotspot or home network works).")
    print(f"\nConnected wirelessly as {target}.\nNow UNPLUG the USB cable, then run:\n"
          f"  python energy_probe.py collect --serial {target}")


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def sampler_script(columns: list[tuple[str, str]], interval: float) -> str:
    reads = "\n".join(f"  read x{i} < {path}" for i, (_, path) in enumerate(columns))
    fields = " ".join(f"$x{i}" for i in range(len(columns)))
    header = " ".join(label for label, _ in columns)
    return f"""#!/system/bin/sh
OUT={DEV_DIR}/samples.txt
rm -f {DEV_DIR}/stop
echo "# up {header}" > $OUT
while [ ! -e {DEV_DIR}/stop ]; do
  read up _ < /proc/uptime
{reads}
  echo "$up {fields}"
  sleep {interval}
done >> $OUT
"""


LOAD_SCRIPT = """#!/system/bin/sh
# usage: load.sh <busy threads> <seconds>; prints "<uptime start> <uptime end>"
N=$1; D=$2; i=0; pids=""
read t0 _ < /proc/uptime
while [ $i -lt $N ]; do
  timeout $D sh -c 'while :; do :; done' &
  pids="$pids $!"
  i=$((i+1))
done
sleep $D
kill $pids 2>/dev/null
read t1 _ < /proc/uptime
echo "$t0 $t1"
"""


def push_text(adb: Adb, text: str, remote: str):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, newline="\n") as f:
        f.write(text)
        local = f.name
    adb.run(["push", local, remote], timeout=60)
    Path(local).unlink(missing_ok=True)


def start_perfetto(adb: Adb) -> int | None:
    push_text(adb, PERFETTO_CONFIG, f"{DEV_DIR}/perfetto.pbtx")
    adb.sh(f"rm -f {TRACE_REMOTE}")
    out = adb.sh(f"cat {DEV_DIR}/perfetto.pbtx | perfetto --txt -c - -o {TRACE_REMOTE} --background-wait 2>&1")
    m = re.search(r"^\s*(\d+)\s*$", out, re.M)
    if not m:
        print(f"[WARN] perfetto did not start, Perfetto methods skipped: {out.strip()[:200]}")
        return None
    return int(m.group(1))


def stop_perfetto(adb: Adb, pid: int, run_dir: Path) -> str | None:
    adb.sh(f"kill -TERM {pid}")
    for _ in range(60):
        if "alive" not in adb.sh(f"kill -0 {pid} 2>/dev/null && echo alive"):
            break
        time.sleep(1)
    local = run_dir / "trace.pftrace"
    adb.run(["pull", TRACE_REMOTE, str(local)], timeout=300)
    adb.sh(f"rm -f {TRACE_REMOTE}")
    return local.name if local.exists() else None


def cmd_collect(args):
    adb = Adb(args.serial)
    info = discover(adb)
    print_discovery(info)

    if info["state"]["externally_powered"] and not info["powercap"]["usable"] and not args.allow_powered:
        sys.exit("\nThe phone is externally powered (USB/charger), so battery-based energy would measure the charger, "
                 "not the workload, and this phone has no rail counters to fall back on.\n"
                 "Run `python energy_probe.py wireless` while the cable is plugged in, unplug the cable, then run "
                 "collect again with the wireless serial. (--allow-powered records anyway, for testing.)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', info['model'])}_{stamp}"
    run_dir.mkdir(parents=True)
    print(f"\nResults -> {run_dir}")

    columns = []
    bdir = info["battery_dir"]
    for f in info["battery_files"]:
        columns.append((f"sysfs:{f}", f"{bdir}/{f}"))
    pc = info["powercap"]
    for z in pc["zones"]:
        if z["shell_readable"] or z.get("su_readable"):
            label = re.sub(r"\s+", "_", z["name"] or Path(z["path"]).name)
            columns.append((f"powercap:{label}", f"{z['path']}/energy_uj"))
    if not columns and args.no_perfetto:
        sys.exit("Nothing readable to sample (no battery sysfs files, no powercap) and --no-perfetto was given.")
    if not columns:
        print("No sysfs/powercap files readable: the phone-side sampler records timestamps only (for sleep "
              "detection) and energy comes from the Perfetto battery methods.")

    adb.sh(f"mkdir -p {DEV_DIR}; rm -f {DEV_DIR}/stop {DEV_DIR}/samples.txt")
    push_text(adb, sampler_script(columns, args.interval), f"{DEV_DIR}/sampler.sh")
    push_text(adb, LOAD_SCRIPT, f"{DEV_DIR}/load.sh")
    sampler_cmd = f"setsid sh {DEV_DIR}/sampler.sh"
    if pc["usable"] and pc["needs_su"]:
        sampler_cmd = f"su -c '{sampler_cmd}'"
    if not info["has_timeout"]:
        sys.exit("The phone's shell has no `timeout` command, which the load script needs.")

    workloads = {"idle": 0, "cpu1": 1, "cpu_all": info["ncpu"]}
    plan = [(w, r) for r in range(args.repeats) for w in workloads]
    total_min = len(plan) * (args.window + args.rest) / 60
    print(f"Plan: {len(plan)} windows of {args.window}s ({', '.join(workloads)} x {args.repeats}), "
          f"{args.rest}s rest each, about {total_min:.0f} min. Keep the phone still and unplugged.\n")

    meta = {"discovery": info, "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                          if k != "func"},
            "columns": [c[0] for c in columns], "start": datetime.now().isoformat()}
    windows, pid = [], None
    try:
        if args.screen == "off":
            adb.sh("input keyevent 223")
        adb.sh(f"{sampler_cmd} </dev/null >/dev/null 2>&1 &")
        if not args.no_perfetto:
            pid = start_perfetto(adb)
            if pid is None and not columns:
                raise SystemExit("Perfetto did not start and nothing else is readable; no energy method to test.")
        time.sleep(args.rest)
        for i, (w, rep) in enumerate(plan, 1):
            st = bench_common.device_state(adb)
            print(f"[{i}/{len(plan)}] {w} rep {rep + 1}: battery {st['battery_temp_c']} C, "
                  f"powered={st['externally_powered']}", flush=True)
            out = adb.sh(f"sh {DEV_DIR}/load.sh {workloads[w]} {args.window}", timeout=args.window + 120).split()
            if len(out) != 2:
                print(f"   !! load script returned {out!r}; window skipped")
                continue
            windows.append({"workload": w, "rep": rep, "t0": float(out[0]), "t1": float(out[1]),
                            "battery_temp_c": st["battery_temp_c"], "externally_powered": st["externally_powered"],
                            "battery_level": st["battery_level"]})
            (run_dir / "windows.json").write_text(json.dumps(windows, indent=1))
            time.sleep(args.rest)
    finally:
        adb.sh(f"touch {DEV_DIR}/stop")
        time.sleep(1)
        # Separate calls: pkill -f also matches the invoking shell's own command line. Busy loops left
        # behind die on their own via `timeout`.
        adb.sh(f"pkill -f {DEV_DIR}/load.sh")
        if pid is not None:
            meta["trace"] = stop_perfetto(adb, pid, run_dir)
        adb.run(["pull", f"{DEV_DIR}/samples.txt", str(run_dir / "samples.txt")], timeout=120)
        if args.screen == "off":
            adb.sh("input keyevent 224")
        meta["end"] = datetime.now().isoformat()
        meta["state_end"] = bench_common.device_state(adb)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        (run_dir / "windows.json").write_text(json.dumps(windows, indent=1))

    analyze(run_dir)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_samples(path: Path) -> tuple[list[str], list[list]]:
    lines = path.read_text(errors="replace").splitlines()
    header = lines[0].lstrip("#").split() if lines and lines[0].startswith("#") else []
    rows = []
    for line in lines[1:]:
        p = line.split()
        if len(p) != len(header):
            continue
        row = []
        for x in p:
            try:
                row.append(float(x))
            except ValueError:
                row.append(None)
        if row[0] is not None:
            rows.append(row)
    return header, rows


def current_scale(values: list[float]) -> float:
    """Raw battery current -> amps. Gauges report uA or mA in the same field."""
    med = statistics.median(abs(v) for v in values)
    return 1e-3 if med < bench_common.MA_UNIT_THRESHOLD else 1e-6


def voltage_scale(values: list[float]) -> float:
    med = statistics.median(values)
    return 1e-6 if med > 1e5 else 1e-3 if med > 100 else 1.0


def series_window(series: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    return [(t, v) for t, v in series if a <= t <= b]


def time_weighted_mean(series, a, b):
    """Zero-order hold mean of series over [a, b]; None if no samples in range."""
    prev = [(t, v) for t, v in series if t <= a]
    pts = ([(a, prev[-1][1])] if prev else []) + [(t, v) for t, v in series if a < t < b]
    if not pts:
        return None
    total = 0.0
    for (t, v), nxt in zip(pts, pts[1:] + [(b, None)]):
        total += v * (nxt[0] - t)
    return total / (b - pts[0][0]) if b > pts[0][0] else None


def counter_rate(series, a, b, wrap=None):
    """(last - first) / dt over samples inside [a, b] for a cumulative counter."""
    inside = series_window(series, a, b)
    if len(inside) < 2 or inside[-1][0] <= inside[0][0]:
        return None
    delta = inside[-1][1] - inside[0][1]
    if delta < 0 and wrap:
        delta += wrap
    return delta / (inside[-1][0] - inside[0][0])


def updates_per_s(series, a, b):
    vals = [v for _, v in series_window(series, a, b)]
    return sum(1 for x, y in zip(vals, vals[1:]) if x != y) / (b - a) if b > a else 0.0


def sign_from_load(rates: dict) -> float:
    """+1/-1 so that cpu_all draws more than idle (handles each gauge's charge/discharge convention)."""
    idle = [r for r in rates.get("idle", []) if r is not None]
    load = [r for r in rates.get("cpu_all", []) if r is not None]
    if not idle or not load:
        return 1.0
    return 1.0 if statistics.mean(load) >= statistics.mean(idle) else -1.0


def perfetto_series(trace: Path) -> dict:
    """{track name: [(boottime s, value)]} for battery counters and power rails; {} if unavailable."""
    try:
        from perfetto.trace_processor import TraceProcessor
    except ImportError:
        print("[INFO] Python package 'perfetto' not installed: Perfetto methods skipped "
              "(pip install perfetto, then rerun: python energy_probe.py analyze <run dir>)")
        return {}
    tp = TraceProcessor(trace=str(trace))
    try:
        rows = tp.query("select t.name as name, c.ts as ts, c.value as v from counter c "
                        "join counter_track t on c.track_id = t.id "
                        "where t.name like 'batt.%' or t.name like 'power.%' order by c.ts")
        out: dict = {}
        for r in rows:
            out.setdefault(r.name, []).append((r.ts / 1e9, r.v))
        return out
    finally:
        tp.close()


def method_rates(run_dir: Path, meta: dict, windows: list, settle: float) -> tuple[dict, dict]:
    """Returns ({method: {workload: [mean power mW per window]}}, {method: {"updates_hz": .., "notes": ..}})."""
    header, rows = load_samples(run_dir / "samples.txt")
    idx = {name: i for i, name in enumerate(header)}
    col = {name: [(r[0], r[i]) for r in rows if r[i] is not None] for i, name in enumerate(header) if i}

    def paired(a_name, b_name):
        """[(t, a, b)] from rows where both columns were read in the same sample."""
        ia, ib = idx[a_name], idx[b_name]
        return [(r[0], r[ia], r[ib]) for r in rows if r[ia] is not None and r[ib] is not None]
    spans = [(w, w["t0"] + settle, w["t1"] - 0.5) for w in windows]
    powers: dict = {}
    quality: dict = {}

    def add(method, fn, series_for_updates, note=""):
        raw = {}
        for w, a, b in spans:
            raw.setdefault(w["workload"], []).append(fn(a, b))
        sign = sign_from_load(raw)
        powers[method] = {k: [None if v is None else sign * v for v in vs] for k, vs in raw.items()}
        upd = [updates_per_s(series_for_updates, a, b) for _, a, b in spans] if series_for_updates else []
        quality[method] = {"updates_hz": round(statistics.median(upd), 3) if upd else None, "notes": note}

    vname = next((n for n in ("sysfs:voltage_now", "sysfs:voltage_avg") if col.get(n)), None)
    volt = col.get(vname) if vname else None
    vs = voltage_scale([v for _, v in volt]) if volt else None
    for f in SAMPLED_CURRENT:
        cur = col.get(f"sysfs:{f}")
        if cur and volt:
            cs = current_scale([v for _, v in cur])
            p = [(t, i * cs * v * vs * 1e3) for t, i, v in paired(f"sysfs:{f}", vname)]
            add(f"sysfs:{f}", lambda a, b, p=p: time_weighted_mean(p, a, b), cur,
                f"current unit {'mA' if cs == 1e-3 else 'uA'}")
    pw = col.get("sysfs:power_now")
    if pw:
        scale = 1e-3 if statistics.median(abs(v) for _, v in pw) > 1e4 else 1.0
        add("sysfs:power_now", lambda a, b: (lambda m: None if m is None else m * scale)(time_weighted_mean(pw, a, b)),
            pw, f"assumed {'uW' if scale == 1e-3 else 'mW'}")
    q = col.get("sysfs:charge_counter")
    if q and volt:
        def charge_power(a, b):
            r = counter_rate(q, a, b)
            v = time_weighted_mean(volt, a, b)
            return None if r is None or v is None else r * 3.6e-3 * v * vs * 1e3  # uAh/s -> C/s, x V -> W -> mW
        add("sysfs:charge_counter", charge_power, q, "coulomb counter (uAh)")
    zones = {z["name"]: z for z in meta["discovery"]["powercap"]["zones"]}
    for name, series in col.items():
        if name.startswith("powercap:"):
            wrap = (zones.get(name.split(":", 1)[1]) or {}).get("max_range_uj")
            add(name, lambda a, b, s=series, w=wrap: (lambda r: None if r is None else r / 1e3)(counter_rate(s, a, b, w)),
                series, "energy_uj counter (uJ)")

    if meta.get("trace") and (run_dir / meta["trace"]).exists():
        ps = perfetto_series(run_dir / meta["trace"])
        pc, pv, pq = ps.get("batt.current_ua"), ps.get("batt.voltage_uv"), ps.get("batt.charge_uah")
        if pc and pv:
            cs, pvs = current_scale([v for _, v in pc]), voltage_scale([v for _, v in pv])

            def perf_power(a, b):
                i, v = time_weighted_mean(pc, a, b), time_weighted_mean(pv, a, b)
                return None if i is None or v is None else i * cs * v * pvs * 1e3
            add("perfetto:current", perf_power, pc, f"battery HAL current, unit {'mA' if cs == 1e-3 else 'uA'}")
        if pq and pv:
            pvs = voltage_scale([v for _, v in pv])

            def perf_charge(a, b):
                r, v = counter_rate(pq, a, b), time_weighted_mean(pv, a, b)
                return None if r is None or v is None else r * 3.6e-3 * v * pvs * 1e3
            add("perfetto:charge", perf_charge, pq, "battery HAL charge counter")
        rails = {k: v for k, v in ps.items() if k.startswith("power.")}
        for name, series in rails.items():
            add(f"rail:{name[6:]}", lambda a, b, s=series: (lambda r: None if r is None else r / 1e3)(counter_rate(s, a, b)),
                series, "Power Stats rail energy (uWs)")
        if rails:
            def rails_total(a, b):
                rs = [counter_rate(s, a, b) for s in rails.values()]
                return None if any(r is None for r in rs) else sum(rs) / 1e3
            add("rail:TOTAL", rails_total, None, f"sum of {len(rails)} rails")
    return powers, quality


def stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, None
    m = statistics.mean(v)
    s = statistics.stdev(v) if len(v) > 1 else 0.0
    return m, s, (s / m if m else None)


def analyze(run_dir: Path):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    windows = json.loads((run_dir / "windows.json").read_text())
    settle = float(meta["params"].get("settle", 5))
    powers, quality = method_rates(run_dir, meta, windows, settle)

    rows = {}
    for m, by_w in powers.items():
        r = {"quality": quality[m]}
        for w in ("idle", "cpu1", "cpu_all"):
            r[w] = stats(by_w.get(w, []))
        idle = r["idle"][0]
        for w in ("cpu1", "cpu_all"):
            net = [x - idle for x in by_w.get(w, []) if x is not None] if idle is not None else []
            r[f"net_{w}"] = stats(net)
        cvs = [r[w][2] for w in ("cpu1", "cpu_all") if r[w][2] is not None]
        r["load_cv"] = max(cvs) if cvs else None
        rows[m] = r

    nets = [r["net_cpu_all"][0] for m, r in rows.items() if r["net_cpu_all"][0] and not m.startswith(("rail:", "powercap:"))]
    ref = statistics.median(nets) if nets else None
    for r in rows.values():
        n = r["net_cpu_all"][0]
        r["agreement"] = n / ref if (n is not None and ref) else None

    _, samples = load_samples(run_dir / "samples.txt")
    times = [r[0] for r in samples]
    gaps = {}
    for w in windows:
        ts = [t for t in times if w["t0"] <= t <= w["t1"]]
        g = max((b - a for a, b in zip(ts, ts[1:])), default=None)
        if g is not None:
            gaps[w["workload"]] = max(gaps.get(w["workload"], 0), g)

    rec, reasons = recommend(rows)
    if gaps.get("idle", 0) > 2.0:
        reasons.append(f"the phone slept during idle windows (largest sample gap {gaps['idle']:.1f} s), so the idle "
                       "baseline is a suspended-phone baseline, lower than the awake idle an inference run sees; "
                       "rerun with --screen on to compare")
    powered = any(w.get("externally_powered") for w in windows) or meta.get("state_end", {}).get("externally_powered")
    meta["max_sample_gap_s"] = {k: round(v, 2) for k, v in gaps.items()}
    report = render(meta, windows, rows, rec, reasons, powered)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps({"recommended": rec, "reasons": reasons, "methods": rows,
                                                     "externally_powered": powered}, indent=1, default=str))
    print("\n" + report)
    print(f"Report written to {run_dir / 'report.md'}")


def recommend(rows: dict) -> tuple[str | None, list[str]]:
    """Direct counters (powercap, rails) beat battery methods if they are repeatable; among battery methods
    pick the most repeatable one that updates often enough, preferring agreement with the others."""
    def usable(r, min_hz):
        return (r["load_cv"] is not None and r["net_cpu_all"][0] and r["net_cpu_all"][0] > 0
                and (r["quality"]["updates_hz"] or 0) >= min_hz)

    for prefix, label in (("powercap:", "paper-equivalent powercap counters"), ("rail:", "Power Stats rails")):
        cands = {m: r for m, r in rows.items() if m.startswith(prefix) and usable(r, 0.5) and r["load_cv"] < 0.15}
        if cands:
            total = next((m for m in cands if m.endswith("TOTAL")), None)
            best = total or max(cands, key=lambda m: cands[m]["net_cpu_all"][0])
            return best, [f"{label} are available and repeatable (load CV {cands[best]['load_cv']:.1%}); "
                          "they measure chip rails directly, are unaffected by charging, and match the paper's method"]
    batt = {m: r for m, r in rows.items() if not m.startswith(("powercap:", "rail:")) and usable(r, 0.05)}
    if not batt:
        return None, ["no method produced a positive, repeatable load-minus-idle power; see the table"]

    def score(m):
        r = batt[m]
        disagreement = abs((r["agreement"] or 1) - 1)
        return r["load_cv"] + 0.5 * disagreement + (0.05 if (r["quality"]["updates_hz"] or 0) < 0.5 else 0)
    ranked = sorted(batt, key=score)
    best = ranked[0]
    r = batt[best]
    reasons = [f"most repeatable battery method: load CV {r['load_cv']:.1%}, updates {r['quality']['updates_hz']} /s, "
               f"net all-core power {r['net_cpu_all'][0]:.0f} mW ({(r['agreement'] or 0):.2f}x the median of battery methods)"]
    if len(ranked) > 1:
        r2 = batt[ranked[1]]
        reasons.append(f"runner-up {ranked[1]}: load CV {r2['load_cv']:.1%}, agreement {(r2['agreement'] or 0):.2f}x")
    reasons.append("battery methods measure the whole phone (not SoC-only like the paper): use idle-subtracted "
                   "values and expect them to read higher than the paper's uJ/token")
    return best, reasons


def render(meta, windows, rows, rec, reasons, powered) -> str:
    d = meta["discovery"]

    def f(t, nd=0):
        m, s, _ = t
        return "n/a" if m is None else f"{m:.{nd}f} +/- {s:.{nd}f}"

    lines = [
        f"# Energy-method probe: {d['manufacturer']} {d['model']} ({d['soc']})",
        "",
        f"- Run: {meta['start']} -> {meta.get('end')}; {len(windows)} windows of {meta['params']['window']} s "
        f"(first {meta['params'].get('settle', 5)} s of each dropped); screen {meta['params']['screen']}",
        f"- Root: {d['root']['rooted']}; powercap usable: {d['powercap']['usable']}; "
        f"Power Stats HAL: {d['powerstats']['service'] or 'none'}",
        f"- Battery temperature across windows: "
        f"{min(w['battery_temp_c'] for w in windows) if windows else '?'} - "
        f"{max(w['battery_temp_c'] for w in windows) if windows else '?'} C",
        f"- Largest gap between phone-side samples (s), per workload: {meta.get('max_sample_gap_s')}",
    ]
    if powered:
        lines.append("- **WARNING: the phone was externally powered during the run; battery-based methods are invalid.**")
    lines += [
        "",
        "Power in mW, mean +/- std over repeats. Net = workload minus idle. Load CV = worst coefficient of "
        "variation of the two load workloads (lower = more repeatable). Agreement = net all-core power / median "
        "of the battery methods.",
        "",
        "| Method | Idle | 1 core | All cores | Net 1 core | Net all cores | Load CV | Updates/s | Agreement | Notes |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for m, r in sorted(rows.items()):
        cv = "n/a" if r["load_cv"] is None else f"{r['load_cv']:.1%}"
        ag = "n/a" if r["agreement"] is None else f"{r['agreement']:.2f}x"
        lines.append(f"| {m}{' **(recommended)**' if m == rec else ''} | {f(r['idle'])} | {f(r['cpu1'])} | "
                     f"{f(r['cpu_all'])} | {f(r['net_cpu1'])} | {f(r['net_cpu_all'])} | {cv} | "
                     f"{r['quality']['updates_hz']} | {ag} | {r['quality']['notes']} |")
    lines += ["", f"**Recommended: {rec or 'none'}**", ""] + [f"- {x}" for x in reasons] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("discover", help="list the phone's energy sources (USB is fine)")
    p.add_argument("--serial")
    p.add_argument("--json", type=Path, help="also write the discovery result here")

    p = sub.add_parser("wireless", help="switch a USB-connected phone to adb over Wi-Fi")
    p.add_argument("--serial")
    p.add_argument("--port", type=int, default=5555)

    p = sub.add_parser("collect", help="measure idle / 1-core / all-core loads with every method")
    p.add_argument("--serial")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--window", type=int, default=45, help="seconds per window (fuel gauges update every few s)")
    p.add_argument("--rest", type=int, default=20, help="seconds between windows")
    p.add_argument("--settle", type=float, default=5.0, help="seconds dropped from the start of each window")
    p.add_argument("--interval", type=float, default=0.1, help="sysfs sampling interval on the phone (s)")
    p.add_argument("--screen", choices=("off", "on"), default="off")
    p.add_argument("--no-perfetto", action="store_true")
    p.add_argument("--allow-powered", action="store_true", help="run even while plugged in (results flagged)")
    p.add_argument("--out-dir", type=Path, default=HERE / "energy_probe_results")

    p = sub.add_parser("analyze", help="re-analyze a collect run directory")
    p.add_argument("run_dir", type=Path)

    args = ap.parse_args()
    if args.cmd == "discover":
        info = discover(Adb(args.serial))
        print_discovery(info)
        if args.json:
            args.json.write_text(json.dumps(info, indent=1))
    elif args.cmd == "wireless":
        cmd_wireless(args)
    elif args.cmd == "collect":
        cmd_collect(args)
    else:
        analyze(args.run_dir)


if __name__ == "__main__":
    main()
