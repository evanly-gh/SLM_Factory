#!/usr/bin/env python3
"""
End-to-end test of run_paper_table5.py + report.py against a simulated phone (no adb, no device).

The fake phone answers the harness's adb commands: getprop / dumpsys / sysfs probes, detached benchmark
runs that emit PB_MARK lines at known speeds, and a Perfetto trace whose battery draw is 700 mW idle and
5000 mW while a benchmark window is active. Checks: every configuration runs, throughput matches the
simulated speeds, battery energy is extended to >= --min-energy-seconds, net energy per token matches
(5000 - 700) mW / speed, failures are recorded, and the report renders.

  python tests/fake_phone_test.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_paper_table5 as rpt  # noqa: E402
import report  # noqa: E402

SPEEDS = {  # (framework, backend): (prefill t/s, decode t/s)
    ("llama", "cpu"): (300.0, 55.0), ("llama", "gpu"): (750.0, 50.0),
    ("mnn", "cpu"): (228.0, 47.0), ("mnn", "gpu"): (434.0, 12.0),
}
IDLE_MW, LOAD_MW, V = 700.0, 5000.0, 4.0
REP_GAP_S = 0.002  # idle time between timed repetitions (llama-bench: a KV-cache clear, ~ms)
TIME_SCALE = 0.01  # real seconds per simulated benchmark second (the virtual clock jumps ahead)


# Perfetto trace protobuf written by hand (Trace.packet=1; TracePacket timestamp=8, trusted_packet_sequence_id=10,
# battery=38; BatteryCounters charge_counter_uah=1, current_ua=3, voltage_uv=7): the generated perfetto_trace_pb2
# needs protobuf >= 6, while requirements.txt pins protobuf < 5 for the llama.cpp converter.
def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1  # int64 two's complement
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _pb_varint(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _pb_bytes(field: int, payload: bytes) -> bytes:
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


class FakePhone:
    def __init__(self, sizes: dict, fail_model: str | None = None, swap_model: str | None = None):
        self.t0 = time.time()
        self.extra = 0.0
        self.files: dict = {}
        self.jobs: list = []  # (start, end) virtual windows of benchmark activity
        self.sizes = sizes
        self.fail_model = fail_model
        self.swap_model = swap_model  # its benchmark never finishes and has 900 MB swapped out until killed
        self.swapping = False
        self.trace_start = None

    def now(self) -> float:
        return 1000.0 + (time.time() - self.t0) + self.extra

    def run_job(self, script: str):
        work = re.search(r"> (\S+)/stdout\.txt", script).group(1)
        cmd = script.split("&&", 1)[1]
        out, err, code = "", [], 0
        start = self.now()
        t = start + 0.5  # model load
        if self.swap_model and self.swap_model in cmd:
            self.swapping = True
            return
        if self.fail_model and self.fail_model in cmd:
            code, err = 137, ["Killed"]
        elif "llama-bench" in cmd:
            p = int(re.search(r"-p (\d+)", cmd).group(1))
            n = int(re.search(r"-n (\d+)", cmd).group(1))
            r = int(re.search(r"-r (\d+)", cmd).group(1))
            d = int((re.search(r"-d (\d+)", cmd) or [0, 0])[1])
            backend = "gpu" if "-ngl 99" in cmd else "cpu"
            pp, tg = SPEEDS[("llama", backend)]
            t += 0.3  # llama-bench's own warm-up
            for i in range(r):
                dur = p / pp if p else n / tg
                self.jobs.append((t, t + dur))
                err.append(f"PB_MARK llama rep={i} begin={int(t * 1e9)} end={int((t + dur) * 1e9)} "
                           f"n_prompt={p} n_gen={n} n_depth={d}")
                t += dur + REP_GAP_S
            ts = (p or n) / (p / pp if p else n / tg)
            out = json.dumps([{"build_commit": "eadc418", "backends": "OpenCL" if backend == "gpu" else "CPU",
                               "n_threads": 8, "avg_ts": ts, "stddev_ts": 0.1, "samples_ts": [ts] * r,
                               "samples_ns": [int((p / pp if p else n / tg) * 1e9)] * r}])
        elif "llm_bench" in cmd:
            p = int(re.search(r"-p (\d+)", cmd).group(1))
            n = int(re.search(r"-n (\d+)", cmd).group(1))
            rep = int(re.search(r"-rep (\d+)", cmd).group(1))
            kv = "-kv true" in cmd
            backend = "gpu" if "-a opencl" in cmd else "cpu"
            pp, tg = SPEEDS[("mnn", backend)]
            for i in range(rep + 1):
                pre = p / pp
                dec = n / tg if kv else 1 / tg
                self.jobs.append((t, t + pre + dec))
                err.append(f"PB_MARK mnn mode={'kv' if kv else 'pp'} rep={i} begin={int(t * 1e9)} "
                           f"end={int((t + pre + dec) * 1e9)} prefill_us={int(pre * 1e6)} decode_us={int(dec * 1e6)} "
                           f"prompt={p} gen={n if kv else 1}")
                t += pre + dec + REP_GAP_S
        self.extra += t - start
        time.sleep((t - start) * TIME_SCALE)
        self.files[f"{work}/stdout.txt"] = out
        self.files[f"{work}/stderr.txt"] = "\n".join(err) + "\n"
        self.files[f"{work}/exit"] = str(code)

    def write_trace(self, path: Path):
        trace = bytearray()
        t, end = self.trace_start, self.now()
        q = 4_000_000.0
        while t < end:
            mw = LOAD_MW if any(a <= t < b for a, b in self.jobs) else IDLE_MW
            ua = mw / V * 1000
            q -= ua * 0.1 / 3600
            battery = _pb_varint(3, int(-ua)) + _pb_varint(7, int(V * 1e6)) + _pb_varint(1, int(q))
            pkt = _pb_varint(8, int(t * 1e9)) + _pb_varint(10, 1) + _pb_bytes(38, battery)
            trace += _pb_bytes(1, pkt)
            t += 0.1
        path.write_bytes(bytes(trace))


class FakeAdb:
    phone: FakePhone = None

    def __init__(self, serial=None, adb_bin=None):
        self.serial = serial or "FAKE123"
        self.bin = "adb"
        self.p = FakeAdb.phone

    def run(self, args, timeout=60):
        import subprocess
        if args[0] == "shell":
            return subprocess.CompletedProcess(args, 0, stdout=self.sh(args[1]), stderr="")
        if args[0] == "push":
            local, remote = args[1], args[2]
            if Path(local).is_file() and Path(local).stat().st_size < 100_000:
                self.p.files[remote] = Path(local).read_text(errors="replace")
        elif args[0] == "pull":
            remote, local = args[1], Path(args[2])
            if remote.endswith(".pftrace"):
                self.p.write_trace(local)
            elif remote.endswith("samples.txt"):
                t, end, lines = self.p.trace_start, self.p.now(), ["# up"]
                while t < end:
                    lines.append(f"{t:.2f}")
                    t += 0.5
                local.write_text("\n".join(lines) + "\n")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def sh(self, cmd: str, timeout=60) -> str:
        p = self.p
        props = {"ro.product.manufacturer": "samsung", "ro.product.model": "SM-S911U", "ro.product.device": "dm1q",
                 "ro.soc.model": "SM8550", "ro.board.platform": "kalama", "ro.build.version.release": "16",
                 "ro.build.version.sdk": "36", "ro.build.fingerprint": "fake"}
        m = re.fullmatch(r"getprop (\S+)", cmd)
        if m:
            return props.get(m.group(1), "") + "\n"
        if cmd == "echo ok":
            return "ok\n"
        if cmd == "cat /proc/uptime":
            return f"{p.now():.2f} 0.00\n"
        if cmd == "head -1 /proc/stat":
            busy = int(p.extra * 100 * 8)  # busy jiffies grow while benchmarks run
            return f"cpu  {busy} 0 0 {int(p.now() * 100 * 8) - busy} 0 0 0 0 0 0\n"
        if cmd == "cat /proc/meminfo":
            return "MemTotal:        7654321 kB\n"
        if "cpufreq/policy*; do echo $(basename $p) $(cat $p/cpuinfo_max_freq) $(cat $p/related_cpus)" in cmd:
            return "policy0 2016000 0 1 2\npolicy3 2803200 3 4 5 6\npolicy7 3360000 7\n"
        if cmd.startswith("dumpsys battery;"):
            return ("Current Battery Service state:\n  AC powered: false\n  USB powered: false\n  Wireless powered: false\n"
                    "  status: 3\n  level: 80\n  temperature: 265\n@@CAPS\npolicy0 2016000 2016000\npolicy3 2803200 2803200\n"
                    "policy7 3360000 3360000\n@@POWER\n  mWakefulness=Asleep\n@@THERMAL\nThermal Status: 0\n")
        if cmd.startswith("su -c id"):
            return "/system/bin/sh: su: inaccessible or not found\n"
        if cmd == "dumpsys -l":
            return "Currently running services:\n  power\n"
        if "perfetto --version" in cmd:
            return "Perfetto v50\n"
        if cmd == "command -v timeout":
            return "/system/bin/timeout\n"
        if cmd.startswith("test -x"):
            return "ok\n"
        if "--list-devices" in cmd:
            return "Available devices:\n  GPUOpenCL: QUALCOMM Adreno(TM) 740\n"
        if "find . -type f -exec stat" in cmd or cmd.startswith("stat -c '%n %s'"):
            return "".join(f"./{n} {s}\n" if "find" in cmd else f"/data/x/{n} {s}\n" for n, s in p.sizes.items())
        m = re.match(r"setsid sh (\S+/run\.sh)", cmd)
        if m:
            p.run_job(p.files[m.group(1)])
            return ""
        if "grep VmSwap" in cmd:
            return "VmSwap:\t  921600 kB\n" if p.swapping else ""
        if cmd.startswith("pkill") or "; pkill " in cmd:
            p.swapping = False
        if "perfetto --txt" in cmd:
            p.trace_start = p.now()
            return "4242\n"
        m = re.match(r"cat (\S+)( 2>/dev/null)?$", cmd)
        if m:
            return p.files.get(m.group(1), "")
        if "samples.txt" in cmd and "sampler.sh" not in cmd:
            return ""
        if "sampler.sh" in cmd and cmd.startswith("setsid"):
            p.trace_start = p.trace_start or p.now()
            return ""
        return ""


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pb_fake_"))
    models = tmp / "models"
    (models / "gguf").mkdir(parents=True)
    (models / "mnn" / "m1-mnn-q4").mkdir(parents=True)
    manifest = {"models": [
        {"name": "llama3.2-1b", "framework": "llama.cpp", "quant": "Q4_0", "bits": 4,
         "path": "gguf/llama3.2-1b-q4_0.gguf", "files": {"llama3.2-1b-q4_0.gguf": 123}},
        {"name": "llama3.2-1b", "framework": "llama.cpp", "quant": "Q4_K_M", "bits": 4,
         "path": "gguf/llama3.2-1b-q4_k_m.gguf", "files": {"llama3.2-1b-q4_k_m.gguf": 123}},
        {"name": "llama3.2-1b", "framework": "mnn", "quant": "Q4", "bits": 4,
         "path": "mnn/m1-mnn-q4", "files": {"config.json": 123}},
        {"name": "qwen2.5-7b", "framework": "llama.cpp", "quant": "Q4_0", "bits": 4,
         "path": "gguf/qwen2.5-7b-q4_0.gguf", "files": {"qwen2.5-7b-q4_0.gguf": 123}},
        {"name": "llama3.2-3b", "framework": "llama.cpp", "quant": "Q4_0", "bits": 4,
         "path": "gguf/llama3.2-3b-q4_0.gguf", "files": {"llama3.2-3b-q4_0.gguf": 123}},
    ]}
    (models / "manifest.json").write_text(json.dumps(manifest))
    FakeAdb.phone = FakePhone({"llama3.2-1b-q4_0.gguf": 123, "llama3.2-1b-q4_k_m.gguf": 123, "config.json": 123,
                               "qwen2.5-7b-q4_0.gguf": 123, "llama3.2-3b-q4_0.gguf": 123}, fail_model="qwen2.5-7b",
                              swap_model="llama3.2-3b")
    rpt.Adb = FakeAdb
    rpt.MODELS_DIR = models
    rpt.POLL_S = 0.05
    rpt.bench_common.ReadinessGate.wait.__defaults__  # noqa: B018 - gate polls a fake state that is always ready
    args = Namespace(serial="FAKE123", ref="pinned", models=None, quants=None, frameworks=["llama.cpp", "mnn"],
                     backends=["cpu", "gpu"], trials=3, n_prompt=256, n_gen=256, max_temp=None, gate_timeout=60, gate_plateau=None, min_battery=0, smallest_first=True,
                     rest=0, idle_seconds=1, pre_idle_seconds=0.5, min_energy_seconds=30.0, prime_seconds=0, timeout=600,
                     llama_threads=None, mnn_threads=None, no_controls=False, no_perfetto=False, screen="auto",
                     results_dir=str(tmp / "results"), force=False, plan=False)
    rpt.Session(args).run()

    out = tmp / "results"
    results = {f.stem: json.loads(f.read_text()) for f in (out / "configs").glob("*.json")}
    failures = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    check(len(results) == 10, f"10 configurations recorded (got {len(results)})")
    check(results["llama3.2-1b__Q4_K_M__llama.cpp__gpu"].get("gpu_partial_offload") is True,
          "Q4_K_M on the pinned OpenCL backend flagged as partly on the CPU")
    check(not results["llama3.2-1b__Q4_0__llama.cpp__gpu"].get("gpu_partial_offload"),
          "Q4_0 on the pinned OpenCL backend not flagged")
    for cid, r in sorted(results.items()):
        c = r["config"]
        if c["model"] == "qwen2.5-7b":
            check(r["status"] == "failed" and "oom" in r["error"], f"{cid}: failure recorded ({r.get('error')})")
            continue
        if c["model"] == "llama3.2-3b":
            check(r["status"] == "failed" and "does not fit in RAM" in r["error"],
                  f"{cid}: swapping benchmark stopped and recorded ({r.get('error')})")
            continue
        fw = "llama" if c["framework"] == "llama.cpp" else "mnn"
        pp, tg = SPEEDS[(fw, c["backend"])]
        check(r["status"] == "ok", f"{cid}: status ok")
        check(not r.get("suspended"), f"{cid}: no suspend detected (ratios "
              f"{[inv.get('suspend_ratio') for inv in r['invocations'].values()]})")
        check(abs(r["prefill"]["tps"] - pp) / pp < 0.01, f"{cid}: prefill {r['prefill']['tps']} ~ {pp}")
        check(abs(r["decode"]["tps"] - tg) / tg < 0.01, f"{cid}: decode {r['decode']['tps']} ~ {tg}")
        check(len(r["prefill"]["reps"]) == 3, f"{cid}: 3 recorded prefill trials (warm-up dropped)")
        e = r["energy"][r["energy_headline_method"]]
        check(e["prefill"]["seconds"] >= 29, f"{cid}: prefill energy window extended to {e['prefill']['seconds']} s")
        exp_pre = (LOAD_MW - IDLE_MW) / pp * 1e3
        exp_dec = (LOAD_MW - IDLE_MW) / tg * 1e3
        check(abs(r["prefill"]["uj_per_token"] - exp_pre) / exp_pre < 0.05,
              f"{cid}: prefill net {r['prefill']['uj_per_token']} uJ/tok ~ {exp_pre:.0f}")
        check(abs(r["decode"]["uj_per_token"] - exp_dec) / exp_dec < 0.05,
              f"{cid}: decode net {r['decode']['uj_per_token']} uJ/tok ~ {exp_dec:.0f}")
        check(r["prefill"]["energy_valid"] and r["decode"]["energy_valid"], f"{cid}: energy valid")
        check(r.get("energy_idle") == "per invocation", f"{cid}: idle baseline measured before each invocation")

    sys.argv = ["report.py", str(out)]
    report.main()
    check((out / "TABLE5.html").exists() and (out / "TABLE5.md").exists(), "report written")
    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'} (artifacts in {tmp})")
    if not failures:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
