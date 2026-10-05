#!/usr/bin/env python3
"""
Table 5 replication harness (arXiv 2607.05475): 256-token prefill and 256-token decode throughput and
energy (uJ/token) per model x quantization x framework (llama.cpp, MNN) x backend (CPU, GPU/OpenCL), using
the frameworks' native benchmark tools built by build_binaries.py and models from prepare_models.py.

  python run_paper_table5.py --serial S --models llama3.2-1b --quants Q4_0 Q4
  python run_paper_table5.py --serial S                    # every model/quant in models/manifest.json
  python run_paper_table5.py --serial S --plan             # list the configurations and exit
  python run_paper_table5.py --serial S --results-dir results/<dir>   # resume a session

Protocol (paper Section 3.4, plus what its measurement definitions imply):
  * phone in airplane mode (Wi-Fi kept on only for wireless adb), screen off, Do Not Disturb, background
    apps killed; only the benchmark process runs.
  * before every benchmark invocation: wait until battery temperature <= the profile's limit (paper: 28 C)
    and CPU frequency caps are back at the session baseline (bench_common.ReadinessGate).
  * 1 warm-up repetition + --trials (>= 3) recorded repetitions; the mean is reported.
  * llama.cpp: llama-bench pp256 (prefill) and tg256 at depth 256 (decode after a 256-token context,
    as in the paper's single prompt+generate run), separate invocations.
  * MNN: llm_bench -kv true -p 256 -n 256 (prompt then generation in one response, its "llm_demo test
    standard"), EOS ignored (PB_IGNORE_EOS) so decode always produces 256 tokens.
  * prefill t/s = prompt tokens / prefill time; decode t/s = generated tokens / decode time.
  * energy: see energy.py; windows are the benchmark's own timed regions (PB_MARK lines, CLOCK_BOOTTIME).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import bench_common  # noqa: E402
import energy_probe  # noqa: E402
from common import (DEV_MODELS, DEV_ROOT, MODELS_DIR, RESULTS_DIR, DeviceControls, boottime_s,  # noqa: E402
                    dev_bin_dir, device_info, match_profile)
from devenv import Adb  # noqa: E402
from energy import EnergyRecorder, is_chip_method, probe_recommendation  # noqa: E402

BACKENDS = ("cpu", "gpu")
FRAMEWORKS = ("llama.cpp", "mnn")
MARK_RE = re.compile(r"^PB_MARK (\w+) (.*)$", re.M)
DEV_RUN = f"{DEV_ROOT}/run"
POLL_S = 10  # seconds between completion checks of a detached benchmark


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_marks(stderr: str) -> list[dict]:
    marks = []
    for m in MARK_RE.finditer(stderr):
        d = {"tool": m.group(1)}
        for kv in m.group(2).split():
            k, _, v = kv.partition("=")
            d[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
        marks.append(d)
    return marks


def mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    path = MODELS_DIR / "manifest.json"
    if not path.exists():
        sys.exit(f"{path} not found: run prepare_models.py first")
    return json.loads(path.read_text())["models"]


def ensure_model_on_device(adb: Adb, entry: dict) -> str:
    """Push the model if the phone's copy is missing or differs in size; returns the device path."""
    local = MODELS_DIR / entry["path"]
    remote = f"{DEV_MODELS}/{local.name}"
    expected = entry["files"]  # {relative name: bytes}
    listing = adb.sh(f"cd {remote} 2>/dev/null && find . -type f -exec stat -c '%n %s' {{}} + 2>/dev/null"
                     if local.is_dir() else f"stat -c '%n %s' {remote} 2>/dev/null")
    have = {}
    for line in listing.splitlines():
        name, _, size = line.rpartition(" ")
        if size.isdigit():
            have[name.removeprefix("./") if local.is_dir() else Path(name).name] = int(size)
    if all(have.get(n) == s for n, s in expected.items()):
        return remote
    if not local.exists():
        sys.exit(f"{entry['name']} {entry['quant']}: not on the phone and not on this computer ({local})")
    log(f"pushing {local.name} ({sum(expected.values()) / 2**30:.2f} GiB) to the phone...")
    adb.sh(f"mkdir -p {DEV_MODELS}; rm -rf {remote}")
    r = adb.run(["push", str(local), remote if local.is_file() else f"{DEV_MODELS}/"], timeout=7200)
    if r.returncode != 0:
        sys.exit(f"adb push failed: {r.stderr.strip()[-500:]}")
    return remote


# ---------------------------------------------------------------------------
# Running one benchmark invocation on the phone
# ---------------------------------------------------------------------------

def run_on_device(adb: Adb, controls: DeviceControls, bin_dir: str, cmd: str, env: str, tag: str,
                  raw_dir: Path, timeout_s: int) -> dict:
    """Run detached (survives an adb disconnect), poll for completion, fetch stdout/stderr."""
    work = f"{DEV_RUN}/{tag}"
    adb.sh(f"rm -rf {work}; mkdir -p {work}")
    script = (f"cd {bin_dir} && {env} {cmd} > {work}/stdout.txt 2> {work}/stderr.txt; "
              f"echo $? > {work}/exit")
    energy_probe.push_text(adb, script + "\n", f"{work}/run.sh")
    adb.sh(f"setsid sh {work}/run.sh </dev/null >/dev/null 2>&1 &")
    t0 = time.time()
    exit_code = None
    while time.time() - t0 < timeout_s:
        time.sleep(POLL_S)
        out = adb.sh(f"cat {work}/exit 2>/dev/null", timeout=30).strip()
        if out == "" and controls.wireless and adb.sh("echo ok", timeout=15).strip() != "ok":
            controls.reconnect()
            continue
        if out.lstrip("-").isdigit():
            exit_code = int(out)
            break
    else:
        adb.sh(f"pkill -f {work}/run.sh; pkill -f {cmd.split()[0]}")
    stdout = adb.sh(f"cat {work}/stdout.txt", timeout=120)
    stderr = adb.sh(f"cat {work}/stderr.txt", timeout=120)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{tag}.stdout.txt").write_text(stdout, encoding="utf-8")
    (raw_dir / f"{tag}.stderr.txt").write_text(stderr, encoding="utf-8")
    (raw_dir / f"{tag}.cmd.txt").write_text(f"{env} {cmd}\n", encoding="utf-8")
    return {"exit": exit_code, "timed_out": exit_code is None, "stdout": stdout, "stderr": stderr,
            "seconds": round(time.time() - t0, 1), "marks": parse_marks(stderr)}


def failure_reason(res: dict) -> str | None:
    if res["timed_out"]:
        return "timeout"
    if res["exit"] != 0:
        tail = " | ".join(res["stderr"].strip().splitlines()[-3:])
        kind = "oom/killed" if res["exit"] in (137, -9) or "Killed" in res["stderr"] else "crash/error"
        return f"{kind} (exit {res['exit']}): {tail[-300:]}"
    if not res["marks"]:
        return "no PB_MARK output (binary not built by build_binaries.py?)"
    return None


# ---------------------------------------------------------------------------
# Framework adapters: commands, and PB_MARK -> throughput + energy windows
# ---------------------------------------------------------------------------

def llama_cmd(model_path: str, backend: str, n_p: int, n_g: int, depth: int, reps: int, threads: int | None) -> str:
    cmd = f"./llama-bench -m {model_path} -p {n_p} -n {n_g} -r {reps} -ngl {99 if backend == 'gpu' else 0} -o json"
    if depth:
        cmd += f" -d {depth}"
    if threads:
        cmd += f" -t {threads}"
    return cmd


def llama_windows(marks: list[dict], phase: str, source: str) -> list[dict]:
    out = []
    for m in marks:
        if m.get("tool") != "llama" or m["rep"] == 0:  # rep 0 is the warm-up
            continue
        tokens = m["n_prompt"] if phase == "prefill" else m["n_gen"]
        out.append({"phase": phase, "source": source, "rep": m["rep"], "a": m["begin"] / 1e9,
                    "b": m["end"] / 1e9, "tokens": tokens})
    return out


def mnn_cmd(model_dir: str, backend: str, n_p: int, n_g: int, reps: int, kv: bool, threads: int | None) -> str:
    cmd = (f"./llm_bench -m {model_dir}/config.json -a {'opencl' if backend == 'gpu' else 'cpu'} "
           f"-p {n_p} -n {n_g} -rep {reps} -kv {'true' if kv else 'false'}")
    if threads:
        cmd += f" -t {threads}"
    return cmd


def mnn_windows(marks: list[dict], source: str) -> list[dict]:
    """kv: prefill = [begin, begin + prefill_us], decode = [end - decode_us, end]; pp: prefill only."""
    out = []
    for m in marks:
        if m.get("tool") != "mnn" or m["rep"] == 0:  # llm_bench discards its first response too
            continue
        a, b = m["begin"] / 1e9, m["end"] / 1e9
        if m["mode"] in ("kv", "pp"):
            out.append({"phase": "prefill", "source": source, "rep": m["rep"], "a": a,
                        "b": a + m["prefill_us"] / 1e6, "tokens": m["prompt"]})
        if m["mode"] == "kv":
            out.append({"phase": "decode", "source": source, "rep": m["rep"], "a": b - m["decode_us"] / 1e6,
                        "b": b, "tokens": m["gen"]})
    return out


def suspend_ratio(marks: list[dict], llama_json: dict | None = None) -> float | None:
    """Largest ratio of a repetition's CLOCK_BOOTTIME duration (PB_MARK, counts suspend) to the tool's own
    monotonic timing (does not count suspend). Above ~1.05 the phone slept during the timed work."""
    ratios = []
    for i, m in enumerate(marks):
        wall = (m["end"] - m["begin"]) / 1e9
        if m.get("tool") == "llama" and llama_json and i < len(llama_json.get("samples_ns") or []):
            mono = llama_json["samples_ns"][i] / 1e9
        elif m.get("tool") == "mnn":
            mono = (m["prefill_us"] + m["decode_us"]) / 1e6
        else:
            continue
        if mono > 0.05:
            ratios.append(wall / mono)
    return round(max(ratios), 3) if ratios else None


def throughput(windows: list[dict], phase: str) -> dict:
    tps = [w["tokens"] / (w["b"] - w["a"]) for w in windows
           if w["phase"] == phase and w["source"] == "main" and w["b"] > w["a"]]
    m, s = mean_std(tps)
    return {"tps": round(m, 2) if m is not None else None, "tps_std": round(s, 2) if s is not None else None,
            "reps": [round(x, 2) for x in tps],
            "tokens": sorted({w["tokens"] for w in windows if w["phase"] == phase and w["source"] == "main"})}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, args):
        self.args = args
        self.adb = Adb(args.serial)
        self.info = device_info(self.adb)
        self.profile = match_profile(self.info)
        self.out = Path(args.results_dir) if args.results_dir else (
            RESULTS_DIR / f"{self.profile['id']}_{args.ref}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        (self.out / "configs").mkdir(parents=True, exist_ok=True)
        self.raw = self.out / "raw"
        self.controls = DeviceControls(self.adb, log, screen=args.screen)
        self.gate = None
        self.recorder = None
        self.idle_window = None
        self.max_temp = args.max_temp if args.max_temp is not None else self.profile.get("max_temp_c", 28.0)

    # -- configurations ----------------------------------------------------
    def configs(self) -> list[dict]:
        manifest = load_manifest()
        out = []
        for e in manifest:
            if self.args.models and e["name"] not in self.args.models:
                continue
            if self.args.quants and e["quant"] not in self.args.quants:
                continue
            if e["framework"] not in self.args.frameworks:
                continue
            if self.args.ref == "pinned" and e.get("ref", "pinned") != "pinned":
                continue  # converted for the newer runtime; run it in a --ref head session
            for backend in self.args.backends:
                cid = f"{e['name']}__{e['quant']}__{e['framework']}__{backend}"
                if self.args.ref != "pinned":
                    cid += f"__{self.args.ref}"
                out.append({"id": cid.replace("/", "-"), "model": e["name"], "quant": e["quant"], "bits": e.get("bits"),
                            "framework": e["framework"], "backend": backend, "ref": self.args.ref, "entry": e})
        return out

    def done(self, cfg) -> bool:
        return (self.out / "configs" / f"{cfg['id']}.json").exists() and not self.args.force

    # -- preflight ---------------------------------------------------------
    def preflight(self, cfgs):
        needed = set()
        for c in cfgs:
            needed.add("mnn" if c["framework"] == "mnn" else f"llama_{c['backend']}")
        for t in sorted(needed):
            exe = "llm_bench" if t == "mnn" else "llama-bench"
            if self.adb.sh(f"test -x {dev_bin_dir(self.args.ref, t)}/{exe} && echo ok").strip() != "ok":
                sys.exit(f"{dev_bin_dir(self.args.ref, t)}/{exe} missing on the phone: "
                         f"python build_binaries.py --ref {self.args.ref} --push --serial {self.adb.serial}")
        if "llama_gpu" in needed:
            listing = self.adb.sh(f"cd {dev_bin_dir(self.args.ref, 'llama_gpu')} && "
                                  f"LD_LIBRARY_PATH=.:/vendor/lib64 ./llama-bench --list-devices 2>&1")
            if "OpenCL" not in listing:
                sys.exit(f"llama.cpp OpenCL build sees no OpenCL device:\n{listing[-800:]}")

    # -- per-invocation helpers -------------------------------------------
    def wait_ready(self) -> dict:
        res = self.gate.wait()
        return {"passed": res["passed"], "waited_s": res["waited_s"], "issues": res.get("issues"),
                "battery_temp_c": res["state"]["battery_temp_c"], "cpu_caps_khz": res["state"]["cpu_caps_khz"],
                "thermal_status": res["state"]["thermal_status"]}

    def prime(self):
        """Optional all-core burst (stop-file busy loops; `timeout` does not reliably stop children on Samsung)."""
        if self.args.prime_seconds <= 0:
            return
        dev = energy_probe.DEV_DIR
        self.adb.sh(f"rm -f {dev}/stop_load")
        for _ in range(self.info["ncpu"]):
            self.adb.sh(f"setsid sh {dev}/busy.sh </dev/null >/dev/null 2>&1 &")
        time.sleep(self.args.prime_seconds)
        energy_probe.cleanup_loads(self.adb)

    def invoke(self, cfg, tag, bin_dir, cmd, env, phase_label) -> dict:
        gate = self.wait_ready()
        self.controls.screen_off()
        self.prime()
        before = bench_common.device_state(self.adb)
        log(f"   {phase_label}: {cmd}")
        j0 = energy_probe.read_cpu_jiffies(self.adb)
        res = run_on_device(self.adb, self.controls, bin_dir, cmd, env, tag, self.raw, self.args.timeout)
        busy = energy_probe.busy_fraction(j0, energy_probe.read_cpu_jiffies(self.adb))
        after = bench_common.device_state(self.adb)
        res.update({"gate": gate, "state_before": before, "state_after": after, "cpu_busy": busy})
        if busy is not None and busy < 0.05 and not failure_reason(res):
            log(f"   [WARN] CPU only {busy:.0%} busy during this run: the process may have been frozen or throttled")
        time.sleep(self.args.rest)
        return res

    # -- one configuration -------------------------------------------------
    def run_config(self, cfg) -> dict:
        a = self.args
        model_path = ensure_model_on_device(self.adb, cfg["entry"])
        result = {"config": {k: v for k, v in cfg.items() if k != "entry"}, "model_file": cfg["entry"],
                  "started": datetime.now().isoformat(), "invocations": {}, "windows": []}
        reps = a.trials + 1
        if cfg["framework"] == "llama.cpp":
            target = f"llama_{cfg['backend']}"
            bin_dir = dev_bin_dir(a.ref, target)
            env = "LD_LIBRARY_PATH=.:/vendor/lib64"
            plan = [("prefill", llama_cmd(model_path, cfg["backend"], a.n_prompt, 0, 0, reps, a.llama_threads)),
                    ("decode", llama_cmd(model_path, cfg["backend"], 0, a.n_gen, a.n_prompt, reps, a.llama_threads))]
            for phase, cmd in plan:
                res = self.invoke(cfg, f"{cfg['id']}__{phase}", bin_dir, cmd, env, phase)
                result["invocations"][phase] = self._summ(res)
                err = failure_reason(res)
                if err:
                    return self._fail(result, err)
                result["windows"] += llama_windows(res["marks"], phase, "main")
                try:
                    j = json.loads(res["stdout"])[0]
                    result["invocations"][phase]["llama_bench"] = {k: j.get(k) for k in (
                        "build_commit", "backends", "n_threads", "flash_attn", "type_k", "type_v", "n_batch",
                        "n_ubatch", "use_mmap", "avg_ts", "stddev_ts", "samples_ts", "model_type", "model_size")}
                except (ValueError, IndexError, KeyError):
                    j = None
                result["invocations"][phase]["suspend_ratio"] = suspend_ratio(res["marks"], j)
            backend_seen = (result["invocations"]["prefill"].get("llama_bench") or {}).get("backends", "")
            if cfg["backend"] == "gpu" and "OpenCL" not in str(backend_seen):
                result["warning"] = f"GPU requested but llama-bench reports backends={backend_seen!r}"
        else:
            bin_dir = dev_bin_dir(a.ref, "mnn")
            env = "LD_LIBRARY_PATH=. PB_IGNORE_EOS=1"
            cmd = mnn_cmd(model_path, cfg["backend"], a.n_prompt, a.n_gen, a.trials, True, a.mnn_threads)
            res = self.invoke(cfg, f"{cfg['id']}__kv", bin_dir, cmd, env, "prefill+decode")
            result["invocations"]["kv"] = self._summ(res)
            err = failure_reason(res)
            if err:
                return self._fail(result, err)
            result["windows"] += mnn_windows(res["marks"], "main")
            result["invocations"]["kv"]["suspend_ratio"] = suspend_ratio(res["marks"])
            short = [w for w in result["windows"] if w["phase"] == "decode" and w["tokens"] != a.n_gen]
            if short:
                result["warning"] = f"{len(short)} decode run(s) stopped early (EOS not suppressed?)"

        result["prefill"] = throughput(result["windows"], "prefill")
        result["decode"] = throughput(result["windows"], "decode")
        worst = max((inv.get("suspend_ratio") or 0) for inv in result["invocations"].values())
        if worst > 1.05:
            result["suspended"] = worst
            result["warning"] = ((result.get("warning") or "") + f" phone suspended during timed work (boottime/monotonic "
                                 f"up to {worst:.2f}x): keep it awake (--screen on over wireless adb)").strip()
            log(f"   [WARN] {result['warning']}")
        self._extend_for_energy(cfg, result, model_path)
        result["status"] = "ok"
        result["finished"] = datetime.now().isoformat()
        log(f"   prefill {result['prefill']['tps']} +/- {result['prefill']['tps_std']} t/s, "
            f"decode {result['decode']['tps']} +/- {result['decode']['tps_std']} t/s")
        return result

    def _extend_for_energy(self, cfg, result, model_path):
        """Battery gauges update every ~1-5 s: give each phase at least --min-energy-seconds of timed work
        (extra repetitions, kept apart from the paper-protocol throughput numbers)."""
        a = self.args
        if self.headline_is_chip or a.min_energy_seconds <= 0:
            return
        for phase in ("prefill", "decode"):
            ws = [w for w in result["windows"] if w["phase"] == phase]
            secs = sum(w["b"] - w["a"] for w in ws)
            if not ws or secs >= a.min_energy_seconds:
                continue
            per_rep = secs / len(ws)
            reps = math.ceil((a.min_energy_seconds - secs) / per_rep) + 1
            if cfg["framework"] == "llama.cpp":
                bin_dir, env = dev_bin_dir(a.ref, f"llama_{cfg['backend']}"), "LD_LIBRARY_PATH=.:/vendor/lib64"
                cmd = (llama_cmd(model_path, cfg["backend"], a.n_prompt, 0, 0, reps, a.llama_threads) if phase == "prefill"
                       else llama_cmd(model_path, cfg["backend"], 0, a.n_gen, a.n_prompt, reps, a.llama_threads))
            else:
                bin_dir, env = dev_bin_dir(a.ref, "mnn"), "LD_LIBRARY_PATH=. PB_IGNORE_EOS=1"
                cmd = (mnn_cmd(model_path, cfg["backend"], a.n_prompt, 0, reps - 1, False, a.mnn_threads)
                       if phase == "prefill" else
                       mnn_cmd(model_path, cfg["backend"], a.n_prompt, a.n_gen, reps - 1, True, a.mnn_threads))
            res = self.invoke(cfg, f"{cfg['id']}__energy_{phase}", bin_dir, cmd, env, f"energy extension ({phase})")
            result["invocations"][f"energy_{phase}"] = self._summ(res)
            if failure_reason(res):
                continue
            new = (llama_windows(res["marks"], phase, "energy") if cfg["framework"] == "llama.cpp"
                   else [w for w in mnn_windows(res["marks"], "energy") if w["phase"] == phase])
            result["windows"] += new

    @staticmethod
    def _summ(res: dict) -> dict:
        return {"exit": res["exit"], "timed_out": res["timed_out"], "seconds": res["seconds"],
                "marks": len(res["marks"]), "cpu_busy": res.get("cpu_busy"), "gate": res.get("gate"),
                "state_before": res.get("state_before"), "state_after": res.get("state_after")}

    def _fail(self, result, reason) -> dict:
        result["status"] = "failed"
        result["error"] = reason
        result["finished"] = datetime.now().isoformat()
        log(f"   FAILED: {reason}")
        return result

    # -- energy --------------------------------------------------------------
    def attach_energy(self):
        self.recorder.load()
        preferred = self.profile.get("energy_method") or probe_recommendation(self.info["model"])
        headline = self.recorder.headline(preferred)
        session = json.loads((self.out / "session.json").read_text())
        session["energy_headline_method"] = headline
        session["energy_methods"] = list(self.recorder.funcs)
        (self.out / "session.json").write_text(json.dumps(session, indent=1))
        log(f"[ENERGY] methods: {list(self.recorder.funcs)}; headline: {headline}")
        for f in sorted((self.out / "configs").glob("*.json")):
            r = json.loads(f.read_text())
            if r.get("status") != "ok" or r.get("energy_session") not in (None, session["started"]):
                continue
            if r.get("energy_session") == session["started"] and "energy" in r:
                continue
            powered = any((inv.get(k) or {}).get("externally_powered")
                          for inv in r["invocations"].values() for k in ("state_before", "state_after"))
            r["energy"] = self.recorder.phase_energy(r["windows"], self.idle_window, powered)
            r["energy_session"] = session["started"]
            r["energy_headline_method"] = headline
            h = r["energy"].get(headline, {}) if headline else {}
            chip = bool(headline) and is_chip_method(headline)
            for phase in ("prefill", "decode"):
                e = h.get(phase) or {}
                r[phase]["uj_per_token"] = e.get("uj_per_token") if chip else e.get("net_uj_per_token")
                r[phase]["uj_per_token_gross"] = e.get("uj_per_token")
                r[phase]["energy_valid"] = e.get("valid", False)
                r[phase]["energy_reason"] = e.get("reason")
            r["energy_kind"] = "chip (gross)" if chip else "whole device (net of idle)"
            f.write_text(json.dumps(r, indent=1))

    # -- main loop -----------------------------------------------------------
    def run(self):
        a = self.args
        cfgs = self.configs()
        todo = [c for c in cfgs if not self.done(c)]
        log(f"device: {self.info['manufacturer']} {self.info['model']} ({self.info['soc']}), profile "
            f"{self.profile['id']} (paper column: {self.profile.get('paper_column')}); serial {self.adb.serial}")
        log(f"{len(cfgs)} configurations, {len(todo)} to run; results -> {self.out}")
        for c in cfgs:
            print(f"   {'done ' if self.done(c) else 'todo '} {c['id']}")
        if a.plan or not todo:
            return
        self.preflight(todo)
        self.adb.sh(f"mkdir -p {energy_probe.DEV_DIR}")
        energy_probe.push_text(self.adb, energy_probe.BUSY_SCRIPT, f"{energy_probe.DEV_DIR}/busy.sh")
        energy_probe.cleanup_loads(self.adb)
        idle_busy = energy_probe.idle_cpu_check(self.adb, 5.0)
        if idle_busy is not None and idle_busy > 0.15:
            log(f"[WARN] the phone is {idle_busy:.0%} busy with nothing running; something else is loading it, "
                f"which skews throughput and energy (busiest: "
                f"{' | '.join(self.adb.sh('top -b -n 1 -m 4 2>/dev/null | tail -4').split(chr(10)))[:300]})")
        disc = energy_probe.discover(self.adb)
        session = {"started": datetime.now().isoformat(), "device": self.info, "profile": self.profile,
                   "energy_discovery": {k: disc[k] for k in ("root", "powercap", "powerstats", "battery_dir",
                                                              "battery_files")},
                   "params": {k: v for k, v in vars(a).items() if k not in ("func",)},
                   "max_temp_c": self.max_temp, "idle_cpu_busy_at_start": idle_busy,
                   "screen_on": self.controls.screen_on, "wireless_adb": self.controls.wireless,
                   "binaries": {t: self.adb.sh(f"cat {dev_bin_dir(a.ref, t)}/build_info.json 2>/dev/null")
                                for t in ("llama_cpu", "llama_gpu", "mnn")}}
        for k, v in list(session["binaries"].items()):
            try:
                session["binaries"][k] = json.loads(v)
            except ValueError:
                session["binaries"][k] = None
        (self.out / "session.json").write_text(json.dumps(session, indent=1, default=str))
        self.recorder = EnergyRecorder(self.adb, self.out / "energy", disc, use_perfetto=not a.no_perfetto, log=log)
        self.headline_is_chip = disc["powercap"]["usable"]
        if disc["state"]["externally_powered"] and not self.headline_is_chip:
            log("[WARN] the phone is externally powered and has no chip counters: throughput is fine, but energy "
                "will be marked invalid (use wireless adb with the cable unplugged for energy).")
        try:
            if not a.no_controls:
                self.controls.apply()
            self.gate = bench_common.ReadinessGate(self.adb, self.max_temp, timeout_s=a.gate_timeout, log=log)
            self.gate.set_baseline(bench_common.device_state(self.adb))
            self.recorder.start()
            self.wait_ready()
            log(f"[ENERGY] idle baseline ({a.idle_seconds}s)")
            t0 = boottime_s(self.adb)
            time.sleep(a.idle_seconds)
            self.idle_window = (t0 + min(2.0, a.idle_seconds / 4), boottime_s(self.adb))
            session["idle_window"] = self.idle_window
            (self.out / "session.json").write_text(json.dumps(session, indent=1, default=str))
            for i, cfg in enumerate(todo, 1):
                log(f"[{i}/{len(todo)}] {cfg['id']}")
                r = self.run_config(cfg)
                (self.out / "configs" / f"{cfg['id']}.json").write_text(json.dumps(r, indent=1, default=str))
        finally:
            if self.recorder:
                self.recorder.stop()
            if not a.no_controls:
                self.controls.restore()
        self.attach_energy()
        write_summary(self.out)
        log(f"done -> {self.out}  (python report.py {self.out})")


def write_summary(out: Path):
    rows = []
    for f in sorted((out / "configs").glob("*.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        row = {k: c[k] for k in ("model", "quant", "framework", "backend")}
        row["status"] = r.get("status")
        row["error"] = r.get("error")
        for phase in ("prefill", "decode"):
            p = r.get(phase) or {}
            row[f"{phase}_tps"] = p.get("tps")
            row[f"{phase}_tps_std"] = p.get("tps_std")
            row[f"{phase}_reps"] = " ".join(str(x) for x in p.get("reps", []))
            row[f"{phase}_uj_per_token"] = p.get("uj_per_token")
            row[f"{phase}_energy_valid"] = p.get("energy_valid")
        row["energy_kind"] = r.get("energy_kind")
        row["energy_method"] = r.get("energy_headline_method")
        row["warning"] = r.get("warning")
        rows.append(row)
    if rows:
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial")
    ap.add_argument("--ref", choices=["pinned", "head"], default="pinned", help="which build_binaries.py build to use")
    ap.add_argument("--models", nargs="+", help="model names from models/manifest.json (default: all)")
    ap.add_argument("--quants", nargs="+", help="quant labels, e.g. Q4_0 Q4_K_M Q8_0 F16 (llama.cpp), Q4 Q8 F16 (MNN)")
    ap.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=list(FRAMEWORKS))
    ap.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    ap.add_argument("--trials", type=int, default=3, help="recorded repetitions after 1 warm-up (paper: >= 3)")
    ap.add_argument("--n-prompt", type=int, default=256)
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--max-temp", type=float, default=None, help="cool-down gate in C (default: device profile, 28)")
    ap.add_argument("--gate-timeout", type=int, default=1800, help="max seconds to wait at the gate (then flagged)")
    ap.add_argument("--rest", type=int, default=15, help="seconds after each invocation before the next gate check")
    ap.add_argument("--idle-seconds", type=int, default=40, help="idle-power baseline at session start")
    ap.add_argument("--min-energy-seconds", type=float, default=30.0,
                    help="battery energy only: extra repetitions until each phase has this much timed work (0 = off)")
    ap.add_argument("--prime-seconds", type=int, default=0, help="all-core CPU burst before each invocation (off)")
    ap.add_argument("--timeout", type=int, default=5400, help="seconds per benchmark invocation")
    ap.add_argument("--llama-threads", type=int, default=None, help="llama-bench -t (default: its own default)")
    ap.add_argument("--mnn-threads", type=int, default=None, help="llm_bench -t (default: its own default, 4)")
    ap.add_argument("--no-controls", action="store_true", help="skip airplane mode / screen off / DND")
    ap.add_argument("--screen", choices=["auto", "off", "on"], default="auto",
                    help="screen during runs: off = the paper; on = minimum brightness, needed over wireless adb "
                         "where a screen-off phone suspends mid-run; auto = off on USB, on over Wi-Fi")
    ap.add_argument("--no-perfetto", action="store_true")
    ap.add_argument("--results-dir", help="resume into an existing results directory")
    ap.add_argument("--force", action="store_true", help="rerun configurations that already have results")
    ap.add_argument("--plan", action="store_true", help="list configurations and exit")
    args = ap.parse_args()
    if args.trials < 3:
        print("[WARN] the paper uses at least 3 recorded trials")
    Session(args).run()


if __name__ == "__main__":
    main()
