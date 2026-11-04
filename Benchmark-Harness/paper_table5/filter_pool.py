#!/usr/bin/env python3
"""
Filter a model pool for one phone: try each model, quantization, runtime (llama.cpp GGUF / MNN) and backend (CPU /
GPU) on the phone, top to bottom, and drop what does not fit or breaks a limit. Accuracy is not considered.

  python filter_pool.py --serial S --max-memory 2048 --min-prefill 128 --min-decode 10 --max-energy-j 60
  python filter_pool.py --serial S --max-memory 40% --min-decode 5 --prompt-tokens 32 --answer-tokens 45
  python filter_pool.py --serial S --no-measure --min-decode 20        # re-filter what this phone already measured
  python filter_pool.py --from results/filter_oneplus8pro_pinned results/filter_oneplus8pro_head --min-decode 10

  from filter_pool import filter_pool
  result = filter_pool("SERIAL", {"max_peak_memory_mb": 2048, "min_decode_tps": 10, "max_energy_j": 60})

For each configuration, in pool order (models as listed, then 4 / 8 / 16-bit):
  1. weights larger than the phone's RAM allows: removed without running ("does not fit")
  2. peak memory from one 256 + 256-token inference; over --max-memory: removed, not benchmarked
  3. benchmark: 1 warm-up + 1 timed run of 256-token prefill and 256-token decode, each behind the same cool-down gate
     as the research protocol (battery <= the profile's limit, 28 C; CPU caps back at baseline), with the phone in
     airplane mode, background apps killed, screen per the device profile; prefill and decode limits checked
  4. energy (only when --max-energy-j is given): measured during the benchmark, attached when the session ends
Limits not given are not checked. Each model x quantization is kept if at least one runtime/backend passes; its entry
names the passing one with the fastest decode (--rank-by prefill / energy), ties to lower energy.

Energy needs the phone on battery. With --max-energy-j over USB (and no chip power counters), the tool explains how
to switch to wireless adb and waits for the phone to come back that way, unplugged (--wireless-timeout).

Everything resumes: measurements go to results/<prefix>_<ref> (default prefix filter_<device profile>), and a rerun
only measures what is missing. A session stops below 20% battery (--min-battery); charge and rerun. Output contract:
README.md ("Filtered model pool for a phone").
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run_paper_table5 as rpt  # noqa: E402
from common import (PAPER_MODELS, POOL_MODELS, RESULTS_DIR, THIRD_PARTY, dev_bin_dir, device_info,  # noqa: E402
                    is_wireless_serial, match_profile)

CONTRACT_VERSION = 1
DEFAULT_POOL = HERE / "pools" / "slm_factory_android_pool.py"
FORMAT = {"llama.cpp": "GGUF", "mnn": "MNN"}
QUANT_BITS = {"Q4_0": 4, "Q4_K_M": 4, "Q4": 4, "Q8_0": 8, "Q8": 8, "F16": 16}
MNN_QUANT = {4: "Q4", 8: "Q8", 16: "F16"}
COMBOS = [("llama.cpp", "cpu"), ("llama.cpp", "gpu"), ("mnn", "cpu"), ("mnn", "gpu")]
OPENCL_LIBS = ("/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so", "/system/lib64/libOpenCL.so")
LIMIT_KEYS = ("max_peak_memory_mb", "min_prefill_tps", "min_decode_tps", "max_energy_j")
OVER = "over limit: "  # error prefix of a configuration removed at the memory check (not benchmarked)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def step(cmd: list[str]):
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"step failed ({r.returncode}): {' '.join(cmd)}")


# ---------------------------------------------------------------------------
# The phone
# ---------------------------------------------------------------------------

def llama_march(features: set[str]) -> str:
    """-march for llama.cpp from /proc/cpuinfo Features. The default armv8.7a (llama.cpp's Android recipe, used on
    every i8mm phone measured so far) dies with SIGILL on cores without i8mm, e.g. the Snapdragon 865."""
    from build_binaries import DEFAULT_MARCH
    if {"i8mm", "asimddp"} <= features:
        return DEFAULT_MARCH
    if {"asimddp", "fphp"} <= features:
        return "armv8.2-a+dotprod+fp16"
    if "fphp" in features:
        return "armv8.2-a+fp16"
    return "armv8-a"


def phone_facts(adb) -> dict:
    info = device_info(adb)
    feats = set()
    for line in adb.sh("grep -m1 -i '^Features' /proc/cpuinfo").splitlines():
        feats |= set(line.split(":", 1)[-1].split())
    info["serialno"] = adb.sh("getprop ro.serialno").strip()
    info["abi64"] = adb.sh("getprop ro.product.cpu.abilist64").strip()
    info["cpu_features"] = sorted(feats)
    info["llama_march"] = llama_march(feats)
    libs = adb.sh(f"ls {' '.join(OPENCL_LIBS)} 2>/dev/null").split()
    info["opencl_lib"] = libs[0] if libs else None
    return info


def wait_for_wireless(adb, serialno: str, timeout_s: int):
    """The phone over wireless adb and on battery, or exit after timeout_s. Returns the Adb to use."""
    import energy_probe
    from devenv import Adb
    log("[ENERGY] the energy limit needs the phone on battery, which a USB cable prevents. To switch:\n"
        f"   1. phone and this computer on the same Wi-Fi network\n"
        f"   2. python energy_probe.py wireless --serial {adb.serial}\n"
        f"      (or: adb -s {adb.serial} tcpip 5555, then adb connect <phone Wi-Fi IP>:5555)\n"
        f"   3. unplug the USB cable (do not run the 'energy_probe.py collect' it suggests: this tool runs it\n"
        f"      itself on a new phone, and two at once load the phone during the benchmark)\n"
        f"   Waiting up to {timeout_s // 60} min for the phone to reconnect wirelessly, unplugged ...")
    t0, last = time.time(), ""
    while time.time() - t0 < timeout_s:
        for serial, state in energy_probe.list_devices(adb.bin):
            if state != "device" or not is_wireless_serial(serial):
                continue
            cand = Adb(serial)
            if cand.sh("getprop ro.serialno", timeout=15).strip() != serialno:
                continue
            if not re.search(r"(AC|USB|Wireless|Dock) powered: true", cand.sh("dumpsys battery", timeout=15)):
                log(f"[ENERGY] connected wirelessly as {serial}, on battery")
                return cand
            if last != serial:
                log(f"[ENERGY] {serial} connected; still charging: unplug the USB cable")
                last = serial
        time.sleep(5)
    sys.exit(f"the phone did not come back over wireless adb on battery within {timeout_s} s; rerun when it has, "
             f"or leave out --max-energy-j to filter on memory, prefill and decode over USB")


def built_march(ref: str, target: str) -> str | None:
    f = THIRD_PARTY / "out" / ref / target / "build_info.json"
    if not f.is_file():
        return None
    args = json.loads(f.read_text()).get("cmake_args", [])
    return next((a.split("-march=", 1)[1] for a in args if a.startswith("-DCMAKE_C_FLAGS=-march=")), None)


def ensure_binaries(ref: str, info: dict, serial: str):
    targets = ["llama_cpu", "mnn"] + (["llama_gpu"] if info["opencl_lib"] else [])
    rebuild = [t for t in targets if t.startswith("llama") and built_march(ref, t) != info["llama_march"]]
    if not (THIRD_PARTY / "out" / ref / "mnn" / "llm_bench").is_file():
        rebuild.append("mnn")
    if rebuild:
        step(["build_binaries.py", "--ref", ref, "--targets", *rebuild, "--llama-march", info["llama_march"]])
    step(["build_binaries.py", "--ref", ref, "--targets", *targets, "--push-only", "--serial", serial])


def gpu_paths(adb, ref: str, info: dict) -> dict:
    """{framework: None if its GPU backend can run, else why not}."""
    if not info["opencl_lib"]:
        why = "no OpenCL driver on the phone"
        return {"llama.cpp": why, "mnn": why}
    listing = adb.sh(f"cd {dev_bin_dir(ref, 'llama_gpu')} && LD_LIBRARY_PATH=.:/vendor/lib64 "
                     f"./llama-bench --list-devices 2>&1", timeout=600)
    llama = None if "OpenCL" in listing else (
        f"llama.cpp {ref} OpenCL build finds no device: {' '.join(listing.split()[-25:])}")
    return {"llama.cpp": llama, "mnn": None}


# ---------------------------------------------------------------------------
# Measuring, top to bottom
# ---------------------------------------------------------------------------

class FilterSession(rpt.Session):
    """run_paper_table5's session (phone controls, cool-down gate, energy recording) over the pool in order, with
    peak memory measured first: a configuration over the memory limit is not benchmarked."""

    def __init__(self, args, limits: dict, order: dict, no_gpu: set):
        super().__init__(args)
        self.limits, self.order, self.no_gpu = limits, order, no_gpu
        (self.out / "memory").mkdir(exist_ok=True)

    def configs(self) -> list[dict]:
        cfgs = [c for c in super().configs() if not (c["backend"] == "gpu" and c["framework"] in self.no_gpu)]
        return sorted(cfgs, key=lambda c: (self.order.get(c["model"], len(self.order)), c.get("bits") or 99,
                                           COMBOS.index((c["framework"], c["backend"]))))

    def too_big(self, cfg) -> bool:
        b = sum(cfg["entry"]["files"].values())
        if cfg["framework"] == "mnn" and cfg["quant"].upper() == "F16" and cfg["backend"] == "cpu":
            b = int(b * rpt.MNN_F16_RAM_FACTOR)
        ram = (self.info.get("mem_total_kb") or 0) * 1024
        return bool(ram) and b > rpt.RAM_FIT_FRACTION * ram

    def memory(self, cfg) -> dict:
        """memory_usage.py's measurement, cached in <results>/memory/ in its format."""
        import memory_usage
        f = self.out / "memory" / f"{cfg['id']}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            if rec.get("exit") == 0:
                return rec
        path = rpt.ensure_model_on_device(self.adb, cfg["entry"])
        self.adb.sh("am kill-all")
        rec = {"config": {k: v for k, v in cfg.items() if k != "entry"},
               "storage_bytes": sum(cfg["entry"]["files"].values()), "measured": datetime.now().isoformat()}
        res = memory_usage.run_one(self.adb, cfg, path, self.args.ref, self.args.llama_threads, self.args.timeout)
        if res.get("exit") is None and self.adb.run(["get-state"], timeout=15).stdout.strip() != "device":
            # the phone dropped off (Wi-Fi, this computer asleep): stop the session; a rerun measures this again
            raise RuntimeError("phone unreachable during the memory measurement")
        rec.update(res)
        f.write_text(json.dumps(rec, indent=1))
        time.sleep(10)  # let the kernel return the memory before the benchmark's free-RAM check
        return rec

    def run_config(self, cfg) -> dict:
        from results_all_md import peak_mb
        if self.too_big(cfg):
            return super().run_config(cfg)  # records "does not fit in RAM" without running
        rec = self.memory(cfg)
        peak, lim = peak_mb(rec), self.limits["max_peak_memory_mb"]
        log(f"   peak memory {peak:.0f} MB" if peak else f"   peak memory not measured (exit {rec.get('exit')})")
        if lim is not None and (peak is None or peak > lim):
            result = {"config": {k: v for k, v in cfg.items() if k != "entry"}, "model_file": cfg["entry"],
                      "started": datetime.now().isoformat(), "invocations": {}, "windows": []}
            why = f"peak memory {peak:.0f} MB > {lim:g} MB" if peak else "peak memory not measured"
            return self._fail(result, OVER + why)
        r = super().run_config(cfg)
        if r.get("status") == "ok":
            miss = misses(dict(runs=True, peak_memory_mb=peak, prefill_tps=r["prefill"]["tps"],
                               decode_tps=r["decode"]["tps"]), dict(self.limits, max_energy_j=None))
            log("   REMOVED: " + "; ".join(miss) if miss else "   passes memory, prefill and decode (energy: at "
                "the end of the session)" if self.limits["max_energy_j"] is not None else "   passes")
        return r


def recover_stopped_session(out: Path, serial: str):
    """A session that stopped before its energy step (phone unreachable) left its recording on the phone: pull it
    and attach it before the next session overwrites it. If that fails, its configurations are measured again."""
    starts = {f.stem: json.loads(f.read_text())["started"] for f in (out / "sessions").glob("*.json")}
    if not starts:
        return
    stamp = max(starts, key=starts.get)
    edir = out / "energy" / stamp
    if edir.is_dir() and any(edir.iterdir()):
        return
    log(f"[ENERGY] session {stamp} stopped before its energy step; recovering its recording from the phone")
    try:
        rpt.recover_energy(argparse.Namespace(results_dir=str(out), serial=serial))
        return
    except (SystemExit, RuntimeError, OSError, ValueError) as e:
        log(f"[ENERGY] could not recover it ({e}); its configurations are measured again")
    for f in (out / "configs").glob("*.json"):
        r = json.loads(f.read_text())
        if r.get("status") == "ok" and not r.get("energy") and r.get("started", "") >= starts[stamp]:
            f.unlink()


def session_args(serial, ref, models, quants, out, screen, energy: bool, extra) -> argparse.Namespace:
    argv = ["--serial", serial, "--ref", ref, "--models", *models, "--quants", *quants, "--results-dir", str(out),
            "--trials", "1", "--screen", screen, *extra]
    if not energy:  # no energy limit: skip the energy-only work (extra repetitions, idle windows)
        argv += ["--min-energy-seconds", "0", "--pre-idle-seconds", "0"]
    return rpt.build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------

def parse_memory_limit(value, mem_total_kb: int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.endswith("%"):
        if not mem_total_kb:
            raise SystemExit("a memory limit as a percentage needs the phone's RAM (MemTotal); give MB instead")
        return round(float(value[:-1]) / 100 * mem_total_kb * 1024 / 1e6)
    return float(value)


def config_row(k, r, mem, request) -> dict:
    from results_all_md import peak_mb, short_reason
    from results_md import val
    model, backend, fw, quant = k
    ok = r.get("status") == "ok"
    removed_early = str(r.get("error", "")).startswith(OVER)
    uj = {ph: val(r, ph, "uj") if ok else None for ph in ("prefill", "decode")}
    energy = (request["prompt_tokens"] * uj["prefill"] + request["answer_tokens"] * uj["decode"]) / 1e6 \
        if uj["prefill"] and uj["decode"] else None
    peak = peak_mb(mem) if ok or removed_early else None
    return {"model": model, "quant": quant, "bits": r.get("config", {}).get("bits") or QUANT_BITS.get(quant),
            "runtime": fw, "format": FORMAT[fw], "backend": backend,
            "runs": ok or removed_early, "benchmarked": ok,
            "error": None if ok or removed_early else short_reason(r, "", k),
            "peak_memory_mb": round(peak) if peak else None,
            "prefill_tps": round(val(r, "prefill", "tps"), 2) if ok else None,
            "decode_tps": round(val(r, "decode", "tps"), 2) if ok else None,
            "energy_j_per_request": round(energy, 3) if energy is not None else None}


def misses(c: dict, lim: dict) -> list[str]:
    if not c["runs"]:
        return [f"does not run: {c['error']}"]
    out = []
    if lim["max_peak_memory_mb"] is not None:
        if c["peak_memory_mb"] is None:
            out.append("peak memory not measured")
        elif c["peak_memory_mb"] > lim["max_peak_memory_mb"]:
            out.append(f"peak memory {c['peak_memory_mb']} MB > {lim['max_peak_memory_mb']:g} MB")
    if c.get("benchmarked") is False:
        return out or ["not benchmarked: removed at the memory check under earlier limits"]
    if lim["min_prefill_tps"] is not None and c["prefill_tps"] < lim["min_prefill_tps"]:
        out.append(f"prefill {c['prefill_tps']:g} < {lim['min_prefill_tps']:g} tokens/s")
    if lim["min_decode_tps"] is not None and c["decode_tps"] < lim["min_decode_tps"]:
        out.append(f"decode {c['decode_tps']:g} < {lim['min_decode_tps']:g} tokens/s")
    if lim["max_energy_j"] is not None:
        if c["energy_j_per_request"] is None:
            out.append("energy not measured")
        elif c["energy_j_per_request"] > lim["max_energy_j"]:
            out.append(f"energy {c['energy_j_per_request']:g} J > {lim['max_energy_j']:g} J")
    return out


def rank_key(rank_by: str):
    inf = float("inf")
    e = lambda c: c["energy_j_per_request"] if c["energy_j_per_request"] is not None else inf  # noqa: E731
    if rank_by == "energy":
        return lambda c: (e(c), -c["decode_tps"], -c["prefill_tps"])
    if rank_by == "prefill":
        return lambda c: (-c["prefill_tps"], e(c), -c["decode_tps"])
    return lambda c: (-c["decode_tps"], e(c), -c["prefill_tps"])


METRICS = ("peak_memory_mb", "prefill_tps", "decode_tps", "energy_j_per_request")


def expected_matrix(pool: dict, gguf, mnn) -> list[tuple]:
    """Every (model, backend, framework, quant) the pool asks for."""
    quants = {"llama.cpp": list(gguf), "mnn": [MNN_QUANT[b] for b in mnn]}
    return [(m, b, fw, q) for m in pool for fw in quants for q in quants[fw] for b in ("cpu", "gpu")]


def not_tried_reasons(pool: dict, expected, gpu: dict, conversion: dict) -> dict:
    """{key: why it was not tried}: its runtime's GPU path cannot run, or the model could not be converted."""
    out = {}
    for k in expected:
        m, b, fw, q = k
        conv = (conversion.get(m) or {}).get(fw) or {}
        why_gpu = (gpu.get(pool[m][1]) or {}).get(fw) if b == "gpu" else None
        if q in conv:
            out[k] = conv[q]
        elif why_gpu:
            out[k] = why_gpu
    return out


def _blank(k, error, reason) -> dict:
    m, b, fw, q = k
    return {"model": m, "quant": q, "bits": QUANT_BITS.get(q), "runtime": fw, "format": FORMAT[fw], "backend": b,
            "runs": False, "benchmarked": False, "error": error, **{x: None for x in METRICS},
            "reasons": [reason], "passes": False}


def build_contract(phone: dict, dirs: list[Path], gpu: dict, lim: dict, request: dict, rank_by: str,
                   pool: dict | None = None, expected=None, not_tried: dict | None = None,
                   conversion: dict | None = None) -> dict:
    """pool: {name: (hf_id, ref)} in pool order (None: every model measured). expected: the (model, backend,
    framework, quant) the pool asks for (None: only what was measured; completeness unknown). not_tried: {key: why}.
    """
    from results_all_md import merge_dirs
    from results_md import load_dir, load_memory
    bench = merge_dirs(dirs, lambda d: load_dir(d) if (d / "configs").is_dir() else {})
    memory = load_memory([d for d in dirs if (d / "memory").is_dir()])
    if pool is None:
        names = list(dict.fromkeys([*POOL_MODELS, *PAPER_MODELS, *sorted({k[0] for k in bench})]))
        pool = {n: ((POOL_MODELS.get(n) or (PAPER_MODELS.get(n),))[0], None) for n in names
                if any(k[0] == n for k in bench)}
    not_tried = not_tried or {}
    keys = set(expected) if expected is not None else {k for k in bench if k[0] in pool}
    order = {n: i for i, n in enumerate(pool)}
    keys = sorted(keys, key=lambda k: (order[k[0]], QUANT_BITS.get(k[3], 99), COMBOS.index((k[2], k[1]))))
    configs, missing = [], []
    for k in keys:
        if k in bench:
            c = config_row(k, bench[k], memory.get(k), request)
            c["reasons"] = misses(c, lim)
            c["passes"] = not c["reasons"]
        elif k in not_tried:
            c = _blank(k, not_tried[k], f"not tried: {not_tried[k]}")
        else:
            c = _blank(k, "not measured yet", "not measured yet")
            missing.append(f"{k[0]}__{k[3]}__{k[2]}__{k[1]}")
        configs.append(c)
    groups: dict = {}
    for c in configs:
        groups.setdefault((c["model"], c["bits"]), []).append(c)
    entries = []
    for (model, bits), cs in groups.items():
        passing = sorted((c for c in cs if c["passes"]), key=rank_key(rank_by))
        best = passing[0] if passing else None
        entries.append({
            "model": model, "hf_id": pool[model][0], "bits": bits, "kept": bool(passing),
            "best": {k: best[k] for k in ("quant", "runtime", "format", "backend", *METRICS)} if best else None,
            "alternatives": [{k: c[k] for k in ("quant", "runtime", "format", "backend", *METRICS)}
                             for c in passing[1:]],
            "tried": [{k: c[k] for k in ("quant", "runtime", "format", "backend", "passes", "reasons", *METRICS)}
                      for c in cs]})
    kept = [e for e in entries if e["kept"]]
    count = lambda field: {v: sum(e["best"][field] == v for e in kept)  # noqa: E731
                           for v in sorted({e["best"][field] for e in kept})}
    return {
        "contract_version": CONTRACT_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "complete": (not missing) if expected is not None else None,
        "not_measured": missing,
        "phone": phone,
        "gpu": gpu,
        "conversion_failed": conversion or {},
        "measurement": {"results_dirs": [Path(d).as_posix() for d in dirs], "request": request,
                        "configs_expected": len(keys),
                        "configs_measured": sum(k in bench for k in keys),
                        "configs_benchmarked": sum(c["benchmarked"] for c in configs),
                        "energy_methods": sorted({str(r.get("energy_headline_method")) for r in bench.values()
                                                  if r.get("status") == "ok" and r.get("energy_headline_method")})},
        "limits": lim,
        "rank_by": rank_by,
        "filtered_pool": [{"model": e["model"], "hf_id": e["hf_id"], "bits": e["bits"],
                           **{k: e["best"][k] for k in ("quant", "runtime", "format", "backend")}} for e in kept],
        "entries": entries,
        "summary": {"entries": len(entries), "kept": len(kept), "removed": len(entries) - len(kept),
                    "models_kept": len({e["model"] for e in kept}), "backend": count("backend"),
                    "runtime": count("runtime")},
    }


def print_table(c: dict):
    s = c["summary"]
    print(f"\n{c['phone'].get('label')}: {s['kept']} of {s['entries']} model/quantization entries kept"
          + ("" if c["complete"] is not False else f"  (INCOMPLETE: {len(c['not_measured'])} configurations not "
                                                  f"measured yet; rerun the same command)"))
    print(f"  {'model':26s} {'bits':>4s}  {'quant':6s} {'runtime':10s} {'backend':7s} {'mem MB':>7s} {'prefill':>8s} "
          f"{'decode':>7s} {'J/req':>7s}")
    fmt = lambda v: "-" if v is None else f"{v:g}"  # noqa: E731
    for e in c["entries"]:
        if e["kept"]:
            b = e["best"]
            print(f"  {e['model']:26s} {e['bits']:>4d}  {b['quant']:6s} {b['runtime']:10s} {b['backend']:7s} "
                  f"{fmt(b['peak_memory_mb']):>7s} {fmt(b['prefill_tps']):>8s} {fmt(b['decode_tps']):>7s} "
                  f"{fmt(b['energy_j_per_request']):>7s}")
        else:
            reasons = "; ".join(sorted({r for t in e["tried"] for r in t["reasons"]}))
            print(f"  {e['model']:26s} {e['bits']:>4d}  removed: {reasons[:110]}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _limits(limits: dict | None, mem_total_kb) -> dict:
    limits = {k: v for k, v in (limits or {}).items() if v is not None}
    unknown = set(limits) - set(LIMIT_KEYS)
    if unknown:
        raise ValueError(f"unknown limits {sorted(unknown)}; expected {list(LIMIT_KEYS)}")
    lim = {k: limits.get(k) for k in LIMIT_KEYS}
    lim["max_peak_memory_mb"] = parse_memory_limit(lim["max_peak_memory_mb"], mem_total_kb)
    for k in ("min_prefill_tps", "min_decode_tps", "max_energy_j"):
        lim[k] = float(lim[k]) if lim[k] is not None else None
    bad = [k for k, v in lim.items() if v is not None and v < 0]
    if bad:
        raise ValueError(f"limits must not be negative: {bad}")
    return lim


def _phone_summary(info: dict, profile: dict) -> dict:
    out = {k: info.get(k) for k in ("serial", "manufacturer", "model", "soc", "android", "abi64", "llama_march",
                                    "opencl_lib")}
    out.update(label=profile.get("label"), profile=profile.get("id"),
               ram_gib=round(info["mem_total_kb"] / 2**20, 2) if info.get("mem_total_kb") else None)
    return out


def _run_tail(cmd: list[str]) -> tuple[int, str]:
    """Run a script here, streaming its output; return its exit code and last lines."""
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    p = subprocess.Popen([sys.executable, *cmd], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    tail: list[str] = []
    for line in p.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line.rstrip()])[-6:]
    return p.wait(), " | ".join(t for t in tail if t)[-400:]


def prepare(name: str, hf_id: str, ref: str, gguf, mnn, serial: str, quantize_bin: str | None) -> dict:
    """Convert one model for both runtimes, each separately so one failing does not stop the other or the run.
    Returns {framework: {quant: why it is missing}} for the artifacts that could not be made."""
    failed = {}
    for fw, fmt_args in (("llama.cpp", ["--gguf", *gguf, "--mnn"]), ("mnn", ["--gguf", "--mnn", *map(str, mnn)])):
        want = list(gguf) if fw == "llama.cpp" else [MNN_QUANT[b] for b in mnn]
        if not want:
            continue
        cmd = ["prepare_models.py", "--model", f"{name}={hf_id}", "--ref", ref, *fmt_args, "--serial", serial]
        code, tail = _run_tail(cmd + (["--quantize-bin", quantize_bin] if quantize_bin else []))
        have = {e["quant"] for e in rpt.load_manifest()
                if e["name"] == name and e["framework"] == fw and e.get("ref", "pinned") == ref}
        lost = [q for q in want if q not in have]
        if lost:
            why = f"conversion to {FORMAT[fw]} failed (prepare_models.py exit {code}): {tail}"
            failed[fw] = {q: why for q in lost}
            log(f"[PREPARE] {name} {FORMAT[fw]} {', '.join(lost)}: {why}")
    return failed


def filter_pool(serial: str, limits: dict | None = None, *, pool_file=DEFAULT_POOL, gguf=("Q4_0", "Q8_0", "F16"),
                mnn=(4, 8, 16), prompt_tokens: int = 256, answer_tokens: int = 256, rank_by: str = "decode",
                measure: bool = True, wireless_timeout_s: int = 900, screen: str | None = None,
                results_prefix: str | None = None, quantize_bin: str | None = None, extra_args=(),
                out: str | Path | None = None) -> dict:
    """Filter the pool for the phone `serial` and return the output contract (README.md; schema in
    filter_pool_contract.schema.json).

    limits: any of max_peak_memory_mb (MB, or a share of RAM such as "40%"), min_prefill_tps, min_decode_tps,
    max_energy_j (per request of prompt_tokens + answer_tokens); a limit left out (or None) is not checked."""
    from devenv import Adb
    from pool import load_pool
    if rank_by not in ("decode", "prefill", "energy"):
        raise ValueError(f"rank_by must be decode, prefill or energy, got {rank_by!r}")
    _limits({k: v for k, v in (limits or {}).items() if not (isinstance(v, str) and v.endswith("%"))}, None)
    rpt.keep_host_awake()
    adb = Adb(serial)
    info = phone_facts(adb)
    profile = match_profile(info)
    prefix = results_prefix or f"filter_{profile['id']}"
    log(f"{info['manufacturer']} {info['model']} ({info['soc']}), {info['mem_total_kb'] / 2**20:.1f} GiB, "
        f"profile {profile['id']}, llama.cpp -march={info['llama_march']}, OpenCL {info['opencl_lib'] or 'none'}")
    if not info["abi64"]:
        raise SystemExit("this phone runs 32-bit Android only (no arm64-v8a ABI); the benchmark binaries are arm64")
    lim = _limits(limits, info.get("mem_total_kb"))
    pool = load_pool(pool_file)
    refs = [r for r in ("pinned", "head") if any(ref == r for _, ref in pool.values())]
    quants = list(dict.fromkeys([*gguf, *(MNN_QUANT[b] for b in mnn)]))
    dirs = [RESULTS_DIR / f"{prefix}_{r}" for r in refs]
    state_file = RESULTS_DIR / f"{prefix}_state.json"
    state = json.loads(state_file.read_text()) if state_file.is_file() else {"gpu": {}, "conversion_failed": {}}
    if measure:
        rpt.keep_host_awake()  # a computer that sleeps drops the phone's wireless adb connection mid-run
        if lim["max_energy_j"] is not None:
            import energy_probe
            disc = energy_probe.discover(adb)
            if disc["state"]["externally_powered"] and not disc["powercap"]["usable"]:
                adb = wait_for_wireless(adb, info["serialno"], wireless_timeout_s)
                info["serial"] = adb.serial
            from energy import probe_recommendation
            if not disc["powercap"]["usable"] and not profile.get("energy_method") \
                    and not probe_recommendation(info["model"]):
                # a phone seen for the first time: find which battery reading is repeatable before relying on it
                log("[ENERGY] first run on this phone model: energy_probe.py collect picks the battery energy "
                    "method (~15 min)")
                code, tail = _run_tail(["energy_probe.py", "collect", "--serial", adb.serial])
                if code or not probe_recommendation(info["model"]):
                    log(f"[ENERGY] energy probe gave no recommendation (exit {code}: {tail}); using the default "
                        f"battery method order")
        screen = screen or profile.get("screen", "auto")
        order = {n: i for i, n in enumerate(pool)}
        state["conversion_failed"] = {}
        for ref, out_dir in zip(refs, dirs):
            ensure_binaries(ref, info, adb.serial)
            state["gpu"][ref] = gpu_paths(adb, ref, info)
            for fw, why in state["gpu"][ref].items():
                if why:
                    log(f"[GPU] {ref} {fw}: {why}; CPU only")
            models = []
            for name, (hf_id, r) in pool.items():
                if r != ref:
                    continue
                failed = prepare(name, hf_id, ref, gguf, mnn, adb.serial, quantize_bin)
                if failed:
                    state["conversion_failed"][name] = failed
                if any(e["name"] == name and e.get("ref", "pinned") == ref for e in rpt.load_manifest()):
                    models.append(name)
            state_file.write_text(json.dumps(state, indent=1))
            if not models:
                continue
            if lim["max_energy_j"] is not None:
                recover_stopped_session(out_dir, adb.serial)
            args = session_args(adb.serial, ref, models, quants, out_dir, screen, lim["max_energy_j"] is not None,
                                list(extra_args))
            FilterSession(args, lim, order, {fw for fw, why in state["gpu"][ref].items() if why}).run()
    elif not any(d.is_dir() for d in dirs):
        raise SystemExit(f"nothing measured yet under results/{prefix}_*; run with measure=True")
    expected = expected_matrix(pool, gguf, mnn)
    contract = build_contract(_phone_summary(info, profile), [d for d in dirs if d.is_dir()], state["gpu"], lim,
                              {"prompt_tokens": prompt_tokens, "answer_tokens": answer_tokens}, rank_by, pool,
                              expected, not_tried_reasons(pool, expected, state["gpu"], state["conversion_failed"]),
                              state["conversion_failed"])
    out = Path(out) if out else RESULTS_DIR / f"{prefix}_filtered_pool.json"
    out.write_text(json.dumps(contract, indent=1))
    return contract


def from_results(dirs, limits: dict | None = None, *, prompt_tokens: int = 256, answer_tokens: int = 256,
                 rank_by: str = "decode") -> dict:
    """The same contract from results folders already on this computer (no phone)."""
    dirs = [Path(d) for d in dirs]
    session = next((json.loads((d / "session.json").read_text()) for d in dirs if (d / "session.json").is_file()), {})
    info = session.get("device", {})
    profile = session.get("profile") or (match_profile(info) if info else {"id": dirs[0].name})
    return build_contract(_phone_summary(info, profile), dirs, {}, _limits(limits, info.get("mem_total_kb")),
                          {"prompt_tokens": prompt_tokens, "answer_tokens": answer_tokens}, rank_by)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--serial", help="the phone (adb serial)")
    src.add_argument("--from", dest="from_dirs", nargs="+", metavar="DIR",
                     help="filter results folders already on this computer instead (no phone)")
    ap.add_argument("--no-measure", action="store_true", help="with --serial: re-filter what is already measured")
    ap.add_argument("--pool-file", default=str(DEFAULT_POOL))
    ap.add_argument("--gguf", nargs="+", default=["Q4_0", "Q8_0", "F16"], help="GGUF quantizations")
    ap.add_argument("--mnn", nargs="+", type=int, default=[4, 8, 16], help="MNN quant_bit levels")
    ap.add_argument("--max-memory", help="peak memory limit: MB, or a share of the phone's RAM such as 40%%")
    ap.add_argument("--min-prefill", type=float, help="prefill tokens/s")
    ap.add_argument("--min-decode", type=float, help="decode tokens/s")
    ap.add_argument("--max-energy-j", type=float, help="energy per request (prompt + answer tokens), J")
    ap.add_argument("--prompt-tokens", type=int, default=256, help="request shape for the energy limit")
    ap.add_argument("--answer-tokens", type=int, default=256)
    ap.add_argument("--rank-by", choices=["decode", "prefill", "energy"], default="decode",
                    help="which passing runtime/backend an entry names")
    ap.add_argument("--wireless-timeout", type=int, default=900,
                    help="seconds to wait for the switch to wireless adb when --max-energy-j is given over USB")
    ap.add_argument("--screen", choices=["auto", "off", "on", "panel", "awake"],
                    help="run_paper_table5.py --screen (default: the device profile's 'screen', else auto)")
    ap.add_argument("--quantize-bin", help="host llama-quantize, passed to prepare_models.py")
    ap.add_argument("--results-prefix", help="results/<prefix>_<ref> (default: filter_<device profile>)")
    ap.add_argument("--out", help="output JSON (default: results/<prefix>_filtered_pool.json)")
    args, extra = ap.parse_known_args()  # anything else goes to run_paper_table5.py's session
    if extra and args.from_dirs:
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    limits = {"max_peak_memory_mb": args.max_memory, "min_prefill_tps": args.min_prefill,
              "min_decode_tps": args.min_decode, "max_energy_j": args.max_energy_j}
    shape = dict(prompt_tokens=args.prompt_tokens, answer_tokens=args.answer_tokens, rank_by=args.rank_by)
    try:
        _limits({k: v for k, v in limits.items() if not (isinstance(v, str) and v.endswith("%"))}, None)
    except ValueError as e:
        ap.error(str(e))
    if args.from_dirs:
        contract = from_results(args.from_dirs, limits, **shape)
        out = Path(args.out) if args.out else RESULTS_DIR / f"{Path(args.from_dirs[0]).name}_filtered_pool.json"
        out.write_text(json.dumps(contract, indent=1))
    else:
        contract = filter_pool(args.serial, limits, pool_file=args.pool_file, gguf=args.gguf, mnn=args.mnn,
                               measure=not args.no_measure, wireless_timeout_s=args.wireless_timeout,
                               screen=args.screen, results_prefix=args.results_prefix,
                               quantize_bin=args.quantize_bin, extra_args=extra, out=args.out, **shape)
        prefix = args.results_prefix or f"filter_{contract['phone']['profile']}"
        out = Path(args.out) if args.out else RESULTS_DIR / f"{prefix}_filtered_pool.json"
    print_table(contract)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
