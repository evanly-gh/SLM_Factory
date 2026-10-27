#!/usr/bin/env python3
"""
Storage and memory use of every model configuration, for the Table 5 results (run after the benchmarks).

For each configuration (model x quantization x framework x backend) of one build ref:
  * storage: bytes of the model files (GGUF file, or the MNN export directory) as pushed to the phone
  * memory: one inference of the paper's workload (256-token prompt, 256 generated tokens) while a sampler on the phone
    reads, every ~0.5 s, the benchmark process's VmHWM / VmRSS / VmSwap, its GPU memory and total memory from
    `dumpsys meminfo <pid>` ("GL mtrack": buffers the Adreno driver allocated; "TOTAL PSS": everything, GPU included),
    and the phone's MemAvailable. Recorded: peak total (PSS incl. GPU), peak GPU, peak CPU-side RSS, peak swap, and
    the largest drop in MemAvailable from the 5 s before the run. RSS alone undercounts GPU runs badly: Qwen3-0.6B
    Q8_0 on the GPU held 0.24 GB RSS but 1.42 GB of GPU memory.
Throughput of this single run is not used: the benchmarks measure that under the full protocol.

  python memory_usage.py --serial S --ref pinned --pool-file pools/slm_factory_android_pool.py --results-dir results/slm_pool_s23_pinned
  python memory_usage.py --serial S --ref head   --pool-file pools/slm_factory_android_pool.py --results-dir results/slm_pool_s23_head

Writes <results-dir>/memory/<config id>.json; resumable (--force reruns). Configurations whose benchmark result says
"does not fit in RAM" are recorded with their storage only.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import energy_probe
import pool as poolmod
import run_paper_table5 as rpt
from common import DeviceControls, dev_bin_dir
from devenv import Adb

WORK = "/data/local/tmp/paper_bench/memprobe"
SAMPLER = r"""
exe=$1; out=$2
: > $out
while [ ! -f {work}/exit ]; do
  t=$(cat /proc/uptime); t=${{t%% *}}
  avail=$(grep MemAvailable /proc/meminfo); avail=${{avail//[!0-9]/}}
  gsys=$(grep GPUTotalUsed /proc/meminfo); gsys=${{gsys//[!0-9]/}}
  pid=$(pidof $exe); pid=${{pid%% *}}
  if [ -n "$pid" ]; then
    set -- $(grep -E '^(VmRSS|VmHWM|VmSwap):' /proc/$pid/status | tr -s ' ' | cut -d' ' -f2)
    hwm=$1; rss=$2; swp=$3
    m=$(dumpsys meminfo $pid 2>/dev/null)
    gl=$(echo "$m" | grep 'GL mtrack' | tr -s ' ' | cut -d' ' -f4)
    pss=$(echo "$m" | grep 'TOTAL PSS:' | tr -s ' ' | cut -d' ' -f4)
    echo "$t $avail $pid $hwm $rss $swp ${{gl:--}} ${{pss:--}} ${{gsys:--}}" >> $out
  else
    echo "$t $avail - - - - - - ${{gsys:--}}" >> $out
  fi
  sleep 0.5
done
"""


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_one(adb: Adb, cfg: dict, model_path: str, ref: str, threads: int | None, timeout_s: int) -> dict:
    if cfg["framework"] == "llama.cpp":
        bin_dir, exe = dev_bin_dir(ref, f"llama_{cfg['backend']}"), "llama-bench"
        env = "LD_LIBRARY_PATH=.:/vendor/lib64"
        # -pg 256,256: one prompt of 256 tokens then 256 generated tokens, the paper's workload in one test
        cmd = (f"./llama-bench -m {model_path} -p 0 -n 0 -pg 256,256 -r 1 -ngl {99 if cfg['backend'] == 'gpu' else 0} "
               f"-o json" + (f" -t {threads}" if threads else ""))
    else:
        bin_dir, exe = dev_bin_dir(ref, "mnn"), "llm_bench"
        env = "LD_LIBRARY_PATH=. PB_IGNORE_EOS=1"
        cmd = rpt.mnn_cmd(model_path, cfg["backend"], 256, 256, 1, True, None)
    adb.sh(f"rm -rf {WORK}; mkdir -p {WORK}")
    energy_probe.push_text(adb, SAMPLER.format(work=WORK), f"{WORK}/sampler.sh")
    energy_probe.push_text(adb, f"sleep 5; cd {bin_dir} && {env} {cmd} > {WORK}/stdout.txt 2> {WORK}/stderr.txt; "
                                f"echo $? > {WORK}/exit\n", f"{WORK}/run.sh")
    adb.sh(f"setsid sh {WORK}/sampler.sh {exe} {WORK}/samples.txt </dev/null >/dev/null 2>&1 &")
    adb.sh(f"setsid sh {WORK}/run.sh </dev/null >/dev/null 2>&1 &")
    t0, code, swap_kb = time.time(), None, 0
    while time.time() - t0 < timeout_s:
        time.sleep(3)
        out = adb.sh(f"cat {WORK}/exit 2>/dev/null", timeout=30).strip()
        if out.lstrip("-").isdigit():
            code = int(out)
            break
        tail = adb.sh(f"tail -n 1 {WORK}/samples.txt 2>/dev/null").split()
        if len(tail) >= 8 and tail[5].isdigit():
            swap_kb = max(swap_kb, int(tail[5]))
        if swap_kb > rpt.SWAP_LIMIT_MB * 1024:  # same guard as the benchmarks: do not drive the phone into swap
            rpt.stop_benchmark(adb, WORK, exe)
            code = -9
            break
    else:
        rpt.stop_benchmark(adb, WORK, exe)
    adb.sh(f"touch {WORK}/exit")  # stops the sampler
    time.sleep(1)
    rows = [l.split() for l in adb.sh(f"cat {WORK}/samples.txt", timeout=60).splitlines()]
    stderr = adb.sh(f"tail -c 2000 {WORK}/stderr.txt")
    base = [int(r[1]) for r in rows if r[2] == "-" and r[1].isdigit()]
    run = [r for r in rows if r[2] != "-" and r[3].isdigit()]
    num = lambda i: [int(r[i]) for r in run if r[i].isdigit()]  # noqa: E731
    mib = lambda kb: round(kb / 1024, 1) if kb is not None else None  # noqa: E731
    # Some phones (OnePlus 15, Android 16) report no per-process GPU memory in dumpsys meminfo ("GL mtrack" stays ~0.2 MiB while
    # the GPU holds gigabytes), but /proc/meminfo has a system-wide GPUTotalUsed. Its rise above the pre-run baseline is the
    # process's GPU memory (the device is otherwise idle: airplane mode, background apps killed). Where the counter is absent
    # (e.g. the Galaxy S23) this stays inactive and the original GL-mtrack/PSS numbers are used.
    gbase = sorted(int(r[8]) for r in rows[:len(rows)] if r[2] == "-" and len(r) > 8 and r[8].isdigit())
    gbase_kb = gbase[len(gbase) // 2] if gbase else None
    gpu_rise, totals = [], []
    for r in run:
        pss = int(r[7]) if r[7].isdigit() else None
        gl = int(r[6]) if r[6].isdigit() else 0
        gs = int(r[8]) if len(r) > 8 and r[8].isdigit() else None
        rise = max(0, gs - gbase_kb) if (gs is not None and gbase_kb is not None) else None
        if rise is not None:
            gpu_rise.append(rise)
        if pss is not None:
            totals.append(pss + (max(0, rise - gl) if rise is not None else 0))
    return {
        "exit": code, "seconds": round(time.time() - t0, 1), "samples": len(run),
        # columns (kB): uptime, MemAvailable, pid, VmHWM, VmRSS, VmSwap, GL mtrack, TOTAL PSS (dumpsys meminfo)
        "peak_rss_mib": mib(max(num(3) + num(4), default=None)),  # CPU-side resident memory (VmHWM high-water mark)
        "peak_swap_mib": mib(max(num(5), default=None)),
        "peak_gpu_mib": mib(max(gpu_rise, default=None)) if gpu_rise else mib(max(num(6), default=None)),  # GPU buffers of the process
        "peak_total_mib": mib(max(totals, default=None)),  # PSS + GPU memory not in PSS: the process's memory footprint
        "gpu_source": "GPUTotalUsed rise over baseline (/proc/meminfo)" if gpu_rise else "GL mtrack (dumpsys meminfo)",
        "gpu_baseline_mib": mib(gbase_kb),
        "mem_available_before_mib": mib(max(base[-20:], default=None)) if base else None,
        "mem_available_drop_mib": (mib(max(base[-20:]) - min(int(r[1]) for r in run)) if base and run else None),
        "stderr_tail": stderr[-600:] if code not in (0, None) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial")
    ap.add_argument("--ref", choices=["pinned", "head"], default="pinned")
    ap.add_argument("--pool-file")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--results-dir", required=True, help="the benchmark results folder of this ref")
    ap.add_argument("--llama-threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-battery", type=int, default=5, help="stop below this battery level (%%)")
    ap.add_argument("--backends", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    ap.add_argument("--screen", choices=["off", "on", "panel"], default="off",
                    help="as in run_paper_table5.py; panel for phones that cap the CPU while asleep (OnePlus 8 Pro)")
    a = ap.parse_args()
    out = Path(a.results_dir)
    (out / "memory").mkdir(parents=True, exist_ok=True)
    models = a.models
    if a.pool_file:
        models = [n for n, (_, ref) in poolmod.load_pool(a.pool_file).items() if ref == a.ref]
    adb = Adb(a.serial)
    controls = DeviceControls(adb, log, screen=a.screen)
    controls.apply()  # airplane mode, background apps killed, screen off + wake lock: same memory state as the benchmarks
    try:
        manifest = rpt.load_manifest()
        cfgs = []
        for e in manifest:
            if models and e["name"] not in models or e.get("ref", "pinned") != a.ref:
                continue
            for b in a.backends:
                cid = f"{e['name']}__{e['quant']}__{e['framework']}__{b}" + ("" if a.ref == "pinned" else f"__{a.ref}")
                cfgs.append({"id": cid.replace("/", "-"), "model": e["name"], "quant": e["quant"],
                             "framework": e["framework"], "backend": b, "entry": e})
        cfgs.sort(key=lambda c: sum(c["entry"]["files"].values()))
        for i, cfg in enumerate(cfgs, 1):
            f = out / "memory" / f"{cfg['id']}.json"
            if f.exists() and not a.force and json.loads(f.read_text()).get("exit") in (0, "skipped"):
                continue
            rec = {"config": {k: v for k, v in cfg.items() if k != "entry"},
                   "storage_bytes": sum(cfg["entry"]["files"].values()), "measured": datetime.now().isoformat()}
            bench = out / "configs" / f"{cfg['id']}.json"
            err = json.loads(bench.read_text()).get("error", "") if bench.exists() else ""
            if str(err).startswith("does not fit in RAM:"):
                rec.update(exit="skipped", reason=err)
            else:
                level = rpt.bench_common.device_state(adb).get("battery_level")
                if level is not None and level < a.min_battery:
                    log(f"[BATTERY] {level}% < --min-battery {a.min_battery}%: stopping; charge the phone, unplug it "
                        "and rerun the same command (measured configurations are kept)")
                    break
                log(f"[{i}/{len(cfgs)}] {cfg['id']} ({rec['storage_bytes'] / 2**30:.2f} GiB)")
                path = rpt.ensure_model_on_device(adb, cfg["entry"])
                adb.sh("am kill-all")
                rec.update(run_one(adb, cfg, path, a.ref, a.llama_threads, a.timeout))
                log(f"   exit {rec['exit']}: peak total {rec['peak_total_mib']} MiB (GPU {rec['peak_gpu_mib']}, "
                    f"RSS {rec['peak_rss_mib']}), "
                    f"MemAvailable drop {rec['mem_available_drop_mib']} MiB")
                time.sleep(10)  # let the kernel return the memory before the next baseline
            f.write_text(json.dumps(rec, indent=1))
    finally:
        controls.restore()


if __name__ == "__main__":
    main()
