#!/usr/bin/env python3
"""
Fig. 9a/9b on-phone validation: run a few hundred consecutive held-out requests of a factory-built model on its
target phone, under the paper's Section 3 profiling conditions, and compare the measured prefill/decode speed, peak
memory and energy per request with the requirement.

  # once per phone, over USB: model + request set + binaries onto the phone (no measurement)
  python fig9_validation/run_fig9.py --scenario m07 --serial <usb serial> --prepare
  # energy needs the phone on battery: switch to wireless adb, unplug, then
  python energy_probe.py wireless --serial <usb serial>
  python fig9_validation/run_fig9.py --scenario m07 --serial <ip:port>
  # recompute the summary from a finished results folder (no phone)
  python fig9_validation/run_fig9.py --summarize results/fig9_m07_oneplus8pro_<stamp>

Scenarios: m07 (Fig. 9a, BC5CDR NER, OnePlus 8 Pro, Qwen3.5-0.8B 4-bit MNN CPU) and h09 (Fig. 9b, DialogSum, Galaxy
S23, Qwen3-0.6B 8-bit MNN CPU). Run from paper_table5/ with the harness environment (README "One-time setup").

What one batch is (repeated --repeats times, each from a cool start):
  1. cool-down gate (battery <= the profile's limit, 28 C, or the S23's plateau rule; CPU caps back at baseline),
     airplane mode (Wi-Fi kept for adb), Do Not Disturb, background apps killed, screen off (OnePlus 8 Pro: awake
     with only the panel off), on battery: the paper's Section 3 conditions, through the harness's own controls;
  2. --idle-seconds of idle power, right before the batch;
  3. pb_requests loads the model once (4 CPU threads, precision low, memory low: the published config.json and the
     paper's protocol), runs the warm-up requests, then the measured requests back to back;
  4. --post-idle-seconds of idle power after the batch (a second idle reference).
The phone-side energy sampler (energy.py's EnergyRecorder: battery gauge at 100 ms, Perfetto, battery service, or
chip counters where readable) runs for the whole session.

Per request (pb_requests PB_MARK lines, CLOCK_BOOTTIME): prefill tokens/s = prompt tokens / time to first token
(embedding + prefill forward); decode tokens/s = decode forward passes / (end - first token); peak memory = the
process's Pss (+ GPU via dumpsys for OpenCL) and VmHWM. Energy per request = the batch's whole-phone energy net of
idle / number of requests (the battery gauge cannot resolve one ~1 s request; see README).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PT5 = HERE.parent
sys.path.insert(0, str(PT5))
sys.path.insert(0, str(PT5.parent))
import bench_common  # noqa: E402
import energy_probe  # noqa: E402
from build_pb_requests import FIG9_DEV, MNN_SHA, dev_bin_dir  # noqa: E402
from common import (WAKE_KICK_CMD, DeviceControls, boottime_s, device_info, match_profile,  # noqa: E402
                    portable_record, recorded_serial)
from energy import EnergyRecorder, is_chip_method, probe_recommendation  # noqa: E402

POLL_S = 15
TIMELINE_EVERY_S = 30
MEMINFO_EVERY_S = 60

# Requirement numbers: config/filtered_pools/<phone>_<scenario>.json "limits" (what the factory filtered with) and
# hardware/docs/SCENARIOS.md (request shape, latency limits, the profile's values for the chosen configuration).
SCENARIOS = {
    "m07": {
        "title": "Fig. 9a / M07: BC5CDR NER on a OnePlus 8 Pro, 'Paper-alert abstract index' (overnight job)",
        "profile": "oneplus8pro",
        "run": "slm-ner-bc5cdr-m07-op8pro-l40s-42809109",
        "model": "Qwen3.5-0.8B, LoRA-merged, MNN 4-bit HQQ block 32, 8-bit lm_head (span F1 0.811)",
        "hf_repo": "evanlyhf/MobileAIFactory-BC5CDR-NER-Qwen3.5-0.8B-MNN-4bit-OnePlus8Pro",
        "cluster_artifact": "artifacts/mnn/Qwen_Qwen3.5-0.8B/1d0d381b8a74/model-mnn-q4",
        "device_name": "m07-qwen3.5-0.8b-mnn-q4",
        "backend": "cpu", "threads": 4,
        "limits": {"max_peak_memory_mb": 5800.0, "min_prefill_tps": 2.96, "min_decode_tps": 5.0, "max_energy_j": 15.66},
        "latency": {"max_ttft_s": 10.8, "max_e2e_s": 21.6},
        "requirement_request": {"prompt_tokens": 32, "answer_tokens": 45, "cached_prefix_tokens": 40},
        "profile_values": {"peak_memory_mb": 550, "prefill_tps": 167, "decode_tps": 45.9, "energy_j": 7.04,
                           "source": "SCENARIOS.md M07, Qwen3.5-0.8B Q4 MNN CPU row (MNN default 4-bit export, "
                                     "256-token prompt + 256-token answer, energy for 32 + 45 tokens)"},
        "files": {  # model-mnn-q4.validation.json fingerprint (config.json differs: the HF copy sets greedy/no template)
            "llm.mnn": (2600496, "13286f705b4f97cd5670e25279cb213c5bb7fea2f74c9c2932bad1efdb3ac0f7"),
            "llm.mnn.weight": (597498250, "882d7631321d37d59b1a83d0cac07a999fbb4d1a71b82badc0a157350352bf13"),
            "llm_config.json": (9114, "fab4dbcbb8b50f6e1cafdee342fc62d453dae70ff86468a20e941251a0500114"),
            "tokenizer.mtok": (7342245, "52fc876ae30a44898ad8da9c04d1118d64bf70eb2bbdeebdb45e8ab167599fcc"),
            "visual.mnn": (250360, "db0378863f3a4c643c1b8b37e3be4f4a3d84fc73c3e6148b0467fced68046ef3"),
            "visual.mnn.weight": (63101558, "936f5264fad74974daca3068ccab43768d20c6ee48e3007de94ecc7f56b45f16"),
        },
    },
    "h09": {
        "title": "Fig. 9b / H09: DialogSum on a Galaxy S23, 'Dialer call summary' (feature inside an app)",
        "profile": "galaxy_s23",
        "run": "slm-dialogsum-h09-s23-l40s-43019548",
        "model": "Qwen3-0.6B, LoRA-merged, MNN 8-bit block 64 (ROUGE-L 0.469)",
        "hf_repo": "evanlyhf/MobileAIFactory-DialogSum-Qwen3-0.6B-MNN-8bit-GalaxyS23",
        "cluster_artifact": "artifacts/mnn/Qwen_Qwen3-0.6B/50602192f8e7/model-mnn-q8",
        "device_name": "h09-qwen3-0.6b-mnn-q8",
        "backend": "cpu", "threads": 4,
        "limits": {"max_peak_memory_mb": 1024.0, "min_prefill_tps": 350.0, "min_decode_tps": 5.0, "max_energy_j": 90.0},
        "latency": {"max_ttft_s": 0.5, "max_e2e_s": None},
        "requirement_request": {"prompt_tokens": 175, "answer_tokens": 40, "cached_prefix_tokens": 30},
        "profile_values": {"peak_memory_mb": 856, "prefill_tps": 368, "decode_tps": 49.8, "energy_j": 10.4,
                           "source": "SCENARIOS.md H09 / galaxys23_h09.json, Qwen3-0.6B Q8 MNN CPU (MNN default 8-bit "
                                     "export, 256 + 256 tokens, energy for 175 + 40 tokens)"},
        "files": {
            "llm.mnn": (1282520, "394d7d570fe6efeab622ac31db5009d00945abb3c74966cdbca0f4c7ea38966a"),
            "llm.mnn.weight": (633751970, "397fb8f02ffa935ce253563cf59a516136d4de8efbb2721474c7b9b64e895142"),
            "llm_config.json": (4865, "42535229850dc306bf71c92167e1342142ff23cba6c60863aca4ee26423d22e0"),
            "tokenizer.mtok": (4111427, "fac7fb81ae32cf115dec36921b2d5a23bab72ca5d9d2998805b2452ddbc6f1ab"),
        },
    },
}


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def pct(xs, q):
    s = sorted(x for x in xs if x is not None)
    if not s:
        return None
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def rnd(x, n=3):
    return None if x is None else round(x, n)


def describe(xs) -> dict:
    v = [x for x in xs if x is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "min": rnd(min(v)), "p05": rnd(pct(v, 5)), "median": rnd(pct(v, 50)),
            "mean": rnd(statistics.mean(v)), "p95": rnd(pct(v, 95)), "max": rnd(max(v))}


# ---------------------------------------------------------------------------
# Model files and requests
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_model_dir(sc: dict, given: str | None) -> Path:
    if given:
        d = Path(given)
    else:
        d = HERE / "models" / sc["device_name"]
        if not (d / "llm.mnn.weight").is_file():
            log(f"downloading {sc['hf_repo']} -> {d}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=sc["hf_repo"], local_dir=str(d))
    bad = []
    for name, (size, digest) in sc["files"].items():
        f = d / name
        if not f.is_file():
            bad.append(f"{name} missing")
        elif f.stat().st_size != size:
            bad.append(f"{name} is {f.stat().st_size} bytes, expected {size}")
        elif sha256(f) != digest:
            bad.append(f"{name} sha256 differs from the factory's validation sidecar")
    if not (d / "config.json").is_file():
        bad.append("config.json missing")
    if bad:
        sys.exit(f"model files in {d} are not the published Fig. 9 model: " + "; ".join(bad))
    log(f"model files verified against the factory's sidecar: {d}")
    return d


def push_model(adb, local: Path, remote: str, names: list[str]):
    """Push the model's MNN files (skipping those already on the phone with the same size)."""
    have = {}
    for line in adb.sh(f"cd {remote} 2>/dev/null && stat -c '%n %s' * 2>/dev/null").splitlines():
        n, _, s = line.rpartition(" ")
        if s.isdigit():
            have[n] = int(s)
    todo = [local / n for n in names if have.get(n) != (local / n).stat().st_size]
    if not todo:
        log(f"model already on the phone: {remote}")
        return
    adb.sh(f"mkdir -p {remote}")
    for p in todo:
        log(f"pushing {p.name} ({p.stat().st_size / 1e6:.0f} MB) -> {remote}")
        r = adb.run(["push", str(p), f"{remote}/{p.name}"], timeout=7200)
        if r.returncode != 0:
            sys.exit(f"adb push {p} failed: {r.stderr.strip()[-300:]}")


def load_requests(name: str, n: int, warmup: int, mode: str) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in (HERE / "requests" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = json.loads((HERE / "requests" / f"{name}.meta.json").read_text(encoding="utf-8"))
    warm = [r for r in rows if r["warmup"]][:warmup]
    meas = [r for r in rows if not r["warmup"]][:n]
    if len(meas) < n:
        sys.exit(f"{name}: only {len(meas)} measured requests in the file (asked for {n})")
    key = "max_new_tokens_reference" if mode == "reference" else "max_new_tokens_eos"
    for r in warm + meas:
        r["max_new"] = r[key]
    return warm + meas, meta


def device_request_file(reqs: list[dict]) -> str:
    lines = ["# seq warm max_new n ids..."]
    for r in reqs:
        lines.append(f"{r['seq']} {int(r['warmup'])} {r['max_new']} {len(r['prompt_ids'])} "
                     + " ".join(map(str, r["prompt_ids"])))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Session on the phone
# ---------------------------------------------------------------------------

def read_meminfo(adb, exe: str) -> dict | None:
    out = adb.sh(f"pid=$(pidof {exe}); [ -n \"$pid\" ] && dumpsys meminfo $pid | grep -E 'TOTAL PSS|GL mtrack'",
                 timeout=60)
    pss = re.search(r"TOTAL PSS:\s+(\d+)", out)
    gl = re.search(r"GL mtrack\s+(\d+)", out)
    if not pss:
        return None
    return {"total_pss_kb": int(pss.group(1)), "gl_mtrack_kb": int(gl.group(1)) if gl else None}


def run_batch(adb, controls, sess: dict, a, tag: str, out: Path, kick: bool) -> dict:
    """One pb_requests invocation (load + warm-up + measured requests), with timeline and memory sampling."""
    work = f"{FIG9_DEV}/run/{tag}"
    bin_dir = dev_bin_dir(a.mnn_sha)
    cfg = json.dumps({"backend_type": "opencl" if a.backend == "gpu" else "cpu", "thread_num": a.threads,
                      "precision": "low", "memory": "low"}, separators=(",", ":"))
    cmd = (f"./pb_requests {sess['model_remote']}/config.json {sess['requests_remote']} --set '{cfg}' "
           f"--ignore-eos {1 if a.answer_length == 'reference' else 0} --out {work}/outputs.jsonl "
           f"--tune {1 if a.backend == 'gpu' else 0} --gap-ms {a.gap_ms}")
    script = ((WAKE_KICK_CMD if kick else "") + f"cd {bin_dir} && LD_LIBRARY_PATH=. {cmd} > {work}/stdout.txt "
              f"2> {work}/stderr.txt; echo $? > {work}/exit\n")
    adb.sh(f"rm -rf {work}; mkdir -p {work} {bin_dir}/tmp")
    energy_probe.push_text(adb, script, f"{work}/run.sh")
    adb.sh(f"setsid sh {work}/run.sh </dev/null >/dev/null 2>&1 &")
    t_start, exit_code = time.time(), None
    timeline, meminfo = [], []
    last_tl = last_mem = 0.0
    n_total = sess["n_requests"]
    while time.time() - t_start < a.timeout:
        time.sleep(POLL_S)
        ex = adb.sh(f"cat {work}/exit 2>/dev/null", timeout=30).strip()
        if ex == "" and controls.wireless and adb.sh("echo ok", timeout=15).strip() != "ok":
            controls.reconnect(900)
            continue
        if ex.lstrip("-").isdigit():
            exit_code = int(ex)
            break
        now = time.time()
        if now - last_tl >= TIMELINE_EVERY_S:
            last_tl = now
            st = bench_common.device_state(adb)
            done = adb.sh(f"grep -c '^PB_MARK req' {work}/stderr.txt 2>/dev/null").strip()
            timeline.append({"boottime_s": boottime_s(adb), "battery_temp_c": st["battery_temp_c"],
                             "battery_level": st["battery_level"], "externally_powered": st["externally_powered"],
                             "cpu_capped": st["cpu_capped"], "cpu_caps_khz": st["cpu_caps_khz"],
                             "thermal_status": st["thermal_status"], "screen": st["screen"],
                             "requests_done": int(done) if done.isdigit() else None})
            log(f"   {tag}: {done or 0}/{n_total} requests, battery {st['battery_temp_c']} C {st['battery_level']}%"
                + (", CPU capped" if st["cpu_capped"] else "") + (", POWERED" if st["externally_powered"] else ""))
        if now - last_mem >= MEMINFO_EVERY_S:
            last_mem = now
            m = read_meminfo(adb, "pb_requests")
            if m:
                meminfo.append({"t": now, **m})
    else:
        from run_paper_table5 import stop_benchmark
        stop_benchmark(adb, work, "pb_requests")
        log(f"   {tag}: timed out after {a.timeout}s")
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("stdout.txt", "stderr.txt", "outputs.jsonl"):
        adb.run(["pull", f"{work}/{name}", str(raw / f"{tag}.{name}")], timeout=300)
    (raw / f"{tag}.cmd.txt").write_text(script, encoding="utf-8")
    return {"tag": tag, "exit": exit_code, "seconds": round(time.time() - t_start, 1), "timeline": timeline,
            "meminfo": meminfo}


def session(a):
    from devenv import Adb
    sc = SCENARIOS[a.scenario]
    adb = Adb(a.serial)
    info = device_info(adb)
    profile = match_profile(info)
    if profile["id"] != sc["profile"] and not a.any_phone:
        sys.exit(f"this is a {info['manufacturer']} {info['model']} (profile {profile['id']}); scenario {a.scenario} "
                 f"is for {sc['profile']} (--any-phone to run anyway)")
    model_remote = f"{FIG9_DEV}/models/{sc['device_name']}"
    reqs, req_meta = load_requests(a.scenario, a.n, a.warmup, a.answer_length)
    req_name = f"{a.scenario}_{a.answer_length}_{a.n}.txt"
    req_remote = f"{FIG9_DEV}/requests/{req_name}"
    bin_dir = dev_bin_dir(a.mnn_sha)

    if a.prepare or adb.sh(f"test -f {model_remote}/llm.mnn.weight && echo ok").strip() != "ok":
        push_model(adb, local_model_dir(sc, a.model_dir), model_remote, [*sc["files"], "config.json"])
    local_req = HERE / "requests" / req_name
    local_req.write_text(device_request_file(reqs), encoding="utf-8")
    adb.sh(f"mkdir -p {FIG9_DEV}/requests")
    adb.run(["push", str(local_req), req_remote], timeout=300)
    if adb.sh(f"test -x {bin_dir}/pb_requests && echo ok").strip() != "ok":
        sys.exit(f"{bin_dir}/pb_requests missing on the phone: python fig9_validation/build_pb_requests.py "
                 f"--push --serial {adb.serial}" + ("" if a.mnn_sha == MNN_SHA else f" --mnn-ref {a.mnn_sha}"))
    if a.prepare:
        log(f"prepared: model {model_remote}, requests {req_remote}, binary {bin_dir}/pb_requests. "
            "Now switch to wireless adb, unplug, and run without --prepare.")
        return None

    out = Path(a.results_dir) if a.results_dir else (
        PT5 / "results" / f"fig9_{a.scenario}_{profile['id']}_{a.answer_length}_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    screen = a.screen or profile.get("screen") or "auto"
    max_temp = a.max_temp if a.max_temp is not None else profile.get("max_temp_c", 28.0)
    plateau = a.gate_plateau if a.gate_plateau is not None else profile.get("gate_plateau_s", 0)
    adb.sh("pkill pb_requests; pkill llm_bench; pkill llama-bench")
    st0 = bench_common.device_state(adb)
    disc = energy_probe.discover(adb)
    chip = disc["powercap"]["usable"]
    if st0["externally_powered"] and not chip and not a.allow_powered:
        sys.exit("the phone is on external power and has no chip energy counters: energy would follow the charger. "
                 "Switch to wireless adb (python energy_probe.py wireless --serial <usb serial>), unplug, rerun with "
                 "the ip:port serial (or --allow-powered to measure speed and memory only).")
    if st0["battery_level"] is not None and st0["battery_level"] < a.min_battery:
        sys.exit(f"battery at {st0['battery_level']}% < --min-battery {a.min_battery}%: charge it first")
    binfo = adb.sh(f"cat {bin_dir}/build_info.json 2>/dev/null")
    sess = {
        "scenario": a.scenario, "scenario_def": {k: v for k, v in sc.items() if k != "files"},
        "started": datetime.now().isoformat(), "device": info, "profile": profile,
        "params": {k: recorded_serial(v) if k == "serial" else portable_record(v) for k, v in vars(a).items()},
        "model_remote": model_remote, "requests_remote": req_remote, "request_meta": req_meta,
        "n_requests": len(reqs), "answer_length": a.answer_length,
        "requests": [{k: r[k] for k in ("seq", "eval_index", "warmup", "prompt_tokens", "answer_tokens", "max_new")}
                     for r in reqs],
        "runtime": {"engine": "MNN", "mnn_commit": a.mnn_sha, "backend": a.backend, "threads": a.threads,
                    "precision": "low", "memory": "low", "sampler": "greedy", "use_template": False,
                    "reuse_kv": False, "ignore_eos": a.answer_length == "reference",
                    "build_info": json.loads(binfo) if binfo.strip().startswith("{") else None},
        "conditions": {"screen": screen, "max_temp_c": max_temp, "gate_plateau_s": plateau,
                       "wireless_adb": None, "state_at_start": st0},
        "energy_discovery": {k: disc[k] for k in ("root", "powercap", "powerstats", "battery_dir", "battery_files",
                                                   "battery_service")},
        "batches": [],
    }
    write_session(out, sess)
    controls = DeviceControls(adb, log, screen=screen)
    recorder = EnergyRecorder(adb, out / "energy", disc, use_perfetto=not a.no_perfetto, log=log)
    kick = bool(profile.get("wake_kick"))
    try:
        if not a.no_controls:
            controls.apply()
        sess["conditions"].update({"wireless_adb": controls.wireless, "radio_mode": controls.radio_mode,
                                   "screen_on": controls.screen_on,
                                   "panel_off": bool(controls.panel) and not controls.panel.keep_only,
                                   "wake_lock": bool(controls.waker)})
        kick = kick and not controls.screen_on
        gate = bench_common.ReadinessGate(adb, max_temp, timeout_s=a.gate_timeout, log=log, plateau_s=plateau,
                                          plateau_max_c=profile.get("gate_plateau_max_c"))
        gate.set_baseline(bench_common.device_state(adb))
        recorder.start()
        for rep in range(1, a.repeats + 1):
            tag = f"rep{rep}"
            lvl = bench_common.device_state(adb).get("battery_level")
            if lvl is not None and lvl < a.min_battery:
                log(f"[BATTERY] {lvl}% < {a.min_battery}%: stopping before {tag}")
                break
            log(f"[{tag}] cool-down gate")
            g = gate.wait()
            controls.ensure_screen()
            busy = energy_probe.idle_cpu_check(adb, 5.0)
            if busy is not None and busy > 0.15:
                log(f"[WARN] the phone is {busy:.0%} busy with nothing running")
            log(f"[{tag}] idle power ({a.idle_seconds}s)")
            t0 = boottime_s(adb)
            time.sleep(a.idle_seconds)
            idle_pre = [t0 + min(2.0, a.idle_seconds / 4), boottime_s(adb)]
            before = bench_common.device_state(adb)
            log(f"[{tag}] {len(reqs)} requests ({a.warmup} warm-up), {a.answer_length} answer lengths, "
                f"{a.backend.upper()} x{a.threads}, battery {before['battery_temp_c']} C")
            res = run_batch(adb, controls, sess, a, tag, out, kick)
            after = bench_common.device_state(adb)
            log(f"[{tag}] exit {res['exit']} after {res['seconds']}s; post-idle ({a.post_idle_seconds}s)")
            t1 = boottime_s(adb)
            time.sleep(a.post_idle_seconds)
            idle_post = [t1 + min(5.0, a.post_idle_seconds / 4), boottime_s(adb)]
            res.update({"gate": {"passed": g["passed"], "passed_by": g.get("passed_by"), "waited_s": g["waited_s"],
                                 "issues": g.get("issues"), "battery_temp_c": g["state"]["battery_temp_c"]},
                        "idle_pre": idle_pre, "idle_post": idle_post, "idle_cpu_busy": busy,
                        "state_before": before, "state_after": after})
            sess["batches"].append(res)
            write_session(out, sess)
            if res["exit"] != 0:
                log(f"[{tag}] pb_requests failed (exit {res['exit']}); see {out / 'raw'}")
                break
            if rep < a.repeats:
                time.sleep(a.rest)
    finally:
        try:
            recorder.stop()
        finally:
            if not a.no_controls:
                controls.restore()
            sess["finished"] = datetime.now().isoformat()
            write_session(out, sess)
    return out


def write_session(out: Path, sess: dict):
    (out / "session.json").write_text(json.dumps(portable_record(sess), indent=1, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Analysis (online and --summarize)
# ---------------------------------------------------------------------------

def parse_lines(text: str, tag: str) -> list[dict]:
    out = []
    for m in re.finditer(rf"^{tag} (.*)$", text, re.M):
        d = {}
        for kv in m.group(1).split():
            k, _, v = kv.partition("=")
            d[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
        out.append(d)
    return out


def summarize(out: Path) -> dict:
    sess = json.loads((out / "session.json").read_text(encoding="utf-8"))
    sc = SCENARIOS[sess["scenario"]]
    lim, lat = sc["limits"], sc["latency"]
    reqmeta = {r["seq"]: r for r in sess["requests"]}
    texts = {}
    recorder = EnergyRecorder(None, out / "energy", sess["energy_discovery"], log=log)
    funcs = recorder.load() if (out / "energy").exists() else {}
    preferred = sess["profile"].get("energy_method") or probe_recommendation(sess["device"].get("model"))
    headline = recorder.headline(preferred) if funcs else None
    rows, batches = [], []
    for b in sess["batches"]:
        tag = b["tag"]
        err_f = out / "raw" / f"{tag}.stderr.txt"
        if not err_f.exists():
            continue
        err = err_f.read_text(encoding="utf-8", errors="replace")
        marks = parse_lines(err, "PB_MARK req")
        load = (parse_lines(err, "PB_LOAD") or [{}])[0]
        done = (parse_lines(err, "PB_DONE") or [{}])[0]
        outs = out / "raw" / f"{tag}.outputs.jsonl"
        if outs.exists():
            for line in outs.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    o = json.loads(line)
                    texts[(tag, o["seq"])] = o.get("text")
                except ValueError:
                    pass
        powered = any((b.get(k) or {}).get("externally_powered") for k in ("state_before", "state_after")) or any(
            t.get("externally_powered") for t in b.get("timeline", []))
        meas = []
        for m in marks:
            a_, mid, e_ = m["begin"] / 1e9, m["mid"] / 1e9, m["end"] / 1e9
            pre_s, dec_s = mid - a_, e_ - mid
            r = {"batch": tag, "seq": m["seq"], "warmup": bool(m["warm"]),
                 "eval_index": reqmeta.get(m["seq"], {}).get("eval_index"),
                 "prompt_tokens": m["prompt"], "answer_tokens_reference": reqmeta.get(m["seq"], {}).get("answer_tokens"),
                 "max_new_tokens": m["max_new"], "generated_tokens": m["gen"], "decode_forwards": m["fwd"],
                 "stopped_at_eos": m["status"] == 1,
                 "ttft_s": rnd(pre_s, 4), "decode_s": rnd(dec_s, 4), "e2e_s": rnd(e_ - a_, 4),
                 "prefill_tps": rnd(m["prompt"] / pre_s if pre_s > 0 else None, 2),
                 "decode_tps": rnd(m["fwd"] / dec_s if dec_s > 0 and m["fwd"] > 0 else None, 2),
                 "prefill_tps_mnn": rnd(m["prompt"] / (m["prefill_us"] / 1e6) if m["prefill_us"] > 0 else None, 2),
                 "decode_tps_mnn": rnd(m["fwd"] / (m["decode_us"] / 1e6) if m["decode_us"] > 0 and m["fwd"] else None, 2),
                 "suspend_ratio": rnd((e_ - a_) / ((m["prefill_us"] + m["decode_us"]) / 1e6)
                                      if m["prefill_us"] + m["decode_us"] > 0 else None, 3),
                 "pss_mb": rnd(m["pss_kb"] * 1024 / 1e6, 1) if m["pss_kb"] >= 0 else None,
                 "rss_mb": rnd(m["rss_kb"] * 1024 / 1e6, 1) if m["rss_kb"] >= 0 else None,
                 "hwm_mb": rnd(m["hwm_kb"] * 1024 / 1e6, 1) if m["hwm_kb"] >= 0 else None,
                 "swap_mb": rnd(m["swap_kb"] * 1024 / 1e6, 1) if m["swap_kb"] >= 0 else None,
                 "begin_boottime_s": round(a_, 3), "mid_boottime_s": round(mid, 3), "end_boottime_s": round(e_, 3)}
            temps = [t for t in b.get("timeline", []) if t.get("battery_temp_c") is not None]
            if temps:
                near = min(temps, key=lambda t: abs(t["boottime_s"] - a_))
                r["battery_temp_c"] = near["battery_temp_c"]
                r["cpu_capped"] = near["cpu_capped"]
            r["output_text"] = texts.get((tag, m["seq"]))
            rows.append(r)
            if not r["warmup"]:
                meas.append(r)
        # Energy: the whole measured batch, net of idle, divided by the number of requests
        energy = {}
        if funcs and meas:
            A, B = meas[0]["begin_boottime_s"], meas[-1]["end_boottime_s"]
            win = [{"phase": "batch", "a": A, "b": B, "tokens": len(meas)}]
            for ref, idle in (("pre_idle", [b["idle_pre"]]), ("post_idle", [b["idle_post"]]),
                              ("both_idle", [b["idle_pre"], b["idle_post"]])):
                res = recorder.phase_energy(win, idle, powered)
                for method, ph in res.items():
                    e = ph.get("batch") or {}
                    chip = is_chip_method(method)
                    per = e.get("uj_per_token") if chip else e.get("net_uj_per_token")
                    energy.setdefault(method, {})[ref] = {
                        "j_per_request": rnd(per / 1e6 if per is not None else None, 3),
                        "gross_j_per_request": rnd(e["uj_per_token"] / 1e6 if e.get("uj_per_token") is not None else None, 3),
                        "avg_power_mw": e.get("avg_power_mw"), "idle_power_mw": e.get("idle_power_mw"),
                        "batch_seconds": e.get("seconds"), "valid": e.get("valid"), "reason": e.get("reason"),
                        "kind": "chip (gross)" if chip else "whole device (net of idle)"}
            if headline in funcs:  # indicative per-request split with the headline gauge
                fn, series, _ = funcs[headline]
                idle_mw = energy.get(headline, {}).get("pre_idle", {}).get("idle_power_mw")
                load = fn(A, B)
                if idle_mw is not None and load is not None:
                    raw_idle = energy_probe_idle(fn, b["idle_pre"])
                    sign = -1.0 if (raw_idle is not None and load < raw_idle) else 1.0
                    chip = is_chip_method(headline)
                    for r in meas:
                        ra, rm, rb = r["begin_boottime_s"], r["mid_boottime_s"], r["end_boottime_s"]
                        for key, (x, y) in (("energy_j_indicative", (ra, rb)), ("prefill_energy_j_indicative", (ra, rm)),
                                            ("decode_energy_j_indicative", (rm, rb))):
                            p = fn(x, y) if y > x else None
                            r[key] = rnd(((sign * p) - (0 if chip else idle_mw)) * (y - x) / 1e3, 3) if p is not None else None
                        r["gauge_updates"] = round(energy_probe.updates_per_s(series, ra, rb) * (rb - ra), 1) if series else None
        mem_dumpsys = [m["total_pss_kb"] * 1024 / 1e6 for m in b.get("meminfo", [])]
        tl = b.get("timeline", [])
        batches.append({
            "tag": tag, "exit": b.get("exit"), "requests": len(meas), "warmup": len(marks) - len(meas),
            "load_s": rnd((load["end"] - load["begin"]) / 1e9, 2) if load.get("begin") else None,
            "batch_s": rnd(meas[-1]["end_boottime_s"] - meas[0]["begin_boottime_s"], 1) if meas else None,
            "gate": b.get("gate"), "externally_powered": powered,
            "battery_temp_c": {"start": (b.get("state_before") or {}).get("battery_temp_c"),
                               "end": (b.get("state_after") or {}).get("battery_temp_c"),
                               "max": max((t["battery_temp_c"] for t in tl if t.get("battery_temp_c") is not None), default=None)},
            "battery_level": {"start": (b.get("state_before") or {}).get("battery_level"),
                              "end": (b.get("state_after") or {}).get("battery_level")},
            "cpu_capped_samples": sum(1 for t in tl if t.get("cpu_capped")), "timeline_samples": len(tl),
            "max_thermal_status": max((t["thermal_status"] for t in tl if t.get("thermal_status") is not None), default=None),
            "peak_dumpsys_total_pss_mb": rnd(max(mem_dumpsys), 1) if mem_dumpsys else None,
            "hwm_at_end_mb": rnd(done["hwm_kb"] * 1024 / 1e6, 1) if done.get("hwm_kb") else None,
            "pss_after_load_mb": rnd(load["pss_kb"] * 1024 / 1e6, 1) if load.get("pss_kb") else None,
            "energy": energy,
            "suspended_requests": sum(1 for r in meas if (r["suspend_ratio"] or 0) > 1.3),
        })

    meas = [r for r in rows if not r["warmup"]]
    with open(out / "requests.csv", "w", newline="", encoding="utf-8") as f:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    pre, dec = [r["prefill_tps"] for r in meas], [r["decode_tps"] for r in meas]
    ttft, e2e = [r["ttft_s"] for r in meas], [r["e2e_s"] for r in meas]
    peak_pss = max([r["pss_mb"] for r in rows if r["pss_mb"] is not None]
                   + [b["peak_dumpsys_total_pss_mb"] for b in batches if b["peak_dumpsys_total_pss_mb"]], default=None)
    peak_hwm = max([r["hwm_mb"] for r in rows if r["hwm_mb"] is not None]
                   + [b["hwm_at_end_mb"] for b in batches if b["hwm_at_end_mb"]], default=None)
    # pooled energy: total net energy over all batches / all requests, headline method, pre-batch idle
    e_tot, n_tot, e_valid, reasons = 0.0, 0, True, []
    for b in batches:
        e = (b["energy"].get(headline) or {}).get("pre_idle") if headline else None
        if not e or e["j_per_request"] is None:
            e_valid = False
            reasons.append(f"{b['tag']}: no energy for {headline}")
            continue
        e_tot += e["j_per_request"] * b["requests"]
        n_tot += b["requests"]
        if not e["valid"]:
            e_valid = False
            reasons.append(f"{b['tag']}: {e['reason']}")
    e_req = e_tot / n_tot if n_tot else None
    e_alt = {}
    for ref in ("post_idle", "both_idle"):
        vals = [((b["energy"].get(headline) or {}).get(ref) or {}).get("j_per_request") for b in batches] if headline else []
        if vals and all(v is not None for v in vals):
            e_alt[ref] = rnd(sum(v * b["requests"] for v, b in zip(vals, batches)) / max(1, sum(b["requests"] for b in batches)), 3)
    ind = [r.get("energy_j_indicative") for r in meas]
    # indicative per-phase split: gauge power integrated over all prefill (decode) windows / their tokens
    pe = [r.get("prefill_energy_j_indicative") for r in meas]
    de = [r.get("decode_energy_j_indicative") for r in meas]
    split = {}
    if meas and all(x is not None for x in pe + de):
        split = {"prefill_mj_per_token": rnd(1e3 * sum(pe) / max(1, sum(r["prompt_tokens"] for r in meas)), 3),
                 "decode_mj_per_token": rnd(1e3 * sum(de) / max(1, sum(r["decode_forwards"] for r in meas)), 3)}

    def check(name, stat, value, op, limit, note=""):
        if value is None or limit is None:
            ok = None
        else:
            ok = value <= limit if op == "<=" else value >= limit
        return {"metric": name, "statistic": stat, "measured": rnd(value, 3), "requirement": f"{op} {limit:g}"
                if limit is not None else None, "pass": ok, "note": note}

    s_pre, s_dec = describe(pre), describe(dec)
    verdict = [
        check("peak memory (MB)", "max Pss over the run (dumpsys/smaps; PSS + GPU as in the profile)", peak_pss, "<=",
              lim["max_peak_memory_mb"]),
        check("peak memory (MB)", "max VmHWM (resident high-water mark; conservative)", peak_hwm, "<=",
              lim["max_peak_memory_mb"]),
        check("prefill (tokens/s)", "median", s_pre.get("median"), ">=", lim["min_prefill_tps"]),
        check("prefill (tokens/s)", "p95 (slow tail: 95% of requests at least this fast = 5th percentile)",
              s_pre.get("p05"), ">=", lim["min_prefill_tps"]),
        check("decode (tokens/s)", "median", s_dec.get("median"), ">=", lim["min_decode_tps"]),
        check("decode (tokens/s)", "p95 (slow tail: 95% of requests at least this fast = 5th percentile)",
              s_dec.get("p05"), ">=", lim["min_decode_tps"]),
        check("energy (J per request)", f"average over the batch, {headline}, net of pre-batch idle", e_req, "<=",
              lim["max_energy_j"], "" if e_valid else "INVALID: " + "; ".join(reasons)),
    ]
    if not e_valid:  # an energy reading on external power / without idle baseline is not a measurement
        verdict[-1]["pass"] = None
    secondary = []
    if lat.get("max_ttft_s") is not None:
        secondary += [check("time to first token (s)", "median", pct(ttft, 50), "<=", lat["max_ttft_s"]),
                      check("time to first token (s)", "p95", pct(ttft, 95), "<=", lat["max_ttft_s"])]
    if lat.get("max_e2e_s") is not None:
        secondary += [check("whole answer (s)", "p95", pct(e2e, 95), "<=", lat["max_e2e_s"])]
    summary = {
        "scenario": sess["scenario"], "title": sc["title"], "model": sc["model"], "hf_repo": sc["hf_repo"],
        "phone": {k: sess["device"].get(k) for k in ("manufacturer", "model", "soc", "android", "build")},
        "runtime": {k: v for k, v in sess["runtime"].items() if k != "build_info"},
        "conditions": sess["conditions"], "answer_length": sess["answer_length"],
        "request_set": {"measured": len(meas), "batches": len(batches),
                        "prompt_tokens": describe([r["prompt_tokens"] for r in meas]),
                        "answer_tokens_reference": describe([r["answer_tokens_reference"] for r in meas]),
                        "decode_forwards": describe([r["decode_forwards"] for r in meas]),
                        "stopped_at_eos": sum(1 for r in meas if r["stopped_at_eos"]),
                        "requirement_request_shape": sc["requirement_request"]},
        "results": {"prefill_tps": s_pre, "decode_tps": s_dec, "ttft_s": describe(ttft), "e2e_s": describe(e2e),
                    "prefill_tps_mnn_internal": describe([r["prefill_tps_mnn"] for r in meas]),
                    "decode_tps_mnn_internal": describe([r["decode_tps_mnn"] for r in meas]),
                    "peak_memory_mb": {"pss": rnd(peak_pss, 1), "vmhwm": rnd(peak_hwm, 1)},
                    "energy_j_per_request": {"headline_method": headline, "net_pre_idle": rnd(e_req, 3),
                                             "net_post_idle": e_alt.get("post_idle"), "net_both_idle": e_alt.get("both_idle"),
                                             "valid": e_valid,
                                             "per_request_indicative": describe(ind),
                                             "phase_split_indicative": split,
                                             "gauge_updates_per_request": describe([r.get("gauge_updates") for r in meas])}},
        "limits": lim, "latency_limits": lat, "profile_values": sc["profile_values"],
        "verdict": verdict, "secondary_latency_checks": secondary,
        "all_pass": all(v["pass"] for v in verdict) if all(v["pass"] is not None for v in verdict) else None,
        "batches": batches,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (out / "summary.md").write_text(render_md(summary), encoding="utf-8")
    print(render_md(summary))
    return summary


def energy_probe_idle(fn, idle):
    try:
        return fn(idle[0], idle[1])
    except Exception:  # noqa: BLE001
        return None


def render_md(s: dict) -> str:
    r = s["results"]
    rs = s["request_set"]
    lines = [f"# {s['title']}", "",
             f"Model: {s['model']} ({s['hf_repo']}); MNN {str(s['runtime'].get('mnn_commit'))[:12]}, "
             f"{s['runtime'].get('backend', '').upper()} x{s['runtime'].get('threads')}, answer lengths: "
             f"{s['answer_length']}.",
             f"Phone: {s['phone'].get('manufacturer')} {s['phone'].get('model')} ({s['phone'].get('soc')}), "
             f"Android {s['phone'].get('android')}. Conditions: {s['conditions'].get('radio_mode')}; screen "
             f"{s['conditions'].get('screen')}; gate {s['conditions'].get('max_temp_c')} C.",
             f"Requests: {rs['measured']} measured in {rs['batches']} batch(es); prompt tokens median "
             f"{rs['prompt_tokens'].get('median')} / p95 {rs['prompt_tokens'].get('p95')}; reference answer tokens "
             f"median {rs['answer_tokens_reference'].get('median')} / p95 {rs['answer_tokens_reference'].get('p95')}; "
             f"decode forwards median {rs['decode_forwards'].get('median')}. Requirement shape: "
             f"{rs['requirement_request_shape']}.", "",
             "| Metric | Statistic | Measured | Requirement | Pass |", "|---|---|--:|---|---|"]
    for v in s["verdict"] + s["secondary_latency_checks"]:
        p = {True: "PASS", False: "**FAIL**", None: "n/a"}[v["pass"]]
        lines.append(f"| {v['metric']} | {v['statistic']} | {v['measured']} | {v['requirement']} | {p} "
                     f"{v['note']} |")
    e = r["energy_j_per_request"]
    lines += ["", f"Prefill tokens/s: {r['prefill_tps']}", f"Decode tokens/s: {r['decode_tps']}",
              f"Time to first token (s): {r['ttft_s']}", f"Whole request (s): {r['e2e_s']}",
              f"Energy per request (J): {e['net_pre_idle']} net of pre-batch idle ({e['headline_method']}); "
              f"{e['net_post_idle']} net of post-batch idle; per-request indicative {e['per_request_indicative']}; "
              f"gauge updates per request {e['gauge_updates_per_request']}",
              f"Peak memory (MB = 10^6 B): Pss {r['peak_memory_mb']['pss']}, VmHWM {r['peak_memory_mb']['vmhwm']}",
              f"Profile values for this configuration: {s['profile_values']}", ""]
    for b in s["batches"]:
        lines.append(f"- {b['tag']}: exit {b['exit']}, {b['requests']} requests in {b['batch_s']} s (load "
                     f"{b['load_s']} s), battery {b['battery_temp_c']} C, level {b['battery_level']}, CPU capped in "
                     f"{b['cpu_capped_samples']}/{b['timeline_samples']} samples, max thermal status "
                     f"{b['max_thermal_status']}, suspended requests {b['suspended_requests']}, powered "
                     f"{b['externally_powered']}, gate {b['gate']}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS))
    ap.add_argument("--serial")
    ap.add_argument("--prepare", action="store_true", help="push model, requests (and check the binary); no run")
    ap.add_argument("--summarize", metavar="RESULTS_DIR", help="recompute summary.json/.md from a results folder")
    ap.add_argument("--model-dir", help="local model directory (default: download the HF repo into ./models)")
    ap.add_argument("--answer-length", choices=["reference", "eos"], default="reference",
                    help="reference: decode exactly the reference answer's token count, EOS ignored (default); "
                         "eos: greedy until the model's own end-of-turn, capped at the eval's 512")
    ap.add_argument("--n", type=int, default=300, help="measured requests (consecutive held-out rows)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1, help="batches, each after its own cool-down gate and idle")
    ap.add_argument("--backend", choices=["cpu", "gpu"], default=None, help="default: the factory's choice (CPU)")
    ap.add_argument("--threads", type=int, default=None, help="default 4 (paper protocol and config.json)")
    ap.add_argument("--mnn-sha", default=MNN_SHA, help="which pb_requests build on the phone (build_pb_requests.py)")
    ap.add_argument("--screen", choices=["auto", "off", "on", "panel", "awake"], default=None,
                    help="default: the device profile's (OnePlus 8 Pro: panel), else auto (off + wake lock)")
    ap.add_argument("--max-temp", type=float, default=None)
    ap.add_argument("--gate-plateau", type=int, default=None)
    ap.add_argument("--gate-timeout", type=int, default=1800)
    ap.add_argument("--idle-seconds", type=int, default=60)
    ap.add_argument("--post-idle-seconds", type=int, default=60)
    ap.add_argument("--rest", type=int, default=15)
    ap.add_argument("--gap-ms", type=int, default=0, help="pause between requests (default 0: back to back)")
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="seconds per batch")
    ap.add_argument("--min-battery", type=int, default=50)
    ap.add_argument("--allow-powered", action="store_true", help="run on external power (energy marked invalid)")
    ap.add_argument("--any-phone", action="store_true")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--no-perfetto", action="store_true")
    ap.add_argument("--results-dir")
    a = ap.parse_args()
    if a.summarize:
        summarize(Path(a.summarize))
        return
    if not a.scenario:
        ap.error("--scenario is required")
    sc = SCENARIOS[a.scenario]
    a.backend = a.backend or sc["backend"]
    a.threads = a.threads or sc["threads"]
    out = session(a)
    if out is not None:
        summarize(out)
        log(f"done -> {out}")


if __name__ == "__main__":
    main()
