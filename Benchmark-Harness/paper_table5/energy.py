"""
Energy recording for run_paper_table5.py: one phone-side sampler (powercap counters + readable battery
sysfs files, 10 Hz) and one Perfetto trace (battery HAL counters + Power Stats rails) span a whole session;
afterwards every method is integrated over the benchmark's own timed windows (PB_MARK lines).

Energy per token for a phase = sum of window energies / sum of tokens, in uJ/token (the paper's unit).
  * powercap / rail methods: chip-level counters, gross energy (what the paper reports), valid on a cable.
  * battery methods: whole phone; reported gross and net of the session's idle power. Invalid while the
    phone is externally powered.
Headline method: the device profile's energy_method if available, else powercap (the zone with the most
energy, normally the SoC total), else rail:TOTAL, else the first available battery method in
BATTERY_PREFERENCE (energy_probe.py measures which one is most repeatable on a given phone).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import energy_probe as ep

BATTERY_PREFERENCE = ("perfetto:charge", "sysfs:charge_counter", "perfetto:current", "sysfs:current_now",
                      "sysfs:current_avg", "sysfs:power_now")


def is_chip_method(method: str) -> bool:
    return method.startswith(("powercap:", "rail:"))


def probe_recommendation(model: str, results_dir: Path | None = None) -> str | None:
    """The method energy_probe.py recommended in its newest valid run for this phone model, if any."""
    results_dir = results_dir or Path(ep.HERE) / "energy_probe_results"
    best = None
    for rep in results_dir.glob("*/report.json"):
        try:
            meta = json.loads((rep.parent / "meta.json").read_text())
            data = json.loads(rep.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("discovery", {}).get("model") != model or not data.get("recommended") \
                or data.get("externally_powered"):
            continue
        if best is None or rep.stat().st_mtime > best[0]:
            best = (rep.stat().st_mtime, data["recommended"])
    return best[1] if best else None


def merge_contiguous(windows: list[dict], max_gap_s: float = 0.5) -> list[dict]:
    """Merge consecutive same-phase windows separated by less than max_gap_s into one span (tokens summed);
    a window of another phase in between breaks the span."""
    spans: list[dict] = []
    for w in sorted(windows, key=lambda w: w["a"]):
        last = spans[-1] if spans else None
        if last and last["phase"] == w["phase"] and 0 <= w["a"] - last["b"] < max_gap_s:
            last["b"] = max(last["b"], w["b"])
            last["tokens"] += w["tokens"]
            last["covered"] += w["b"] - w["a"]
            last["n"] += 1
        else:
            spans.append({"phase": w["phase"], "a": w["a"], "b": w["b"], "tokens": w["tokens"],
                          "covered": w["b"] - w["a"], "n": 1})
    return spans


class EnergyRecorder:
    def __init__(self, adb, out_dir: Path, discovery: dict, use_perfetto: bool = True, log=print):
        self.adb, self.out_dir, self.disc, self.log = adb, Path(out_dir), discovery, log
        self.use_perfetto = use_perfetto
        self.columns: list[tuple[str, str]] = []
        self.perfetto_pid = None
        self.trace_name = None
        self.funcs: dict | None = None

    def start(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        bdir = self.disc.get("battery_dir")
        self.columns = [(f"sysfs:{f}", f"{bdir}/{f}") for f in self.disc.get("battery_files", [])]
        pc = self.disc["powercap"]
        for z in pc["zones"]:
            if z["shell_readable"] or z.get("su_readable"):
                label = (z["name"] or Path(z["path"]).name).replace(" ", "_")
                self.columns.append((f"powercap:{label}", f"{z['path']}/energy_uj"))
        self.adb.sh(f"mkdir -p {ep.DEV_DIR}; rm -f {ep.DEV_DIR}/stop {ep.DEV_DIR}/samples.txt")
        ep.push_text(self.adb, ep.sampler_script(self.columns, 0.1), f"{ep.DEV_DIR}/sampler.sh")
        cmd = f"setsid sh {ep.DEV_DIR}/sampler.sh"
        if pc["usable"] and pc["needs_su"]:
            cmd = f"su -c '{cmd}'"
        self.adb.sh(f"{cmd} </dev/null >/dev/null 2>&1 &")
        if self.use_perfetto:
            self.perfetto_pid = ep.start_perfetto(self.adb)
        self.log(f"[ENERGY] sampler columns: {[c for c, _ in self.columns] or 'timestamps only'}; "
                 f"perfetto: {'on' if self.perfetto_pid else 'off'}")

    def stop(self):
        self.adb.sh(f"touch {ep.DEV_DIR}/stop")
        time.sleep(1)
        if self.perfetto_pid is not None:
            self.trace_name = ep.stop_perfetto(self.adb, self.perfetto_pid, self.out_dir)
            self.perfetto_pid = None
        self.adb.run(["pull", f"{ep.DEV_DIR}/samples.txt", str(self.out_dir / "samples.txt")], timeout=300)

    def load(self):
        trace = self.out_dir / "trace.pftrace"
        self.funcs = ep.power_functions(self.out_dir / "samples.txt", trace if trace.exists() else None,
                                        self.disc["powercap"]["zones"])
        return self.funcs

    def headline(self, preferred: str | None) -> str | None:
        methods = list(self.funcs or {})
        if preferred and preferred in methods:
            return preferred
        caps = [m for m in methods if m.startswith("powercap:")]
        if caps:
            return caps[0]
        if "rail:TOTAL" in methods:
            return "rail:TOTAL"
        return next((m for m in BATTERY_PREFERENCE if m in methods), None)

    def phase_energy(self, windows: list[dict], idle: tuple[float, float] | None, powered: bool) -> dict:
        """windows: [{"phase", "a", "b", "tokens"}] (CLOCK_BOOTTIME seconds). Returns
        {method: {phase: {uj_per_token, net_uj_per_token, avg_power_mw, seconds, tokens, windows, valid, reason}}}."""
        out = {}
        spans = merge_contiguous(windows)
        for method, (fn, _series, note) in (self.funcs or {}).items():
            idle_mw = fn(*idle) if idle else None
            # Battery gauges update every ~0.1-5 s: integrating each sub-second repetition separately would bill
            # every window's first reading to the preceding gap. Back-to-back repetitions are integrated as one span.
            use = windows if is_chip_method(method) else spans
            load = [fn(w["a"], w["b"]) for w in use]
            usable = [p for p in load if p is not None]
            sign = 1.0
            if idle_mw is not None and usable and statistics.mean(usable) < idle_mw:
                sign = -1.0  # this gauge reports discharge as negative
            if idle_mw is not None:
                idle_mw *= sign
            per_phase: dict = {}
            for w, p in zip(use, load):
                d = per_phase.setdefault(w["phase"], {"mj": 0.0, "net_mj": 0.0, "s": 0.0, "tokens": 0, "n": 0,
                                                      "missing": 0})
                if p is None:
                    d["missing"] += 1
                    continue
                # a span's average power applies to the time spent in its own windows, not to the gaps between
                # them (which can hold other work, e.g. the token MNN generates after each prefill)
                dur = w.get("covered", w["b"] - w["a"])
                d["mj"] += sign * p * dur
                d["net_mj"] += (sign * p - (idle_mw or 0.0)) * dur
                d["s"] += dur
                d["tokens"] += w["tokens"]
                d["n"] += 1
            res = {}
            for phase, d in per_phase.items():
                reasons = []
                if not is_chip_method(method) and powered:
                    reasons.append("externally powered: battery reading follows the charger")
                if d["missing"]:
                    reasons.append(f"{d['missing']} window(s) without samples")
                if not d["tokens"]:
                    reasons.append("no usable windows")
                res[phase] = {
                    "uj_per_token": round(d["mj"] * 1e3 / d["tokens"], 1) if d["tokens"] else None,
                    "net_uj_per_token": (round(d["net_mj"] * 1e3 / d["tokens"], 1)
                                         if d["tokens"] and idle_mw is not None and not is_chip_method(method) else None),
                    "avg_power_mw": round(d["mj"] / d["s"], 1) if d["s"] else None,
                    "idle_power_mw": round(idle_mw, 1) if idle_mw is not None else None,
                    "seconds": round(d["s"], 2), "tokens": d["tokens"], "windows": d["n"],
                    "valid": not reasons, "reason": "; ".join(reasons) or None, "note": note,
                }
            out[method] = res
        return out
