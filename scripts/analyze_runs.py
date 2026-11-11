"""Extract per-run, per-model time / cost / score metrics for the COST, EXPLORATION and ESCALATION docs.

Reads only what each run already wrote — `run.log`, `timing-events.jsonl`, `cost-events.jsonl`,
`scores.json` — plus `sacct` for the GPU allocation, and writes one JSON file:
`docs/SLMF_Results/data/run_metrics.json`. `scripts/plot_run_analysis.py` draws the figures from it.

HOW TIME AND MONEY ARE ATTRIBUTED TO A MODEL TIER.
  * Every `[evaluate][<model> [<quant>]] Score:` line in run.log is one train->eval iteration, and
    they pair one-to-one, in order, with the `evaluate` graph-node events in timing-events.jsonl
    (both are written by the same node; the count is checked and a mismatch is reported).
  * Iteration k owns every graph-node event that ENDS after evaluate k-1 and at or before
    evaluate k — curate, train, build, evaluate, iterate, rollback, escalate. Events before the
    first iteration's window (task analysis, eval setup, model selection) are `setup`; events after
    the last evaluate are `wrapup`.
  * API spend is bucketed the same way by each cost event's timestamp.
  * Graph-node time is the run's ACTIVE time — it excludes a requeued job's downtime. GPU-hours are
    graph-node hours x the job's allocated GPUs; what `sacct` billed beyond that (vLLM boot, Python
    startup, the time between a USR1 checkpoint and the kill) is reported as `overhead`, not spread
    over tiers.
  * A model's tier is read from the run's last `Model Improvement Report` table.

Login-node safe: single-threaded, reads a few text files per run.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "logs" / "runs"
OUT = ROOT / "docs" / "SLMF_Results" / "data" / "run_metrics.json"

# Reference cloud price for one L40S GPU-hour — the median on-demand rate across ~32 providers
# (getdeploying.com / gpucostcompare.com, October 2026). Low/high bound the sensitivity band:
# RunPod $0.99 and AWS g6e.xlarge $1.86.
GPU_HOUR_USD = 1.60
GPU_HOUR_USD_LOW = 0.99
GPU_HOUR_USD_HIGH = 1.86

# family -> task -> (run directory, [slurm job ids whose allocation belongs to that run directory])
RUN_SETS: dict[str, dict[str, tuple[str, list[str]]]] = {
    "main": {
        "clinc150": ("slm-clinc150-ablation-nostop-l40s-40105479", ["40105479", "40214984"]),
        "sms_spam": ("slm-sms-spam-ablation-nostop-l40s-40105478", ["40105478"]),
        # 42138096 / 42138097 replaced 39294409 / 39311801 (2026-10-21): the 08-29 runs used the
        # pre-08-30 model pool and fed our own eval scores into model selection.
        "calendar_json": ("slm-calendar-json-l40s-42138096", ["42138096"]),
        # 41959026 replaced 39361648 (2026-10-21): the original became a DeepSeek-teacher, 1-GPU run
        # through the submitting shell's environment; this one is the standard launcher, local Qwen.
        "xlam_bfcl": ("slm-xlam-bfcl-l40s-41959026", ["41959026"]),
        "ner_bc5cdr": ("slm-ner-bc5cdr-l40s-42138097", ["42138097"]),
        "goemotions": ("slm-goemotions-ckpt-39708679", ["39708679"]),
        "gec_bea19": ("slm-gec-bea19-l40s-39881529", ["39881529"]),
        # 42138121 replaced 39881531 after the loader fix (the prefix slice had 27 of 33 types).
        "multiconer": ("slm-multiconer-l40s-42138121", ["42138121"]),
        "dialogsum": ("slm-dialogsum-l40s-39969104", ["39969104"]),
        "topv2": ("slm-topv2-l40s-39969102", ["39969102"]),
    },
    "single_model": {
        "sms_spam": ("slm-sms-spam-ablation-singlemodel-l40s-41496759", ["41496759"]),
        "goemotions": ("slm-goemotions-ablation-singlemodel-l40s-41496760", ["41496760"]),
        "calendar_json": ("slm-calendar-json-ablation-singlemodel-l40s-41496761", ["41496761"]),
        "gec_bea19": ("slm-gec-bea19-ablation-singlemodel-l40s-41496762", ["41496762"]),
        # 42138122 replaced 41496763 after the multiconer loader fix.
        "multiconer": ("slm-multiconer-ablation-singlemodel-l40s-42138122", ["42138122"]),
        "dialogsum": ("slm-dialogsum-ablation-singlemodel-l40s-41496764", ["41496764"]),
        "clinc150": ("slm-clinc150-ablation-singlemodel-l40s-41496765", ["41496765"]),
        "xlam_bfcl": ("slm-xlam-bfcl-ablation-singlemodel-l40s-41496766", ["41496766"]),
        "topv2": ("slm-topv2-ablation-singlemodel-l40s-41496767", ["41496767"]),
        "ner_bc5cdr": ("slm-ner-bc5cdr-ablation-singlemodel-l40s-40831542", ["40831542"]),
    },
    "size_sweep": {
        # 41517188 died at startup on config drift; 41940584 is the fixed resume (killed by a proxy 503
        # at 11:21 on 10-17) and 41958216 resumes it again. Same run dir throughout.
        "ner_bc5cdr": ("slm-ner-bc5cdr-ablation-sizesweep-l40s-40827150",
                       ["40827150", "41517188", "41940584", "41958216"]),
    },
    "scarce": {
        "sms_spam/nosynth": ("slm-sms-spam-ablation-scarce-nosynth-l40s-40128751", ["40128751"]),
        "sms_spam/synth": ("slm-sms-spam-ablation-scarce-synth-l40s-40126479", ["40126479"]),
        "clinc150/nosynth": ("slm-clinc150-ablation-scarce-nosynth-l40s-40175896", ["40175896"]),
        "clinc150/synth": ("slm-clinc150-ablation-scarce-synth-l40s-40175895", ["40175895"]),
        "xlam_bfcl/nosynth": ("slm-xlam-bfcl-ablation-scarce-nosynth-l40s-40315641", ["40315641"]),
        "xlam_bfcl/synth": ("slm-xlam-bfcl-ablation-scarce-synth-l40s-40315640",
                            ["40315640", "40388434"]),
        "calendar_json": ("slm-calendar-json-ablation-scarce-l40s-41940585", ["41940585", "41958217"]),
        "ner_bc5cdr": ("slm-ner-bc5cdr-ablation-scarce-l40s-41940586", ["41940586"]),
        "goemotions": ("slm-goemotions-ablation-scarce-l40s-41940587", ["41940587"]),
        "gec_bea19": ("slm-gec-bea19-ablation-scarce-l40s-41940588", ["41940588"]),
        "multiconer": ("slm-multiconer-ablation-scarce-l40s-41940589", ["41940589"]),
        # 41940590 died on a transient llama.cpp CUDA init error; 42138095 resumes the same run dir.
        "dialogsum": ("slm-dialogsum-ablation-scarce-l40s-41940590", ["41940590", "42138095"]),
        "topv2": ("slm-topv2-ablation-scarce-l40s-41940591", ["41940591"]),
    },
    # Runs replaced for a non-uniform configuration or a loader bug. COST and ESCALATION do not
    # read this family; EXPLORATION reads the three that met their goal, because their raise
    # decisions are real decisions whatever the run's other confounds.
    "superseded": {
        "xlam_bfcl (DeepSeek teacher)": ("slm-xlam-bfcl-l40s-39361648", ["39361648"]),
        "calendar_json (08-29)": ("slm-calendar-json-l40s-39294409", ["39294409"]),
        "ner_bc5cdr (08-30)": ("slm-ner-bc5cdr-l40s-39311801", ["39311801"]),
        "multiconer (27 types)": ("slm-multiconer-l40s-39881531", ["39881531"]),
        "multiconer single (27 types)": ("slm-multiconer-ablation-singlemodel-l40s-41496763",
                                         ["41496763"]),
    },
    # Other runs that met their goal at least once — used only by EXPLORATION.md.
    "other_converged": {
        "sms_spam (original)": ("slm-sms-spam-l40s-39040698", ["39040698"]),
        "ner_bc5cdr reset-ablation": ("slm-ner-ablation-reset-cse-39562028", []),
        "ner_bc5cdr nosynth-ablation": ("slm-ner-ablation-nosynth-cse-39562030", []),
        "ner_bc5cdr deepseek-ablation": ("slm-ner-ablation-deepseek-cse-39562029", []),
        "address_bench": ("slm-address-bench-l40s-41497353", ["41497353"]),
    },
    # Default-goal reruns (2026-11-04) of the two tasks whose main runs pin the goal at 0.99. ESCALATION
    # compares the one-shot pick against these, so both arms use the default goal on all ten tasks;
    # EXPLORATION counts their goal decisions. COST keeps the main family (the paper's ten main runs).
    "default_goal": {
        "clinc150": ("slm-clinc150-calibrated-l40s-42806317", ["42806317"]),
        "sms_spam": ("slm-sms-spam-calibrated-l40s-42806318", ["42806318"]),
    },
}

SCORE_RE = re.compile(
    r"\[evaluate\]\[(?P<model>.+?) \[(?P<quant>[\w.]+)\]\] Score: (?P<metric>\w+)=(?P<score>[\d.]+)"
)
REPORT_ROW_RE = re.compile(
    r"^\s*(?:\[\d\d:\d\d:\d\d\])?\s+(?P<tier>\d+)\s+(?P<model>\S+)\s+(?P<quant>Q\d\w*|bf16|fp16|F16)\s+"
    r"(?P<base>[\d.]+)\s+(?P<first>[\d.]+)\s+(?P<best>[\d.]+)"
)
SELECTED_RE = re.compile(r"Selected (?P<model>\S+) \[(?P<quant>[\w.]+)\] \(tier=(?P<tier>\d+)")
GOAL_MET_RE = re.compile(
    r"Goal (?P<goal>[\d.]+) met with (?P<score>[\d.]+) at iteration (?P<iter>\d+)")
RAISE_RE = re.compile(
    r"RAISING accuracy goal (?P<from>[\d.]+) → (?P<to>[\d.]+) .*?iteration (?P<iter>\d+)\)\. "
    r"Reason: (?P<reason>.*)$")
DECLINE_RE = re.compile(
    r"Stretch goal DECLINED by the orchestrator — keeping the goal at (?P<goal>[\d.]+) and "
    r"finishing at (?P<score>[\d.]+) .*?Reason: (?P<reason>.*)$")
DECLINE_STEP_RE = re.compile(r"Stretch goal DECLINED: proposed (?P<proposed>[\d.]+)")
ESCALATE_RE = re.compile(r"\[escalate\]|ESCALAT", re.I)


def _ts(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _elapsed_s(text: str) -> float:
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [float(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def sacct_allocation(job_ids: list[str]) -> dict:
    """Allocated GPU-hours and elapsed time across every segment (requeues included) of the jobs."""
    out = {"jobs": [], "gpu_hours": 0.0, "elapsed_hours": 0.0, "gpus": None, "states": []}
    for job in job_ids:
        try:
            text = subprocess.run(
                ["sacct", "-j", job, "-D", "-X", "-n", "-P",
                 "-o", "JobID,State,Elapsed,AllocTRES,Start,End"],
                capture_output=True, text=True, timeout=60, check=False,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in text.strip().splitlines():
            jid, state, elapsed, tres, start, end = line.split("|")
            gpus = int(m.group(1)) if (m := re.search(r"gres/gpu=(\d+)", tres)) else 0
            hours = _elapsed_s(elapsed) / 3600
            out["jobs"].append({"job": jid, "state": state, "hours": round(hours, 4),
                                "gpus": gpus, "start": start, "end": end})
            out["gpu_hours"] += hours * gpus
            out["elapsed_hours"] += hours
            out["states"].append(state)
            if gpus:
                out["gpus"] = gpus
    return out


def tier_map(log_lines: list[str]) -> dict[str, int]:
    """selector -> tier, from the LAST Model Improvement Report (falls back to selection lines)."""
    tiers: dict[str, int] = {}
    for line in log_lines:
        if m := SELECTED_RE.search(line):
            tiers.setdefault(f"{m['model']}@{m['quant']}", int(m["tier"]))
    starts = [i for i, line in enumerate(log_lines) if "Model Improvement Report" in line]
    if starts:
        for line in log_lines[starts[-1] + 3: starts[-1] + 60]:
            m = REPORT_ROW_RE.match(line)
            if not m:
                if "DAG Traversal" in line:
                    break
                continue
            tiers[f"{m['model']}@{m['quant']}"] = int(m["tier"])
    return tiers


def parse_run(run_dir: Path) -> dict:
    lines = (run_dir / "run.log").read_text(errors="replace").splitlines()

    evals: list[dict] = []
    goal_events: list[dict] = []
    for line in lines:
        if m := SCORE_RE.search(line):
            evals.append({"selector": f"{m['model']}@{m['quant']}", "metric": m["metric"],
                          "score": float(m["score"])})
        elif m := GOAL_MET_RE.search(line):
            goal_events.append({"kind": "met", "eval_index": len(evals), "goal": float(m["goal"]),
                                "score": float(m["score"]), "tier_iteration": int(m["iter"])})
        elif m := RAISE_RE.search(line):
            goal_events.append({"kind": "raise", "eval_index": len(evals),
                                "from": float(m["from"]), "to": float(m["to"]),
                                "tier_iteration": int(m["iter"]), "reason": m["reason"].strip()})
        elif m := DECLINE_RE.search(line):
            goal_events.append({"kind": "decline", "eval_index": len(evals),
                                "goal": float(m["goal"]), "score": float(m["score"]),
                                "reason": m["reason"].strip()})
        elif m := DECLINE_STEP_RE.search(line):
            goal_events.append({"kind": "decline_min_step", "eval_index": len(evals),
                                "proposed": float(m["proposed"])})

    tiers = tier_map(lines)

    # Graph-node events, ordered by end time.
    nodes = []
    for raw in (run_dir / "timing-events.jsonl").read_text().splitlines():
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if e.get("kind") == "graph_node":
            nodes.append((_ts(e["timestamp"]), e["name"], float(e.get("duration_ms") or 0) / 1000,
                          e.get("status", "success")))
    nodes.sort()
    # A FAILED evaluate (e.g. a GGUF build crash right before a job died) writes no Score line, so
    # only successful ones pair with run.log. Its time still counts, in the next iteration's window.
    eval_ends = [t for t, name, _, status in nodes if name == "evaluate" and status == "success"]
    pairing_note = None
    if len(eval_ends) != len(evals):
        pairing_note = (f"{len(evals)} Score lines vs {len(eval_ends)} evaluate events; "
                        f"paired the first {min(len(evals), len(eval_ends))}")
    n = min(len(evals), len(eval_ends))
    evals = evals[:n]
    eval_ends = eval_ends[:n]

    def bucket(ts: float) -> int:
        """-1 = setup, 0..n-1 = iteration index, n = wrap-up."""
        lo, hi = 0, n
        while lo < hi:  # first eval_end >= ts
            mid = (lo + hi) // 2
            if eval_ends[mid] < ts:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            # Before (or at) the first evaluate: setup unless it is part of the first iteration,
            # which starts at the first `curate`/`train` node.
            return 0 if ts >= first_iter_start else -1
        return lo

    first_iter_start = next((t - d for t, name, d, _ in nodes if name in ("curate", "train")),
                            eval_ends[0] if eval_ends else 0.0)

    iter_seconds = defaultdict(float)
    for t, _name, d, _status in nodes:
        iter_seconds[bucket(t)] += d

    iter_usd = defaultdict(float)
    usd_by_provider = defaultdict(float)
    usd_by_stage = defaultdict(float)
    cost_path = run_dir / "cost-events.jsonl"
    for raw in (cost_path.read_text().splitlines() if cost_path.exists() else []):
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        usd = float(e.get("estimated_usd") or 0.0)
        if not usd:
            continue
        iter_usd[bucket(_ts(e["timestamp"]))] += usd
        usd_by_provider[f"{e.get('provider')}:{e.get('model')}"] += usd
        usd_by_stage[str(e.get("stage"))] += usd

    cumulative = 0.0
    best = 0.0
    for k, ev in enumerate(evals):
        ev["index"] = k
        ev["tier"] = tiers.get(ev["selector"])
        ev["active_seconds"] = round(iter_seconds[k], 2)
        ev["usd"] = round(iter_usd[k], 6)
        cumulative += iter_seconds[k] + (iter_seconds[-1] if k == 0 else 0.0)
        ev["cumulative_active_hours"] = round(cumulative / 3600, 4)
        best = max(best, ev["score"])
        ev["best_so_far"] = round(best, 4)

    scores = json.loads((run_dir / "scores.json").read_text())
    return {
        "run_dir": run_dir.name,
        "evals": evals,
        "pairing_note": pairing_note,
        "tiers": tiers,
        "setup_active_seconds": round(iter_seconds[-1], 2),
        "wrapup_active_seconds": round(iter_seconds[n], 2),
        "setup_usd": round(iter_usd[-1], 6),
        "wrapup_usd": round(iter_usd[n], 6),
        "active_hours": round(sum(iter_seconds.values()) / 3600, 4),
        "usd_total": round(sum(iter_usd.values()), 4),
        "usd_by_provider": {k: round(v, 4) for k, v in usd_by_provider.items()},
        "usd_by_stage": {k: round(v, 4) for k, v in usd_by_stage.items()},
        "goal_events": goal_events,
        "scores_json": {k: scores.get(k) for k in (
            "best_score", "final_selector", "trajectory_selector", "stop_threshold",
            "initial_stop_threshold", "converged", "iterations", "threshold_raises",
            "threshold_lowers", "convergence_banked")},
        "threshold_source": (scores.get("threshold_calibration") or {}).get("reason"),
    }


def per_model(run: dict, gpus: int) -> list[dict]:
    """Aggregate a run's iterations by selector, in order of first appearance."""
    rows: dict[str, dict] = {}
    for ev in run["evals"]:
        r = rows.setdefault(ev["selector"], {
            "selector": ev["selector"], "tier": ev["tier"], "iterations": 0,
            "active_hours": 0.0, "usd": 0.0, "best": 0.0, "first": ev["score"]})
        r["iterations"] += 1
        r["active_hours"] += ev["active_seconds"] / 3600
        r["usd"] += ev["usd"]
        r["best"] = max(r["best"], ev["score"])
    for r in rows.values():
        r["gpu_hours"] = round(r["active_hours"] * gpus, 4)
        r["gpu_usd"] = round(r["gpu_hours"] * GPU_HOUR_USD, 4)
        r["active_hours"] = round(r["active_hours"], 4)
        r["usd"] = round(r["usd"], 4)
    return list(rows.values())


def main() -> int:
    result = {"gpu_hour_usd": GPU_HOUR_USD, "gpu_hour_usd_low": GPU_HOUR_USD_LOW,
              "gpu_hour_usd_high": GPU_HOUR_USD_HIGH,
              "generated_at": datetime.now().isoformat(timespec="seconds"), "families": {}}
    for family, runs in RUN_SETS.items():
        result["families"][family] = {}
        for task, (dirname, jobs) in runs.items():
            run_dir = RUNS / dirname
            if not (run_dir / "run.log").exists():
                result["families"][family][task] = {"run_dir": dirname, "missing": True,
                                                    "jobs": jobs}
                print(f"[{family}] {task}: no run.log yet ({dirname})", file=sys.stderr)
                continue
            run = parse_run(run_dir)
            alloc = sacct_allocation(jobs)
            gpus = alloc["gpus"] or 2
            run["allocation"] = alloc
            run["gpus"] = gpus
            run["models"] = per_model(run, gpus)
            graph_gpu_h = run["active_hours"] * gpus
            run["gpu_hours_total"] = round(alloc["gpu_hours"] or graph_gpu_h, 4)
            run["gpu_hours_setup"] = round(
                (run["setup_active_seconds"] + run["wrapup_active_seconds"]) / 3600 * gpus, 4)
            run["gpu_hours_overhead"] = round(max(0.0, run["gpu_hours_total"] - graph_gpu_h), 4)
            run["gpu_usd_total"] = round(run["gpu_hours_total"] * GPU_HOUR_USD, 2)
            result["families"][family][task] = run
            note = f"  ({run['pairing_note']})" if run["pairing_note"] else ""
            print(f"[{family}] {task}: {len(run['evals'])} evals, "
                  f"{run['gpu_hours_total']:.1f} GPU-h, ${run['usd_total']:.2f} API{note}",
                  file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
