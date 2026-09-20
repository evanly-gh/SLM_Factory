#!/usr/bin/env python
"""Rebuild a finished run's final report from its run directory.

WHY THIS EXISTS. The report is emitted inline at the end of `tests/pipeline/run.py`, from the live
`last_state`, and it is wrapped in a broad `except BaseException` so a broken report can never take
a finished run down with it. That is the right trade — but it means a formatting bug in the report
destroys the report of a run that otherwise succeeded, with no way to get it back short of
re-running the whole job. It happened on 2026-09-05 to all three NER ablation arms: each COMPLETED
with exit 0 after 4-9 hours, and each printed

    !! final report failed: TypeError: unsupported format string passed to NoneType.__format__

mid-way through the DAG traversal table, losing every section after it.

Everything the report reads is already durable on disk (`checkpoint.json`, `baselines.json`,
`cost.json`, `scores.json`), so the report is reconstructible. This script reuses the SAME helpers
the inline report uses — `build_run_progression`, `format_health_summary`,
`format_intervention_detail`, `describe_threshold_provenance` — rather than reimplementing them, so
the two cannot drift into disagreeing about the same run.

    python scripts/regenerate_run_report.py logs/runs/<run-dir> [...]
    python scripts/regenerate_run_report.py --all          # every run dir missing a report

Writes `<run_dir>/final-report.txt` and echoes it to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
os.environ.setdefault("ANTHROPIC_API_KEY", "unused-for-reporting")


def _load(run_dir: Path, name: str, default):
    path = run_dir / name
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _state(run_dir: Path) -> dict:
    blob = _load(run_dir, "checkpoint.json", {})
    return blob.get("state") or blob


def _progression(state: dict, baselines: list[dict]) -> list[dict]:
    """Reconstruct the per-tier progression from durable state.

    `build_run_progression` is used where it can be: it is the function the live report calls, and
    it is what correctly separates post-convergence downward probes from a model's own trajectory.
    It expects `state["selected_model"]` to be a ModelSpec OBJECT, though, and the checkpoint
    serialises it to a string — so the final tier is appended by hand from `state["dag"]`, which
    carries the selector, model_id, quant and tier on every node anyway.
    """
    from agent.pipeline_status import build_run_progression

    by_selector = {b.get("selector", b.get("model_id")): b for b in baselines or []}
    entries = []
    try:
        entries = [
            {**e, "kind": e.get("kind") or "model_trajectory"}
            for e in build_run_progression({**state, "selected_model": None}, baselines or [])
        ]
    except Exception:  # noqa: BLE001 — fall back to the raw history rather than emitting nothing
        entries = [
            {**e, "kind": "model_trajectory"} for e in (state.get("escalation_history") or [])
        ]

    dag = state.get("dag") or []
    if dag:
        selector = dag[-1].get("selector")
        if not any(e.get("selector") == selector and e.get("dag") for e in entries):
            base = by_selector.get(selector, {})
            entries.append({
                "kind": "model_trajectory",
                "selector": selector,
                "model_id": dag[-1].get("model_id") or base.get("model_id") or selector,
                "quant": dag[-1].get("quant") or base.get("quant"),
                "tier": dag[-1].get("tier", "?"),
                "baseline_f1": base.get("baseline_f1"),
                "first_finetuned_f1": base.get("first_finetuned_f1"),
                "best_score": state.get("best_score"),
                "iterations": state.get("iteration", len(dag)),
                "scores": list(state.get("scores") or []),
                "dag": dag,
            })
    return entries


def _num(value, spec=".4f", absent="n/a"):
    """Format a number, or say it is absent. The whole reason this script was needed."""
    return f"{value:{spec}}" if isinstance(value, (int, float)) else absent


def regenerate(run_dir: Path) -> Path:
    from agent.pipeline_status import format_downward_probe_history, format_intervention_detail
    from agent.run_health import format_health_summary
    from agent.threshold import describe_threshold_provenance

    state = _state(run_dir)
    if not state:
        raise SystemExit(f"{run_dir}: no checkpoint.json to rebuild from")
    baselines = _load(run_dir, "baselines.json", [])
    cost = _load(run_dir, "cost.json", {})
    scores_blob = _load(run_dir, "scores.json", {})
    progression = _progression(state, baselines)

    out: list[str] = []

    def log(line=""):
        out.append(line)

    threshold = state.get("stop_threshold") or scores_blob.get("stop_threshold")
    # MAX ACROSS EVERYTHING, not `lifetime_best_score or best_score`. Those two are not
    # interchangeable: `lifetime_best_score` is only updated AT AN ESCALATION, so on a run that
    # converged on its final tier it still holds the PREVIOUS tier's best. Taking it first reported
    # the no-synth arm at 0.7756 (tier 1) when it had in fact converged at 0.8041 on tier 2 —
    # turning a success into an apparent failure.
    candidates = [
        state.get("lifetime_best_score"),
        state.get("best_score"),
        *[e.get("best_score") for e in progression],
    ]
    numeric = [c for c in candidates if isinstance(c, (int, float))]
    best = max(numeric) if numeric else None
    best_entry = next(
        (e for e in progression if e.get("best_score") == best), None
    )
    converged = isinstance(best, (int, float)) and isinstance(threshold, (int, float)) \
        and best >= threshold

    log("=" * 78)
    log(f"  FINAL REPORT (regenerated) — {run_dir.name}")
    log("=" * 78)
    # `or {}` rather than a `.get` default, because `task_plan` is present-and-None on a run that
    # never reached task analysis. This is the same trap the whole script exists to clean up after,
    # and it caught me writing it.
    _task = (state.get("task_plan") or {}).get("task") or state.get("task") or "?"
    log(f"  task            : {_task}")
    log(f"  best score      : {_num(best)}")
    log(f"  accuracy goal   : {_num(threshold)}")
    log(f"  converged       : {'YES' if converged else 'NO'}"
        + (f"  (on {best_entry.get('model_id')}@{best_entry.get('quant') or 'bf16'}, "
           f"tier {best_entry.get('tier','?')})" if best_entry else ""))
    log(f"  models tried    : {len(progression)}")
    log(f"  total iterations: {sum(int(e.get('iterations', 0) or 0) for e in progression)}")
    calibration = state.get("threshold_calibration")
    if calibration:
        log(f"  goal source     : {describe_threshold_provenance(calibration)}")

    if len(progression) > 1:
        log("")
        log(f"  Full run progression ({len(progression)} ordered model attempts):")
        log(f"  {'Tier':>4}  {'Model':<32} {'Quant':<8} {'Iters':>5} {'Best':>7}  Trajectory")
        log(f"  {'-' * 92}")
        for p in progression:
            traj = " → ".join(f"{x:.3f}" for x in p.get("scores") or []) or "(reset)"
            log(f"  {str(p.get('tier','?')):>4}  {str(p.get('model_id','?')):<32} "
                f"{str(p.get('quant') or 'bf16'):<8} {p.get('iterations',0):>5} "
                f"{_num(p.get('best_score')):>7}  {traj}")

    if progression:
        log("")
        log("  Model Improvement Report (per quantized variant):")
        log(f"  {'Tier':>4}  {'Model':<32} {'Quant':<8} {'Baseline':>9} {'First FT':>9} "
            f"{'Best FT':>9} {'Δ base':>9} {'Δ search':>9} {'Format':>8}")
        log(f"  {'-' * 110}")
        for p in progression:
            bl, ft, first = p.get("baseline_f1"), p.get("best_score"), p.get("first_finetuned_f1")
            delta = f"{ft - bl:+.4f}" if isinstance(bl, (int, float)) and isinstance(ft, (int, float)) else "n/a"
            search = f"{ft - first:+.4f}" if isinstance(first, (int, float)) and isinstance(ft, (int, float)) else "n/a"
            best_fmt = next((n.get("format_valid") for n in reversed(p.get("dag") or [])
                             if n.get("format_valid") is not None), None)
            log(f"  {str(p.get('tier','?')):>4}  {str(p.get('model_id','?')):<32} "
                f"{str(p.get('quant') or 'bf16'):<8} {_num(bl):>9} {_num(first):>9} "
                f"{_num(ft):>9} {delta:>9} {search:>9} {_num(best_fmt):>8}")

    if progression:
        log("")
        log(f"  DAG Traversal (all {len(progression)} models):")
        for p in progression:
            pdag = p.get("dag") or []
            log("")
            log(f"  ── Tier {p.get('tier','?')}: {p.get('model_id','?')} "
                f"[{p.get('quant') or 'bf16'}]  ({len(pdag)} iterations) ──")
            if not pdag:
                log("     (no completed iterations recorded)")
                continue
            log(f"     {'Iter':>4}  {'Content':>8}  {'Format':>7}  {'Pruned':>6}  "
                f"{'Config':<34}  {'Intervention'}")
            for node in pdag:
                pruned = "✗" if node.get("pruned") else ""
                iteration = node.get("iteration")
                log(f"     {iteration if isinstance(iteration, int) else '?':>4}  "
                    f"{_num(node.get('score'), '8.4f', ' skipped'):>8}  "
                    f"{_num(node.get('format_valid')):>7}  {pruned:>6}  "
                    f"{str(node.get('best_config','?')):<34}  "
                    f"{format_intervention_detail(node)}")

    probe = format_downward_probe_history(state.get("downward_probe_history"))
    if probe:
        log("")
        out.extend(probe)

    try:
        health = format_health_summary(state)
    except Exception as error:  # noqa: BLE001
        health = [f"  (curriculum ledger unavailable: {type(error).__name__}: {error})"]
    if health:
        log("")
        out.extend(health)

    fitness = state.get("teacher_fitness") or {}
    if fitness:
        from agent.teacher_fitness import format_fitness_measurement

        log("")
        log(f"  teacher measurement: {format_fitness_measurement(fitness)}")
        log(f"    → vs a {_num(fitness.get('threshold'), '.2f', '0.80')} gate: synthetic data "
            f"{'ALLOWED' if fitness.get('synthesis_allowed') else 'REFUSED'}")
        if fitness.get("bypassed"):
            log("    ⚠ SLM_TEACHER_SYNTH_BYPASS=1: the gate was NOT cleared and synthetic data "
                "was allowed anyway.")
        if fitness.get("disallowed_by_operator"):
            log("    ⚠ SLM_SYNTH_DISALLOW=1: synthetic data was refused by operator override "
                "rather than by the gate.")

    by_provider = (cost or {}).get("by_provider") or {}
    if by_provider:
        log("")
        log(f"  cost: total ${_num(cost.get('total_cost_usd'), '.4f', '0.0000')}")
        for name, entry in sorted(by_provider.items()):
            log(f"    {name:<12} calls={entry.get('calls', 0):<7} "
                f"in={entry.get('input_tokens', 0):<12,} out={entry.get('output_tokens', 0):<12,} "
                f"${_num(entry.get('estimated_usd'), '.4f', '0.0000')}")

    manifest = _load(run_dir, "run-manifest.json", {})
    flags = {
        k: v for k, v in ((manifest.get("effective_config") or {}).items())
        if k in ("SLM_ABLATION_RESET_DATA_ON_ESCALATION", "SLM_SYNTH_DISALLOW",
                 "SLM_STOP_THRESHOLD", "SYNTH_MODEL", "SYNTH_API_MODE")
    }
    if flags:
        log("")
        log("  run configuration (from the resume manifest):")
        for k, v in sorted(flags.items()):
            log(f"    {k} = {v!r}")

    log("")
    log(f"  regenerated by scripts/regenerate_run_report.py from {run_dir}")

    text = "\n".join(out) + "\n"
    target = run_dir / "final-report.txt"
    target.write_text(text, encoding="utf-8")
    return target


FAILED_MARKER = "!! final report failed"
APPEND_BANNER = "REGENERATED FINAL REPORT"


def _logs_for(run_dir: Path) -> list[Path]:
    """Every log file that belongs to this run and lost its report.

    Found by CONTENT, not by filename. A run directory is named after the job that created it, but
    a resumed run continues it under a different job id — the three NER arms each have two
    segments, and it is the SECOND one whose report failed. Every segment's log names its run
    directory in the "Run dir:" line, so matching on that finds all of them regardless of job id.
    """
    found = [run_dir / "run.log"] if (run_dir / "run.log").is_file() else []
    slurm = PROJ / "logs" / "slurm"
    if slurm.is_dir():
        for path in sorted(slurm.glob("*.out")):
            try:
                # Read the head only; "Run dir:" is printed in the first few lines.
                with open(path, encoding="utf-8", errors="replace") as handle:
                    head = "".join(next(handle, "") for _ in range(40))
            except OSError:
                continue
            if str(run_dir.resolve()) in head or run_dir.name in head:
                found.append(path)
    return found


def append_to_logs(run_dir: Path, report: str) -> list[Path]:
    """Append the regenerated report to each of this run's logs, unmistakably marked.

    Marked, and idempotent. These files are the run's primary record and someone reading one months
    from now must not mistake a reconstruction for what the job actually printed — the sections
    below were assembled from the checkpoint afterwards, not emitted live. Re-running the script
    replaces a previous appended block rather than stacking copies.
    """
    touched: list[Path] = []
    banner_start = f"\n{'=' * 78}\n=== {APPEND_BANNER} (appended post-hoc by "
    for path in _logs_for(run_dir):
        try:
            existing = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if APPEND_BANNER in existing:
            existing = existing.split(banner_start)[0].rstrip("\n") + "\n"
        block = (
            banner_start
            + "scripts/regenerate_run_report.py) ===\n"
            + "=== The report this run emitted live was cut short by a formatting bug\n"
            + "===   TypeError: unsupported format string passed to NoneType.__format__\n"
            + "=== in the DAG-traversal table (run.py applied a numeric format to a SKIPPED\n"
            + "=== iteration's score, which is None). The run itself COMPLETED normally; only\n"
            + "=== the report was lost. Everything below is rebuilt from checkpoint.json and is\n"
            + "=== NOT original job output.\n"
            + "=" * 78 + "\n\n"
            + report
        )
        try:
            path.write_text(existing + block, encoding="utf-8")
        except OSError:
            continue
        touched.append(path)
    return touched


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="*", help="run directories under logs/runs/")
    parser.add_argument("--all", action="store_true",
                        help="every run dir with a checkpoint but no final-report.txt")
    parser.add_argument("--quiet", action="store_true", help="write files, do not echo")
    parser.add_argument("--into-logs", action="store_true",
                        help="also append the report to run.log and this run's slurm logs")
    args = parser.parse_args(argv)

    targets = [Path(d) for d in args.run_dirs]
    if args.all:
        for d in sorted((PROJ / "logs" / "runs").iterdir()):
            if (d / "checkpoint.json").is_file() and not (d / "final-report.txt").is_file():
                targets.append(d)
    if not targets:
        parser.error("give at least one run directory, or --all")

    for run_dir in targets:
        path = regenerate(run_dir)
        print(f"wrote {path}")
        if args.into_logs:
            for touched in append_to_logs(run_dir, path.read_text(encoding="utf-8")):
                print(f"  appended to {touched}")
        if not args.quiet:
            print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
