"""Health verdict for in-flight pipeline runs (2026-08-25).

WHY A PARSER RATHER THAN `tail`
    Watching a run means answering specific questions, and none of them is answerable from the last
    twenty lines: did the score move by a plausible amount, is the model failing on FORM or on
    CONTENT, did the orchestrator's intervention target the failures that actually cost points, is
    the judge deciding or abstaining, is the curriculum growing. Each of those is a number that
    appears once per iteration, thousands of lines apart.

    So this reads the whole log and prints one block per run with the things that would make a
    human stop the run, plus explicit ALERTs for the patterns that have cost runs before:

      score 0.0000 with format_valid ~0     the model never produced a readable answer — a prompt or
                                            chat-template problem, not a data problem (B290).
      score 0.0000 with format_valid ~1     it answers in the right shape and is wrong, which is a
                                            data problem. The pair is the diagnosis; either alone
                                            is ambiguous.
      a jump larger than JUMP_ALERT         on an all-or-nothing metric a single convention can be
                                            worth the whole score (calendar_json swung 0.0019 →
                                            0.4598 on byte-identical data), so a large jump is a
                                            prompt for suspicion rather than celebration.
      judge_unsure above UNSURE_ALERT       the metric is measuring the judge's uncertainty.
      rebuild added 0 rows                  the intervention did nothing; several in a row is how a
                                            run burns hours going nowhere.
      identical consecutive scores          usually a curriculum that did not change.

USAGE
    python scripts/monitor_runs.py                 # every slm-*-l40s log with a live job
    python scripts/monitor_runs.py 38832586 ...    # specific job ids
    python scripts/monitor_runs.py --all           # include finished runs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

PROJ = "/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory"

# A single-iteration move larger than this is not necessarily wrong, but it is never uninteresting.
JUMP_ALERT = 0.25
# Above this share of queries the judge declined to decide, the score is about the judge.
UNSURE_ALERT = 0.15

_SCORE = re.compile(
    r"Score: (?P<metric>\w+)=(?P<score>[0-9.]+)\s+format_valid=(?P<fv>[0-9.]+)"
    r".*?failures=(?P<failures>\d+)/(?P<total>\d+)"
)
_BASELINE = re.compile(r"Baseline (?P<metric>\w+)=(?P<score>[0-9.]+)\s+format_valid=(?P<fv>[0-9.]+)")
_ITERATION = re.compile(r"ITERATION (\d+) EVAL")
_PINNED = re.compile(r"PINNED to (\S+)")
# `smallest_first` announces its pick differently from `single_model`, and both have to be read or
# the monitor reports "(not selected yet)" on a run that chose a model half an hour ago — which is
# worse than printing nothing, because it looks like a stalled selection step.
_SELECTED = re.compile(r"\[model_selection:\w+\] Selected (\S+ \[\S+\])")
_ORCH_REASON = re.compile(r"\[model_selection:\w+\]\s+reason: (.+)")
_TOP_FAILURES = re.compile(r"top failures: (.+)")
_REBUILD = re.compile(r"DATA REBUILD: (\S+) — asking for (\d+) new row\(s\)")
_CURRICULUM = re.compile(r"CURRICULUM: (\d+) → (\d+) row\(s\) \(([+-]\d+) added")
_ADDED_ZERO = re.compile(r"added 0 new rows")
_PERCLASS = re.compile(r"(judge_unsure_rate|undeclared_api_rate|judged_rows)[=: ]+([0-9.]+)")
_GOAL = re.compile(r"goal (?P<goal>[0-9.]+)")
_TERMINAL = re.compile(r"(RUN COMPLETE|RUN FAILED|CONVERGED|TERMINAT\w*|STOPPING)")
_ERRORS = re.compile(
    r"(Traceback|OutOfMemoryError|CudaWorkerError|BadRequestError|JudgeInfrastructureError"
    r"|RUN FAILED|GGML_ASSERT|exit=-6)"
)


def squeue_state(job_id: str) -> str:
    try:
        out = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T %M %R"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
    return out or "not in queue"


def worker_ops(run_dir: str) -> list[tuple[str, float, str]]:
    path = os.path.join(run_dir, "timing-events.jsonl")
    ops: list[tuple[str, float, str]] = []
    if not os.path.exists(path):
        return ops
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("kind") == "worker_op":
                ops.append((
                    str(event.get("name")),
                    float(event.get("duration_ms", 0)) / 60000.0,
                    str(event.get("status")),
                ))
    return ops


def analyse(log_path: str, show_all: bool) -> None:
    job_id = re.sub(r"\D", "", os.path.basename(log_path).rsplit("-", 1)[-1])
    state = squeue_state(job_id)
    if state == "not in queue" and not show_all:
        return
    with open(log_path, encoding="utf-8", errors="ignore") as handle:
        text = handle.read()

    print("=" * 78)
    print(f"{os.path.basename(log_path)}")
    print(f"  job {job_id}  state: {state}")
    mtime = os.path.getmtime(log_path)
    import time as _time

    print(f"  log silent for {(_time.time() - mtime) / 60:.0f} min  "
          f"({os.path.getsize(text and log_path) / 1e6:.1f} MB)")

    model = None
    for pattern in (_SELECTED, _PINNED):
        found = pattern.findall(text)
        if found:
            model = found[-1]
            break
    goal = _GOAL.findall(text)
    print(f"  model: {model or '(not selected yet)'}    goal: {goal[-1] if goal else '?'}")
    reason = _ORCH_REASON.findall(text)
    if reason:
        print(f"  orchestrator reason: {reason[-1][:120]}")
    fitness = re.findall(
        r"\[teacher\]\s+(\d+)-shot: (\w+)=([0-9.]+) \(format_valid=([0-9.]+)\)", text
    )
    if fitness:
        print("  teacher fitness: " + "  ".join(
            f"{shots}-shot {metric}={score}" for shots, metric, score, _fv in fitness
        ))

    run_dir = os.path.join(PROJ, "logs", "runs",
                           os.path.basename(log_path)[:-4].replace("slm-", "slm-", 1))
    candidates = glob.glob(os.path.join(PROJ, "logs", "runs", f"*{job_id}"))
    if candidates:
        run_dir = candidates[0]
    ops = worker_ops(run_dir)
    if ops:
        totals: dict[str, list[float]] = {}
        for name, minutes, _status in ops:
            totals.setdefault(name, []).append(minutes)
        summary = "  ".join(
            f"{name}x{len(v)} {sum(v) / len(v):.0f}min avg" for name, v in sorted(totals.items())
        )
        print(f"  worker ops: {summary}")
        failed = [(n, s) for n, _m, s in ops if s != "success"]
        if failed:
            print(f"  ALERT worker ops not successful: {failed}")

    # ---- score trajectory ----
    scores: list[tuple[str, float, float, str]] = []
    for match in _BASELINE.finditer(text):
        scores.append(("baseline", float(match["score"]), float(match["fv"]), ""))
    for iteration, match in zip(_ITERATION.findall(text), _SCORE.finditer(text)):
        scores.append((f"iter {iteration}", float(match["score"]), float(match["fv"]),
                       f"{match['failures']}/{match['total']}"))
    if not scores:
        print("  no eval yet")
    else:
        print("  scores:")
        previous = None
        for label, score, fv, failures in scores:
            delta = "" if previous is None else f"  Δ={score - previous:+.4f}"
            print(f"    {label:9s} score={score:.4f}  format_valid={fv:.4f}  "
                  f"failures={failures}{delta}")
            if score == 0.0 and fv < 0.1:
                print("      ALERT score 0 AND format_valid ~0 → FORMAT problem "
                      "(prompt / chat template), not data")
            elif score == 0.0 and fv > 0.9:
                print("      ALERT score 0 with format_valid ~1 → CONTENT problem; "
                      "output is well-formed and wrong")
            if previous is not None and abs(score - previous) >= JUMP_ALERT:
                print(f"      ALERT jump of {score - previous:+.4f} in one iteration — on an "
                      "all-or-nothing metric one convention can be worth the whole score; verify")
            if previous is not None and score == previous and label != "baseline":
                print("      ALERT identical to the previous score — curriculum probably unchanged")
            previous = score

    # ---- judge / fabrication rates ----
    rates = {}
    for name, value in _PERCLASS.findall(text):
        rates[name] = float(value)
    if rates:
        print("  rates: " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(rates.items())))
        if rates.get("judge_unsure_rate", 0) > UNSURE_ALERT:
            print(f"      ALERT judge_unsure_rate {rates['judge_unsure_rate']:.3f} — the metric is "
                  "measuring the judge's uncertainty rather than the model")

    # ---- failures and interventions ----
    failures = _TOP_FAILURES.findall(text)
    if failures:
        print(f"  top failures (latest): {failures[-1][:150]}")
    rebuilds = _REBUILD.findall(text)
    if rebuilds:
        print(f"  interventions: {[f'{s}({n})' for s, n in rebuilds][-6:]}")
    growth = _CURRICULUM.findall(text)
    if growth:
        print(f"  curriculum: {[f'{a}→{b}({d})' for a, b, d in growth][-6:]}")
    empties = len(_ADDED_ZERO.findall(text))
    if empties:
        print(f"      ALERT {empties} rebuild(s) added 0 rows")

    # ---- errors / termination ----
    errors = _ERRORS.findall(text)
    if errors:
        counts: dict[str, int] = {}
        for error in errors:
            counts[error] = counts.get(error, 0) + 1
        print(f"      ALERT errors seen: {counts}")
    terminal = _TERMINAL.findall(text)
    if terminal:
        print(f"  terminal markers: {sorted(set(terminal))}")
    if "SLM_TEACHER_SYNTH_BYPASS=1" in text:
        print("  synthesis gate: BYPASSED (operator override, as intended)")
    elif "synthetic data is REFUSED" in text:
        print("      ALERT synthesis REFUSED and no bypass line — the gate is on")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="*", help="job ids; default is every recent l40s log")
    parser.add_argument("--all", action="store_true", help="include runs no longer in the queue")
    args = parser.parse_args()

    if args.jobs:
        paths = [p for job in args.jobs
                 for p in glob.glob(os.path.join(PROJ, "logs", "slurm", f"*{job}.out"))]
    else:
        # Both accounts. The default used to match `slm-*-l40s-*.out` only, which silently skipped
        # every run submitted to the CSE quota — and which quota a task lands on is decided by
        # whichever has a free GPU, so half the suite was invisible to the monitor at random.
        paths = sorted(
            glob.glob(os.path.join(PROJ, "logs", "slurm", "slm-*-l40s-*.out"))
            + glob.glob(os.path.join(PROJ, "logs", "slurm", "slm-*-cse-*.out")),
            key=os.path.getmtime, reverse=True,
        )[:12]
    if not paths:
        print("no matching logs")
        return
    for path in paths:
        try:
            analyse(path, args.all or bool(args.jobs))
        except Exception as error:  # noqa: BLE001 — a monitor must not die on one bad log
            print(f"  (could not analyse {os.path.basename(path)}: "
                  f"{type(error).__name__}: {error})")
    print("=" * 78)
    subprocess.run(["squeue", "-u", os.environ.get("USER", "evanly"),
                    "-o", "%.10i %.30j %.8T %.10M %.20R"], check=False)


if __name__ == "__main__":
    sys.exit(main())
