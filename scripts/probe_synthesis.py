"""Synthesis + verifier audit per task, driven by the local teacher, with no orchestrator.

WHY THIS EXISTS
    Two of the six failure modes the suite has to be checked against live entirely inside the
    synthesis path — is the generated data legitimate, and do the verifier passes behave — and both
    are normally only observable partway through a multi-day agent run. As of 2026-09-08 those runs
    cannot start at all: every one dies at its first orchestrator decision with
    `FatalLLMError: Claude API call failed — 'Your credit balance is too low'`.

    Synthesis itself needs no orchestrator. `data.curriculum.synthesize_examples` takes a
    `generate_fn`, and the teacher behind it is the local vLLM server the pipeline stands up
    anyway. So the questions can be answered now, per task, instead of being rediscovered days into
    a seven-day run:

      * How many generated rows does the EXACT programmatic verifier reject, and for what?
      * How many would the TEACHER pass reject, and for what? That is the shadow measurement — it
        is recorded and NOT acted on, so a false-rejection cascade shows up as a number instead of
        as a run that quietly lost half its data.
      * Do the generated rows survive their own task's training turn and scorer?

    `goemotions` already has this data from run 39708679, which is where the teacher's
    JSON-wrapper false positives were first seen (38-72% would-reject, all of it formatting
    complaints about correctly-formatted rows). The other four tasks have none.

WHAT IT DELIBERATELY DOES NOT DO
    It does not train, score a model, or touch `stop_threshold`. It generates, verifies, and
    reports. Nothing it does can change a run's accuracy goal.

USAGE
    python scripts/probe_synthesis.py --tasks topv2 multiconer gec_bea19 dialogsum --n 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

# `config/config.py` reads `os.environ["ANTHROPIC_API_KEY"]` at MODULE SCOPE, so importing
# anything that touches config fails with a bare KeyError unless `.env` is loaded first — which
# `tests/pipeline/run.py` does and a standalone script otherwise would not. Run 39871824 died this
# way on all five tasks after loading their anchors successfully.
#
# The key only has to EXIST. This probe never calls Anthropic: synthesis and both verifier passes
# go to the local vLLM teacher, which is the whole reason the audit can run while the agent runs
# are blocked on that same account having no credits.
from dotenv import load_dotenv  # noqa: E402

# `override=True` matches `tests/pipeline/run.py`: without it a stale key already exported in the
# submitting shell shadows the real `.env` value (load_dotenv defaults to override=False) and the
# result is a 401 rather than a missing-key error, which is harder to read.
load_dotenv(os.path.join(_PROJ, ".env"), override=True)

DEFAULT_TASKS = ("topv2", "multiconer", "gec_bea19", "goemotions", "dialogsum")


def _log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def _anchors(task: str, n: int, log=_log) -> list[dict]:
    """Real gold rows to anchor generation on, exactly as `curate` would supply them."""
    from tasks import get_task

    spec = get_task(task)
    train, _eval = spec.load(max_train=max(n * 4, 200), max_test=1, log=lambda *a: None)
    for row in train:
        row.setdefault("_provenance", "train_anchor")
    log(f"  anchors: {len(train)} gold row(s)")
    return train


# Enough to read a pattern, few enough that the JSON stays openable. The audits generate 60.
_ROWS_KEPT = 60


def probe(task: str, n: int) -> dict:
    """Generate `n` rows for `task`, then report both verifier passes over them."""
    from data.curriculum import (
        VERIFY_MODE_SHADOW,
        _apply_verdicts,
        synthesize_examples,
    )
    from data.synth_client import get_generate_fn
    from tasks import get_task
    from tasks._builders import TrainingContext

    spec = get_task(task)
    result: dict = {"task": task, "requested": n}
    _log(f"── {task} ──")
    anchors = _anchors(task, n)
    # The same factory `agent/nodes/curate.py` uses, so the teacher, its sampling settings and its
    # cost accounting are identical to a real run's.
    generate_fn = get_generate_fn(log=_log)

    # The task's own exact verifier, run in ENFORCE mode inside `synthesize_examples` exactly as
    # the pipeline runs it. Passing it here is what makes the reject count below real rather than
    # a separate after-the-fact opinion.
    lines: list[str] = []
    rows = synthesize_examples(
        anchors,
        task=task,
        n=n,
        generate_fn=generate_fn,
        verify_fn=spec.synth_verifier,
        log=lines.append,
        label_definitions=dict(spec.label_definitions) or None,
    )
    result["generated"] = len(rows)
    result["synthesis_log"] = lines[-40:]
    # The ROWS, not just the counts. "Inspect generated rows" is one of the six things these runs
    # are audited for, and a rate cannot answer it: topv2's 2026-09-09 audit reported 14 identical
    # "should be CREATE_REMINDER, not UPDATE_REMINDER" verdicts, and deciding whether those were
    # synthesis mislabels or teacher false positives needs the command each one judged. Both
    # readings fit the number equally well, and they call for opposite fixes.
    result["rows"] = [
        {k: v for k, v in row.items() if not str(k).startswith("_")}
        for row in rows[:_ROWS_KEPT]
    ]
    _log(f"  generated {len(rows)} row(s) that passed the exact verifier")
    if not rows:
        result["note"] = ("no rows survived generation + the exact verifier; see synthesis_log. "
                          "That is itself the finding.")
        return result

    # Did the survivors keep their shape? A row that cannot become a training turn, or that its
    # own verifier now rejects, is a defect the pipeline would carry into training.
    ctx = TrainingContext(labels=(), instruction="")
    turn_errors, verifier_rejects, reasons = 0, 0, Counter()
    for row in rows:
        try:
            prompt, target, _marker = spec.build_training_turn(row, ctx)
            if not str(prompt).strip() or not str(target).strip():
                turn_errors += 1
        except Exception:  # noqa: BLE001 — the point is to count these, not to raise
            turn_errors += 1
        if spec.synth_verifier is not None:
            ok, why = spec.synth_verifier.checker(row)
            if not ok:
                verifier_rejects += 1
                reasons[str(why)[:70]] += 1
    result["training_turn_errors"] = turn_errors
    result["post_hoc_verifier_rejects"] = verifier_rejects
    result["post_hoc_verifier_reasons"] = dict(reasons)

    # THE SHADOW MEASUREMENT COMES FROM `synthesize_examples` ITSELF, not from a second pass.
    #
    # Run 39873026 got this wrong and the mistake is worth recording. This probe used to call
    # `verify_generated_answers` again afterwards with SLM_VERIFY_SYNTH set only around that call
    # — so the teacher pass INSIDE `synthesize_examples` ran in ENFORCE mode and dropped rows
    # (topv2: "teacher validated 32/57; rejected 25"), and the outer call then shadow-judged the
    # 32 survivors. Two passes, one of them binding, and neither matching what a real run does.
    #
    # The task launchers export `SLM_VERIFY_SYNTH=shadow` for the whole process, so the internal
    # pass is the shadow pass and logs `[verify:shadow] would have rejected N/M`. Parsing that is
    # both simpler and the only faithful reading.
    shadow_lines = [l for l in lines if "verify:shadow" in l]
    exact_lines = [l for l in lines if "verify:exact" in l]
    result["shadow_log"] = [l.strip() for l in shadow_lines]
    result["exact_log"] = [l.strip() for l in exact_lines]
    result["mode"] = os.environ.get("SLM_VERIFY_SYNTH", "<unset>")
    if not shadow_lines:
        result["warning"] = (
            f"no [verify:shadow] line was logged (SLM_VERIFY_SYNTH={result['mode']!r}). The "
            "teacher pass either did not run or ran in ENFORCE mode, in which case `generated` "
            "is a post-rejection count and the shadow rate was not measured."
        )
    for line in shadow_lines + exact_lines:
        _log("  " + line.strip()[:170])
    return result


def _gold_field_for(spec) -> str:
    from data.curriculum import _gold_field

    return _gold_field(spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--n", type=int, default=60, help="rows to generate per task")
    parser.add_argument("--out", default="logs/probes/synthesis-audit.json")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HOME", "/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache")
    results = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_per_task": args.n,
               "tasks": []}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    for task in args.tasks:
        try:
            cell = probe(task, args.n)
        except Exception as exc:  # noqa: BLE001 — one task's failure must not hide the others
            import traceback
            cell = {"task": task, "fatal": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
            _log(f"  FATAL {type(exc).__name__}: {exc}")
        results["tasks"].append(cell)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=str)

    _log("=== SUMMARY ===")
    _log(f"  {'task':12s} {'gen':>5s} {'turn_err':>9s} {'verif_rej':>10s} {'shadow_kept':>12s}")
    for cell in results["tasks"]:
        if cell.get("fatal"):
            _log(f"  {cell['task']:12s} FATAL: {cell['fatal'][:60]}")
            continue
        _log(f"  {cell['task']:12s} {cell.get('generated', 0):>5d} "
             f"{cell.get('training_turn_errors', 0):>9d} "
             f"{cell.get('post_hoc_verifier_rejects', 0):>10d} "
             f"{cell.get('shadow_kept', 0):>12d}")
    _log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
