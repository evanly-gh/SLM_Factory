#!/usr/bin/env python3
"""
run_fallback_agent_gguf.py — Stage 2 of the MNN-first, GGUF-fallback pipeline.

Reads needs_gguf_fallback.json (written by Stage 1, run_fallback_agent_mnn.py)
and, for each flagged model, ensures a local GGUF conversion exists, then
runs run_autobench.py as a genuine TOP-LEVEL subprocess covering ALL of that
model's flagged remaining questions in ONE invocation (not one subprocess
call per question).

THIS SCRIPT ITSELF MUST BE LAUNCHED AS A FRESH, TOP-LEVEL PYTHON PROCESS -
NOT IMPORTED OR CALLED FROM WITHIN ANOTHER PYTHON PROCESS. That's the whole
point of the two-stage split: an earlier single-process design ran
run_autobench.py as a subprocess NESTED inside a long-running Python
process (the old run_fallback_agent.py), and that nesting had a confirmed,
unresolved hang that survived 7 separate fix attempts across fundamentally
different mechanisms - all failed identically, strongly indicating a
structural incompatibility with the nesting itself, not anything fixable at
the invocation level. Since THIS script is meant to be launched fresh by
run_pipeline.sh (a shell wrapper, not a parent Python process) after Stage 1
has fully exited, run_autobench.py's invocation here is no longer nested at
all - so a plain subprocess.run() (no capture-avoidance tricks, no
os.system(), no polling workarounds) is expected to just work, matching
every one of the confirmed-working manual, direct-in-terminal invocations
throughout this debugging saga.

Every flagged question is checked in order, via is_garbage() - not just the
first one (the exact question that originally failed on MNN in Stage 1).
The moment ANY question comes back garbage, first or later, processing
stops immediately: no further flagged questions for that model are
attempted, the model is marked "failed" in this stage's summary, and only
the questions that genuinely passed the garbage check BEFORE the failure
point are recorded into gguf_results.json (Stage 1's mnn_results.json,
produced separately, already holds whatever succeeded on MNN before its own
failure point - untouched by this script). This mirrors Stage 1's own
stop-on-first-garbage rule exactly, just continuing from wherever Stage 1
left off. Only if EVERY flagged question passes is the model marked
"success", with every one of its results (scored via score_accuracy(),
display-annotated via extract_final_answer()) recorded.

Reuses existing, proven infrastructure by importing response_quality.py by
path; run_autobench.py itself is invoked as a subprocess (its own full
main() flow - download/convert/push already handled by ensure_gguf_
converted() below, force-restart+settle, batch broadcast+poll over every
flagged question at once), not imported as a module.

Usage (standalone - only meaningful after Stage 1 has produced
needs_gguf_fallback.json):
    python3 run_fallback_agent_gguf.py --fallback-file needs_gguf_fallback.json

Usage (full pipeline, both stages):
    ./run_pipeline.sh --fit-report model_fit_report_q4q8.json --questions eval_questions_phase3.json
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

RESPONSE_QUALITY_SCRIPT = SCRIPT_DIR / "response_quality.py"

# Only needed by --full-run mode, to reuse its load_fit_report()/
# load_eval_questions() exactly rather than reimplementing that parsing here
# - same cross-file-by-path reuse pattern already used throughout this
# project (e.g. run_sanity_check.py).
RUN_FALLBACK_AGENT_MNN_SCRIPT = SCRIPT_DIR / "run_fallback_agent_mnn.py"

# The GGUF/SmolChat side lives in a sibling worktree, not this repo - same
# cross-worktree reuse pattern already used elsewhere in this project.
SMOLCHAT_WORKTREE = Path.home() / "SLM_Factory-SmolChat"
RUN_AUTOBENCH_GGUF_SCRIPT = SMOLCHAT_WORKTREE / "Benchmark-Harness" / "run_autobench.py"
CONVERT_TO_GGUF_SCRIPT = SMOLCHAT_WORKTREE / "Model-Conversion" / "convert_to_gguf.py"

REQUIRED_SCRIPTS = [
    (RESPONSE_QUALITY_SCRIPT, "response_quality.py"),
    (RUN_AUTOBENCH_GGUF_SCRIPT, "run_autobench.py (SmolChat, sibling worktree)"),
    (CONVERT_TO_GGUF_SCRIPT, "convert_to_gguf.py (SmolChat, sibling worktree)"),
]

# Extra time budgeted (beyond N questions * per-question --timeout) for a
# flagged model's ONE run_autobench.py invocation, covering the one-time
# app force-restart+settle at the start of that run - conversion/push
# themselves are already done ahead of time by ensure_gguf_converted().
SETUP_OVERHEAD_SECONDS = 180

# Per question, not per model: NOT a "retry until success" mechanism -
# every question ALWAYS runs exactly TOTAL_ATTEMPTS_PER_QUESTION total
# attempts (the initial batch call, plus WARMUP_ATTEMPTS - 1 more
# single-question run_autobench.py invocations, each paying the
# app-restart+settle overhead again). The first WARMUP_ATTEMPTS attempts
# are unconditionally discarded regardless of whether they pass or fail
# is_garbage() - only the FINAL attempt's result is ever recorded, even if
# it's also garbage. is_garbage() itself is unchanged - this only wraps
# additional attempts around it and decides which one gets kept.
WARMUP_ATTEMPTS = 5
TOTAL_ATTEMPTS_PER_QUESTION = WARMUP_ATTEMPTS + 1

_SUBPROCESS_OUTPUT_TAIL_CHARS = 2000


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tail(text, n=_SUBPROCESS_OUTPUT_TAIL_CHARS):
    text = text or ""
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def load_fallback_queue(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data.get("models", [])


def shorten(text, n=150):
    if not text:
        return text
    return text if len(text) <= n else text[:n] + "..."


def _fmt(v, unit="", nd=1):
    return f"{v:.{nd}f}{unit}" if isinstance(v, (int, float)) else "N/A"


def cooldown_between_models(minutes: float, index: int, total: int) -> None:
    """Real time.sleep() pause between one model+quant's run and the next,
    letting the phone's thermal state return to baseline before starting the
    next benchmark - mirrors run_fallback_agent_mnn.py's own
    cooldown_between_models() exactly. Skipped after the LAST model (nothing
    left to cool down for) and entirely when minutes <= 0
    (--cooldown-minutes 0, for quick tests where thermal accuracy doesn't matter).
    """
    if minutes <= 0 or index >= total:
        return
    print(f"\n[COOLDOWN] Waiting {minutes:g} minute(s) before next model...")
    time.sleep(minutes * 60)


def print_question_result(qlabel: str, metrics: dict, is_correct, response) -> None:
    """Standardized multi-line per-question result block, matching
    run_fallback_agent_mnn.py's print_question_result() exactly:

        [1/1] Qwen/Qwen3-1.7B [Q4_K_M] Q1/2 [gguf]
          ColdLoad=643ms  TTFT=4745.0ms  PrefillTPS=N/A  DecodeTPS=7.1
          RSS=1691596KB  Power=345.60mA (BatteryMgr) / N/A (Monsoon)
          ThermalCPU=57.1°C  ThermalSkin=42.3°C  Correct=False
          Response: "<truncated to ~150 chars>..."

    cold_load_ms and power_ma_monsoon are both confirmed real, at
    raw_result["metrics"]["cold_load_ms"] / ["power_ma_monsoon"] (verified
    directly against a saved run_autobench.py output,
    old_test_results/monsoon_integration_test.json). PrefillTPS is always
    N/A - GGUF/SmolChat only ever reports one combined "tps" (mapped to
    decode_tps above), never a separate prefill figure - shown explicitly
    via _fmt() rather than omitted, so the line's shape stays comparable
    to MNN's own output.
    """
    print(qlabel)
    print(
        f"  ColdLoad={_fmt(metrics.get('cold_load_ms'), 'ms', 0)}  TTFT={_fmt(metrics.get('ttft_ms'), 'ms')}  "
        f"PrefillTPS={_fmt(metrics.get('prefill_tps'))}  DecodeTPS={_fmt(metrics.get('decode_tps'))}"
    )
    print(
        f"  RSS={_fmt(metrics.get('peak_rss_kb'), 'KB', 0)}  "
        f"Power={_fmt(metrics.get('power_ma'), 'mA', 2)} (BatteryMgr) / "
        f"{_fmt(metrics.get('power_ma_monsoon'), 'mA', 2)} (Monsoon)"
    )
    print(
        f"  ThermalCPU={_fmt(metrics.get('thermal_cpu_c'), '°C')}  ThermalSkin={_fmt(metrics.get('thermal_skin_c'), '°C')}  "
        f"Correct={is_correct}"
    )
    print(f'  Response: "{shorten(response)}"')


# ---------------------------------------------------------------------------
# GGUF conversion (reused/duplicated from the old single-process agent -
# these two scripts are meant to be independently launchable, so this
# helper is self-contained here rather than imported cross-file)
# ---------------------------------------------------------------------------

def gguf_model_prefix(model_id: str) -> str:
    """Mirror convert_to_gguf.py's own model_prefix() exactly, for the HF-ID
    case (the only case this pipeline ever passes in - model_id is always a
    HuggingFace ID, never a local path)."""
    return model_id.split("/")[-1].lower().replace("_", "-").replace(" ", "-")


def gguf_local_path(model_id: str, quant: str) -> Path:
    """The local .gguf file path convert_to_gguf.py produces for this
    model_id/quant, using its own naming convention exactly:
    <SmolChat-worktree>/Model-Conversion/output-<prefix>/<prefix>-<quant-lower>.gguf
    """
    prefix = gguf_model_prefix(model_id)
    output_dir = CONVERT_TO_GGUF_SCRIPT.parent / f"output-{prefix}"
    return output_dir / f"{prefix}-{quant.lower()}.gguf"


def ensure_gguf_converted(model_id: str, quant: str, label: str) -> dict:
    """Guarantee a local, already-converted .gguf file exists for this
    model_id/quant, converting it directly via convert_to_gguf.py (WITHOUT
    --deploy - this only needs to produce the local file, not push it) if
    it doesn't already exist.

    Returns {"ok": True, "local_path": Path} or {"ok": False, "error": str}.
    """
    local_path = gguf_local_path(model_id, quant)
    if local_path.exists():
        print(f"{label} [gguf-convert] - reusing existing local GGUF at {local_path}")
        return {"ok": True, "local_path": local_path}

    cmd = [
        sys.executable, str(CONVERT_TO_GGUF_SCRIPT),
        "--model", model_id,
        "--output", str(local_path.parent),
        "--quant", quant,
    ]
    print(f"{label} [gguf-convert] - RUN: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"ok": False, "error": f"convert_to_gguf.py exited {result.returncode}. "
                                       f"stdout: {_tail(result.stdout)} stderr: {_tail(result.stderr)}"}

    if not local_path.exists():
        return {"ok": False, "error": f"convert_to_gguf.py exited 0 but expected output file is missing: "
                                       f"{local_path}. stdout tail: {_tail(result.stdout)}"}

    return {"ok": True, "local_path": local_path}


# ---------------------------------------------------------------------------
# GGUF: run ALL of a model's flagged questions in ONE run_autobench.py call
# ---------------------------------------------------------------------------

def run_gguf_batch(local_path: Path, quant: str, questions: list, timeout: int, no_think: bool, label: str) -> dict:
    """Invoke run_autobench.py ONCE, covering exactly the given questions -
    either the full flagged-question batch (every question's first
    attempt), or a single question (one of its warmup or final attempts).
    Same subprocess-invocation logic either way. Returns
    {"ok": True, "results_list": [...]} (one entry per question, same
    order as `questions`) or {"ok": False, "error": "..."}.
    """
    with tempfile.TemporaryDirectory(prefix="fallback_gguf_stage2_") as tmpdir:
        tmpdir = Path(tmpdir)
        questions_path = tmpdir / "questions.txt"
        output_path = tmpdir / "output.json"

        lines = []
        for q in questions:
            text = q["text"]
            lines.append(f"{text} /no_think" if no_think else text)
        questions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        overall_timeout = timeout * len(questions) + SETUP_OVERHEAD_SECONDS
        # "--quiet" is always baked in here, not a user-facing flag on this
        # script - run_autobench.py's own routine per-question/pre-flight
        # output would otherwise duplicate the reporting this script already
        # does after parsing its output JSON. [ERROR]/[WARN] lines and the
        # final "Results saved"/"DONE" confirmation still always print.
        cmd = [
            sys.executable, "-u", str(RUN_AUTOBENCH_GGUF_SCRIPT),
            "--model", str(local_path),
            "--quant", quant,
            "--questions", str(questions_path),
            "--output", str(output_path),
            "--timeout", str(timeout),
            "--quiet",
        ]
        print(f"{label} - RUN (timeout={overall_timeout}s): {' '.join(cmd)}")

        # Plain subprocess.run(), no capture-avoidance/os.system()/polling
        # tricks - this script is a fresh top-level process (launched by
        # run_pipeline.sh, not nested inside another Python process), which
        # was the actual root cause identified after 7 failed fix attempts
        # at the invocation-mechanism level alone.
        try:
            proc = subprocess.run(cmd, timeout=overall_timeout)
        except subprocess.TimeoutExpired:
            print(f"{label} - FAILED: run_autobench.py did not finish within {overall_timeout}s")
            return {"ok": False, "error": f"run_autobench.py timed out after {overall_timeout}s"}

        if not output_path.exists():
            print(f"{label} - FAILED: run_autobench.py exited (code={proc.returncode}) but {output_path} was not written")
            return {"ok": False, "error": f"run_autobench.py exited (code={proc.returncode}) but produced no output file"}

        try:
            data = json.loads(output_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"{label} - FAILED: could not parse {output_path}: {exc}")
            return {"ok": False, "error": f"failed to parse output JSON: {exc}"}

    results_list = data.get("results", [])
    if not results_list:
        print(f"{label} - FAILED: run_autobench.py produced no per-question results (run_info={data.get('run_info')})")
        return {"ok": False, "error": f"no per-question results (run_info={data.get('run_info')})"}

    return {"ok": True, "results_list": results_list}


def extract_response_and_metrics(quality, raw_result):
    """From one raw run_autobench.py per-question result, return
    (response, metrics, run_id, garbage_flag). A missing/non-success
    raw_result is treated as garbage - exactly the existing rule, just
    extracted so it can be reused for both the first attempt and retries.
    is_garbage() itself is called exactly as before, unmodified.
    """
    if raw_result is None or raw_result.get("status") != "success":
        run_id = raw_result.get("run_id") if raw_result else None
        return None, {}, run_id, True

    raw_metrics = raw_result.get("metrics") or {}
    # Renamed to match MNN's own metrics field names for consistency
    # across mnn_results.json/gguf_results.json - SmolChat/GGUF only
    # ever reports one combined "tps" (not separate prefill/decode).
    metrics = {
        "cold_load_ms": raw_metrics.get("cold_load_ms"),
        "ttft_ms": raw_metrics.get("ttft_ms"),
        "decode_tps": raw_metrics.get("tps"),
        "peak_rss_kb": raw_metrics.get("memory_kb"),
        "power_ma": raw_metrics.get("power_ma"),
        "power_ma_monsoon": raw_metrics.get("power_ma_monsoon"),
        "thermal_cpu_c": raw_metrics.get("thermal_temp_cpu_c"),
        "thermal_skin_c": raw_metrics.get("thermal_temp_skin_c"),
    }
    response = raw_result.get("response")
    run_id = raw_result.get("run_id")
    garbage_flag = quality.is_garbage(response, metrics)
    return response, metrics, run_id, garbage_flag


def build_result_entry(quality, model_id, quant, question_number, question_text, reference_answer,
                        response, is_correct, is_garbage_flag, metrics, run_id) -> dict:
    return {
        "model_id": model_id,
        "quant": quant,
        "engine": "gguf",
        "question_number": question_number,
        "question": question_text,
        "reference_answer": reference_answer,
        "response": response,
        "extracted_answer": quality.extract_final_answer(response),
        "is_correct": is_correct,
        "is_garbage": is_garbage_flag,
        "metrics": metrics,
        "run_id": run_id,
    }


def process_flagged_model(quality, model_entry: dict, index: int, total: int, timeout: int,
                           no_think: bool, max_tokens) -> dict:
    """Returns {"model_id", "quant", "status" ("success"|"failed"), "error",
    "gguf_results": [...]}."""
    model_id = model_entry["model_id"]
    quant = model_entry["quant"]
    flagged_questions = model_entry["questions"]
    label = f"[{index}/{total}] {model_id} [{quant}]"

    result = {"model_id": model_id, "quant": quant, "status": None, "error": None, "gguf_results": []}

    if max_tokens is not None:
        print(f"{label} [NOTE] --max-tokens has no effect here - SmolChat's broadcast protocol has no max_tokens extra.")

    print(f"\n{'=' * 70}\n{label} ({len(flagged_questions)} flagged question(s))\n{'=' * 70}")

    conv = ensure_gguf_converted(model_id, quant, label)
    if not conv["ok"]:
        print(f"{label} - FAILED (GGUF conversion): {conv['error']}")
        result["status"] = "failed"
        result["error"] = f"GGUF conversion failed: {conv['error']}"
        return result

    batch = run_gguf_batch(conv["local_path"], quant, flagged_questions, timeout, no_think, label)
    if not batch["ok"]:
        result["status"] = "failed"
        result["error"] = batch["error"]
        return result
    results_list = batch["results_list"]

    # Check EVERY flagged question in order, not just the first. The moment
    # ANY question comes back garbage - first or later, and after exhausting
    # its own retries - stop immediately: no further questions after it are
    # attempted, the model is marked "failed", and only the entries recorded
    # BEFORE the garbage question are kept. This mirrors Stage 1's own
    # stop-on-first-garbage rule exactly, just continuing from wherever
    # Stage 1 left off - a later garbage question doesn't get a pass just
    # because an earlier one in this same batch happened to succeed.
    entries = []
    garbage_question_number = None
    total_q = len(flagged_questions)
    for idx, q in enumerate(flagged_questions):
        question_number = q["question_number"]
        reference_answer = q["answer"]
        # question_number is the ORIGINAL question's number from the full
        # question set (this batch is a filtered subset, not necessarily
        # 1..total_q sequentially) - only the /total_q denominator was
        # missing, so it's kept as the numerator rather than reindexed to
        # match MNN's always-sequential 1..N (that would break the "GARBAGE
        # ALSO on Q{garbage_question_number}" messaging below, which relies
        # on this being the true original number).
        qlabel = f"{label} Q{question_number}/{total_q} [gguf]"

        raw_result = results_list[idx] if idx < len(results_list) else None
        response, metrics, run_id, garbage_flag = extract_response_and_metrics(quality, raw_result)

        # Prints the FULL metrics+response block for every attempt,
        # immediately after that attempt completes - not just a one-line
        # summary for the discarded ones - so warmup attempts are just as
        # inspectable as the final, recorded one. Returns that attempt's own
        # is_correct (computed fresh per attempt, not reused across
        # attempts) so the caller can track the final attempt's value
        # without re-printing or re-scoring it.
        def print_attempt_block(attempt_num, response, metrics):
            label_suffix = "discarded" if attempt_num < TOTAL_ATTEMPTS_PER_QUESTION else "final, recorded"
            attempt_label = f"{qlabel} attempt {attempt_num}/{TOTAL_ATTEMPTS_PER_QUESTION} ({label_suffix})"
            attempt_is_correct = quality.score_accuracy(response, reference_answer)
            print_question_result(attempt_label, metrics, attempt_is_correct, response)
            return attempt_is_correct

        # ALWAYS run exactly TOTAL_ATTEMPTS_PER_QUESTION total attempts -
        # NOT "retry until success". The batch call above was attempt 1 for
        # every question (a warmup attempt like all the others). The first
        # WARMUP_ATTEMPTS attempts are unconditionally discarded regardless
        # of whether they pass or fail is_garbage() - is_garbage() itself is
        # still called on every attempt, unchanged, but its result only
        # matters for the FINAL attempt. Only that final attempt's
        # response/metrics/run_id/garbage_flag survive this loop and get
        # recorded below, even if it's also garbage.
        attempt = 1
        is_correct = print_attempt_block(attempt, response, metrics)
        while attempt < TOTAL_ATTEMPTS_PER_QUESTION:
            attempt += 1
            retry_batch = run_gguf_batch(conv["local_path"], quant, [q], timeout, no_think, qlabel)
            if retry_batch["ok"] and retry_batch["results_list"]:
                retry_raw_result = retry_batch["results_list"][0]
            else:
                retry_raw_result = None  # counts as garbage, same as a missing/non-success result
            response, metrics, run_id, garbage_flag = extract_response_and_metrics(quality, retry_raw_result)
            is_correct = print_attempt_block(attempt, response, metrics)

        # Record the final attempt's result unconditionally - no early
        # stopping, no conditional acceptance, even if it's garbage. Already
        # printed above (the "final, recorded" block), so not printed again.
        entry = build_result_entry(quality, model_id, quant, question_number, q["text"], reference_answer,
                                    response, is_correct, garbage_flag, metrics, run_id)
        entries.append(entry)

        if garbage_flag:
            print(f"{qlabel} -> final attempt is GARBAGE - recorded anyway (no retry-until-success), "
                  f"but still stopping this model's batch here, unchanged.")
            garbage_question_number = question_number
            break  # stop immediately - no further flagged questions attempted

    if garbage_question_number is not None:
        first_was_the_failure = garbage_question_number == flagged_questions[0]["question_number"]
        origin = "the question that originally failed on MNN" if first_was_the_failure else "a later question in this batch"
        # entries includes the just-recorded garbage question itself now
        # (recorded unconditionally, per the fixed-attempts design above) -
        # so the count below is split to avoid implying all of it succeeded.
        successful_before_it = len(entries) - 1
        print(
            f"{label} - GGUF ALSO garbage on Q{garbage_question_number} ({origin}) after {TOTAL_ATTEMPTS_PER_QUESTION} "
            f"total attempts - marking model FAILED, stopping (keeping {successful_before_it} genuinely-successful "
            f"GGUF question(s) recorded before it, plus the garbage Q{garbage_question_number} entry itself, "
            f"plus whatever Stage 1's mnn_results.json already holds)."
        )
        result["status"] = "failed"
        result["error"] = f"garbage on Q{garbage_question_number}: stopped GGUF fallback for this model"
        result["gguf_results"] = entries
        return result

    result["status"] = "success"
    result["gguf_results"] = entries
    return result


def process_full_model(quality, variant: dict, questions: list, index: int, total: int, timeout: int,
                        no_think: bool, max_tokens) -> dict:
    """--full-run mode: runs EVERY question in `questions` against this
    model+quant, completely independent of anything Stage 1 (MNN) did - no
    handoff, no reading of needs_gguf_fallback.json at all. This is a
    deliberate near-duplicate of process_flagged_model() rather than a
    shared refactor of it, so that mode is guaranteed untouched by this
    addition.

    The retry mechanism (WARMUP_ATTEMPTS discarded + 1 final attempt
    recorded, TOTAL_ATTEMPTS_PER_QUESTION total) and all metrics collection
    are identical to process_flagged_model() - only the source of the
    question list changes (the full eval set, numbered 1..N here, instead
    of a Stage-1-flagged subset carrying its own original numbering) and the
    stop-early messaging no longer references MNN/Stage 1 at all, since
    there's no handoff for it to refer to.

    Returns {"model_id", "quant", "status" ("success"|"failed"), "error",
    "gguf_results": [...]}."""
    model_id = variant["model_id"]
    quant = variant["quant"]
    numbered_questions = [
        {"question_number": i, "text": q["text"], "answer": q["answer"]}
        for i, q in enumerate(questions, start=1)
    ]
    label = f"[{index}/{total}] {model_id} [{quant}]"

    result = {"model_id": model_id, "quant": quant, "status": None, "error": None, "gguf_results": []}

    if max_tokens is not None:
        print(f"{label} [NOTE] --max-tokens has no effect here - SmolChat's broadcast protocol has no max_tokens extra.")

    print(f"\n{'=' * 70}\n{label} ({len(numbered_questions)} question(s), full run)\n{'=' * 70}")

    conv = ensure_gguf_converted(model_id, quant, label)
    if not conv["ok"]:
        print(f"{label} - FAILED (GGUF conversion): {conv['error']}")
        result["status"] = "failed"
        result["error"] = f"GGUF conversion failed: {conv['error']}"
        return result

    batch = run_gguf_batch(conv["local_path"], quant, numbered_questions, timeout, no_think, label)
    if not batch["ok"]:
        result["status"] = "failed"
        result["error"] = batch["error"]
        return result
    results_list = batch["results_list"]

    # Check every question in order. The moment ANY question comes back
    # garbage - first or later, and after exhausting its own retries - stop
    # immediately: no further questions after it are attempted, the model
    # is marked "failed", and only the entries recorded BEFORE the garbage
    # question are kept. Identical mechanics to process_flagged_model()'s
    # own stop-on-first-garbage rule, just applied to the full question set
    # rather than a flagged subset.
    entries = []
    garbage_question_number = None
    total_q = len(numbered_questions)
    for idx, q in enumerate(numbered_questions):
        question_number = q["question_number"]
        reference_answer = q["answer"]
        qlabel = f"{label} Q{question_number}/{total_q} [gguf]"

        raw_result = results_list[idx] if idx < len(results_list) else None
        response, metrics, run_id, garbage_flag = extract_response_and_metrics(quality, raw_result)

        # Prints the FULL metrics+response block for every attempt,
        # immediately after that attempt completes - not just a one-line
        # summary for the discarded ones - so warmup attempts are just as
        # inspectable as the final, recorded one. Returns that attempt's own
        # is_correct (computed fresh per attempt, not reused across
        # attempts) so the caller can track the final attempt's value
        # without re-printing or re-scoring it.
        def print_attempt_block(attempt_num, response, metrics):
            label_suffix = "discarded" if attempt_num < TOTAL_ATTEMPTS_PER_QUESTION else "final, recorded"
            attempt_label = f"{qlabel} attempt {attempt_num}/{TOTAL_ATTEMPTS_PER_QUESTION} ({label_suffix})"
            attempt_is_correct = quality.score_accuracy(response, reference_answer)
            print_question_result(attempt_label, metrics, attempt_is_correct, response)
            return attempt_is_correct

        # ALWAYS run exactly TOTAL_ATTEMPTS_PER_QUESTION total attempts -
        # NOT "retry until success". The batch call above was attempt 1 for
        # every question (a warmup attempt like all the others). The first
        # WARMUP_ATTEMPTS attempts are unconditionally discarded regardless
        # of whether they pass or fail is_garbage() - is_garbage() itself is
        # still called on every attempt, unchanged, but its result only
        # matters for the FINAL attempt. Only that final attempt's
        # response/metrics/run_id/garbage_flag survive this loop and get
        # recorded below, even if it's also garbage.
        attempt = 1
        is_correct = print_attempt_block(attempt, response, metrics)
        while attempt < TOTAL_ATTEMPTS_PER_QUESTION:
            attempt += 1
            retry_batch = run_gguf_batch(conv["local_path"], quant, [q], timeout, no_think, qlabel)
            if retry_batch["ok"] and retry_batch["results_list"]:
                retry_raw_result = retry_batch["results_list"][0]
            else:
                retry_raw_result = None  # counts as garbage, same as a missing/non-success result
            response, metrics, run_id, garbage_flag = extract_response_and_metrics(quality, retry_raw_result)
            is_correct = print_attempt_block(attempt, response, metrics)

        # Record the final attempt's result unconditionally - no early
        # stopping, no conditional acceptance, even if it's garbage. Already
        # printed above (the "final, recorded" block), so not printed again.
        entry = build_result_entry(quality, model_id, quant, question_number, q["text"], reference_answer,
                                    response, is_correct, garbage_flag, metrics, run_id)
        entries.append(entry)

        if garbage_flag:
            print(f"{qlabel} -> final attempt is GARBAGE - recorded anyway (no retry-until-success), "
                  f"but still stopping this model's run here, unchanged.")
            garbage_question_number = question_number
            break  # stop immediately - no further questions attempted

    if garbage_question_number is not None:
        successful_before_it = len(entries) - 1
        print(
            f"{label} - GGUF garbage on Q{garbage_question_number} after {TOTAL_ATTEMPTS_PER_QUESTION} "
            f"total attempts - marking model FAILED, stopping (keeping {successful_before_it} genuinely-successful "
            f"GGUF question(s) recorded before it, plus the garbage Q{garbage_question_number} entry itself)."
        )
        result["status"] = "failed"
        result["error"] = f"garbage on Q{garbage_question_number}: stopped GGUF full run for this model"
        result["gguf_results"] = entries
        return result

    result["status"] = "success"
    result["gguf_results"] = entries
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(model_results: list):
    print("\n" + "=" * 90)
    print("STAGE 2 (GGUF) SUMMARY")
    print("=" * 90)
    header = f"{'Model':<32}{'Quant':<10}{'Status':<20}{'Accuracy':>10}{'N':>6}"
    print(header)
    print("-" * len(header))
    for r in model_results:
        entries = r["gguf_results"]
        total = len(entries)
        correct = sum(1 for e in entries if e["is_correct"])
        acc = (correct / total) if total > 0 else None
        acc_disp = f"{acc * 100:.1f}%" if acc is not None else "N/A"
        row = (
            f"{str(r['model_id']):<32}"
            f"{str(r['quant']):<10}"
            f"{r['status']:<20}"
            f"{acc_disp:>10}"
            f"{total:>6}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2 (GGUF-only) of the MNN-first, GGUF-fallback evaluation pipeline."
    )
    p.add_argument("--fallback-file", default=str(SCRIPT_DIR / "logs" / "needs_gguf_fallback.json"),
                    help="Path to the queue written by Stage 1 (run_fallback_agent_mnn.py's --fallback-file). "
                         "Ignored when --full-run is set.")
    p.add_argument("--full-run", action="store_true", dest="full_run",
                    help="Run the ENTIRE question set against every model+quant in --fit-report, "
                         "completely independent of anything MNN (Stage 1) did - no fallback-file "
                         "handoff at all. Requires --fit-report and --questions (same format/flags as "
                         "run_fallback_agent_mnn.py). The fallback-file mode above stays available "
                         "and unchanged when this flag is omitted.")
    p.add_argument("--fit-report", default=None,
                    help="Required with --full-run: same fit-report JSON ({'fits': [...]}) run_fallback_agent_mnn.py takes.")
    p.add_argument("--questions", default=None,
                    help="Required with --full-run: same eval-questions JSON run_fallback_agent_mnn.py takes.")
    p.add_argument("--timeout", type=int, default=180,
                    help="Seconds to wait per question's broadcast result on GGUF (default: 180). "
                         "The overall per-model subprocess timeout is this multiplied by that model's "
                         f"flagged question count, plus a {SETUP_OVERHEAD_SECONDS}s setup buffer.")
    p.add_argument("--no-think", action="store_true", dest="no_think",
                    help="Append ' /no_think' to every question's prompt. Default: off.")
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                    help="Accepted for CLI compatibility with Stage 1, but has no effect - SmolChat's "
                         "broadcast protocol has no max_tokens extra.")
    p.add_argument("--cooldown-minutes", type=float, default=5.0, dest="cooldown_minutes",
                    help="Real-time pause between one model+quant's run and the next, letting the phone's "
                         "thermal state return to baseline (default: 5.0). Not applied after the last model. "
                         "Use 0 to skip entirely (quick tests where thermal accuracy doesn't matter).")
    p.add_argument("--gguf-output", default=str(SCRIPT_DIR / "logs" / "gguf_results.json"),
                    help="Flat list of every question result recorded while on GGUF, across all flagged models.")
    p.add_argument("--summary-output", default=str(SCRIPT_DIR / "logs" / "agent_summary_gguf.json"),
                    help="Per-model status summary for Stage 2.")
    # parse_known_args(), not parse_args(): run_pipeline.sh passes the same
    # argv to BOTH stage scripts, and Stage 1 has its own flags (e.g.
    # --fit-report, --mnn-output) this script doesn't define - unknown args
    # are ignored rather than raising, so one shared invocation works for both.
    args, _unknown = p.parse_known_args()
    return args


def run_full(quality, args) -> None:
    """--full-run mode: runs the ENTIRE question set against every
    model+quant in --fit-report, independent of anything MNN (Stage 1) did -
    no needs_gguf_fallback.json handoff at all. Mirrors main()'s existing
    fallback-file flow (load -> loop -> flatten results -> write outputs ->
    print summary), just with a different input source (--fit-report/
    --questions instead of --fallback-file) and process_full_model() instead
    of process_flagged_model().
    """
    if not args.fit_report or not args.questions:
        print("[ERROR] --full-run requires both --fit-report and --questions")
        sys.exit(1)
    if not RUN_FALLBACK_AGENT_MNN_SCRIPT.exists():
        print(f"[ERROR] run_fallback_agent_mnn.py not found at {RUN_FALLBACK_AGENT_MNN_SCRIPT} "
              "(needed by --full-run to reuse its --fit-report/--questions loaders)")
        sys.exit(1)

    # Reused by path, not reimplemented, so parsing can never silently drift
    # out of sync with Stage 1's own --fit-report/--questions format.
    mnn_fallback = _load_module(RUN_FALLBACK_AGENT_MNN_SCRIPT, "_run_fallback_agent_mnn_for_gguf_full_run")

    try:
        variants = mnn_fallback.load_fit_report(args.fit_report)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to load fit report '{args.fit_report}': {exc}")
        sys.exit(1)
    if not variants:
        print(f"[ERROR] No variants found in '{args.fit_report}' (expected a non-empty 'fits' list).")
        sys.exit(1)

    try:
        questions = mnn_fallback.load_eval_questions(args.questions)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] Failed to load questions '{args.questions}': {exc}")
        sys.exit(1)
    if not questions:
        print(f"[ERROR] No questions found in '{args.questions}'.")
        sys.exit(1)

    total = len(variants)
    print("=" * 70)
    print("Stage 2: GGUF Full Run (independent of MNN)")
    print(f"  Pool: {args.fit_report} ({total} models)  Questions: {args.questions} ({len(questions)})  "
          f"Per-question timeout: {args.timeout}s")
    print(f"  No-think: {'ON' if args.no_think else 'OFF'}")
    print("=" * 70)

    model_results = []
    for i, variant in enumerate(variants, start=1):
        r = process_full_model(quality, variant, questions, i, total, args.timeout, args.no_think, args.max_tokens)
        model_results.append(r)
        cooldown_between_models(args.cooldown_minutes, i, total)

    gguf_flat = []
    for r in model_results:
        gguf_flat.extend(r["gguf_results"])

    with open(args.gguf_output, "w") as f:
        json.dump(gguf_flat, f, indent=2, default=str)
    print(f"\n[OUTPUT] GGUF results saved: {args.gguf_output} ({len(gguf_flat)} question results)")

    succeeded = sum(1 for r in model_results if r["status"] == "success")
    failed = sum(1 for r in model_results if r["status"] == "failed")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fit_report": args.fit_report,
        "questions_file": args.questions,
        "total_models": total,
        "succeeded": succeeded,
        "failed": failed,
        "models": [
            {
                "model_id": r["model_id"],
                "quant": r["quant"],
                "status": r["status"],
                "error": r["error"],
                "total_questions_recorded": len(r["gguf_results"]),
            }
            for r in model_results
        ],
    }
    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[OUTPUT] Summary saved: {args.summary_output}")

    print_summary_table(model_results)

    print("\n" + "=" * 70)
    print(f"STAGE 2 (FULL RUN) DONE - {succeeded} succeeded, {failed} failed (of {total} models)")
    print("=" * 70)


def main():
    args = parse_args()

    for path, label in REQUIRED_SCRIPTS:
        if not path.exists():
            print(f"[ERROR] {label} not found at {path}")
            sys.exit(1)

    quality = _load_module(RESPONSE_QUALITY_SCRIPT, "_response_quality")

    Path(args.gguf_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)

    if args.full_run:
        run_full(quality, args)
        return

    try:
        flagged_models = load_fallback_queue(args.fallback_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to load fallback queue '{args.fallback_file}': {exc}")
        sys.exit(1)

    if not flagged_models:
        print(f"[OK] No models flagged for GGUF fallback in '{args.fallback_file}' - nothing to do.")
        with open(args.gguf_output, "w") as f:
            json.dump([], f, indent=2)
        with open(args.summary_output, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fallback_file": args.fallback_file,
                "total_models": 0, "succeeded": 0, "failed": 0, "models": [],
            }, f, indent=2)
        return

    total = len(flagged_models)
    print("=" * 70)
    print("Stage 2: GGUF Fallback Evaluation")
    print(f"  Fallback queue: {args.fallback_file} ({total} model(s) flagged)  Per-question timeout: {args.timeout}s")
    print(f"  No-think: {'ON' if args.no_think else 'OFF'}")
    print("=" * 70)

    model_results = []
    for i, model_entry in enumerate(flagged_models, start=1):
        r = process_flagged_model(quality, model_entry, i, total, args.timeout, args.no_think, args.max_tokens)
        model_results.append(r)
        cooldown_between_models(args.cooldown_minutes, i, total)

    gguf_flat = []
    for r in model_results:
        gguf_flat.extend(r["gguf_results"])

    with open(args.gguf_output, "w") as f:
        json.dump(gguf_flat, f, indent=2, default=str)
    print(f"\n[OUTPUT] GGUF results saved: {args.gguf_output} ({len(gguf_flat)} question results)")

    succeeded = sum(1 for r in model_results if r["status"] == "success")
    failed = sum(1 for r in model_results if r["status"] == "failed")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fallback_file": args.fallback_file,
        "total_models": total,
        "succeeded": succeeded,
        "failed": failed,
        "models": [
            {
                "model_id": r["model_id"],
                "quant": r["quant"],
                "status": r["status"],
                "error": r["error"],
                "total_questions_recorded": len(r["gguf_results"]),
            }
            for r in model_results
        ],
    }
    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[OUTPUT] Summary saved: {args.summary_output}")

    print_summary_table(model_results)

    print("\n" + "=" * 70)
    print(f"STAGE 2 DONE - {succeeded} succeeded, {failed} failed (of {total} flagged)")
    print("=" * 70)


if __name__ == "__main__":
    main()
