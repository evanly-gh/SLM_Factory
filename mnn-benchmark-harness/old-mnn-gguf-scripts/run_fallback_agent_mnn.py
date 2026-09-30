#!/usr/bin/env python3
"""
run_fallback_agent_mnn.py — Stage 1 of the MNN-first, GGUF-fallback pipeline.

For each model in a check_model_fit.py fit-report, runs every question from
an eval set (text + reference answer pairs) against MNN. The MOMENT a
response is garbage (response_quality.is_garbage(), including its Python-
side zero-width-space-robust repetition check), processing for THAT MODEL
stops immediately - whatever questions already succeeded stay recorded in
mnn_results.json, and the model_id/quant/remaining-questions-starting-from-
the-failure-point get recorded into needs_gguf_fallback.json for Stage 2
(run_fallback_agent_gguf.py) to pick up. The agent then moves on to the next
model in the pool.

THIS SCRIPT DOES NOT TALK TO GGUF/SMOLCHAT AT ALL. That's the whole point of
the split: an earlier single-process design ran GGUF/SmolChat as a nested
subprocess of this same script, and that nesting had a confirmed,
unresolved hang that survived 7 separate fix attempts across fundamentally
different mechanisms (stdin handling, an MNN force-stop, uncaptured output
with polling, session isolation via start_new_session, os.system()
invocation instead of subprocess.run(), an ADB server restart) - all failed
identically. That strongly indicates a structural incompatibility with
nesting a Python-invoked benchmark script inside another running Python
process, not something fixable at this level. Splitting into two SEPARATE
top-level processes (this one, then run_fallback_agent_gguf.py, run
sequentially by run_pipeline.sh - Stage 1 fully exits before Stage 2 starts)
sidesteps the nesting problem entirely, since every confirmed-working
invocation this whole debugging saga was a fresh, non-nested process.

Reuses existing, proven infrastructure directly (by importing each script by
path, the pattern already used throughout this project):
  - convert_to_mnn.py       : HF download + llmexport.py MNN export
  - agent_mnn_quantize.py   : push_to_device() (MNN) and its shared local
                               model cache directory
  - run_mnn_autobench.py    : find_adb()/Adb, run_one() (single-question
                               broadcast+poll), build_metrics(),
                               extract_response()/extract_error()
  - response_quality.py     : is_garbage(), score_accuracy(),
                               extract_final_answer()

Usage (standalone):
    python3 run_fallback_agent_mnn.py --fit-report model_fit_report_q4q8.json --questions eval_questions_phase3.json

Usage (full pipeline, both stages):
    ./run_pipeline.sh --fit-report model_fit_report_q4q8.json --questions eval_questions_phase3.json
"""

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

RUN_MNN_AUTOBENCH_SCRIPT = SCRIPT_DIR / "run_mnn_autobench.py"
AGENT_QUANTIZE_SCRIPT = SCRIPT_DIR / "agent_mnn_quantize.py"
RESPONSE_QUALITY_SCRIPT = SCRIPT_DIR / "response_quality.py"
CONVERT_TO_MNN_SCRIPT = SCRIPT_DIR.parent / "Model-Conversion" / "convert_to_mnn.py"
CHECK_MODEL_FIT_SCRIPT = SCRIPT_DIR.parent / "check_model_fit.py"

REQUIRED_SCRIPTS = [
    (RUN_MNN_AUTOBENCH_SCRIPT, "run_mnn_autobench.py"),
    (AGENT_QUANTIZE_SCRIPT, "agent_mnn_quantize.py"),
    (RESPONSE_QUALITY_SCRIPT, "response_quality.py"),
    (CONVERT_TO_MNN_SCRIPT, "convert_to_mnn.py"),
    (CHECK_MODEL_FIT_SCRIPT, "check_model_fit.py"),
]

# check_model_fit.py's "fits" list carries GGUF-style quant names (its own
# SUPPORTED_QUANTS = {"Q4_K_M", "Q8_0", "F16"}) - this is the one place that
# name gets mapped to MNN's numeric --quant_bit.
QUANT_BIT_MAP = {"Q4_K_M": 4, "Q8_0": 8, "F16": 16}

# Mirrors run_fallback_agent_gguf.py's own WARMUP_ATTEMPTS/
# TOTAL_ATTEMPTS_PER_QUESTION exactly: NOT a "retry until success"
# mechanism - every question ALWAYS runs exactly TOTAL_ATTEMPTS_PER_QUESTION
# total attempts. The first WARMUP_ATTEMPTS are unconditionally discarded
# regardless of whether they pass or fail is_garbage() - only the FINAL
# attempt's result is ever recorded, even if it's also garbage. is_garbage()
# itself is unchanged - this only wraps additional attempts around it and
# decides which one gets kept.
WARMUP_ATTEMPTS = 5
TOTAL_ATTEMPTS_PER_QUESTION = WARMUP_ATTEMPTS + 1


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fit_report(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data.get("fits", [])


def load_eval_questions(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list of {{'text':..., 'answer':...}} objects, got {type(data).__name__}")
    for i, q in enumerate(data):
        if "text" not in q or "answer" not in q:
            raise ValueError(f"question at index {i} is missing 'text' or 'answer': {q}")
    return data


def shorten(text, n=150):
    if not text:
        return text
    return text if len(text) <= n else text[:n] + "..."


def _fmt(v, unit="", nd=1):
    return f"{v:.{nd}f}{unit}" if isinstance(v, (int, float)) else "N/A"


def cooldown_between_models(minutes: float, index: int, total: int) -> None:
    """Real time.sleep() pause between one model+quant's run and the next,
    letting the phone's thermal state return to baseline before starting the
    next benchmark - thermal state materially affects TTFT/TPS, so back-to-
    back runs without a rest period can make later models look artificially
    slower/hotter than they'd be measured cold. Skipped after the LAST
    model (nothing left to cool down for) and entirely when minutes <= 0
    (--cooldown-minutes 0, for quick tests where thermal accuracy doesn't matter).
    """
    if minutes <= 0 or index >= total:
        return
    print(f"\n[COOLDOWN] Waiting {minutes:g} minute(s) before next model...")
    time.sleep(minutes * 60)


def print_question_result(qlabel: str, metrics: dict, is_correct, response) -> None:
    """Standardized multi-line per-question result block:

        [1/2] Qwen/Qwen3-0.6B [Q4_K_M] Q1/2 [mnn]
          ColdLoad=698ms  TTFT=1063.4ms  PrefillTPS=75.6  DecodeTPS=23.7
          RSS=560036KB  Power=350.66mA (BatteryMgr) / N/A (Monsoon)
          ThermalCPU=61.7°C  ThermalSkin=40.0°C  Correct=True
          Response: "<truncated to ~150 chars>..."
    """
    battery_power_disp = metrics.get("power_ma_raw") if metrics.get("power_ma_raw") is not None else "N/A"
    monsoon_power_disp = _fmt(metrics.get("power_ma_monsoon"), "mA", 2)
    print(qlabel)
    print(
        f"  ColdLoad={_fmt(metrics.get('cold_load_ms'), 'ms', 0)}  TTFT={_fmt(metrics.get('ttft_ms'), 'ms')}  "
        f"PrefillTPS={_fmt(metrics.get('prefill_tps'))}  DecodeTPS={_fmt(metrics.get('decode_tps'))}"
    )
    print(
        f"  RSS={_fmt(metrics.get('peak_rss_kb'), 'KB', 0)}  "
        f"Power={battery_power_disp}mA (BatteryMgr) / {monsoon_power_disp} (Monsoon)"
    )
    print(
        f"  ThermalCPU={_fmt(metrics.get('thermal_cpu_c'), '°C')}  ThermalSkin={_fmt(metrics.get('thermal_skin_c'), '°C')}  "
        f"Correct={is_correct}"
    )
    print(f'  Response: "{shorten(response)}"')


# ---------------------------------------------------------------------------
# MNN: convert + push (via convert_to_mnn.py + agent_mnn_quantize.py)
# ---------------------------------------------------------------------------

def convert_and_push_mnn(mnn_convert, agent, adb_bin: str, model_id: str, quant_bit: int, label: str) -> dict:
    """Returns {"ok": True, "device_path": str} or {"ok": False, "error": str}.

    Never raises/exits: convert_to_mnn.py's own functions call sys.exit(1)
    directly on some failure paths (missing MNNConvert binary, a failed HF
    download) since it was written as a single-shot CLI tool - SystemExit is
    caught here alongside ordinary exceptions so one model's failure can
    never take down the whole pool run.
    """
    try:
        if not mnn_convert.LLMEXPORT_SCRIPT.exists():
            return {"ok": False, "error": f"llmexport.py not found at {mnn_convert.LLMEXPORT_SCRIPT}"}
        if not mnn_convert.MNNCONVERT_BIN.exists():
            return {"ok": False, "error": f"MNNConvert binary not found at {mnn_convert.MNNCONVERT_BIN}"}

        # agent.LOCAL_MODELS_DIR must exist before resolve_model() is called -
        # its own check_disk_space() call requires the directory to already
        # exist (shutil.disk_usage raises otherwise). agent_mnn_quantize.py's
        # own resolve_local_model_dir() normally creates this lazily; calling
        # convert_to_mnn.py's resolve_model() directly bypasses that, so it's
        # created explicitly here instead.
        agent.LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        prefix = mnn_convert.model_prefix(model_id)
        slug = mnn_convert.quant_slug(prefix, str(quant_bit))
        export_dir = agent.LOCAL_MODELS_DIR / slug

        if mnn_convert.is_export_complete(export_dir):
            print(f"{label} [MNN] - reusing existing export at {export_dir}")
        else:
            interpreter = mnn_convert.find_llmexport_interpreter()
            if interpreter is None:
                return {"ok": False, "error": f"llmexport.py venv not found at {mnn_convert.LLMEXPORT_VENV_PYTHON}"}

            print(f"{label} [MNN] - downloading (if needed)...")
            model_dir = mnn_convert.resolve_model(model_id, agent.LOCAL_MODELS_DIR)

            print(f"{label} [MNN] - exporting to Q{quant_bit}...")
            result = mnn_convert.export_one(interpreter, model_dir, str(quant_bit), export_dir)
            if not result["ok"]:
                return {"ok": False, "error": f"MNN export failed: {result['error']}"}

        print(f"{label} [MNN] - pushing to device...")
        push = agent.push_to_device(adb_bin, export_dir, slug)
        if not push["ok"]:
            return {"ok": False, "error": f"MNN push failed: {push['error']}"}

        return {"ok": True, "device_path": push["device_path"]}
    except SystemExit as exc:
        return {"ok": False, "error": f"MNN conversion/push aborted internally (convert_to_mnn.py called sys.exit): {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"MNN conversion/push raised: {exc}"}


def run_mnn_question(mnn_module, mnn_adb, device_path: str, question_text: str, n: int, timeout: int,
                      no_think: bool, max_tokens) -> dict:
    """Single-question MNN broadcast+poll, via run_mnn_autobench.py's own
    run_one() (already exactly one question at a time - the "batch" in
    run_benchmark() is just a Python for-loop calling run_one() per
    question, so no adaptation is needed, just calling it directly here
    instead of going through run_benchmark())."""
    # Only pass max_tokens through when the caller actually set it - passing
    # max_tokens=None explicitly would override run_one()'s own default
    # (4096) with None instead of letting that default take over, which is
    # exactly the bug this replaces (a stale hardcoded 256 fallback here
    # silently beat run_one()'s real default whenever --max-tokens wasn't
    # passed on the command line).
    run_one_kwargs = {"no_think": no_think}
    if max_tokens is not None:
        run_one_kwargs["max_tokens"] = max_tokens
    outcome = mnn_module.run_one(mnn_adb, device_path, question_text, n, timeout, **run_one_kwargs)
    status = outcome["status"]
    if status == "done":
        metrics = mnn_module.build_metrics(outcome["tag_lines"])
        # Mirrors run_benchmark()'s own merge - run_one() itself doesn't
        # fold the Monsoon reading into metrics, run_benchmark() does.
        metrics["power_ma_monsoon"] = outcome["monsoon"].get("power_ma_mean")
        response = mnn_module.extract_response(outcome["run_lines"], outcome["run_id"])
        return {"engine_status": "done", "response": response, "metrics": metrics, "run_id": outcome["run_id"], "error": None}
    if status == "error":
        reason, message = mnn_module.extract_error(outcome["tag_lines"].get("RUN_ERROR", ""))
        return {"engine_status": "error", "response": None, "metrics": None, "run_id": outcome["run_id"],
                "error": {"reason": reason, "message": message}}
    return {"engine_status": "timeout", "response": None, "metrics": None, "run_id": outcome["run_id"],
            "error": {"reason": "timeout", "message": f"No RUN_DONE/RUN_ERROR within {timeout}s"}}


# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------

def build_result_entry(quality, model_id, quant, engine, question_number, question_text, reference_answer,
                        response, is_correct, metrics, run_id) -> dict:
    # extract_final_answer() is display-only (best-effort, not always exact)
    # - is_correct above already came from score_accuracy() against the raw,
    # unmodified response, so an imperfect extraction here can't skew scoring.
    return {
        "model_id": model_id,
        "quant": quant,
        "engine": engine,
        "question_number": question_number,
        "question": question_text,
        "reference_answer": reference_answer,
        "response": response,
        "extracted_answer": quality.extract_final_answer(response),
        "is_correct": is_correct,
        "metrics": metrics,
        "run_id": run_id,
    }


def finish(result: dict) -> dict:
    """Compute mnn_accuracy over this model's mnn_results - called on every
    return path (skipped/failed/needs_gguf_fallback/success) so downstream
    summary code never has to special-case a missing key."""
    total = len(result["mnn_results"])
    correct = sum(1 for e in result["mnn_results"] if e["is_correct"])
    result["mnn_accuracy"] = (correct / total) if total > 0 else None
    result["total_questions_recorded"] = total
    return result


# ---------------------------------------------------------------------------
# Post-evaluation RAM fit check (ported from run_model_pool.py's
# fits_based_on_measured_rss logic - see check_current_fit()/process_variant()
# there for the original)
# ---------------------------------------------------------------------------

def compute_fits_based_on_measured_rss(check_fit, result: dict, label: str):
    """After this model's MNN evaluation has finished (all questions
    completed, or stopped early due to garbage), take the highest
    peak_rss_kb seen across its successfully-recorded questions and compare
    it against the phone's CURRENT free RAM - queried fresh via the same
    method check_model_fit.py uses (get_phone_free_ram_mb(): /proc/meminfo
    MemAvailable, no multiplier).

    Ported directly from run_model_pool.py's own fits_based_on_measured_rss
    computation: same comparison (measured_rss_mb <= free_ram_mb), same
    "unknown, not fatal" handling of a free-RAM query failure - the only
    difference is WHEN the query happens (there: once, before that
    variant's benchmark ever ran; here: once, after this model's questions
    are done, since Stage 1 processes questions one at a time rather than
    via a single batch benchmark_one() call with one metrics dict).

    Returns True/False, or None if there's no measured RSS to compare (e.g.
    every question failed before producing metrics) or the free-RAM query
    itself failed.
    """
    peak_rss_values = [
        e["metrics"].get("peak_rss_kb")
        for e in result["mnn_results"]
        if e.get("metrics") and e["metrics"].get("peak_rss_kb") is not None
    ]
    if not peak_rss_values:
        return None
    max_peak_rss_kb = max(peak_rss_values)

    try:
        current_free_ram_mb = check_fit.get_phone_free_ram_mb()
    except RuntimeError as exc:
        print(f"{label} - [WARN] could not query current free RAM for fits_based_on_measured_rss: {exc}")
        return None

    measured_rss_mb = max_peak_rss_kb / 1024
    return measured_rss_mb <= current_free_ram_mb


def process_model(mnn_module, mnn_convert, agent, quality, check_fit, mnn_adb, adb_bin: str,
                   variant: dict, questions: list, index: int, total: int, timeout: int,
                   no_think: bool, max_tokens) -> tuple:
    """Returns (result_dict, needs_gguf_entry_or_None)."""
    model_id = variant.get("model_id")
    quant = variant.get("quant")
    label = f"[{index}/{total}] {model_id} [{quant}]"

    result = {
        "model_id": model_id, "quant": quant, "status": None, "error": None,
        "mnn_results": [], "fits_based_on_measured_rss": None,
    }

    quant_bit = QUANT_BIT_MAP.get(quant)
    if quant_bit is None:
        print(f"{label} - SKIPPED: unrecognized quant {quant!r} (expected Q4_K_M, Q8_0, or F16)")
        result["status"] = "skipped"
        result["error"] = f"unrecognized quant value {quant!r}"
        return finish(result), None

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")

    conv = convert_and_push_mnn(mnn_convert, agent, adb_bin, model_id, quant_bit, label)
    if not conv["ok"]:
        print(f"{label} - FAILED (MNN setup): {conv['error']}")
        result["status"] = "failed"
        result["error"] = f"MNN conversion/push failed: {conv['error']}"
        return finish(result), None

    mnn_module.reset_mnnchat_for_clean_process(mnn_adb)
    mnn_device_path = conv["device_path"]

    total_q = len(questions)
    for i, q in enumerate(questions, start=1):
        question_text, reference_answer = q["text"], q["answer"]
        qlabel = f"{label} Q{i}/{total_q} [mnn]"

        # Prints the FULL metrics+response block for every attempt,
        # immediately after that attempt completes - not just a one-line
        # summary for the discarded ones - mirroring run_fallback_agent_gguf.py's
        # identical change. Returns that attempt's own is_correct (computed
        # fresh per attempt, not reused across attempts).
        def print_attempt_block(attempt_num, response, metrics):
            label_suffix = "discarded" if attempt_num < TOTAL_ATTEMPTS_PER_QUESTION else "final, recorded"
            attempt_label = f"{qlabel} attempt {attempt_num}/{TOTAL_ATTEMPTS_PER_QUESTION} ({label_suffix})"
            attempt_is_correct = quality.score_accuracy(response, reference_answer)
            print_question_result(attempt_label, metrics, attempt_is_correct, response)
            return attempt_is_correct

        attempt = 1
        outcome = run_mnn_question(mnn_module, mnn_adb, mnn_device_path, question_text, i, timeout, no_think, max_tokens)
        response = outcome["response"]
        metrics = outcome["metrics"] or {}

        # A hard engine error/timeout is treated the same as garbage output
        # here - either way this question produced nothing usable on MNN.
        # is_garbage() itself is fully active and unchanged on every attempt.
        garbage = outcome["engine_status"] != "done" or quality.is_garbage(response, metrics)
        is_correct = print_attempt_block(attempt, response, metrics)

        # NOT "retry until success": every question ALWAYS runs exactly
        # TOTAL_ATTEMPTS_PER_QUESTION total attempts. The first
        # WARMUP_ATTEMPTS are unconditionally discarded regardless of
        # whether they pass or fail is_garbage() - only the FINAL attempt's
        # outcome/response/metrics is ever recorded or fed into the
        # GGUF-fallback trigger below, even if it's also garbage.
        while attempt < TOTAL_ATTEMPTS_PER_QUESTION:
            attempt += 1
            outcome = run_mnn_question(mnn_module, mnn_adb, mnn_device_path, question_text, i, timeout, no_think, max_tokens)
            response = outcome["response"]
            metrics = outcome["metrics"] or {}
            garbage = outcome["engine_status"] != "done" or quality.is_garbage(response, metrics)
            is_correct = print_attempt_block(attempt, response, metrics)

        if outcome["engine_status"] != "done":
            print(f"{qlabel} - {outcome['engine_status'].upper()}: {outcome['error']}")
        elif garbage:
            print(f"{qlabel} - final attempt is GARBAGE after {TOTAL_ATTEMPTS_PER_QUESTION} total attempts - recorded anyway (no retry-until-success): {shorten(response)!r}")
        else:
            print(f"{qlabel} -> final attempt passed")

        # Record the final attempt's result unconditionally - even when
        # garbage - mirroring run_fallback_agent_gguf.py's identical change.
        # Already printed above (the "final, recorded" block), so not
        # printed again here.
        entry = build_result_entry(quality, model_id, quant, "mnn", i, question_text, reference_answer,
                                    response, is_correct, metrics, outcome["run_id"])
        result["mnn_results"].append(entry)

        if garbage:
            # Stop processing further questions for THIS model on MNN
            # immediately - the moment the final attempt is still garbage,
            # not just on Q1. Qi's garbage entry was already recorded into
            # mnn_results above (unconditional recording, matching
            # run_fallback_agent_gguf.py's identical change) - it stays
            # there even though Qi is ALSO flagged below for GGUF fallback,
            # so Stage 2 may end up recording a (hopefully better) result
            # for the same question number in gguf_results.json too.
            # question_number is preserved explicitly (not just list order)
            # so Stage 2 can record results under their ORIGINAL numbering
            # (e.g. Q5-Q10), not restart at 1.
            remaining_questions = [
                {"question_number": qn, "text": q2["text"], "answer": q2["answer"]}
                for qn, q2 in zip(range(i, total_q + 1), questions[i - 1:])
            ]
            print(
                f"{label} - MNN garbage on Q{i}/{total_q} - stopping MNN for this model, flagging "
                f"Q{i}-Q{total_q} ({len(remaining_questions)} question(s)) for GGUF fallback."
            )
            result["status"] = "needs_gguf_fallback"
            result["error"] = f"garbage on Q{i}; {len(remaining_questions)} question(s) flagged for GGUF fallback"
            result["fits_based_on_measured_rss"] = compute_fits_based_on_measured_rss(check_fit, result, label)
            needs_gguf_entry = {"model_id": model_id, "quant": quant, "questions": remaining_questions}
            return finish(result), needs_gguf_entry

        time.sleep(2)

    result["status"] = "success"
    result["fits_based_on_measured_rss"] = compute_fits_based_on_measured_rss(check_fit, result, label)
    return finish(result), None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(model_results: list):
    print("\n" + "=" * 90)
    print("STAGE 1 (MNN) SUMMARY")
    print("=" * 90)
    header = f"{'Model':<32}{'Quant':<10}{'Status':<20}{'Accuracy':>10}{'N':>6}"
    print(header)
    print("-" * len(header))
    for r in model_results:
        acc = r.get("mnn_accuracy")
        acc_disp = f"{acc * 100:.1f}%" if acc is not None else "N/A"
        row = (
            f"{str(r['model_id']):<32}"
            f"{str(r['quant']):<10}"
            f"{r['status']:<20}"
            f"{acc_disp:>10}"
            f"{r.get('total_questions_recorded', 0):>6}"
        )
        print(row)


def save_needs_gguf_fallback(path: str, entries: list):
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "models": entries}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[OUTPUT] GGUF fallback queue saved: {path} ({len(entries)} model(s) flagged)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1 (MNN-only) of the MNN-first, GGUF-fallback evaluation pipeline."
    )
    p.add_argument("--fit-report", required=True,
                    help="Path to a check_model_fit.py fit-report JSON; its 'fits' list is the pool to process.")
    p.add_argument("--questions", required=True,
                    help="Path to an eval_questions_phase3.json-style JSON: a list of {'text':..., 'answer':...} objects.")
    p.add_argument("--timeout", type=int, default=180,
                    help="Seconds to wait for a single question's broadcast result on MNN (default: 180).")
    p.add_argument("--no-think", action="store_true", dest="no_think",
                    help="Append ' /no_think' to every question's prompt. Default: off.")
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                    help="Passed through to MNN's --ei max_tokens extra.")
    p.add_argument("--cooldown-minutes", type=float, default=5.0, dest="cooldown_minutes",
                    help="Real-time pause between one model+quant's run and the next, letting the phone's "
                         "thermal state return to baseline (default: 5.0). Not applied after the last model. "
                         "Use 0 to skip entirely (quick tests where thermal accuracy doesn't matter).")
    p.add_argument("--mnn-output", default=str(SCRIPT_DIR / "logs" / "mnn_results.json"),
                    help="Flat list of every question result recorded while on MNN, across all models.")
    p.add_argument("--fallback-file", default=str(SCRIPT_DIR / "logs" / "needs_gguf_fallback.json"),
                    help="Where to write models/questions flagged for Stage 2's GGUF fallback "
                         "(run_fallback_agent_gguf.py reads this same file, via the same flag name, "
                         "as its input).")
    p.add_argument("--summary-output", default=str(SCRIPT_DIR / "logs" / "agent_summary_mnn.json"),
                    help="Per-model status/accuracy summary for Stage 1.")
    # parse_known_args(), not parse_args(): run_pipeline.sh passes the same
    # argv to BOTH stage scripts, and Stage 2 has its own flags (e.g.
    # --gguf-output) this script doesn't define - unknown args are ignored
    # rather than raising, so one shared invocation works for both.
    args, _unknown = p.parse_known_args()
    return args


def main():
    args = parse_args()

    for path, label in REQUIRED_SCRIPTS:
        if not path.exists():
            print(f"[ERROR] {label} not found at {path}")
            sys.exit(1)

    mnn_module = _load_module(RUN_MNN_AUTOBENCH_SCRIPT, "_run_mnn_autobench")
    agent = _load_module(AGENT_QUANTIZE_SCRIPT, "_agent_mnn_quantize")
    mnn_convert = _load_module(CONVERT_TO_MNN_SCRIPT, "_convert_to_mnn")
    quality = _load_module(RESPONSE_QUALITY_SCRIPT, "_response_quality")
    check_fit = _load_module(CHECK_MODEL_FIT_SCRIPT, "_check_model_fit")

    adb_bin = mnn_module.find_adb()
    mnn_adb = mnn_module.Adb(adb_bin)

    print("[PRE-FLIGHT] Checking device and MNN Chat...")
    mnn_module.check_device(mnn_adb)
    mnn_module.check_mnnchat_installed(mnn_adb)

    try:
        variants = load_fit_report(args.fit_report)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to load fit report '{args.fit_report}': {exc}")
        sys.exit(1)
    if not variants:
        print(f"[ERROR] No variants found in '{args.fit_report}' (expected a non-empty 'fits' list).")
        sys.exit(1)

    try:
        questions = load_eval_questions(args.questions)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] Failed to load questions '{args.questions}': {exc}")
        sys.exit(1)
    if not questions:
        print(f"[ERROR] No questions found in '{args.questions}'.")
        sys.exit(1)

    total = len(variants)
    print("=" * 70)
    print("Stage 1: MNN Evaluation")
    print(f"  Pool: {args.fit_report} ({total} models)  Questions: {args.questions} ({len(questions)})  Timeout: {args.timeout}s")
    print(f"  No-think: {'ON' if args.no_think else 'OFF'}  MaxTokens: {args.max_tokens if args.max_tokens is not None else 'default'}")
    print("=" * 70)

    model_results = []
    needs_gguf_entries = []
    for i, variant in enumerate(variants, start=1):
        result, needs_gguf_entry = process_model(mnn_module, mnn_convert, agent, quality, check_fit, mnn_adb, adb_bin,
                                                   variant, questions, i, total, args.timeout, args.no_think, args.max_tokens)
        model_results.append(result)
        if needs_gguf_entry is not None:
            needs_gguf_entries.append(needs_gguf_entry)
        cooldown_between_models(args.cooldown_minutes, i, total)

    mnn_flat = []
    for r in model_results:
        mnn_flat.extend(r["mnn_results"])

    Path(args.mnn_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.mnn_output, "w") as f:
        json.dump(mnn_flat, f, indent=2, default=str)
    print(f"\n[OUTPUT] MNN results saved: {args.mnn_output} ({len(mnn_flat)} question results)")

    save_needs_gguf_fallback(args.fallback_file, needs_gguf_entries)

    succeeded = sum(1 for r in model_results if r["status"] == "success")
    needs_fallback = sum(1 for r in model_results if r["status"] == "needs_gguf_fallback")
    failed = sum(1 for r in model_results if r["status"] == "failed")
    skipped = sum(1 for r in model_results if r["status"] == "skipped")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fit_report": args.fit_report,
        "questions_file": args.questions,
        "total_models": total,
        "succeeded": succeeded,
        "needs_gguf_fallback": needs_fallback,
        "failed": failed,
        "skipped": skipped,
        "models": [
            {
                "model_id": r["model_id"],
                "quant": r["quant"],
                "status": r["status"],
                "error": r["error"],
                "fits_based_on_measured_rss": r["fits_based_on_measured_rss"],
                "mnn_accuracy": r["mnn_accuracy"],
                "total_questions_recorded": r["total_questions_recorded"],
            }
            for r in model_results
        ],
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[OUTPUT] Summary saved: {args.summary_output}")

    print_summary_table(model_results)

    print("\n" + "=" * 70)
    print(f"STAGE 1 DONE - {succeeded} succeeded, {needs_fallback} need GGUF fallback, "
          f"{failed} failed, {skipped} skipped (of {total})")
    print("=" * 70)


if __name__ == "__main__":
    main()
