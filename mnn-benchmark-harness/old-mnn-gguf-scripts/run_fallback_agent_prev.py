#!/usr/bin/env python3
"""
run_fallback_agent.py — MNN-first, GGUF-fallback evaluation orchestrator.

For each model in a check_model_fit.py fit-report, this runs every question
from an eval set (text + reference answer pairs) against MNN first. If a
response is garbage (response_quality.is_garbage()), the SAME model is
converted to the equivalent GGUF quant level and deployed via SmolChat, the
SAME question is retried there, and - if it succeeds - every remaining
question for that model continues on GGUF. If GGUF is ALSO garbage (either
on the retried question, or on any later question after an earlier
successful switch), the model is marked "failed" and the agent moves on to
the next model, keeping whatever results were already recorded.

Reuses existing, proven infrastructure directly rather than reimplementing
any of it:
  - convert_to_mnn.py       : HF download + llmexport.py MNN export
                               (imported by path, called as functions)
  - agent_mnn_quantize.py   : push_to_device() (MNN) and its shared local
                               model cache directory (imported by path)
  - run_mnn_autobench.py    : find_adb()/Adb, run_one() (single-question
                               broadcast+poll), build_metrics(),
                               extract_response()/extract_error() (imported
                               by path)
  - run_autobench.py        : the GGUF/SmolChat side - invoked as a genuine
    (SmolChat, sibling         SUBPROCESS per question (see "GGUF: run one
     worktree)                question via subprocess" below), NOT imported
                               as a module. Only its Adb/check_smolchat_
                               installed() are still imported by path, for
                               the one-time pre-flight check.
  - response_quality.py     : is_garbage(), score_accuracy(),
                               extract_final_answer() (imported by path)

GGUF/SmolChat runs as a subprocess, not an import, because an earlier
version of this script imported run_autobench.py's resolve_model()/run_one()/
reset_smolchat_for_clean_process() directly and manually re-sequenced them to
mirror its main() flow - this consistently timed out even after one ordering
bug was found and fixed, implying other subtle timing/state assumptions
baked into main()'s exact flow weren't being faithfully replicated by manual
reconstruction. Running run_autobench.py as an actual subprocess sidesteps
that whole class of bug: it's exactly the same code path already proven to
work standalone, no reconstruction involved. The trade-off is that EVERY
GGUF question - not just the first retry - now pays the full download/
convert/push/restart cost of a fresh run_autobench.py invocation, since
there's no shared device_path or process state to reuse between calls. This
only affects the fallback path (triggered occasionally, not on every
question/model), so the reliability trade is worth the added time.

IMPORTANT: convert_to_mnn.py was written as a single-shot CLI tool that
calls sys.exit(1) directly on several failure paths (missing MNNConvert
binary, a failed HF download). Since this script must isolate a failure to
ONE model and continue with the rest of the pool - never abort the whole
run - every call into its conversion/push logic is wrapped in a try/except
that also catches SystemExit and converts it into a normal per-model
failure result. The GGUF/SmolChat subprocess path doesn't need this same
treatment: a subprocess's sys.exit() just becomes a non-zero return code,
which is already handled as an ordinary failure.

SmolChat's confirmed broadcast protocol (model_path/prompt/run_id extras
only, no max_tokens extra) has no equivalent of MNN's --ei max_tokens, so
--max-tokens only affects the MNN engine; a one-time note is printed if the
GGUF fallback is ever actually used while --max-tokens was requested.
--no-think DOES carry over to GGUF, since it's implemented purely as a
" /no_think" suffix appended to the prompt text (a Qwen3 chat-template
behavior, not an app-specific broadcast flag), which works identically
regardless of which app relays the prompt to the model.

Usage:
    python3 run_fallback_agent.py --fit-report model_fit_report_q4q8.json --questions eval_questions_phase3.json
"""

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolved once at import time: "timeout" is the standard name on Linux,
# "gtimeout" is what GNU coreutils installs it as on macOS (to avoid
# clobbering the BSD toolset) - e.g. via `brew install coreutils`. Confirmed
# absent by default in this environment; installed specifically for the
# os.system()-based invocation below.
TIMEOUT_BIN = shutil.which("timeout") or shutil.which("gtimeout")

SCRIPT_DIR = Path(__file__).resolve().parent

RUN_MNN_AUTOBENCH_SCRIPT = SCRIPT_DIR / "run_mnn_autobench.py"
AGENT_QUANTIZE_SCRIPT = SCRIPT_DIR / "agent_mnn_quantize.py"
RESPONSE_QUALITY_SCRIPT = SCRIPT_DIR / "response_quality.py"
CONVERT_TO_MNN_SCRIPT = SCRIPT_DIR.parent / "Model-Conversion" / "convert_to_mnn.py"

# The GGUF/SmolChat fallback path lives in a sibling worktree, not this repo -
# same cross-worktree reuse pattern already used elsewhere in this project.
SMOLCHAT_WORKTREE = Path.home() / "SLM_Factory-SmolChat"
RUN_AUTOBENCH_GGUF_SCRIPT = SMOLCHAT_WORKTREE / "Benchmark-Harness" / "run_autobench.py"
CONVERT_TO_GGUF_SCRIPT = SMOLCHAT_WORKTREE / "Model-Conversion" / "convert_to_gguf.py"

REQUIRED_SCRIPTS = [
    (RUN_MNN_AUTOBENCH_SCRIPT, "run_mnn_autobench.py"),
    (AGENT_QUANTIZE_SCRIPT, "agent_mnn_quantize.py"),
    (RESPONSE_QUALITY_SCRIPT, "response_quality.py"),
    (CONVERT_TO_MNN_SCRIPT, "convert_to_mnn.py"),
    (RUN_AUTOBENCH_GGUF_SCRIPT, "run_autobench.py (SmolChat, sibling worktree)"),
    (CONVERT_TO_GGUF_SCRIPT, "convert_to_gguf.py (SmolChat, sibling worktree)"),
]

# check_model_fit.py's "fits" list carries GGUF-style quant names (its own
# SUPPORTED_QUANTS = {"Q4_K_M", "Q8_0"}) even for entries destined for MNN -
# this is the one place that name gets mapped to MNN's numeric --quant_bit.
# Going the other way (MNN -> GGUF fallback) needs no inverse mapping: the
# variant's own "quant" field is already the correct GGUF quant name as-is.
QUANT_BIT_MAP = {"Q4_K_M": 4, "Q8_0": 8}


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


def shorten(text, n=120):
    if not text:
        return text
    return text if len(text) <= n else text[: n - 3] + "..."


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
    outcome = mnn_module.run_one(
        mnn_adb, device_path, question_text, n, timeout,
        no_think=no_think, max_tokens=(max_tokens if max_tokens is not None else 256),
    )
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
# GGUF: run one question via a genuine run_autobench.py subprocess
# ---------------------------------------------------------------------------

# Cap on how much of a failed subprocess's stdout/stderr gets embedded in an
# error message - enough to actually diagnose the failure, not so much that
# one bad question buries the rest of the run's output.
_SUBPROCESS_OUTPUT_TAIL_CHARS = 2000


def _tail(text, n=_SUBPROCESS_OUTPUT_TAIL_CHARS):
    text = text or ""
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def gguf_model_prefix(model_id: str) -> str:
    """Mirror convert_to_gguf.py's own model_prefix() exactly, for the HF-ID
    case (the only case this agent ever passes in - variant["model_id"] is
    always a HuggingFace ID, never a local path)."""
    return model_id.split("/")[-1].lower().replace("_", "-").replace(" ", "-")


def gguf_local_path(model_id: str, quant: str) -> Path:
    """The local .gguf file path convert_to_gguf.py produces for this
    model_id/quant, using its own naming convention exactly:
    <SmolChat-worktree>/Model-Conversion/output-<prefix>/<prefix>-<quant-lower>.gguf
    (matches run_autobench.py's own convert_to_gguf() wrapper, which passes
    this same "output-<prefix>" directory as --output; the filename itself
    comes from convert_to_gguf.py's quantize_model(), which always names
    each level's file f"{prefix}-{level.lower()}.gguf")."""
    prefix = gguf_model_prefix(model_id)
    output_dir = CONVERT_TO_GGUF_SCRIPT.parent / f"output-{prefix}"
    return output_dir / f"{prefix}-{quant.lower()}.gguf"


def ensure_gguf_converted(model_id: str, quant: str, label: str) -> dict:
    """Guarantee a local, already-converted .gguf file exists for this
    model_id/quant, converting it directly via convert_to_gguf.py (WITHOUT
    --deploy - this only needs to produce the local file, not push it) if
    it doesn't already exist.

    This exists because passing a bare model_id as run_autobench.py's
    --model forces its resolve_model() through the full download+convert+
    deploy chain internally, which has been confirmed to hang/fail when
    invoked that way - whereas passing an already-existing LOCAL .gguf path
    is confirmed reliable every time: resolve_model() detects the .gguf
    extension and just pushes the file directly, skipping convert_to_gguf()
    entirely. Converting it ourselves first, then handing run_autobench.py
    only the resulting local path, gets that same reliable path.

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


def run_gguf_question_subprocess(model_id: str, quant: str, question_text: str, n: int, timeout: int,
                                  no_think: bool, label: str) -> dict:
    """Run exactly ONE question against GGUF/SmolChat by invoking
    run_autobench.py as a genuine subprocess (its own full main() flow:
    push, force-restart+settle, single broadcast+poll) - not by importing
    and re-sequencing its internal functions. See the module docstring for
    why: manual reconstruction of that sequence proved unreliable even
    after fixing a found ordering bug, while the actual script is already
    proven to work standalone.

    The model is converted to a local .gguf file first (ensure_gguf_
    converted(), reusing it if already present) and THAT LOCAL PATH is
    passed as run_autobench.py's --model - not the bare model_id. A bare
    model_id forces run_autobench.py's own resolve_model() through its full
    download+convert+deploy chain internally, which is confirmed to hang/
    fail; a local .gguf path takes resolve_model()'s other branch (just
    push the file), confirmed reliable every time this session.

    --no-think is applied as a plain prompt-text suffix in the temp
    questions file, the same way it always has been - it's a Qwen3
    chat-template behavior triggered by the prompt text itself, not
    something run_autobench.py's CLI has a dedicated flag for.

    Returns the same shape as before, so callers don't need to change:
    {"engine_status": "done"|"error"|"timeout", "response": str|None,
    "metrics": dict|None, "run_id": str|None, "error": dict|None}.
    """
    conv = ensure_gguf_converted(model_id, quant, label)
    if not conv["ok"]:
        return {"engine_status": "error", "response": None, "metrics": None, "run_id": None,
                "error": {"reason": "gguf_conversion_failed", "message": conv["error"]}}

    prompt_text = f"{question_text} /no_think" if no_think else question_text

    with tempfile.TemporaryDirectory(prefix="fallback_gguf_") as tmpdir:
        tmpdir = Path(tmpdir)
        questions_path = tmpdir / "question.txt"
        output_path = tmpdir / "output.json"
        questions_path.write_text(prompt_text + "\n", encoding="utf-8")

        if TIMEOUT_BIN is None:
            return {"engine_status": "error", "response": None, "metrics": None, "run_id": None,
                    "error": {"reason": "timeout_util_missing",
                              "message": "Neither 'timeout' nor 'gtimeout' is on PATH - install GNU "
                                         "coreutils (e.g. `brew install coreutils` on macOS) to provide it."}}

        # os.system() goes through /bin/sh -c directly - a genuinely
        # different, lower-level invocation mechanism than subprocess.run(),
        # tried after capture_output, stdin handling, and start_new_session
        # were each ruled out individually as the hang's cause across 4
        # prior subprocess.run()-based attempts. Each argument is
        # shlex.quote()'d before being joined into the shell string, since
        # os.system() has no argv-list form - unlike subprocess.run(cmd),
        # this command is genuinely parsed by a shell, so unquoted paths
        # would be vulnerable to word-splitting/injection.
        #
        # os.system() gives no Python-level handle to enforce a timeout, so
        # the OS's own `timeout`/`gtimeout` utility wraps the command
        # instead and enforces the limit itself, killing the child directly
        # if it's exceeded (exit code 124, the utility's own convention).
        inner_cmd = " ".join(shlex.quote(str(part)) for part in [
            sys.executable, "-u", str(RUN_AUTOBENCH_GGUF_SCRIPT),
            "--model", str(conv["local_path"]),
            "--quant", quant,
            "--questions", str(questions_path),
            "--output", str(output_path),
            "--timeout", str(timeout),
        ])
        full_cmd = f"{shlex.quote(TIMEOUT_BIN)} {timeout} {inner_cmd}"
        print(f"{label} Q{n} [gguf-os.system] - RUN: {full_cmd}")

        # Output-file polling/parsing below is unchanged from the
        # subprocess.run() version - only the launch mechanism above it has
        # changed. os.system() blocks until the shell (and therefore the
        # `timeout`-wrapped child) exits, same as subprocess.run() did.
        deadline = time.time() + timeout
        wait_status = os.system(full_cmd)
        returncode = os.waitstatus_to_exitcode(wait_status)
        subprocess_timed_out = returncode == 124

        data = None
        while time.time() < deadline:
            if output_path.exists():
                try:
                    data = json.loads(output_path.read_text())
                    break
                except (json.JSONDecodeError, OSError):
                    pass  # file exists but not fully flushed yet - keep polling
            time.sleep(1)

        if data is None:
            if subprocess_timed_out:
                return {"engine_status": "timeout", "response": None, "metrics": None, "run_id": None,
                        "error": {"reason": "subprocess_timeout",
                                  "message": f"run_autobench.py did not finish within {timeout}s and no valid "
                                             f"{output_path} appeared (see terminal output above for details)."}}
            return {"engine_status": "error", "response": None, "metrics": None, "run_id": None,
                    "error": {"reason": "no_output_file",
                              "message": f"run_autobench.py exited (code={returncode}) but no valid "
                                         f"{output_path} appeared within {timeout}s "
                                         f"(see terminal output above for details)."}}

        results_list = data.get("results", [])
        if not results_list:
            return {"engine_status": "error", "response": None, "metrics": None, "run_id": None,
                    "error": {"reason": "no_results",
                              "message": f"run_autobench.py produced no per-question results (run_info={data.get('run_info')})"}}

        q_result = results_list[0]  # single-question file -> exactly one entry
        run_id = q_result.get("run_id")

        if q_result.get("status") != "success":
            q_error = q_result.get("error") or {}
            return {"engine_status": "error", "response": None, "metrics": None, "run_id": run_id,
                    "error": {"reason": q_error.get("reason"), "message": q_error.get("message")}}

        raw_metrics = q_result.get("metrics") or {}
        # Renamed to match MNN's own metrics field names for consistency in
        # the output JSON - SmolChat/GGUF only ever reports one combined
        # "tps" (not separate prefill/decode), so decode_tps here is that
        # same single number, not a genuinely decode-only measurement.
        metrics = {
            "ttft_ms": raw_metrics.get("ttft_ms"),
            "decode_tps": raw_metrics.get("tps"),
            "peak_rss_kb": raw_metrics.get("memory_kb"),
            "power_ma": raw_metrics.get("power_ma"),
            "thermal_cpu_c": raw_metrics.get("thermal_temp_cpu_c"),
            "thermal_skin_c": raw_metrics.get("thermal_temp_skin_c"),
        }
        response = q_result.get("response")
        return {"engine_status": "done", "response": response, "metrics": metrics, "run_id": run_id, "error": None}


# ---------------------------------------------------------------------------
# Per-model state machine
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
    """Compute combined_accuracy across BOTH engines' recorded results for
    this model - every recorded result counts, regardless of which engine
    produced it. Called on every return path (skip/failed/success) so
    downstream summary code never has to special-case a missing key."""
    all_entries = result["mnn_results"] + result["gguf_results"]
    total = len(all_entries)
    correct = sum(1 for e in all_entries if e["is_correct"])
    result["combined_accuracy"] = (correct / total) if total > 0 else None
    result["total_questions_recorded"] = total
    return result


def process_model(mnn_module, mnn_convert, agent, quality, mnn_adb, adb_bin: str,
                   variant: dict, questions: list, index: int, total: int, timeout: int,
                   no_think: bool, max_tokens, warn_state: dict) -> dict:
    model_id = variant.get("model_id")
    quant = variant.get("quant")
    label = f"[{index}/{total}] {model_id} [{quant}]"

    result = {
        "model_id": model_id, "quant": quant, "status": None, "error": None,
        "engines_used": [], "mnn_results": [], "gguf_results": [],
    }

    quant_bit = QUANT_BIT_MAP.get(quant)
    if quant_bit is None:
        print(f"{label} - SKIPPED: unrecognized quant {quant!r} (expected Q4_K_M or Q8_0)")
        result["status"] = "skipped"
        result["error"] = f"unrecognized quant value {quant!r}"
        return finish(result)

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")

    conv = convert_and_push_mnn(mnn_convert, agent, adb_bin, model_id, quant_bit, label)
    if not conv["ok"]:
        print(f"{label} - FAILED (MNN setup): {conv['error']}")
        result["status"] = "failed"
        result["error"] = f"MNN conversion/push failed: {conv['error']}"
        return finish(result)

    mnn_module.reset_mnnchat_for_clean_process(mnn_adb)
    mnn_device_path = conv["device_path"]
    current_engine = "mnn"
    result["engines_used"].append("mnn")

    total_q = len(questions)
    for i, q in enumerate(questions, start=1):
        question_text, reference_answer = q["text"], q["answer"]
        qlabel = f"{label} Q{i}/{total_q} [{current_engine}]"

        if current_engine == "mnn":
            outcome = run_mnn_question(mnn_module, mnn_adb, mnn_device_path, question_text, i, timeout, no_think, max_tokens)
        else:
            outcome = run_gguf_question_subprocess(model_id, quant, question_text, i, timeout, no_think, label)

        response = outcome["response"]
        metrics = outcome["metrics"] or {}
        # A hard engine error/timeout is treated the same as garbage output
        # for fallback purposes - either way this question produced nothing
        # usable on the current engine.
        garbage = outcome["engine_status"] != "done" or quality.is_garbage(response, metrics)

        if outcome["engine_status"] != "done":
            print(f"{qlabel} - {outcome['engine_status'].upper()}: {outcome['error']}")
        elif garbage:
            print(f"{qlabel} - GARBAGE detected: {shorten(response)!r}")

        if garbage:
            if current_engine == "mnn":
                print(f"{label} - MNN produced garbage on Q{i}, falling back to GGUF for the rest of this model...")

                if "gguf" not in result["engines_used"]:
                    # 5 different subprocess invocation mechanisms all hung
                    # identically at the same point, ruling out the
                    # invocation mechanism itself - the one true constant
                    # was extensive prior ADB activity in the same parent
                    # process/session before every failure, vs. a fresh
                    # session with no prior ADB usage before every success.
                    # Killing the shared adb server here tests/clears that:
                    # it restarts clean on the very next adb command issued.
                    # Allowed to fail silently (e.g. if the server wasn't
                    # even running) - this is a best-effort reset, not a
                    # required step.
                    print(f"{label} - killing adb server (clears any bad shared-daemon state before switching to GGUF)...")
                    mnn_adb.run(["kill-server"], timeout=15)

                    # Confirmed root cause (manual testing): MNN Chat stays
                    # resident on the phone after its own inference attempt,
                    # and SmolChat trying to load a second large model
                    # concurrently causes genuine resource contention that
                    # hangs the GGUF subprocess. Force-stopping MNN Chat here
                    # - once, at the actual engine-switch moment, not before
                    # every GGUF question - resolved it completely.
                    print(f"{label} - force-stopping MNN Chat ({mnn_module.PACKAGE}) before switching to GGUF...")
                    mnn_adb.run(["shell", "am", "force-stop", mnn_module.PACKAGE], timeout=15)
                    result["engines_used"].append("gguf")
                    if max_tokens is not None and not warn_state["gguf_max_tokens_warned"]:
                        print(
                            "[NOTE] --max-tokens has no effect on the GGUF/SmolChat fallback path - "
                            "SmolChat's confirmed broadcast protocol has no max_tokens extra."
                        )
                        warn_state["gguf_max_tokens_warned"] = True

                current_engine = "gguf"
                # Every GGUF question - including this first retry - runs
                # via a fresh run_autobench.py subprocess, which handles its
                # own download/convert/push/restart/settle internally. No
                # separate setup step is needed here anymore.
                retry_outcome = run_gguf_question_subprocess(model_id, quant, question_text, i, timeout, no_think, label)
                retry_response = retry_outcome["response"]
                retry_metrics = retry_outcome["metrics"] or {}
                retry_garbage = retry_outcome["engine_status"] != "done" or quality.is_garbage(retry_response, retry_metrics)

                if retry_outcome["engine_status"] != "done":
                    print(f"{label} Q{i}/{total_q} [gguf-retry] - {retry_outcome['engine_status'].upper()}: {retry_outcome['error']}")
                elif retry_garbage:
                    print(f"{label} Q{i}/{total_q} [gguf-retry] - GARBAGE detected: {shorten(retry_response)!r}")

                if retry_garbage:
                    print(f"{label} - GGUF ALSO garbage on Q{i} - marking model FAILED, stopping (keeping {len(result['mnn_results'])} MNN + {len(result['gguf_results'])} GGUF results already recorded).")
                    result["status"] = "failed"
                    result["error"] = f"double-garbage on Q{i}: MNN and GGUF both produced unusable output"
                    return finish(result)

                is_correct = quality.score_accuracy(retry_response, reference_answer)
                entry = build_result_entry(quality, model_id, quant, "gguf", i, question_text, reference_answer,
                                            retry_response, is_correct, retry_metrics, retry_outcome["run_id"])
                result["gguf_results"].append(entry)
                print(f"{qlabel} -> [gguf-retry] recovered, correct={is_correct}  response: {shorten(retry_response)!r}")
                time.sleep(2)
                continue

            else:  # current_engine == "gguf", garbage on a LATER question after an earlier successful switch
                print(f"{label} - GGUF garbage on Q{i} (already on GGUF from an earlier switch this model) - marking model FAILED, stopping (keeping {len(result['mnn_results'])} MNN + {len(result['gguf_results'])} GGUF results already recorded).")
                result["status"] = "failed"
                result["error"] = f"GGUF garbage on Q{i} after switching from MNN earlier in this model's run"
                return finish(result)

        # Not garbage: score and record on the CURRENT engine, continue.
        is_correct = quality.score_accuracy(response, reference_answer)
        entry = build_result_entry(quality, model_id, quant, current_engine, i, question_text, reference_answer,
                                    response, is_correct, metrics, outcome["run_id"])
        if current_engine == "mnn":
            result["mnn_results"].append(entry)
        else:
            result["gguf_results"].append(entry)
        print(f"{qlabel} -> correct={is_correct}  response: {shorten(response)!r}")
        time.sleep(2)

    result["status"] = "success"
    return finish(result)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(model_results: list):
    print("\n" + "=" * 100)
    print("FALLBACK AGENT SUMMARY")
    print("=" * 100)
    header = f"{'Model':<32}{'Quant':<10}{'Status':<10}{'Engines':<14}{'Accuracy':>10}{'N':>6}"
    print(header)
    print("-" * len(header))
    for r in model_results:
        acc = r.get("combined_accuracy")
        acc_disp = f"{acc * 100:.1f}%" if acc is not None else "N/A"
        row = (
            f"{str(r['model_id']):<32}"
            f"{str(r['quant']):<10}"
            f"{r['status']:<10}"
            f"{(','.join(r['engines_used']) or '-'):<14}"
            f"{acc_disp:>10}"
            f"{r.get('total_questions_recorded', 0):>6}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MNN-first, GGUF-fallback evaluation orchestrator across a check_model_fit.py pool."
    )
    p.add_argument("--fit-report", required=True,
                    help="Path to a check_model_fit.py fit-report JSON; its 'fits' list is the pool to process.")
    p.add_argument("--questions", required=True,
                    help="Path to an eval_questions_phase3.json-style JSON: a list of {'text':..., 'answer':...} objects.")
    p.add_argument("--timeout", type=int, default=180,
                    help="Seconds to wait for a single question's broadcast result, on either engine (default: 180).")
    p.add_argument("--no-think", action="store_true", dest="no_think",
                    help="Append ' /no_think' to every question's prompt, on both engines. Default: off.")
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                    help="Passed through to MNN's --ei max_tokens extra. Has no effect on the GGUF/SmolChat "
                         "fallback path - SmolChat's broadcast protocol has no max_tokens extra.")
    p.add_argument("--mnn-output", default="mnn_results.json",
                    help="Flat list of every question result recorded while on MNN, across all models.")
    p.add_argument("--gguf-output", default="gguf_results.json",
                    help="Flat list of every question result recorded while on GGUF, across all models.")
    p.add_argument("--summary-output", default="agent_summary.json",
                    help="Per-model status/engines/accuracy summary.")
    return p.parse_args()


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
    gguf_module = _load_module(RUN_AUTOBENCH_GGUF_SCRIPT, "_run_autobench_gguf")

    adb_bin = mnn_module.find_adb()
    mnn_adb = mnn_module.Adb(adb_bin)
    gguf_adb = gguf_module.Adb(adb_bin, "phone")

    print("[PRE-FLIGHT] Checking device and apps...")
    mnn_module.check_device(mnn_adb)
    mnn_module.check_mnnchat_installed(mnn_adb)
    try:
        gguf_module.check_smolchat_installed(gguf_adb)
    except SystemExit:
        print("[WARN] SmolChat is not installed - the GGUF fallback path will fail for any model that needs it.")

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
    print("MNN -> GGUF Fallback Agent")
    print(f"  Pool: {args.fit_report} ({total} models)  Questions: {args.questions} ({len(questions)})  Timeout: {args.timeout}s")
    print(f"  No-think: {'ON' if args.no_think else 'OFF'}  MaxTokens (MNN only): {args.max_tokens if args.max_tokens is not None else 'default'}")
    print("=" * 70)

    warn_state = {"gguf_max_tokens_warned": False}
    model_results = []
    for i, variant in enumerate(variants, start=1):
        r = process_model(mnn_module, mnn_convert, agent, quality, mnn_adb, adb_bin,
                           variant, questions, i, total, args.timeout, args.no_think, args.max_tokens, warn_state)
        model_results.append(r)

    mnn_flat, gguf_flat = [], []
    for r in model_results:
        mnn_flat.extend(r["mnn_results"])
        gguf_flat.extend(r["gguf_results"])

    with open(args.mnn_output, "w") as f:
        json.dump(mnn_flat, f, indent=2, default=str)
    print(f"\n[OUTPUT] MNN results saved: {args.mnn_output} ({len(mnn_flat)} question results)")

    with open(args.gguf_output, "w") as f:
        json.dump(gguf_flat, f, indent=2, default=str)
    print(f"[OUTPUT] GGUF results saved: {args.gguf_output} ({len(gguf_flat)} question results)")

    succeeded = sum(1 for r in model_results if r["status"] == "success")
    failed = sum(1 for r in model_results if r["status"] == "failed")
    skipped = sum(1 for r in model_results if r["status"] == "skipped")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fit_report": args.fit_report,
        "questions_file": args.questions,
        "total_models": total,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "models": [
            {
                "model_id": r["model_id"],
                "quant": r["quant"],
                "status": r["status"],
                "error": r["error"],
                "engines_used": r["engines_used"],
                "combined_accuracy": r["combined_accuracy"],
                "total_questions_recorded": r["total_questions_recorded"],
            }
            for r in model_results
        ],
    }
    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[OUTPUT] Summary saved: {args.summary_output}")

    print_summary_table(model_results)

    print("\n" + "=" * 70)
    print(f"DONE - {succeeded} succeeded, {failed} failed, {skipped} skipped (of {total})")
    print("=" * 70)


if __name__ == "__main__":
    main()
