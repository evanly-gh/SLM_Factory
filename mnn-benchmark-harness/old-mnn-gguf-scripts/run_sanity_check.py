#!/usr/bin/env python3
"""
run_sanity_check.py — Simple, direct pass across models/quants/engines.

Deliberately dumb: no fallback logic, no retry, no garbage-checking-driven
branching (response_quality.GARBAGE_CHECK_ENABLED is currently False anyway,
so that branching would be a no-op regardless). One question run once, one
response recorded, printed in full. The point is a fast eyeball check across
every combination, not a scored benchmark - that's what run_fallback_agent_mnn.py
/run_fallback_agent_gguf.py are for.

Reuses existing infrastructure directly rather than reimplementing it:
  - MNN:  run_fallback_agent_mnn.py's convert_and_push_mnn() (conversion,
          cached via convert_to_mnn.py's own is_export_complete(), + push via
          agent_mnn_quantize.py's push_to_device()) and run_mnn_question()
          (run_mnn_autobench.py's own run_one()/build_metrics()/
          extract_response(), single-question broadcast+poll).
  - GGUF: run_fallback_agent_gguf.py's ensure_gguf_converted() (conversion,
          cached via a local-file-exists check, shelling out to
          convert_to_gguf.py - a fresh top-level process, not the nested-
          subprocess pattern that hangs) and, directly imported from
          run_autobench.py: push_to_app_files_dir(), run_one(),
          build_metrics(), extract_response()/extract_error(). Calling these
          functions in-process (rather than shelling out to run_autobench.py
          itself) sidesteps the confirmed nested-subprocess hang documented
          at length in run_fallback_agent_mnn.py/run_fallback_agent_gguf.py -
          there's no "already-running Python process launches run_autobench.py
          as a subprocess" here, just this script calling adb directly via
          the imported Adb class, same as the MNN side already does.

Usage:
    python3 run_sanity_check.py --questions generic_5_questions.json
    python3 run_sanity_check.py --questions generic_5_questions.json --no-think --max-tokens 2048
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

RUN_FALLBACK_AGENT_MNN_SCRIPT = SCRIPT_DIR / "run_fallback_agent_mnn.py"
RUN_FALLBACK_AGENT_GGUF_SCRIPT = SCRIPT_DIR / "run_fallback_agent_gguf.py"
RUN_MNN_AUTOBENCH_SCRIPT = SCRIPT_DIR / "run_mnn_autobench.py"
AGENT_QUANTIZE_SCRIPT = SCRIPT_DIR / "agent_mnn_quantize.py"
CONVERT_TO_MNN_SCRIPT = SCRIPT_DIR.parent / "Model-Conversion" / "convert_to_mnn.py"

# GGUF/SmolChat side lives in a sibling worktree - same cross-worktree reuse
# pattern already used by run_fallback_agent_gguf.py.
SMOLCHAT_WORKTREE = Path.home() / "SLM_Factory-SmolChat"
RUN_AUTOBENCH_GGUF_SCRIPT = SMOLCHAT_WORKTREE / "Benchmark-Harness" / "run_autobench.py"

REQUIRED_SCRIPTS = [
    (RUN_FALLBACK_AGENT_MNN_SCRIPT, "run_fallback_agent_mnn.py"),
    (RUN_FALLBACK_AGENT_GGUF_SCRIPT, "run_fallback_agent_gguf.py"),
    (RUN_MNN_AUTOBENCH_SCRIPT, "run_mnn_autobench.py"),
    (AGENT_QUANTIZE_SCRIPT, "agent_mnn_quantize.py"),
    (CONVERT_TO_MNN_SCRIPT, "convert_to_mnn.py"),
    (RUN_AUTOBENCH_GGUF_SCRIPT, "run_autobench.py (SmolChat, sibling worktree)"),
]

# One shared GGUF-style quant name per pair, same convention as this
# project's fit reports - run_fallback_agent_mnn.py's QUANT_BIT_MAP maps it
# to MNN's numeric --quant_bit (F16/Q4_K_M/Q8_0 -> 16/4/8, i.e. MNN's own
# F16/Q4/Q8 naming); GGUF/convert_to_gguf.py takes the string as-is.
DEFAULT_PAIRS = [
    {"model_id": "Qwen/Qwen3-0.6B", "quant": "F16"},
    {"model_id": "Qwen/Qwen3-0.6B", "quant": "Q4_K_M"},
    {"model_id": "Qwen/Qwen3-0.6B", "quant": "Q8_0"},
    {"model_id": "Qwen/Qwen3-1.7B", "quant": "F16"},
    {"model_id": "Qwen/Qwen3-1.7B", "quant": "Q4_K_M"},
    {"model_id": "Qwen/Qwen3-1.7B", "quant": "Q8_0"},
]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_questions(path: str) -> list:
    """Parse the eval questions JSON file (a list of {"text":..., "answer":
    ...} objects) and return just the question text strings, in order -
    matching how run_fallback_agent_mnn.py's own load_eval_questions() reads
    this same format. This script doesn't score responses against the
    reference answer (no retry/accept-reject logic at all), so only "text"
    is actually used, but "answer" is still validated/extracted here for
    parity with that format.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list of {{'text':..., 'answer':...}} objects, got {type(data).__name__}")
    for i, q in enumerate(data):
        if "text" not in q or "answer" not in q:
            raise ValueError(f"question at index {i} is missing 'text' or 'answer': {q}")
    return [q["text"] for q in data]


def shorten(text, n=100):
    if not text:
        return text
    return text if len(text) <= n else text[:n] + "..."


# ---------------------------------------------------------------------------
# MNN: one (model, quant) pair, straight through, no retry
# ---------------------------------------------------------------------------

def run_mnn_pair(mnn_fallback, mnn_module, mnn_convert, agent, mnn_adb, adb_bin: str,
                  model_id: str, quant: str, questions: list, timeout: int,
                  no_think: bool, max_tokens) -> dict:
    label = f"{model_id} [{quant}] [mnn]"
    print(f"\n{'-' * 70}\n{label}\n{'-' * 70}")

    quant_bit = mnn_fallback.QUANT_BIT_MAP.get(quant)
    if quant_bit is None:
        print(f"{label} - SKIPPED: unrecognized quant {quant!r}")
        return {"model_id": model_id, "quant": quant, "engine": "mnn",
                "responses": [None] * len(questions), "error": f"unrecognized quant {quant!r}"}

    conv = mnn_fallback.convert_and_push_mnn(mnn_convert, agent, adb_bin, model_id, quant_bit, label)
    if not conv["ok"]:
        print(f"{label} - FAILED (convert/push): {conv['error']}")
        return {"model_id": model_id, "quant": quant, "engine": "mnn",
                "responses": [None] * len(questions), "error": conv["error"]}

    mnn_module.reset_mnnchat_for_clean_process(mnn_adb)

    responses = []
    for i, question_text in enumerate(questions, start=1):
        qlabel = f"{label} Q{i}/{len(questions)}"
        outcome = mnn_fallback.run_mnn_question(mnn_module, mnn_adb, conv["device_path"], question_text,
                                                  i, timeout, no_think, max_tokens)
        response = outcome["response"]
        responses.append(response)
        if outcome["engine_status"] != "done":
            print(f"{qlabel} - {outcome['engine_status'].upper()}: {outcome['error']}")
        else:
            print(f"{qlabel}\n  Q: {question_text}\n  A: {response}\n")

    return {"model_id": model_id, "quant": quant, "engine": "mnn", "responses": responses, "error": None}


# ---------------------------------------------------------------------------
# GGUF: one (model, quant) pair, straight through, no retry
# ---------------------------------------------------------------------------

def run_gguf_pair(gguf_fallback, gguf_module, gguf_adb, model_id: str, quant: str,
                   questions: list, timeout: int, no_think: bool, max_tokens: int) -> dict:
    label = f"{model_id} [{quant}] [gguf]"
    print(f"\n{'-' * 70}\n{label}\n{'-' * 70}")

    conv = gguf_fallback.ensure_gguf_converted(model_id, quant, label)
    if not conv["ok"]:
        print(f"{label} - FAILED (convert): {conv['error']}")
        return {"model_id": model_id, "quant": quant, "engine": "gguf",
                "responses": [None] * len(questions), "error": conv["error"]}

    device_path = gguf_module.push_to_app_files_dir(gguf_adb, conv["local_path"])
    gguf_module.reset_smolchat_for_clean_process(gguf_adb)

    responses = []
    for i, question_text in enumerate(questions, start=1):
        qlabel = f"{label} Q{i}/{len(questions)}"
        prompt = f"{question_text} /no_think" if no_think else question_text
        outcome = gguf_module.run_one(gguf_adb, device_path, prompt, i, timeout, max_tokens)
        if outcome["status"] == "done":
            response = gguf_module.extract_response(outcome["lines"]["RUN_DONE"])
        else:
            response = None
        responses.append(response)
        if outcome["status"] != "done":
            reason, message = (None, None)
            if outcome["status"] == "error" and "RUN_ERROR" in outcome["lines"]:
                reason, message = gguf_module.extract_error(outcome["lines"]["RUN_ERROR"])
            print(f"{qlabel} - {outcome['status'].upper()}: reason={reason} message={message}")
        else:
            print(f"{qlabel}\n  Q: {question_text}\n  A: {response}\n")

    return {"model_id": model_id, "quant": quant, "engine": "gguf", "responses": responses, "error": None}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: list, num_questions: int) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Model':<24} {'Quant':<8} {'Engine':<6} " + " ".join(f"{'Q' + str(i):<25}" for i in range(1, num_questions + 1))
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r["error"] is not None:
            print(f"{r['model_id']:<24} {r['quant']:<8} {r['engine']:<6} ERROR: {shorten(r['error'], 200)}")
            continue
        cells = []
        for resp in r["responses"]:
            cell = shorten(resp, 100) if resp else "(none)"
            cells.append(f"{cell:<25}")
        print(f"{r['model_id']:<24} {r['quant']:<8} {r['engine']:<6} " + " ".join(cells))
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Direct, no-fallback, no-retry sanity pass across models/quants/engines.")
    p.add_argument("--questions", required=True,
                    help="Path to a JSON file: a list of {'text':..., 'answer':...} objects (5 questions expected), "
                         "same format as run_fallback_agent_mnn.py's --questions.")
    p.add_argument("--timeout", type=int, default=180, help="Per-question timeout in seconds (default: 180).")
    p.add_argument("--no-think", action="store_true", dest="no_think", help="Append ' /no_think' to every question. Default: off.")
    p.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens", help="Max tokens per response (default: 4096).")
    p.add_argument("--device", choices=["phone", "emulator"], default="phone", help="GGUF/SmolChat device kind (default: phone).")
    return p.parse_args()


def main():
    args = parse_args()

    for path, label in REQUIRED_SCRIPTS:
        if not path.exists():
            print(f"[ERROR] {label} not found at {path}")
            sys.exit(1)

    mnn_fallback = _load_module(RUN_FALLBACK_AGENT_MNN_SCRIPT, "_run_fallback_agent_mnn")
    gguf_fallback = _load_module(RUN_FALLBACK_AGENT_GGUF_SCRIPT, "_run_fallback_agent_gguf")
    mnn_module = _load_module(RUN_MNN_AUTOBENCH_SCRIPT, "_run_mnn_autobench")
    gguf_module = _load_module(RUN_AUTOBENCH_GGUF_SCRIPT, "_run_autobench_gguf")
    agent = _load_module(AGENT_QUANTIZE_SCRIPT, "_agent_mnn_quantize")
    mnn_convert = _load_module(CONVERT_TO_MNN_SCRIPT, "_convert_to_mnn")

    try:
        questions = load_questions(args.questions)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] Failed to load questions '{args.questions}': {exc}")
        sys.exit(1)
    if not questions:
        print(f"[ERROR] No questions found in '{args.questions}'.")
        sys.exit(1)
    print(f"[OK] Loaded {len(questions)} questions from {args.questions}")

    print("[PRE-FLIGHT] MNN: checking device and MNN Chat...")
    mnn_adb_bin = mnn_module.find_adb()
    mnn_adb = mnn_module.Adb(mnn_adb_bin)
    mnn_module.check_device(mnn_adb)
    mnn_module.check_mnnchat_installed(mnn_adb)

    print("[PRE-FLIGHT] GGUF: checking device and SmolChat...")
    gguf_adb_bin = gguf_module.find_adb()
    gguf_adb = gguf_module.Adb(gguf_adb_bin, args.device)
    gguf_module.check_device(gguf_adb, args.device)
    gguf_module.check_smolchat_installed(gguf_adb)

    print("=" * 70)
    print("Sanity Check: direct pass, no fallback, no retry")
    print(f"  Pairs: {len(DEFAULT_PAIRS)}  Questions: {args.questions} ({len(questions)})  Timeout: {args.timeout}s")
    print(f"  No-think: {'ON' if args.no_think else 'OFF'}  MaxTokens: {args.max_tokens}")
    print("=" * 70)

    all_results = []
    for pair in DEFAULT_PAIRS:
        model_id, quant = pair["model_id"], pair["quant"]
        all_results.append(run_mnn_pair(mnn_fallback, mnn_module, mnn_convert, agent, mnn_adb, mnn_adb_bin,
                                          model_id, quant, questions, args.timeout, args.no_think, args.max_tokens))
        all_results.append(run_gguf_pair(gguf_fallback, gguf_module, gguf_adb,
                                           model_id, quant, questions, args.timeout, args.no_think, args.max_tokens))

    print_summary_table(all_results, len(questions))


if __name__ == "__main__":
    main()
