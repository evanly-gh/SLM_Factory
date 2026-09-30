#!/usr/bin/env python3
"""
energy_latency_agent.py — LATENCY and ENERGY measurement for base MNN
models (Qwen3/Qwen3.5 family), across two cold-load modes.

For each model:
  1. Convert + push once (cached - reuses run_fallback_agent_mnn.py's own
     convert_and_push_mnn(), which is itself cached via convert_to_mnn.py's
     is_export_complete() check).
  2. For each cold-load mode ("cold", "cached"):
       COLD:   force-stop the app ONCE, at the very start of this mode's
               run (reset_mnnchat_for_clean_process()), guaranteeing a
               genuinely fresh app process before any question runs.
       CACHED: do NOT force-stop at all; instead fire one throwaway warmup
               question first (discarded, not recorded) so the app process
               is confirmed warm/responsive before real measurement starts.
     For each question, run it 5 times in a row (no force-stop between
     repeats, in EITHER mode) - repeats 1-4 are unconditionally discarded
     (a "warmup attempt X/5" progress line only), only the 5th is recorded.

Reuses existing infrastructure directly, no duplicated logic:
  - run_fallback_agent_mnn.py : convert_and_push_mnn() (conversion+push,
                                  cached), QUANT_BIT_MAP
  - run_mnn_autobench.py      : find_adb()/Adb, check_device(),
                                  check_mnnchat_installed(),
                                  reset_mnnchat_for_clean_process(),
                                  run_one() (single-question broadcast+poll,
                                  called directly rather than through
                                  run_fallback_agent_mnn.py's
                                  run_mnn_question() wrapper, since this
                                  script needs tag_lines/raw_log access that
                                  wrapper doesn't expose), build_metrics(),
                                  extract_response()/extract_error()

Per-question metrics recorded (5th repeat only):
  - ttft_ms      : straight from build_metrics()
  - ttlt_ms      : (prefill_time_us + decode_time_us) / 1000 - MNN reports
                    prefill/decode as two separate native-timed windows;
                    TTLT ("time to last token") is their sum, i.e. total
                    generation time excluding broadcast/JNI dispatch
                    overhead that ttft_ms's wall-clock measurement includes.
  - energy_mas_sampled / energy_mj_sampled : straight from build_metrics()
    (added there directly, in the immediately preceding change to
    run_mnn_autobench.py - trapezoidal integration over PowerSampler's real
    per-sample timestamps, current-only and current*voltage respectively).
  - backend_requested / backend_actual / backend_mismatch : parsed from the
    MNN_LLM_ACTUAL_BACKEND native log line (no run_id of its own - located
    by searching run_one()'s raw_log, the exact buffer that produced this
    run's verdict, rather than a second separate logcat read that could
    race against the ring buffer evicting that line under heavy per-token
    RESPDEBUG logging).

Usage:
    python3 energy_latency_agent.py --models "Qwen/Qwen3-0.6B:Q4_K_M,Qwen/Qwen3-1.7B:F16" --questions generic_5_questions.txt
    python3 energy_latency_agent.py --models model_pool.json --questions questions.txt --cold-load-mode cold --backend-type vulkan
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

RUN_FALLBACK_AGENT_MNN_SCRIPT = SCRIPT_DIR / "run_fallback_agent_mnn.py"
RUN_MNN_AUTOBENCH_SCRIPT = SCRIPT_DIR / "run_mnn_autobench.py"
AGENT_QUANTIZE_SCRIPT = SCRIPT_DIR / "agent_mnn_quantize.py"
CONVERT_TO_MNN_SCRIPT = SCRIPT_DIR.parent / "Model-Conversion" / "convert_to_mnn.py"
# Only loaded/required when --score-accuracy is set (see main()) - omitting
# the flag needs no NER-scoring machinery at all, matching every other
# purely-additive flag in this script.
NER_METRICS_SCRIPT = SCRIPT_DIR / "ner_metrics.py"

REQUIRED_SCRIPTS = [
    (RUN_FALLBACK_AGENT_MNN_SCRIPT, "run_fallback_agent_mnn.py"),
    (RUN_MNN_AUTOBENCH_SCRIPT, "run_mnn_autobench.py"),
    (AGENT_QUANTIZE_SCRIPT, "agent_mnn_quantize.py"),
    (CONVERT_TO_MNN_SCRIPT, "convert_to_mnn.py"),
]

# Every question runs exactly this many total repeats; the first N-1 are
# unconditionally discarded (warmup) and only the final one is recorded -
# same fixed-attempts concept as run_fallback_agent_mnn.py's/
# run_fallback_agent_gguf.py's own retry mechanism, but implemented
# independently here (no is_garbage()/quality-check logic attached at all -
# this script always keeps the Nth repeat regardless of what it looks like).
REPEATS_PER_QUESTION = 5

BACKEND_LOG_RE = re.compile(
    r"MNN_LLM_ACTUAL_BACKEND.*?requested=(\d+)\((\w+)\)\s+actual=(\d+)\((\w+)\)"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_models_arg(models_arg: str) -> list:
    """Accepts either a JSON pool file (same {'fits': [{'model_id','quant'}]}
    shape run_fallback_agent_mnn.py's load_fit_report() reads) or a plain
    comma-separated 'model_id:quant,model_id:quant' string - whichever the
    caller finds more convenient for a short, focused model list. Returns a
    list of {'model_id', 'quant'} dicts either way.
    """
    path = Path(models_arg)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        variants = data.get("fits", data if isinstance(data, list) else [])
        if not variants:
            raise ValueError(f"no models found in '{models_arg}' (expected a non-empty 'fits' list or a JSON list)")
        return variants

    variants = []
    for pair in models_arg.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"expected 'model_id:quant', got {pair!r} (no ':' found)")
        model_id, quant = pair.rsplit(":", 1)
        variants.append({"model_id": model_id.strip(), "quant": quant.strip()})
    if not variants:
        raise ValueError(f"--models produced no entries: {models_arg!r}")
    return variants


def find_backend_confirmation(raw_log: str):
    """Parses the MNN_LLM_ACTUAL_BACKEND native log line (no run_id of its
    own, since it's logged from llm.cpp's initRuntime(), not the Kotlin
    per-call logging) out of a run's raw_log. Uses the LAST match in the
    buffer - this run's own load() is always the most recent one, since
    every run_one() call is a fully isolated create-load-generate-release
    cycle (confirmed by reading HeadlessBenchmarkRunner.kt) and logcat was
    cleared immediately before this specific broadcast fired.

    Returns {"requested": "cpu", "actual": "cpu", "mismatch": False} or
    None if the line isn't present (e.g. genuinely evicted from the ring
    buffer under unusually heavy per-token logging - a known, rare edge
    case, not silently treated as "no mismatch").
    """
    matches = list(BACKEND_LOG_RE.finditer(raw_log))
    if not matches:
        return None
    _requested_code, requested_name, _actual_code, actual_name = matches[-1].groups()
    requested_name = requested_name.lower()
    actual_name = actual_name.lower()
    return {
        "requested": requested_name,
        "actual": actual_name,
        "mismatch": requested_name != actual_name,
    }


def run_repeated_question(mnn_module, adb, device_path: str, question_text: str, question_number: int,
                           timeout: int, max_tokens, backend_type: str, qlabel: str,
                           ner_metrics=None, gold_entities=None) -> dict:
    """Runs `question_text` REPEATS_PER_QUESTION times in a row (no
    force-stop between repeats), discarding the first REPEATS_PER_QUESTION-1
    and recording only the final one. Returns one per_question_results
    entry.

    `ner_metrics` is the loaded ner_metrics module, or None (the default) to
    skip accuracy scoring entirely - matching --score-accuracy being off,
    the exact pre-existing behavior with no entity_f1/format_valid/
    error_type keys added to the entry at all.
    """
    outcome = None
    for repeat in range(1, REPEATS_PER_QUESTION + 1):
        if repeat < REPEATS_PER_QUESTION:
            print(f"{qlabel} -> warmup attempt {repeat}/{REPEATS_PER_QUESTION}")
        else:
            print(f"{qlabel} -> final attempt {repeat}/{REPEATS_PER_QUESTION} (recording)")
        outcome = mnn_module.run_one(
            adb, device_path, question_text, question_number, timeout,
            max_tokens=max_tokens, backend_type=backend_type,
        )

    entry = {
        "question_number": question_number,
        "question": question_text,
        "run_id": outcome["run_id"],
        "status": None,
        "ttft_ms": None,
        "ttlt_ms": None,
        "energy_mas_sampled": None,
        "energy_mj_sampled": None,
        "backend_requested": None,
        "backend_actual": None,
        "backend_mismatch": None,
        "response": None,
        "error": None,
    }

    backend_info = find_backend_confirmation(outcome["raw_log"])
    if backend_info is not None:
        entry["backend_requested"] = backend_info["requested"]
        entry["backend_actual"] = backend_info["actual"]
        entry["backend_mismatch"] = backend_info["mismatch"]
        if backend_info["mismatch"]:
            print(f"{qlabel} [WARN] BACKEND MISMATCH: requested={backend_info['requested']} "
                  f"actual={backend_info['actual']} - MNN silently fell back to a different backend")
    else:
        print(f"{qlabel} [WARN] MNN_LLM_ACTUAL_BACKEND line not found in this run's log "
              "(possible ring-buffer eviction under heavy logging) - backend_requested/actual left null")

    if outcome["status"] == "done":
        metrics = mnn_module.build_metrics(outcome["tag_lines"])
        response = mnn_module.extract_response(outcome["run_lines"], outcome["run_id"])
        ttlt_ms = None
        if metrics["prefill_time_us"] is not None and metrics["decode_time_us"] is not None:
            ttlt_ms = (metrics["prefill_time_us"] + metrics["decode_time_us"]) / 1000.0

        entry["status"] = "success"
        entry["ttft_ms"] = metrics["ttft_ms"]
        entry["ttlt_ms"] = round(ttlt_ms, 3) if ttlt_ms is not None else None
        entry["energy_mas_sampled"] = metrics["energy_mas_sampled"]
        entry["energy_mj_sampled"] = metrics["energy_mj_sampled"]
        entry["response"] = response

        accuracy_disp = ""
        if ner_metrics is not None:
            # Reuses score()'s own real logic (JSON parsing with regex
            # fallback, the None-vs-[] distinction, failure_category_of())
            # unmodified, just applied to a single-question "eval set" of
            # length 1 rather than a full corpus - entity_f1/format_valid
            # over one example is exactly what score() already computes,
            # nothing reimplemented here.
            example = {"text": question_text, "entities": gold_entities or []}
            preds = ner_metrics.extract_predictions([response], [example])
            result = ner_metrics.score([example], preds)
            entry["entity_f1"] = result["f1"]
            entry["format_valid"] = result["format_valid"]
            entry["error_type"] = result["failures"][0]["error_type"] if result["failures"] else None
            accuracy_disp = f"  F1={entry['entity_f1']:.3f}  ErrorType={entry['error_type'] or 'none'}"

        def fmt(v, unit="", nd=1):
            return f"{v:.{nd}f}{unit}" if isinstance(v, (int, float)) else "N/A"

        print(
            f"{qlabel}  TTFT={fmt(entry['ttft_ms'], 'ms')}  TTLT={fmt(entry['ttlt_ms'], 'ms')}  "
            f"Energy={fmt(entry['energy_mj_sampled'], 'mJ')} ({fmt(entry['energy_mas_sampled'], 'mA*s')})  "
            f"Backend={entry['backend_actual'] or 'unknown'}{accuracy_disp}"
        )
    elif outcome["status"] == "error":
        reason, message = mnn_module.extract_error(outcome["tag_lines"].get("RUN_ERROR", ""))
        entry["status"] = "failed"
        entry["error"] = {"reason": reason, "message": message}
        print(f"{qlabel} - FAILED: reason={reason} message={message}")
    else:  # timeout
        entry["status"] = "failed"
        entry["error"] = {"reason": "timeout", "message": f"No RUN_DONE/RUN_ERROR within {timeout}s"}
        print(f"{qlabel} - FAILED: timeout after {timeout}s")

    return entry


def compute_task_summary(per_question_results: list, score_accuracy: bool = False) -> dict:
    def vals(key):
        return [r[key] for r in per_question_results if r["status"] == "success" and r.get(key) is not None]

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    ttft_vals = vals("ttft_ms")
    ttlt_vals = vals("ttlt_ms")
    energy_mj_vals = vals("energy_mj_sampled")
    energy_mas_vals = vals("energy_mas_sampled")

    summary = {
        "n_questions": len(per_question_results),
        "mean_ttft_ms": mean(ttft_vals),
        "mean_ttlt_ms": mean(ttlt_vals),
        "mean_energy_mj_sampled": mean(energy_mj_vals),
        "total_energy_mj_sampled": round(sum(energy_mj_vals), 3) if energy_mj_vals else None,
        "mean_energy_mas_sampled": mean(energy_mas_vals),
        "total_energy_mas_sampled": round(sum(energy_mas_vals), 3) if energy_mas_vals else None,
    }
    # Only added when --score-accuracy is set - matching the per-question
    # entity_f1/format_valid/error_type fields being similarly absent
    # otherwise, so the JSON schema is byte-for-byte identical to before
    # this flag existed when it's omitted.
    if score_accuracy:
        summary["mean_entity_f1"] = mean(vals("entity_f1"))
    return summary


def run_mode(mnn_module, mnn_fallback, adb, adb_bin, device_path: str, model_id: str, quant: str,
             mode: str, questions: list, timeout: int, max_tokens, backend_type: str, output_dir: Path,
             ner_metrics=None, gold_entities_list=None) -> dict:
    label = f"{model_id} [{quant}] [{mode}]"
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")

    if mode == "cold":
        # Force-stop ONCE, at the very start of this mode's run - guarantees
        # a genuinely fresh app process before question 1's first repeat.
        # Nothing else in this mode's loop force-stops again.
        mnn_module.reset_mnnchat_for_clean_process(adb)
    else:
        # No force-stop at all. One throwaway warmup question first, fully
        # discarded (not recorded anywhere), just to confirm the app process
        # is genuinely warm/responsive before real measurement begins.
        print(f"{label} -> cached-mode warmup question (discarded, not recorded)")
        mnn_module.run_one(adb, device_path, questions[0], 0, timeout, max_tokens=max_tokens, backend_type=backend_type)

    per_question_results = []
    total_q = len(questions)
    for i, question_text in enumerate(questions, start=1):
        qlabel = f"{label} Q{i}/{total_q}"
        gold_entities = gold_entities_list[i - 1] if gold_entities_list else None
        entry = run_repeated_question(mnn_module, adb, device_path, question_text, i, timeout, max_tokens,
                                        backend_type, qlabel, ner_metrics=ner_metrics, gold_entities=gold_entities)
        per_question_results.append(entry)

    task_summary = compute_task_summary(per_question_results, score_accuracy=ner_metrics is not None)

    report = {
        "model_id": model_id,
        "quant": quant,
        "cold_load_mode": mode,
        "per_question_results": per_question_results,
        "task_summary": task_summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model_id.split("/")[-1].lower().replace(".", "").replace("_", "-")
    output_path = output_dir / f"{model_slug}_{quant.lower()}_{mode}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OUTPUT] {label} -> {output_path}")
    f1_disp = f"  mean_entity_f1={task_summary['mean_entity_f1']}" if "mean_entity_f1" in task_summary else ""
    print(f"  n_questions={task_summary['n_questions']}  mean_ttft_ms={task_summary['mean_ttft_ms']}  "
          f"mean_ttlt_ms={task_summary['mean_ttlt_ms']}  mean_energy_mj_sampled={task_summary['mean_energy_mj_sampled']}{f1_disp}")

    return report


def parse_args():
    p = argparse.ArgumentParser(
        description="LATENCY and ENERGY measurement for base MNN models, across cold/cached load modes."
    )
    p.add_argument("--models", required=True,
                    help="Comma-separated 'model_id:quant' pairs (e.g. 'Qwen/Qwen3-0.6B:Q4_K_M,Qwen/Qwen3-1.7B:F16'), "
                         "or a JSON pool file (same {'fits': [{'model_id','quant'}]} shape as run_fallback_agent_mnn.py's --fit-report).")
    p.add_argument("--questions", required=True,
                    help="Path to a .txt file, one question per line (same format as run_mnn_autobench.py's "
                         "--questions) - or, with --score-accuracy, a NER eval-set JSON "
                         "({'pos':[...],'neg':[...],'boundary':[...]}, each entry {'text','entities'}).")
    p.add_argument("--score-accuracy", action="store_true", dest="score_accuracy",
                    help="Score each question's final (5th) response against its real gold NER entities "
                         "(ner_metrics.py's extract_predictions()/score()). Requires --questions to point to "
                         "a NER eval-set JSON (not a plain question-per-line .txt) - adds entity_f1/"
                         "format_valid/error_type to each per-question result and mean_entity_f1 to "
                         "task_summary. Default: off (unchanged behavior, no NER-scoring fields added at all).")
    p.add_argument("--cold-load-mode", choices=["cold", "cached", "both"], default="both", dest="cold_load_mode",
                    help="Which cold-load mode(s) to run each model through, as separate outputs (default: both).")
    p.add_argument("--backend-type", choices=["cpu", "vulkan", "opencl"], default="cpu", dest="backend_type",
                    help="Forces a specific MNN backend via the backend_type broadcast extra (default: cpu).")
    p.add_argument("--timeout", type=int, default=180,
                    help="Seconds to wait for a single question's broadcast result (default: 180).")
    p.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens",
                    help="Max tokens to generate per question (default: 4096).")
    p.add_argument("--output-dir", default=str(SCRIPT_DIR / "energy_latency_results"), dest="output_dir",
                    help="Directory to write one JSON per model+cold_load_mode into.")
    return p.parse_args()


def main():
    args = parse_args()

    for path, label in REQUIRED_SCRIPTS:
        if not path.exists():
            print(f"[ERROR] {label} not found at {path}")
            sys.exit(1)

    try:
        variants = parse_models_arg(args.models)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] Failed to parse --models '{args.models}': {exc}")
        sys.exit(1)

    mnn_module = _load_module(RUN_MNN_AUTOBENCH_SCRIPT, "_run_mnn_autobench")
    mnn_fallback = _load_module(RUN_FALLBACK_AGENT_MNN_SCRIPT, "_run_fallback_agent_mnn")
    agent = _load_module(AGENT_QUANTIZE_SCRIPT, "_agent_mnn_quantize")
    mnn_convert = _load_module(CONVERT_TO_MNN_SCRIPT, "_convert_to_mnn")

    ner_metrics = None
    gold_entities_list = None
    if args.score_accuracy:
        if not NER_METRICS_SCRIPT.exists():
            print(f"[ERROR] ner_metrics.py not found at {NER_METRICS_SCRIPT} (required by --score-accuracy)")
            sys.exit(1)
        ner_metrics = _load_module(NER_METRICS_SCRIPT, "_ner_metrics")
        try:
            eval_rows = ner_metrics.load_ner_eval_set(args.questions)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[ERROR] Failed to load NER eval set '{args.questions}': {exc}")
            sys.exit(1)
        if not eval_rows:
            print(f"[ERROR] No rows found in '{args.questions}' (expected a non-empty pos/neg/boundary JSON).")
            sys.exit(1)
        questions = [row["text"] for row in eval_rows]
        gold_entities_list = [row["entities"] for row in eval_rows]
        print(f"[OK] Loaded {len(questions)} questions with gold entities from {args.questions} (--score-accuracy)")
    else:
        questions = mnn_module.load_questions(args.questions)

    modes = ["cold", "cached"] if args.cold_load_mode == "both" else [args.cold_load_mode]

    # --backend-type mirrors run_mnn_autobench.py's own default-omission
    # rule exactly: "cpu" (the default) is never actually sent as a
    # broadcast extra, since every shipped model's config.json already
    # defaults to cpu - only a genuine override value gets forwarded.
    broadcast_backend_type = args.backend_type if args.backend_type != "cpu" else None

    adb_bin = mnn_module.find_adb()
    adb = mnn_module.Adb(adb_bin)
    print("[PRE-FLIGHT] Checking device and MNN Chat...")
    mnn_module.check_device(adb)
    mnn_module.check_mnnchat_installed(adb)

    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Energy/Latency Agent")
    print(f"  Models: {len(variants)}  Questions: {args.questions} ({len(questions)})  Modes: {modes}  "
          f"Timeout: {args.timeout}s  Backend: {args.backend_type}")
    print("=" * 70)

    all_reports = []
    for variant in variants:
        model_id, quant = variant["model_id"], variant["quant"]
        label = f"{model_id} [{quant}]"
        quant_bit = mnn_fallback.QUANT_BIT_MAP.get(quant)
        if quant_bit is None:
            print(f"{label} - SKIPPED: unrecognized quant {quant!r} (expected Q4_K_M, Q8_0, or F16)")
            continue

        conv = mnn_fallback.convert_and_push_mnn(mnn_convert, agent, adb_bin, model_id, quant_bit, label)
        if not conv["ok"]:
            print(f"{label} - FAILED (convert/push): {conv['error']}")
            continue

        for mode in modes:
            report = run_mode(mnn_module, mnn_fallback, adb, adb_bin, conv["device_path"], model_id, quant,
                               mode, questions, args.timeout, args.max_tokens, broadcast_backend_type, output_dir,
                               ner_metrics=ner_metrics, gold_entities_list=gold_entities_list)
            all_reports.append(report)

    print("\n" + "=" * 70)
    print(f"DONE - {len(all_reports)} model+mode report(s) written to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
