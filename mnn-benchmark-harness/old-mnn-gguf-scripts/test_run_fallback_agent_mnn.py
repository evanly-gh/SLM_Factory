#!/usr/bin/env python3
"""
test_run_fallback_agent_mnn.py — Verifies run_fallback_agent_mnn.py's
per-question logic via a mocked run_mnn_question(), so these scenarios can be
re-run any time without a real device/model.

Current design (NOT "retry until success"): every question ALWAYS runs
exactly TOTAL_ATTEMPTS_PER_QUESTION (6) total attempts. The first
WARMUP_ATTEMPTS (5) are unconditionally discarded regardless of outcome, and
only the FINAL attempt's result is ever recorded into mnn_results - even if
it's also garbage. If the final attempt is garbage, the model still stops
processing further questions and flags the rest for GGUF fallback, unchanged.

Run directly: python3 test_run_fallback_agent_mnn.py
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m1 = _load_module(SCRIPT_DIR / "run_fallback_agent_mnn.py", "_run_fallback_agent_mnn")
quality = _load_module(SCRIPT_DIR / "response_quality.py", "_response_quality")
quality.GARBAGE_CHECK_ENABLED = True
m1.time.sleep = lambda *_a, **_k: None

GOOD_2_PLUS_2 = (
    "Let's work through this carefully. We start with the number two, and we "
    "add another two to it. Combining these two quantities together gives a "
    "final total of 4."
)
GOOD_3_PLUS_3 = (
    "Let's work through this carefully. We start with the number three, and "
    "we add another three to it. Combining these two quantities together "
    "gives a final total of 6."
)


def done_outcome(response):
    return {"engine_status": "done", "response": response, "metrics": {"decode_len": 50},
            "run_id": "r1", "error": None}


def run_with_fake_responses(questions, responses_by_question_number):
    """responses_by_question_number: {question_number: response_text}, returned
    for every attempt of that question. Returns (result, needs_gguf_entry, call_count)."""
    call_count = [0]

    def fake_run_mnn_question(mnn_module, mnn_adb, device_path, question_text, n, timeout, no_think, max_tokens):
        call_count[0] += 1
        return done_outcome(responses_by_question_number[n])

    m1.run_mnn_question = fake_run_mnn_question
    m1.convert_and_push_mnn = lambda *a, **k: {"ok": True, "device_path": "/fake/path"}
    mnn_module = MagicMock()
    mnn_module.reset_mnnchat_for_clean_process = lambda *a, **k: None
    check_fit = MagicMock()

    variant = {"model_id": "Qwen/Qwen3-0.6B", "quant": "Q4_K_M"}
    result, needs_gguf_entry = m1.process_model(
        mnn_module, MagicMock(), MagicMock(), quality, check_fit, MagicMock(), "adb",
        variant, questions, 1, 1, 30, False, None,
    )
    return result, needs_gguf_entry, call_count[0]


def main():
    questions = [
        {"text": "What is 2+2?", "answer": "4"},
        {"text": "What is 3+3?", "answer": "6"},
    ]

    # Scenario 1: both questions' final attempts pass -> success, both
    # recorded, 6 attempts each = 12 total calls.
    r1, entry1, calls1 = run_with_fake_responses(questions, {1: GOOD_2_PLUS_2, 2: GOOD_3_PLUS_3})
    assert r1["status"] == "success"
    assert entry1 is None
    assert len(r1["mnn_results"]) == 2
    assert all(e["is_correct"] for e in r1["mnn_results"])
    assert calls1 == 12, f"expected 12 total calls (6 x 2 questions), got {calls1}"
    print("Scenario 1 PASS - both questions succeed on final attempt, 6 attempts each, status=success")

    # Scenario 2: Q1's final attempt is still garbage after all 6 attempts ->
    # recorded anyway (unconditional), model flagged needs_gguf_fallback,
    # Q2 never attempted.
    r2, entry2, calls2 = run_with_fake_responses(questions, {1: "", 2: GOOD_3_PLUS_3})
    assert r2["status"] == "needs_gguf_fallback"
    assert len(r2["mnn_results"]) == 1, "the garbage final attempt IS recorded now, not dropped"
    assert r2["mnn_results"][0]["question_number"] == 1
    assert r2["mnn_results"][0]["response"] == "", "garbage final attempt's raw response is still recorded as-is"
    assert entry2 is not None
    assert [q["question_number"] for q in entry2["questions"]] == [1, 2], \
        "Q1 (the garbage one, already recorded) stays flagged for GGUF fallback too"
    assert calls2 == 6, f"expected 6 total calls for Q1 alone (Q2 never attempted), got {calls2}"
    print("Scenario 2 PASS - Q1 final attempt still garbage after 6 attempts -> recorded anyway, flagged for fallback, Q2 never attempted")

    print("\nALL SCENARIOS PASSED")


if __name__ == "__main__":
    main()
