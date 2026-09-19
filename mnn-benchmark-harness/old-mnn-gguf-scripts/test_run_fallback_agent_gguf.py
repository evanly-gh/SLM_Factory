#!/usr/bin/env python3
"""
test_run_fallback_agent_gguf.py — Verifies run_fallback_agent_gguf.py's
per-question logic via a mocked subprocess.run(), so these scenarios can be
re-run any time without a real device/model.

Current design (NOT "retry until success"): every question ALWAYS runs
exactly TOTAL_ATTEMPTS_PER_QUESTION (6) total attempts. The first
WARMUP_ATTEMPTS (5) are unconditionally discarded regardless of outcome,
and only the FINAL attempt's result is ever recorded - even if it's also
garbage. The per-batch stop-on-garbage rule (mark the model "failed", don't
attempt any later flagged questions) still applies based on that final
attempt's is_garbage() result, unchanged.

Run directly: python3 test_run_fallback_agent_gguf.py
"""

import importlib.util
import json
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m2 = _load_module(SCRIPT_DIR / "run_fallback_agent_gguf.py", "_run_fallback_agent_gguf")
quality = m2._load_module(m2.RESPONSE_QUALITY_SCRIPT, "_response_quality")

# Real garbage-checking must be active for these tests to mean anything -
# GARBAGE_CHECK_ENABLED defaults to False in production right now (a
# separate, deliberate kill switch), which would make every non-length rule
# a no-op and defeat the point of these scenarios.
quality.GARBAGE_CHECK_ENABLED = True

# Conversion is mocked out for every scenario - these tests are about the
# per-question garbage-checking/attempt/stop logic, not GGUF conversion.
m2.ensure_gguf_converted = lambda model_id, quant, label: {"ok": True, "local_path": Path("/tmp/fake.gguf")}

# Realistic, non-repetitive "good" response text - long enough (150+ chars)
# to clear is_garbage()'s always-active length-based rule, and varied enough
# to not trip its independent repeated-substring rule either.
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
GOOD_4_PLUS_4 = (
    "Let's work through this carefully. We start with the number four, and "
    "we add another four to it. Combining these two quantities together "
    "gives a final total of 8."
)


def success_result(rid, response, tps=10.0):
    return {"status": "success", "run_id": rid, "response": response, "metrics": {"tps": tps}}


def run_with_fake_results(flagged_model, responses_by_question_number):
    """Runs process_flagged_model() with subprocess.run() mocked.

    responses_by_question_number: {question_number: response_text}. The
    SAME response is returned for every attempt of that question (all 6
    attempts, including the shared initial batch call) - sufficient for
    these scenarios since none of them depend on a response that changes
    attempt-to-attempt, only on whether the FINAL (always-recorded) result
    is garbage or not. Returns (result, call_count).
    """
    orig_run = subprocess.run
    call_count = [0]

    def fake_run(cmd, timeout=None):
        call_count[0] += 1
        output_path = cmd[cmd.index("--output") + 1]
        questions_path = cmd[cmd.index("--questions") + 1]
        num_questions_this_call = len(Path(questions_path).read_text().strip().splitlines())

        # Reconstruct which question(s) this call covers, in order, by
        # matching against the full flagged list (batch call covers all of
        # them; a retry call covers exactly one) via the actual question
        # TEXT written into the temp file, which is unique per question.
        all_qns = [q["question_number"] for q in flagged_model["questions"]]
        if num_questions_this_call == len(all_qns):
            qns_this_call = all_qns
        else:
            written_lines = Path(questions_path).read_text().splitlines()
            qns_this_call = [
                q["question_number"] for q in flagged_model["questions"]
                if any(q["text"] in line for line in written_lines)
            ]
            assert len(qns_this_call) == 1, f"expected exactly one matching question, got {qns_this_call}"

        results = [
            success_result(f"r{call_count[0]}_{qn}", responses_by_question_number[qn])
            for qn in qns_this_call
        ]
        with open(output_path, "w") as f:
            json.dump({"run_info": {"completed": len(results), "total": len(results)}, "results": results}, f)
        class R:
            returncode = 0
        return R()

    subprocess.run = fake_run
    try:
        result = m2.process_flagged_model(quality, flagged_model, 1, 1, 30, False, None)
        return result, call_count[0]
    finally:
        subprocess.run = orig_run


def main():
    # -------------------------------------------------------------------
    # Scenario 1: every flagged question's FINAL attempt passes -> whole
    # model succeeds, everything recorded. 2 questions x 6 attempts each =
    # 12 total subprocess calls (1 shared batch call + 5 single-question
    # retries per question).
    # -------------------------------------------------------------------
    flagged = {
        "model_id": "Qwen/Qwen3-0.6B", "quant": "Q4_K_M",
        "questions": [
            {"question_number": 5, "text": "What is 2+2?", "answer": "4"},
            {"question_number": 6, "text": "What is 3+3?", "answer": "6"},
        ],
    }
    r1, calls1 = run_with_fake_results(flagged, {5: GOOD_2_PLUS_2, 6: GOOD_3_PLUS_3})
    assert r1["status"] == "success"
    assert len(r1["gguf_results"]) == 2
    assert [e["question_number"] for e in r1["gguf_results"]] == [5, 6]
    assert all(e["is_correct"] for e in r1["gguf_results"])
    assert all(e["is_garbage"] is False for e in r1["gguf_results"])
    assert calls1 == 1 + 5 + 5, f"expected 11 total calls (1 batch + 5 retries x 2 questions), got {calls1}"
    print("Scenario 1 PASS - all questions' final attempts pass, all recorded, status=success, 6 attempts each")

    # -------------------------------------------------------------------
    # Scenario 2: the FIRST flagged question's FINAL attempt (after all 6
    # attempts, every one of which is garbage here) is ALSO garbage ->
    # model failed. Under the current design this entry IS still recorded
    # (unconditional recording of the final attempt), unlike the old
    # "retry until success" design where a garbage question was simply
    # dropped. The second question is never attempted at all.
    # -------------------------------------------------------------------
    r2, calls2 = run_with_fake_results(flagged, {5: "", 6: GOOD_3_PLUS_3})
    assert r2["status"] == "failed"
    assert len(r2["gguf_results"]) == 1, "the garbage final attempt IS recorded now, not dropped"
    assert r2["gguf_results"][0]["question_number"] == 5
    assert r2["gguf_results"][0]["is_garbage"] is True
    assert "Q5" in r2["error"]
    assert calls2 == 1 + 5, f"expected 6 total calls for Q5 alone (Q6 never attempted), got {calls2}"
    print("Scenario 2 PASS - Q5's final attempt still garbage after 6 attempts -> recorded anyway, failed, Q6 never attempted")

    # -------------------------------------------------------------------
    # Scenario 3: the FIRST flagged question's final attempt SUCCEEDS, the
    # SECOND's final attempt is garbage (every one of its 6 attempts is
    # garbage here) -> stop there. Q1 (genuinely successful) is kept, Q2
    # (garbage) IS recorded anyway (new behavior), and Q3 is never
    # attempted/recorded at all.
    # -------------------------------------------------------------------
    flagged_three = {
        "model_id": "Qwen/Qwen3-1.7B", "quant": "Q8_0",
        "questions": [
            {"question_number": 5, "text": "What is 2+2?", "answer": "4"},
            {"question_number": 6, "text": "What is 3+3?", "answer": "6"},
            {"question_number": 7, "text": "What is 4+4?", "answer": "8"},
        ],
    }
    r3, calls3 = run_with_fake_results(flagged_three, {5: GOOD_2_PLUS_2, 6: "", 7: GOOD_4_PLUS_4})
    assert r3["status"] == "failed", f"expected failed, got {r3['status']}"
    assert len(r3["gguf_results"]) == 2, f"expected Q5 (kept) + Q6 (garbage, recorded anyway) = 2, got {len(r3['gguf_results'])}"
    assert r3["gguf_results"][0]["question_number"] == 5
    assert r3["gguf_results"][0]["is_garbage"] is False
    assert r3["gguf_results"][0]["is_correct"] is True
    assert r3["gguf_results"][1]["question_number"] == 6
    assert r3["gguf_results"][1]["is_garbage"] is True
    recorded_question_numbers = [e["question_number"] for e in r3["gguf_results"]]
    assert 7 not in recorded_question_numbers, "questions after the garbage one must still never be attempted/recorded"
    assert "Q6" in r3["error"]
    # Q5: 1 batch call (covers all 3) + 5 retries = 6. Q6: 5 more single-question
    # retries (its "attempt 1" was already part of the shared batch call) = 5.
    # Q7 never attempted at all.
    assert calls3 == 1 + 5 + 5, f"expected 11 total calls (Q5's 6 + Q6's remaining 5, Q7 never touched), got {calls3}"
    print("Scenario 3 PASS - Q5 succeeds and is kept, Q6 garbage after 6 attempts recorded anyway, Q7 never attempted, status=failed")

    print("\nALL SCENARIOS PASSED")


if __name__ == "__main__":
    main()
