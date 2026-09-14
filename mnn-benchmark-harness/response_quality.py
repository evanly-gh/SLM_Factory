#!/usr/bin/env python3
"""
response_quality.py — Post-hoc quality checks for MNN benchmark responses.

is_garbage() flags responses that are structurally broken (not wrong, just
not real output) based on confirmed real failure patterns from this
project's own benchmark runs:
  - REPETITION_DETECTED firing on the Android side (a model looping on the
    same token/phrase, e.g. "resse\n\n\n\n\n..." repeated until max_tokens)
  - Empty/whitespace-only responses
  - decode_len <= 5: near-instant termination producing 1-2 tokens, seen
    repeatedly with certain quant/model combinations
  - A Python-side text repetition check, independent of the Android-side
    flag above: a confirmed real response ("resse​​...\n\n\n...")
    looped without ever setting repetition_detected, likely because
    invisible zero-width unicode characters broke the on-device exact-match
    check. Zero-width characters are stripped before checking here, then
    any short substring repeated 4+ times is treated as garbage regardless
    of what the on-device flag reported.

score_accuracy() checks whether a (non-garbage) response's FINAL, concluding
answer - as extracted by extract_final_answer() - matches the expected
answer, via fuzzy substring matching on normalized text. It deliberately
does NOT match against the raw response: a smoke test confirmed a real
false positive where the model mentioned the correct number mid-reasoning
while explicitly rejecting it (e.g. "...Total is 45 kg? No." before landing
on a different final answer), which a raw-substring check scored as
correct. Scoring against extract_final_answer()'s output instead - which is
built specifically to find the model's actual concluding answer, not just
any mention of a number - avoids that class of false positive.
"""

import re
import string


# ---------------------------------------------------------------------------
# Garbage detection
# ---------------------------------------------------------------------------

# Single, easy-to-flip switch: when False, is_garbage() short-circuits to
# False immediately, before any rule runs - no exceptions, every rule
# (including the length-based truncation check) stays fully intact in the
# code, just skipped, so flipping this back to True re-enables all of them
# exactly as they were, with no logic to reconstruct.
GARBAGE_CHECK_ENABLED = False

# metrics dict field names observed/plausible for signaling that MNN Chat's
# own repetition guard fired on-device - checked case-insensitively so this
# doesn't silently miss a differently-cased key from a future logcat tag.
REPETITION_FIELD_NAMES = ["repetition_detected", "REPETITION_DETECTED"]

DECODE_LEN_GARBAGE_THRESHOLD = 5

# Confirmed real truncation bug: responses that stop mid-thought well short
# of a real answer (e.g. "We are told:", "Let's break down..."). Gated by
# GARBAGE_CHECK_ENABLED like every other rule here - a genuine, complete
# answer under 150 characters is rare for this question set, and a truncated
# one is exactly what this is meant to catch.
TRUNCATION_LENGTH_THRESHOLD = 150

# Zero-width/invisible characters that can silently break exact-substring
# matching (confirmed real case: broke the Android-side repetition guard).
# U+200B zero-width space, U+200C ZWNJ, U+200D ZWJ, U+2060 word joiner,
# U+FEFF zero-width no-break space / BOM.
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")

# A short window repeated this many times or more, anywhere in the
# (zero-width-stripped) response, is treated as a model stuck looping -
# short enough to reliably catch a repeated phrase/token, long enough that
# ordinary prose essentially never repeats an identical window this often.
TEXT_REPETITION_WINDOW_CHARS = 12
TEXT_REPETITION_MIN_OCCURRENCES = 4


def _strip_zero_width(text: str) -> str:
    return _ZERO_WIDTH_RE.sub("", text or "")


def _has_repeated_substring(text: str, window: int = TEXT_REPETITION_WINDOW_CHARS,
                             min_occurrences: int = TEXT_REPETITION_MIN_OCCURRENCES) -> bool:
    """True if any window-length substring appears at least min_occurrences
    times in text - a cheap, Python-side stand-in for "the model is stuck
    looping", independent of any on-device flag."""
    if len(text) < window:
        return False
    counts = {}
    for i in range(len(text) - window + 1):
        chunk = text[i:i + window]
        count = counts.get(chunk, 0) + 1
        if count >= min_occurrences:
            return True
        counts[chunk] = count
    return False


def is_garbage(response: str, metrics: dict) -> bool:
    """Return True if `response` is structurally broken output, not just a
    wrong answer. `metrics` is a per-question metrics dict; its exact shape
    is engine-dependent (e.g. run_mnn_autobench.py's build_metrics() vs.
    run_autobench.py's SmolChat/GGUF build_metrics()), so every rule here
    must degrade gracefully when a given engine simply doesn't report a
    field, rather than treating "field not reported by this engine" the
    same as "field reported as a bad value".
    """
    if not GARBAGE_CHECK_ENABLED:
        return False

    # No longer exempt from the flag above - see TRUNCATION_LENGTH_THRESHOLD's
    # own comment. Gated the same as every rule below it.
    if len((response or "").strip()) < TRUNCATION_LENGTH_THRESHOLD:
        return True

    metrics = metrics or {}

    # Rule 1: on-device repetition guard fired.
    for field in REPETITION_FIELD_NAMES:
        if metrics.get(field) is True:
            return True

    # Rule 2: empty/whitespace-only response.
    if response is None or not response.strip():
        return True

    # Rule 3: decode_len <= 5 - near-instant termination, no real answer.
    # Only applies when the engine actually reports decode_len at all (MNN
    # does; SmolChat/GGUF's build_metrics() has no such field) - a key that's
    # simply absent from this engine's metrics shape is "unknown", not "0",
    # and must not silently make every response from that engine garbage.
    # A key that IS present but explicitly None (a failed extraction on an
    # engine that does report it) still defaults to 0 -> garbage, as before.
    if "decode_len" in metrics:
        decode_len = metrics.get("decode_len")
        if decode_len is None:
            decode_len = 0
        if decode_len <= DECODE_LEN_GARBAGE_THRESHOLD:
            return True

    # Rule 4: Python-side text repetition check, independent of Rule 1's
    # on-device flag - confirmed gap: a real response looped without ever
    # setting repetition_detected, likely because invisible zero-width
    # characters broke the on-device exact-match check. Stripping them here
    # first closes that blind spot.
    if _has_repeated_substring(_strip_zero_width(response)):
        return True

    return False


# ---------------------------------------------------------------------------
# Final-answer extraction (used by score_accuracy() below for correctness
# scoring, AND kept available standalone for display purposes)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_LATEX_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_MD_EMPHASIS_RE = re.compile(r"[*_]{1,3}")
_MD_BACKTICK_RE = re.compile(r"`+")
_DOLLAR_RE = re.compile(r"\$")
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+$")
# Requires at least one digit after a decimal point to count as a decimal,
# so a sentence-ending period right after a whole number (e.g. "87.") isn't
# swallowed into the match.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _strip_markdown(text: str) -> str:
    if not text:
        return text
    text = _LATEX_FRAC_RE.sub(r"\1/\2", text)  # \frac{a}{b} -> a/b
    text = _LATEX_CMD_RE.sub("", text)  # \boxed, \text, etc. -> removed (braces/content left as-is)
    text = _DOLLAR_RE.sub("", text)  # $...$ math delimiters
    text = _MD_EMPHASIS_RE.sub("", text)  # **bold**, __bold__, *italic*
    text = _MD_BACKTICK_RE.sub("", text)  # `code`
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_final_answer(response: str) -> str:
    """Best-effort extraction of just the final, concluding answer from a
    longer response - both for readability in results output AND as the
    basis for score_accuracy()'s correctness check below, since a model's
    concluding answer is what actually matters, not every number it happens
    to mention while reasoning towards (or explicitly rejecting) a value.

    Priority order:
      1. The last \\boxed{...} (a common, explicit "final answer" marker in
         math-reasoning model output - eval_questions_phase3.json is exactly
         this kind of math word problem set).
      2. The last standalone number anywhere in the (markdown-stripped)
         response - searched across the WHOLE response, not just the final
         sentence, so a trailing non-numeric remark ("Hope that helps!")
         doesn't hide the real answer stated just before it.
      3. The last non-empty sentence, for non-numeric answers.
    """
    if not response or not response.strip():
        return ""

    boxed_matches = _BOXED_RE.findall(response)
    if boxed_matches:
        return _strip_markdown(boxed_matches[-1]).strip()

    cleaned = _strip_markdown(response)
    if not cleaned:
        return ""

    numbers = _NUMBER_RE.findall(cleaned)
    if numbers:
        return numbers[-1].replace(",", "")

    chunks = [c.strip() for c in re.split(r"[.\n!?]+", cleaned) if c.strip()]
    if not chunks:
        return cleaned
    return _TRAILING_PUNCT_RE.sub("", chunks[-1]).strip()


# ---------------------------------------------------------------------------
# Accuracy scoring
# ---------------------------------------------------------------------------

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.translate(_PUNCTUATION_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_accuracy(response: str, reference_answer: str) -> bool:
    """Fuzzy match: True if the normalized reference_answer appears anywhere
    within the normalized FINAL ANSWER extracted from response (via
    extract_final_answer()), NOT the raw response. Matching against the raw
    response caused a confirmed false positive: a model that mentions the
    correct number mid-reasoning while explicitly rejecting it (e.g.
    "...Total is 45 kg? No." before landing on a different final answer)
    would score as correct even though its actual answer was wrong.
    Extracting the model's concluding answer first avoids that. An empty
    normalized reference never counts as a match - it would trivially match
    everything.
    """
    normalized_reference = _normalize(reference_answer)
    if not normalized_reference:
        return False
    normalized_final_answer = _normalize(extract_final_answer(response))
    return normalized_reference in normalized_final_answer


# ---------------------------------------------------------------------------
# Inline tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # These tests validate is_garbage()'s rule logic itself, which stays
    # fully intact regardless of GARBAGE_CHECK_ENABLED's default - so the
    # flag is switched on for the duration of the tests below, otherwise
    # every rule-logic assertion here would trivially fail on the disabled
    # short-circuit rather than actually exercising the rules.
    GARBAGE_CHECK_ENABLED = True

    # --- is_garbage: repetition loop (confirmed real failure pattern) ---
    repetition_response = "resse\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
    repetition_metrics = {"decode_len": 80, "repetition_detected": True}
    assert is_garbage(repetition_response, repetition_metrics) is True, "repetition_detected should be garbage regardless of decode_len"

    # Same shape but the field only fires under its Android-tag-style name.
    assert is_garbage(repetition_response, {"decode_len": 80, "REPETITION_DETECTED": True}) is True

    # --- is_garbage: decode_len <= 5 near-empty response ---
    near_empty_response = "The"
    near_empty_metrics = {"decode_len": 2, "repetition_detected": False}
    assert is_garbage(near_empty_response, near_empty_metrics) is True, "decode_len=2 should be garbage even with non-empty text"

    assert is_garbage("OK.", {"decode_len": 5}) is True, "decode_len exactly at threshold should still be garbage"
    # Padded to 150+ chars so this isolates decode_len's own threshold,
    # independent of the always-on truncation-length check below.
    long_enough_response = "A real six token answer here, " + "padding to clear the always-on truncation length check. " * 3
    assert len(long_enough_response) >= TRUNCATION_LENGTH_THRESHOLD
    assert is_garbage(long_enough_response, {"decode_len": 6}) is False, "decode_len just above threshold should not be garbage on its own"

    # --- is_garbage: empty/whitespace-only response ---
    assert is_garbage("", {"decode_len": 30}) is True
    assert is_garbage("   \n\t  ", {"decode_len": 30}) is True
    assert is_garbage(None, {"decode_len": 30}) is True

    # --- is_garbage: missing/partial metrics shouldn't crash ---
    # A "decode_len" key that's entirely ABSENT (e.g. GGUF/SmolChat's own
    # build_metrics(), which has no such field at all) means "unknown" for
    # rule 3, not "0" - a normal-looking response must not be auto-flagged
    # garbage just because this engine doesn't report token counts.
    # Padded to 150+ chars throughout this section so these isolate the
    # rule each is meant to test, independent of the always-on truncation
    # check above (a short response would now be garbage on length alone,
    # regardless of what these specific rules do).
    normal_response = "A perfectly normal, complete answer to the question, explained in enough detail to comfortably clear the always-on truncation length check on its own."
    assert len(normal_response) >= TRUNCATION_LENGTH_THRESHOLD
    assert is_garbage(normal_response, {}) is False, "absent decode_len is 'unknown' (rule 3 doesn't apply), not '0'"
    assert is_garbage(normal_response, None) is False, "None metrics dict should be handled gracefully (same as {})"

    # A key that IS present but explicitly None (a failed extraction on an
    # engine that DOES normally report it) still defaults to 0 -> garbage.
    assert is_garbage(normal_response, {"decode_len": None}) is True, "present-but-None decode_len still defaults to 0 -> garbage"

    # GGUF/SmolChat-shaped metrics (no decode_len key at all, real fields
    # instead) - rule 3 must not fire; only rules 1/2 apply on this engine.
    gguf_shaped_metrics = {"cold_load_ms": 900, "ttft_ms": 150, "tps": 12.3, "memory_kb": 500000, "power_ma": 300.0, "thermal": "NONE"}
    gguf_shaped_response = (
        "The capital of France is Paris, a city on the Seine river, and has been the "
        "country's capital for many centuries, known worldwide for its museums and architecture."
    )
    assert len(gguf_shaped_response) >= TRUNCATION_LENGTH_THRESHOLD
    assert is_garbage(gguf_shaped_response, gguf_shaped_metrics) is False, \
        "GGUF metrics shape has no decode_len - must not be auto-garbage"

    # --- is_garbage: a real, healthy response should NOT be garbage ---
    healthy_response = "The capital of France is Paris, a city on the Seine river known for the Eiffel Tower, the Louvre, the Notre-Dame Cathedral, and centuries of history as the nation's capital."
    healthy_metrics = {"decode_len": 14, "repetition_detected": False}
    assert len(healthy_response) >= TRUNCATION_LENGTH_THRESHOLD
    assert is_garbage(healthy_response, healthy_metrics) is False

    # --- is_garbage: truncation-length check (CONFIRMED REAL BUG) ---
    # Actual observed truncated responses, e.g. "We are told:" and "Let's
    # break down..." - the model stops mid-thought well short of a real
    # answer. Gated by GARBAGE_CHECK_ENABLED like every other rule here.
    assert is_garbage("We are told:", {}) is True, "confirmed real truncation case"
    assert is_garbage("Let's break down...", {}) is True, "confirmed real truncation case"
    # Non-repetitive filler (not "x" * N) so this isolates the length
    # boundary itself, independent of the separate repetition rule (also
    # enabled here) which real repeated characters would otherwise trip.
    _boundary_filler = ("This is a non-repetitive filler sentence used only to test the exact-length "
                         "boundary of the truncation check without tripping the repetition rule. " * 2)
    assert is_garbage(_boundary_filler[:TRUNCATION_LENGTH_THRESHOLD - 1], {}) is True, "one character under the threshold is still garbage"
    assert is_garbage(_boundary_filler[:TRUNCATION_LENGTH_THRESHOLD], {}) is False, "exactly at the threshold is not garbage on length grounds alone"

    GARBAGE_CHECK_ENABLED = False
    assert is_garbage("We are told:", {}) is False, "truncation check no longer fires while GARBAGE_CHECK_ENABLED is False - no rule is exempt from the flag anymore"
    assert is_garbage(healthy_response, healthy_metrics) is False, "a long-enough response is still correctly not garbage while disabled"
    GARBAGE_CHECK_ENABLED = True

    # --- is_garbage: CONFIRMED REAL GAP, now fixed ---
    # Actual observed response: the model looped on "resse" with zero-width
    # spaces (U+200B) interspersed between newline runs, e.g.
    # "resse​​\n\n\n...\n\n\n...". repetition_detected did NOT
    # fire on the Android side (likely because the zero-width characters
    # broke its exact-match check), and neither the empty-response rule nor
    # decode_len<=5 applied - decode_len was reported as a large number
    # since the model kept generating (garbage) tokens the whole time. This
    # is exactly the case Rule 4 exists for: a Python-side check on the
    # response text itself, independent of the on-device flag.
    zero_width_repetition_response = (
        "resse​​\n\n\n"
        "resse﻿\n\n\n"
        "resse‌\n\n\n"
        "resse‍​\n\n\n"
        "resse⁠\n\n\n"
        "resse​﻿\n\n\n"
    )
    zero_width_repetition_metrics = {"decode_len": 80, "repetition_detected": False}
    assert is_garbage(zero_width_repetition_response, zero_width_repetition_metrics) is True, \
        "Rule 4 must catch this even though repetition_detected didn't fire and decode_len > 5"

    # Sanity: stripping zero-width characters must not itself cause false
    # positives on ordinary, non-repetitive text that merely happens to
    # contain a few invisible characters (e.g. copy-pasted from elsewhere).
    # Padded to 150+ chars, same reasoning as the section above.
    zero_width_but_normal_response = (
        "The capital of​ France is​ Paris, a​ lovely city on the Seine known for its "
        "museums,​ architecture, gardens, and centuries of history as the nation's capital."
    )
    assert len(zero_width_but_normal_response) >= TRUNCATION_LENGTH_THRESHOLD
    assert is_garbage(zero_width_but_normal_response, {"decode_len": 14, "repetition_detected": False}) is False

    print("is_garbage: all test cases passed")

    # --- extract_final_answer: realistic multi-step math response (\boxed{} priority) ---
    boxed_response = (
        "Let's work through this step by step.\n"
        "Dora is 15 years old. Her father's age is eight more than twice Dora's age: "
        "2 * 15 + 8 = 38.\n"
        "Her mother is four years younger than her father: 38 - 4 = 34.\n"
        "Combined total: 15 + 38 + 34 = 87.\n"
        "The final answer is $\\boxed{87}$."
    )
    assert extract_final_answer(boxed_response) == "87"

    # --- extract_final_answer: no \boxed{} - last standalone number in a multi-step response ---
    stepwise_response = (
        "Mary has 30 sheep. Half of them (15) give 1 kg each = 15 kg. "
        "The other half (15) give 2 kg each = 30 kg. "
        "Total milk collected per day is 15 + 30 = 45 kg."
    )
    assert extract_final_answer(stepwise_response) == "45"

    # --- extract_final_answer: trailing non-numeric commentary after the real answer ---
    trailing_commentary_response = (
        "Katina starts with $3000. Over 2 years (24 months) she removes $100/month, "
        "a total of $2400. Remaining balance: 3000 - 2400 = 600. Hope that helps!"
    )
    assert extract_final_answer(trailing_commentary_response) == "600"

    # --- extract_final_answer: markdown-bolded final answer, no boxed/latex ---
    markdown_response = "After simplifying, we get **87** as the combined age."
    assert extract_final_answer(markdown_response) == "87"

    # --- extract_final_answer: non-numeric answer falls back to last sentence ---
    non_numeric_response = "Let's think about this. France is a country in Europe. Its capital is Paris."
    assert extract_final_answer(non_numeric_response) == "Its capital is Paris"

    # --- extract_final_answer: garbage/empty responses degrade gracefully, no crash ---
    assert extract_final_answer("") == ""
    assert extract_final_answer(None) == ""
    assert extract_final_answer(repetition_response) == "resse"  # no digits/sentence delimiters - falls back to the sole leftover word, not a crash

    print("extract_final_answer: all test cases passed")

    # --- score_accuracy: reference appearing inside a longer response ---
    assert score_accuracy("The total is 87 years old by our count.", "87") is True

    # --- score_accuracy: real correct Q&A example ---
    assert score_accuracy(
        "The capital of France is Paris, located on the Seine river.",
        "Paris",
    ) is True

    # --- score_accuracy: real incorrect Q&A example ---
    assert score_accuracy(
        "The capital of France is Lyon.",
        "Paris",
    ) is False

    # --- score_accuracy: case/punctuation/whitespace normalization ---
    # (single-sentence responses, so extract_final_answer()'s "last sentence"
    # step doesn't fragment the answer away from the rest of the text)
    assert score_accuracy("I'm confident the   answer is PARIS!", "paris") is True
    assert score_accuracy("The chemical symbol is Au.", "au") is True
    assert score_accuracy("The chemical symbol is gold (Au).", "Au.") is True

    # --- score_accuracy: garbage/near-empty responses should just fail to match, not crash ---
    assert score_accuracy(repetition_response, "Paris") is False
    assert score_accuracy("", "Paris") is False
    assert score_accuracy(None, "Paris") is False

    # --- score_accuracy: empty reference never trivially matches ---
    assert score_accuracy("Any response at all", "") is False
    assert score_accuracy("Any response at all", None) is False

    # --- score_accuracy: CONFIRMED REAL FALSE-POSITIVE, now fixed ---
    # The model considers 45 mid-reasoning, explicitly REJECTS it ("No."),
    # then lands on a different final answer (38). The old raw-substring
    # check scored this as correct against reference "45" purely because
    # "45" appears somewhere in the text - even though the model's actual,
    # concluding answer was 38, not 45. Scoring against
    # extract_final_answer()'s output (which finds the LAST number in the
    # response, i.e. the actual final answer) fixes this.
    rejected_midway_response = (
        "Mary has 30 sheep. Half give 1 kg, half give 2 kg. "
        "Total is 45 kg? No, wait - I miscounted, only 20 of the sheep are "
        "actually producing milk today. Let me redo this: 20 sheep at "
        "1.9 kg average is 38 kg."
    )
    assert extract_final_answer(rejected_midway_response) == "38", "extract_final_answer should land on the actual concluding number, not the rejected mid-reasoning one"
    assert score_accuracy(rejected_midway_response, "45") is False, "FALSE POSITIVE FIX: '45' was mentioned and rejected mid-reasoning, not the model's final answer"
    assert score_accuracy(rejected_midway_response, "38") is True, "38 is the model's actual concluding answer and should score correct"

    print("score_accuracy: all test cases passed")
