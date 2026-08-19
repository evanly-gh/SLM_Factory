"""LlamaPIE when-to-respond loader (arXiv:2505.04066).

The task: at each pause in a conversation, whisper a hint (`interrupt`) or stay silent (`wait`).
This is LlamaPIE's small on-device model — the component that runs continuously and decides *when*
to respond, as opposed to the larger model that decides *what* to say.

The tests below pin the two things that were actually wrong on the first implementation, because
both were silent: decision points are ANY `|MARKER >` token (not only `|SILENCE >`), and the
whisper marker must never leak into the model's input.
"""
import json

import pytest

from data.loaders.proactive_listening import (
    DEFAULT_SOURCES,
    LABEL_INTERRUPT,
    LABEL_WAIT,
    _decision_points,
    _profile_text,
    convert_proactive_rows,
)


# One synthetic dialogue in the authors' exact on-disk format: `|SILENCE >` pauses, ` ^^` marking a
# whisper at the immediately preceding pause.
DIALOGUE = (
    " User: So we went to the museum in |SILENCE > ^^ |SILENCE > the one downtown. "
    "|SILENCE > Speaker 1: Which one was that? |SILENCE > User: It was the |SILENCE > ^^ "
    "|SILENCE > Natural History one. |SILENCE >"
)

# `synthetic0` attaches whispers to EMOTION markers instead. Matching only `|SILENCE >` found zero
# positives in every one of its 7,065 dialogues.
EMOTION_DIALOGUE = (
    " User: They've had me on this for months now! |ANGRY > ^^ |SILENCE > "
    "Speaker 1: I understand. |NEUTRAL > |SILENCE > User: I guess that might help. |NEUTRAL > ^^ "
    "|SILENCE > Speaker 1: Right. |SILENCE >"
)


# --------------------------------------------------------------------------
# Decision points
# --------------------------------------------------------------------------

def test_every_silence_marker_is_a_decision_point():
    points = _decision_points(DIALOGUE)
    assert len(points) == DIALOGUE.count("|SILENCE >")


def test_whisper_marker_labels_the_preceding_pause():
    points = _decision_points(DIALOGUE)
    positives = [i for i, (_, w) in enumerate(points) if w]
    assert len(positives) == DIALOGUE.count(" ^^")
    # The FIRST pause is the one the first `^^` follows.
    assert positives[0] == 0


def test_emotion_markers_are_decision_points_too():
    """Regression: `synthetic0` annotates whispers on |ANGRY >/|NEUTRAL >, and the authors' own
    dataset code masks on the ` >` token rather than on `|SILENCE >`."""
    points = _decision_points(EMOTION_DIALOGUE)
    assert len(points) == EMOTION_DIALOGUE.count(" >"), (
        "every |MARKER > token is a decision point, not just |SILENCE >"
    )
    # Two of them are the emotion markers a silence-only matcher would have missed entirely.
    assert len(points) == 7
    assert sum(1 for _, w in points if w) == 2


def test_whisper_marker_never_leaks_into_the_input():
    """The model must not be shown where the answer is."""
    for dialogue in (DIALOGUE, EMOTION_DIALOGUE):
        for prefix, _ in _decision_points(dialogue):
            assert "^^" not in prefix


def test_prefix_grows_monotonically_and_ends_at_its_marker():
    points = _decision_points(DIALOGUE)
    lengths = [len(p) for p, _ in points]
    assert lengths == sorted(lengths)
    for prefix, _ in points:
        assert prefix.rstrip().endswith(">")


def test_dialogue_with_no_markers_yields_nothing():
    assert _decision_points("User: hello there. Speaker 1: hi.") == []


# --------------------------------------------------------------------------
# Row conversion
# --------------------------------------------------------------------------

def _dialogue_record(raw=DIALOGUE, memory=""):
    return {"raw": raw, "memory": memory, "source": "soda", "sample_id": "000001"}


def test_conversion_emits_only_the_two_task_labels():
    rows = convert_proactive_rows([_dialogue_record()], points_per_dialogue=4)
    assert rows
    assert {r["label"] for r in rows} <= {LABEL_INTERRUPT, LABEL_WAIT}
    assert all(r["text"].strip() for r in rows)


def test_balanced_sampling_guarantees_a_positive_when_one_exists():
    """The natural positive rate is ~12%; a small unbalanced draw can easily contain none, and a
    binary task whose minority class is absent cannot be scored at all."""
    rows = convert_proactive_rows([_dialogue_record()], points_per_dialogue=3)
    assert any(r["label"] == LABEL_INTERRUPT for r in rows)


def test_dialogue_with_no_whisper_yields_only_wait():
    no_whisper = DIALOGUE.replace(" ^^", "")
    rows = convert_proactive_rows([_dialogue_record(no_whisper)], points_per_dialogue=3)
    assert rows and all(r["label"] == LABEL_WAIT for r in rows)


def test_instruction_and_transcript_are_both_present():
    rows = convert_proactive_rows([_dialogue_record()], points_per_dialogue=2)
    text = rows[0]["text"]
    assert "in-ear assistant" in text
    assert "Conversation so far:" in text


def test_memory_is_included_when_supplied():
    """A reminder-type whisper is unguessable without the memory — the detail to be recalled is
    only in the user profile."""
    memory = json.dumps({
        "profile": "Krew is a high school senior teaching her brother Donna to read.",
        "events": {"event0": "Krew was accepted to her top-choice university."},
    })
    rows = convert_proactive_rows([_dialogue_record(memory=memory)], points_per_dialogue=2)
    assert "What you know about your user:" in rows[0]["text"]
    assert "Donna" in rows[0]["text"]
    assert "top-choice university" in rows[0]["text"]


def test_memory_absent_omits_the_profile_block():
    rows = convert_proactive_rows([_dialogue_record(memory="")], points_per_dialogue=2)
    assert "What you know about your user:" not in rows[0]["text"]


def test_context_is_bounded():
    long_dialogue = (" User: " + "word " * 4000 + "|SILENCE > ^^ |SILENCE >")
    rows = convert_proactive_rows([_dialogue_record(long_dialogue)], points_per_dialogue=2)
    # Instruction + a truncated transcript, not the whole 20k-character dialogue.
    assert all(len(r["text"]) < 4000 for r in rows)
    assert any("…" in r["text"] for r in rows)


def test_conversion_is_deterministic_for_a_fixed_seed():
    a = convert_proactive_rows([_dialogue_record()], points_per_dialogue=3, seed=7)
    b = convert_proactive_rows([_dialogue_record()], points_per_dialogue=3, seed=7)
    assert a == b


def test_empty_and_malformed_dialogues_are_skipped_not_fatal():
    rows = convert_proactive_rows(
        [{"raw": ""}, {"raw": "   "}, {}, _dialogue_record()], points_per_dialogue=2
    )
    assert rows, "the one valid dialogue must still produce rows"


def test_profile_text_handles_non_json_memory():
    assert _profile_text("just a plain sentence") == "just a plain sentence"
    assert _profile_text("") == ""


def test_synthetic0_is_excluded_by_default():
    """It is half the training bundle but carries emotion markers the held-out split has none of —
    including it would put a surface feature in training that is absent at eval."""
    assert "synthetic0" not in DEFAULT_SOURCES
    assert set(DEFAULT_SOURCES) == {"synthetic", "perl", "soda"}


# --------------------------------------------------------------------------
# The vendored bundle (skipped if absent, e.g. a fresh clone)
# --------------------------------------------------------------------------

def _bundle_available() -> bool:
    import os
    from data.loaders.proactive_listening import LOCAL_BUNDLE
    return os.path.isfile(os.path.join(LOCAL_BUNDLE, "test.jsonl"))


@pytest.mark.skipif(not _bundle_available(), reason="vendored LlamaPIE bundle not present")
def test_real_bundle_train_and_eval_have_matching_label_distributions():
    """A train/eval prior mismatch is its own bug: the first implementation drew 4.2% positives for
    training against a 33.4% eval rate, because the bundle is grouped by sub-corpus and truncation
    took only the first one."""
    from data.loaders.proactive_listening import load_proactive_listening

    train, test = load_proactive_listening(max_train=600, max_test=300, log=lambda _m: None)
    assert train and test

    def rate(rows):
        return sum(1 for r in rows if r["label"] == LABEL_INTERRUPT) / len(rows)

    assert abs(rate(train) - rate(test)) < 0.05
    assert 0.15 < rate(test) < 0.60, "both classes must be well represented"


@pytest.mark.skipif(not _bundle_available(), reason="vendored LlamaPIE bundle not present")
def test_real_bundle_has_no_split_overlap_and_no_marker_leak():
    from data.loaders.proactive_listening import load_proactive_listening

    train, test = load_proactive_listening(max_train=600, max_test=300, log=lambda _m: None)
    norm = lambda s: " ".join(s.lower().split())
    assert not ({norm(r["text"]) for r in train} & {norm(r["text"]) for r in test})
    assert not [r for r in train + test if "^^" in r["text"]]


@pytest.mark.skipif(not _bundle_available(), reason="vendored LlamaPIE bundle not present")
def test_real_bundle_scores_one_on_gold_and_zero_on_a_degenerate_answer():
    """The pre-flight that `calendar_json` needed and did not get: gold-vs-gold proves the harness
    is wired up, and an all-majority prediction scoring 0.0 proves the metric refuses to reward
    collapse."""
    from data.eval_set import build_eval_set
    from data.loaders.proactive_listening import load_proactive_listening
    from eval.scorers import classification

    _train, test = load_proactive_listening(max_train=50, max_test=300, log=lambda _m: None)
    eval_set = build_eval_set(test, task="proactive_listening", target=300)
    gold = [e["label"] for e in eval_set.all]
    # The task NAMES minority-class F1 on its spec. It used to be selected implicitly by counting
    # classes, and the metric string said `macro_f1` either way.
    score = eval_set.spec.score
    assert score is classification.score_minority_f1
    assert score(eval_set, gold)["f1"] == pytest.approx(1.0)
    assert score(eval_set, [LABEL_WAIT] * len(eval_set.all))["f1"] == 0.0
