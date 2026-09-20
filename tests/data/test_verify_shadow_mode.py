"""Shadow mode: run the teacher's self-check, keep every row, report what it would have dropped.

WHY THIS MODE EXISTS
    `enforce` has been measured doing net harm. On run 38832588 the teacher rejected 25 of 25
    `ner_bc5cdr` rows that `verify_ner_row` had just accepted, twice running, and that stopped the
    run; on `calendar_json` it rejected 1,965 of 8,704 (22.6%) including "7pm start plus 60 mins is
    8pm, not 20:00" — 8pm IS 20:00. Both tasks therefore ship with the pass off, and their
    launchers say it should be re-measured once `verifier_notes` tells the teacher the conventions
    it was missing.

    Turning it back on to find out is the expensive way to ask: a false-rejection cascade does not
    merely discard good rows, it can end a multi-day run. Shadow mode is the cheap way.
"""
from __future__ import annotations

import pytest

from data import curriculum


@pytest.fixture(autouse=True)
def _clear_feedback():
    """The rejection-feedback buffer is module state; isolate each test from the last."""
    curriculum._RECENT_REJECTIONS[:] = []
    yield
    curriculum._RECENT_REJECTIONS[:] = []


def _results(rows, verdicts):
    return [(row, valid, reason) for row, (valid, reason) in zip(rows, verdicts)]


ROWS = [
    {"text": "first generated row", "label": "spam"},
    {"text": "second generated row", "label": "ham"},
    {"text": "third generated row", "label": "spam"},
]
VERDICTS = [
    (True, "looks right"),
    (False, "wrong format"),
    (False, "wrong format"),
]


# --------------------------------------------------------------------------
# Mode selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", curriculum.VERIFY_MODE_ENFORCE),
        ("0", curriculum.VERIFY_MODE_OFF),
        ("shadow", curriculum.VERIFY_MODE_SHADOW),
        ("SHADOW", curriculum.VERIFY_MODE_SHADOW),
        ("dry-run", curriculum.VERIFY_MODE_SHADOW),
    ],
)
def test_the_env_var_selects_the_mode_explicitly(monkeypatch, value, expected):
    monkeypatch.setenv("SLM_VERIFY_SYNTH", value)
    assert curriculum._verify_synth_mode() == expected


def test_shadow_counts_as_enabled_because_the_pass_really_runs(monkeypatch):
    """`_verify_synth_enabled` gates whether the teacher is CALLED, and in shadow mode it is —
    that is the entire point. A shadow mode that skipped the call would measure nothing."""
    monkeypatch.setenv("SLM_VERIFY_SYNTH", "shadow")
    assert curriculum._verify_synth_enabled() is True
    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")
    assert curriculum._verify_synth_enabled() is False


def test_an_unrecognised_value_does_not_silently_become_shadow(monkeypatch):
    """Shadow must be asked for by name. A typo turning enforcement into a no-op would leave rows
    unchecked while the log still said the pass ran."""
    monkeypatch.setenv("SLM_VERIFY_SYNTH", "yes")
    assert curriculum._verify_synth_mode() == curriculum.VERIFY_MODE_OFF


# --------------------------------------------------------------------------
# The two guarantees
# --------------------------------------------------------------------------


def test_shadow_mode_keeps_every_row_including_the_rejected_ones():
    """Guarantee one. Two of three rows are rejected and all three survive."""
    logs: list[str] = []
    kept = curriculum._apply_verdicts(
        ROWS, _results(ROWS, VERDICTS), kind="row",
        mode=curriculum.VERIFY_MODE_SHADOW, log=logs.append,
    )
    assert kept == ROWS


def test_enforce_mode_still_drops_what_the_teacher_rejects():
    """The control. If enforce behaved like shadow, the mode would be decoration."""
    kept = curriculum._apply_verdicts(
        ROWS, _results(ROWS, VERDICTS), kind="row",
        mode=curriculum.VERIFY_MODE_ENFORCE, log=None,
    )
    assert kept == [ROWS[0]]


def test_shadow_mode_does_not_feed_rejection_reasons_into_the_next_prompt():
    """GUARANTEE TWO, AND THE SUBTLE ONE.

    `record_rejection_reasons` injects the last round's rejections into the NEXT generation prompt
    as mistakes not to repeat. In shadow mode that would change what the teacher generates — so
    the pass would be altering the run it is supposed to be passively measuring, and the measured
    rejection rate would describe a trajectory that exists only because we measured it.
    """
    curriculum._apply_verdicts(
        ROWS, _results(ROWS, VERDICTS), kind="row",
        mode=curriculum.VERIFY_MODE_SHADOW, log=None,
    )
    assert curriculum._RECENT_REJECTIONS == []
    assert curriculum._rejection_feedback_block() == ""


def test_enforce_mode_does_feed_them_back():
    """The control for guarantee two: the feedback loop is real and shadow is what suppresses it."""
    curriculum._apply_verdicts(
        ROWS, _results(ROWS, VERDICTS), kind="row",
        mode=curriculum.VERIFY_MODE_ENFORCE, log=None,
    )
    assert "wrong format" in curriculum._RECENT_REJECTIONS
    assert "wrong format" in curriculum._rejection_feedback_block()


# --------------------------------------------------------------------------
# The measurement itself
# --------------------------------------------------------------------------


def test_the_log_reports_the_rate_and_says_plainly_that_nothing_was_dropped():
    """A log that read like the enforce path would be worse than no log: the whole risk being
    managed here is somebody concluding rows were discarded when they were not."""
    logs: list[str] = []
    curriculum._apply_verdicts(
        ROWS, _results(ROWS, VERDICTS), kind="row",
        mode=curriculum.VERIFY_MODE_SHADOW, log=logs.append,
    )
    joined = "\n".join(logs)

    assert "[verify:shadow]" in joined
    assert "would have rejected 2/3" in joined
    assert "66.7%" in joined
    assert "ALL 3 KEPT, nothing was dropped" in joined
    # And the per-row lines must not say REJECTED, which is the enforce path's word.
    assert "WOULD REJECT" in joined
    assert "\n        REJECTED" not in joined


def test_the_reasons_are_reported_as_a_frequency_histogram():
    """The actual deliverable. A list of 25 individual rejections does not answer "is this one
    systematic misunderstanding or 25 different ones", and that distinction is what decides
    whether `verifier_notes` can fix it."""
    logs: list[str] = []
    curriculum._apply_verdicts(
        ROWS, _results(ROWS, VERDICTS), kind="row",
        mode=curriculum.VERIFY_MODE_SHADOW, log=logs.append,
    )
    histogram = [line for line in logs if "reasons by frequency" in line]
    assert histogram, "shadow mode logged no reason histogram"
    assert "'wrong format': 2" in histogram[0]


def test_a_clean_batch_reports_zero_without_a_histogram():
    logs: list[str] = []
    clean = [(row, True, "fine") for row in ROWS]
    kept = curriculum._apply_verdicts(
        ROWS, clean, kind="row", mode=curriculum.VERIFY_MODE_SHADOW, log=logs.append,
    )
    joined = "\n".join(logs)
    assert kept == ROWS
    assert "would have rejected 0/3" in joined
    assert "reasons by frequency" not in joined


def test_the_label_pass_and_the_answer_pass_share_one_implementation():
    """Both call `_apply_verdicts`, so they cannot disagree about what shadow mode means — which
    is the drift that would make a measurement taken on one task inapplicable to another."""
    import inspect

    for name in ("verify_generated_labels", "verify_generated_answers"):
        source = inspect.getsource(getattr(curriculum, name))
        assert "_apply_verdicts(" in source, f"{name} does not route through _apply_verdicts"
        assert "record_rejection_reasons" not in source, (
            f"{name} records rejection feedback itself, which would bypass shadow mode's "
            f"suppression of it"
        )


def test_the_generator_is_told_the_same_conventions_the_verifier_judges_it_against():
    """Rows were being held to a closed vocabulary that was never shown to the generator.

    `_task_context_block` reached the in-class synthesis path and the teacher's verification pass,
    but not `_synthesize_new_correct` — the path taken by every `closed_label_space=False` task,
    which is topv2, multiconer, ner_bc5cdr, gec_bea19 and dialogsum.

    The 2026-09-09 topv2 audit measured the cost: the exact verifier rejected 62 of 98 generated
    rows, 33 for labels that do not exist. `SL:TIME` 12 times where TOPv2 says `DATE_TIME`,
    `IN:SET_ALARM` 10 times where it says `CREATE_ALARM`, `IN:SET_TIMER` 8 times where it says
    `CREATE_TIMER` — every one a near-miss synonym, the signature of guessing a list you were never
    given. Demonstrations cannot close that gap: five rows exhibit a handful of 166 names.
    """
    from data.curriculum import _new_example_prompt, _task_context_block
    from tasks import get_task

    context = _task_context_block(get_task("topv2"), [])
    anchor = {"text": "wake me up at six", "answer": "[IN:CREATE_ALARM ...]"}
    prompt = _new_example_prompt(anchor, "parse commands", task_context=context)

    # The real labels the generator kept missing.
    for real in ("CREATE_ALARM", "CREATE_TIMER", "DATE_TIME"):
        assert real in prompt, real
    # And it is the vocabulary being shown, not an accident of the anchor text.
    assert "SET_ALARM" not in prompt

    # ORDER: the context is run-stable, so it must sit in the cacheable prefix, ahead of the
    # per-batch rejection block and the per-row schema request. See the ordering note in
    # `_new_example_prompt`.
    assert prompt.index("CREATE_ALARM") < prompt.index("Generate ONE new")

    # A task with nothing to declare is unaffected.
    assert _new_example_prompt(anchor, "parse commands", task_context="") == \
        _new_example_prompt(anchor, "parse commands")


def test_the_generator_is_told_the_same_conventions_the_verifier_judges_it_against():
    """Rows were being held to a closed vocabulary that was never shown to the generator.

    `_task_context_block` reached the in-class synthesis path and the teacher's verification pass,
    but not `_synthesize_new_correct` — the path taken by every `closed_label_space=False` task,
    which is topv2, multiconer, ner_bc5cdr, gec_bea19 and dialogsum.

    The 2026-09-09 topv2 audit measured the cost: the exact verifier rejected 62 of 98 generated
    rows, 33 for labels that do not exist. `SL:TIME` 12 times where TOPv2 says `DATE_TIME`,
    `IN:SET_ALARM` 10 times where it says `CREATE_ALARM`, `IN:SET_TIMER` 8 times where it says
    `CREATE_TIMER` — every one a near-miss synonym, the signature of guessing a list you were never
    given. Demonstrations cannot close that gap: five rows exhibit a handful of 166 names.
    """
    from data.curriculum import _new_example_prompt, _task_context_block
    from tasks import get_task

    context = _task_context_block(get_task("topv2"), [])
    anchor = {"text": "wake me up at six", "answer": "[IN:CREATE_ALARM ...]"}
    prompt = _new_example_prompt(anchor, "parse commands", task_context=context)

    # The real labels the generator kept missing.
    for real in ("CREATE_ALARM", "CREATE_TIMER", "DATE_TIME"):
        assert real in prompt, real
    # And it is the vocabulary being shown, not an accident of the anchor text.
    assert "SET_ALARM" not in prompt

    # ORDER: the context is run-stable, so it must sit in the cacheable prefix, ahead of the
    # per-batch rejection block and the per-row schema request. See the ordering note in
    # `_new_example_prompt`.
    assert prompt.index("CREATE_ALARM") < prompt.index("Generate ONE new")

    # A task with nothing to declare is unaffected.
    assert _new_example_prompt(anchor, "parse commands", task_context="") == \
        _new_example_prompt(anchor, "parse commands")


@pytest.mark.parametrize(
    "task", ["topv2", "multiconer", "gec_bea19", "goemotions", "dialogsum"]
)
def test_no_loader_ships_a_train_row_that_is_verbatim_in_its_own_eval_split(task):
    """Leakage removed at LOAD, not left to the firewall downstream.

    `curate` has an eval firewall that strips train rows colliding with the eval set, so a run was
    never contaminated by this. Two things still made it worth fixing where it originates. The
    reported curriculum size becomes a lie — the loader claims 3,250 rows and training gets fewer,
    with no record of why. And the guarantee ends up resting on a downstream net rather than on the
    component that knows the splits, which is the kind of arrangement that holds until someone
    reorders the pipeline.

    `scripts/preflight_tasks.py` flagged this on 2026-09-09: topv2 shipped 4 such rows and
    goemotions 1, while multiconer and gec_bea19 already deduped at load. Parametrized across all
    five so a new loader cannot reintroduce it.
    """
    from data.loaders.dataset_integrity import normalized_text_overlap
    from tasks import get_task

    spec = get_task(task)
    train, eval_rows = spec.load(max_train=3250, max_test=1000, log=lambda *a: None)

    overlap = normalized_text_overlap(list(train), list(eval_rows))
    count = len(overlap) if not isinstance(overlap, int) else overlap
    assert not count, (
        f"{task}: {count} train row(s) appear verbatim in the eval split as loaded; dedupe in the "
        f"loader (see `remove_normalized_train_overlap`) rather than relying on curate's firewall"
    )
