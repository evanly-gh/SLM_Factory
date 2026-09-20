"""The single teacher measurement: one 5-shot pass over the full eval set, two consumers.

WHY THIS FILE EXISTS
    `surgical_synthesis` used to be offered on every task at every score, and the only evidence
    anyone had about the teacher was its ZERO-SHOT number from `measure_endpoint_baseline` — which
    on a format-bound task mostly reports whether it guessed the output contract (BC5CDR: 0.1131
    zero-shot, 0.7190 five-shot, B276) and was never consulted before spending the budget anyway.
    `agent/teacher_fitness.py` measures the teacher the way synthesis actually prompts it and
    refuses synthetic data for the whole run when it cannot clear `MIN_ACCURACY`.

    That verdict now ALSO sets the run's accuracy goal (see `tests/test_qwen_baseline_goal.py`),
    which raises the stakes on three things this file pins: the measurement covers the whole eval
    set rather than a 200-row subsample, it reports format validity beside the score, and a
    wholesale generation failure yields "unmeasured" rather than a plausible-looking 0.0000.

    Three further properties are load-bearing and each has been a bug somewhere in this pipeline:

      the direction of the default   An unmeasured teacher is REFUSED, not allowed. `state.get(...)`
                                     defaulting to True would mean a run that skipped the gate
                                     silently regained synthesis, which is what the gate prevents.
      never fatal                    The measurement is a network call per row. It answers a
                                     go/no-go question, so a failure has to produce a verdict —
                                     a conservative one — rather than end the run.
      contamination                  Demonstrations come from TRAIN. Drawing them from the eval set
                                     would put held-out rows into a prompt that scores them, and
                                     the resulting number would authorise the teacher on the
                                     strength of having already been shown the answers.

    Nothing here touches the network: `generate_fn` is always supplied, and the one test that
    exercises the "no endpoint" path stubs the client factory.
"""
from __future__ import annotations

import dataclasses

import pytest

from agent.teacher_fitness import (
    FITNESS_SHOTS,
    MIN_ACCURACY,
    measure_teacher_fitness,
    synthesis_allowed,
)
from data.eval_set import EvalSet
from tasks import get_task

SPEC = get_task("routerbench")

# Deliberately disjoint text markers. Every assertion below about which rows reached a prompt is a
# substring test, so a train row and an eval row must never be confusable.
EVAL_ROWS = (
    [{"text": f"eval utterance {i:03d}", "label": "route"} for i in range(14)]
    + [{"text": f"eval utterance {i:03d}", "label": "local"} for i in range(14, 20)]
)
TRAIN_ROWS = (
    [{"text": f"train utterance {i:03d}", "label": "route"} for i in range(8)]
    + [{"text": f"train utterance {i:03d}", "label": "local"} for i in range(8, 12)]
)


def _eval_set(rows=None) -> EvalSet:
    return EvalSet(all=[dict(row) for row in (rows if rows is not None else EVAL_ROWS)],
                   task=SPEC.name)


def _perfect_teacher(prompts: list[str] | None = None):
    """A teacher that answers every eval row with its gold label.

    The row is identified by which eval marker the prompt carries — the demonstrations prepended to
    it carry train markers only, which is itself part of what the contamination test checks.
    """
    gold = {row["text"]: row["label"] for row in EVAL_ROWS}

    def generate(prompt, *_args, **_kwargs):
        if prompts is not None:
            prompts.append(prompt)
        for text, label in gold.items():
            if text in prompt:
                return label
        return ""

    return generate


def _majority_teacher(prompt, *_args, **_kwargs):
    """Always answers the majority class. Scored by minority F1, that is 0.0 — the honest reading
    of a model that has not done the task, and well under the gate."""
    return "route"


# --------------------------------------------------------------------------
# The verdict, and what it authorises
# --------------------------------------------------------------------------


def test_a_teacher_above_the_gate_is_allowed_to_generate():
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_perfect_teacher(), log=lambda *_: None,
    )
    assert verdict["status"] == "measured"
    assert verdict["score"] >= MIN_ACCURACY
    assert verdict["synthesis_allowed"] is True
    assert verdict["n"] == len(EVAL_ROWS)
    assert verdict["shots"] == FITNESS_SHOTS
    assert verdict["threshold"] == MIN_ACCURACY
    assert verdict["metric"] == SPEC.metric_name


def test_a_teacher_below_the_gate_is_refused():
    """A teacher that gets the task right less often than the student is expected to become is a
    source of labelled noise: training on its output caps the student at its error rate."""
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_majority_teacher, log=lambda *_: None,
    )
    assert verdict["status"] == "measured"
    assert verdict["score"] < MIN_ACCURACY
    assert verdict["synthesis_allowed"] is False


def test_the_refusal_says_what_it_costs_the_run():
    """The verdict and its consequence are logged together, so neither can be read alone — a score
    with no stated consequence reads as a diagnostic nobody acted on."""
    logs: list[str] = []
    measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_majority_teacher, log=logs.append,
    )
    joined = " ".join(logs)
    assert "REFUSED" in joined
    assert "surgical_synthesis is off the menu" in joined
    # The other two interventions are explicitly unaffected, or a reader concludes the run is over.
    assert "mine_new_real" in joined


def test_an_unreachable_endpoint_refuses_rather_than_raising(monkeypatch):
    """The measurement is a network call per row and it answers a go/no-go question, so it must
    always return a verdict. "We could not check" is recorded as UNMEASURED and refused — the
    conservative direction, and the honest one: the alternative silently authorises a teacher
    nobody measured.
    """
    import data.synth_client as synth_client

    monkeypatch.setattr(synth_client, "get_generate_fn", lambda **_kwargs: None)
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=None, log=lambda *_: None,
    )
    assert verdict["status"] == "unmeasured"
    assert verdict["synthesis_allowed"] is False
    assert verdict["score"] is None
    assert "unreachable" in verdict["reason"]


def test_a_scorer_that_blows_up_refuses_rather_than_raising():
    """Same contract one layer down: an unmeasurable teacher is refused, not fatal."""
    def _explode(*_args, **_kwargs):
        raise RuntimeError("scorer disagreed with the prediction shape")

    broken = dataclasses.replace(SPEC, score=_explode)
    verdict = measure_teacher_fitness(
        broken, _eval_set(), TRAIN_ROWS, generate_fn=_perfect_teacher(), log=lambda *_: None,
    )
    assert verdict["status"] == "unmeasured"
    assert verdict["synthesis_allowed"] is False
    assert "RuntimeError" in verdict["reason"]


def test_an_empty_eval_set_is_refused_rather_than_scored():
    verdict = measure_teacher_fitness(
        SPEC, _eval_set([]), TRAIN_ROWS, generate_fn=_perfect_teacher(), log=lambda *_: None,
    )
    assert verdict["status"] == "unmeasured"
    assert verdict["synthesis_allowed"] is False


def test_the_whole_eval_set_is_scored_by_default():
    """The default is the FULL set, because the verdict also sets the run's accuracy goal.

    It used to be a 200-row subsample, which was a defensible cost compromise for a go/no-go
    decision and is not one for a convergence target: the student is scored on all of E, so a goal
    calibrated on a fifth of it carries an error bar wider than the score differences the run is
    trying to detect.
    """
    prompts: list[str] = []
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_perfect_teacher(prompts),
        log=lambda *_: None,
    )
    assert verdict["n"] == len(EVAL_ROWS)
    # ONE pass, not two: the second zero-shot pass was removed once nothing consumed it.
    assert len(prompts) == len(EVAL_ROWS)
    assert all("### Solved example" in p for p in prompts)


def test_n_rows_can_still_subsample_for_a_cheap_bring_up_run():
    """The cap survives as an explicit opt-in, so a new task can be brought up without paying for
    a full-eval measurement against a teacher that may not work on it yet."""
    prompts: list[str] = []
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_perfect_teacher(prompts),
        n_rows=6, log=lambda *_: None,
    )
    assert verdict["n"] == 6
    assert len(prompts) == 6


# --------------------------------------------------------------------------
# Contamination: demonstrations come from TRAIN
# --------------------------------------------------------------------------


def test_demonstrations_are_drawn_from_train_and_never_from_the_eval_set():
    """The whole measurement is worthless if it leaks. A demonstration taken from the eval set puts
    a held-out row and its gold answer into the prompt that later scores that same row, so the
    teacher would clear the gate on the strength of having been shown the answers.
    """
    prompts: list[str] = []
    measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_perfect_teacher(prompts),
        log=lambda *_: None,
    )

    assert len(prompts) == len(EVAL_ROWS)
    few_shot = [p for p in prompts if "### Solved example" in p]
    assert len(few_shot) == len(EVAL_ROWS)
    train_texts = {row["text"] for row in TRAIN_ROWS}
    for prompt in few_shot:
        shown_eval = [row["text"] for row in EVAL_ROWS if row["text"] in prompt]
        assert len(shown_eval) == 1, (
            f"{len(shown_eval)} eval row(s) in one prompt — the row being scored is the only one "
            f"that may appear: {shown_eval}"
        )
        shown_train = [text for text in train_texts if text in prompt]
        assert len(shown_train) == FITNESS_SHOTS, (
            f"expected {FITNESS_SHOTS} train demonstrations, prompt carried {len(shown_train)}"
        )


def test_a_demonstration_shows_the_gold_answer_in_the_task_s_own_training_shape():
    """A demonstration is a prompt/answer PAIR built by the task's own `build_training_turn`, so
    the teacher sees exactly the shape fine-tuning would show the student. Showing the inputs
    without the answers would demonstrate nothing about the output contract, which is the whole
    reason the measurement is five-shot rather than zero-shot (B276).
    """
    from tasks._builders import TrainingContext

    prompts: list[str] = []
    measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_perfect_teacher(prompts),
        log=lambda *_: None,
    )
    ctx = TrainingContext(
        labels=tuple(sorted({row["label"] for row in TRAIN_ROWS})),
        instruction="Answer the following question:",
    )
    pairs = {}
    for row in TRAIN_ROWS:
        turn_prompt, target, _marker = SPEC.build_training_turn(row, ctx)
        pairs[row["text"]] = f"{turn_prompt}\n{target}"

    for prompt in [p for p in prompts if "### Solved example" in p]:
        shown = [text for text, pair in pairs.items() if pair in prompt]
        assert len(shown) == FITNESS_SHOTS, (
            "the demonstrations are not full prompt/answer pairs in the training shape; "
            f"matched {len(shown)} of {FITNESS_SHOTS}"
        )


def test_each_demonstration_is_fenced_and_the_real_question_is_marked():
    """Otherwise the teacher cannot tell which block is the question, and demonstrations make it WORSE.

    `build_training_turn` returns a COMPLETE prompt, so k demonstrations repeat the whole task
    instruction k+1 times. With only a blank line between them, measured 2026-08-21: xlam_bfcl scored
    0.8350 five-shot against 0.8680 zero-shot and calendar_json 0.7350 against 0.8368, both with
    format_valid at ~1.0 — so not a contract problem, an addressing one. calendar_json shows the
    mechanism plainly: every block carries its own "Current date and time", so six reference instants
    arrive with nothing saying which governs the answer (B320).
    """
    prompts: list[str] = []
    measure_teacher_fitness(
        SPEC, _eval_set(EVAL_ROWS[:2]), TRAIN_ROWS,
        generate_fn=_perfect_teacher(prompts), log=lambda *_: None,
    )
    few_shot = [p for p in prompts if "### Solved example" in p]
    assert few_shot, "no few-shot prompt was issued"
    for prompt in few_shot:
        # Every demonstration numbered, so the count is unambiguous rather than inferred.
        for index in range(1, FITNESS_SHOTS + 1):
            assert f"### Solved example {index} of {FITNESS_SHOTS}" in prompt
        # The question is marked, and comes after every demonstration.
        marker = "### Now answer THIS request only"
        assert marker in prompt
        assert prompt.index(marker) > prompt.rindex("### Solved example")
        # And the instruction that fixes calendar_json specifically: ignore the demos' own context.
        assert "reference instant" in prompt


def test_only_the_k_shot_prompt_is_measured_because_that_is_what_synthesis_sends():
    """One measurement, in the shape synthesis actually uses.

    The gate authorises SYNTHESIS and synthesis prompts k-shot, so a teacher authorised on a
    zero-shot score it will never be asked to reproduce is authorised on the wrong evidence. That
    reading was already fixed; what this pins is that the second, zero-shot pass is no longer TAKEN
    either. It survived as commentary after nothing consumed it, doubling the cost of a
    measurement that now also sets the run's accuracy goal. The B320 comparison it provided lives
    in `scripts/probe_teacher_fewshot.py`, which exists for exactly that.
    """
    def _teacher(prompt, *args, **kwargs):
        # Deliberately incompetent WITH demonstrations, perfect without them.
        if "### Solved example" in prompt:
            return "definitely-not-a-label"
        for row in EVAL_ROWS:
            if row["text"] in prompt:
                return row["label"]
        return ""

    prompts: list[str] = []

    def _recording(prompt, *args, **kwargs):
        prompts.append(prompt)
        return _teacher(prompt, *args, **kwargs)

    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_recording, log=lambda *_: None,
    )
    assert verdict["shots"] == FITNESS_SHOTS, "the gate must read the shots synthesis sends"
    assert verdict["synthesis_allowed"] is False, "the k-shot teacher is incompetent here"
    assert len(prompts) == len(EVAL_ROWS), "the zero-shot pass must not be paid for"
    assert all("### Solved example" in p for p in prompts)


def test_the_verdict_reports_format_validity_beside_the_score():
    """They fail differently and are fixed differently, so neither may be read alone.

    A low score at format_valid 1.0 is a capability ceiling; a low format_valid is a broken output
    contract that bounds the score no matter how capable the model is. Reporting only the first is
    what let B290 read as "fine-tuning does not help this task" for two whole runs — and this
    number now sets the accuracy goal as well as the synthesis gate, so a goal depressed by a
    formatting failure has to be visible as one.
    """
    logs: list[str] = []
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_perfect_teacher(), log=logs.append,
    )
    assert isinstance(verdict["format_valid"], float)
    assert any("format_valid" in line for line in logs)


def test_the_verdict_names_the_teacher_it_measured():
    """0.87 from Qwen3.6 and 0.87 from deepseek-v4-flash are different claims, and the verdict now
    sets the run's accuracy goal, so the goal is only interpretable with the model named."""
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_perfect_teacher(), log=lambda *_: None,
    )
    import config.config as config

    assert verdict["model"] == config.SYNTH_MODEL


def test_a_teacher_that_fails_most_rows_is_unmeasured_rather_than_scored_zero():
    """B313: an endpoint that errors on every row produces a clean-looking 0.0000.

    A failed generation scores as an empty prediction, so a broken harness is indistinguishable
    from an incapable teacher — and on run 38661753 all 1,000 rows failed, the baseline read
    0.0000, and the accuracy goal was quietly floored at 0.80 on the strength of it. The
    failure-rate guard used to live in `eval/endpoint_eval`, which protected the goal; that path no
    longer runs, so the guard moved here with the measurement.
    """
    def _broken(prompt, *args, **kwargs):
        raise RuntimeError("'str' object is not callable")

    logs: list[str] = []
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_broken, log=logs.append,
    )
    assert verdict["status"] == "unmeasured"
    assert verdict["score"] is None, "a harness failure must not be reported as a score of zero"
    assert verdict["synthesis_allowed"] is False
    assert "generation failed" in verdict["reason"]
    assert any("describe the harness rather than the model" in line for line in logs)


def test_a_few_failed_rows_still_produce_a_measurement():
    """Zero tolerance would be wrong: a handful of refusals or truncations is normal, and the
    surviving rows still measure something. The guard is for wholesale failure, not for noise."""
    state = {"n": 0}

    def _flaky(prompt, *args, **kwargs):
        state["n"] += 1
        if state["n"] % 10 == 0:
            raise RuntimeError("transient")
        for row in EVAL_ROWS:
            if row["text"] in prompt:
                return row["label"]
        return ""

    logs: list[str] = []
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_flaky, log=logs.append,
    )
    assert verdict["status"] == "measured"
    assert any("failed to generate and scored as empty" in line for line in logs)


def test_no_train_rows_measures_zero_shot_and_says_so():
    """Honest rather than silent: a zero-shot number understates a format-bound task by up to 6.4x
    (B276), so a verdict measured without demonstrations has to be readable as one."""
    logs: list[str] = []
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), [], generate_fn=_perfect_teacher(), log=logs.append,
    )
    assert verdict["status"] == "measured"
    assert any("measuring zero-shot" in line for line in logs)


def test_a_short_train_pool_uses_what_it_has_and_says_so():
    logs: list[str] = []
    prompts: list[str] = []
    measure_teacher_fitness(
        SPEC, _eval_set(EVAL_ROWS[:3]), TRAIN_ROWS[:2],
        generate_fn=_perfect_teacher(prompts), log=logs.append,
    )
    assert any("only 2 of" in line and "demonstration" in line for line in logs)
    for prompt in [p for p in prompts if "### Solved example" in p]:
        assert sum(1 for row in TRAIN_ROWS[:2] if row["text"] in prompt) == 2


# --------------------------------------------------------------------------
# Reading the verdict off the state
# --------------------------------------------------------------------------


def test_an_absent_verdict_is_a_refusal():
    """The single most important line in the module. A run that somehow skipped the gate must not
    silently regain synthesis — "we could not check" is not evidence of a fit teacher."""
    assert synthesis_allowed({}) is False


@pytest.mark.parametrize("verdict", [None, "measured", 0.9, [], {"score": 0.95}])
def test_a_verdict_that_is_not_a_decision_is_a_refusal(verdict):
    """Includes the shape that matters most: a dict carrying a high SCORE but no explicit
    `synthesis_allowed`. Reading the score directly here would duplicate the threshold comparison
    in a second place and let the two drift."""
    assert synthesis_allowed({"teacher_fitness": verdict}) is False


def test_an_explicit_pass_is_the_only_thing_that_allows_synthesis():
    assert synthesis_allowed({"teacher_fitness": {"synthesis_allowed": True}}) is True
    assert synthesis_allowed({"teacher_fitness": {"synthesis_allowed": False}}) is False


def test_the_measured_verdict_is_what_the_state_reader_consumes():
    """The two halves are tested apart everywhere else; this is the round trip, so a change to the
    verdict's key names cannot pass both halves separately."""
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_perfect_teacher(), log=lambda *_: None,
    )
    assert synthesis_allowed({"teacher_fitness": verdict}) is True

    refused = measure_teacher_fitness(
        SPEC, _eval_set(), TRAIN_ROWS, generate_fn=_majority_teacher, log=lambda *_: None,
    )
    assert synthesis_allowed({"teacher_fitness": refused}) is False


def test_the_gate_matches_the_shots_synthesis_actually_sends():
    """The gate authorises a prompt shape. Measuring at a different number of demonstrations than
    synthesis sends would authorise a teacher on the strength of a prompt nothing else issues."""
    from data.curriculum import SYNTH_SHOTS

    assert FITNESS_SHOTS == SYNTH_SHOTS


# --------------------------------------------------------------------------
# A demonstration block that cannot fit is skipped, not sent
# --------------------------------------------------------------------------


def test_a_demonstration_block_too_large_for_the_context_is_skipped():
    """Measured on toolbench job 38817759: all 199 five-shot requests returned HTTP 400.

    `build_training_turn` returns a COMPLETE prompt, so k demonstrations cost k full prompts. On a
    task with large rows that exceeds any context — ToolBench's median row is 2,292 tokens against a
    teacher served at 8,192 — and the failure was actively misleading rather than merely wasteful:
    every request returned an empty string, the k-shot pass scored `format_valid=0.0000`, and because
    the gate takes the BEST of its measurements the code logged "demonstrations made this teacher
    WORSE on this task". That is the B320 wording for a prompt-assembly defect, and it pointed at the
    model instead of at the prompt size.

    So an over-large prefix is dropped BEFORE the calls, and the k-shot pass simply does not happen.
    """
    from agent.teacher_fitness import _prefix_fits
    from config.config import SYNTH_MAX_MODEL_LEN

    logs: list[str] = []
    # The TEACHER's served context, not the task's. See `_prefix_fits`.
    budget_chars = int((SYNTH_MAX_MODEL_LEN - SPEC.max_new_tokens) * 2.5)
    longest_prompt = 400
    room = budget_chars - longest_prompt

    assert _prefix_fits("x" * room, SPEC, longest_prompt, log=logs.append) is True
    assert logs == [], "a prefix that fits must not be reported"

    assert _prefix_fits("x" * (room + 1), SPEC, longest_prompt, log=logs.append) is False
    joined = " ".join(logs)
    assert "SKIPPED" in joined
    # The log must name the real cause, or the next reader repeats the B320 misdiagnosis.
    assert "NOT evidence" in joined and "B320" in joined
    # And it must say what to do, because synthesis hits the identical wall.
    assert "SLM_SYNTH_SHOTS" in joined


def test_an_oversized_prefix_degrades_to_zero_shot_rather_than_to_no_verdict():
    """A task whose rows crowd out the demonstrations still gets a decision.

    The gate's contract is that it always returns one; falling through to "unmeasured" — which
    REFUSES synthesis and, now, also aborts the run for want of an accuracy goal — would be the
    wrong answer for the wrong reason. The shot count RECORDED must be the one actually sent, so a
    reader can tell "this teacher is weak" from "this task left no room to show it the format".
    """
    huge_train = tuple(
        {"text": "train row " + "padding " * 20000, "label": "route"} for _ in range(5)
    )
    prompts: list[str] = []
    verdict = measure_teacher_fitness(
        SPEC, _eval_set(), huge_train,
        generate_fn=_perfect_teacher(prompts), log=lambda *_: None,
    )
    assert verdict["status"] == "measured"
    assert verdict["shots"] == 0
    assert verdict["shots_requested"] == FITNESS_SHOTS
    assert not any("### Solved example" in p for p in prompts)
