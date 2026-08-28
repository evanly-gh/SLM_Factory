"""Is the teacher good enough to generate training data for THIS task?

WHY THIS EXISTS
    Synthetic data is only worth having if the model writing it can do the task. Until now nothing
    checked: `surgical_synthesis` was offered on every task at every score, and the only evidence
    anyone had about teacher competence was its ZERO-SHOT score from
    `eval/endpoint_eval.measure_endpoint_baseline`. That number is not the right input, twice over:

      * it is measured zero-shot, so on a format-bound task it mostly reports whether the teacher
        guessed our output contract. BC5CDR measures 0.1131 zero-shot and 0.7190 with five
        demonstrations — a 6.4x difference that says nothing changed about the teacher's competence,
        only about what it had been told (B276);
      * it was never consulted before spending the teacher's budget anyway.

    So the gate measures the teacher the way synthesis actually prompts it — FIVE-SHOT, through the
    task's own scorer — and if it cannot clear `MIN_ACCURACY` on the task's own held-out eval set,
    synthetic data is refused for the rest of the run. A teacher that gets a task right less than
    four times in five is not a source of training targets for it; it is a source of labelled noise,
    and the project's best measured result came from a gold-only curriculum.

WHAT IT IS NOT
    Not a substitute for per-row verification. A teacher that clears the gate still has every
    generated row checked — programmatically where an exact check exists, by the teacher itself
    otherwise. This gate answers a coarser question that has to be asked first: is generating rows
    for this task a sensible thing to spend calls on at all.

CONTAMINATION
    Demonstrations are drawn from TRAIN only, never from the eval set, so measuring here cannot leak
    held-out rows into a prompt that later scores them.
"""
from __future__ import annotations

import os
import random

# The bar a teacher must clear on a task before its output may enter that task's curriculum.
# 0.80 is the number the project already treats as "can do the task": it is the floor the accuracy
# goal itself is calibrated against (`agent/threshold.py`), so a teacher below it is worse than the
# student is expected to become, and training a student on its output caps the student at the
# teacher's error rate.
MIN_ACCURACY = float(os.environ.get("SLM_TEACHER_SYNTH_MIN_ACCURACY", "0.80"))

# Operator override: measure the teacher honestly, then let synthesis proceed anyway.
#
# WHY THIS IS A SEPARATE FLAG RATHER THAN `SLM_TEACHER_SYNTH_MIN_ACCURACY=0`
#     Lowering the threshold to zero reaches the same decision and destroys the evidence on the way:
#     the verdict then records `threshold=0.00`, `log_fitness` prints "against a 0.00 gate", and a
#     reader six weeks later cannot tell a teacher that PASSED a real gate from one that was waved
#     through. This flag changes only the DECISION. The measurement, the real threshold and the
#     comparison are all still recorded, and `bypassed=True` is stamped on the verdict so any report
#     reading it can say so.
#
# WHEN IT IS LEGITIMATE
#     Bringing up a NEW task, where the question being answered is "does the synthesis path run at
#     all on this task" rather than "is this curriculum any good". `toolbench` is the case it was
#     added for: its metric is a judged pass rate, so the teacher's fitness score is itself a judged
#     number, and gating a first bring-up run on it means a cold judge or an unlucky rubric silently
#     removes the intervention under test.
#
#     It is NOT legitimate for a results run. A teacher below the gate is a source of labelled noise,
#     and the project's best measured result came from a gold-only curriculum — so a run that sets
#     this and then reports an accuracy is reporting an accuracy obtained with known-noisy data.
BYPASS = str(os.environ.get("SLM_TEACHER_SYNTH_BYPASS", "0")).strip().lower() in {
    "1", "true", "yes", "on",
}

# Demonstrations shown when measuring. Must match what synthesis actually sends, or the gate
# measures a different prompt than the one it is authorising — see data/curriculum.SYNTH_SHOTS.
FITNESS_SHOTS = int(os.environ.get("SLM_SYNTH_SHOTS", "5"))

# Eval rows scored. The full set is unnecessary for a go/no-go decision and costs teacher calls that
# are better spent generating; 200 rows put the standard error on a proportion near 3 points, which
# is far tighter than the margin this decision turns on.
FITNESS_EVAL_ROWS = int(os.environ.get("SLM_TEACHER_FITNESS_ROWS", "200"))


def _apply_bypass(verdict: dict, spec, *, log=print) -> dict:
    """Stamp the operator override onto a verdict, loudly, and return it.

    Called at every exit from `measure_teacher_fitness` so there is no path on which the flag is set
    and quietly ignored — including the unmeasured ones, which is the point: an unreachable endpoint
    normally refuses synthesis, and during a bring-up run that is exactly the refusal an operator
    means to override.
    """
    if not BYPASS or verdict.get("synthesis_allowed"):
        return verdict
    verdict["synthesis_allowed"] = True
    verdict["bypassed"] = True
    score = verdict.get("score")
    measured = f"{score:.4f}" if isinstance(score, (int, float)) else "unmeasured"
    log(
        f"      [teacher] ⚠ SLM_TEACHER_SYNTH_BYPASS=1 — {spec.name}'s teacher measured "
        f"{measured} against a {MIN_ACCURACY:.2f} gate it did NOT clear, and synthetic data is "
        f"being ALLOWED anyway by operator override. Every generated row is still verified "
        f"individually. Any accuracy from this run was obtained with data the gate would have "
        f"refused; do not report it as a clean result."
    )
    return verdict


def _training_context(rows: list[dict], spec):
    """Dataset-level context the task's turn builder needs (label vocabulary, instruction).

    Resolved exactly as `training.lora_trainer` resolves it, so a demonstration shows the teacher
    the same prompt/answer pair fine-tuning would have shown the student.
    """
    from eval.scorers.generation import resolve_generation_instruction
    from tasks._builders import TrainingContext

    labels = tuple(sorted({
        str(row.get("label", "")) for row in rows if row.get("label")
    })) if spec.closed_label_space else ()
    return TrainingContext(labels=labels, instruction=resolve_generation_instruction(rows))


# Framing around each demonstration. Load-bearing, and it was missing.
#
# `build_training_turn` returns a COMPLETE prompt — task instruction, the tool/label vocabulary, the
# request — because that is what fine-tuning shows the student. Concatenating k of them therefore
# repeats the whole instruction k+1 times, and with only a blank line between them there is nothing to
# say where one example ends, which are answered already, and which one is the question.
#
# Measured 2026-08-21 on two tasks, both times with format_valid at or near 1.0 (so not a contract
# problem): xlam_bfcl scored 0.8350 five-shot against 0.8680 zero-shot, and calendar_json 0.7350
# against 0.8368. Demonstrations making a model WORSE is the signature of exactly this ambiguity, and
# calendar_json shows why it bites hardest there — every block carries its own
# "Current date and time", so six different reference instants arrive with nothing marking which one
# governs the answer. The gate then read the depressed number and refused synthetic data (B320).
_DEMO_HEADER = "### Solved example {n} of {total}"
_QUESTION_HEADER = (
    "### Now answer THIS request only\n"
    "The examples above are already answered and are shown only to demonstrate the required output "
    "format and conventions. Ignore their specifics — in particular any date, time or reference "
    "instant they mention — and answer strictly the request below, using only ITS own context."
)


# Characters per token when checking whether a demonstration block can fit. Deliberately LOW, so the
# estimate over-counts tokens and the guard errs toward skipping — a skipped k-shot pass costs one
# measurement, while an over-long one costs a request per row and returns an empty string that scores
# as a format failure.
_FIT_CHARS_PER_TOKEN = 2.5


def _prefix_fits(prefix: str, spec, longest_prompt_chars: int, *, log=print) -> bool:
    """Whether a demonstration block leaves room for the LONGEST question and the answer.

    WHY THIS EXISTS (measured 2026-08-24, toolbench job 38817759)
        `build_training_turn` returns a COMPLETE prompt, so k demonstrations cost k full prompts. On
        a task whose rows are large that does not fit any context: ToolBench's median training row is
        2,292 tokens, so the default five demonstrations are ~11,500 tokens before the question is
        appended, against a teacher served at 8,192. Every single request 400'd.

        The damage was not the wasted calls. `measure_teacher_fitness` takes the BEST of its 0-shot
        and k-shot measurements, so a k-shot pass where every request failed returned
        `format_valid=0.0000` and score 0.0000 — and the code then logged "demonstrations made this
        teacher WORSE on this task", which is the B320 diagnosis for a prompt-assembly defect. It was
        not wrong about that, but the real cause was that the prompt did not fit, and nothing said so.

        So the check happens before the calls, and the log says which of the two it is.

    The headroom is measured against the LONGEST prompt that will actually be sent, not a fraction of
    the budget. That matters in both directions: an arbitrary fraction would disable demonstrations on
    small-context tasks where they fit perfectly well, and a bound taken against the MEDIAN prompt
    would let the longest rows fail while the rest succeeded — a partial failure that shows up as a
    depressed score rather than as an error. Using the longest prompt makes the guarantee "no request
    in this measurement will exceed the context", which is the only version worth having.

    The bound is the TEACHER's served context, not the task's. Using `spec.max_seq_length` here was
    wrong and routerbench is the proof: its five-shot prefix is ~2,240 tokens against a 974-token
    student budget, and it fits the teacher's 8,192 perfectly well. Bounding a teacher prompt by the
    student's window would have disabled demonstrations on exactly the tasks B276 says need them.
    """
    from eval.harness import eval_output_token_reserve

    from config.config import SYNTH_MAX_MODEL_LEN

    budget_tokens = max(
        int(SYNTH_MAX_MODEL_LEN) - int(eval_output_token_reserve(spec.name)), 256
    )
    allowed_chars = int(budget_tokens * _FIT_CHARS_PER_TOKEN) - int(longest_prompt_chars)
    if len(prefix) <= allowed_chars:
        return True
    log(
        f"      [teacher] {spec.name}: a {len(prefix)}-char demonstration block does not fit beside "
        f"the longest {longest_prompt_chars}-char prompt inside a {budget_tokens}-token input budget "
        f"(room for {max(allowed_chars, 0)} chars of demonstrations), so the k-shot measurement is "
        f"SKIPPED and the gate uses the zero-shot number. This is a prompt-SIZE limit on a task with "
        f"large rows, NOT evidence that demonstrations hurt (B320) — and synthesis prompts the same "
        f"way, so RAISE SLM_SYNTH_MAX_MODEL_LEN for this task (and lower SLM_SYNTH_MAX_NUM_SEQS to "
        f"pay for the KV cache) until five demonstrations fit.\n"
        f"      [teacher] Do NOT respond by lowering SLM_SYNTH_SHOTS. That has been tried twice on "
        f"toolbench and failed both times: at 0 shots the verifier kept 0 rows of 1,019 attempts, and "
        f"at the 1-2 shots that DID fit an 8,192 context it kept 0 of 133 (runs 38832586, 38985393). A "
        f"format-bound task cannot be synthesized from a prompt that does not show the format enough "
        f"times (B276), so fewer shots trades a size error for a silent total loss of yield."
    )
    return False


def fit_demonstrations(pool: list[dict], shots: int, budget_chars: int, rng=None) -> list[dict]:
    """Pick `shots` demonstrations that actually FIT the budget, shortest-first.

    WHY NOT A RANDOM SAMPLE (measured 2026-08-25/26)
        `build_training_turn` returns a COMPLETE prompt, so k demonstrations cost k full prompts. On
        `toolbench` the mean training row is 2,308 tokens, so five RANDOM demonstrations are ~11,500
        tokens against a teacher served at 8,192 and every request 400s. The reflex fix — drop to
        zero-shot — is worse than it looks: zero-shot synthesis on that task generated 636, 326 and
        57 candidate rows across three rebuilds and the programmatic verifier kept **zero** of them,
        every time. A format-bound task cannot be synthesized without showing the format (B276).

        The rows are not uniformly large, though: toolbench's shortest complete paths are ~809
        tokens, so five SHORT demonstrations are ~4,000 and fit with room to spare. Choosing which
        five to show is free, and it is the difference between five-shot synthesis and none.

    Shortest-first rather than a random draw among the fitting rows: the point is to spend as little
    of the context as possible on demonstrations so the question itself has room, and a demonstration
    is showing FORM, which the shortest example shows as well as the longest.
    """
    if shots <= 0 or not pool:
        return []
    ordered = sorted(pool, key=lambda row: len(str(row.get("text", ""))) + len(str(row.get("answer", ""))))
    chosen: list[dict] = []
    used = 0
    for row in ordered:
        cost = len(str(row.get("text", ""))) + len(str(row.get("answer", ""))) + 64
        if chosen and used + cost > budget_chars:
            break
        chosen.append(row)
        used += cost
        if len(chosen) >= shots:
            break
    return chosen


def _demo_block(demos: list[dict], spec, ctx) -> str:
    """k demonstrations in the same prompt/answer shape the model is about to be asked for.

    Built from the task's `build_training_turn`, which imports its prompt from the eval scorer, so
    this cannot drift into measuring a prompt nothing else in the pipeline sends. Each one is fenced
    by a numbered header — see `_DEMO_HEADER` for why that is not cosmetic.
    """
    parts = []
    total = len(demos)
    for index, row in enumerate(demos, start=1):
        prompt, target, _marker = spec.build_training_turn(row, ctx)
        parts.append(f"{_DEMO_HEADER.format(n=index, total=total)}\n{prompt}\n{target}")
    return "\n\n".join(parts)


def measure_teacher_fitness(
    spec,
    eval_set,
    train_rows: list[dict],
    *,
    generate_fn=None,
    shots: int = FITNESS_SHOTS,
    n_rows: int = FITNESS_EVAL_ROWS,
    seed: int = 20260819,
    log=print,
) -> dict:
    """Score the teacher `shots`-shot on this task's own eval set.

    Returns a verdict dict — always, never raises. An unreachable endpoint yields
    ``status="unmeasured"``, and an unmeasured teacher is NOT trusted: synthesis is refused, because
    "we could not check" is not evidence of fitness. That is the conservative direction, and it is
    also the honest one — the alternative silently authorises a teacher nobody measured.
    """
    from data.eval_set import EvalSet

    verdict = {
        "status": "unmeasured",
        "shots": shots,
        "score": None,
        "format_valid": None,
        "n": 0,
        "threshold": MIN_ACCURACY,
        "synthesis_allowed": False,
        "metric": spec.metric_name,
    }
    if generate_fn is None:
        from data.synth_client import get_generate_fn

        generate_fn = get_generate_fn(log=log)
    if generate_fn is None:
        log("      [teacher] endpoint unreachable — teacher fitness UNMEASURED, so synthetic data "
            "is refused for this run. A teacher nobody checked is not evidence of a fit teacher.")
        verdict["reason"] = "synthesis endpoint unreachable"
        return _apply_bypass(verdict, spec, log=log)

    rows = list(getattr(eval_set, "all", []) or [])
    if not rows:
        verdict["reason"] = "eval set is empty"
        return _apply_bypass(verdict, spec, log=log)
    rng = random.Random(seed)
    scored_rows = rows[:max(1, n_rows)]
    probe_set = EvalSet(all=scored_rows, task=spec.name)

    from eval.harness import eval_output_token_reserve

    max_tokens = eval_output_token_reserve(spec.name)
    # Built BEFORE the demonstration block, because whether that block fits depends on the longest
    # prompt it has to sit beside.
    base_prompts = spec.build_prompts(probe_set)

    pool = [row for row in (train_rows or []) if isinstance(row, dict)]
    ctx = _training_context(pool or scored_rows, spec)
    prefix = ""
    if shots and pool:
        from config.config import SYNTH_MAX_MODEL_LEN

        longest = max((len(prompt) for prompt in base_prompts), default=0)
        budget = int(
            (SYNTH_MAX_MODEL_LEN - max_tokens) * _FIT_CHARS_PER_TOKEN
        ) - longest
        demos = fit_demonstrations(pool, shots, budget, rng=rng)
        prefix = _demo_block(demos, spec, ctx) + "\n\n" + _QUESTION_HEADER + "\n" if demos else ""
        if len(demos) < shots:
            log(f"      [teacher] only {len(demos)} of {shots} demonstration(s) fit beside the "
                f"longest {longest}-char prompt in the teacher's {SYNTH_MAX_MODEL_LEN}-token "
                f"context; showing what fits rather than sending a request that cannot be served")
        if prefix and not _prefix_fits(prefix, spec, longest, log=log):
            prefix = ""
    elif shots:
        log("      [teacher] no train rows to draw demonstrations from — measuring zero-shot, "
            "which understates a format-bound task (B276)")

    log(f"      [teacher] measuring fitness: {len(base_prompts)} eval row(s), "
        f"{shots if prefix else 0}-shot AND zero-shot, metric {spec.metric_name}, "
        f"gate {MIN_ACCURACY:.2f}")

    def _one(prompt: str) -> str:
        try:
            return generate_fn(prompt, temperature=0.0, max_tokens=max_tokens)
        except Exception as error:  # noqa: BLE001 — one bad row must not abort the measurement
            log(f"      [teacher] generation failed on one row: {type(error).__name__}: {error}")
            return ""

    from concurrent.futures import ThreadPoolExecutor

    def _measure(prompt_prefix: str) -> dict:
        prompts = [prompt_prefix + prompt for prompt in base_prompts]
        workers = max(1, min(16, len(prompts)))
        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            raw = list(pool_exec.map(_one, prompts))
        return spec.score(probe_set, spec.extract_predictions(raw, probe_set))

    # BOTH prompting schemes, and the gate reads the better one.
    #
    # The gate answers "can this teacher do this task", and the honest answer is the best measurement
    # available rather than one arbitrary prompt shape. Measuring only k-shot let a prompt-assembly
    # defect masquerade as an incapable teacher and refuse synthetic data on both tasks (B320); a
    # future one would do the same. `best_shots` records which won, so synthesis can prompt the way
    # the authorising measurement was taken instead of contradicting it.
    try:
        measurements = {shots: _measure(prefix)} if prefix else {}
        measurements[0] = _measure("")
    except Exception as error:  # noqa: BLE001 — an unmeasurable teacher is refused, not fatal
        log(f"      [teacher] fitness measurement FAILED ({type(error).__name__}: {error}); "
            "synthetic data is refused for this run")
        verdict["reason"] = f"{type(error).__name__}: {error}"
        return _apply_bypass(verdict, spec, log=log)

    for used_shots, result in sorted(measurements.items()):
        log(f"      [teacher]   {used_shots}-shot: {result.get('metric', spec.metric_name)}="
            f"{float(result.get('f1', 0.0)):.4f} "
            f"(format_valid={float(result.get('format_valid', 1.0)):.4f})")
    # THE GATE READS THE k-SHOT NUMBER, because that is the prompt synthesis actually sends.
    #
    # It used to read the BEST of the two, which was a defensible reading of "can this teacher do the
    # task" and the wrong reading of the question the gate exists to answer. The gate authorises
    # SYNTHESIS, synthesis prompts k-shot, so the number that decides it must be the k-shot one — a
    # teacher authorised on a zero-shot score it will never be asked to reproduce is authorised on
    # the wrong evidence. Both are still measured and both are still reported, so the comparison
    # that motivated best-of (B320: demonstrations making a teacher look worse is a prompt-assembly
    # smell) stays visible in the log.
    gate_shots = shots if shots in measurements else 0
    result = measurements[gate_shots]
    if len(measurements) > 1 and float(measurements[0].get("f1", 0.0)) > float(result.get("f1", 0.0)):
        log(f"      [teacher]   NOTE: this teacher scores HIGHER zero-shot "
            f"({float(measurements[0].get('f1', 0.0)):.4f}) than {gate_shots}-shot "
            f"({float(result.get('f1', 0.0)):.4f}). The gate still uses the {gate_shots}-shot number "
            f"because that is what synthesis sends, but demonstrations making a teacher worse is the "
            f"signature of a prompt-assembly defect — see B320 before assuming the model is at fault.")
    best_shots = gate_shots

    score = float(result.get("f1", 0.0))
    verdict.update({
        "status": "measured",
        "score": score,
        "format_valid": float(result.get("format_valid", 1.0)),
        "n": len(base_prompts),
        "shots": best_shots,
        "shots_measured": sorted(measurements),
        "synthesis_allowed": score >= MIN_ACCURACY,
        "metric": result.get("metric", spec.metric_name),
    })
    log_fitness(verdict, spec, log=log)
    return _apply_bypass(verdict, spec, log=log)


def log_fitness(verdict: dict, spec, *, log=print) -> None:
    """State the verdict and its consequence in the same place, so neither can be read alone."""
    if verdict.get("status") != "measured":
        log(f"      [teacher] {spec.name}: fitness UNMEASURED "
            f"({verdict.get('reason', 'no reason recorded')}) — synthetic data REFUSED")
        return
    score = verdict["score"]
    allowed = verdict["synthesis_allowed"]
    log(f"      [teacher] {spec.name}: teacher scores {verdict['metric']}={score:.4f} "
        f"(format_valid={verdict['format_valid']:.4f}) {verdict['shots']}-shot on "
        f"{verdict['n']} eval row(s), against a {verdict['threshold']:.2f} gate")
    if allowed:
        log("      [teacher] → synthetic data is ALLOWED for this run. Every generated row is "
            "still verified individually.")
    else:
        # Two decimal places, not a rounded percentage. calendar_json measured 0.7950 against a 0.80
        # gate and the old `{score:.0%}` rendered it as "gets this task right 80% of the time" directly
        # above a REFUSAL, which reads like a bug in the comparison rather than a near miss.
        log(f"      [teacher] → synthetic data is REFUSED for this run: {score:.4f} is below the "
            f"{verdict['threshold']:.2f} gate, and a teacher at that accuracy is a source of labelled "
            f"noise rather than training targets. surgical_synthesis is off the menu; mine_new_real "
            f"and hyperparameter interventions are unaffected.")


def synthesis_allowed(state) -> bool:
    """Whether this run may generate synthetic rows.

    Absent a measurement the answer is NO. `state.get(...)` defaulting to True would mean a run that
    somehow skipped the gate silently regained synthesis, which is the failure mode the gate exists
    to prevent.

    `BYPASS` is honoured here as well as on the verdict, so the override holds even on a run that
    never reached the gate at all — a resumed checkpoint written before the flag was set, for one.
    Reading the flag in both places means the answer cannot depend on which of them ran.
    """
    verdict = state.get("teacher_fitness")
    if not isinstance(verdict, dict):
        return BYPASS
    return bool(verdict.get("synthesis_allowed")) or BYPASS
