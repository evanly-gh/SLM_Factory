"""THE teacher measurement: one 5-shot pass over the full held-out eval set, two consumers.

WHAT IT ANSWERS
    1. Is the teacher good enough to generate training data for this task? (the synthesis gate)
    2. What accuracy must the student beat? (the run's stop_threshold, via `agent/threshold.py`)

    Those are two readings of ONE number, so they are measured once. They did not used to be.
    `measure_teacher_fitness` scored 200 rows five-shot AND zero-shot for the gate, while
    `eval/endpoint_eval.measure_endpoint_baseline` separately scored the full eval set ZERO-shot for
    the goal — 1,400 teacher calls producing two numbers that disagreed with each other by
    construction, because they were different prompts over different samples. The goal was set by
    the one nothing else in the pipeline ever sends.

WHY FIVE-SHOT
    Because that is how synthesis prompts the teacher. A zero-shot number on a format-bound task
    mostly reports whether the teacher guessed our output contract: BC5CDR measures 0.1131
    zero-shot and 0.7190 with five demonstrations — a 6.4x difference that says nothing changed
    about the teacher's competence, only about what it had been told (B276). Gating synthesis on a
    prompt shape synthesis never uses is authorising on the wrong evidence; calibrating the run's
    accuracy goal against it sets the bar in the wrong place for the same reason.

    A task whose rows are too large for five demonstrations to fit degrades to zero-shot rather
    than sending requests that cannot be served (see `_prefix_fits`), and the shot count actually
    used is recorded on the verdict so a reader can tell which happened.

WHY THE FULL EVAL SET
    It is the set the student is scored on, so the teacher's number and the student's number are
    finally the same measurement on the same rows — which is the only way "the student matched the
    teacher" means anything. The old 200-row subsample was a cost compromise that no longer buys
    much: at the 1,000-row `select_cap` the standard error on a proportion is about 1.5 points, and
    the extra 800 calls cost cents even on the paid API teacher.

WHAT IT IS NOT
    Not a substitute for per-row verification. A teacher that clears the gate still has every
    generated row checked programmatically where an exact check exists. This gate answers a coarser
    question that has to be asked first: is generating rows for this task a sensible thing to spend
    calls on at all.

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

# Eval rows scored. 0 (the default) means THE WHOLE EVAL SET — the same rows the student is scored
# on, so the two numbers are comparable and "the student matched the teacher" is a claim about one
# measurement rather than two.
#
# This used to default to 200 as a cost compromise, back when the verdict fed only the go/no-go
# synthesis gate. It now also sets the run's accuracy goal, and a goal calibrated on a fifth of the
# eval set is a goal with a wider error bar than the differences the run is trying to detect. Set a
# positive value to subsample during bring-up on a new task; leave it at 0 for a results run.
FITNESS_EVAL_ROWS = int(os.environ.get("SLM_TEACHER_FITNESS_ROWS", "0"))

# Above this share of failed generations the measurement is discarded rather than scored — see the
# B313 note at the guard itself. Kept numerically identical to the
# `eval.endpoint_eval.MAX_GENERATION_FAILURE_RATE` this replaces, so the protection the accuracy
# goal used to have is the protection it still has.
MAX_GENERATION_FAILURE_RATE = 0.25

# Output tokens the TEACHER is given when it is a hosted API model, as a floor under the task's own
# eval reserve.
#
# WHY THE STUDENT'S RESERVE IS THE WRONG BUDGET FOR A REASONING TEACHER
#     `TaskSpec.max_new_tokens` is how much room the STUDENT gets to answer — 256 on xlam_bfcl,
#     sized for a small non-reasoning model emitting a JSON array and nothing else. Measuring the
#     teacher under the same number looks like the fair comparison, and for a non-reasoning teacher
#     it is: the local Qwen is rendered with thinking disabled, so every token it spends is answer.
#
#     A reasoning model spends completion tokens on its own reasoning FIRST, and those count against
#     the same limit. On run 39361189 `deepseek-v4-flash` came back with format_valid=0.7630 — a
#     quarter of its replies unparseable — and the pipeline's own warning said its score was
#     "bounded by a formatting failure rather than by competence". Probed directly on 24 real xlam
#     eval prompts, 3 of 24 hit `finish_reason=length` at 256 tokens against 1 of 24 at 2048, with
#     mean completion rising 140 -> 252: the budget was truncating answers mid-JSON.
#
#     That score is not cosmetic. It gates synthetic data AND calibrates the run's accuracy goal, so
#     a teacher truncated into looking incompetent lowers the bar the student is then held to.
#
# SCOPED TO API MODE ON PURPOSE. Applying it locally would change every teacher measurement this
# project has recorded, for a teacher that does not reason and therefore was never truncated. The
# floor only ever RAISES the reserve, so a task already asking for more keeps its own number.
TEACHER_API_OUTPUT_RESERVE = int(
    os.environ.get("SLM_TEACHER_API_OUTPUT_RESERVE", "2048")
)


def _teacher_output_reserve(spec) -> int:
    """Output tokens for one teacher call during the fitness measurement.

    Reads `SLM_SYNTH_API_MODE` from the environment rather than importing `config.config`, which
    raises on any unset API key — the same trap that once made `token_budget`'s context clamp
    silently inactive whenever an unrelated credential was missing.
    """
    from eval.harness import eval_output_token_reserve

    reserve = eval_output_token_reserve(spec.name)
    if os.environ.get("SLM_SYNTH_API_MODE", "0") != "1":
        return reserve
    return max(reserve, TEACHER_API_OUTPUT_RESERVE)


def _measurement_concurrency() -> int:
    """In-flight requests while measuring, matched to the teacher's configured fan-out.

    The measurement now covers the whole eval set rather than a 200-row subsample, so the old
    hardcoded 16 workers turned a five-minute pass into a twenty-five-minute one on a server
    already provisioned for `SLM_SYNTH_CONCURRENCY` (48 on the two-GPU profile). It is the same
    endpoint under the same budget as synthesis, so it should use the same number.
    """
    try:
        return max(1, int(os.environ.get("SLM_SYNTH_CONCURRENCY", "16")))
    except (TypeError, ValueError):
        return 16


def _apply_operator_overrides(verdict: dict, spec, *, log=print) -> dict:
    """Stamp any operator override onto a verdict, loudly, and return it.

    Called at every exit from `measure_teacher_fitness` so there is no path on which a flag is set
    and quietly ignored — including the unmeasured ones, which is the point: an unreachable endpoint
    normally refuses synthesis, and during a bring-up run that is exactly the refusal an operator
    means to override.

    The two overrides pull in opposite directions and REFUSAL WINS, checked first and returning
    immediately. `SLM_SYNTH_DISALLOW` is the ablation asking what the pipeline achieves with no
    synthetic data; letting `SLM_TEACHER_SYNTH_BYPASS` — which the curated launchers all set, to
    protect against a teacher drifting below the gate mid-experiment — hand synthesis back would
    make the ablation silently unanswerable.
    """
    # Imported here, as in `synthesis_allowed` below, because `agent.ablations` reaches
    # `agent.checkpoint` for its atomic JSONL writer and this module is on that import path.
    from agent.ablations import synthesis_disallowed

    if synthesis_disallowed():
        verdict["synthesis_allowed"] = False
        verdict["disallowed_by_operator"] = True
        score = verdict.get("score")
        measured = f"{score:.4f}" if isinstance(score, (int, float)) else "unmeasured"
        log(
            f"      [teacher] ⚠ SLM_SYNTH_DISALLOW=1 — synthetic data is REFUSED for this run by "
            f"operator override, not by the gate. {spec.name}'s teacher measured {measured} "
            f"against a {MIN_ACCURACY:.2f} gate"
            + (
                " and WOULD have been allowed to generate"
                if isinstance(score, (int, float)) and score >= MIN_ACCURACY
                else ""
            )
            + ". surgical_synthesis is off the menu for the whole run; mine_new_real and "
            "hyperparameter interventions are unaffected. This is the no-synthetic-data ablation: "
            "whatever accuracy this run reaches was reached without a single generated row."
        )
        return verdict
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


def _teacher_identity() -> tuple[str, str]:
    """The model and endpoint the measurement was taken against, for the audit trail.

    Named on the verdict because the verdict now sets the run's accuracy goal, and a goal is only
    interpretable if the reader knows which teacher set it — a 0.87 from Qwen3.6-35B and a 0.87
    from deepseek-v4-flash are not the same claim.
    """
    try:
        from config.config import SYNTH_ENDPOINT, SYNTH_MODEL
    except Exception:  # noqa: BLE001 — config may be unimportable in a bare unit test
        return "", ""
    return str(SYNTH_MODEL or ""), str(SYNTH_ENDPOINT or "")


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
    """Score the teacher `shots`-shot on this task's full held-out eval set.

    ONE measurement with TWO consumers: the synthesis gate reads `synthesis_allowed`, and
    `agent/nodes/cold_start/eval_setup` reads `score` to calibrate the run's accuracy goal. Both
    therefore see the same prompt shape over the same rows, which they did not before.

    Returns a verdict dict — always, never raises. An unreachable endpoint yields
    ``status="unmeasured"``, and an unmeasured teacher is NOT trusted: synthesis is refused, because
    "we could not check" is not evidence of fitness. That is the conservative direction, and it is
    also the honest one — the alternative silently authorises a teacher nobody measured. Goal
    calibration treats the same verdict as fatal, because a run with no measured reference has no
    honest target to converge against.
    """
    from data.eval_set import EvalSet

    model, endpoint = _teacher_identity()
    verdict = {
        "status": "unmeasured",
        "shots": shots,
        "shots_requested": shots,
        "score": None,
        "format_valid": None,
        "n": 0,
        "threshold": MIN_ACCURACY,
        "synthesis_allowed": False,
        "metric": spec.metric_name,
        "model": model,
        "endpoint": endpoint,
    }
    if generate_fn is None:
        from data.synth_client import get_generate_fn

        generate_fn = get_generate_fn(log=log)
    if generate_fn is None:
        log("      [teacher] endpoint unreachable — teacher fitness UNMEASURED, so synthetic data "
            "is refused for this run. A teacher nobody checked is not evidence of a fit teacher.")
        verdict["reason"] = "synthesis endpoint unreachable"
        return _apply_operator_overrides(verdict, spec, log=log)

    rows = list(getattr(eval_set, "all", []) or [])
    if not rows:
        verdict["reason"] = "eval set is empty"
        return _apply_operator_overrides(verdict, spec, log=log)
    rng = random.Random(seed)
    # n_rows <= 0 means the whole set, which is the default and what a results run wants: the
    # teacher's score is then measured on exactly the rows the student is measured on.
    scored_rows = rows if n_rows <= 0 else rows[:n_rows]
    probe_set = EvalSet(all=scored_rows, task=spec.name)

    max_tokens = _teacher_output_reserve(spec)
    # Built BEFORE the demonstration block, because whether that block fits depends on the longest
    # prompt it has to sit beside.
    base_prompts = spec.build_prompts(probe_set)

    pool = [row for row in (train_rows or []) if isinstance(row, dict)]
    ctx = _training_context(pool or scored_rows, spec)
    prefix = ""
    n_demos = 0
    if shots and pool:
        from config.config import SYNTH_MAX_MODEL_LEN

        longest = max((len(prompt) for prompt in base_prompts), default=0)
        budget = int(
            (SYNTH_MAX_MODEL_LEN - max_tokens) * _FIT_CHARS_PER_TOKEN
        ) - longest
        demos = fit_demonstrations(pool, shots, budget, rng=rng)
        prefix = _demo_block(demos, spec, ctx) + "\n\n" + _QUESTION_HEADER + "\n" if demos else ""
        n_demos = len(demos) if prefix else 0
        if len(demos) < shots:
            log(f"      [teacher] only {len(demos)} of {shots} demonstration(s) fit beside the "
                f"longest {longest}-char prompt in the teacher's {SYNTH_MAX_MODEL_LEN}-token "
                f"context; showing what fits rather than sending a request that cannot be served")
        if prefix and not _prefix_fits(prefix, spec, longest, log=log):
            prefix = ""
            n_demos = 0
    elif shots:
        log("      [teacher] no train rows to draw demonstrations from — measuring zero-shot, "
            "which understates a format-bound task (B276)")

    # The shots actually SENT, not the shots requested. They differ whenever a task's rows are too
    # large for the full block to fit, and recording the request would describe a prompt that was
    # never issued — which is the difference between "this teacher is weak" and "this task's rows
    # do not leave room for demonstrations" (B276/B320).
    used_shots = n_demos
    verdict["shots"] = used_shots
    log(f"      [teacher] measuring {model or 'teacher'} on the FULL eval set: "
        f"{len(base_prompts)} row(s), {used_shots}-shot, metric {spec.metric_name}. "
        f"This one number both gates synthetic data (gate {MIN_ACCURACY:.2f}) and calibrates the "
        f"run's accuracy goal.")

    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    failures: list[str] = []

    def _one(prompt: str) -> str:
        try:
            return generate_fn(prompt, temperature=0.0, max_tokens=max_tokens)
        except Exception as error:  # noqa: BLE001 — one bad row must not abort the measurement
            # COUNTED, not printed. This runs once per eval row and the set is now the full
            # thousand, so a broken endpoint used to mean a thousand identical stack-trace lines
            # burying the summary that explains them. The tally is reported once, below.
            failures.append(f"{type(error).__name__}: {error}"[:160])
            return ""

    def _measure(prompt_prefix: str) -> dict:
        prompts = [prompt_prefix + prompt for prompt in base_prompts]
        workers = max(1, min(_measurement_concurrency(), len(prompts)))
        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            raw = list(pool_exec.map(_one, prompts))
        return spec.score(probe_set, spec.extract_predictions(raw, probe_set))

    # ONE prompting scheme: the one synthesis actually sends.
    #
    # This used to measure k-shot AND zero-shot and report both, a habit from when the verdict was
    # read best-of. That reading was removed once it became clear the gate authorises SYNTHESIS and
    # synthesis prompts k-shot, so a teacher authorised on a zero-shot score it will never be asked
    # to reproduce is authorised on the wrong evidence. The second pass then survived purely as
    # commentary — it doubled the cost of the measurement to print a line nothing consumed — and it
    # became indefensible once the same verdict started setting the run's accuracy goal too.
    #
    # The diagnostic it used to provide (B320: demonstrations making a teacher LOOK worse is the
    # signature of a prompt-assembly defect, not an incapable model) is still reachable on demand
    # through `scripts/probe_teacher_fewshot.py`, which exists for exactly that comparison.
    try:
        result = _measure(prefix)
    except Exception as error:  # noqa: BLE001 — an unmeasurable teacher is refused, not fatal
        log(f"      [teacher] fitness measurement FAILED ({type(error).__name__}: {error}); "
            "synthetic data is refused for this run")
        verdict["reason"] = f"{type(error).__name__}: {error}"
        return _apply_operator_overrides(verdict, spec, log=log)

    # TOO MANY FAILED ROWS MEANS THERE IS NO MEASUREMENT (B313).
    #
    # A failed generation scores as an empty prediction, so an endpoint that errors on every row
    # produces a clean-looking 0.0000 that is indistinguishable from a genuinely incapable teacher.
    # On run 38661753 all 1,000 rows failed with `'str' object is not callable`, the baseline read
    # 0.0000, and the accuracy goal was quietly floored at 0.80 on the strength of it. This guard
    # came from `eval/endpoint_eval`, whose failure-rate check used to protect the goal; that path
    # no longer runs, so the check moves here with it.
    #
    # Not zero-tolerance: a few refusals or truncations are normal and the surviving rows still
    # measure something. Well under half, because a teacher that cannot answer most of the set is
    # not being measured either.
    if base_prompts and len(failures) / len(base_prompts) > MAX_GENERATION_FAILURE_RATE:
        common = Counter(failures).most_common(3)
        reason = (
            f"generation failed on {len(failures)} of {len(base_prompts)} eval row(s) "
            f"({len(failures) / len(base_prompts):.0%}); most common: "
            + "; ".join(f"{message} (x{count})" for message, count in common)
        )
        log(f"      [teacher] {reason}")
        log("      [teacher] that score would describe the harness rather than the model, so the "
            "teacher is UNMEASURED: synthetic data is refused and the accuracy goal has no "
            "calibrated reference.")
        verdict["reason"] = reason
        return _apply_operator_overrides(verdict, spec, log=log)
    if failures:
        common = Counter(failures).most_common(3)
        log(f"      [teacher] {len(failures)} of {len(base_prompts)} row(s) failed to generate and "
            f"scored as empty; most common: "
            + "; ".join(f"{message} (x{count})" for message, count in common))

    score = float(result.get("f1", 0.0))
    format_valid = float(result.get("format_valid", 1.0))
    verdict.update({
        "status": "measured",
        "score": score,
        "format_valid": format_valid,
        "n": len(base_prompts),
        "shots": used_shots,
        "synthesis_allowed": score >= MIN_ACCURACY,
        "metric": result.get("metric", spec.metric_name),
    })
    # FORMAT VALIDITY BESIDE THE SCORE, ALWAYS. They fail differently and are fixed differently: a
    # low score with format_valid near 1.0 is a capability limit, while a low format_valid is a
    # prompt or output-contract problem that caps the score no matter how capable the model is.
    # Reporting only the first is what let B290 read as "fine-tuning does not help this task".
    log(f"      [teacher]   {used_shots}-shot on {len(base_prompts)} row(s): "
        f"{verdict['metric']}={score:.4f}  format_valid={format_valid:.4f}")
    if format_valid < 0.95:
        log(f"      [teacher]   ⚠ format_valid {format_valid:.4f} — {1 - format_valid:.0%} of the "
            f"teacher's replies did not parse as this task's output contract, so its "
            f"{verdict['metric']} is bounded by a formatting failure rather than by competence. "
            f"Both the synthesis gate and the run's accuracy goal are being set from this number.")
    log_fitness(verdict, spec, log=log)
    return _apply_operator_overrides(verdict, spec, log=log)


def format_fitness_measurement(verdict: dict | None) -> str:
    """One line naming WHAT was measured, HOW, and how well — including format validity.

    Shared by the fitness log, the accuracy-goal provenance and the end-of-run report so all three
    describe the measurement identically. They previously each rendered their own subset of these
    fields, which is how a run could report a `threshold 0.8000` beside a teacher that had scored
    0.0999 without the two lines ever contradicting each other on the page.
    """
    if not isinstance(verdict, dict) or verdict.get("status") != "measured":
        reason = (verdict or {}).get("reason") or "no reason recorded"
        return f"UNMEASURED ({reason})"
    model = verdict.get("model") or "teacher"
    score = float(verdict.get("score") or 0.0)
    format_valid = verdict.get("format_valid")
    format_text = (
        f"format_valid={float(format_valid):.4f}"
        if isinstance(format_valid, (int, float))
        else "format_valid=unmeasured"
    )
    return (
        f"{model} {verdict.get('shots', 0)}-shot {verdict.get('metric', 'score')}={score:.4f} "
        f"{format_text} on {verdict.get('n', 0)} eval row(s)"
    )


def log_fitness(verdict: dict, spec, *, log=print) -> None:
    """State the verdict and its consequence in the same place, so neither can be read alone."""
    if verdict.get("status") != "measured":
        log(f"      [teacher] {spec.name}: fitness UNMEASURED "
            f"({verdict.get('reason', 'no reason recorded')}) — synthetic data REFUSED")
        return
    score = verdict["score"]
    allowed = verdict["synthesis_allowed"]
    log(f"      [teacher] {spec.name}: {format_fitness_measurement(verdict)}, "
        f"against a {verdict['threshold']:.2f} gate")
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

    `SLM_SYNTH_DISALLOW` is checked FIRST, ahead of both the verdict and `BYPASS`, because it is an
    operator refusal rather than a measurement: the ablation it serves asks what the pipeline
    achieves with no synthetic data at all, and a teacher that happens to clear the 0.80 gate must
    not be able to answer that question with synthesized rows. It is deliberately the only input
    that can force the answer to False — the gate itself is still measured and still reported, so
    the run says what the teacher COULD have done alongside the fact that it was not asked to.

    `BYPASS` is honoured here as well as on the verdict, so the override holds even on a run that
    never reached the gate at all — a resumed checkpoint written before the flag was set, for one.
    Reading the flag in both places means the answer cannot depend on which of them ran.
    """
    from agent.ablations import synthesis_disallowed

    if synthesis_disallowed():
        return False
    verdict = state.get("teacher_fitness")
    if not isinstance(verdict, dict):
        return BYPASS
    return bool(verdict.get("synthesis_allowed")) or BYPASS
