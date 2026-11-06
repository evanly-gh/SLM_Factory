"""Generate the nine `single_model` ablation launchers from their baselines (2026-10-06).

THE ARM. `ner_bc5cdr` already has one (`run_ner_bc5cdr_ablation_singlemodel_l40s.slurm`, job
40831542). This extends it to the other nine tasks in the ten-task suite, so the question the NER
arm answered on one task — *does one informed model choice get to the ladder's answer faster than
climbing?* — has an answer across the suite.

WHY GENERATED RATHER THAN HAND-WRITTEN. The arm's validity rests on each ablation differing from
its baseline in EXACTLY ONE WAY: the model-selection strategy. Nine hand-copied files is nine
chances to drop an `SLM_VERIFY_SYNTH=shadow` and silently turn an ablation into a different
experiment — which is the failure `tests/test_ablations.py`'s docstring calls the one that wastes
a multi-day L40S job. This reads each baseline and re-emits it with two lines added, so the diff
is mechanical and reviewable.

THE TWO LINES:
  * `SLM_MODEL_SELECTION_STRATEGY=single_model` — the orchestrator sees all 27 feasible variants
    at once, picks one, and the run stays on it. No escalation on failure, no downward probe on
    success. It ends when that model meets the goal, stagnates, or hits the 30-eval cap.
  * `unset SLM_SYNTH_API_MODE` — sbatch forwards the submitting shell's environment, which is how
    run 39361648 silently became a DeepSeek run. This pins the local Qwen teacher the baselines
    used.

THE GOAL IS LEFT CALIBRATED, deliberately, exactly as the NER arm left it. It appears in the
selection prompt, so pinning it at 0.99 to force a run-to-stagnation would push the orchestrator
toward the largest model and turn this into a different experiment.

Re-running this script is safe: it rewrites the nine files from their current baselines.
"""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent / "tests" / "pipeline"

# THE WALL CLOCK IS NOT THE BASELINE'S, AND THAT IS DELIBERATE.
#
# Every weeklong launcher asks for `--time=7-00:00:00`, which is right for a run that climbs five
# tiers. A `single_model` arm cannot: it is pinned to one variant and bounded by the 30-eval cap,
# so its worst case is hours, not days. Carrying the 7-day request over costs twice.
#
#   * IT CAN BLOCK THE JOB OUTRIGHT. Submitted 2026-10-06, all nine sat `ReqNodeNotAvail,
#     Reserved_for_maintenance`: the cluster holds an `October13_Maintenance` reservation over
#     every g31xx L40S node from 2026-10-13 09:00, and SLURM will not start a job whose wall clock
#     runs past a reservation it would collide with. Nine jobs of a few hours each were unstartable
#     for a week because of a time limit none of them needed.
#   * IT COSTS BACKFILL. A short job slots into gaps a seven-day request cannot, which on a
#     10-GPU account shared by a dozen users is most of how it ever starts.
#
# `wall_hours` below is roughly 2x the 30-eval worst case in the header — enough that a slow
# iteration or a requeue does not truncate the run, short enough to schedule. The USR1
# checkpoint-and-requeue contract still applies and still fires two hours before the wall, so an
# overrun resumes rather than dying.
#
# task -> (baseline log this arm is read against, its best score + tier, min/iter from the 09-16
# note's IV.3 table, the wall-clock request in hours, and what makes this task's pick interesting).
TASKS = {
    "clinc150": (
        "slm-clinc150-ablation-nostop-l40s-40105479 (+ -40214984 resume)",
        "0.9372 macro_f1 at T5 Qwen3-4B-Instruct-2507@Q8_0",
        25,
        36,
        "The baseline is a no-convergence rerun with the goal pinned at 0.99, so it climbed all "
        "five tiers rather than stopping where a calibrated goal would have. This arm's goal IS "
        "calibrated, so a faster finish here is partly that difference and not purely the "
        "selector — the one confound in this task's row, stated rather than removed. clinc150 is "
        "also the suite's in-distribution control (teacher 0.89, fine-tuning adds +0.003), so a "
        "one-shot pick has unusually little to gain by going large.",
    ),
    "sms_spam": (
        "slm-sms-spam-ablation-nostop-l40s-40105478",
        "0.9609 minority_f1 at T3 Qwen3-1.7B@Q4_K_M",
        6,
        12,
        "The cheapest task in the suite per iteration and the only one whose score AND search gain "
        "peak at the same tier. Same pinned-goal confound as clinc150. Minority-class F1 means a "
        "pick too small to separate the classes reports 0.0 rather than a flattering 0.87, so a "
        "bad choice here is unambiguous.",
    ),
    "calendar_json": (
        "slm-calendar-json-l40s-39294409",
        "0.8748 ast_arg_match at T3 Qwen3.5-2B@Q4_K_M",
        12,
        18,
        "One of two tasks whose ladder stopped at three tiers, so the orchestrator's pick has a "
        "well-identified target. Its highest-gain tier was T1 and the gain was ENTIRELY mining — "
        "the one task in the suite where that is true — so a one-shot run that skips the small "
        "tiers also skips where its baseline's data lever actually paid.",
    ),
    "xlam_bfcl": (
        "slm-xlam-bfcl-l40s-39361648",
        "0.8740 ast_arg_match at T5 Qwen3-4B-Instruct-2507@Q8_0",
        27,
        36,
        "The baseline used the DeepSeek teacher and converged against a 0.87 goal, so it is a "
        "reference point rather than a matched control — the same caveat the 09-16 note attaches "
        "to it in Part III. This arm uses the local Qwen teacher, so teacher and selector both "
        "differ and the comparison is weaker than the other eight.",
    ),
    "goemotions": (
        "slm-goemotions-ckpt-39708679",
        "0.5979 ekman_macro_f1 at T5 Qwen3-4B-Instruct-2507@Q8_0",
        9.5,
        16,
        "Fine-tuning gain of +0.0000 over the first fine-tune: the baseline's search bought "
        "nothing at the tier that scored best. If one informed pick reaches 0.5979 directly, this "
        "is the task where the ladder is most clearly unnecessary.",
    ),
    "gec_bea19": (
        "slm-gec-bea19-l40s-39881529",
        "0.5647 errant_f05 at T5 Qwen3-4B-Instruct-2507@Q8_0",
        16,
        24,
        "The one task whose highest-gain tier split its gain three ways across hyperparameters, "
        "mining and synthesis. A single-model run keeps all three levers but loses the tier "
        "changes between them.",
    ),
    "multiconer": (
        "slm-multiconer-l40s-39881531",
        "0.6388 micro_f1 at T5 Qwen3-4B-Instruct-2507@Q8_0",
        17,
        24,
        "33 classes, and the only task where selection and reporting metrics genuinely differ "
        "(micro selects, macro reports). A pick too small to hold 33 classes should show up as a "
        "micro/macro gap rather than as a flat score.",
    ),
    "dialogsum": (
        "slm-dialogsum-l40s-39969104",
        "0.4817 rouge_l at T5 Qwen3-4B-Instruct-2507@Q8_0",
        20,
        30,
        "The suite's other +0.0000 task: 15 iterations of search moved nothing, and the 09-16 "
        "note's honest reading is 'ship the base model'. A one-shot arm that lands on the same "
        "score in a fraction of the time is the cleanest possible statement of that finding.",
    ),
    "topv2": (
        "slm-topv2-l40s-39969102",
        "0.5990 exact_match at T5 Qwen3-4B-Instruct-2507@Q8_0",
        34,
        48,
        "The most expensive task per iteration and the largest synthesis-driven gain in the suite "
        "(+0.1930 at T1). Exact match on a nested parse is unforgiving, so a pick that cannot "
        "hold the bracket grammar will sit near zero rather than degrade gracefully.",
    ),
}

HEADER = """#
# ============================================================================================
# ABLATION — ONE SHOT AT THE WHOLE POOL (orchestrator picks once, no ladder)
#
# Added 2026-10-06, extending `run_ner_bc5cdr_ablation_singlemodel_l40s.slurm` (job 40831542) to
# the rest of the ten-task suite. GENERATED by `scripts/gen_singlemodel_ablations.py` from
# `{baseline_script}` — every line below the header is that file's, unchanged, plus the two
# exports at the bottom. Do not hand-edit; change the generator and re-run it.
#
# BASELINE: {baseline_log}
#   best: {baseline_best}
#
# Same task, same data, same teacher and same goal — but instead of starting in the smallest tier
# and climbing, the orchestrator sees ALL 27 feasible variants at once, picks one, and the run
# stays on it: no escalation on failure, no downward probe on success. It ends when that model
# meets the goal, stagnates, or hits the 30-eval cap.
#
# The pick is `orchestrator_choice_node` verbatim (`single_model` delegates to it): candidates
# listed smallest first with their tier, size and published metrics, and an instruction to take
# the most resource-efficient variant that can plausibly reach the goal after LoRA fine-tuning.
# The chosen selector and the orchestrator's one-sentence reason are printed at the top of the log.
#
# WHAT IT ANSWERS: does one informed choice get to the baseline's answer faster than climbing?
# If the orchestrator picks the baseline's winning tier it skips every iteration below it; if it
# picks smaller it may never reach the baseline's best; if it picks larger it pays in on-device
# size for accuracy the task did not need. All three outcomes are the finding.
#
{note}
#
# THE GOAL IS LEFT CALIBRATED, deliberately. It appears in the selection prompt, so pinning it at
# 0.99 to force a run-to-stagnation would push the orchestrator toward the largest model and turn
# this into a different experiment. The orchestrator may raise it mid-run, as the ner baseline's
# did (0.80 -> 0.85), so both runs stop under the same rule.
#
# EXPECTED COST: ~{per_iter:g} min per train+eval iteration (09-16 note, IV.3), and the ner arm
# took 9 iterations. Budget {central:.0f}-{worst:.0f}h including ~50 min of fixed setup (vLLM boot,
# eval_setup, the four contamination firewalls, difficulty labelling, teacher measurement). The
# upper figure is the 30-eval cap. That per-iteration number is a LADDER AVERAGE across every tier
# the baseline visited; a single-model run sits on one variant, so expect +/-40% depending on how
# large the orchestrator picks.
#
# WALL CLOCK: {wall_hours}h, NOT the baseline's seven days. This arm is pinned to one variant and
# bounded by the 30-eval cap, so it cannot run for days; and a seven-day request is unschedulable
# ahead of the cluster's October13_Maintenance reservation and gets no backfill on a 10-GPU account
# shared by a dozen users. {wall_hours}h is ~2x the worst case above. The USR1
# checkpoint-and-requeue contract still fires two hours before the wall, so an overrun resumes
# rather than dying.
#
# `unset SLM_SYNTH_API_MODE` keeps the local Qwen teacher the baseline used: sbatch forwards the
# submitting shell's environment, which is how run 39361648 silently became a DeepSeek run.
# ============================================================================================"""

ABLATION_EXPORTS = """
export SLM_MODEL_SELECTION_STRATEGY=single_model
unset SLM_SYNTH_API_MODE
"""


def build(task: str) -> tuple[Path, str]:
    baseline_script = f"run_{task}_l40s.slurm"
    baseline = (PIPELINE / baseline_script).read_text()
    baseline_log, baseline_best, per_iter, wall_hours, note = TASKS[task]

    job = f"slm-{task.replace('_', '-')}-ablation-singlemodel-l40s"
    text = baseline

    # Rename the job and its log, so the ablation never writes over its baseline's output.
    text = re.sub(r"^#SBATCH --job-name=.*$", f"#SBATCH --job-name={job}", text, flags=re.M)
    text = re.sub(r"^#SBATCH --output=.*$",
                  "#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/"
                  f"logs/slurm/{job}-%j.out", text, flags=re.M)
    # Cut the baseline's seven-day request down to this arm's actual envelope — see the
    # `wall_hours` note on TASKS. Left at 7 days, every one of these is unschedulable ahead of the
    # October13_Maintenance reservation and gets no backfill besides.
    text = re.sub(r"^#SBATCH --time=.*$",
                  f"#SBATCH --time={wall_hours}:00:00", text, flags=re.M)

    header = HEADER.format(
        baseline_script=baseline_script,
        baseline_log=baseline_log,
        baseline_best=baseline_best,
        # Wrapped to the launchers' ~100-column comment convention, with every line commented.
        note="\n".join(
            f"# {line}" for line in textwrap.wrap(
                f"ON THIS TASK SPECIFICALLY: {note}", width=96)
        ),
        per_iter=per_iter,
        wall_hours=wall_hours,
        central=(50 + 18 * per_iter) / 60,
        worst=(50 + 30 * per_iter) / 60,
    )

    # Splice the header in after the #SBATCH block, replacing the baseline's leading bare `#`.
    lines = text.splitlines()
    last_sbatch = max(i for i, line in enumerate(lines) if line.startswith("#SBATCH"))
    rest = lines[last_sbatch + 1:]
    if rest and rest[0].strip() == "#":
        rest = rest[1:]
    text = "\n".join(lines[: last_sbatch + 1]) + "\n" + header + "\n" + "\n".join(rest)

    # The two exports go immediately BEFORE the body source line, so they are in scope for it.
    text = text.replace(
        "source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/"
        "_l40s_task_body.sh",
        ABLATION_EXPORTS.strip()
        + "\nsource /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/"
        "_l40s_task_body.sh",
    )
    if not text.endswith("\n"):
        text += "\n"
    return PIPELINE / f"run_{task}_ablation_singlemodel_l40s.slurm", text


def main() -> int:
    for task in TASKS:
        path, text = build(task)
        path.write_text(text)
        path.chmod(0o755)
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
