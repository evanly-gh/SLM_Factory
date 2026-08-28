"""Calendar NL→JSON — TOPv2 reminder / SGD Calendar_1 turned into `calendar.events.insert` calls.

Shares its scorer, prompt and training turn with `xlam_bfcl`: both emit a JSON call against a
declared tool, and the argument matcher is the same computation. Everything that differs is stated
below rather than inferred. Under the old `task_type` design both were `function_call`, which is
why the verifier registry had already been forced to key on the benchmark name instead — the one
place the channel abstraction had visibly failed before this refactor.
"""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import function_call as scorer
from tasks._builders import function_call_turn
from tasks.spec import MiningSource, TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.calendar_json import load_calendar_json

    return load_calendar_json(max_train=max_train, max_test=max_test, log=log)


def _check(row: dict) -> tuple[bool, str]:
    """The verdict AND the reason. Exposed as `_verify.checker` below."""
    from data.synth_verifiers import verify_calendar_row

    return verify_calendar_row(row)


def _verify(row: dict) -> bool:
    return _check(row)[0]


# WHY THE REASON IS PUBLISHED SEPARATELY
#     `TaskSpec.synth_verifier` only has to answer yes/no, so this wrapper used to be
#     `return verify_calendar_row(row)[0]` and the reason string was thrown away on the spot. `data.curriculum`
#     looks for a `.checker` attribute to recover it, finds nothing, and its
#     "[verify:exact] programmatic verifier rejected N row(s)" block is then unreachable.
#
#     Run 38985393 is what that costs. Synthesis generated 519 rows, the exact verifier rejected all
#     519, and the log recorded only the total — so which of the five checks fired (unparseable path,
#     undeclared API, bad argument schema, no terminal Finish, over the call budget) had to be
#     reverse-engineered afterwards from the vLLM access log. The information existed at the moment of
#     rejection and was discarded one character from where it was needed.
_verify.checker = _check


SPEC = TaskSpec(
    name="calendar_json",
    title="Calendar NL→JSON (TOPv2 reminder / SGD Calendar_1)",
    category="format_bound",
    family="structured_output",

    load=_load,
    required_fields=("text", "answer"),
    # Deliberately BELOW the pool size, unlike the other tasks. Calendar has ~4,460 usable rows in
    # total (TOPv2 + SGD, deduplicated), so a 5,000 cap consumed every one of them at cold start and
    # `mine_new_real` had nothing left to find on its very first attempt. With synthesis also refused
    # by the fitness gate on this task, `data_rebuild` then could not add a row by either route, every
    # rebuild came back empty, and `run_health` correctly stopped run 38734724 as unable to progress.
    #
    # 3,000 leaves ~900 rows of headroom, which is one or two rebuilds at the 1,000-row ceiling. The
    # cost is a smaller starting curriculum; the gain is that the data intervention exists at all,
    # which matters more on the one task where synthesis is unavailable.
    initial_train_cap=3000,
    eval_cap=1000,
    eval_sampling="shuffled",
    closed_label_space=False,
    label_definitions={},
    # The four conventions `verify_calendar_row` enforces exactly, restated for the TEACHER — which
    # judges the same rows afterwards and, without them, invents its own. See TaskSpec.verifier_notes
    # for the measurement that made this necessary.
    verifier_notes=(
        "This task has FIXED datetime conventions. Judge against these and nothing else:\n"
        "  - Times are 24-hour ISO. 8 pm IS 20:00, 7 pm IS 19:00 — the SAME time written two ways, "
        "never a mismatch.\n"
        "  - An event whose duration the request does not state lasts exactly 60 minutes, so `end` "
        "is `start` plus one hour.\n"
        "  - 'tonight' means 20:00. A bare date with no time means 09:00.\n"
        "  - Dates resolve FORWARD from the 'Current date and time' line in the request: a date "
        "that has already passed this year rolls to NEXT year. A year later than the reference "
        "year is therefore usually CORRECT, not an error.\n"
        "  - `summary` is the event TITLE only, with the imperative and the date/time phrase "
        "removed ('remind me to', 'schedule', 'after work', 'next saturday at 8am')."
    ),
    quality_controls=(
        qc.require_fields("text", "answer", "tools"),
        qc.valid_json_answer(),
        qc.length_outliers(key="text"),
        # NOT deduplicated. Calendar utterances are short and highly templated ("remind me to X
        # at Y"), so a trigram-Jaccard duplicate filter removes legitimately distinct events that
        # differ only in the entity — which is the part the model has to learn to extract.
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="ast_arg_match",
    max_new_tokens=256,
    max_seq_length=2048,
    eval_batch_size=32,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    build_training_turn=function_call_turn,

    # The stricter of the two verifiers: as well as the schema check xlam gets, this one checks
    # that the datetimes parse, that `end` follows `start`, that an unstated duration is the
    # 60-minute gold convention, and that the event resolves near the request's own reference
    # instant. That last check is the one that catches the year-rollover defect which made the
    # task score 0.0000 and look like a model failure.
    synth_verifier=_verify,
    cot_annotation=False,

    # TOPv2's reminder split is far larger than the initial load takes, so re-reading it with a
    # bigger slice is the cheapest source of new real rows. (An earlier version of this file
    # declared no sources at all, on the mistaken assumption the pool was fully consumed.)
    mining_sources=(
        MiningSource(
            hf_id="TOPv2/reminder",
            config=None,
            split="train",
            url="https://github.com/facebookresearch/TOPv2",
            supports_offset=True,
        ),
    ),
    allow_paid_discovery=True,

    model_ranking_metric=None,
)
