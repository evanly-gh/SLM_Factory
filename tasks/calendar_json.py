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


def _verify(row: dict) -> bool:
    from data.synth_verifiers import verify_calendar_row

    return verify_calendar_row(row)[0]


SPEC = TaskSpec(
    name="calendar_json",
    title="Calendar NL→JSON (TOPv2 reminder / SGD Calendar_1)",
    category="format_bound",
    family="structured_output",

    load=_load,
    required_fields=("text", "answer"),
    initial_train_cap=5000,
    eval_cap=1000,
    eval_sampling="shuffled",
    closed_label_space=False,
    label_definitions={},
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
