"""BC5CDR — chemical and disease spans from biomedical abstracts."""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import ner as scorer
from tasks._builders import ner_turn
from tasks.spec import MiningSource, TaskSpec


def _verify(row: dict) -> bool:
    from data.synth_verifiers import verify_ner_row

    return verify_ner_row(row)[0]


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.ner_bc5cdr import load_ner_bc5cdr

    return load_ner_bc5cdr(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="ner_bc5cdr",
    title="BC5CDR (Chemical/Disease spans)",
    category="format_bound",
    family="extraction",

    load=_load,
    required_fields=("text", "entities"),
    initial_train_cap=5000,
    eval_cap=1000,
    eval_sampling="shuffled",
    # `entities` is the target, not `label`; there is no class vocabulary to pin.
    closed_label_space=False,
    label_definitions={},
    quality_controls=(
        qc.require_fields("text", "entities"),
        # Abstracts repeat the same drug names constantly; without a cap the curriculum teaches
        # a handful of surface forms rather than the span task.
        qc.entity_diversity(cap=3),
        qc.length_outliers(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="span_f1",
    max_new_tokens=512,
    max_seq_length=2048,
    eval_batch_size=16,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    build_training_turn=ner_turn,

    # Span synthesis is checkable, which is what makes it safe enough to allow: every generated
    # span must be a real substring of the generated text at the offset it claims. That catches the
    # dominant teacher error (a plausible entity that does not appear verbatim) for free, before any
    # teacher call. What it cannot catch is a MISSED entity, so the teacher pass still runs.
    synth_verifier=_verify,
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="tner/bc5cdr",
            config=None,
            split="train",
            url="https://huggingface.co/datasets/tner/bc5cdr",
            supports_offset=True,
        ),
    ),
    allow_paid_discovery=True,

    model_ranking_metric=None,
)
