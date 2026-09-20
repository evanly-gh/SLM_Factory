"""RouterBench — decide whether a request can be answered on-device or must be escalated."""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import classification as scorer
from tasks._builders import classification_turn
from tasks.spec import MiningSource, TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.routerbench import load_routerbench

    return load_routerbench(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="routerbench",
    title="RouterBench (escalate or not)",
    category="out_of_distribution",
    family="classification",

    load=_load,
    required_fields=("text", "label"),
    initial_train_cap=5000,
    select_cap=1000,
    eval_sampling="label_balanced",
    closed_label_space=True,
    # Without these the teacher reads the label as an English word: it rejected a grade-school
    # math problem for `local` because "the utterance is a math problem, not a local query",
    # discarding 70% of generated rows for the wrong reason (B267).
    label_definitions={
        "local": (
            "the request is simple enough that a SMALL on-device model can answer it correctly "
            "— judge difficulty for a small model, NOT whether the topic is 'local' in the "
            "everyday sense of nearby/location-based"
        ),
        "route": (
            "the request is hard enough that it should be escalated to a LARGER cloud model "
            "— judge difficulty, NOT whether the topic involves routing or networking"
        ),
    },
    entity_type_vocabulary=(),  # extracts no spans
    verifier_notes="",
    quality_controls=(
        qc.require_fields("text", "label"),
        qc.label_space(),
        qc.balance_labels(max_ratio=3),
        qc.length_outliers(key="text"),
        qc.dedup_surface(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    # Binary. Macro-F1 over two classes flatters a model that always predicts the majority, and
    # the base rate here is roughly 30/70.
    score=scorer.score_minority_f1,
    metric_name="minority_f1",
    max_new_tokens=50,
    max_seq_length=1024,
    eval_batch_size=32,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # Selection and reporting coincide: two classes with label-balanced sampling, so the minority
    # class has support at selection size.
    report_load=None,
    report_score=scorer.score_minority_f1,
    report_metric_name="minority_f1",

    build_training_turn=classification_turn,

    # A generated row inherits a real anchor's label, so the TARGET cannot be wrong; only the
    # phrasing can be. That is what the teacher pass checks, and it is why no programmatic check is
    # needed here.
    synth_verifier=None,
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="withmartian/routerbench",
            config=None,
            split="train",
            url="https://huggingface.co/datasets/withmartian/routerbench",
            supports_offset=True,
        ),
    ),
    # The label is DERIVED from whether a small model answered correctly, so no other corpus on
    # the hub carries it natively. Discovery is still allowed as the last rung of the mining ladder,
    # but the closed label space means any mapped row whose label falls outside {local, route} is
    # dropped per-row — which is what stops a repeat of the four hallucinated classes (B259).
    allow_paid_discovery=True,

    model_ranking_metric="MMLU",
)
