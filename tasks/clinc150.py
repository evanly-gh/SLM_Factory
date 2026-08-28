"""CLINC150 — 151-way intent classification, including an out-of-scope class."""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import classification as scorer
from tasks._builders import classification_turn
from tasks.spec import MiningSource, TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.clinc150 import load_clinc150

    return load_clinc150(max_train=max_train, max_test=max_test)


SPEC = TaskSpec(
    name="clinc150",
    title="CLINC150 (clinc_oos/plus)",
    category="out_of_distribution",
    family="classification",

    load=_load,
    required_fields=("text", "label"),
    initial_train_cap=5000,
    eval_cap=1000,
    # 151 classes against an 800-row eval set: without round-robin sampling some classes would be
    # absent entirely and macro-F1 would average over whichever ones happened to be drawn.
    eval_sampling="label_balanced",
    closed_label_space=True,
    # No definitions, deliberately. CLINC150's labels ARE plain descriptions of the utterance's
    # intent (`accept_reservations`, `transfer`, `oos`), so naming the label already tells the
    # teacher what the class means; a gloss for 151 classes would be prompt noise.
    label_definitions={},
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
    # Genuinely multi-class, so macro-F1 across all 151 classes is the honest headline.
    score=scorer.score_macro_f1,
    metric_name="macro_f1",
    max_new_tokens=50,
    max_seq_length=1024,
    eval_batch_size=32,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    build_training_turn=classification_turn,

    # The generated row inherits a real anchor's intent, so the target cannot be wrong.
    synth_verifier=None,
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="clinc/clinc_oos",
            config="plus",
            split="train",
            url="https://huggingface.co/datasets/clinc/clinc_oos",
            supports_offset=True,
        ),
    ),
    allow_paid_discovery=True,

    model_ranking_metric="MMLU",
)
