"""Proactive listening (LlamaPIE) — at this pause, whisper a hint or stay silent."""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import classification as scorer
from tasks._builders import classification_turn
from tasks.spec import MiningSource, TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.proactive_listening import load_proactive_listening

    return load_proactive_listening(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="proactive_listening",
    title="Proactive listening (LlamaPIE interrupt/wait)",
    category="out_of_distribution",
    family="classification",

    load=_load,
    required_fields=("text", "label"),
    initial_train_cap=5000,
    eval_cap=1000,
    eval_sampling="label_balanced",
    closed_label_space=True,
    label_definitions={
        "interrupt": (
            "at the pause ending this transcript, the user is about to need a specific detail "
            "they may not recall, or clearly needs help continuing — so a 1-3 word hint should be "
            "whispered NOW. Judge the conversational moment, NOT whether the topic sounds urgent"
        ),
        "wait": (
            "the assistant should stay silent at this pause — the user is mid-flow, does not need "
            "anything, or a hint would be intrusive. Most pauses are this. Judge the moment, NOT "
            "whether the topic is calm or uninteresting"
        ),
    },
    verifier_notes="",
    quality_controls=(
        qc.require_fields("text", "label"),
        qc.label_space(),
        # NOT label-balanced. `wait` is the overwhelming majority by construction — most pauses
        # are not interruption points — and forcing a 3:1 cap would train the model on a base
        # rate the eval set does not have. The minority-class F1 below is what keeps a
        # majority-always model from scoring well.
        qc.length_outliers(key="text"),
        qc.dedup_surface(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score_minority_f1,
    metric_name="minority_f1",
    max_new_tokens=50,
    # Transcripts are long: the decision depends on the whole conversation up to the pause, so
    # this task gets more context than the other two classification tasks.
    max_seq_length=2048,
    eval_batch_size=32,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    build_training_turn=classification_turn,

    # The generated row inherits a real anchor's label, so the target cannot be wrong.
    synth_verifier=None,
    cot_annotation=False,

    # The vendored bundle holds 14,130 train rows — far more than the initial load takes — so
    # re-reading it with a bigger slice is the cheapest source of new real rows. (An earlier version
    # of this file declared no sources, on the mistaken assumption the pool was fully consumed.)
    mining_sources=(
        MiningSource(
            hf_id="local:proactive_listening",
            config=None,
            split="train",
            url="https://github.com/kwentar/LlamaPIE",
            supports_offset=True,
        ),
    ),
    allow_paid_discovery=True,

    model_ranking_metric="MMLU",
)
