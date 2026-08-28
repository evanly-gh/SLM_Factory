"""DialogSum + SAMSum — summarise a conversation in one to three sentences."""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import generation as scorer
from tasks._builders import generation_turn
from tasks.spec import MiningSource, TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.dialogsum_samsum import load_dialogsum_samsum

    return load_dialogsum_samsum(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="dialogsum",
    title="DialogSum + SAMSum",
    category="in_distribution",
    family="generation",

    load=_load,
    required_fields=("text", "answer"),
    initial_train_cap=5000,
    eval_cap=1000,
    eval_sampling="shuffled",
    closed_label_space=False,
    label_definitions={},
    verifier_notes="",
    quality_controls=(
        qc.require_fields("text", "answer"),
        qc.length_outliers(key="text"),
        # NOT deduplicated. Chat transcripts share a great deal of surface form — greetings,
        # scheduling small talk — so a trigram-Jaccard filter removes genuinely distinct
        # conversations. The old branch reached the same conclusion but expressed it as
        # "generation is diverse enough to skip dedup", which was a statement about a channel
        # rather than about this corpus.
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score_with_judge,
    metric_name="judge_mean_0_1",
    max_new_tokens=512,
    max_seq_length=2048,
    eval_batch_size=16,
    failure_category=scorer.judge_failure_category_of,
    # The only task that scores through the LLM judge. A judge outage must fail the run loudly
    # rather than record a baseline of zero and let the loop chase a phantom regression.
    needs_judge=True,
    judge_overlap=True,
    attach_reasoning=True,

    build_training_turn=generation_turn,

    # Summary quality is not decidable by computation, so the teacher pass is the only gate.
    synth_verifier=None,
    # No chain-of-thought. A summary is not reached by reasoning steps, and prepending a
    # `<reasoning>` block to the target teaches the model to emit text the judge then scores as
    # part of the summary.
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="knkarthick/dialogsum",
            config=None,
            split="train",
            url="https://huggingface.co/datasets/knkarthick/dialogsum",
            supports_offset=True,
        ),
        MiningSource(
            hf_id="Samsung/samsum",
            config=None,
            split="train",
            url="https://huggingface.co/datasets/Samsung/samsum",
            supports_offset=True,
        ),
    ),
    allow_paid_discovery=True,

    model_ranking_metric=None,
)
