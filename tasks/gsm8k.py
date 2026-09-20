"""GSM8K — grade-school math word problems with a checkable numeric answer.

Promoted to a first-class curated task on 2026-08-18. It was the only one of the eight that still
ran through the autonomous Exa/web-acquire path: its rows were built by an inline converter buried
in `web_acquire.load_benchmark_dataset`, it had no entry in the task registry, and it carried no
`_instruction`, so it silently inherited the family default "Answer the following question:" —
which happens to be right for GSM8K and was wrong for every other task that inherited it.
"""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import generation as scorer
from tasks._builders import generation_turn
from tasks.spec import MiningSource, TaskSpec

INSTRUCTION = (
    "Solve the following grade-school math problem. Show your working, then give the final "
    "numeric answer on its own line after ####."
)


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.gsm8k import load_gsm8k

    return load_gsm8k(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="gsm8k",
    title="GSM8K (grade-school math)",
    category="in_distribution",
    family="generation",

    load=_load,
    required_fields=("text", "answer"),
    initial_train_cap=5000,
    select_cap=1000,
    eval_sampling="shuffled",
    closed_label_space=False,
    label_definitions={},
    entity_type_vocabulary=(),  # extracts no spans
    verifier_notes="",
    quality_controls=(
        qc.require_fields("text", "answer"),
        # `key="text"` matters: the old generation-family branch filtered on `"prompt"`, a field
        # GSM8K rows have never carried, so both of these were no-ops and a 100,000-character row
        # would have survived (B299).
        qc.length_outliers(key="text"),
        qc.dedup_surface(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score_exact_match,
    metric_name="exact_match",
    max_new_tokens=512,
    max_seq_length=2048,
    eval_batch_size=16,
    failure_category=scorer.exact_match_failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # Selection and reporting coincide for this task: exact match on a checkable numeric answer is
    # the published metric AND a sound ranking signal, with no threshold or class-support problem
    # for it to hide. The report pass only widens the split.
    report_load=None,
    report_score=scorer.score_exact_match,
    report_metric_name="exact_match",

    build_training_turn=generation_turn,

    # No free exact check: deciding whether a generated word problem's stated answer is correct
    # requires solving it, which is the task itself. The teacher pass is the only gate, and it is
    # weaker than the programmatic checks the format-bound tasks get.
    synth_verifier=None,
    # GSM8K ships gold chain-of-thought in its `####` split; `annotate_cot` preserves an existing
    # `cot_reasoning` rather than regenerating, so this only fills rows that lack one.
    cot_annotation=True,

    mining_sources=(
        MiningSource(
            hf_id="openai/gsm8k",
            config="main",
            split="train",
            url="https://huggingface.co/datasets/openai/gsm8k",
            supports_offset=True,
        ),
    ),
    allow_paid_discovery=True,

    model_ranking_metric="GSM8K",
)
