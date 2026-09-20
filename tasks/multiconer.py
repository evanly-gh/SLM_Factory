"""MultiCoNER II (English) — 33 fine-grained entity classes on queries and short sentences.

WHY THIS TASK IS IN THE SUITE
    Entity extraction over what the user typed and what is on screen is what feeds local search,
    smart replies and link-outs. It is latency-bound and privacy-sensitive, and the inputs are the
    shortest in the suite — most rows under 50 words, the majority 10 to 20 — so it is cheap per
    example and the label space, not the sequence length, is the difficulty.

    It is also the suite's clearest evidence that training choices beat scale: the published ladder
    runs RoBERTa-base 0.31 -> XLM-R-Large 0.53 -> 0.61, and roughly 30 of those points come from
    feature, model and loss engineering rather than from a bigger model.

THE TWO METRICS ARE NOT THE SAME NUMBER
    Selection is micro-F1 on the official 871-row dev split; the headline is macro-F1 on a fixed
    stratified 20,000-row slice of the 249,980-row test split. Both halves of that are forced. A
    33-class macro average over 871 rows is noise about a handful of entities, so the loop cannot
    select on it; and the 871 rows cannot support a published number regardless of which metric is
    computed over them. See `data/loaders/multiconer.py` for the slice's seed and hash.

WHAT IS NOT REPORTED, AND WHY
    The clean-versus-corrupted robustness gap. The corrupted test portion is the task's headline
    feature, but the partition is not in the release: all 249,980 test sentence headers carry only
    `# id <uuid>`, with none of the `domain=en` attribute train and dev have. Reconstructing it
    heuristically would mean reporting a gap measured against a partition we invented.
"""
from __future__ import annotations

from data import quality_controls as qc
from data.loaders.multiconer import ENTITY_TYPES  # noqa: F401  (spec vocabulary)
from eval.scorers import fine_ner as scorer
from tasks._builders import fine_ner_turn
from tasks.spec import MiningSource, TaskSpec


def _check(row: dict) -> tuple[bool, str]:
    """The verdict AND the reason. Exposed as `_verify.checker` below."""
    from data.synth_verifiers import verify_fine_ner_row

    return verify_fine_ner_row(row)


def _verify(row: dict) -> bool:
    return _check(row)[0]


_verify.checker = _check


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.multiconer import load_multiconer

    return load_multiconer(max_train=max_train, max_test=max_test, log=log)


def _load_report(log=print):
    from data.loaders.multiconer import load_report_split

    return load_report_split(log=log)


SPEC = TaskSpec(
    name="multiconer",
    title="MultiCoNER II EN (33-class fine-grained NER)",
    category="format_bound",
    family="extraction",

    load=_load,
    required_fields=("text", "entities"),
    initial_train_cap=5000,
    # The dev split is 871 rows, so this cap never actually bites — the loop evaluates the whole
    # official dev split every iteration. Left at the suite default rather than lowered to 871,
    # because the cap is a ceiling and pinning it to today's split size would be a second place to
    # update if the release changed.
    select_cap=1000,
    eval_sampling="shuffled",
    # `entities` is the target and the 33 TYPES are the closed vocabulary, but `closed_label_space`
    # is about a row's `label` field being a class to predict — which is not the shape here. The
    # vocabulary is pinned where it actually matters instead: enumerated in the prompt, and
    # enforced by the scorer, which compares types exactly.
    closed_label_space=False,
    label_definitions={},
    entity_type_vocabulary=ENTITY_TYPES,
    verifier_notes=(
        "You are shown a short sentence or query and, as the proposed answer, ONLY its entity "
        "list — a JSON array of {\"text\", \"type\"} objects. That is the expected shape; it is "
        "not a malformed row and it is not missing a wrapper object.\n"
        "The following are ALREADY verified by computation before you see the row, so never "
        "reject for them: the answer parses, every span appears verbatim in the text, no span is "
        "empty, and no (text, type) pair repeats.\n"
        "Types are compared EXACTLY against the fixed 33-class taxonomy listed below, so a synonym "
        "is a wrong answer: it is OtherPER and not PERSON, ORG and not ORGANIZATION, "
        "Medication/Vaccine and not DRUG. That list is EXHAUSTIVE and every name on it is valid — "
        "never reject a row for using one. Membership is also checked by computation before you "
        "see the row, so a type outside the list will never reach you, and a type you do see is "
        "one you should accept.\n"
        "The text is lowercased throughout this corpus; that is the corpus convention and not an "
        "error.\n"
        "Judge ONLY what computation cannot: whether a real entity was MISSED, whether something "
        "labelled is not an entity, and whether each span carries the RIGHT one of the 33 types."
        ),
    quality_controls=(
        qc.require_fields("text", "entities"),
        # Wikipedia-derived sentences repeat prominent names heavily, and without a cap the
        # curriculum teaches a few hundred surface forms rather than the tagging task. Same
        # reasoning as BC5CDR, where drug names recur constantly.
        qc.entity_diversity(cap=3),
        qc.length_outliers(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="micro_f1",
    # Short inputs, but the prompt carries all 33 type names and a query can hold many entities.
    max_new_tokens=512,
    max_seq_length=2048,
    eval_batch_size=16,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # THE ONE TASK WHERE THE REPORT SPLIT IS A DIFFERENT FILE, not a larger draw from the same
    # one. See the module docstring: the 871-row dev split cannot support a 33-class macro-F1 at
    # any sample size, because it does not contain the mentions.
    report_load=_load_report,
    report_score=scorer.score_report,
    report_metric_name="macro_f1",

    build_training_turn=fine_ner_turn,

    # `verify_fine_ner_row`, NOT the BC5CDR verifier this task originally borrowed. Spans appearing
    # verbatim is the shared half; the difference is the TAXONOMY, and on 2026-09-09 that turned out
    # to be checkable after all — the audit found generated rows typed `OtherORG`, `Org` and
    # `OtherPer`, none of which exist among the 33. The scorer compares types exactly, so those are
    # targets the model can only be marked wrong on. Two types are unguessable; 33 are not.
    synth_verifier=_verify,
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="MultiCoNER/multiconer_v2",
            config=None,
            split="train",
            url="https://huggingface.co/datasets/MultiCoNER/multiconer_v2",
            supports_offset=True,
        ),
    ),
    # 16,778 train rows against a 5,000-row cap leaves ~11,800 unmined, and the 12 other languages
    # sit in the same repo. Paid discovery would only rediscover mirrors of a corpus already in the
    # cache, which is the B297 trap.
    allow_paid_discovery=False,

    model_ranking_metric=None,
)
