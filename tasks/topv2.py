"""TOPv2 — assistant command to nested intent/slot parse, under the low-resource protocol.

WHY THIS TASK IS IN THE SUITE
    It is the canonical on-device NLU task: a personal command should not leave the phone, and a
    cloud round-trip blows the wake-word latency budget outright. It is also the only task here
    with existing META-LEARNING baselines rather than only vanilla SFT numbers, so a data-curation
    method has something to be compared against other than itself.

    And it is where seed variance is worst, which is the reason to have it. A production-sized
    TOPv2 baseline churns on ~21% of examples across retrains with identical data and identical
    hyperparameters (EM 83.74 / EM@10 73.18 / AGR 78.47, arXiv:2204.04735). This task trains on
    501 and 177 rows. `scripts/report_eval.py` defaults to five seeds here and three elsewhere for
    exactly that reason.

WHAT THE SCORE IS
    The mean of exact match on the two target domains, with both components always reported. The
    two are deliberately unalike — `weather` is flat and will saturate, `reminder` is 21.5%
    compositional and will not — so the mean is a good loop signal (pressure shifts to the hard
    domain as the easy one tops out) and a bad headline on its own.

THE SPIS SPLITS ARE RECONSTRUCTED. The official low-resource files are not in the public mirror
and no mirror of them exists; `data/loaders/topv2.py` reimplements the sampling rule from its
definition and lands within 1.6% of the released 25-SPIS sizes. Our EM is comparable between our
own runs, not against published RINE or shift-reduce numbers — which was already true because EM
depends on the parse serialization.
"""
from __future__ import annotations

from data import quality_controls as qc
from data.loaders.topv2 import INTENTS, SLOTS
from eval.scorers import semantic_parse as scorer
from tasks._builders import semantic_parse_turn
from tasks.spec import MiningSource, TaskSpec


def _check(row: dict) -> tuple[bool, str]:
    """The verdict AND the reason. Exposed as `_verify.checker` below."""
    from data.synth_verifiers import verify_semantic_parse_row

    return verify_semantic_parse_row(row)


def _verify(row: dict) -> bool:
    return _check(row)[0]


_verify.checker = _check


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.topv2 import load_topv2

    return load_topv2(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="topv2",
    title="TOPv2 (compositional semantic parsing, low-resource)",
    category="format_bound",
    family="structured_output",

    load=_load,
    # `domain` is required, not incidental: the scorer's headline is the mean of the per-domain
    # exact-match rates, so a row without a domain cannot be scored into the number that matters.
    required_fields=("text", "answer", "domain"),
    initial_train_cap=5000,
    select_cap=1000,
    # `shuffled`, not `label_balanced`. The two target test splits are 5,767 and 5,682, so a
    # shuffled 1,000-row draw is close to even by itself, and `label_balanced` reads `label` —
    # a field these rows do not carry, because the target here is a parse and not a class.
    eval_sampling="shuffled",
    closed_label_space=False,
    label_definitions={},
    entity_type_vocabulary=(),  # extracts no spans
    # The teacher has to be told the extractive constraint, because it is the one property a
    # fluent generated parse will violate. Left unstated, a teacher paraphrases the command into
    # the slots — "set an alarm" for "wake me up" — which is well-formed, plausible, and teaches
    # the model to invent span text at eval time, where exact match scores it zero.
    verifier_notes=(
        "You are shown a short assistant command and, as the proposed answer, ONLY its parse "
        "tree: one balanced expression rooted at an intent. That is the expected shape; it is not "
        "a malformed row and it is not missing a wrapper object.\n"
        "NESTED INTENTS ARE LEGITIMATE AND COMMON. A slot may contain a whole intent, and 21.5% "
        "of this corpus's `reminder` parses contain more than one intent. The gold itself looks "
        "like this:\n"
        "  [IN:CREATE_REMINDER remind [SL:PERSON_REMINDED me ] to [SL:TODO take my meds ] "
        "[SL:RECURRING_DATE_TIME [IN:GET_RECURRING_DATE_TIME [SL:DATE_TIME at 8 am ] "
        "[SL:FREQUENCY daily ] ] ] ]\n"
        "So NEVER reject a row for nesting an intent inside a slot, and never claim slots must be "
        "flat. That reasoning is a measured false positive: it accounted for 3 of 10 sampled "
        "rejections on the 2026-09-09 synthesis audit, every one of them on a correctly nested "
        "row.\n"
        "The following are ALREADY verified by computation before you see the row, so never "
        "reject for them: the brackets balance, the answer is exactly one tree rooted at an "
        "intent, every word inside the tree appears in the command verbatim and in order, and "
        "EVERY LABEL IS A REAL TOPv2 LABEL drawn from the closed vocabulary of 82 intents and 84 "
        "slots listed at the end of these notes. Never reject a row because a label looks "
        "unfamiliar — if it reached you, it is in the vocabulary. On the 2026-09-09 synthesis "
        "audit the teacher rejected SL:DATE_TIME_NEW as invalid, and it is a real slot that "
        "appears in gold.\n"
        "TOPv2 parses are EXTRACTIVE. Slot contents are copied from the command, never "
        "paraphrased, reordered or reworded, and words the command does not contain must not "
        "appear.\n"
        "CREATE VERSUS UPDATE is the single most confused distinction on this task, so state it "
        "plainly: CREATE_REMINDER makes a NEW reminder, and the UPDATE_REMINDER* intents modify "
        "one that ALREADY EXISTS. Measured over the reminder domain of this corpus, a command "
        "opening with `remind` is CREATE in 103 of 105 gold rows and one opening with `set` in 29 "
        "of 29; `change` is UPDATE in 35 of 35, `update` in 18 of 18, and `add` in 26 of 28, "
        "because `add to reminder` extends a reminder already there. So `set a reminder for 5pm` "
        "is CREATE_REMINDER, not UPDATE_REMINDER. On the 2026-09-09 synthesis audit that exact "
        "confusion was the largest single source of bad rows.\n"
        "Judge ONLY what computation cannot: whether the chosen intent is the right one for this "
        "command, whether a slot that should have been filled was left out, and whether each span "
        "is labelled with the slot it actually is.\n"
        "THE CLOSED VOCABULARY. Intents: " + ", ".join(INTENTS) + ".\n"
        "Slots: " + ", ".join(SLOTS) + "."
    ),
    quality_controls=(
        qc.require_fields("text", "answer", "domain"),
        # No `dedup_surface`. Assistant commands are short and formulaic by nature — "set an
        # alarm for 6am" and "set an alarm for 7am" are near-identical in trigram Jaccard and are
        # genuinely different parses — so a surface-similarity filter here removes exactly the
        # systematic variation the parser has to learn.
        qc.length_outliers(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="exact_match",
    # Parses nest: reminder's deepest train parse reaches bracket depth 8. 512 tokens is ample for
    # the longest of them and leaves the 2,048-token context almost entirely to the prompt.
    max_new_tokens=512,
    max_seq_length=2048,
    eval_batch_size=16,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # Selection and reporting coincide: exact match is the published metric and a sound ranking
    # signal, with no threshold and no per-class averaging to destabilize at selection size. The
    # report pass widens the eval from 1,000 rows to the full 11,449 and adds seeds, which is
    # where this task actually needs the help.
    report_load=None,
    report_score=scorer.score,
    report_metric_name="exact_match",

    build_training_turn=semantic_parse_turn,

    # Verifiable, which is what makes synthesis safe enough to allow here: a generated parse's
    # brackets, labels and extractive spans are all decidable by computation. What it cannot check
    # is whether the intent is the correct one, so the teacher pass still runs.
    synth_verifier=_verify,
    # No chain-of-thought. A parse is a structured transcription of the command, not a conclusion
    # reached by steps, and a `<reasoning>` block prepended to the target would become part of the
    # string exact match compares.
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="WillHeld/top_v2",
            config=None,
            split="train",
            url="https://huggingface.co/datasets/WillHeld/top_v2",
            supports_offset=True,
        ),
    ),
    # The mirror holds 83,703 unused source-domain rows behind the 5,000-row cap, so mining has
    # somewhere real to go and paid discovery would only rediscover mirrors of what is already
    # cached. Off for that reason, not as a cost measure.
    allow_paid_discovery=False,

    model_ranking_metric=None,
)
