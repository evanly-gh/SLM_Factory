"""GoEmotions — multi-label emotion over 27 emotions plus neutral, on short Reddit comments.

WHY THIS TASK IS IN THE SUITE
    On-device tone detection for reply suggestion and notification triage. Reading someone's
    messages in order to classify how they feel is close to the definition of what should not be
    sent to a server.

    It is also the most training-data-sensitive task here, which is the point for a data-curation
    method: the BERT baseline sits at 0.46 macro-F1, data augmentation moves it to 0.52, and a
    clipped asymmetric loss moves it to 0.54. Nothing about that ladder is scale.

TWO METRICS, BECAUSE THE HONEST ONE IS NOT COMPUTABLE IN THE LOOP
    The headline is threshold-free macro AUPRC over the 28 labels. Macro-F1 there is a
    thresholding artifact — the same model moves several points between a fixed 0.5, a fixed 0.3
    and a dev-tuned sweep — and AUPRC cannot be moved by a cutoff.

    But AUPRC needs a RANKING, and generation returns a hard decision. The report pass therefore
    scores all 28 labels per row through `training.slm_helpers.infer_label_scores_batch`, which is
    the one new inference capability this suite required. The loop selects on Ekman-7 macro-F1
    instead, computed from ordinary generation, whose seven classes all have real support.

THE TAIL IS STATISTICALLY EMPTY AND MUST NEVER CARRY A HEADLINE
    Test support: grief 6, relief 11, pride 16, nervousness 23, against neutral 1,787. A published
    result moving `grief` from 0.00 to 0.57 F1 is +2 macro-F1 points earned on six examples. The
    scorer therefore reports per-label F1 in frequency BANDS — head >=300, mid 50-300, tail <50 —
    banded by gold support in the split actually scored.

    Contamination: HIGH. 2020, canonical on the Hub, so a zero-shot baseline here is inflated.
    That compresses the apparent delta rather than exaggerating it, but it should be stated.
"""
from __future__ import annotations

from data import quality_controls as qc
from data.loaders.goemotions import EMOTIONS
from eval.scorers import multilabel_emotion as scorer
from tasks._builders import multilabel_emotion_turn
from tasks.spec import MiningSource, TaskSpec


def _check(row: dict) -> tuple[bool, str]:
    """The verdict AND the reason. Exposed as `_verify.checker` below."""
    from data.synth_verifiers import verify_multilabel_emotion_row

    return verify_multilabel_emotion_row(row)


def _verify(row: dict) -> bool:
    return _check(row)[0]


_verify.checker = _check


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.goemotions import load_goemotions

    return load_goemotions(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="goemotions",
    title="GoEmotions (28-label multi-label emotion)",
    category="in_distribution",
    family="classification",

    load=_load,
    # `label` FIRST, and the order is load-bearing. `data.curriculum._gold_field` takes the first
    # required field that is not an input as the gold to show the synthesis teacher, so this is
    # what the teacher is asked to judge — and it must be the string the model is actually trained
    # to emit (`confusion, neutral`), not the list (`["confusion", "neutral"]`).
    #
    # THE BUG THIS ORDER FIXES, measured in run 39708679. With `labels` first, the teacher was
    # shown `["confusion", "neutral"]` as the proposed answer while the task brief told it the
    # output contract is plain text with no brackets — so it rejected 38-72% of generated rows
    # across rebuilds, with reasons like "Output contains brackets and quotes, violating the plain
    # text contract" (22 of one batch) and "must be plain text, not JSON or list" (8 more). Every
    # one of those rows was CORRECTLY FORMATTED; the teacher was judging the JSON rendering of the
    # field rather than the answer, which is the BC5CDR failure (B267/B269/B314) in a new costume.
    #
    # It was caught rather than paid for because the pass ran in SHADOW mode: nothing was dropped.
    # In enforce mode this would have discarded roughly half of every synthesis batch and looked
    # like the teacher doing its job.
    #
    # `labels` stays required — it is the field the SCORER grades — so nothing has to re-derive
    # the label set at a second site with a second chance to disagree.
    required_fields=("text", "label", "labels"),
    initial_train_cap=5000,
    select_cap=1000,
    # `shuffled`, NOT `label_balanced`. Round-robin sampling reads `label`, which here is a
    # multi-label STRING like "annoyance, disapproval" — so it would balance across label
    # COMBINATIONS, of which there are hundreds, and produce a draw whose emotion distribution
    # resembles nothing. The tail is unfixable by sampling anyway: `grief` has 6 test rows in
    # total, so no draw can give it usable support, which is exactly why the headline is AUPRC and
    # the diagnostics are banded.
    eval_sampling="shuffled",
    # The 28 emotions ARE a closed vocabulary and it is pinned in the loader rather than read off
    # the eval set: `qc_context_labels` collects distinct `label` VALUES, which for a multi-label
    # task are combinations, not classes. Declaring this closed would hand quality control and the
    # teacher a vocabulary of hundreds of comma-joined strings.
    closed_label_space=False,
    label_definitions={},
    entity_type_vocabulary=(),  # extracts no spans
    verifier_notes=(
        "You are shown a short Reddit comment and, as the proposed answer, ONLY its emotion "
        "labels — one or more of the 28 names, comma-separated, as plain text. That is the "
        "expected shape; it is not a malformed row and it is not missing a wrapper object.\n"
        "The following are ALREADY verified by computation before you see the row, so never "
        "reject for them: every label is one of the 28 names spelled exactly, the list is "
        "non-empty, and no label repeats.\n"
        "NEVER reject for formatting, punctuation, brackets, quotes or JSON. Those are properties "
        "of how the row was rendered for you, not of the answer, and rejecting on them is a "
        "measured failure mode: on run 39708679 that reasoning would have discarded 38-72% of "
        "generated rows whose formatting was in fact correct.\n"
        "The label set is FIXED and compared exactly, so a synonym is a wrong answer: it is "
        "`annoyance` and not `irritation`, `gratitude` and not `thankful`, `neutral` and not "
        "`none`. Multiple labels are legitimate and common.\n"
        "`neutral` means the comment expresses no particular emotion; it is a real label, not a "
        "fallback for uncertainty.\n"
        "Comments are Reddit text: informal, profane, sarcastic, often missing context. That is "
        "the corpus, not a defect, and a comment being crude is not a reason to reject the row.\n"
        "Judge ONLY whether the labels fit the comment: whether an emotion clearly expressed was "
        "left out, and whether a label was applied that the text does not support."
    ),
    quality_controls=(
        qc.require_fields("text", "labels", "label"),
        # NOT `balance_labels` or `label_space`. Both operate on a single-label `label` field: the
        # first would balance across comma-joined combinations, and the second would drop every
        # multi-label row for not matching a single-class vocabulary. The real skew here is
        # per-label, not per-row, and it is handled by the metric rather than by resampling.
        qc.length_outliers(key="text"),
        # Reddit comments repeat heavily — "thank you so much", "this is awesome" — and the corpus
        # is large enough that removing near-duplicates costs nothing.
        qc.dedup_surface(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="ekman_macro_f1",
    # 128, not 64. The gold answer is tiny — measured max 8 tokens over 400 dev rows — so 64 was
    # ample for a correct reply and that is not what the reserve has to cover. A BASE model on
    # this task writes prose ("The comment expresses a mix of emotions, including admiration,
    # amusement, ..."), and at 64 tokens that was being cut off mid-sentence. The strict extractor
    # scores such a reply as a format failure either way, so the verdict does not change — but a
    # zero-shot baseline should be low because the model did not follow the contract, not because
    # the harness stopped it mid-answer, and only the larger reserve lets us say which.
    #
    # 28 labels is also the theoretical maximum a compliant answer could name, which is ~138
    # tokens; 128 covers any realistic over-prediction rather than silently trimming it into a
    # shorter and possibly better-scoring one.
    max_new_tokens=128,
    max_seq_length=1024,
    eval_batch_size=32,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # Selection and reporting differ in METRIC but not in split: the report pass runs the full
    # 5,427-row test split and computes AUPRC from per-label rankings, which the loop cannot do.
    # `report_load` stays None because it IS the same split, just uncapped.
    report_load=None,
    report_score=scorer.score_report,
    report_metric_name="macro_auprc",

    build_training_turn=multilabel_emotion_turn,

    # ADDED 2026-09-08, having first been declared unnecessary. The original reasoning — that
    # whether a comment expresses `annoyance` or `disapproval` is a judgement, not a computation —
    # is true of CORRECTNESS and false of the LABEL SPACE, which is a fixed list of 28 names the
    # scorer compares exactly. Run 39708679 paid for the distinction: 32 of 1,214 synthetic rows
    # carried labels the taxonomy does not contain (`frustration` among them), and with no exact
    # verifier they went into training as targets the scorer can only mark wrong.
    #
    # It still cannot check whether the labels FIT the comment, so the teacher pass still runs.
    synth_verifier=_verify,
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="google-research-datasets/go_emotions",
            config="simplified",
            split="train",
            url="https://huggingface.co/datasets/google-research-datasets/go_emotions",
            supports_offset=True,
        ),
    ),
    # 43,410 train rows against a 5,000-row cap leaves ~38,000 unmined, so mining has somewhere
    # real to go and paid discovery would only rediscover mirrors of a cached corpus (B297).
    allow_paid_discovery=False,

    model_ranking_metric=None,
)

# Re-exported so a reader of this module can see the vocabulary the prompt pins without chasing
# it into the loader.
LABELS = EMOTIONS
