"""UCI SMS Spam Collection — is this text message unsolicited spam or a real message?"""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import classification as scorer
from tasks._builders import classification_turn
from tasks.spec import MiningSource, TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.sms_spam import load_sms_spam

    return load_sms_spam(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="sms_spam",
    title="SMS Spam Collection (spam vs ham)",
    # Train and eval are a stratified split of ONE corpus, so the model is scored on the
    # distribution it was taught. That is the definition the other two in-distribution tasks use.
    category="in_distribution",
    family="classification",

    load=_load,
    required_fields=("text", "label"),
    # The deduplicated corpus is ~5,150 rows and 20% is held out, so the train half is ~4,120 —
    # below the 5,000 the other classification tasks ask for. The cap is set under that on purpose
    # so `mine_new_real` has somewhere to go on its first attempt, which is the lesson calendar_json
    # taught: a cap that consumes the whole pool at cold start kills the data intervention outright.
    initial_train_cap=3000,
    select_cap=1000,
    # Binary at a ~87/13 base rate. Without round-robin sampling a 1,000-row eval draw is ~870 ham,
    # and the minority class the metric is computed over would rest on a few dozen rows.
    eval_sampling="label_balanced",
    closed_label_space=True,
    # `ham` is domain jargon, not English. Left undefined, the teacher reads it as the food and
    # rejects perfectly good rows for not being about pork — the same failure RouterBench hit when
    # it read `local` as "nearby" (B267). `spam` needs the complement stated for the same reason.
    label_definitions={
        "spam": (
            "an unsolicited bulk message — advertising, a prize/competition claim, a premium-rate "
            "number, a subscription service, or a phishing lure. Judge whether the message was "
            "sent without the recipient's consent for commercial or fraudulent gain"
        ),
        "ham": (
            "a legitimate personal or transactional message the recipient expected — friends and "
            "family, arrangements, replies, reminders. NOT a reference to the food; `ham` is the "
            "conventional name for the non-spam class in this corpus"
        ),
    },
    entity_type_vocabulary=(),  # extracts no spans
    verifier_notes="",
    quality_controls=(
        qc.require_fields("text", "label"),
        qc.label_space(),
        # Deliberately looser than the 3:1 the other classification tasks use. The real base rate
        # is ~6.5:1 ham:spam, and clamping the curriculum to 3:1 would teach a prior the eval set
        # does not have. 8:1 trims the extreme tail without rewriting the distribution.
        qc.balance_labels(max_ratio=8),
        qc.length_outliers(key="text"),
        # SMS messages are short and formulaic, and the corpus repeats near-identical spam
        # templates. The loader removes only EXACT duplicates (before splitting, so no copy
        # straddles the firewall); this removes the near-duplicates that survive that.
        qc.dedup_surface(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    # Binary and heavily imbalanced. Accuracy scores 0.87 for always answering `ham`, and macro-F1
    # still scores ~0.47 for it. Minority-class F1 scores it 0.0, which is the only one of the
    # three that reports collapse as failure.
    score=scorer.score_minority_f1,
    metric_name="minority_f1",
    max_new_tokens=50,
    max_seq_length=1024,
    eval_batch_size=32,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # Selection and reporting coincide: two classes, and `eval_sampling="label_balanced"` guarantees
    # the minority class has support in any draw, so the selection metric is already the honest one.
    report_load=None,
    report_score=scorer.score_minority_f1,
    report_metric_name="minority_f1",

    build_training_turn=classification_turn,

    # Closed label space: a generated row copies a real anchor's label, so the target cannot be
    # wrong — only the phrasing can be, and that is what the teacher pass checks.
    synth_verifier=None,
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="ucirvine/sms_spam",
            config="plain_text",
            split="train",
            url="https://huggingface.co/datasets/ucirvine/sms_spam",
            supports_offset=True,
        ),
    ),
    # The canonical corpus is small and fully consumed within a couple of rebuilds, so discovery
    # is the only route to more real rows. The closed label space bounds the risk: a mapped row
    # whose label is not `ham` or `spam` is dropped per-row.
    allow_paid_discovery=True,

    model_ranking_metric="MMLU",
)
