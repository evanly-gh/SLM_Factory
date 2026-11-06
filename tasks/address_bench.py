"""address-bench — is the latest transcript line addressed to the wearer's voice assistant?

Ported 2026-10-06 from `bubble-agent-pipeline/data/bench/address-bench`. The loader
(`data/loaders/address_bench.py`) explains the corpus and the rendering; the scorer
(`eval/scorers/address_gate.py`) explains the prompt, the bucket taxonomy and the metric.
"""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import address_gate as scorer
from tasks.spec import TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.address_bench import load_address_bench

    return load_address_bench(max_train=max_train, max_test=max_test, log=log)


def _training_turn(row: dict, _ctx) -> tuple[str, str, str]:
    """`(prompt, target, marker)` for completion-only SFT.

    Not `_builders.classification_turn`: that builds its prompt from `ctx.labels`, which would
    enumerate the two label words and drop the definition of "addressed" entirely. The definition
    IS the task here (see the scorer's docstring), so the turn imports the eval prompt builder —
    the same defence against train/serve drift every other builder in `_builders.py` uses (B290).
    """
    return (
        scorer.build_prompt(row.get("text", "")),
        str(row.get("label", "")),
        "Answer",
    )


SPEC = TaskSpec(
    name="address_bench",
    title="Address gate — is this line for the assistant? (bubble-agent address-bench)",
    # The label is not readable off the line's surface form: the N4/N10 buckets are built to punish
    # a model that keys on wake-word lookalikes and imperative mood. Same family as
    # `proactive_listening`, and for the same reason — fine-tuning supplies the decision boundary.
    category="out_of_distribution",
    family="classification",

    load=_load,
    required_fields=("text", "label"),
    # The corpus is 6,000 rows and 20% is held out, so the train half is ~4,800. The cap sits under
    # that on purpose so `mine_new_real` has somewhere to go on its first attempt — the
    # calendar_json lesson that a cap consuming the whole pool at cold start kills the data
    # intervention outright.
    initial_train_cap=3000,
    select_cap=1000,
    # Binary at 25/75. A 1,000-row unstratified draw is ~750 not_addressed, and recall — the thing
    # the metric is built around — would rest on the other 250.
    eval_sampling="label_balanced",
    closed_label_space=True,
    # Both labels are opaque to a teacher that has not been told the gate's rule. `not_addressed`
    # especially: left undefined it reads as "nobody was addressed", and the teacher rejects
    # perfectly good rows where the User is plainly talking to another person — the B267 failure
    # `sms_spam`'s `ham` and RouterBench's `local` both hit.
    label_definitions={
        "addressed": (
            "the User asks the assistant a question, gives it a request or command, or calls on "
            "it by name; or directly follows up on what the Assistant just said; or adds to or "
            "changes a request they just made to it"
        ),
        "not_addressed": (
            "the User is talking to another person, to themselves, or ABOUT the assistant "
            "(describing how it works, quoting a command as an example, a hypothetical). A "
            "request-shaped line is for the person the User is talking with unless something "
            "marks it as for the assistant; a name merely mentioned is not a call; and a line "
            "spoken by anyone other than the User is always not_addressed"
        ),
    },
    entity_type_vocabulary=(),  # extracts no spans
    # The conventions a teacher cannot infer from a sample of rows, stated so it does not invent
    # its own (the `calendar_json` lesson recorded on TaskSpec.verifier_notes).
    verifier_notes=(
        "Transcript conventions: `User` is the wearer of the assistant; `Speaker2`, `Speaker3`, … "
        "are other people in the room; `Assistant` is what the assistant itself said. The "
        "assistant has NO fixed name — it may be addressed as `agent`, `earbuds`, `glasses`, or "
        "with no name at all, so the absence of a wake word is not evidence of `not_addressed`. "
        "Transcripts come from speech recognition and may contain mishearings, filler and lines "
        "split mid-sentence; that noise is part of the task and must not be corrected. Only the "
        "LATEST line is being labelled — the context lines are evidence, never the target."
    ),
    quality_controls=(
        qc.require_fields("text", "label"),
        qc.label_space(),
        # Looser than the 3:1 the other classification tasks use. The real base rate is 3:1
        # not_addressed:addressed and the eval set keeps it; clamping the curriculum tighter would
        # teach a prior the eval set does not have. 4:1 trims the tail without rewriting the
        # distribution.
        qc.balance_labels(max_ratio=4),
        # Anchored on the trusted rows' median. Transcript blocks vary a lot in length (a row with
        # no context is ~60 chars, one with 8 context lines ~400), so the ratio is loosened from
        # the 3.0 default to avoid discarding the short, context-free rows — which are exactly the
        # P1/P10 shapes the gate most needs to get right.
        qc.length_outliers(key="text", max_ratio=5.0),
        # The corpus repeats scenario templates across buckets with small substitutions, and
        # synthesis will produce more of the same. The loader removes only EXACT duplicates
        # (before splitting, so no copy straddles the firewall); this catches the near ones.
        qc.dedup_surface(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    # F2 on `addressed` — recall weighted 2x precision, because a miss (assistant stays silent
    # when spoken to) is the failure that matters and a false yes is one wasted judge call. The
    # upstream harness expresses the same preference as "the threshold that misses no addressed
    # turn", which needs a per-row P(true) this pipeline cannot produce from a quantized
    # generative artifact. Full argument, including F2's 0.625 always-yes floor and the three
    # diagnostics that expose it, in `eval/scorers/address_gate.py`.
    score=scorer.score_address_gate,
    metric_name="f2_addressed",
    # One word. The slack above that is for a model that prefixes its verdict with a clause; the
    # extractor reads the tail, so it does not need room to ramble.
    max_new_tokens=16,
    # A rendered transcript block runs ~400 chars at the 8-line default, and the system prompt
    # carrying the gate definition is ~1,100 — call it ~500 tokens of prompt at the top end.
    max_seq_length=2048,
    eval_batch_size=32,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # The report split is the SAME split, just not truncated to `select_cap` — which is exactly
    # what `report_load=None` already means (`scripts/report_eval.py` re-calls `load` with an
    # enormous `max_test`). Only a task whose report split is a genuinely different split declares
    # its own, as `multiconer` does. That untruncated draw is ~1,200 rows, which puts ~60 rows in
    # each of the 20 buckets instead of the ~50 a 1,000-row label-balanced draw leaves.
    #
    # F2 is the honest headline at both sizes, so unlike `multiconer` (micro selects, macro
    # reports) and `goemotions` (F1 selects, AUPRC reports) this task names the same scorer twice:
    # the question was asked and the answer was that they coincide.
    report_load=None,
    report_score=scorer.score_address_gate,
    report_metric_name="f2_addressed",

    build_training_turn=_training_turn,

    # Closed label space: a generated row copies a real anchor's label, so the target cannot be
    # wrong — only the transcript phrasing can be, and that is what the teacher pass checks. The
    # same reasoning `sms_spam` records.
    synth_verifier=None,
    cot_annotation=False,

    # NO MINING SOURCES. The corpus was generated by bubble-agent-pipeline's
    # `scripts/build_gate_synthetic.py` and exists nowhere else — there is no HuggingFace mirror
    # to offset into and no public corpus of transcripts labelled for assistant-addressivity.
    # `mine_new_real` will retire itself on the first attempt, which is the honest state: more real
    # rows for this task means generating them upstream, not discovering them.
    mining_sources=(),
    # Paid discovery is OFF for the same reason. Exa cannot find this corpus, and the rows it would
    # find instead (chatbot intent datasets, wake-word corpora) carry a different label definition
    # — they would be contamination dressed as acquisition, not data.
    allow_paid_discovery=False,

    model_ranking_metric="MMLU",
)
