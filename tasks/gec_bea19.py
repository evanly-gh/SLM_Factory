"""W&I+LOCNESS (BEA-2019) — grammatical error correction. Pure text in, text out.

WHY THIS TASK IS IN THE SUITE
    It is the strongest privacy case here by a distance: a keyboard correction model sees literally
    everything the user types, so it cannot go to a server. It is also the only pure generation
    task, and it has the cheapest rows in the suite at ~18.6 tokens a sentence.

    The SFT delta is the largest of any task and points the right way for this project: zero-shot
    GPT-4 scores 43.0 F0.5 on BEA-19 dev, fine-tuned small models reach 62-75, and fine-tuning
    GPT-4o gained +22.07 F0.5 over its own zero-shot. A local model beating a frontier cloud model
    outright is the claim, and it is measurable here.

    34,308 training sentences IS the official BEA-2019 low-resource track, so published prior work
    exists at exactly this data scale rather than at a scale we would have to extrapolate from.

TWO NUMBERS, NEVER ONE
    F0.5 alone hides which failure mode you are in. Fine-tuned small models under-correct — high
    precision, low recall — and large untuned models over-correct, and both produce a middling
    F0.5. The scorer therefore reports precision and recall beside it always, and the failure
    taxonomy splits `missed_correction` from `overcorrected_correct_sentence` for the same reason.

WHAT OUR F0.5 IS AND IS NOT COMPARABLE TO
    W&I+LOCNESS train and dev are SINGLE-reference, which caps measurable recall. F0.5 is strongly
    reference-count dependent: CoNLL-14 with two references puts top systems near 68, and the same
    systems re-scored against a 10-annotator extension reach 80-81 against a human 72.58. So this
    number is comparable to other single-reference BEA-19 dev numbers and to nothing else. The
    scorer records `references_per_sentence` so a report cannot lose that.

    Also contaminated: BEA and CoNLL are old and public, and Lang-8 is scraped, so a zero-shot
    baseline here is inflated. That compresses the apparent delta rather than exaggerating it,
    which is a conservative bias — but it should be stated.
"""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import gec as scorer
from tasks._builders import gec_turn
from tasks.spec import MiningSource, TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.gec_bea19 import load_gec_bea19

    return load_gec_bea19(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="gec_bea19",
    title="W&I+LOCNESS BEA-2019 (grammatical error correction)",
    category="in_distribution",
    family="generation",

    load=_load,
    # `m2` is required, not decorative: it is the corpus's own gold edit annotation, and the
    # scorer reassembles a reference M2 file from it for whatever subset it is handed. Without it
    # the reference would have to be regenerated from the target string, which re-segments the
    # annotator's edits and changes the number. `cefr` is the episode axis.
    required_fields=("text", "answer", "cefr", "m2"),
    initial_train_cap=5000,
    select_cap=1000,
    eval_sampling="shuffled",
    closed_label_space=False,
    label_definitions={},
    entity_type_vocabulary=(),  # extracts no spans
    verifier_notes=(
        "You are shown a learner sentence and, as the proposed answer, ONLY its corrected form. "
        "The text is TOKENIZED — punctuation is separated by spaces, and contractions are split "
        "as in \"do n't\". That is the corpus convention and not an error to fix.\n"
        "A correction that is IDENTICAL to the input is legitimate and common: roughly a fifth of "
        "this corpus needs no correction at all, and a sentence that was already correct must be "
        "repeated unchanged rather than reworded.\n"
        "Judge ONLY whether the correction is right: whether a real grammatical error was left "
        "uncorrected, and whether anything was changed that was not an error. Do NOT reward "
        "rewriting for style, concision or naturalness — this task changes as little as possible, "
        "and a fluent rewrite of a correct sentence is a FALSE correction, which the metric "
        "penalizes twice as heavily as a missed one."
    ),
    quality_controls=(
        qc.require_fields("text", "answer", "cefr", "m2"),
        # NOT deduplicated. Learner corpora repeat short sentences heavily — "I like it very
        # much", "Thank you for your letter" — and those recur because learners genuinely write
        # them, often with different errors in each instance. A surface-similarity filter would
        # remove the systematic error patterns the model is here to learn.
        qc.length_outliers(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="errant_f05",
    # A correction is about as long as its input, and inputs average ~18.6 tokens. 256 is generous
    # and keeps the context almost entirely available to the prompt.
    max_new_tokens=256,
    max_seq_length=1024,
    eval_batch_size=32,
    failure_category=scorer.failure_category_of,
    # NOT judged. ERRANT is a programmatic scorer — it just happens to live in another
    # interpreter. `needs_judge` is about the LLM judge, whose outage must fail the run loudly;
    # ERRANT's unavailability fails loudly too, via `ErrantUnavailable`, for the same reason.
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # Selection and reporting coincide: ERRANT F0.5 is the published metric and a sound ranking
    # signal, with no thresholding or class-support problem. The report pass widens the eval to
    # the full 4,384-sentence dev split and adds seeds. Variance here is the suite's most
    # favourable — a recent system reports 71.24 +/- 0.28 — so three seeds is ample, and the
    # AGR/EM@N churn machinery TOPv2 needs is deliberately NOT imported.
    report_load=None,
    report_score=scorer.score,
    report_metric_name="errant_f05",

    build_training_turn=gec_turn,

    # No exact verifier. Whether a correction is GRAMMATICALLY right is not decidable by
    # computation, which is the whole task — so unlike the span and parse tasks there is nothing
    # free to check, and the teacher pass is the only gate. Stated as an explicit None rather
    # than left to a default so the weakness is visible where the decision was made.
    synth_verifier=None,
    # No chain-of-thought. The target is one sentence that must appear on one line, and a
    # `<reasoning>` block would become extra lines — which the extractor correctly rejects as a
    # format failure, so CoT here would train the model to fail its own output contract.
    cot_annotation=False,

    mining_sources=(
        # FCE, from the same BEA-2019 release and in the same M2 format: 28,350 more sentences of
        # learner English under the same licence. The scale-up path past that is cLang-8 at 2.37M
        # pairs, which is deliberately NOT declared here — the repo is archived, ships targets
        # only, and needs a Google Form and a reassembly script, so it cannot be a mid-run fetch.
        MiningSource(
            hf_id="cl.cam.ac.uk/research/nl/bea2019st",
            config=None,
            split="train",
            url="https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz",
            supports_offset=True,
        ),
    ),
    # Research/educational licence only (Lang-8 research-only, cLang-8 CC BY-NC-SA 4.0). Paid
    # discovery would go looking for more learner corpora, and the licence patchwork around this
    # task is exactly what should not be expanded by an autonomous fetch.
    allow_paid_discovery=False,

    model_ranking_metric=None,
)
