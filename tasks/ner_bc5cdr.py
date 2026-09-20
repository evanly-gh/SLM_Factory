"""BC5CDR — chemical and disease spans from biomedical abstracts."""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import ner as scorer
from tasks._builders import ner_turn
from tasks.spec import MiningSource, TaskSpec


def _check(row: dict) -> tuple[bool, str]:
    """The verdict AND the reason. Exposed as `_verify.checker` below."""
    from data.synth_verifiers import verify_ner_row

    return verify_ner_row(row)


def _verify(row: dict) -> bool:
    return _check(row)[0]


# WHY THE REASON IS PUBLISHED SEPARATELY
#     `TaskSpec.synth_verifier` only has to answer yes/no, so this wrapper used to be
#     `return verify_ner_row(row)[0]` and the reason string was thrown away on the spot. `data.curriculum`
#     looks for a `.checker` attribute to recover it, finds nothing, and its
#     "[verify:exact] programmatic verifier rejected N row(s)" block is then unreachable.
#
#     Run 38985393 is what that costs. Synthesis generated 519 rows, the exact verifier rejected all
#     519, and the log recorded only the total — so which of the five checks fired (unparseable path,
#     undeclared API, bad argument schema, no terminal Finish, over the call budget) had to be
#     reverse-engineered afterwards from the vLLM access log. The information existed at the moment of
#     rejection and was discarded one character from where it was needed.
_verify.checker = _check


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.ner_bc5cdr import load_ner_bc5cdr

    return load_ner_bc5cdr(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="ner_bc5cdr",
    title="BC5CDR (Chemical/Disease spans)",
    category="format_bound",
    family="extraction",

    load=_load,
    required_fields=("text", "entities"),
    initial_train_cap=5000,
    select_cap=1000,
    eval_sampling="shuffled",
    # `entities` is the target, not `label`; there is no class vocabulary to pin.
    closed_label_space=False,
    label_definitions={},
    entity_type_vocabulary=("Chemical", "Disease"),
    # The teacher rejected 25 of 25 generated rows on run 38832588 — twice running, which is what
    # stopped the run — with reasons like "Output format is invalid; must be a JSON object with
    # 'text' and 'entities' fields". Every one of those rows had just passed `verify_ner_row`
    # 25/25, which checks the substantive property: that each span appears VERBATIM in the row's own
    # text. The teacher was judging the shape of what it was shown rather than the row, so it is
    # told here what it is looking at and what has already been checked by computation.
    verifier_notes=(
        "You are shown the abstract and, as the proposed answer, ONLY its entity list — a JSON "
        "array of {\"text\", \"type\"} objects. That is the expected shape; it is not a malformed "
        "row and it is not missing a wrapper object.\n"
        "The following are ALREADY verified by computation before you see the row, so never reject "
        "for them: the answer parses, every span appears verbatim in the abstract, no span is "
        "empty, and no (text, type) pair repeats.\n"
        "Judge ONLY the one thing computation cannot: whether the labelled spans are the right "
        "CHEMICALS and DISEASES for this abstract — in particular whether a real one was MISSED, "
        "or something labelled that is not a chemical or a disease."
    ),
    quality_controls=(
        qc.require_fields("text", "entities"),
        # Abstracts repeat the same drug names constantly; without a cap the curriculum teaches
        # a handful of surface forms rather than the span task.
        qc.entity_diversity(cap=3),
        qc.length_outliers(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="span_f1",
    max_new_tokens=512,
    max_seq_length=2048,
    eval_batch_size=16,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # Selection and reporting coincide: two entity types, both with thousands of spans, so micro
    # span-F1 is stable at selection size and is what the BC5CDR literature reports. Contrast
    # `multiconer`, where 33 classes and a 0.18%-of-entities tail force micro for selection and
    # macro for the headline.
    report_load=None,
    report_score=scorer.score,
    report_metric_name="span_f1",

    build_training_turn=ner_turn,

    # Span synthesis is checkable, which is what makes it safe enough to allow: every generated
    # span must be a real substring of the generated text at the offset it claims. That catches the
    # dominant teacher error (a plausible entity that does not appear verbatim) for free, before any
    # teacher call. What it cannot catch is a MISSED entity, so the teacher pass still runs.
    synth_verifier=_verify,
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="tner/bc5cdr",
            config=None,
            split="train",
            url="https://huggingface.co/datasets/tner/bc5cdr",
            supports_offset=True,
        ),
    ),
    allow_paid_discovery=True,

    model_ranking_metric=None,
)
