"""MultiCoNER II scoring: 33-class entity extraction, micro to select and macro to publish.

WHY THE PROMPT ENUMERATES ALL 33 TYPES
    This repo has already paid for the alternative. `eval.scorers.ner.NER_PROMPT` asks for entities
    without naming the vocabulary, and the scorer then compares types EXACTLY. On the BC5CDR
    teacher probe that cost a measured 0.1011 against a real 0.6140: the teacher returned the right
    spans under its own label names — MEDICAL, SUBSTANCE, DISEASE instead of Chemical and Disease —
    and every one scored as a miss. With two guessable types that was a 6x understatement. With 33
    types including `OtherPROD`, `AerospaceManufacturer` and `Medication/Vaccine`, no model can
    guess the label set, so an unenumerated prompt would measure vocabulary telepathy rather than
    entity recognition.

    The cost is a long prompt on every row, which is why the enumeration is a bare comma-separated
    list rather than 33 glosses.

WHY MICRO SELECTS AND MACRO PUBLISHES
    Macro-F1 is the honest headline for a fine-grained label space: averaging equally over 33
    classes is what stops a model scoring well by learning the four common ones. It is also
    unusable as a per-iteration ranking signal, because the selection split is 871 dev rows and
    several classes have single-digit support there — a macro average over those is mostly noise,
    and the loop would chase it. Micro pools every mention, so it is stable at that size and still
    moves in the same direction as real improvement.
"""
from collections import Counter

from data.eval_set import EvalSet
from data.loaders.multiconer import COARSE_GROUPS, ENTITY_TYPES
from eval.metrics import entity_macro_f1, entity_micro_f1, entity_prf_by_type

# Below this many GOLD mentions in the scored split, a class's F1 is a fact about a handful of
# rows. Such classes are reported with their support and EXCLUDED from the headline macro, which
# is the task's own stated rule.
HEADLINE_MIN_SUPPORT = 20

FINE_NER_PROMPT = (
    "Extract named entities from the text and label each with one of these types:\n"
    "{types}\n\n"
    'Reply with a JSON list of objects with "text" and "type" keys, using the type names above '
    "exactly as written. Reply with [] if there are no entities.\n\n"
    "Text: {text}"
)


def build_prompt(text: str) -> str:
    """The ONE prompt. Imported by the training turn builder so the two cannot drift (B250)."""
    return FINE_NER_PROMPT.format(types=", ".join(ENTITY_TYPES), text=text)


def build_prompts(eval_set: EvalSet) -> list[str]:
    return [build_prompt(example.get("text", "")) for example in eval_set.all]


def extract_predictions(raw_outputs: list[str], eval_set: EvalSet):
    """Shared with `eval.scorers.ner`: the output contract is the same JSON span list.

    Reused rather than reimplemented so the `None` (unreadable) versus `[]` (a real prediction of
    no entities) distinction stays identical across both extraction tasks. That distinction is the
    whole basis of `format_valid`.
    """
    from eval.scorers.ner import extract_predictions as extract_spans

    return extract_spans(raw_outputs, eval_set)


def _pairs(spans) -> Counter:
    return Counter((s["text"], s["type"]) for s in spans or [] if isinstance(s, dict))


def _coarsen(spans):
    """Re-label spans with their coarse group, dropping types outside the taxonomy.

    A predicted type the taxonomy does not contain is dropped rather than mapped to a catch-all:
    at the coarse level it is not evidence about any of the six groups, and folding it into one
    would credit or penalize a group for a label the model invented. It is still counted as a
    false positive in the fine-grained numbers, which is where it belongs.
    """
    return [
        {"text": s["text"], "type": COARSE_GROUPS[s["type"]]}
        for s in spans or []
        if isinstance(s, dict) and s.get("type") in COARSE_GROUPS
    ]


def _score(eval_set: EvalSet, predictions, metric_name: str, headline: str) -> dict:
    gold = [example.get("entities", []) for example in eval_set.all]
    # An unreadable reply predicts nothing, which is what it is worth. It is counted separately in
    # `format_valid` so a low score can be read as wrong spans or as unreadable output.
    scoreable = [pred if pred is not None else [] for pred in predictions]

    micro = entity_micro_f1(scoreable, gold)
    per_type = entity_prf_by_type(scoreable, gold)
    macro_all = entity_macro_f1(scoreable, gold)
    macro_headline = entity_macro_f1(scoreable, gold, min_support=HEADLINE_MIN_SUPPORT)
    coarse_macro = entity_macro_f1(
        [_coarsen(pred) for pred in scoreable], [_coarsen(g) for g in gold]
    )

    readable = sum(1 for pred in predictions if pred is not None)
    format_valid = readable / len(predictions) if predictions else 0.0
    thin = sorted(
        name for name, stats in per_type.items()
        if 0 < stats["support"] < HEADLINE_MIN_SUPPORT
    )
    # The failure record carries the RAW prediction, with `None` preserved — not the `[]` the
    # metric scored it as. Those are different diagnoses: `None` is output the scorer could not
    # read, `[]` is the model correctly reporting no entities. Handing `scoreable` to
    # `failure_category_of` makes its `unparseable_output` branch unreachable and files every
    # format failure as `no_entities_predicted`, so a broken chat template would present as a
    # recall problem and send the orchestrator looking for more positive examples.
    failures = [
        {**example, "predicted": raw,
         "error_type": failure_category_of({**example, "predicted": raw})}
        for example, raw, pred, g in zip(eval_set.all, predictions, scoreable, gold)
        if _pairs(pred) != _pairs(g)
    ]
    return {
        "f1": {"micro_f1": micro, "macro_f1": macro_headline}[headline],
        "metric": metric_name,
        "per_class": {
            "micro_f1": micro,
            # The headline macro EXCLUDES thin classes; `macro_f1_all_classes` includes every
            # class present. Both are reported because the literature reports the unfiltered
            # number and the gap between them says how much of the macro rests on near-empty
            # classes.
            "macro_f1": macro_headline,
            "macro_f1_all_classes": macro_all,
            "coarse6_macro_f1": coarse_macro,
            "headline_min_support": HEADLINE_MIN_SUPPORT,
            "classes_below_support": thin,
            "classes_scored": len(per_type),
            "format_valid": format_valid,
            **{
                f"f1_{name}": stats["f1"] for name, stats in sorted(per_type.items())
            },
            **{
                f"support_{name}": stats["support"] for name, stats in sorted(per_type.items())
            },
        },
        "failures": failures,
        "format_valid": format_valid,
    }


def score(eval_set: EvalSet, predictions) -> dict:
    """SELECTION scoring: micro-F1 is the comparison scalar. Runs every iteration on the 871 dev rows."""
    return _score(eval_set, predictions, "micro_f1", headline="micro_f1")


def score_report(eval_set: EvalSet, predictions) -> dict:
    """REPORT scoring: macro-F1 over classes with real support. Runs once on the 20k test slice."""
    return _score(eval_set, predictions, "macro_f1", headline="macro_f1")


def failure_category_of(failure: dict) -> str:
    """Why this row's span set is wrong, in a category that suggests a different fix.

    `wrong_entity_type` is separated from the span errors because on a 33-class taxonomy it is the
    dominant and most actionable failure: the model found the entity and chose the wrong label,
    which is a label-space problem, not an extraction problem.
    """
    if failure.get("predicted") is None:
        return "unparseable_output"
    predicted = _pairs(failure.get("predicted"))
    gold = _pairs(failure.get("entities"))
    if not predicted and gold:
        return "no_entities_predicted"
    if predicted and not gold:
        return "entities_hallucinated"
    predicted_types = {t for _s, t in predicted}
    if predicted_types - set(COARSE_GROUPS):
        return "type_outside_taxonomy"
    if {s for s, _t in predicted} == {s for s, _t in gold} and predicted_types != {
        t for _s, t in gold
    }:
        return "wrong_entity_type"
    if predicted - gold and gold - predicted:
        return "wrong_span_boundaries"
    return "partial_span_set"
