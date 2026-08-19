# eval/scorers/ner.py
from eval.metrics import entity_f1
from data.eval_set import EvalSet
from collections import Counter
import json
import re

NER_PROMPT = (
    'Extract named entities from the text. '
    'Reply with a JSON list of objects with "text" and "type" keys. '
    'Reply with [] if there are no entities.\n\nText: {text}'
)

def build_prompts(eval_set: EvalSet) -> list[str]:
    return [NER_PROMPT.format(text=ex.get("text", "")) for ex in eval_set.all]

def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[list[dict] | None]:
    """Parse each reply into a span list, or None when it did not parse at all.

    `None` and `[]` used to be the same value, which made a model emitting prose indistinguishable
    from one correctly reporting no entities — and since most rows DO have entities, both scored
    zero and the format failure was invisible. A legitimate `[]` is a real prediction; unparseable
    output is not a prediction.
    """
    results: list[list[dict] | None] = []
    for raw in raw_outputs:
        try:
            try:
                spans = json.loads(str(raw).strip())
            except Exception:
                match = re.search(r'\[.*\]', str(raw), re.DOTALL)
                if not match:
                    results.append(None)
                    continue
                spans = json.loads(match.group())
            if not isinstance(spans, list):
                results.append(None)
                continue
            results.append([s for s in spans if isinstance(s, dict) and "text" in s and "type" in s])
        except Exception:
            results.append(None)
    return results

def _pairs(spans) -> Counter:
    return Counter((s["text"], s["type"]) for s in spans or [])


def score(eval_set: EvalSet, predictions: list[list[dict] | None]) -> dict:
    gold = [ex.get("entities", []) for ex in eval_set.all]
    # An unparseable reply scores as predicting nothing, which is what it is worth, but it is
    # counted separately in `format_valid` so a span-F1 of 0.2 can be read as "wrong spans" or
    # "unreadable output" rather than being ambiguous between them.
    scoreable = [pred if pred is not None else [] for pred in predictions]
    f1 = entity_f1(scoreable, gold)
    readable = sum(1 for pred in predictions if pred is not None)
    format_valid = readable / len(predictions) if predictions else 0.0
    failures = [
        {**ex, "predicted": pred, "error_type": failure_category_of({"predicted": pred, **ex})}
        for ex, pred, g in zip(eval_set.all, scoreable, gold)
        if _pairs(pred) != _pairs(g)
    ]
    return {
        "f1": f1,
        "metric": "span_f1",
        "per_class": {"entity_f1": f1, "format_valid": format_valid},
        "failures": failures,
        "format_valid": format_valid,
    }


def failure_category_of(failure: dict) -> str:
    """Why this row's span set is wrong, in a category that suggests a different fix.

    Missed spans and invented spans are opposite errors — one wants more positive examples, the
    other wants harder negatives — and reporting both as a single aggregate told the orchestrator
    nothing it could act on.
    """
    if failure.get("predicted") is None:
        return "unparseable_output"
    predicted = _pairs(failure.get("predicted"))
    gold = _pairs(failure.get("entities"))
    if not predicted and gold:
        return "no_entities_predicted"
    if predicted and not gold:
        return "entities_hallucinated"
    if {t for _s, t in predicted} != {t for _s, t in gold}:
        return "wrong_entity_type"
    if predicted - gold and gold - predicted:
        return "wrong_span_boundaries"
    return "partial_span_set"
