#!/usr/bin/env python3
"""
ner_metrics.py — NER scoring for energy_latency_agent.py's --score-accuracy
mode: entity_f1()/load_ner_eval_set() (written from scratch for this
project - no EvalSet class, no task-registry dependency), plus
build_prompts()/extract_predictions()/score()/failure_category_of() adapted
from /Users/ksd/Downloads/ner.py (the real scorer implementation, with the
None-vs-[] distinction and per-example failure categorization - a simpler
version without either exists at
/Users/ksd/SLM_Factory_GitHub/eval/scorers/ner.py, not used here). The only
change from that source: `eval_set` is a plain list of {"text","entities"}
dicts (from load_ner_eval_set() above) instead of an EvalSet object, so
`eval_set.all` becomes just `eval_set` throughout - the JSON-parsing/regex-
fallback logic, the None-vs-[] distinction, and failure_category_of()'s
categorization are otherwise unchanged.
"""

import json
import re
from collections import Counter


def entity_f1(predictions: list, gold: list) -> float:
    """Entity-level F1 over exact (text, type) pair multisets.

    predictions[i] may be `None` (the model's output didn't parse into a
    span list at all) - treated the same as an empty list for THIS
    function's counting purposes (zero predicted entities for that
    example). Callers that need to distinguish "genuinely predicted no
    entities" from "unparseable output" for failure-category reporting do
    that separately (see ner.py's own format_valid/failure_category_of) -
    entity_f1 itself only computes the aggregate score.

    Uses Counter (multiset) arithmetic, not set intersection, so a
    repeated entity mention is counted correctly on both sides: if gold
    lists the same (text, type) pair twice and a prediction lists it once,
    that's 1 true positive and 1 false negative, not a full match.
    """
    tp = fp = fn = 0
    for pred_spans, gold_spans in zip(predictions, gold):
        pred_counter = Counter((s["text"], s["type"]) for s in (pred_spans or []))
        gold_counter = Counter((s["text"], s["type"]) for s in (gold_spans or []))
        tp_counter = pred_counter & gold_counter  # multiset intersection: min count per key
        tp += sum(tp_counter.values())
        fp += sum((pred_counter - gold_counter).values())
        fn += sum((gold_counter - pred_counter).values())

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def load_ner_eval_set(path) -> list:
    """Reads a JSON file shaped {"pos": [...], "neg": [...], "boundary": [...]}
    (each entry a {"text":..., "entities":...} dict - the legacy slice
    format) and returns one flat list of {"text":..., "entities":...} dicts,
    in pos+neg+boundary order. No EvalSet, no task-type validation, no
    stratified sampling - just the flattening this scorer actually needs.
    """
    with open(path) as f:
        data = json.load(f)
    rows = list(data.get("pos") or []) + list(data.get("neg") or []) + list(data.get("boundary") or [])
    return [{"text": r.get("text", ""), "entities": r.get("entities", [])} for r in rows]


# ---------------------------------------------------------------------------
# Scorer (adapted from /Users/ksd/Downloads/ner.py - see module docstring)
# ---------------------------------------------------------------------------

NER_PROMPT = (
    'Extract named entities from the text. '
    'Reply with a JSON list of objects with "text" and "type" keys. '
    'Reply with [] if there are no entities.\n\nText: {text}'
)


def build_prompts(eval_set: list) -> list:
    return [NER_PROMPT.format(text=ex.get("text", "")) for ex in eval_set]


def extract_predictions(raw_outputs: list, eval_set: list) -> list:
    """Parse each reply into a span list, or None when it did not parse at all.

    `None` and `[]` used to be the same value, which made a model emitting prose indistinguishable
    from one correctly reporting no entities — and since most rows DO have entities, both scored
    zero and the format failure was invisible. A legitimate `[]` is a real prediction; unparseable
    output is not a prediction.
    """
    results: list = []
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


def score(eval_set: list, predictions: list) -> dict:
    gold = [ex.get("entities", []) for ex in eval_set]
    # An unparseable reply scores as predicting nothing, which is what it is worth, but it is
    # counted separately in `format_valid` so a span-F1 of 0.2 can be read as "wrong spans" or
    # "unreadable output" rather than being ambiguous between them.
    scoreable = [pred if pred is not None else [] for pred in predictions]
    f1 = entity_f1(scoreable, gold)
    readable = sum(1 for pred in predictions if pred is not None)
    format_valid = readable / len(predictions) if predictions else 0.0
    failures = [
        {**ex, "predicted": pred, "error_type": failure_category_of({"predicted": pred, **ex})}
        for ex, pred, g in zip(eval_set, scoreable, gold)
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
