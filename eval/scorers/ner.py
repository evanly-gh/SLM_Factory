# eval/scorers/ner.py
from eval.metrics import entity_f1, per_slice_scores
from data.eval_set import EvalSet
import json
import re

NER_PROMPT = (
    'Extract named entities from the text. '
    'Reply with a JSON list of objects with "text" and "type" keys. '
    'Reply with [] if there are no entities.\n\nText: {text}'
)

def build_prompts(eval_set: EvalSet) -> list[str]:
    return [NER_PROMPT.format(text=ex.get("text", "")) for ex in eval_set.all]

def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[list[dict]]:
    results = []
    for raw in raw_outputs:
        try:
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            spans = json.loads(match.group()) if match else []
            results.append([s for s in spans if "text" in s and "type" in s])
        except Exception:
            results.append([])
    return results

def score(eval_set: EvalSet, predictions: list[list[dict]]) -> dict:
    gold = [ex.get("entities", []) for ex in eval_set.all]
    f1 = entity_f1(predictions, gold)
    # Per-slice: convert to binary has-entity labels for slice scoring
    pred_labels = ["entity" if p else "no_entity" for p in predictions]
    gold_labels = ["entity" if g else "no_entity" for g in gold]
    slices = per_slice_scores(eval_set, pred_labels)
    failures = [
        {**ex, "predicted": pred}
        for ex, pred, g in zip(eval_set.all, predictions, gold)
        if set(tuple(s.items()) for s in pred) != set(tuple(s.items()) for s in g)
    ]
    return {"f1": f1, "per_class": {"entity_f1": f1}, "slices": slices, "failures": failures}
