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

def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[list[dict]]:
    results = []
    for raw in raw_outputs:
        try:
            try:
                spans = json.loads(raw.strip())
            except Exception:
                match = re.search(r'\[.*\]', raw, re.DOTALL)
                spans = json.loads(match.group()) if match else []
            results.append([s for s in spans if "text" in s and "type" in s])
        except Exception:
            results.append([])
    return results

def score(eval_set: EvalSet, predictions: list[list[dict]]) -> dict:
    gold = [ex.get("entities", []) for ex in eval_set.all]
    f1 = entity_f1(predictions, gold)
    failures = [
        {**ex, "predicted": pred}
        for ex, pred, g in zip(eval_set.all, predictions, gold)
        if Counter((s['text'], s['type']) for s in pred) != Counter((s['text'], s['type']) for s in g)
    ]
    return {
        "f1": f1,
        "metric": "span_f1",
        "per_class": {"entity_f1": f1},
        "failures": failures,
    }
