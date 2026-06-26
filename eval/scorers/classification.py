# eval/scorers/classification.py
from eval.metrics import binary_f1, per_slice_scores
from data.eval_set import EvalSet

CLASSIFY_PROMPT = (
    'Classify this message. Reply with exactly one word — the label.\n\nMessage: {text}'
)

def build_prompts(eval_set: EvalSet) -> list[str]:
    return [CLASSIFY_PROMPT.format(text=ex["text"]) for ex in eval_set.all]

_UNKNOWN_LABEL = "__EXTRACTION_FAILED__"

def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]:
    """Extract label from raw model output. Uses __EXTRACTION_FAILED__ when no
    known label is found — this counts as a failure in the score function rather
    than silently inflating majority-class accuracy."""
    all_labels = {e["label"] for e in eval_set.all}
    def extract(raw: str) -> str:
        cleaned = raw.strip().lower()
        for lbl in all_labels:
            if lbl.lower() in cleaned:
                return lbl
        return _UNKNOWN_LABEL
    return [extract(r) for r in raw_outputs]

def score(eval_set: EvalSet, predictions: list[str]) -> dict:
    labels = [e["label"] for e in eval_set.all]
    all_labels = list({e["label"] for e in eval_set.all})
    per_class = {lbl: binary_f1(predictions, labels, pos_label=lbl) for lbl in all_labels}
    if len(all_labels) > 2:
        # Multi-class: macro-averaged F1 across every label.
        f1 = sum(per_class.values()) / len(per_class) if per_class else 0.0
    else:
        # Binary: F1 of the (minority) positive class.
        pos_label = min(all_labels, key=lambda l: labels.count(l))
        f1 = binary_f1(predictions, labels, pos_label=pos_label)
    slices = per_slice_scores(eval_set, predictions)
    failures = [
        {**ex, "predicted": pred}
        for ex, pred, lbl in zip(eval_set.all, predictions, labels)
        if pred != lbl
    ]
    return {"f1": f1, "per_class": per_class, "slices": slices, "failures": failures}
