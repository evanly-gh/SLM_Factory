# eval/scorers/classification.py
from eval.metrics import binary_f1, per_slice_scores
from data.eval_set import EvalSet

CLASSIFY_PROMPT = (
    'Classify this message. Reply with exactly one word — the label.\n\nMessage: {text}'
)

def build_prompts(eval_set: EvalSet) -> list[str]:
    return [CLASSIFY_PROMPT.format(text=ex["text"]) for ex in eval_set.all]

def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]:
    """Extract label from raw model output. Falls back to majority neg label."""
    all_labels = {e["label"] for e in eval_set.all}
    pos_label = min(
        {e["label"] for e in eval_set.pos} or all_labels,
        key=lambda l: sum(1 for e in eval_set.all if e["label"] == l)
    )
    def extract(raw: str) -> str:
        cleaned = raw.strip().lower()
        for lbl in all_labels:
            if lbl.lower() in cleaned:
                return lbl
        return next(iter(all_labels - {pos_label}), pos_label)
    return [extract(r) for r in raw_outputs]

def score(eval_set: EvalSet, predictions: list[str]) -> dict:
    labels = [e["label"] for e in eval_set.all]
    all_labels = list({e["label"] for e in eval_set.all})
    pos_label = min(all_labels, key=lambda l: labels.count(l))
    f1 = binary_f1(predictions, labels, pos_label=pos_label)
    per_class = {lbl: binary_f1(predictions, labels, pos_label=lbl) for lbl in all_labels}
    slices = per_slice_scores(eval_set, predictions)
    failures = [
        {**ex, "predicted": pred}
        for ex, pred, lbl in zip(eval_set.all, predictions, labels)
        if pred != lbl
    ]
    return {"f1": f1, "per_class": per_class, "slices": slices, "failures": failures}
