# eval/scorers/classification.py
import re
from eval.metrics import binary_f1, per_slice_scores
from data.eval_set import EvalSet

CLASSIFY_PROMPT = (
    'Classify this message into exactly one of these labels: {labels}.\n'
    'Reply with only the label word — nothing else.\n\nMessage: {text}'
)


def _labels_str(labels) -> str:
    """Deterministic, deduped, sorted label list for the prompt. Sorting makes the string
    identical between training and eval as long as they see the same label vocabulary
    (train/serve parity, B161)."""
    return ", ".join(sorted({str(x) for x in labels if x is not None and str(x).strip()}))


def build_classify_prompt(text: str, labels) -> str:
    """Build the classification prompt WITH the allowed label set enumerated. Listing the
    labels is what lets a model (especially a strong instruct model like Qwen3-4B) emit an
    in-vocabulary label instead of a synonym the exact-match extractor can't score — the
    root cause of the tier-3 100% __EXTRACTION_FAILED__ collapse (B161)."""
    return CLASSIFY_PROMPT.format(labels=_labels_str(labels), text=text)


def build_prompts(eval_set: EvalSet) -> list[str]:
    labels = [e["label"] for e in eval_set.all]
    return [build_classify_prompt(ex["text"], labels) for ex in eval_set.all]

_UNKNOWN_LABEL = "__EXTRACTION_FAILED__"

def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]:
    """Extract label from raw model output.

    Priority: (1) exact word-boundary match, (2) substring match, (3) __EXTRACTION_FAILED__.
    Word-boundary matching prevents "positive" from matching "very_positive".
    """
    all_labels = {e["label"] for e in eval_set.all}

    def extract(raw: str) -> str:
        cleaned = raw.strip().lower()
        # Pass 1: word-boundary match (most precise)
        for lbl in sorted(all_labels, key=len, reverse=True):  # longest label first
            if re.search(r'\b' + re.escape(lbl.lower()) + r'\b', cleaned):
                return lbl
        # Pass 2: substring match (fallback for labels without word boundaries)
        for lbl in sorted(all_labels, key=len, reverse=True):
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
    return {
        "f1": f1,
        "metric": "macro_f1",
        "per_class": per_class,
        "slices": slices,
        "failures": failures,
    }
