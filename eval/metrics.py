# eval/metrics.py
from collections import Counter

def binary_f1(predictions: list[str], labels: list[str], pos_label: str) -> float:
    """Compute binary F1 for the positive class."""
    tp = sum(p == pos_label and l == pos_label for p, l in zip(predictions, labels))
    fp = sum(p == pos_label and l != pos_label for p, l in zip(predictions, labels))
    fn = sum(p != pos_label and l == pos_label for p, l in zip(predictions, labels))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def entity_f1(predictions: list[list[dict]], labels: list[list[dict]]) -> float:
    """
    Compute entity-level F1 for NER.
    Each element is a list of {"text": str, "type": str} entity spans.

    Uses Counter (multiset) arithmetic so repeated entity mentions are counted
    correctly. Set intersection would deduplicate, undercounting TP/FN when the
    same (text, type) pair appears multiple times in gold or predictions.
    """
    tp = fp = fn = 0
    for pred_spans, gold_spans in zip(predictions, labels):
        pred_counter = Counter((s["text"], s["type"]) for s in pred_spans)
        gold_counter = Counter((s["text"], s["type"]) for s in gold_spans)
        # Multiset intersection: min count per key
        tp_counter = pred_counter & gold_counter
        tp += sum(tp_counter.values())
        fp += sum((pred_counter - gold_counter).values())
        fn += sum((gold_counter - pred_counter).values())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)
