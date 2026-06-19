# eval/metrics.py
from data.eval_set import EvalSet

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
    """
    tp = fp = fn = 0
    for pred_spans, gold_spans in zip(predictions, labels):
        pred_set = {(s["text"], s["type"]) for s in pred_spans}
        gold_set = {(s["text"], s["type"]) for s in gold_spans}
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def per_slice_scores(eval_set: EvalSet, predictions: list[str]) -> dict[str, float]:
    """Compute accuracy per slice (pos/neg/boundary)."""
    n_pos = len(eval_set.pos)
    n_neg = len(eval_set.neg)

    pos_preds = predictions[:n_pos]
    neg_preds = predictions[n_pos:n_pos + n_neg]
    bnd_preds = predictions[n_pos + n_neg:]

    def acc(preds, examples):
        if not examples:
            return 0.0
        return sum(p == e["label"] for p, e in zip(preds, examples)) / len(examples)

    return {
        "pos": acc(pos_preds, eval_set.pos),
        "neg": acc(neg_preds, eval_set.neg),
        "boundary": acc(bnd_preds, eval_set.boundary),
    }
