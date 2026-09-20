# eval/metrics.py
"""Scoring primitives. Deliberately dependency-free where the definition is short.

WHY NOT seqeval / scikit-learn
    Both would be one `pip install` and both are the wrong trade here. The definitions below are
    twenty to sixty lines each, they are the definitions the papers state, and writing them out
    means the exact convention is READABLE at the point it is used — whether an entity is matched
    as a typed multiset or a token span, whether average precision interpolates, whether a class
    with no support counts as a zero or is excluded. Those choices move the reported number by
    more than any implementation detail, and a library call hides all of them behind a name.

    `seqeval` in particular would have been actively wrong for this suite: it reconstructs spans
    from BIO tags, and a stray `I-` continuation with no `B-` is read as the start of a new span.
    That is a documented contamination source in PIIBench's own test file, and MultiCoNER's
    corrupted test portion is exactly where such tags appear.

    The two places a real dependency IS justified are the two where the metric is defined by an
    implementation rather than an equation: `errant` for GEC edit alignment, and `rouge_score` for
    ROUGE's stemming and tokenization. Reimplementing either would produce numbers that are not
    comparable to published ones, which is the whole reason to use those metrics at all. Both live
    outside the training environment; see `scripts/setup_metric_envs.sh`.
"""
from collections import Counter, defaultdict

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


def _f_beta(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    """F-beta from raw counts. `beta` > 1 weights recall, < 1 weights precision."""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def entity_counts_by_type(
    predictions: list[list[dict]], labels: list[list[dict]]
) -> dict[str, dict[str, int]]:
    """Per-entity-type TP/FP/FN over typed spans, as a multiset per row.

    The unit is the (text, type) pair within one row, counted with multiplicity — the same
    convention `entity_f1` uses, for the same reason: set arithmetic would deduplicate repeated
    mentions and undercount both TP and FN. Biomedical abstracts and Wikipedia sentences both
    repeat the same surface form many times, so that is not a corner case.

    A type appears in the result if it occurs in EITHER gold or predictions. Including
    prediction-only types matters: a model that invents a class nobody asked for should show up as
    a column of pure false positives rather than vanishing from the breakdown.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for pred_spans, gold_spans in zip(predictions, labels):
        pred_counter = Counter(
            (s["text"], s["type"]) for s in pred_spans or []
            if isinstance(s, dict) and "text" in s and "type" in s
        )
        gold_counter = Counter(
            (s["text"], s["type"]) for s in gold_spans or []
            if isinstance(s, dict) and "text" in s and "type" in s
        )
        for (_text, entity_type), n in (pred_counter & gold_counter).items():
            counts[entity_type]["tp"] += n
        for (_text, entity_type), n in (pred_counter - gold_counter).items():
            counts[entity_type]["fp"] += n
        for (_text, entity_type), n in (gold_counter - pred_counter).items():
            counts[entity_type]["fn"] += n
    return dict(counts)


def entity_prf_by_type(
    predictions: list[list[dict]], labels: list[list[dict]]
) -> dict[str, dict[str, float]]:
    """Per-type precision, recall, F1 and gold support.

    `support` is the number of GOLD mentions of the type — tp + fn — and is carried alongside the
    scores because a macro average is only as meaningful as the support behind its weakest class.
    MultiCoNER's rarest coarse group is 0.18% of entities; a per-class F1 there is a fact about a
    handful of rows, and the support column is what stops it being read as anything more.
    """
    out: dict[str, dict[str, float]] = {}
    for entity_type, c in entity_counts_by_type(predictions, labels).items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        out[entity_type] = {
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "f1": _f_beta(tp, fp, fn),
            "support": tp + fn,
        }
    return out


def entity_micro_f1(predictions: list[list[dict]], labels: list[list[dict]]) -> float:
    """Entity-level F1 with all types pooled. Frequent types dominate.

    Arithmetically the same thing as `entity_f1`, but computed by SUMMING `entity_counts_by_type`
    rather than by calling it. Two reasons, and the second is the load-bearing one:

      1. A task reporting both micro and macro should name which is which at the call site.
      2. Micro and macro must not be able to disagree about what they counted. Delegating meant
         two independent counting loops with different robustness: `entity_f1` indexes `s["text"]`
         directly and raises KeyError on a malformed span, while the per-type counter skips it. A
         scorer handing the same rows to both would then get a macro number and an exception.
         MultiCoNER's corrupted test portion is precisely where malformed spans turn up.

    `entity_f1` is left as it is because `ner_bc5cdr` scores through it and its extractor already
    filters spans to well-formed dicts, so its stricter indexing is unreachable there.
    """
    counts = entity_counts_by_type(predictions, labels)
    tp = sum(c["tp"] for c in counts.values())
    fp = sum(c["fp"] for c in counts.values())
    fn = sum(c["fn"] for c in counts.values())
    return _f_beta(tp, fp, fn)


def entity_macro_f1(
    predictions: list[list[dict]],
    labels: list[list[dict]],
    min_support: int = 0,
) -> float:
    """Mean of per-type F1, so a type with six mentions counts as much as one with six thousand.

    THE HEADLINE FOR A FINE-GRAINED LABEL SPACE, and the reason `min_support` exists. Averaging
    equally over 33 types is the point — it is what stops a model scoring well by learning the
    four common classes — but it also means a type with almost no gold support contributes a
    number that is nearly noise.

    `min_support=0` averages over every type present, which is the literature's convention and the
    right default for comparability. A caller reporting a headline passes the threshold it
    declared, and the excluded types stay visible in `entity_prf_by_type`.
    """
    per_type = entity_prf_by_type(predictions, labels)
    scores = [stats["f1"] for stats in per_type.values() if stats["support"] >= min_support]
    return sum(scores) / len(scores) if scores else 0.0


def average_precision(scores: list[float], labels: list[int]) -> float:
    """Average precision for one binary label, threshold-free.

    Computed as the step-wise sum `sum_n (R_n - R_{n-1}) * P_n` over the ranking, which is what
    scikit-learn's `average_precision_score` does and what the GoEmotions literature reports as
    AUPRC. It deliberately does NOT interpolate: trapezoidal interpolation over a
    precision-recall curve is optimistically biased, and the two conventions differ by enough to
    matter when comparing against a published number.

    Ties are resolved by consuming every item at one score together, so a model that gives a
    positive and a negative the same score gets credit for neither ordering. Without that, the
    result would depend on the input's incidental order — which, for a model scoring 28 labels
    with a shared prompt, is a real possibility rather than a theoretical one.
    """
    if not scores or not any(labels):
        # No positives means average precision is undefined rather than zero. Callers exclude
        # unsupported classes from the macro average instead of letting a class with nothing to
        # find drag it down; this is only the safe scalar for the degenerate case.
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    total_positive = sum(labels)
    tp = seen = 0
    previous_recall = 0.0
    total = 0.0
    index = 0
    while index < len(order):
        tie_end = index
        while tie_end < len(order) and scores[order[tie_end]] == scores[order[index]]:
            tie_end += 1
        for position in range(index, tie_end):
            tp += labels[order[position]]
            seen += 1
        total += (tp / total_positive - previous_recall) * (tp / seen)
        previous_recall = tp / total_positive
        index = tie_end
    return total


def macro_average_precision(
    scores_by_label: dict[str, list[float]],
    gold_by_label: dict[str, list[int]],
    min_support: int = 1,
) -> tuple[float, dict[str, float]]:
    """Mean average precision across labels, plus the per-label values.

    THE THRESHOLD-FREE ANSWER TO A THRESHOLDING PROBLEM. Macro-F1 over a multi-label space is not
    a property of the model alone — practitioners variously use a fixed 0.5, a fixed 0.3, or a
    dev-tuned per-class sweep, and the same model moves several points between them, which reads
    as a model improvement and is not one. AUPRC asks only whether the model RANKS the correct
    labels highly, so no cutoff can move it.

    Labels with fewer than `min_support` positives are excluded and the survivors reported back,
    so a caller can state how many were dropped. A label with zero positives has no average
    precision at all, and scoring it 0.0 would penalize the model for a class the split lacks.
    """
    per_label = {
        label: average_precision(scores_by_label[label], gold_by_label.get(label, []))
        for label in sorted(scores_by_label)
        if sum(gold_by_label.get(label, [])) >= min_support
    }
    macro = sum(per_label.values()) / len(per_label) if per_label else 0.0
    return macro, per_label


def multi_reference_rouge(
    predictions: list[str],
    references: list[list[str]],
    rouge_types: tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
) -> dict[str, float]:
    """Corpus ROUGE against several references per example, max-then-mean.

    WHY MULTI-REFERENCE IS THE WHOLE POINT ON DIALOGSUM. A summary has many correct forms, so
    scoring against one annotator's wording makes the number partly a lottery about whose phrasing
    the model happened to match. DialogSum's test split ships THREE human summaries per dialogue;
    taking the best-matching reference per example and averaging over examples is the standard
    multi-reference protocol, and it is what makes the published human ceiling — one annotator
    scored against the other two, at ROUGE-1 53.35 / ROUGE-2 26.72 / ROUGE-L 50.84 — the right
    thing to read a 47 against.

    Delegated to `rouge_score` rather than reimplemented: ROUGE is defined by its Porter stemming
    and tokenization, so a hand-rolled version would produce numbers comparable to nothing.
    `use_stemmer=True` matches the summarization literature's convention.
    """
    from rouge_score import rouge_scorer

    totals = {name: 0.0 for name in rouge_types}
    if not predictions:
        return totals
    scorer = rouge_scorer.RougeScorer(list(rouge_types), use_stemmer=True)
    for prediction, refs in zip(predictions, references):
        usable = [str(r) for r in (refs or []) if str(r).strip()]
        if not usable or not str(prediction or "").strip():
            # An empty prediction, or an example with no reference, scores zero rather than being
            # dropped. Dropping it would quietly shrink the denominator and reward a model for
            # declining to answer.
            continue
        best = scorer.score_multi(usable, str(prediction))
        for name in rouge_types:
            totals[name] += best[name].fmeasure
    return {name: totals[name] / len(predictions) for name in rouge_types}
