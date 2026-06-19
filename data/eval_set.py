# data/eval_set.py
import random
from dataclasses import dataclass, field

TASK_TYPES = {"classification", "NER", "generation"}

@dataclass
class EvalSet:
    pos: list[dict]
    neg: list[dict]
    boundary: list[dict]
    task_type: str
    all: list[dict] = field(init=False)

    def __post_init__(self):
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, got {self.task_type!r}")
        self.all = self.pos + self.neg + self.boundary

def build_eval_set(
    examples: list[dict],
    task_type: str,
    n_pos: int = 40,
    n_neg: int = 40,
    n_boundary: int = 20,
    seed: int = 42,
) -> EvalSet:
    """
    Build E = Epos ∪ Eneg ∪ Eboundary. All slices are disjoint.
    task_type determines what Eneg and Eboundary represent.

    classification:
      pos = clear positive-class examples
      neg = clear negative-class examples
      boundary = confusable pairs at the class boundary

    NER:
      pos = entity-rich passages with gold annotations
      neg = entity-free passages (hallucination test)
      boundary = passages with overlapping / near-miss entity types

    generation:
      pos = well-formed problems with unambiguous answers
      neg = adversarial / ill-posed inputs
      boundary = multi-step or edge-case problems
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}")

    rng = random.Random(seed)

    if task_type == "classification":
        pos_label = _infer_pos_label(examples)
        neg_label = _infer_neg_label(examples, pos_label)
        pos_examples = [e for e in examples if e["label"] == pos_label]
        neg_examples = [e for e in examples if e["label"] == neg_label]

        BOUNDARY_KEYWORDS = {"www", "http", "free", "win", "prize", "offer",
                             "click", "call", "urgent", "limited", "txt"}
        boundary_candidates = [
            e for e in neg_examples
            if any(kw in e.get("text", "").lower() for kw in BOUNDARY_KEYWORDS)
        ]
        if len(boundary_candidates) < n_boundary:
            short_pos = [e for e in pos_examples if len(e.get("text", "")) < 50]
            boundary_candidates += short_pos

        rng.shuffle(boundary_candidates)
        boundary = boundary_candidates[:n_boundary]
        boundary_texts = {e["text"] for e in boundary}

        clear_neg = [e for e in neg_examples if e["text"] not in boundary_texts]
        rng.shuffle(clear_neg)
        neg = clear_neg[:n_neg]

        clear_pos = [e for e in pos_examples if e["text"] not in boundary_texts]
        rng.shuffle(clear_pos)
        pos = clear_pos[:n_pos]

    else:
        # NER and generation: simple stratified split — boundary is agent-constructed
        # at runtime via Claude synthesis; here we just partition the examples provided
        rng.shuffle(examples)
        total = len(examples)
        p = min(n_pos, total // 3)
        n = min(n_neg, total // 3)
        b = min(n_boundary, total - p - n)
        pos = examples[:p]
        neg = examples[p:p + n]
        boundary = examples[p + n:p + n + b]

    return EvalSet(pos=pos, neg=neg, boundary=boundary, task_type=task_type)

def _infer_pos_label(examples: list[dict]) -> str:
    """Return the minority label (the positive class)."""
    from collections import Counter
    counts = Counter(e["label"] for e in examples)
    return min(counts, key=counts.get)

def _infer_neg_label(examples: list[dict], pos_label: str) -> str:
    labels = {e["label"] for e in examples}
    others = labels - {pos_label}
    return next(iter(others)) if others else pos_label
