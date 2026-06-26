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

    if task_type == "classification" and len({e["label"] for e in examples}) > 2:
        # Multi-class: stratify all three slices across every label so the eval set
        # covers the full label range (binary pos/neg has no meaning with >2 classes).
        by_label: dict[str, list[dict]] = {}
        for e in examples:
            by_label.setdefault(e["label"], []).append(e)
        for lbl in by_label:
            rng.shuffle(by_label[lbl])
        pools = {lbl: list(v) for lbl, v in by_label.items()}

        avail = len(examples)
        n_p = min(n_pos, max(1, int(avail * 0.6)))
        n_b = min(n_boundary, max(0, int(avail * 0.2)))
        n_n = min(n_neg, max(0, avail - n_p - n_b))

        def _draw(target: int) -> list[dict]:
            out: list[dict] = []
            while target > 0 and any(pools.values()):
                progressed = False
                for lbl in list(pools.keys()):
                    if pools[lbl] and target > 0:
                        out.append(pools[lbl].pop())
                        target -= 1
                        progressed = True
                if not progressed:
                    break
            return out

        pos = _draw(n_p)
        boundary = _draw(n_b)
        neg = _draw(n_n)
        return EvalSet(pos=pos, neg=neg, boundary=boundary, task_type=task_type)

    if task_type == "classification":
        pos_label = _infer_pos_label(examples)
        neg_label = _infer_neg_label(examples, pos_label)
        pos_examples = [e for e in examples if e["label"] == pos_label]
        neg_examples = [e for e in examples if e["label"] == neg_label]

        # Select boundary examples: negative-class examples whose text length is
        # closest to the positive-class mean length (length similarity is a proxy
        # for confusability at the decision boundary). Also include short positive
        # examples as secondary candidates. This is task-agnostic — no hardcoded
        # keywords. (Paper §2.5 Eq. 7: E_boundary = confusable pairs.)
        pos_lengths = [len(e.get("text", "")) for e in pos_examples]
        mean_pos_len = sum(pos_lengths) / max(len(pos_lengths), 1)
        scored_neg = sorted(
            neg_examples,
            key=lambda e: abs(len(e.get("text", "")) - mean_pos_len),
        )
        boundary_candidates = scored_neg[:n_boundary * 2]
        short_pos = [e for e in pos_examples if len(e.get("text", "")) < mean_pos_len * 0.6]
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
