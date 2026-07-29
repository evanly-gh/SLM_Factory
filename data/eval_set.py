# data/eval_set.py
import random
from dataclasses import dataclass, field

# Canonical task types. See task_analysis.py for the full rationale behind each.
# multi_label is a flag on classification, not a separate type here —
# the eval split logic is the same; only the scorer and metric differ.
TASK_TYPES = {"classification", "NER", "math_reasoning", "code_generation", "generation"}

# Types that map to the classification eval split (label-based partitioning).
_CLASSIFICATION_FAMILY = {"classification"}

# Types that map to the NER eval split (entity/schema-based partitioning).
_NER_FAMILY = {"NER"}

# Types that map to the generation eval split (prompt/response partitioning).
_GENERATION_FAMILY = {"math_reasoning", "code_generation", "generation"}

# Flags that travel alongside task_type as separate state fields:
#   multi_label: bool  — set by planner; changes scorer from argmax to per-label threshold
#   schema: dict|None  — set by planner for structured_extraction; changes eval to field-F1
#   multilingual: bool — set by planner; changes eval to target-language metrics


def _canonical_split_type(task_type: str) -> str:
    """Map a task type to its eval-split family: 'classification', 'NER', or 'generation'."""
    if task_type in _CLASSIFICATION_FAMILY:
        return "classification"
    if task_type in _NER_FAMILY:
        return "NER"
    return "generation"


@dataclass
class EvalSet:
    pos: list[dict]
    neg: list[dict]
    boundary: list[dict]
    task_type: str
    all: list[dict] = field(init=False)
    # Optional flags passed through from the task plan
    multi_label: bool = False
    schema: dict | None = None
    multilingual: bool = False

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
    multi_label: bool = False,
    schema: dict | None = None,
    multilingual: bool = False,
) -> EvalSet:
    """
    Build E = Epos ∪ Eneg ∪ Eboundary. All slices are disjoint.

    Split family is determined by _canonical_split_type(task_type):

    classification family (classification):
      pos = clear positive-class examples
      neg = clear negative-class examples
      boundary = confusable pairs at the class boundary
      [multi_label=True]: same split, scorer uses per-label thresholds not argmax

    NER family (NER):
      pos = entity-rich / schema-complete passages with gold annotations
      neg = entity-free passages or schema-empty inputs (hallucination test)
      boundary = passages with overlapping entity types or partial schema matches
      [schema set]: eval is field-level F1 not span-F1

    generation family (math_reasoning, code_generation, generation):
      pos = well-formed problems with unambiguous answers
      neg = adversarial / ill-posed inputs
      boundary = multi-step or edge-case problems
      [math_reasoning]: eval is final-answer exact match
      [code_generation]: eval is execution pass@1
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}")

    split_family = _canonical_split_type(task_type)

    rng = random.Random(seed)

    if split_family == "classification" and len({e["label"] for e in examples}) > 2:
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
        return EvalSet(pos=pos, neg=neg, boundary=boundary, task_type=task_type,
                       multi_label=multi_label, schema=schema, multilingual=multilingual)

    if split_family == "classification":
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
        boundary_texts = {e.get("text", "") for e in boundary}

        clear_neg = [e for e in neg_examples if e.get("text", "") not in boundary_texts]
        rng.shuffle(clear_neg)
        neg = clear_neg[:n_neg]

        clear_pos = [e for e in pos_examples if e.get("text", "") not in boundary_texts]
        rng.shuffle(clear_pos)
        pos = clear_pos[:n_pos]

    else:
        # NER and generation families: simple stratified split — boundary examples
        # are agent-constructed at runtime; here we just partition what we have.
        examples = list(examples)
        rng.shuffle(examples)
        total = len(examples)
        # Honor the caller's requested 40/40/20 budget whenever enough rows exist.
        # Shared-dataset preparation and live eval setup pass the same dynamic sizes.
        p = min(n_pos, total)
        n = min(n_neg, total - p)
        b = min(n_boundary, total - p - n)
        pos = examples[:p]
        neg = examples[p:p + n]
        boundary = examples[p + n:p + n + b]

    return EvalSet(pos=pos, neg=neg, boundary=boundary, task_type=task_type,
                   multi_label=multi_label, schema=schema, multilingual=multilingual)

def _infer_pos_label(examples: list[dict]) -> str:
    """Return the minority label (the positive class)."""
    from collections import Counter
    counts = Counter(e["label"] for e in examples)
    return min(counts, key=counts.get)

def _infer_neg_label(examples: list[dict], pos_label: str) -> str:
    labels = {e["label"] for e in examples}
    others = labels - {pos_label}
    return next(iter(others)) if others else pos_label
