# data/eval_set.py
import random
from dataclasses import dataclass, field


@dataclass
class EvalSet:
    pos: list[dict]       # Epos: genuine spam, full surface-form range
    neg: list[dict]       # Eneg: legitimate messages including borderline ham
    boundary: list[dict]  # Eboundary: confusable pairs at spam/ham boundary
    all: list[dict] = field(init=False)

    def __post_init__(self):
        self.all = self.pos + self.neg + self.boundary


def build_eval_set(
    examples: list[dict],
    n_pos: int = 40,
    n_neg: int = 40,
    n_boundary: int = 20,
    seed: int = 42,
) -> EvalSet:
    """
    Build E = Epos ∪ Eneg ∪ Eboundary from test examples.

    Epos: clear spam examples (high confidence)
    Eneg: clear ham examples (high confidence)
    Eboundary: short promotional ham that could be confused with spam

    All slices are disjoint. Total size = n_pos + n_neg + n_boundary.
    """
    rng = random.Random(seed)
    spam = [e for e in examples if e["label"] == "spam"]
    ham = [e for e in examples if e["label"] == "ham"]

    # Eboundary: messages at the spam/ham boundary — either ham containing
    # spam-like keywords (promotional ham that looks spammy) or short spam
    # messages with minimal spam signals (look almost legitimate).
    boundary_keywords = {"www", "http", "free", "win", "prize", "offer",
                         "click", "call", "urgent", "limited", "txt"}

    # Promotional ham: ham containing spam-like keywords
    promo_ham = [
        e for e in ham
        if any(kw in e["text"].lower() for kw in boundary_keywords)
    ]
    # Short spam: spam messages that are brief and could be confused with ham
    short_spam = sorted(
        [e for e in spam if any(kw in e["text"].lower() for kw in boundary_keywords)],
        key=lambda e: len(e["text"]),
    )

    # Prefer promotional ham; fill remainder with short confusable spam
    boundary_candidates = promo_ham + short_spam
    rng.shuffle(boundary_candidates)
    boundary = boundary_candidates[:n_boundary]
    boundary_texts = {e["text"] for e in boundary}

    # Eneg: clear ham (not in boundary)
    clear_ham = [e for e in ham if e["text"] not in boundary_texts]
    rng.shuffle(clear_ham)
    neg = clear_ham[:n_neg]

    # Epos: clear spam (exclude items already in boundary)
    clear_spam = [e for e in spam if e["text"] not in boundary_texts]
    rng.shuffle(clear_spam)
    pos = clear_spam[:n_pos]

    return EvalSet(pos=pos, neg=neg, boundary=boundary)
