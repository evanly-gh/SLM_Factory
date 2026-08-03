# data/eval_set.py
import random
from dataclasses import dataclass

# Canonical task types. See task_analysis.py for the full rationale behind each.
# multi_label is a flag on classification, not a separate type here —
# only the scorer and metric differ.
TASK_TYPES = {
    "classification", "NER", "math_reasoning", "code_generation", "generation",
    # Format-bound types (2026-08-01): executable, judge-free verifiers.
    #   function_call — BFCL-style AST argument match ({text, answer=gold call JSON}).
    #   diff          — prose edit as a unified diff, verified with `git apply --check`.
    "function_call", "diff",
}

# Flags that travel alongside task_type as separate state fields:
#   multi_label: bool  — set by planner; changes scorer from argmax to per-label threshold
#   schema: dict|None  — set by planner for structured_extraction; changes eval to field-F1
#   multilingual: bool — set by planner; changes eval to target-language metrics


@dataclass
class EvalSet:
    """The frozen held-out eval set. A single flat sample of examples (`all`).

    The eval set used to carry pos/neg/boundary slices, but they had no functional effect —
    every consumer used the `all` union, the difficulty stratification (easy/medium/hard) is the
    real difficulty signal, and for the NER/generation families the slices were a meaningless
    random partition. They were removed 2026-08-02; `all` is the sole content.
    """
    all: list[dict]
    task_type: str
    # Optional flags passed through from the task plan
    multi_label: bool = False
    schema: dict | None = None
    multilingual: bool = False

    def __post_init__(self):
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, got {self.task_type!r}")

    @classmethod
    def from_serialized(cls, d: dict) -> "EvalSet":
        """Rebuild from a serialized dict, folding legacy pos/neg/boundary into `all`
        for backward compatibility with checkpoints/artifacts written before the slices
        were removed."""
        rows = d.get("all")
        if rows is None:
            rows = list(d.get("pos") or []) + list(d.get("neg") or []) + list(d.get("boundary") or [])
        return cls(
            all=list(rows),
            task_type=d["task_type"],
            multi_label=d.get("multi_label", False),
            schema=d.get("schema"),
            multilingual=d.get("multilingual", False),
        )


def build_eval_set(
    examples: list[dict],
    task_type: str,
    target: int = 100,
    seed: int = 42,
    multi_label: bool = False,
    schema: dict | None = None,
    multilingual: bool = False,
) -> EvalSet:
    """Build the held-out eval set as a single sample of up to `target` rows.

    Multi-class classification keeps label-coverage stratification (round-robin across every
    label) so the eval set spans the full label range; every other task type is a plain
    shuffled top-N sample.
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}")

    rng = random.Random(seed)
    target = max(int(target), 0)

    if task_type == "classification" and len({e["label"] for e in examples}) > 2:
        # Multi-class: round-robin across labels so the eval set covers every class even when
        # the target is smaller than the pool. Binary pos/neg has no meaning with >2 classes.
        by_label: dict[str, list[dict]] = {}
        for e in examples:
            by_label.setdefault(e["label"], []).append(e)
        for lbl in by_label:
            rng.shuffle(by_label[lbl])
        pools = {lbl: list(v) for lbl, v in by_label.items()}

        out: list[dict] = []
        remaining = target
        while remaining > 0 and any(pools.values()):
            progressed = False
            for lbl in list(pools.keys()):
                if pools[lbl] and remaining > 0:
                    out.append(pools[lbl].pop())
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return EvalSet(all=out, task_type=task_type,
                       multi_label=multi_label, schema=schema, multilingual=multilingual)

    examples = list(examples)
    rng.shuffle(examples)
    return EvalSet(all=examples[:target], task_type=task_type,
                   multi_label=multi_label, schema=schema, multilingual=multilingual)
