# data/eval_set.py
"""The frozen held-out eval set.

Carries the TASK NAME (`xlam_bfcl`, `clinc150`, ...), not an abstract `task_type` channel. The
channel was what let two tasks sharing a type receive each other's behaviour; every consumer now
resolves `tasks.get_task(eval_set.task)` and reads the decision it needs from that task's spec.
"""
import random
from dataclasses import dataclass


@dataclass
class EvalSet:
    """A single flat sample of held-out examples (`all`), tagged with the task that owns them.

    The eval set used to carry pos/neg/boundary slices, but they had no functional effect — every
    consumer used the `all` union, and the difficulty stratification (easy/medium/hard) is the real
    difficulty signal. They were removed 2026-08-02; `all` is the sole content.
    """

    all: list[dict]
    task: str

    def __post_init__(self):
        from tasks import get_task

        # Resolving here means an eval set can never exist for a task the registry does not know,
        # so no downstream consumer has to handle that case.
        get_task(self.task)

    @property
    def spec(self):
        from tasks import get_task

        return get_task(self.task)

    @classmethod
    def from_serialized(cls, d: dict) -> "EvalSet":
        """Rebuild from a serialized dict, folding legacy pos/neg/boundary slices into `all`."""
        rows = d.get("all")
        if rows is None:
            rows = (
                list(d.get("pos") or [])
                + list(d.get("neg") or [])
                + list(d.get("boundary") or [])
            )
        task = d.get("task")
        if not task:
            # Checkpoints written before 2026-08-18 stored an abstract `task_type`, which does not
            # identify a task: `function_call` was both xlam_bfcl and calendar_json, and
            # `classification` was three tasks. Guessing would silently evaluate one task with
            # another's configuration, so refuse.
            legacy = d.get("task_type")
            raise ValueError(
                "this eval set was written before the task registry existed and records only "
                f"task_type={legacy!r}, which does not identify a task (several tasks shared each "
                "type). Start a fresh run rather than resuming this checkpoint."
            )
        return cls(all=list(rows), task=str(task))


def build_eval_set(
    examples: list[dict],
    task: str,
    target: int = 100,
    seed: int = 42,
) -> EvalSet:
    """Build the held-out eval set as a sample of up to `target` rows.

    Sampling strategy is the task's own `eval_sampling` choice. `label_balanced` round-robins
    across classes so every class appears even when the target is smaller than the pool — which
    matters enormously for CLINC150's 151 classes and not at all for a task with no classes.
    """
    from tasks import get_task

    spec = get_task(task)
    rng = random.Random(seed)
    target = max(int(target), 0)

    if spec.eval_sampling == "label_balanced":
        by_label: dict[str, list[dict]] = {}
        for example in examples:
            by_label.setdefault(example["label"], []).append(example)
        for pool in by_label.values():
            rng.shuffle(pool)

        out: list[dict] = []
        remaining = target
        while remaining > 0 and any(by_label.values()):
            progressed = False
            for label in list(by_label):
                if by_label[label] and remaining > 0:
                    out.append(by_label[label].pop())
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return EvalSet(all=out, task=task)

    shuffled = list(examples)
    rng.shuffle(shuffled)
    return EvalSet(all=shuffled[:target], task=task)
