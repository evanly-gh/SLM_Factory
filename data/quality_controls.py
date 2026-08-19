"""Quality-control steps, as named units a task composes explicitly.

WHY THIS EXISTS
    Quality control used to be one `if task_type == ...` chain in `data/curriculum.py` ending in
    `else: return dataset`. Measured on 2026-08-18, that meant **four of the eight benchmark tasks
    received no quality control at all**, silently:

      * `function_call` (xlam_bfcl, calendar_json) fell into the `else` and was returned untouched;
      * `math_reasoning` (gsm8k) and `generation` (dialogsum_samsum) entered their branch but
        filtered length and duplicates on a `"prompt"` key their rows do not have — a row of
        100,000 characters survived, and nothing was logged.

    Only `classification` and `NER` were actually filtered. Nothing distinguished "this task chose
    not to deduplicate" from "this task fell through a branch nobody updated".

THE RULE
    A task lists the steps it wants, in order, in its `TaskSpec.quality_controls`. An empty tuple is
    a legal and explicit choice; falling through a branch is not possible because there is no
    branch. Every step reports what it removed and why, so a curriculum that shrinks says so.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from collections import Counter
from dataclasses import dataclass, field

# Row provenances that define the task's true length distribution: real rows from the benchmark's
# own train split. Mined and synthesized rows do not — a teacher that rambles would otherwise raise
# the median and license its own outliers (B260).
TRUSTED_LENGTH_PROVENANCE = frozenset({"train_anchor", "resample"})


@dataclass
class QCContext:
    """Everything a step may need beyond the rows themselves."""

    task_name: str
    allowed_labels: set[str] | None = None
    log: Callable[[str], None] | None = field(default=None)

    def report(self, step: str, before: int, after: int, reason: str) -> None:
        if self.log and after < before:
            self.log(f"      [qc] {step}: removed {before - after} row(s) — {reason}")


# A step takes the rows and the context and returns the rows it keeps.
QCStep = Callable[[list[dict], QCContext], list[dict]]


def _text_of(row: dict, key: str) -> str:
    return str(row.get(key) or "")


def require_fields(*names: str) -> QCStep:
    """Drop rows missing any required field.

    Named explicitly per task rather than inferred, because the inference was wrong: the
    generation-family branch accepted a row on `("text" and "label")` and then filtered it on
    `"prompt"`, so every row passed the gate and none was measurable.
    """

    def step(rows: list[dict], ctx: QCContext) -> list[dict]:
        kept = [r for r in rows if isinstance(r, dict) and all(n in r for n in names)]
        ctx.report(
            "schema", len(rows), len(kept),
            f"row missing one of the required field(s) {', '.join(names)}",
        )
        return kept

    step.__name__ = f"require_fields({', '.join(names)})"
    return step


def label_space(strict: bool = True) -> QCStep:
    """Drop rows whose label is outside the task's closed vocabulary.

    A row whose label cannot appear in the eval set can never be scored against it, so it is pure
    training noise. This is where mined sources carrying raw integer class ids get removed
    (B222/B229). No-op when the task has no closed vocabulary.
    """

    def step(rows: list[dict], ctx: QCContext) -> list[dict]:
        if not ctx.allowed_labels:
            return rows
        rejected = Counter(
            str(r.get("label")) for r in rows if str(r.get("label")) not in ctx.allowed_labels
        )
        kept = [r for r in rows if str(r.get("label")) in ctx.allowed_labels]
        if rejected and ctx.log:
            sample = ", ".join(f"{lab!r}x{cnt}" for lab, cnt in rejected.most_common(5))
            ctx.log(
                f"      [qc] label-space: removed {len(rows) - len(kept)} row(s) — label not in "
                f"the task's {len(ctx.allowed_labels)} established classes [{sample}]"
            )
        return kept

    return step


def balance_labels(max_ratio: int = 3) -> QCStep:
    """Cap any class at `max_ratio` × the smallest class."""

    def step(rows: list[dict], ctx: QCContext) -> list[dict]:
        counts = Counter(r.get("label") for r in rows)
        if not counts:
            return rows
        min_count = max(min(counts.values()), 1)
        cap = max_ratio * min_count
        kept: list[dict] = []
        seen: Counter = Counter()
        for row in rows:
            if seen[row.get("label")] < cap:
                kept.append(row)
                seen[row.get("label")] += 1
        ctx.report(
            "label-balance", len(rows), len(kept),
            f"label over the cap of {max_ratio}x the smallest class ({cap} rows/label; "
            f"smallest class has {min_count})",
        )
        return kept

    return step


def length_outliers(key: str = "text", max_ratio: float = 3.0) -> QCStep:
    """Drop rows whose `key` is longer than `max_ratio` × the trusted median.

    The median is computed over TRUSTED rows only — the task's own real training data — so a
    verbose teacher cannot raise the bar and license its own outliers (B260). The `key` is stated
    by the task; guessing it is what made this a no-op for gsm8k and dialogsum.
    """

    def step(rows: list[dict], ctx: QCContext) -> list[dict]:
        if not rows:
            return rows
        if not any(_text_of(r, key) for r in rows):
            # Nothing to measure against. Say so — a silent pass here is precisely the bug this
            # module exists to prevent, and it means the task named a key its rows do not carry,
            # which is exactly how gsm8k and dialogsum went unfiltered on a `"prompt"` key.
            if ctx.log:
                ctx.log(
                    f"      [qc] length-outlier: SKIPPED — no row in {ctx.task_name} carries a "
                    f"non-empty {key!r} field, so no median could be computed. Check the task's "
                    "quality_controls against its row schema."
                )
            return rows
        trusted = [
            r for r in rows
            if str(r.get("_provenance") or "") in TRUSTED_LENGTH_PROVENANCE
        ]
        basis = trusted or rows
        lengths = sorted(len(_text_of(r, key)) for r in basis)
        median = lengths[len(lengths) // 2] or 1
        cutoff = median * max_ratio
        kept = [r for r in rows if len(_text_of(r, key)) <= cutoff]
        ctx.report(
            "length-outlier", len(rows), len(kept),
            f"{key} longer than {max_ratio:g}x the median ({cutoff:.0f} chars)",
        )
        return kept

    return step


def dedup_surface(key: str = "text", threshold: float = 0.9) -> QCStep:
    """Drop near-duplicate rows (word-set Jaccard at or above `threshold`)."""

    def step(rows: list[dict], ctx: QCContext) -> list[dict]:
        kept: list[dict] = []
        seen: list[set[str]] = []
        for row in rows:
            words = set(_text_of(row, key).lower().split())
            if not words:
                kept.append(row)
                continue
            # Compared against a sliding window of recent rows, not the whole set: the curriculum
            # runs to thousands of rows and the full pairwise comparison is quadratic.
            duplicate = False
            for previous in seen[-50:]:
                union = len(words | previous)
                if union and len(words & previous) / union >= threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(row)
                seen.append(words)
        ctx.report(
            "surface-dedup", len(rows), len(kept),
            f"near-duplicate {key} (Jaccard >= {threshold:g})",
        )
        return kept

    return step


def entity_diversity(cap: int = 3) -> QCStep:
    """Cap how often any single entity surface form may appear across the curriculum."""

    def step(rows: list[dict], ctx: QCContext) -> list[dict]:
        totals: Counter = Counter()
        for row in rows:
            for entity in row.get("entities") or []:
                if isinstance(entity, dict):
                    totals[str(entity.get("text", "")).lower()] += 1
        over = {value for value, count in totals.items() if count > cap}
        if not over:
            return rows
        kept: list[dict] = []
        running: Counter = Counter()
        for row in rows:
            values = [
                str(e.get("text", "")).lower()
                for e in (row.get("entities") or []) if isinstance(e, dict)
            ]
            if any(v in over and running[v] >= cap for v in values):
                continue
            kept.append(row)
            for value in values:
                running[value] += 1
        ctx.report(
            "entity-diversity", len(rows), len(kept),
            f"entity surface form already present {cap} times",
        )
        return kept

    return step


def valid_json_answer() -> QCStep:
    """Drop rows whose `answer` is not parseable JSON.

    Format-bound tasks train the model to emit a JSON payload; a row whose own gold does not parse
    teaches it to emit something the scorer will mark wrong. Cheap to check, and it had never run
    on xlam or calendar because those tasks reached the `else` branch.
    """
    import json

    def step(rows: list[dict], ctx: QCContext) -> list[dict]:
        kept: list[dict] = []
        for row in rows:
            answer = row.get("answer")
            if isinstance(answer, (list, dict)):
                kept.append(row)
                continue
            try:
                json.loads(str(answer))
            except (TypeError, ValueError):
                continue
            kept.append(row)
        ctx.report("json-answer", len(rows), len(kept), "gold answer is not parseable JSON")
        return kept

    return step


def apply_quality_controls(
    rows: list[dict],
    steps: Iterable[QCStep],
    *,
    task_name: str,
    allowed_labels: set[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run a task's declared steps in order."""
    ctx = QCContext(task_name=task_name, allowed_labels=allowed_labels, log=log)
    current = list(rows)
    for step in steps:
        current = step(current, ctx)
    return current
