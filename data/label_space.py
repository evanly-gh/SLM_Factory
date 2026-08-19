"""The task's label vocabulary, pinned once and never extended.

WHY THIS EXISTS (B259)
    RouterBench is a two-class task: `local` and `route`. Its runs trained on rows labelled
    `cloud`, `on_device`, `router` and `remote` as well — classes that do not exist in the task.
    They arrived because acquisition asks an LLM to map a mined dataset's columns onto our schema
    (`web_acquire._llm_map_dataset`), the returned `label_map` was applied with
    `lmap.get(str(lab), lab)` so **unmapped values passed through verbatim**, and the mapping was
    re-requested per acquire round and is non-deterministic — so each round could mint a new
    hallucinated class. The out-of-vocabulary set grew monotonically through the run: `cloud`
    only for iterations 1-9, `+on_device` at 10, `+router` at 20, `+remote` at 21.

    The guard that should have caught it accepted a source on ANY overlap with the known labels,
    so a mapping that got *some* rows right admitted all of them. Quality control then deleted
    ~1,166 rows per rebuild for the rest of the run, and the mined rows that survived — 34% of the
    final curriculum — carried invented `local`/`route` labels at a 50/50 rate against the task's
    true 30/70 base rate.

THE RULE
    The label space is established ONCE from the frozen eval set, which is definitionally the set
    of classes the model will be scored against. After that it is CLOSED:

      * a row whose label is outside it is dropped, never added to the vocabulary;
      * a mined source is rejected outright unless its labels are a SUBSET of it;
      * an LLM is never permitted to introduce a label — it may only map onto existing ones, and
        anything it fails to map is discarded rather than passed through.

    "Extend the vocabulary to fit the data" is never the right move: a class absent from the eval
    set cannot be scored, so training rows carrying it are unusable by construction.
"""
from __future__ import annotations

def label_space_from_eval_set(eval_set, task: str) -> set[str] | None:
    """The closed set of classes this run can be scored against, or None if not applicable.

    Read from the frozen eval set rather than from the training pool: the eval set is what the
    score is computed against, so it is the only defensible authority on what counts as a class.
    Whether the task HAS a closed space is `TaskSpec.closed_label_space` — for a span or
    structured-output task the `label` field is a constant tag and there is nothing to police.
    """
    from tasks import get_task

    return get_task(task).qc_context_labels(eval_set)


def partition_rows_by_label(
    rows: list[dict],
    allowed: set[str] | None,
) -> tuple[list[dict], dict[str, int]]:
    """Split rows into (in-vocabulary, rejected-label -> count).

    Per-ROW, not per-source. The previous guard was per-source and passed the whole thing on any
    overlap, which is exactly how the hallucinated classes got in alongside legitimate rows.
    """
    if not allowed:
        return list(rows), {}
    kept: list[dict] = []
    rejected: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        if label is None:
            kept.append(row)
            continue
        key = str(label)
        if key in allowed:
            kept.append(row)
        else:
            rejected[key] = rejected.get(key, 0) + 1
    return kept, rejected


def describe_rejected_labels(rejected: dict[str, int], limit: int = 5) -> str:
    """`'cloud'x688, 'on_device'x269, …` — the top offenders with counts, for a log line."""
    if not rejected:
        return ""
    ordered = sorted(rejected.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{label!r}x{count}" for label, count in ordered[:limit])
    if len(ordered) > limit:
        shown += f", … and {len(ordered) - limit} more"
    return shown


def sanitize_label_map(
    label_map: dict | None,
    allowed: set[str] | None,
) -> tuple[dict[str, str], list[str]]:
    """Keep only LLM label-map entries that target a real class.

    Returns ``(clean_map, dropped_targets)``. An entry whose *target* is not in the task's label
    space is a hallucinated class, and honouring it is how `cloud` entered a two-class task.
    """
    if not isinstance(label_map, dict) or not label_map:
        return {}, []
    if not allowed:
        return {str(k): str(v) for k, v in label_map.items()}, []
    clean: dict[str, str] = {}
    dropped: list[str] = []
    for raw, target in label_map.items():
        if str(target) in allowed:
            clean[str(raw)] = str(target)
        else:
            dropped.append(str(target))
    return clean, sorted(set(dropped))


def label_definitions_for(task: str | None) -> dict[str, str]:
    """What each label MEANS for a task, for prompting the teacher.

    The synthesis and verification prompts previously showed the teacher only the label STRING,
    so it read the label as an English word. For RouterBench that is catastrophic: `local` means
    "a small on-device model answers this correctly", but the teacher read it as "a local-
    information query" and rejected a grade-school math problem with *"the utterance is a math
    problem, not a local query"* — 70% of generated rows discarded for the wrong reason. The
    teacher was not being strict; it was answering a different question than the task asks.

    Absent an entry, prompts fall back to naming the label alone (correct for intent-style tasks
    such as CLINC150, where the label really is a property of the utterance's meaning).
    """
    from tasks import TASKS

    spec = TASKS.get(str(task or ""))
    return dict(spec.label_definitions) if spec else {}


# The definitions themselves now live on each task's spec (`TaskSpec.label_definitions`), next to
# every other decision about that task, rather than in a benchmark-keyed table three modules away
# that had to be remembered separately when a task was added.
