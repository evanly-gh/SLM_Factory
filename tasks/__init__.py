"""The benchmark task registry — the single place a task is defined.

Ten tasks in three categories:

    in_distribution      gsm8k                 DialogSum + SAMSum       sms_spam
    format_bound         xlam_bfcl             calendar_json            ner_bc5cdr
                         toolbench
    out_of_distribution  routerbench           proactive_listening      clinc150

Adding an eleventh means writing one module here and registering it below. Nothing else in the
pipeline dispatches on task identity, so there is no second place to remember — which is the
whole point: the previous design needed five separate side registries kept in sync by hand, and
every one of them was added after a bug caused by their being out of sync.
"""
from __future__ import annotations

from tasks.spec import MiningSource, TaskSpec, spec_field_names

from tasks import (
    calendar_json,
    clinc150,
    dialogsum,
    gec_bea19,
    goemotions,
    gsm8k,
    multiconer,
    ner_bc5cdr,
    proactive_listening,
    routerbench,
    sms_spam,
    toolbench,
    topv2,
    xlam_bfcl,
)

_MODULES = (
    gsm8k,
    dialogsum,
    sms_spam,
    xlam_bfcl,
    calendar_json,
    toolbench,
    ner_bc5cdr,
    routerbench,
    proactive_listening,
    clinc150,
    topv2,
    multiconer,
    gec_bea19,
    goemotions,
)

TASKS: dict[str, TaskSpec] = {}
for _module in _MODULES:
    _spec = _module.SPEC
    if _spec.name in TASKS:
        raise RuntimeError(f"duplicate task name in registry: {_spec.name!r}")
    if _spec.name != _module.__name__.rsplit(".", 1)[-1]:
        # The module name IS the registry key everywhere else (slurm scripts, SLM_BENCHMARK_TASK,
        # log lines). Letting them drift would reintroduce exactly the kind of two-places-to-update
        # bookkeeping this registry replaces.
        raise RuntimeError(
            f"task module {_module.__name__} declares name {_spec.name!r}; they must match"
        )
    TASKS[_spec.name] = _spec

del _module, _spec


def get_task(name: str) -> TaskSpec:
    """The spec for `name`, or a ValueError naming every task that does exist."""
    key = str(name or "").strip().lower()
    try:
        return TASKS[key]
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; the registry holds {sorted(TASKS)}"
        ) from None


def task_names() -> list[str]:
    return sorted(TASKS)


def tasks_in_category(category: str) -> list[TaskSpec]:
    return [spec for spec in TASKS.values() if spec.category == category]


__all__ = [
    "MiningSource",
    "TASKS",
    "TaskSpec",
    "get_task",
    "spec_field_names",
    "task_names",
    "tasks_in_category",
]
