"""The task registry is the single place a task is defined, and it must be complete.

WHY THIS FILE EXISTS
    The pipeline used to route eight concrete benchmarks through five abstract `task_type`
    channels, so behaviour was decided by `if task_type == ...` chains and a task inherited
    whatever the chain's `else` branch happened to do. Three production bugs came from exactly
    that (B291 synthesis returned `[]` for `function_call`, B296 every failure was one constant
    confusion pair, B299 four of eight tasks got no quality control at all).

    `tasks/spec.py` fixes it structurally: no field on `TaskSpec` has a default, so Python refuses
    to construct a spec that has not stated every decision. This file is the guard on the OTHER
    half of that promise — that the dataclass really has no defaults, that all eight tasks are
    registered, that the caps are uniform rather than per-task folklore, and above all that
    `family` never becomes a dispatch key again. `tasks/spec.py`'s own module docstring names this
    file as the thing that enforces the last point.
"""
from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from tasks import TASKS, get_task, spec_field_names, task_names, tasks_in_category
from tasks.spec import CATEGORIES, EVAL_SAMPLING, FAMILIES, TaskSpec

# The suite, as documented in `tasks/__init__.py`. Written out rather than derived from the
# registry, so deleting a task fails here instead of silently shrinking every other assertion.
EXPECTED_TASKS = {
    "gsm8k",
    "dialogsum",
    "xlam_bfcl",
    "calendar_json",
    "ner_bc5cdr",
    "routerbench",
    "proactive_listening",
    "clinc150",
}

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PRODUCTION_PACKAGES = ("agent", "data", "eval", "training", "tasks", "config")


# --------------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------------


def test_all_eight_tasks_are_registered():
    assert set(TASKS) == EXPECTED_TASKS
    assert task_names() == sorted(EXPECTED_TASKS)


def test_no_spec_field_has_a_default():
    """The whole mechanism. A default is a decision nobody made, applied silently.

    With no defaults, "we never considered this for that task" is an import-time TypeError
    instead of a runtime fallthrough three hours into a run.
    """
    defaulted = [
        field.name
        for field in dataclasses.fields(TaskSpec)
        if field.default is not dataclasses.MISSING
        or field.default_factory is not dataclasses.MISSING
    ]
    assert defaulted == [], (
        f"TaskSpec fields {defaulted} have defaults, so a task can omit them and inherit a "
        "decision nobody made for it — the exact shape of B291/B296/B299"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_TASKS))
def test_every_task_declares_every_field(name):
    """Constructing the spec already proves this; reading each field proves the ATTRIBUTE exists
    under the name the rest of the pipeline uses, which a rename would otherwise break at the
    call site rather than here."""
    spec = get_task(name)
    for field in spec_field_names():
        assert hasattr(spec, field), f"{name} is missing {field}"


def test_registry_key_equals_the_module_name():
    """The module name IS the registry key everywhere else — slurm script suffixes,
    `SLM_BENCHMARK_TASK`, log lines. Two places to update is what the registry replaced."""
    for name, spec in TASKS.items():
        assert spec.name == name
        assert (PROJECT_ROOT / "tasks" / f"{name}.py").is_file(), (
            f"{name} is registered but tasks/{name}.py does not exist"
        )


def test_every_task_module_is_registered():
    """The reverse direction: a module nobody added to `_MODULES` is a task that exists in the
    tree, imports cleanly, and can never be selected."""
    modules = {
        path.stem
        for path in (PROJECT_ROOT / "tasks").glob("*.py")
        if not path.stem.startswith("_") and path.stem != "spec"
    }
    assert modules == EXPECTED_TASKS


def test_categories_partition_the_suite():
    covered = [spec.name for category in CATEGORIES for spec in tasks_in_category(category)]
    assert sorted(covered) == sorted(EXPECTED_TASKS)


def test_an_unknown_task_names_every_task_that_does_exist():
    """An unknown task used to fall through to a bare `return []`. The error message carrying the
    registry is what turns "nothing happened" into a diagnosis."""
    with pytest.raises(ValueError, match=r"unknown task 'not_a_task'.*registry holds"):
        get_task("not_a_task")


# --------------------------------------------------------------------------
# Uniform caps
# --------------------------------------------------------------------------


def test_the_data_caps_are_uniform_across_the_suite():
    """`initial_train_cap` is a CAP, not a target and not a fraction of anything.

    It used to be `curriculum_size_target x 0.65`, which is where the mystery 3,250 came from:
    a 5,000-row "target" the curriculum was then never allowed to reach. Every task now asks its
    loader for the same 5,000 and takes as many as the loader has.
    """
    for name, spec in TASKS.items():
        assert spec.initial_train_cap == 5000, name
        assert spec.eval_cap == 1000, name


def test_no_task_declares_a_train_fraction():
    """The removed field. Its absence is worth asserting because its presence was invisible —
    a fraction silently reinterpreted a cap as a target."""
    assert "train_fraction" not in spec_field_names()


def test_token_budgets_leave_room_for_a_prompt():
    """`__post_init__` checks this at import; asserting it here documents WHY the check exists —
    a reserve at or above the context window leaves no prompt budget and every eval row truncates."""
    for name, spec in TASKS.items():
        assert 0 < spec.max_new_tokens < spec.max_seq_length, name
        assert spec.eval_batch_size >= 1, name


# --------------------------------------------------------------------------
# Names a human has to tell apart in a log
# --------------------------------------------------------------------------


def _tokens(name: str) -> tuple[str, ...]:
    return tuple(name.replace("-", "_").split("_"))


def test_no_two_task_names_are_confusable():
    """`fill-synth` and `synth-fill` were one transposition apart and named different mechanisms,
    one of which had already been deleted — so the log appeared to be running something that no
    longer existed. Any two names that are the same tokens in a different order, or differ by a
    single character, are indistinguishable at a glance in a log line.
    """
    names = sorted(TASKS)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert sorted(_tokens(first)) != sorted(_tokens(second)), (
                f"{first!r} and {second!r} are the same tokens reordered"
            )
            assert _levenshtein(first, second) > 1, (
                f"{first!r} and {second!r} differ by one character"
            )


def test_no_two_rebuild_kind_names_are_confusable():
    """The same property for the names a rebuild reports for the mechanism that produced its rows.
    This is the actual `fill-synth`/`synth-fill` regression."""
    from agent.nodes.curate import REBUILD_KIND_NAMES

    display = sorted(set(REBUILD_KIND_NAMES.values()))
    for i, first in enumerate(display):
        for second in display[i + 1:]:
            assert sorted(_tokens(first)) != sorted(_tokens(second)), (
                f"rebuild kinds {first!r} and {second!r} are the same tokens reordered"
            )


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------
# `family` is descriptive, not a dispatch key
# --------------------------------------------------------------------------


def _production_sources():
    for package in PRODUCTION_PACKAGES:
        for path in (PROJECT_ROOT / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_family_is_never_used_to_choose_behaviour():
    """`family` survives only as a report label and a model-selection prompt hint.

    It is the last remnant of the abstract channel, and the danger is precise: the moment a
    behavioural choice reads it, two tasks sharing a family start inheriting each other's
    behaviour again. So `spec.family` may be looked up in a hint table, but it may never appear
    in a comparison or a branch condition.
    """
    offenders: list[str] = []
    for path in _production_sources():
        if path.name == "spec.py" and path.parent.name == "tasks":
            continue  # the validator that pins `family` to FAMILIES lives here
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and _mentions_family(node):
                offenders.append(f"{path}:{node.lineno} compares on .family")
            elif isinstance(node, (ast.If, ast.IfExp)) and _mentions_family(node.test):
                offenders.append(f"{path}:{node.lineno} branches on .family")
    assert offenders == [], (
        "family must never be a dispatch key again:\n  " + "\n  ".join(offenders)
    )


def _mentions_family(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr == "family"
        for child in ast.walk(node)
    )


def test_the_descriptive_tags_are_from_their_declared_vocabularies():
    for name, spec in TASKS.items():
        assert spec.category in CATEGORIES, name
        assert spec.family in FAMILIES, name
        assert spec.eval_sampling in EVAL_SAMPLING, name
        assert spec.title.strip(), name


# --------------------------------------------------------------------------
# Choices that must be stated, not implied
# --------------------------------------------------------------------------


def test_label_definitions_only_exist_where_there_is_a_label_space():
    """A gloss for a class that is not a class is prompt noise the teacher will try to obey."""
    for name, spec in TASKS.items():
        if spec.label_definitions:
            assert spec.closed_label_space, name
        if not spec.closed_label_space:
            assert spec.label_definitions == {}, name


def test_required_fields_are_the_fields_the_scorer_reads():
    """A task whose `required_fields` omit its own target accepts rows it cannot score — the shape
    of the generation-family bug where rows passed a `(text, label)` gate and were then filtered on
    a `"prompt"` key they never carried (B299)."""
    targets = {
        "gsm8k": "answer",
        "dialogsum": "answer",
        "xlam_bfcl": "answer",
        "calendar_json": "answer",
        "ner_bc5cdr": "entities",
        "routerbench": "label",
        "proactive_listening": "label",
        "clinc150": "label",
    }
    for name, target in targets.items():
        spec = get_task(name)
        assert "text" in spec.required_fields, name
        assert target in spec.required_fields, (
            f"{name} does not require {target!r}, the field its scorer grades"
        )


def test_a_judged_task_says_so_and_is_the_only_one():
    """`needs_judge` exists so a judge outage fails loudly instead of scoring zero and sending the
    loop chasing a phantom regression. Exactly one task scores through the judge."""
    judged = {name for name, spec in TASKS.items() if spec.needs_judge}
    assert judged == {"dialogsum"}
    for name, spec in TASKS.items():
        # Overlapping the judge with the next generation batch is only meaningful when there is
        # a judge to overlap.
        if spec.judge_overlap:
            assert spec.needs_judge, name


def test_metric_names_say_what_the_number_is():
    """RouterBench and CLINC150 were both `classification` and both reported `macro_f1`, but
    RouterBench's headline has always been a minority-class F1. The task now names which."""
    assert get_task("clinc150").metric_name == "macro_f1"
    assert get_task("routerbench").metric_name == "minority_f1"
    assert get_task("proactive_listening").metric_name == "minority_f1"
    for name, spec in TASKS.items():
        assert spec.metric_name.strip(), name


def test_the_metric_name_is_read_from_the_spec_not_a_side_table():
    """`TASK_METRIC_NAMES` was one of five hand-maintained side registries the specs replaced."""
    from eval.harness import task_metric_name

    for name, spec in TASKS.items():
        assert task_metric_name(name) == spec.metric_name

    import eval.harness as harness

    for gone in ("TASK_METRIC_NAMES", "TASK_REQUIRED_FIELDS", "NAMED_BENCHMARK_TASK_TYPES"):
        assert not hasattr(harness, gone), f"{gone} is back"


def test_mining_sources_are_declared_where_a_corpus_exists():
    """Without a declared source, `acquire` can only pay Exa to rediscover mirrors of a corpus
    already in the local cache while tens of thousands of unused rows stay unreachable (B297)."""
    for name, spec in TASKS.items():
        assert spec.mining_sources, f"{name} declares no mining source"
        for source in spec.mining_sources:
            assert source.hf_id and source.split and source.url, name
