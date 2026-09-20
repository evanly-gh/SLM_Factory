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
    "sms_spam",
    "xlam_bfcl",
    "calendar_json",
    # Added 2026-08-24: ToolBench / ToolEval pass rate, reimplementing arXiv:2512.15943.
    "toolbench",
    "ner_bc5cdr",
    "routerbench",
    "proactive_listening",
    "clinc150",
    # The on-device SFT suite, added 2026-09-06. Each one covers a capability the ten above did
    # not: nested structured prediction under a published low-resource protocol, a fine-grained
    # label space, and pure text-to-text generation scored on edits.
    "topv2",
    "multiconer",
    "gec_bea19",
    "goemotions",
}

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PRODUCTION_PACKAGES = ("agent", "data", "eval", "training", "tasks", "config")


# --------------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------------


def test_every_task_in_the_suite_is_registered():
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


# Tasks whose `initial_train_cap` is deliberately below the suite default, and why. A task belongs
# here only with a reason recorded in its own spec — the point of the assertion below is that a cap
# cannot drift quietly, not that it can never differ.
CAP_EXCEPTIONS = {
    # ~4,460 usable rows in total, so the default 5,000 consumed the entire pool at cold start and
    # `mine_new_real` had nothing to find on its first attempt. Synthesis is also refused on this task
    # by the fitness gate, so `data_rebuild` could not add a row by either route and every rebuild came
    # back empty. 3,000 leaves roughly 900 rows of mining headroom (B321/B323 follow-up).
    "calendar_json": 3000,
    # 5,159 rows survive deduplication and 20% is held out, so the train half is ~4,128. The
    # default 5,000 would take all of it at cold start and leave `mine_new_real` nothing to find —
    # the same trap calendar_json fell into. 3,000 keeps roughly 1,100 rows in reserve.
    "sms_spam": 3000,
}


def test_the_data_caps_are_uniform_across_the_suite():
    """`initial_train_cap` is a CAP, not a target and not a fraction of anything.

    It used to be `curriculum_size_target x 0.65`, which is where the mystery 3,250 came from:
    a 5,000-row "target" the curriculum was then never allowed to reach. Every task now asks its
    loader for the same 5,000 and takes as many as the loader has — except where a smaller cap is
    recorded in `CAP_EXCEPTIONS` with its reason, which keeps the uniformity meaningful instead of
    letting one task's number drift unremarked.
    """
    for name, spec in TASKS.items():
        assert spec.initial_train_cap == CAP_EXCEPTIONS.get(name, 5000), name


def test_the_in_loop_eval_is_capped_for_selection_not_for_publication():
    """`select_cap` bounds the eval that runs EVERY iteration, so it must stay small.

    The bound is `<= 1000` rather than `== 1000` because the number is a CAP and some tasks have
    less held-out data than that — DialogSum's test split is 500 rows in total — and a task with a
    smaller split is not a task that drifted.

    What must never happen is the reverse. Raising this to get a publishable error bar was the
    tempting move and it is the wrong one: the eval runs on every iteration of the loop, so the
    cost is multiplied by the iteration count, and the number still would not be the one to
    publish. Selection is a PAIRED comparison — the same fixed rows against successive
    checkpoints — so its sampling error is largely common-mode and cancels out of the ranking,
    which is what makes 1,000 defensible here and indefensible in a paper. The reporting half is
    `report_load` / `report_score`, run once by `scripts/report_eval.py`.
    """
    for name, spec in TASKS.items():
        assert spec.select_cap <= 1000, (
            f"{name}: select_cap={spec.select_cap} sizes the per-iteration eval; a bigger number "
            f"belongs in report_load, not here"
        )


def test_the_overloaded_eval_cap_field_is_gone():
    """The removed name. Worth asserting because the rename IS the fix.

    One field called `eval_cap` was sizing two things that want opposite values: checkpoint
    selection, which wants to be small because it runs every iteration, and the published result,
    which wants to be large because a +/-2.5-point error bar is not a result. Nothing in the code
    distinguished them, so the 1,000 could be defended only by ignoring one of the two jobs.
    """
    assert "eval_cap" not in spec_field_names()
    assert "select_cap" in spec_field_names()


def test_a_cap_exception_is_only_worth_having_if_it_leaves_mining_room():
    """An exception exists to give the mining ladder headroom, so it must be BELOW the default.

    A cap at or above 5,000 in that table would be a no-op wearing an explanation, which is worse than
    no exception at all — the next reader would trust the comment.
    """
    for name, cap in CAP_EXCEPTIONS.items():
        assert name in TASKS, f"{name} is not a task"
        assert cap < 5000, f"{name} exception {cap} is not below the default"


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
# The reporting half
# --------------------------------------------------------------------------


# Tasks where the SELECTION metric and the REPORTED metric are deliberately different, with the
# reason. Written out rather than derived so that a task quietly starting to publish its selection
# metric fails here — the two being different is a considered choice, and so is their being equal.
REPORT_METRIC_EXCEPTIONS = {
    # 33 classes with a 0.18%-of-entities tail. A <=1,000-row selection draw can contain ZERO
    # examples of a rare class, which makes macro-F1 undefined or wildly noisy as a ranking signal
    # while remaining the honest headline. Micro selects; macro publishes.
    "multiconer": ("micro_f1", "macro_f1"),
    # Macro-F1 over 28 labels is a thresholding artifact — the same model moves several points
    # between a fixed 0.5, a fixed 0.3 and a dev-tuned sweep — so the headline is threshold-free
    # AUPRC. AUPRC needs per-label rankings, which only the report pass computes; the loop selects
    # on the 7-way Ekman grouping, whose classes all have real support.
    "goemotions": ("ekman_macro_f1", "macro_auprc"),
    # ROUGE-L alone ranks checkpoints fine and is cheap. The headline adds ROUGE-1/2 and BERTScore,
    # because ROUGE punishes a correct summary that is worded differently and BERTScore catches it.
    "dialogsum": ("rouge_l", "rouge_1_2_l_bertscore"),
}


def test_a_task_reporting_a_different_metric_than_it_selects_on_says_so():
    for name, spec in TASKS.items():
        expected = REPORT_METRIC_EXCEPTIONS.get(name)
        if expected is None:
            assert spec.report_metric_name == spec.metric_name, (
                f"{name} selects on {spec.metric_name!r} but reports {spec.report_metric_name!r} "
                f"without an entry in REPORT_METRIC_EXCEPTIONS explaining why"
            )
            continue
        assert (spec.metric_name, spec.report_metric_name) == expected, name


def test_every_task_names_a_report_scorer():
    """`report_score` has no default, so a task cannot reach the registry without deciding.

    A task whose report metric equals its selection metric names the same function twice. That is
    the point: it records that the question was asked, rather than letting a default answer it.
    """
    for name, spec in TASKS.items():
        assert callable(spec.report_score), name
        assert spec.report_metric_name.strip(), name
        if spec.report_metric_name == spec.metric_name:
            assert spec.report_score is spec.score, (
                f"{name} reports the same metric name as it selects on but through a DIFFERENT "
                f"function, which means one of the two is mislabelled"
            )


# Tasks whose report split is a different split, not a bigger draw from the same one.
REPORT_LOAD_TASKS = {
    # Selects on the official 871-row dev, which cannot support a 33-class macro-F1 at all, and
    # reports on a fixed stratified slice of the 249,980-row test split.
    "multiconer",
}


def test_only_the_tasks_that_need_a_separate_report_split_declare_one():
    for name, spec in TASKS.items():
        if name in REPORT_LOAD_TASKS:
            assert spec.report_load is not None, (
                f"{name} is declared as needing its own report split but does not provide one"
            )
            assert callable(spec.report_load), name
        else:
            assert spec.report_load is None, (
                f"{name} declares a report_load; if its report split really is a different split "
                f"from its selection split, add it to REPORT_LOAD_TASKS with the reason"
            )


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
        # `references`, not `answer`. The scorer grades against ALL THREE of DialogSum's human
        # test summaries, and requiring only `answer` would let a source that flattens them —
        # knkarthick/dialogsum's test.csv is 1,500 rows with one `summary` column — load cleanly
        # and silently reduce the metric to single-reference ROUGE.
        "dialogsum": "references",
        "xlam_bfcl": "answer",
        "calendar_json": "answer",
        "ner_bc5cdr": "entities",
        "multiconer": "entities",
        # The parse string. `domain` is required too, because the headline is the mean of the
        # per-domain exact-match rates and a row without a domain cannot enter it.
        "topv2": "answer",
        # `m2` is required alongside it: the corpus's own gold edit annotation, which the scorer
        # reassembles into a reference file. Regenerating it from `answer` would re-segment the
        # annotator's edits and change the number.
        "gec_bea19": "m2",
        # `labels` is the list the scorer grades; `label` is its serialization, which is what the
        # model is trained to emit.
        "goemotions": "labels",
        "routerbench": "label",
        "proactive_listening": "label",
        "clinc150": "label",
        # `query`, not `answer`. ToolEval's pass rate is reference-free — it asks a judge whether
        # the model's own final answer addresses the query — and the ToolBench test queries ship
        # with no gold path at all. `query` is the field the scorer grades against, and requiring
        # `answer` instead would reject every eval row this task has.
        "toolbench": "query",
    }
    for name, target in targets.items():
        spec = get_task(name)
        assert "text" in spec.required_fields, name
        assert target in spec.required_fields, (
            f"{name} does not require {target!r}, the field its scorer grades"
        )


def test_the_judged_tasks_say_so_and_are_the_only_ones():
    """`needs_judge` exists so a judge outage fails loudly instead of scoring zero and sending the
    loop chasing a phantom regression.

    ONE task scores through the judge, and it is the one where the judge is not a proxy for a
    metric but IS the metric: `toolbench`'s ToolEval pass rate is DEFINED as a majority vote of
    judged assessments, and its test queries ship with no reference solution at all.

    `dialogsum` used to be the second, on the reasoning that a summary has no exact gold so
    similarity must be judged. That was wrong about the data. DialogSum's test split ships THREE
    human summaries per dialogue, so multi-reference ROUGE has a real gold to score against — and
    unlike a judge it is comparable to the published baselines and to the human ceiling, is
    deterministic, and costs nothing per eval. It moved off the judge on 2026-09-06.

    Every other task is scored by computation, and adding a second judged task should have to
    argue for itself here.
    """
    judged = {name for name, spec in TASKS.items() if spec.needs_judge}
    assert judged == {"toolbench"}
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


def test_an_exact_synth_verifier_publishes_its_REASON_not_just_its_verdict():
    """`TaskSpec.synth_verifier` returns yes/no, but the reason must stay reachable.

    `data.curriculum` recovers it through a `.checker` attribute and prints the
    "[verify:exact] programmatic verifier rejected N row(s)" breakdown from it. Without the attribute
    that block is unreachable, and a total rejection reports only its own size.

    Run 38985393 paid for this: synthesis generated 519 rows, the exact verifier rejected all 519, and
    the log said nothing about WHICH of the five checks fired — the reason string had been discarded by
    a `[0]` subscript one character from where it was needed, so it had to be reverse-engineered from
    the vLLM access log afterwards.
    """
    import tasks

    offenders = []
    for name, spec in sorted(tasks.TASKS.items()):
        verifier = spec.synth_verifier
        if verifier is None:
            continue
        checker = getattr(verifier, "checker", None)
        if checker is None:
            offenders.append(f"{name}: synth_verifier has no .checker attribute")
            continue
        if not callable(checker):
            offenders.append(f"{name}: .checker is {type(checker).__name__}, not callable")
    assert not offenders, (
        "every exact synth verifier must publish its reason via .checker:\n  "
        + "\n  ".join(offenders)
    )


def test_the_checker_reason_agrees_with_the_verdict_and_explains_a_rejection():
    """A `.checker` that disagreed with the verifier would print a reason for the wrong decision."""
    import tasks

    # A row that is wrong for every task: no answer, no fields, nothing to verify against.
    broken = {"text": "", "answer": "", "entities": [], "tools": []}
    for name, spec in sorted(tasks.TASKS.items()):
        verifier = spec.synth_verifier
        if verifier is None:
            continue
        verdict = verifier(dict(broken))
        checked, reason = verifier.checker(dict(broken))
        assert verdict == checked, f"{name}: verdict {verdict} != checker verdict {checked}"
        assert not verdict, f"{name}: an empty row must not verify"
        # The reason is what a human reads at 3am; an empty string is the failure this test exists for.
        assert reason.strip(), f"{name}: rejected a row with no reason given"
