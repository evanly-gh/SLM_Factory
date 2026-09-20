"""The eval-set size cap is applied ONCE, by the loader, not twice (B288).

`eval_size_target` defaulted to 800 and was applied at TWO places: as the loader's `max_test`, and
again as `build_eval_set`'s `target`. Raising the cap in the first place therefore did nothing —
the observed run still reported `eval set built: 800 examples (available in the held-out split: 1000)`.

This is also a regression test against my own editing mistake: the first fix targeted a string that did
not exist in the file, and `str.replace` with no match is a silent no-op, so the "fix" shipped as
nothing. These tests assert the observable behaviour rather than the source text.
"""
import pytest

from agent.nodes.cold_start.eval_setup import _eval_target


def test_every_task_declares_its_own_select_cap():
    """The cap used to be one module-level constant shared by every benchmark. It is now
    `TaskSpec.select_cap`, because the trade-off it encodes is per task: the eval runs every
    iteration, so an unbounded split makes every loop turn proportionally slower.

    It is also explicitly a SELECTION cap and not the size of anything published — that is what
    the rename bought. The number is defensible for ranking because the comparison is paired
    (identical rows, successive checkpoints, so the sampling error is common-mode), and it is not
    defensible in a paper, where the +/-2.5-point CI half-width at n=1,000 would swallow most of
    the effects being measured. `report_load`/`report_score` and `scripts/report_eval.py` are the
    reporting half."""
    from tasks import TASKS

    assert TASKS, "the registry must not be empty"
    assert all(spec.select_cap >= 1 for spec in TASKS.values())
    assert TASKS["xlam_bfcl"].select_cap == 1000


def test_eval_target_honours_a_large_request():
    """`_eval_target` must not impose a cap of its own — that was the second truncation."""
    assert _eval_target(1000) == 1000
    assert _eval_target(5865) == 5865


def test_eval_target_floors_a_tiny_request():
    """A 30-row floor so a small orchestrator-chosen target cannot starve the eval set (B161)."""
    assert _eval_target(0) == 30
    assert _eval_target(5) == 30


def test_the_cap_reaches_the_loader_and_is_never_applied_a_second_time(monkeypatch):
    """The B288 invariant, driven through the production call path.

    `TaskSpec.select_cap` is handed to the loader as `max_test`, and whatever the loader returns IS
    the target `build_eval_set` is given. That is the whole fix: there is no second number left to
    re-cap the split with. `eval_size_target` — the run-state field that supplied the second one —
    was removed with the autonomous path, so the branch this test used to reproduce at the call
    site no longer exists to be got wrong.
    """
    import dataclasses

    from agent.nodes.cold_start import eval_setup
    from tasks import TASKS, get_task

    asked: dict[str, int] = {}

    def loader(max_train, max_test, log=print):
        asked["max_train"] = max_train
        asked["max_test"] = max_test
        return [], [{"text": f"r{index}", "answer": "[]"} for index in range(max_test)]

    monkeypatch.delenv("SLM_EVAL_SIZE_CAP", raising=False)
    monkeypatch.setitem(
        TASKS, "xlam_bfcl", dataclasses.replace(get_task("xlam_bfcl"), load=loader)
    )

    _train, test_examples = eval_setup._load_named_benchmark(
        "xlam_bfcl", {"task": "xlam_bfcl"}, {}
    )

    assert asked["max_test"] == 1000, "the loader must be asked for the task's own eval cap"
    assert len(test_examples) == 1000
    assert _eval_target(len(test_examples)) == 1000, "the split was re-capped on the way out"


def test_build_eval_set_returns_everything_it_is_given(monkeypatch):
    """End-to-end on the sizing contract: given N rows and a target of N, keep all N."""
    from data.eval_set import build_eval_set

    rows = [{"text": f"r{i}", "answer": "[]"} for i in range(1000)]
    eval_set = build_eval_set(rows, task="xlam_bfcl", target=_eval_target(len(rows)))
    assert len(eval_set.all) == 1000


def test_build_eval_set_covers_every_class_for_multiclass():
    """A label-balanced task round-robins across labels, so a 1,000-row target over 151 classes
    must still span all of them — the property that makes CLINC150's eval set usable."""
    from data.eval_set import build_eval_set

    rows = [
        {"text": f"utterance {i}", "label": f"intent_{i % 151}"}
        for i in range(5000)
    ]
    eval_set = build_eval_set(rows, task="clinc150", target=1000)
    assert len(eval_set.all) == 1000
    assert len({r["label"] for r in eval_set.all}) == 151
