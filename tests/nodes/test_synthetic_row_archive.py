"""The append-only record of what synthesis actually produced.

WHY THIS FILE EXISTS
    The versioned `dataset_vN.jsonl` artifacts ARE the curriculum, and rollback overwrites them. On
    run 38708719 both synthesis rounds regressed, both were rolled back, and the next mining round
    wrote a new `dataset_v2` — so afterwards not one synthesized row survived anywhere on disk.

    The orchestrator had concluded twice that those rows carried noise, and the single most useful
    question about them — WHY did they hurt — could not be asked, because there was nothing left to
    read. `_archive_synthetic_rows` appends every KEPT synthesized row to a per-run audit file that
    rollback does not touch.

    It is EVIDENCE, not state: nothing in the pipeline reads it back, so the only two things that
    can go wrong are losing rows and breaking the rebuild that produced them. Hence the two halves
    of the contract below — every kept row is written with the iteration that will train on it, and
    a failure to write is logged and swallowed. An audit trail that can kill a run costs more than
    it is worth.
"""
from __future__ import annotations

import importlib
import json


def _curate():
    """The LIVE `agent.nodes.curate` module.

    Resolved per call rather than imported at file scope: `tests/config/test_curation_config.py`
    reloads curation modules, and a module object captured at import time can end up being a dead
    copy whose `ARTIFACTS_DIR` production code no longer reads.
    """
    return importlib.import_module("agent.nodes.curate")


def _archive(tmp_path, monkeypatch, rows, *, iteration=2, subdir="artifacts"):
    """Archive `rows` into a temporary artifacts directory and return the audit file's path."""
    curate = _curate()
    monkeypatch.setattr(curate, "ARTIFACTS_DIR", str(tmp_path / subdir))
    curate._archive_synthetic_rows({"iteration": iteration}, rows, model_id="Qwen/Qwen3-1.7B")
    return tmp_path / subdir / "synthetic_rows_audit.jsonl"


def _read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


SYNTHESIZED = [
    {"text": "how far is Portland from Seattle",
     "answer": '[{"name": "calculate_distance", "arguments": {"origin": "Seattle"}}]',
     "_provenance": "synthetic", "_strategy_origin": "surgical_synthesis",
     "_target_category": "wrong_argument_value"},
    {"text": "book me a table for four at 7",
     "answer": '[{"name": "reserve", "arguments": {"party": 4}}]',
     "_provenance": "synthetic", "_strategy_origin": "surgical_synthesis",
     "_target_category": "missing_call"},
]


def test_every_kept_row_is_written_with_the_iteration_that_will_train_on_it(tmp_path, monkeypatch):
    """The iteration is what makes a row attributable to a score.

    Curate runs BEFORE the counter advances, so the rows it archives are trained on the NEXT
    iteration — the same +1 convention the run-health ledger records under. Stamping the current
    value would file a synthesis round's rows under the previous iteration's result, which is
    exactly the join this archive exists to let a human perform by hand.
    """
    path = _archive(tmp_path, monkeypatch, SYNTHESIZED, iteration=2)
    archived = _read(path)

    assert [row["_iteration"] for row in archived] == [3, 3]
    assert [row["text"] for row in archived] == [row["text"] for row in SYNTHESIZED]


def test_the_row_is_archived_whole_rather_than_summarised(tmp_path, monkeypatch):
    """The question being asked of this file is "why did these rows hurt", and the answer is in the
    row: which failure category it was aimed at, and what answer the teacher produced for it. A
    count or a hash would record that synthesis happened without preserving anything to inspect."""
    path = _archive(tmp_path, monkeypatch, SYNTHESIZED)
    first = _read(path)[0]

    for key, value in SYNTHESIZED[0].items():
        assert first[key] == value


def test_a_second_round_appends_rather_than_replacing_the_first(tmp_path, monkeypatch):
    """Run 38708719 had TWO synthesis rounds and lost both. A file rewritten per round would keep
    only the last one, which is the same failure mode as the versioned artifacts it exists to
    survive."""
    path = _archive(tmp_path, monkeypatch, SYNTHESIZED, iteration=2)
    assert len(_read(path)) == 2

    later = [{"text": "cancel my 3pm", "answer": "[]", "_provenance": "synthetic"}]
    path = _archive(tmp_path, monkeypatch, later, iteration=4)
    archived = _read(path)

    assert len(archived) == 3
    assert [row["_iteration"] for row in archived] == [3, 3, 5]
    assert archived[0]["text"] == SYNTHESIZED[0]["text"]


def test_a_round_that_kept_nothing_writes_nothing(tmp_path, monkeypatch):
    """An empty file, or a run of blank lines, reads as "synthesis produced rows and they are
    unreadable". Producing nothing is a different and already well-reported outcome."""
    path = _archive(tmp_path, monkeypatch, [])
    assert not path.exists()


def test_a_round_that_kept_nothing_is_silent(tmp_path, monkeypatch, capsys):
    """No rows, no line. A log entry per empty round trains the reader to skip the archive lines,
    including the one reporting a failure to write."""
    _archive(tmp_path, monkeypatch, [])
    assert "synth] archived" not in capsys.readouterr().out


def test_a_successful_archive_says_where_it_went(tmp_path, monkeypatch, capsys):
    """The file is never read by the pipeline, so the log line is the only thing that tells a human
    it exists — and it has to say the rows survive rollback, or the reader has no reason to look
    there after the artifacts were overwritten."""
    _archive(tmp_path, monkeypatch, SYNTHESIZED)
    out = capsys.readouterr().out

    assert "synthetic_rows_audit.jsonl" in out
    assert "2 kept row(s)" in out
    assert "rollback" in out


def test_an_unwritable_destination_is_logged_and_does_not_break_the_rebuild(tmp_path, monkeypatch,
                                                                           capsys):
    """The audit trail must never be able to fail a run.

    A curate pass that has already spent the teacher budget and assembled a curriculum cannot be
    allowed to die on an evidence file — that trades the thing being protected for the record of
    it. The failure is announced, because silence here is indistinguishable from a synthesis round
    that kept no rows.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("this is a regular file", encoding="utf-8")

    curate = _curate()
    monkeypatch.setattr(curate, "ARTIFACTS_DIR", str(blocker / "artifacts"))
    curate._archive_synthetic_rows({"iteration": 2}, SYNTHESIZED, model_id="Qwen/Qwen3-1.7B")

    out = capsys.readouterr().out
    assert "could not archive generated rows" in out
    assert "NotADirectoryError" in out


def test_a_state_with_no_iteration_still_archives_the_rows(tmp_path, monkeypatch):
    """A curate pass during cold start has no iteration recorded yet. The rows are the evidence;
    losing them because the stamp is unavailable is the outcome this function exists to prevent."""
    curate = _curate()
    monkeypatch.setattr(curate, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    curate._archive_synthetic_rows({}, SYNTHESIZED, model_id="Qwen/Qwen3-1.7B")

    archived = _read(tmp_path / "artifacts" / "synthetic_rows_audit.jsonl")
    assert len(archived) == 2
    assert {row["_iteration"] for row in archived} == {1}


def test_a_total_rejection_is_archived_with_the_reason_for_each_row(tmp_path, monkeypatch):
    """The case the archive existed for and could not see.

    Run 38985393 generated 519 toolbench rows, the exact verifier rejected all 519, and the audit file
    was EMPTY — `_archive_synthetic_rows` returned early on an empty kept-list, so a rebuild that kept
    nothing recorded nothing. A count of reasons says which check fired; the row says what the teacher
    actually wrote, and only the second tells you how to fix the prompt.
    """
    import json

    from agent.nodes import curate as curate_module
    from data.curriculum import note_rejected_row, take_rejected_rows

    take_rejected_rows()  # start from a clean sample regardless of test order
    monkeypatch.setattr(curate_module, "ARTIFACTS_DIR", str(tmp_path))

    note_rejected_row({"text": "find me a hotel", "answer": "Thought: ok\nAction: nope"},
                      "answer is not a sequence of Action / Action Input turns")
    note_rejected_row({"text": "book a flight", "answer": "Action: made_up_api"},
                      "calls undeclared API 'made_up_api'")

    # Zero KEPT rows: exactly the wipeout shape.
    curate_module._archive_synthetic_rows({"iteration": 3}, [], model_id="m")

    written = [json.loads(line) for line in
               (tmp_path / "synthetic_rows_audit.jsonl").read_text().splitlines()]
    assert len(written) == 2, "a total rejection must still be archived"
    assert {row["_verdict"] for row in written} == {"rejected"}
    # `_iteration` is the DISPLAYED number, one ahead of the 0-indexed counter on the state.
    assert all(row["_iteration"] == 4 for row in written)
    # The reason travels WITH the row; that pairing is the whole point.
    reasons = {row["_reject_reason"] for row in written}
    assert "answer is not a sequence of Action / Action Input turns" in reasons
    assert "calls undeclared API 'made_up_api'" in reasons
    assert any("made_up_api" in str(row.get("answer", "")) for row in written)


def test_taking_the_rejected_sample_clears_it_so_iterations_do_not_bleed(tmp_path, monkeypatch):
    """Otherwise iteration 4's audit file would re-report iteration 3's rejections as its own."""
    from data.curriculum import note_rejected_row, take_rejected_rows

    take_rejected_rows()
    note_rejected_row({"text": "a"}, "some reason")
    assert len(take_rejected_rows()) == 1
    assert take_rejected_rows() == []


def test_the_rejected_sample_is_bounded(tmp_path, monkeypatch):
    """A 500-row wipeout must not write a 500-row artifact on every iteration."""
    import data.curriculum as curriculum_module

    curriculum_module.take_rejected_rows()
    monkeypatch.setattr(curriculum_module, "MAX_ARCHIVED_REJECTS", 5)
    for index in range(50):
        curriculum_module.note_rejected_row({"text": f"row {index}"}, "reason")
    assert len(curriculum_module.take_rejected_rows()) == 5
