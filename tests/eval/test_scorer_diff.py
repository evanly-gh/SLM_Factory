"""diff verifier (2026-08-01): `git apply --check` + applied-result match.

Pins the two-column contract: format_valid = the predicted diff cleanly git-applies to src;
content_correct (the f1 scalar) = applying it yields tgt. A well-formed diff that produces the
wrong text is format-valid but content-wrong; garbage is format-invalid. Skips when git is
absent so the suite stays green on a box without git.
"""
import difflib
import shutil

import pytest

from data.eval_set import EvalSet
from eval.scorers.diff import extract_predictions, score

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def _unified(src: str, tgt: str) -> str:
    return "".join(difflib.unified_diff(
        src.splitlines(keepends=True),
        tgt.splitlines(keepends=True),
        fromfile="a/file.txt", tofile="b/file.txt",
    ))


def _row(text, src, tgt):
    return {"text": text, "src": src, "tgt": tgt, "answer": _unified(src, tgt)}


def _eval_set(rows):
    return EvalSet(all=list(rows), task_type="diff")


SRC = "The quick brown fox.\nJumps over the lazy dog.\nThe end.\n"
TGT = "The quick red fox.\nJumps over the lazy dog.\nThe end.\n"
OTHER = "The quick brown fox.\nJumps over the sleepy dog.\nThe end.\n"


def _score_raw(rows, raw_outputs):
    es = _eval_set(rows)
    preds = extract_predictions(raw_outputs, es)
    return score(es, preds)


def test_valid_diff_reproduces_target_scores_one():
    rows = [_row("change brown to red", SRC, TGT)]
    result = _score_raw(rows, [_unified(SRC, TGT)])
    assert result["f1"] == 1.0
    assert result["metric"] == "apply_match"
    assert result["per_class"]["format_valid"] == 1.0


def test_wellformed_diff_wrong_result_is_format_valid_content_wrong():
    rows = [_row("change brown to red", SRC, TGT)]
    # A diff that applies cleanly to SRC but yields OTHER, not TGT.
    result = _score_raw(rows, [_unified(SRC, OTHER)])
    assert result["per_class"]["format_valid"] == 1.0
    assert result["f1"] == 0.0


def test_malformed_diff_is_format_invalid():
    rows = [_row("change brown to red", SRC, TGT)]
    result = _score_raw(rows, ["this is not a diff at all"])
    assert result["per_class"]["format_valid"] == 0.0
    assert result["f1"] == 0.0


def test_diff_that_does_not_apply_is_format_invalid():
    rows = [_row("change brown to red", SRC, TGT)]
    # A diff whose context lines do not match SRC cannot apply.
    bad = _unified("Totally different content.\nNothing matches here.\n",
                   "Totally different content.\nStill nothing.\n")
    result = _score_raw(rows, [bad])
    assert result["per_class"]["format_valid"] == 0.0
    assert result["f1"] == 0.0


def test_fenced_diff_is_unwrapped_and_applies():
    rows = [_row("change brown to red", SRC, TGT)]
    fenced = "```diff\n" + _unified(SRC, TGT) + "```"
    result = _score_raw(rows, [fenced])
    assert result["f1"] == 1.0
