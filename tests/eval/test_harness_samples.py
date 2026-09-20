"""The eval sampler must show rows worth reading, not always rows 0-2."""


def _capture(monkeypatch, rows, raws, preds):
    from types import SimpleNamespace

    from eval import harness

    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    harness._log_prediction_samples(SimpleNamespace(all=rows), raws, preds)
    return "\n".join(printed)


def test_the_sampler_prefers_rows_the_model_got_wrong_once_extraction_is_clean(monkeypatch):
    """The case the old ordering was blind to, and the only symptom a scorer bug has.

    `chosen` was `extraction_failures + everything else in index order`, so a run whose outputs all
    parse showed rows 0, 1, 2 forever. Measured on gec_bea19 run 39881529: the same three rows
    across all 58 eval rounds at `format_valid=1.0000` — 2 unique rows of signal out of 1,000, and
    not one row that was merely SCORED wrong. An extraction failure already shows up in
    `format_valid`; a correct-looking answer graded wrong shows up nowhere else.
    """
    rows = [{"text": f"t{i}", "answer": f"gold{i}"} for i in range(6)]
    raws = [f"raw{i}" for i in range(6)]
    # Rows 0-2 are RIGHT, row 4 is wrong. The old code would never have shown row 4.
    preds = ["gold0", "gold1", "gold2", "gold3", "WRONG", "gold5"]

    out = _capture(monkeypatch, rows, raws, preds)
    assert "WRONG" in out, "the sampler still shows only the leading rows"


def test_extraction_failures_still_outrank_mere_mismatches(monkeypatch):
    """The original priority is preserved: a format failure is the more urgent diagnosis."""
    rows = [{"text": f"t{i}", "answer": f"gold{i}"} for i in range(4)]
    raws = [f"raw{i}" for i in range(4)]
    preds = ["gold0", "WRONG", "__EXTRACTION_FAILED__", "gold3"]

    out = _capture(monkeypatch, rows, raws, preds)
    assert "EXTRACTION FAILED" in out
    assert out.index("EXTRACTION FAILED") < (out.index("WRONG") if "WRONG" in out else len(out))


def test_an_all_correct_eval_still_prints_samples(monkeypatch):
    """No mismatches must not mean no output — the fallback tier still fills the sample."""
    rows = [{"text": f"t{i}", "answer": f"gold{i}"} for i in range(4)]
    out = _capture(monkeypatch, rows, [f"raw{i}" for i in range(4)],
                   [f"gold{i}" for i in range(4)])
    assert "sample predictions" in out
    assert "gold0" in out


def test_the_sampler_cannot_raise_on_a_row_with_no_gold(monkeypatch):
    """It runs on the eval hot path inside a CUDA worker; a display helper must never break eval."""
    rows = [{"text": "t0"}, {"text": "t1", "answer": "g1"}]
    out = _capture(monkeypatch, rows, ["r0", "r1"], ["p0", "p1"])
    assert "sample predictions" in out


def test_the_fields_under_comparison_are_not_truncated_away(monkeypatch):
    """`gold` and `parsed` were clipped to 60 chars while `input`/`raw` got 110 — backwards.

    Gold-vs-parsed is the comparison these samples exist to enable, and a structured prediction is
    exactly where the characters go. On multiconer run 39881531 a two-entity row printed as
    `parsed: [{'text': 'china', ...}, {'text': 'ele…` — clipping away the entity actually in
    dispute. A third of sampled rows were unusable for auditing as a result.
    """
    from types import SimpleNamespace

    from eval import harness

    entities = [{"text": "china", "type": "HumanSettlement"},
                {"text": "electric motor", "type": "OtherPROD"}]
    rows = [{"text": "china 's electric motor industry has been developed for 60 years .",
             "answer": str(entities)}]

    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    harness._log_prediction_samples(SimpleNamespace(all=rows), [str(entities)], [str(entities)])
    out = "\n".join(printed)

    # The SECOND entity — the one that was being cut off — must survive in both fields.
    gold_line = next(l for l in out.splitlines() if "gold  :" in l)
    parsed_line = next(l for l in out.splitlines() if "parsed:" in l)
    assert "electric motor" in gold_line, gold_line
    assert "OtherPROD" in parsed_line, parsed_line
