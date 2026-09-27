#!/usr/bin/env python3
"""
test_ner_metrics.py — Verifies entity_f1() against the 3 specified cases,
plus load_ner_eval_set()'s flattening.

Run directly: python3 test_ner_metrics.py
"""

from ner_metrics import entity_f1, load_ner_eval_set
import json
import tempfile
from pathlib import Path


def test_duplicate_gold_entities_matter():
    # gold has "Apple"/ORG twice, pred has it once -> TP=1, FN=1
    # precision = 1/1 = 1.0, recall = 1/2 = 0.5, F1 = 2*1*0.5/1.5 = 0.6667
    gold = [[{"text": "Apple", "type": "ORG"}, {"text": "Apple", "type": "ORG"}]]
    pred = [[{"text": "Apple", "type": "ORG"}]]
    f1 = entity_f1(pred, gold)
    assert abs(f1 - 0.6667) < 0.001, f"expected ~0.6667, got {f1}"
    print(f"PASS: duplicate gold entities -> F1={f1:.4f} (expected ~0.6667)")


def test_perfect_match():
    gold = [[{"text": "Apple", "type": "ORG"}, {"text": "Paris", "type": "LOC"}]]
    pred = [[{"text": "Apple", "type": "ORG"}, {"text": "Paris", "type": "LOC"}]]
    f1 = entity_f1(pred, gold)
    assert f1 == 1.0, f"expected 1.0, got {f1}"
    print(f"PASS: perfect match -> F1={f1}")


def test_no_match():
    gold = [[{"text": "Apple", "type": "ORG"}]]
    pred = [[{"text": "Google", "type": "ORG"}]]
    f1 = entity_f1(pred, gold)
    assert f1 == 0.0, f"expected 0.0, got {f1}"
    print(f"PASS: no match -> F1={f1}")


def test_none_prediction_treated_as_empty():
    # A None prediction (unparseable output) must score the same as an
    # explicit [] - both contribute zero predicted entities, so an example
    # with real gold entities and a None prediction is pure FN, same as []
    # would be.
    gold = [[{"text": "Apple", "type": "ORG"}]]
    pred_none = [None]
    pred_empty = [[]]
    f1_none = entity_f1(pred_none, gold)
    f1_empty = entity_f1(pred_empty, gold)
    assert f1_none == f1_empty == 0.0, f"expected both 0.0, got none={f1_none} empty={f1_empty}"
    print(f"PASS: None prediction scores identically to [] -> F1={f1_none}")


def test_multi_example_aggregation():
    # Aggregation is over the FULL corpus (summed TP/FP/FN across examples),
    # not a per-example mean of F1 scores - confirms that distinction holds.
    gold = [
        [{"text": "Apple", "type": "ORG"}],
        [{"text": "Paris", "type": "LOC"}],
    ]
    pred = [
        [{"text": "Apple", "type": "ORG"}],  # perfect
        [{"text": "London", "type": "LOC"}],  # wrong
    ]
    # tp=1 (Apple), fp=1 (London), fn=1 (Paris) -> precision=0.5, recall=0.5, F1=0.5
    f1 = entity_f1(pred, gold)
    assert f1 == 0.5, f"expected 0.5, got {f1}"
    print(f"PASS: multi-example corpus-level aggregation -> F1={f1}")


def test_load_ner_eval_set_flattens_slices():
    data = {
        "pos": [{"text": "Apple is in Cupertino.", "entities": [{"text": "Apple", "type": "ORG"}]}],
        "neg": [{"text": "The sky is blue.", "entities": []}],
        "boundary": [{"text": "New York City", "entities": [{"text": "New York City", "type": "LOC"}]}],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "eval_set.json"
        path.write_text(json.dumps(data))
        rows = load_ner_eval_set(str(path))

    assert len(rows) == 3, f"expected 3 flattened rows, got {len(rows)}"
    assert rows[0] == {"text": "Apple is in Cupertino.", "entities": [{"text": "Apple", "type": "ORG"}]}
    assert rows[1] == {"text": "The sky is blue.", "entities": []}
    assert rows[2] == {"text": "New York City", "entities": [{"text": "New York City", "type": "LOC"}]}
    print(f"PASS: load_ner_eval_set flattens pos+neg+boundary -> {len(rows)} rows, correct order/content")


def main():
    test_duplicate_gold_entities_matter()
    test_perfect_match()
    test_no_match()
    test_none_prediction_treated_as_empty()
    test_multi_example_aggregation()
    test_load_ner_eval_set_flattens_slices()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
