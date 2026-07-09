from eval.metrics import entity_f1


def test_duplicate_entity_counted_once_in_set_but_twice_in_counter():
    # Gold has "Apple" ORG twice; pred has it once
    # Correct: TP=1, FN=1 → recall=0.5 → F1 < 1.0
    gold = [[{"text": "Apple", "type": "ORG"}, {"text": "Apple", "type": "ORG"}]]
    pred = [[{"text": "Apple", "type": "ORG"}]]
    f1 = entity_f1(pred, gold)
    assert f1 < 1.0, f"entity_f1 should be < 1.0 for missing duplicate entity, got {f1}"
    # TP=1, FP=0, FN=1 → P=1, R=0.5 → F1=0.667
    assert abs(f1 - 2/3) < 0.01, f"Expected F1≈0.667, got {f1}"


def test_perfect_match():
    gold = [[{"text": "Apple", "type": "ORG"}]]
    pred = [[{"text": "Apple", "type": "ORG"}]]
    assert entity_f1(pred, gold) == 1.0


def test_no_match():
    gold = [[{"text": "Apple", "type": "ORG"}]]
    pred = [[{"text": "Google", "type": "ORG"}]]
    assert entity_f1(pred, gold) == 0.0
