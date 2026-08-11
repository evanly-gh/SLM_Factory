"""fill_synthesize vs surgical_synthesize (2026-08-05).

Surgical spends part of the `synthesize` budget on the classes the model actually confuses, with
per-pair budget PROPORTIONAL to the confusion count, and skips pairs that were targeted before
without improving. It generates in-class GOLD rows for the confused classes, and never anything
derived from held-out eval rows.
"""
from unittest.mock import patch

from agent.nodes.curate import _surgical_synthesize


def _train_rows():
    return [
        {"text": f"utterance {label}-{i}", "label": label}
        for label in ("change_ai_name", "change_language", "cancel")
        for i in range(60)
    ]


def _state(pairs, iteration=5, history=None):
    return {
        "task_type": "classification",
        "iteration": iteration,
        "test_report": {"confusion_pairs": pairs},
        "surgical_pair_history": history or {},
    }


def _fake_synth(anchors, *, task_type, n, generate_fn, log=None, **_kw):
    return [
        {"text": f"synth for {a['label']}", "label": a["label"], "_source": "synth"}
        for a in anchors
    ]


def test_budget_is_proportional_to_confusion_count():
    pairs = [
        {"gold": "change_ai_name", "predicted": "change_user_name", "count": 8},
        {"gold": "change_language", "predicted": "translate", "count": 2},
    ]
    state = _state(pairs)
    with patch("agent.nodes.curate.synthesize_examples", side_effect=_fake_synth):
        rows = _surgical_synthesize(
            state, {"synth_rows": 500}, _train_rows(),
            generate_fn=lambda *a, **k: "x", model_id="m", seed=1,
        )
    by_label = {}
    for row in rows:
        by_label[row["label"]] = by_label.get(row["label"], 0) + 1
    # 8-count pair must receive strictly more than the 2-count pair.
    assert by_label["change_ai_name"] > by_label["change_language"]
    assert all(r["_strategy_origin"] == "surgical_synthesize" for r in rows)


def test_pair_that_did_not_improve_is_skipped_as_exhausted(capsys):
    pairs = [{"gold": "cancel", "predicted": "freeze_account", "count": 7}]
    history = {"cancel->freeze_account": {"count_when_targeted": 7, "iteration": 3}}
    state = _state(pairs, history=history)
    with patch("agent.nodes.curate.synthesize_examples", side_effect=_fake_synth) as synth:
        rows = _surgical_synthesize(
            state, {"synth_rows": 500}, _train_rows(),
            generate_fn=lambda *a, **k: "x", model_id="m", seed=1,
        )
    assert rows == []
    synth.assert_not_called()
    assert "EXHAUSTED" in capsys.readouterr().out


def test_pair_that_improved_is_targeted_again():
    pairs = [{"gold": "cancel", "predicted": "freeze_account", "count": 3}]
    history = {"cancel->freeze_account": {"count_when_targeted": 7, "iteration": 3}}
    state = _state(pairs, history=history)
    with patch("agent.nodes.curate.synthesize_examples", side_effect=_fake_synth):
        rows = _surgical_synthesize(
            state, {"synth_rows": 500}, _train_rows(),
            generate_fn=lambda *a, **k: "x", model_id="m", seed=1,
        )
    assert rows, "a pair whose confusion count fell should still be worth targeting"
    assert state["surgical_pair_history"]["cancel->freeze_account"]["count_when_targeted"] == 3


def test_no_confusion_pairs_means_no_surgical_spend():
    with patch("agent.nodes.curate.synthesize_examples", side_effect=_fake_synth) as synth:
        rows = _surgical_synthesize(
            _state([]), {"synth_rows": 500}, _train_rows(),
            generate_fn=lambda *a, **k: "x", model_id="m", seed=1,
        )
    assert rows == []
    synth.assert_not_called()


def test_anchors_come_from_the_gold_class_not_the_predicted_one():
    pairs = [{"gold": "cancel", "predicted": "change_ai_name", "count": 5}]
    with patch("agent.nodes.curate.synthesize_examples", side_effect=_fake_synth):
        rows = _surgical_synthesize(
            _state(pairs), {"synth_rows": 500}, _train_rows(),
            generate_fn=lambda *a, **k: "x", model_id="m", seed=1,
        )
    # Rows reinforce the class the model SHOULD have predicted.
    assert {r["label"] for r in rows} == {"cancel"}
