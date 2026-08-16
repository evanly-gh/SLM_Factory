"""Two reporting fixes: NER gold is actually shown, and the first fine-tuned score is reported.

B263 — the sample-prediction display read only `answer`/`label`, but NER rows keep their gold in
`entities`, so `gold :` printed BLANK on every NER row in both the baseline and fine-tuned blocks.
That removed the only human check of gold against prediction from the task where it matters most: a
legitimate `Baseline F1 = 0.0000` (the base model emitting ```json []``` on all 800 rows) was
indistinguishable from a broken harness.

Model Improvement Report — `Baseline → Best FT` alone cannot separate what one round of fine-tuning
bought from what the orchestrated search added. BC5CDR's +0.8098 was almost entirely iteration 1
(0.0000 → 0.7701), with four further iterations contributing 0.0397.
"""
from eval.harness import _gold_for_display
from agent.pipeline_status import _first_finetuned_for, build_run_progression


# --------------------------------------------------------------------------
# NER gold display
# --------------------------------------------------------------------------

def test_ner_gold_renders_entity_spans():
    row = {
        "text": "Famotidine - associated delirium .",
        "entities": [
            {"text": "Famotidine", "type": "Chemical"},
            {"text": "delirium", "type": "Disease"},
        ],
    }
    shown = _gold_for_display(row)
    assert shown == "['Famotidine':Chemical, 'delirium':Disease]"


def test_ner_gold_with_no_entities_is_visibly_empty_not_missing():
    """An empty span list is a real gold value; it must render as `[]`, not as a blank line."""
    assert _gold_for_display({"text": "011 ), necrosis ( R = 1 .", "entities": []}) == "[]"


def test_classification_and_generation_gold_unchanged():
    assert _gold_for_display({"text": "t", "label": "accept_reservations"}) == "accept_reservations"
    assert _gold_for_display({"text": "t", "answer": "a summary"}) == "a summary"


def test_answer_wins_over_label_for_generation_rows():
    row = {"text": "t", "answer": "the summary", "label": "generation"}
    assert _gold_for_display(row) == "the summary"


def test_diff_row_without_precomputed_answer_falls_back_to_src_tgt():
    row = {"text": "Paraphrase this", "src": "before\n", "tgt": "after\n"}
    assert "before" in _gold_for_display(row) and "after" in _gold_for_display(row)


def test_row_with_no_gold_at_all_returns_falsy_rather_than_raising():
    assert not _gold_for_display({"text": "t"})


# --------------------------------------------------------------------------
# First fine-tuned score in the progression
# --------------------------------------------------------------------------

def _model(selector="m@Q4_K_M", tier=0):
    class M:
        pass
    m = M()
    m.selector = selector
    m.model_id = "m"
    m.quant = "Q4_K_M"
    m.tier = tier
    return m


def test_first_finetuned_is_looked_up_by_selector():
    baselines = [
        {"selector": "a@bf16", "baseline_f1": 0.1, "first_finetuned_f1": 0.5},
        {"selector": "b@bf16", "baseline_f1": 0.2, "first_finetuned_f1": 0.6},
    ]
    assert _first_finetuned_for("b@bf16", baselines) == 0.6


def test_missing_first_finetuned_reads_as_none_not_zero():
    """Same rule as the baseline: absent must render `n/a`, never a real score of zero."""
    assert _first_finetuned_for("a@bf16", [{"selector": "a@bf16", "baseline_f1": 0.1}]) is None
    assert _first_finetuned_for("missing@bf16", []) is None


def test_progression_carries_first_finetuned_for_the_final_model():
    state = {
        "selected_model": _model(),
        "best_score": 0.8098,
        "iteration": 5,
        "scores": [0.7701, 0.7730, 0.7362, 0.7829, 0.8098],
        "dag": [],
        "best_weights_ref": None,
    }
    baselines = [{"selector": "m@Q4_K_M", "baseline_f1": 0.0, "first_finetuned_f1": 0.7701}]
    progression = build_run_progression(state, baselines)
    entry = progression[-1]
    assert entry["baseline_f1"] == 0.0
    assert entry["first_finetuned_f1"] == 0.7701
    assert entry["best_score"] == 0.8098
    # The two halves of the gain, which is the whole point of the column.
    assert round(entry["first_finetuned_f1"] - entry["baseline_f1"], 4) == 0.7701
    assert round(entry["best_score"] - entry["first_finetuned_f1"], 4) == 0.0397


def test_first_finetuned_is_not_taken_from_scores_zero():
    """scores[0] can BE the baseline when the zero-shot model wins iteration 1, so the two are
    indistinguishable there. The value must come from the recorded field instead."""
    state = {
        "selected_model": _model(),
        "best_score": 0.6,
        "iteration": 3,
        "scores": [0.5443, 0.58, 0.6],  # scores[0] == the baseline below
        "dag": [],
        "best_weights_ref": None,
    }
    baselines = [{
        "selector": "m@Q4_K_M",
        "baseline_f1": 0.5443,
        "first_finetuned_f1": 0.2675,  # the real fine-tuned score, which LOST to the baseline
    }]
    entry = build_run_progression(state, baselines)[-1]
    assert entry["first_finetuned_f1"] == 0.2675
    assert entry["first_finetuned_f1"] != state["scores"][0]
