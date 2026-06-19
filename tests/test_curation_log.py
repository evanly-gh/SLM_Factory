# tests/test_curation_log.py
import os
import tempfile
from data.curation_log import CurationLog
from eval.harness import EvalResult


def _fake_eval_result():
    # EvalResult now uses per_class dict instead of spam_f1/ham_f1
    return EvalResult(
        f1=0.85,
        per_class={"spam": 0.83, "ham": 0.87},
        pos_score=0.9,
        neg_score=0.95,
        boundary_score=0.7,
        failures=[{"text": "promo", "label": "ham", "predicted": "spam"}],
    )


def test_write_and_read_iteration():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        path = f.name
    try:
        log = CurationLog(path)
        log.write_iteration(
            iteration=1,
            task_type="classification",
            dataset_version="v1",
            n_gold=100,
            n_hard=50,
            label_dist={"spam": 50, "ham": 100},
            config_a="LoRA r=8 lr=2e-4 epochs=3",
            config_b="full_ft lr=1e-4 epochs=5",
            best_config="A",
            eval_result=_fake_eval_result(),
            score_band="0.80-0.95",
            next_intervention="hyperparameter tuning",
            hypothesis="Increase epochs to reduce underfitting",
            model_id="Qwen/Qwen3-0.6B",
            int4_size_mb=397,
            tier=1,
        )
        content = log.read_latest()
        assert "Iteration 1" in content
        assert "v1" in content
        assert "0.85" in content
        assert "classification" in content
    finally:
        os.unlink(path)
