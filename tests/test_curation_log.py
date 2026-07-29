from data.curation_log import CurationLog
from eval.harness import EvalResult


def test_default_curation_log_path_uses_explicit_run_environment(
    tmp_path,
    monkeypatch,
):
    expected = tmp_path / "run-a" / "data-curation.md"
    monkeypatch.setenv("SLM_CURATION_LOG_PATH", str(expected))

    assert CurationLog().path == str(expected)


def test_curation_log_persists_complete_dataset_composition(tmp_path):
    path = tmp_path / "data-curation.md"
    log = CurationLog(str(path))
    result = EvalResult(
        f1=0.8,
        per_class={},
        pos_score=0.8,
        neg_score=0.8,
        boundary_score=0.8,
        failures=[{
            "text": "raw held-out prompt secret",
            "label": "gold",
            "predicted": "raw held-out prediction secret",
        }],
    )

    log.write_iteration(
        iteration=2,
        task_type="classification",
        dataset_version="v3",
        total_examples=12,
        n_gold=5,
        n_hard=3,
        n_hard_source=2,
        n_hard_generated=3,
        replay_count=1,
        rebuild_plan_identity="plan-abc",
        strategy_composition=[
            {"strategy": "resample_existing", "rows": 5},
            {"strategy": "mine_new_real_source", "rows": 2},
        ],
        source_novelty={
            "requested": 5,
            "novel_rows": 2,
            "novel_fraction": 0.4,
        },
        plan_yield={
            "status": "novel",
            "final_rows": 12,
            "novel_rows": 2,
        },
        confusion_pairs=[
            {"gold": "a", "predicted": "b", "count": 3},
        ],
        label_dist={"a": 6, "b": 6},
        config_a="a",
        config_b="b",
        best_config="a",
        eval_result=result,
        score_band="0.80-0.95",
        next_intervention="hyperparameter",
        hypothesis="test",
        model_id="test/model@Q4_K_M",
        size_mb=100,
        tier=0,
    )

    text = path.read_text()
    assert "- Total examples: 12" in text
    assert "- Initial gold: 5" in text
    assert "- Source anchors: 2" in text
    assert "- Generated hard rows: 3" in text
    assert "- Replay rows: 1" in text
    assert "- Rebuild plan identity: plan-abc" in text
    assert "- Strategy composition:" in text
    assert "mine_new_real_source" in text
    assert "- Source novelty:" in text
    assert "- Plan yield:" in text
    assert "a→b: 3" in text
    assert "raw held-out prompt secret" not in text
    assert "raw held-out prediction secret" not in text


def test_curation_log_iteration_is_retry_idempotent(tmp_path):
    path = tmp_path / "data-curation.md"
    log = CurationLog(str(path))
    result = EvalResult(0.8, {}, 0.8, 0.8, 0.8, [])
    kwargs = {
        "iteration": 2,
        "task_type": "classification",
        "dataset_version": "v3",
        "n_gold": 5,
        "n_hard": 3,
        "label_dist": {"a": 5, "b": 3},
        "config_a": "a",
        "config_b": "b",
        "best_config": "a",
        "eval_result": result,
        "score_band": "0.80-0.95",
        "next_intervention": "hyperparameter",
        "hypothesis": "test",
        "model_id": "test/model",
        "size_mb": 100,
        "tier": 0,
        "entry_id": "test/model@Q4_K_M:2:/weights",
    }

    log.write_iteration(**kwargs)
    log.write_iteration(**kwargs)

    text = path.read_text(encoding="utf-8")
    assert text.count("## Iteration 2") == 1
    assert text.count("slm-curation-entry:") == 1
