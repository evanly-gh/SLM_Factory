from dataclasses import asdict

from agent.data_rebuild import (
    data_rebuild_plan_identity,
    normalize_data_rebuild_plan,
)
from agent.nodes.rollback import rollback_node
from eval.harness import EvalResult


def _plan(primary, hypothesis):
    return normalize_data_rebuild_plan(
        {
            "primary_strategy": primary,
            "target_rows": 64,
            "elite": {
                "provenance": "best_non_pruned_dataset",
                "dataset_version": 1,
            },
        },
        task_type="classification",
        hypothesis=hypothesis,
    )


def test_rollback_restores_matching_dataset_plan_and_evaluation_state(tmp_path):
    winning_path = tmp_path / "dataset-v1.jsonl"
    winning_path.write_text('{"text":"winner","label":"a"}\n')
    regressed_path = tmp_path / "dataset-v2.jsonl"
    regressed_path.write_text('{"text":"regressed","label":"b"}\n')
    winning_plan = _plan("resample_existing", "winning data balance")
    regressed_plan = _plan(
        "difficulty_weighted_sampling",
        "regressed hard weighting",
    )
    winning_eval = EvalResult(
        f1=0.8,
        per_class={"a": 0.8},
        pos_score=0.8,
        neg_score=0.8,
        boundary_score=0.8,
        failures=[],
    )
    regressed_eval = EvalResult(
        f1=0.7,
        per_class={"a": 0.7},
        pos_score=0.7,
        neg_score=0.7,
        boundary_score=0.7,
        failures=[],
    )
    winning_curation = {
        "total_examples": 10,
        "data_rebuild_plan": winning_plan,
        "data_rebuild_plan_identity": data_rebuild_plan_identity(winning_plan),
        "rebuild_config": {"seed": 11},
        "strategy_composition": [{
            "strategy": "resample_existing",
            "rows": 10,
        }],
        "plan_yield": {"status": "novel", "novel_rows": 10},
    }
    regressed_curation = {
        "total_examples": 12,
        "data_rebuild_plan": regressed_plan,
        "data_rebuild_plan_identity": data_rebuild_plan_identity(regressed_plan),
        "rebuild_config": {"seed": 12},
        "strategy_composition": [{
            "strategy": "difficulty_weighted_sampling",
            "rows": 12,
        }],
        "plan_yield": {"status": "novel", "novel_rows": 2},
    }
    winning_report = {
        "overall": 0.8,
        "confusion_pairs": [
            {"gold": "a", "predicted": "b", "count": 2},
        ],
    }
    regressed_report = {
        "overall": 0.7,
        "confusion_pairs": [
            {"gold": "a", "predicted": "b", "count": 4},
        ],
    }
    state = {
        "selected_model": None,
        "scores": [0.8, 0.7],
        "best_score": 0.8,
        "best_weights_ref": "/weights-v1",
        "current_dataset_path": str(regressed_path),
        "dataset_version": 2,
        "last_curation": regressed_curation,
        "data_rebuild_plan": regressed_plan,
        "data_rebuild_plan_identity": data_rebuild_plan_identity(regressed_plan),
        "last_eval": regressed_eval,
        "test_report": regressed_report,
        "dag": [
            {
                "iteration": 1,
                "score": 0.8,
                "weights_ref": "/weights-v1",
                "best_config": "winner",
                "pruned": False,
                "pi": {
                    "D": {
                        "path": str(winning_path),
                        "version": 1,
                        "plan": winning_plan,
                        "plan_identity": data_rebuild_plan_identity(
                            winning_plan
                        ),
                        "config": winning_curation["rebuild_config"],
                        "composition": winning_curation,
                    },
                    "H": {
                        "lora_rank": 8,
                        "lora_alpha": 16,
                        "learning_rate": 2e-4,
                        "nr_epochs": 3,
                        "micro_batch_size": 2,
                        "gradient_accumulation_steps": 4,
                    },
                },
                "evaluation_state": {
                    "last_eval": asdict(winning_eval),
                    "test_report": winning_report,
                },
            },
            {
                "iteration": 2,
                "score": 0.7,
                "weights_ref": "/weights-v2",
                "best_config": "regressed",
                "pruned": False,
                "pi": {
                    "D": {
                        "path": str(regressed_path),
                        "version": 2,
                        "plan": regressed_plan,
                        "plan_identity": data_rebuild_plan_identity(
                            regressed_plan
                        ),
                        "config": regressed_curation["rebuild_config"],
                        "composition": regressed_curation,
                    },
                    "H": {
                        "lora_rank": 16,
                        "lora_alpha": 32,
                        "learning_rate": 3e-4,
                        "nr_epochs": 4,
                        "micro_batch_size": 2,
                        "gradient_accumulation_steps": 4,
                    },
                },
                "evaluation_state": {
                    "last_eval": asdict(regressed_eval),
                    "test_report": regressed_report,
                },
            },
        ],
    }

    out = rollback_node(state)

    assert out["current_dataset_path"] == str(winning_path)
    assert out["dataset_version"] == 1
    assert out["last_curation"] == winning_curation
    assert out["data_rebuild_plan"] == winning_plan
    assert (
        out["data_rebuild_plan_identity"]
        == data_rebuild_plan_identity(winning_plan)
    )
    assert out["last_eval"] == winning_eval
    assert out["test_report"] == winning_report
    assert out["best_weights_ref"] == "/weights-v1"
    assert out["best_score"] == 0.8
    assert out["dag"][-1]["pruned"] is True
