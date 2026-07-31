import json

import pytest

from agent.nodes.iterate import (
    _ITERATE_SYSTEM,
    _parse_decision_json,
    _tried_hparam_configs,
    _validate_decision_json,
)
from agent.nodes.rollback import rollback_node
from agent.nodes.train import _build_config
from training.hparams import (
    deterministic_neighbor_configs,
    hyperparameter_identity,
)
from training.lora_trainer import TrainingConfig


def _full_hparams(**overrides):
    values = {
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
        "learning_rate": 2e-4,
        "nr_epochs": 4,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 8,
    }
    values.update(overrides)
    return values


def _dag_node(
    *,
    version=1,
    path="/data-v1.jsonl",
    score=0.7,
    pruned=False,
    hparams=None,
):
    return {
        "iteration": version,
        "score": score,
        "pruned": pruned,
        "best_config": "expanded",
        "weights_ref": f"/weights-{version}",
        "pi": {
            "D": {"version": version, "path": path},
            "H": dict(hparams or _full_hparams()),
        },
    }


def _training_config(**overrides):
    values = {
        "base_model": "model",
        "nr_epochs": 4,
        "learning_rate": 2e-4,
        "lora_rank": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "weight_decay": 0.05,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 8,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_training_config_derives_bounded_effective_batch():
    config = _training_config()

    assert config.batch_size == 4
    assert config.micro_batch_size == 4
    assert config.gradient_accumulation_steps == 8
    assert config.effective_batch_size == 32


def test_training_config_accepts_legacy_batch_size_alias():
    config = _training_config(
        batch_size=2,
        micro_batch_size=None,
        gradient_accumulation_steps=4,
    )

    assert config.micro_batch_size == 2
    assert config.batch_size == 2
    assert config.effective_batch_size == 8


def test_training_config_rejects_conflicting_batch_aliases():
    with pytest.raises(ValueError, match="batch_size.*micro_batch_size"):
        _training_config(batch_size=2, micro_batch_size=4)


def test_training_config_rejects_false_effective_batch_claim():
    with pytest.raises(ValueError, match="effective_batch_size.*derived"):
        _training_config(effective_batch_size=64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lora_alpha", 48, "lora_alpha"),
        ("lora_alpha", 64.0, "lora_alpha"),
        ("lora_rank", 16.0, "lora_rank"),
        ("lora_dropout", 0.2, "lora_dropout"),
        ("weight_decay", 0.02, "weight_decay"),
        ("weight_decay", False, "weight_decay"),
        ("micro_batch_size", 3, "micro_batch_size"),
        ("micro_batch_size", 2.0, "micro_batch_size"),
        ("gradient_accumulation_steps", 3, "gradient_accumulation_steps"),
        (
            "gradient_accumulation_steps",
            True,
            "gradient_accumulation_steps",
        ),
        ("learning_rate", 9e-6, "learning_rate"),
        ("learning_rate", 5.01e-4, "learning_rate"),
        ("nr_epochs", 0, "nr_epochs"),
        ("nr_epochs", 9, "nr_epochs"),
    ],
)
def test_training_config_strictly_rejects_out_of_space_values(
    field,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        _training_config(**{field: value})


def test_iterate_parser_snaps_the_five_tunable_values():
    """Out-of-range values on the five tunable axes snap into the bounded space.

    Batch shape and dropout are no longer part of the orchestrator's choice set, so
    this no longer asserts on them — the normalized config still carries derived
    values for them, it just isn't the LLM that picked them.
    """
    decision = _parse_decision_json(json.dumps({
        "intervention": "hyperparameter",
        "hypothesis": "raise capacity and regularize harder",
        "hyperparams": {
            "lora_rank": 15,
            "alpha_ratio": 3,
            "weight_decay": 0.08,
            "learning_rate": 9e-4,
            "nr_epochs": 11,
        },
    }))

    hp = decision["hyperparams"]
    assert hp["lora_rank"] == 16
    # ratio 3 is equidistant from the allowed 2 and 4; _snap tie-breaks to the
    # smaller, so alpha = 16 x 2.
    assert hp["lora_alpha"] == 32
    assert hp["weight_decay"] == 0.1
    assert hp["learning_rate"] == 5e-4
    assert hp["nr_epochs"] == 8
    assert "snapped" in decision["hyperparam_rationale"]


def test_iterate_parser_rejects_retired_hyperparameters():
    """The four retired fields must be refused with an actionable message."""
    for field, value in (
        ("lora_alpha", 64),
        ("lora_dropout", 0.05),
        ("micro_batch_size", 4),
        ("gradient_accumulation_steps", 2),
        ("effective_batch_size", 8),
        ("batch_size", 4),
    ):
        with pytest.raises(ValueError, match="no longer tunable"):
            _parse_decision_json(json.dumps({
                "intervention": "hyperparameter",
                "hypothesis": "try a retired field",
                "hyperparams": {"lora_rank": 16, field: value},
            }))


def test_iterate_parser_rejects_legacy_batch_size_alias():
    """batch_size used to be accepted as a micro_batch_size alias; batch shape is no
    longer the orchestrator's to set, so the alias is refused rather than honored."""
    with pytest.raises(ValueError, match="no longer tunable"):
        _parse_decision_json(
            '{"intervention":"hyperparameter",'
            '"hypothesis":"use a legacy batch alias","hyperparams":'
            '{"lora_rank":8,"batch_size":4}}'
        )


def test_iterate_parser_rejects_conflicting_batch_aliases_actionably():
    with pytest.raises(ValueError, match="batch_size.*micro_batch_size"):
        _parse_decision_json(
            '{"intervention":"hyperparameter",'
            '"hypothesis":"test conflicting aliases","hyperparams":'
            '{"batch_size":2,"micro_batch_size":4}}'
        )


def test_iterate_parser_rejects_false_effective_batch_claim():
    with pytest.raises(ValueError, match="effective_batch_size.*derived"):
        _parse_decision_json(
            '{"intervention":"hyperparameter",'
            '"hypothesis":"test derived batch","hyperparams":'
            '{"micro_batch_size":2,"gradient_accumulation_steps":4,'
            '"effective_batch_size":16}}'
        )


def test_iterate_parser_rejects_unsupported_optimizer_fields():
    with pytest.raises(ValueError, match="unsupported hyperparameter field.*momentum"):
        _parse_decision_json(
            '{"intervention":"hyperparameter",'
            '"hypothesis":"test unsupported optimizer field","hyperparams":'
            '{"lora_rank":8,"momentum":0.9}}'
        )


@pytest.mark.parametrize(
    ("adjustment", "message"),
    [
        ([], "threshold_adjustment must be a JSON object"),
        ({"new_threshold": "0.8", "reason": "capacity"}, "finite numeric"),
        ({"new_threshold": True, "reason": "capacity"}, "finite numeric"),
        ({"new_threshold": float("nan"), "reason": "capacity"}, "finite numeric"),
        ({"new_threshold": float("inf"), "reason": "capacity"}, "finite numeric"),
        ({"new_threshold": 0.8}, "non-empty reason"),
        ({"new_threshold": 0.8, "reason": "   "}, "non-empty reason"),
    ],
)
def test_iterate_rejects_malformed_threshold_adjustment(adjustment, message):
    with pytest.raises(ValueError, match=message):
        _validate_decision_json({
            "intervention": "data_rebuild",
            "hypothesis": "validate threshold payload",
            "data_rebuild": {
                "strategy": "resample",
            },
            "threshold_adjustment": adjustment,
        })


def test_iterate_accepts_null_or_reasoned_finite_threshold_adjustment():
    assert _validate_decision_json({
        "intervention": "data_rebuild",
        "hypothesis": "no threshold adjustment needed",
        "data_rebuild": {
            "strategy": "resample",
        },
        "threshold_adjustment": {"new_threshold": None},
    })["threshold_adjustment"]["new_threshold"] is None
    assert _validate_decision_json({
        "intervention": "data_rebuild",
        "hypothesis": "capacity bounds the remaining score",
        "data_rebuild": {
            "strategy": "resample",
        },
        "threshold_adjustment": {
            "new_threshold": 0.85,
            "reason": "remaining failures are out of distribution",
        },
    })["threshold_adjustment"]["new_threshold"] == 0.85


@pytest.mark.parametrize(
    "decision",
    [
        {"intervention": "hyperparameter"},
        {"intervention": "unknown"},
    ],
)
def test_iterate_rejects_missing_intervention_payloads(decision):
    with pytest.raises(ValueError):
        _validate_decision_json(decision)


def test_iterate_rejects_removed_intervention_even_with_legacy_payload():
    removed_intervention = "sur" + "gical"
    removed_field = "targeted_" + "patterns"
    with pytest.raises(ValueError, match="unsupported|data_rebuild.*hyperparameter"):
        _validate_decision_json({
            "intervention": removed_intervention,
            removed_field: "remaining boundary failures",
        })


def test_iterate_normalizes_bounded_declarative_data_rebuild_plan():
    decision = _validate_decision_json({
        "intervention": "data_rebuild",
        "hypothesis": "hard examples and a→b confusion are underrepresented",
        "data_rebuild": {
            "strategy": "synthesize",
            "target_rows": 99999,
            "resample_fraction": 0.63,
            "new_real_rows": 37,
            "synth_rows": 19,
            "max_acquire_rounds": 99,
            "difficulty_buckets": {
                "easy": 0.1,
                "medium": 0.2,
                "hard": 0.7,
            },
            "confusion_pairs": [
                {"gold": "a", "predicted": "b", "count": 1_000_000},
            ],
            "pattern_hint": "aggregate a→b confusion",
        },
    }, task_type="classification")

    from config.config import DATA_SIZE_CEILING

    plan = decision["data_rebuild"]
    assert plan["strategy"] == "synthesize"
    # target_rows now clamps to DATA_SIZE_CEILING, not the old 2000 cap.
    assert plan["target_rows"] == DATA_SIZE_CEILING
    assert plan["resample_fraction"] == pytest.approx(0.65)
    assert plan["new_real_rows"] == 35
    assert plan["synth_rows"] == 20
    assert plan["max_acquire_rounds"] == 3
    assert sum(plan["difficulty_buckets"].values()) == pytest.approx(1.0)
    assert plan["confusion_pairs"] == [
        {"gold": "a", "predicted": "b", "count": 10_000},
    ]
    assert "hypothesis" in plan["pattern_hint"]
    assert "data_rebuild_plan_identity" not in decision


def test_iterate_schema_documents_the_five_tunable_axes():
    """The prompt must name exactly the tunable axes and say the rest are not."""
    for field in (
        "lora_rank",
        "alpha_ratio",
        "weight_decay",
        "learning_rate",
        "nr_epochs",
    ):
        assert field in _ITERATE_SYSTEM
    assert "EXACTLY FIVE hyperparameters are tunable" in _ITERATE_SYSTEM
    # The retired fields must be named as NOT tunable, so the model doesn't try them.
    assert "is NOT yours to set" in _ITERATE_SYSTEM
    assert "lora_dropout is NOT tunable" in _ITERATE_SYSTEM
    assert "[1e-5, 5e-4]" in _ITERATE_SYSTEM
    assert "[1, 8]" in _ITERATE_SYSTEM
    assert "exact (dataset, hyperparameter) repeat" in _ITERATE_SYSTEM


def test_data_change_carries_every_optimizer_field_forward():
    winning = _full_hparams()
    state = {
        "dag": [_dag_node(hparams=winning)],
        "dataset_version": 2,
        "current_dataset_path": "/data-v2.jsonl",
        "last_intervention": "data_rebuild",
        "llm_iterate_decision": {"intervention": "data_rebuild"},
    }

    config, reason = _build_config(state)

    for key, value in winning.items():
        assert config[key] == value
    assert config["batch_size"] == winning["micro_batch_size"]
    assert "carry-forward" in reason
    assert "effective batch=8" in reason


def test_same_hparams_are_allowed_on_a_new_dataset_identity():
    winning = _full_hparams()
    state = {
        "dag": [_dag_node(version=1, path="/data-v1.jsonl", hparams=winning)],
        "dataset_version": 2,
        "current_dataset_path": "/data-v2.jsonl",
        "last_intervention": "data_rebuild",
        "llm_iterate_decision": {"intervention": "data_rebuild"},
    }

    config, _ = _build_config(state)

    assert hyperparameter_identity(config) == hyperparameter_identity(winning)


def test_exact_repeat_on_same_dataset_is_rejected_and_replaced():
    tried = _full_hparams()
    state = {
        "dag": [_dag_node(hparams=tried)],
        "dataset_version": 1,
        "current_dataset_path": "/data-v1.jsonl",
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": {
            "intervention": "hyperparameter",
            "hypothesis": "try it again",
            "hyperparams": dict(tried),
        },
    }

    config, reason = _build_config(state)

    assert hyperparameter_identity(config) != hyperparameter_identity(tried)
    assert "rejected exact repeat" in reason
    assert "effective batch=" in reason


def test_pruned_config_counts_as_tried_for_exact_repeat_identity():
    tried = _full_hparams()
    state = {
        "dag": [_dag_node(pruned=True, hparams=tried)],
        "dataset_version": 1,
        "current_dataset_path": "/data-v1.jsonl",
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": {
            "intervention": "hyperparameter",
            "hyperparams": dict(tried),
        },
    }

    config, reason = _build_config(state)

    assert hyperparameter_identity(config) != hyperparameter_identity(tried)
    assert "pruned configurations also count as tried" in reason


def test_configless_fallback_uses_an_untried_expanded_identity():
    best = _full_hparams(lora_rank=16, lora_alpha=32)
    tried_nodes = [
        _dag_node(
            hparams=_full_hparams(
                lora_rank=rank,
                lora_alpha=rank * 2,
            ),
            score=0.9 if rank == 16 else 0.5,
        )
        for rank in (4, 8, 16, 32, 64)
    ]
    state = {
        "dag": tried_nodes,
        "dataset_version": 1,
        "current_dataset_path": "/data-v1.jsonl",
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": None,
    }

    config, reason = _build_config(state)
    tried_identities = {
        hyperparameter_identity(node["pi"]["H"])
        for node in tried_nodes
    }

    assert hyperparameter_identity(best) in tried_identities
    assert hyperparameter_identity(config) not in tried_identities
    assert "deterministic untried" in reason


def test_deterministic_fallback_can_reach_multi_axis_combinations():
    base = _full_hparams()

    candidates = list(deterministic_neighbor_configs(base))

    # Multi-axis reach is now across the FIVE tunable axes. dropout and batch shape
    # were removed from the ladder, so the multi-axis pair asserted here is
    # rank + weight_decay rather than dropout + weight_decay.
    assert any(
        candidate["lora_rank"] != base["lora_rank"]
        and candidate["weight_decay"] != base["weight_decay"]
        for candidate in candidates
    )
    # And the removed axes must stay pinned to the base value.
    assert {c["lora_dropout"] for c in candidates[:300]} == {base["lora_dropout"]}
    assert {c["micro_batch_size"] for c in candidates[:300]} == {base["micro_batch_size"]}


def test_tried_prompt_identity_includes_every_field_and_dataset():
    tried = _tried_hparam_configs({
        "dag": [
            _dag_node(
                version=3,
                path="/data-v3.jsonl",
                hparams=_full_hparams(),
                pruned=True,
            )
        ]
    })

    assert tried == [{
        **_full_hparams(),
        "batch_size": 2,
        "dataset_version": 3,
        "dataset_path": "/data-v3.jsonl",
        "score": 0.7,
        "pruned": True,
    }]


def test_nonwinning_trained_candidate_still_counts_as_tried():
    trained = _full_hparams()
    baseline_node = {
        "iteration": 1,
        "score": 0.9,
        "best_config": "baseline (zero-shot, no adapter)",
        "pruned": False,
        "pi": {
            "D": {"version": 1, "path": "/data-v1.jsonl"},
            "H": {"lora_rank": None},
        },
        "trained_configs": [{
            "label": "candidate",
            "score": 0.7,
            "H": trained,
        }],
    }
    state = {
        "dag": [baseline_node],
        "dataset_version": 1,
        "current_dataset_path": "/data-v1.jsonl",
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": {
            "intervention": "hyperparameter",
            "hyperparams": trained,
        },
    }

    tried = _tried_hparam_configs(state)
    replacement, reason = _build_config(state)

    assert len(tried) == 1
    assert hyperparameter_identity(tried[0]) == hyperparameter_identity(trained)
    assert hyperparameter_identity(replacement) != hyperparameter_identity(trained)
    assert "rejected exact repeat" in reason


def test_configless_fallback_does_not_repeat_candidate_that_lost_to_baseline():
    default_candidate = {
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "weight_decay": 0.01,
        "learning_rate": 2e-4,
        "nr_epochs": 3,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 8,
    }
    state = {
        "dag": [{
            "iteration": 1,
            "score": 0.9,
            "best_config": "baseline (zero-shot, no adapter)",
            "pruned": False,
            "pi": {
                "D": {"version": 1, "path": "/data-v1.jsonl"},
                "H": {"lora_rank": None},
            },
            "trained_configs": [{
                "label": "default",
                "score": 0.7,
                "H": default_candidate,
            }],
        }],
        "dataset_version": 1,
        "current_dataset_path": "/data-v1.jsonl",
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": None,
    }

    replacement, reason = _build_config(state)

    assert (
        hyperparameter_identity(replacement)
        != hyperparameter_identity(default_candidate)
    )
    assert "deterministic untried" in reason


def test_data_rebuild_carries_nonwinning_candidate_fields_after_baseline():
    trained = _full_hparams(
        lora_alpha=64,
        lora_dropout=0.1,
        weight_decay=0.05,
        micro_batch_size=1,
        gradient_accumulation_steps=8,
        effective_batch_size=8,
    )
    state = {
        "dag": [{
            "iteration": 1,
            "score": 0.9,
            "best_config": "baseline (zero-shot, no adapter)",
            "pruned": False,
            "pi": {
                "D": {"version": 1, "path": "/data-v1.jsonl"},
                "H": {"lora_rank": None},
            },
            "trained_configs": [{
                "label": "candidate",
                "score": 0.7,
                "H": trained,
            }],
        }],
        "dataset_version": 2,
        "current_dataset_path": "/data-v2.jsonl",
        "last_intervention": "data_rebuild",
        "llm_iterate_decision": {"intervention": "data_rebuild"},
    }

    carried, reason = _build_config(state)

    assert hyperparameter_identity(carried) == hyperparameter_identity(trained)
    assert "carry-forward" in reason


def test_rollback_carries_winning_full_config_not_pruned_config():
    winner = _full_hparams(
        lora_alpha=64,
        lora_dropout=0.1,
        weight_decay=0.05,
        micro_batch_size=1,
        gradient_accumulation_steps=8,
        effective_batch_size=8,
    )
    regressed = _full_hparams(
        lora_alpha=16,
        lora_dropout=0.0,
        weight_decay=0.1,
        micro_batch_size=8,
        gradient_accumulation_steps=1,
        effective_batch_size=8,
    )
    state = {
        "selected_model": None,
        "scores": [0.8, 0.7],
        "best_score": 0.8,
        "best_weights_ref": "/weights-1",
        "dag": [
            _dag_node(score=0.8, hparams=winner),
            _dag_node(version=2, score=0.7, hparams=regressed),
        ],
        "dataset_version": 2,
        "current_dataset_path": "/data-v2.jsonl",
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": {"intervention": "hyperparameter"},
    }

    rolled_back = rollback_node(state)
    config, reason = _build_config(rolled_back)

    assert rolled_back["dag"][-1]["pruned"] is True
    for key, value in winner.items():
        assert config[key] == value
    assert "carry-forward" in reason
