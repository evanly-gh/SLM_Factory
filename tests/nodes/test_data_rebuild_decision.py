from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.data_rebuild import (
    DATA_REBUILD_STRATEGIES,
    data_rebuild_plan_identity,
    fallback_data_rebuild_plan,
    normalize_data_rebuild_plan,
)
from agent.nodes.iterate import (
    _ITERATE_SYSTEM,
    _parse_decision_json,
    _llm_iterate,
    _validate_decision_json,
    apply_iteration_policy,
)
from config.android_pool import ANDROID_POOL


def _raw_plan(primary="resample_existing", supports=None):
    return {
        "primary_strategy": primary,
        "support_strategies": list(supports or []),
        "target_rows": 64,
        "resample_fraction": 0.65,
        "preserve_elite_fraction": 0.25,
        "new_real_rows": 20,
        "synth_rows": 10,
        "max_acquire_rounds": 2,
        "query_variant": 3,
        "difficulty_buckets": {
            "easy": 0.2,
            "medium": 0.3,
            "hard": 0.5,
        },
        "confusion_pairs": [
            {"gold": "a", "predicted": "b", "count": 4},
        ],
        "pattern_hint": "aggregate a to b confusion",
        "elite": {
            "provenance": "best_non_pruned_dataset",
            "dataset_version": 2,
        },
    }


def test_schema_lists_exact_supported_strategies():
    assert DATA_REBUILD_STRATEGIES == (
        "resample_existing",
        "preserve_elite_resample",
        "mine_new_real_source",
        "source_diversification",
        "difficulty_weighted_sampling",
        "targeted_synth_positive",
    )


@pytest.mark.parametrize(
    "task_type",
    ["math_reasoning", "code_generation", "generation"],
)
def test_positive_synthesis_is_ineligible_without_verifiable_labels(task_type):
    with pytest.raises(ValueError, match="not eligible"):
        normalize_data_rebuild_plan(
            _raw_plan("targeted_synth_positive"),
            task_type=task_type,
            hypothesis="remaining errors need positive synthesis",
        )


@pytest.mark.parametrize("task_type", ["classification", "NER"])
def test_positive_synthesis_is_eligible_for_classification_and_ner(task_type):
    plan = normalize_data_rebuild_plan(
        _raw_plan("targeted_synth_positive"),
        task_type=task_type,
        hypothesis="remaining aggregate confusion needs positive examples",
    )
    assert plan["primary_strategy"] == "targeted_synth_positive"


def test_schema_rejects_non_declarative_unknown_fields():
    raw = _raw_plan()
    raw["python"] = "lambda rows: rows"
    with pytest.raises(ValueError, match="unsupported data_rebuild field.*python"):
        normalize_data_rebuild_plan(
            raw,
            task_type="classification",
            hypothesis="rebalance",
        )


@pytest.mark.parametrize(
    ("primary", "overrides"),
    [
        ("preserve_elite_resample", {"preserve_elite_fraction": 0}),
        ("mine_new_real_source", {"new_real_rows": 0}),
        ("targeted_synth_positive", {"synth_rows": 0}),
    ],
)
def test_strategy_specific_material_budgets_must_be_positive(
    primary,
    overrides,
):
    with pytest.raises(ValueError, match="positive material budget"):
        normalize_data_rebuild_plan(
            _raw_plan(primary) | overrides,
            task_type="classification",
            hypothesis="exercise the selected strategy",
        )


def test_sampling_strategies_cannot_be_noop_supports_or_compose_together():
    with pytest.raises(ValueError, match="sampling strategy"):
        normalize_data_rebuild_plan(
            _raw_plan(
                "source_diversification",
                ["difficulty_weighted_sampling"],
            ),
            task_type="classification",
            hypothesis="diversify and weight",
        )
    with pytest.raises(ValueError, match="resample_existing.*primary"):
        normalize_data_rebuild_plan(
            _raw_plan(
                "difficulty_weighted_sampling",
                ["resample_existing"],
            ),
            task_type="classification",
            hypothesis="weight hard rows",
        )


def test_composed_material_budgets_cannot_exceed_final_target():
    with pytest.raises(ValueError, match="material budgets.*target_rows"):
        normalize_data_rebuild_plan(
            _raw_plan(
                "targeted_synth_positive",
                ["preserve_elite_resample", "mine_new_real_source"],
            ) | {
                "target_rows": 16,
                "synth_rows": 10,
                "new_real_rows": 10,
                "preserve_elite_fraction": 0.5,
            },
            task_type="classification",
            hypothesis="compose bounded additions",
        )


def test_difficulty_strategy_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="positive difficulty"):
        normalize_data_rebuild_plan(
            _raw_plan("difficulty_weighted_sampling") | {
                "difficulty_buckets": {
                    "easy": 0,
                    "medium": 0,
                    "hard": 0,
                },
            },
            task_type="classification",
            hypothesis="weight difficulty",
        )


def test_elite_strategy_requires_resolvable_declared_source():
    with pytest.raises(ValueError, match="elite source.*not resolvable"):
        _validate_decision_json(
            {
                "intervention": "data_rebuild",
                "hypothesis": "preserve prior winners",
                "data_rebuild": _raw_plan("preserve_elite_resample"),
            },
            task_type="classification",
            state={
                "dataset_version": 2,
                "current_dataset_path": "/missing/dataset.jsonl",
                "dag": [],
            },
        )


def test_plan_rejects_raw_eval_text_disguised_as_pattern_hint():
    secret = "held out evaluation sentence must remain private"
    state = {
        "task_type": "classification",
        "eval_set": SimpleNamespace(all=[{"text": secret, "label": "a"}]),
        "dataset_version": 1,
        "dag": [],
    }
    raw = _raw_plan()
    raw["pattern_hint"] = secret

    with pytest.raises(ValueError, match="raw eval text"):
        _validate_decision_json({
            "intervention": "data_rebuild",
            "hypothesis": "aggregate confusion requires refinement",
            "data_rebuild": raw,
        }, task_type="classification", state=state)


def test_hyperparameter_hypothesis_cannot_copy_raw_eval_text():
    secret = "another held out sentence that must remain private"
    with pytest.raises(ValueError, match="raw eval text"):
        _validate_decision_json({
            "intervention": "hyperparameter",
            "hypothesis": secret,
            "hyperparams": {"lora_rank": 16},
        }, state={
            "eval_set": SimpleNamespace(
                all=[{"text": secret, "label": "a"}]
            ),
        })


def test_every_iterate_decision_requires_a_nonempty_hypothesis():
    with pytest.raises(ValueError, match="hypothesis"):
        _validate_decision_json({
            "intervention": "hyperparameter",
            "hyperparams": {"lora_rank": 16},
        })


@pytest.mark.parametrize(
    "decision",
    [
        {
            "intervention": "data_rebuild",
            "hypothesis": "rebalance",
            "data_rebuild": {"primary_strategy": "resample_existing"},
            "shell_command": "rm -rf /",
        },
        {
            "intervention": "data_rebuild",
            "hypothesis": "rebalance",
            "data_rebuild": {"primary_strategy": "resample_existing"},
            "data_rebuild_plan_identity": "provider-injected",
        },
        {
            "intervention": "data_rebuild",
            "hypothesis": "rebalance",
            "data_rebuild": {"primary_strategy": "resample_existing"},
            "threshold_adjustment": {
                "new_threshold": None,
                "reason": "",
                "extra": "not allowed",
            },
        },
    ],
)
def test_decision_schema_rejects_unknown_top_level_and_nested_keys(decision):
    with pytest.raises(ValueError, match="unsupported"):
        _validate_decision_json(decision)


def test_hyperparameter_decision_rejects_a_data_rebuild_payload():
    """A hyperparameter decision may not smuggle a data plan alongside it."""
    with pytest.raises(ValueError, match="not allowed"):
        _validate_decision_json({
            "intervention": "hyperparameter",
            "hypothesis": "increase rank",
            "hyperparams": {"lora_rank": 16},
            "data_rebuild": {"primary_strategy": "resample_existing"},
        })


def test_data_rebuild_decision_strips_stray_hyperparams_instead_of_rejecting():
    """The invariant is 'a data_rebuild does not also change hyperparameters'.

    Stripping enforces that exactly. Raising did not — it threw away the whole data
    plan and fell through to a hard-coded heuristic that ALSO held hyperparameters
    fixed, so the invariant was never what was at stake. Claude attaches a
    `hyperparams` block to essentially every data_rebuild it proposes, so rejecting
    meant the NER run executed zero orchestrator-authored data plans in 142
    iterations while still logging them as orchestrator decisions.
    """
    decision = _validate_decision_json({
        "intervention": "data_rebuild",
        "hypothesis": "rebalance toward the failing buckets",
        "data_rebuild": {"primary_strategy": "resample_existing"},
        "hyperparams": {"lora_rank": 16},
    })

    assert decision["intervention"] == "data_rebuild"
    assert "hyperparams" not in decision
    assert "hyperparam_rationale" not in decision
    # The data plan itself survives — that is the whole point.
    assert decision["data_rebuild"]["primary_strategy"] == "resample_existing"


@pytest.mark.parametrize(
    "decision",
    [
        {
            "intervention": "hyperparameter",
            "hypothesis": "increase rank",
            "hyperparams": {"lora_rank": "16"},
        },
        {
            "intervention": "data_rebuild",
            "hypothesis": "rebalance",
            "data_rebuild": {
                "schema_version": True,
                "primary_strategy": "resample_existing",
            },
        },
    ],
)
def test_decision_schema_rejects_wrong_scalar_types(decision):
    with pytest.raises(ValueError, match="type|integer|numeric|schema_version"):
        _validate_decision_json(decision)


@pytest.mark.parametrize(
    "field_update",
    [
        {
            "threshold_adjustment": {
                "new_threshold": 0.8,
                "reason": "held out evaluation sentence must remain private",
            },
        },
        {
            "data_rebuild": {
                **_raw_plan(),
                "confusion_pairs": [{
                    "gold": "held out evaluation sentence must remain private",
                    "predicted": "b",
                    "count": 1,
                }],
            },
        },
    ],
)
def test_all_decision_strings_reject_normalized_eval_text(field_update):
    secret = "held out evaluation sentence must remain private"
    decision = {
        "intervention": "data_rebuild",
        "hypothesis": "aggregate refinement",
        "data_rebuild": _raw_plan(),
        **field_update,
    }
    with pytest.raises(ValueError, match="raw eval text"):
        _validate_decision_json(
            decision,
            state={
                "eval_set": SimpleNamespace(
                    all=[{"text": "  HELD OUT\n evaluation sentence must remain private "}]
                ),
            },
        )


def test_parser_rejects_python_literal_decisions():
    with pytest.raises(ValueError, match="parseable JSON"):
        _parse_decision_json(
            "{'intervention':'hyperparameter',"
            "'hypothesis':'increase rank',"
            "'hyperparams':{'lora_rank':16}}"
        )


def test_plan_identity_is_deterministic_and_sensitive_to_action_fields():
    plan = normalize_data_rebuild_plan(
        _raw_plan(),
        task_type="classification",
        hypothesis="rebalance",
    )
    reordered = dict(reversed(list(plan.items())))
    assert data_rebuild_plan_identity(plan) == data_rebuild_plan_identity(reordered)
    assert normalize_data_rebuild_plan(
        plan,
        task_type="classification",
        hypothesis="rebalance",
    ) == plan

    changed = dict(plan)
    changed["query_variant"] = 4
    assert data_rebuild_plan_identity(plan) != data_rebuild_plan_identity(changed)


def test_pruned_and_zero_yield_plans_count_as_tried_and_exact_repeat_rotates():
    hypothesis = "the source yielded no novel rows"
    plan = normalize_data_rebuild_plan(
        _raw_plan(),
        task_type="classification",
        hypothesis=hypothesis,
    )
    identity = data_rebuild_plan_identity(plan)
    state = {
        "task_type": "classification",
        "dataset_version": 2,
        "curriculum_size_target": 64,
        "source_acquire_rounds_used": 1,
        "dag": [{
            "score": 0.8,
            "pruned": True,
            "pi": {
                "D": {
                    "plan": plan,
                    "plan_identity": identity,
                    "composition": {
                        "plan_yield": {"status": "no_novelty"},
                    },
                },
            },
        }],
    }

    decision = _validate_decision_json({
        "intervention": "data_rebuild",
        "hypothesis": hypothesis,
        "data_rebuild": _raw_plan(),
    }, task_type="classification", state=state)

    assert decision["data_rebuild_plan_identity"] != identity
    assert decision["data_rebuild"]["query_variant"] != plan["query_variant"]
    assert decision["data_rebuild_trial_counts"] == {
        "tried": 1,
        "pruned": 1,
        "zero_yield": 1,
    }


def test_prior_tier_plan_identities_remain_tried_after_dag_reset():
    hypothesis = "repeat across model tiers"
    plan = normalize_data_rebuild_plan(
        _raw_plan(),
        task_type="classification",
        hypothesis=hypothesis,
    )
    identity = data_rebuild_plan_identity(plan)
    state = {
        "task_type": "classification",
        "dataset_version": 2,
        "curriculum_size_target": 64,
        "dag": [],
        "escalation_history": [{
            "selector": "old/model@Q4_K_M",
            "dag": [{
                "score": 0.7,
                "pruned": False,
                "pi": {
                    "D": {
                        "plan": plan,
                        "plan_identity": identity,
                        "composition": {
                            "plan_yield": {"status": "novel"},
                        },
                    },
                },
            }],
        }],
    }

    decision = _validate_decision_json({
        "intervention": "data_rebuild",
        "hypothesis": hypothesis,
        "data_rebuild": _raw_plan(),
    }, task_type="classification", state=state)

    assert decision["data_rebuild_plan_identity"] != identity
    assert decision["data_rebuild_trial_counts"]["tried"] == 1


def test_high_score_fallback_refines_with_data_rebuild_not_removed_route():
    policy = apply_iteration_policy(0.97)
    assert policy["intervention"] == "data_rebuild"
    assert policy["data_rebuild_strategy"] == "targeted_synth_positive"

    plan = fallback_data_rebuild_plan(
        {
            "task_type": "classification",
            "scores": [0.97],
            "dataset_version": 2,
            "curriculum_size_target": 80,
            "dag": [],
            "test_report": {
                "confusion_pairs": [
                    {"gold": "a", "predicted": "b", "count": 3},
                ],
            },
        },
        hypothesis="remaining aggregate a to b confusion",
        score=0.97,
    )
    assert plan["primary_strategy"] == "targeted_synth_positive"
    assert plan["max_acquire_rounds"] == 0


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_iterate_prompt_uses_only_aggregate_eval_and_budget_context():
    captured = {}
    raw_eval_secret = "HELD OUT SECRET EXAMPLE 4921"

    def fake_invoke(_llm, messages, **_kwargs):
        captured["system"] = messages[0].content
        captured["prompt"] = messages[1].content
        return MagicMock(
            tool_calls=[],
            content=(
                '{"intervention":"data_rebuild",'
                '"hypothesis":"hard bucket is underrepresented",'
                '"data_rebuild":{"primary_strategy":'
                '"difficulty_weighted_sampling"}}'
            ),
        )

    chat = MagicMock()
    prior_plan = normalize_data_rebuild_plan(
        _raw_plan(),
        task_type="classification",
        hypothesis="prior hypothesis",
    )
    prior_identity = data_rebuild_plan_identity(prior_plan)
    state = {
        "task_type": "classification",
        "selected_model": ANDROID_POOL[0],
        "iteration": 3,
        "turn_budget": 20,
        "scores": [0.7, 0.78],
        "best_score": 0.78,
        "stop_threshold": 0.9,
        "initial_stop_threshold": 0.9,
        "dataset_version": 2,
        "current_dataset_path": "/dataset-v2.jsonl",
        "last_hypothesis": "prior aggregate hypothesis",
        "last_eval": SimpleNamespace(
            failures=[{
                "text": raw_eval_secret,
                "label": "a",
                "predicted": "b",
            }],
        ),
        "test_report": {
            "overall": 0.78,
            "by_difficulty": {
                "easy": {"n": 10, "accuracy": 0.9},
                "medium": {"n": 10, "accuracy": 0.7},
                "hard": {"n": 10, "accuracy": 0.4},
            },
            "diagnosis": "hard bucket is weak",
            "suggested_intervention": "data_rebuild",
            "confusion_pairs": [
                {"gold": "a", "predicted": "b", "count": 6},
            ],
        },
        "last_curation": {
            "source_novelty": {
                "requested": 20,
                "novel_rows": 7,
                "novel_fraction": 0.35,
            },
            "plan_yield": {
                "status": "novel",
                "novel_rows": 7,
            },
        },
        "source_acquire_rounds_used": 2,
        "dag": [{
            "score": 0.75,
            "pruned": True,
            "pi": {
                "D": {
                    "plan": prior_plan,
                    "plan_identity": prior_identity,
                    "composition": {
                        "plan_yield": {"status": "novel"},
                    },
                },
            },
        }],
    }

    with (
        patch("langchain_anthropic.ChatAnthropic", return_value=chat),
        patch("data.curation_log.CurationLog.read_latest", return_value=""),
        patch("agent.context_manager.should_compact", return_value=False),
        patch(
            "agent.nodes.iterate.tracked_chat_anthropic_invoke",
            side_effect=fake_invoke,
        ),
    ):
        _llm_iterate(state)

    rendered = captured["prompt"]
    assert "prior aggregate hypothesis" in rendered
    assert "gold='a' predicted='b' count=6" in rendered
    assert "novel_rows=7" in rendered
    assert prior_identity in rendered
    assert "tried=1" in rendered and "pruned=1" in rendered
    assert "remaining turn budget" in rendered.lower()
    assert "remaining paid acquisition rounds" in rendered.lower()
    assert raw_eval_secret not in rendered
    assert "raw eval" in captured["system"].lower()
    chat.bind_tools.assert_not_called()
    assert "bash" not in captured["system"].lower()
    assert "edit_file" not in captured["system"].lower()


def test_iterate_prompt_has_separate_branch_specific_json_examples():
    data_example = _ITERATE_SYSTEM.split(
        "Valid data_rebuild JSON example:"
    )[1].split("End data_rebuild example.")[0]
    hyperparameter_example = _ITERATE_SYSTEM.split(
        "Valid hyperparameter JSON example:"
    )[1].split("End hyperparameter example.")[0]

    assert '"intervention": "data_rebuild"' in data_example
    assert '"data_rebuild"' in data_example
    assert '"hyperparams"' not in data_example
    assert '"intervention": "hyperparameter"' in hyperparameter_example
    assert '"hyperparams"' in hyperparameter_example
    assert '"data_rebuild"' not in hyperparameter_example


def test_iterate_prompt_keeps_data_rebuild_schema_constraints():
    for strategy in DATA_REBUILD_STRATEGIES:
        assert strategy in _ITERATE_SYSTEM
    for constraint in (
        "target_rows: integer [16,2000], step 8",
        "resample_fraction: float [0.10,1.00], step 0.05",
        "preserve_elite_fraction: float [0,0.80], step 0.05",
        "new_real_rows: integer [0,500], step 5",
        "synth_rows: integer [0,200], step 5",
        "max_acquire_rounds: integer [0,3]",
        "query_variant: integer [0,7]",
    ):
        assert constraint in _ITERATE_SYSTEM


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_iterate_reasks_once_with_exact_validation_error_and_more_output_room():
    calls = []
    chat_configs = []
    first_llm = MagicMock()
    second_llm = MagicMock()

    def invoke(llm, messages, *, stage, **_kwargs):
        calls.append((llm, stage, messages))
        if stage == "iterate":
            # Prose instead of JSON — the canonical reask trigger. (This used to be a
            # data_rebuild carrying a stray hyperparams block, but that combination is
            # now stripped rather than rejected, so it no longer reaches the reask.)
            return MagicMock(
                content="Let me think about this before answering.",
                tool_calls=[],
            )
        return MagicMock(
            content=(
                '{"intervention":"hyperparameter",'
                '"hypothesis":"increase rank",'
                '"hyperparams":{"lora_rank":16}}'
            ),
            tool_calls=[],
        )

    def make_chat(**kwargs):
        chat_configs.append(kwargs)
        return [first_llm, second_llm][len(chat_configs) - 1]

    state = {
        "task_type": "classification",
        "selected_model": ANDROID_POOL[0],
        "iteration": 1,
        "turn_budget": 20,
        "scores": [0.8],
        "best_score": 0.8,
        "stop_threshold": 0.9,
        "initial_stop_threshold": 0.9,
        "dataset_version": 1,
        "current_dataset_path": "/dataset.jsonl",
        "last_hypothesis": "",
        "test_report": None,
        "last_curation": None,
        "dag": [],
    }
    with (
        patch(
            "langchain_anthropic.ChatAnthropic",
            side_effect=make_chat,
        ),
        patch("data.curation_log.CurationLog.read_latest", return_value=""),
        patch("agent.context_manager.should_compact", return_value=False),
        patch(
            "agent.nodes.iterate.tracked_chat_anthropic_invoke",
            side_effect=invoke,
        ),
    ):
        decision = _llm_iterate(state)

    assert decision["intervention"] == "hyperparameter"
    assert [stage for _, stage, _ in calls] == [
        "iterate",
        "iterate_json_reask",
    ]
    reask = calls[1][2][-1].content
    assert "no parseable JSON object in LLM response" in reask
    assert "contradictory payload sentinel" not in reask
    assert [config["max_tokens"] for config in chat_configs] == [1536, 1536]
    first_llm.bind_tools.assert_not_called()
    second_llm.bind_tools.assert_not_called()
