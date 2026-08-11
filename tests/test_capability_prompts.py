"""Capability-data and prompt-contract regression tests.

These tests intentionally exercise prompt construction with mocked clients. They
must never make a network call.
"""
import importlib
import sys
from unittest.mock import MagicMock, patch

import anthropic

import config.android_pool as android_pool
from config.android_pool import (
    ANDROID_POOL,
    CapabilityMeasurement,
    HardwareConstraints,
    ModelSpec,
    filter_pool_by_task,
    format_capability_metrics,
)
from eval.harness import EvalResult


_QWEN35_SOURCE = "https://huggingface.co/Qwen/Qwen3.5-0.8B"


def _model(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    *,
    size_mb: int = 500,
    tier: int = 0,
    gsm8k: float | None = None,
    knowledge_metric: str = "MMLU-Pro",
    knowledge_score: float = 0.297,
    mmlu_redux: float = 0.485,
    quant: str | None = "Q4_K_M",
) -> ModelSpec:
    measurements = []
    if gsm8k is not None:
        measurements.append(
            CapabilityMeasurement(
                metric="GSM8K",
                value=gsm8k,
                artifact=model_id,
                mode="non-thinking",
                protocol="test-protocol",
                source=_QWEN35_SOURCE,
            )
        )
    if knowledge_metric and knowledge_score is not None:
        measurements.append(
            CapabilityMeasurement(
                metric=knowledge_metric,
                value=knowledge_score,
                artifact=model_id,
                mode="non-thinking",
                protocol="test-protocol",
                source=_QWEN35_SOURCE,
            )
        )
    if mmlu_redux is not None:
        measurements.append(
            CapabilityMeasurement(
                metric="MMLU-Redux",
                value=mmlu_redux,
                artifact=model_id,
                mode="non-thinking",
                protocol="test-protocol",
                source=_QWEN35_SOURCE,
            )
        )
    return ModelSpec(
        model_id=model_id,
        size_mb=size_mb,
        tier=tier,
        capability_measurements=tuple(measurements),
        quant=quant,
        multimodal=True,
    )


def _hw() -> HardwareConstraints:
    return HardwareConstraints(
        storage_mb=10_000,
        memory_mb=10_000,
        latency_ttft_ms=5_000,
    )


def _text_response(model_id: str):
    return MagicMock(
        content=[
            anthropic.types.TextBlock(
                text=f'{{"model_id": "{model_id}", "reason": "sourced fit"}}',
                type="text",
            )
        ]
    )


def _selector_response(selector: str):
    return MagicMock(
        content=[
            anthropic.types.TextBlock(
                text=f'{{"selector": "{selector}", "reason": "exact deployment fit"}}',
                type="text",
            )
        ]
    )


def _orchestrator_choice_module():
    # The model-selection package imports interpolation eagerly. This focused unit
    # test does not exercise interpolation, so avoid requiring NumPy in the slim
    # development-test environment.
    with patch.dict(sys.modules, {"numpy": MagicMock()}):
        return importlib.import_module(
            "agent.nodes.cold_start.model_selection.orchestrator_choice"
        )


def _assert_metric_contract(prompt: str, system: str = "") -> None:
    combined = f"{system}\n{prompt}"
    assert "## Qwen/Qwen3.5-0.8B" in prompt
    assert "MMLU-Pro: 29.7" in prompt
    assert "MMLU-Redux: 48.5" in prompt
    assert "GSM8K: not reported" in prompt
    assert "unknown, not zero" in combined
    assert "MMLU, MMLU-Pro, and MMLU-Redux" in combined
    assert "gsm8k: 0.00" not in prompt.lower()
    assert "mmlu: 0.30" not in prompt.lower()
    assert "Qwen3.6" not in combined


def test_official_pool_capabilities_are_named_and_missing_gsm8k_stays_unknown():
    by_id = {}
    for model in ANDROID_POOL:
        by_id.setdefault(model.model_id, model)

    expected = {
        "Qwen/Qwen3.5-0.8B": ("MMLU-Pro", 0.297, 0.485),
        "Qwen/Qwen3.5-2B": ("MMLU-Pro", 0.553, 0.692),
        "Qwen/Qwen3.5-4B": ("MMLU-Pro", 0.791, 0.888),
        "Qwen/Qwen3-4B-Instruct-2507": ("MMLU-Pro", 0.696, 0.842),
        "Qwen/Qwen3-1.7B": ("MMLU-Pro", 0.402, 0.644),
    }
    for model_id, (metric, score, redux) in expected.items():
        model = by_id[model_id]
        assert model.knowledge_metric == metric
        assert model.knowledge_score == score
        assert model.mmlu_redux == redux
        assert model.benchmark_source.startswith("https://huggingface.co/Qwen/")

    for model_id in (
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3-4B-Instruct-2507",
    ):
        assert by_id[model_id].gsm8k is None


def test_capability_measurements_record_artifact_mode_protocol_and_source():
    model = next(
        model
        for model in ANDROID_POOL
        if model.model_id == "Qwen/Qwen3.5-2B"
    )

    measurement = model.measurement("MMLU-Pro")

    assert measurement is not None
    assert measurement.artifact == "Qwen/Qwen3.5-2B"
    assert measurement.mode == "non-thinking"
    assert measurement.protocol == "qwen3.5-small-nonthinking-card-table"
    assert measurement.source == "https://huggingface.co/Qwen/Qwen3.5-2B"


def test_qwen35_4b_card_values_do_not_claim_non_thinking_mode():
    model = next(
        model
        for model in ANDROID_POOL
        if model.model_id == "Qwen/Qwen3.5-4B"
    )

    measurement = model.measurement("MMLU-Pro")

    assert measurement is not None
    assert measurement.value == 0.791
    assert measurement.mode is None
    assert measurement.protocol == "qwen3.5-4b-card-table-mode-unspecified"


def test_qwen3_06_base_table_mmlu_is_not_attached_to_post_trained_artifact():
    model = next(
        model
        for model in ANDROID_POOL
        if model.model_id == "Qwen/Qwen3-0.6B"
    )

    assert model.measurement("MMLU") is None
    assert model.knowledge_score is None
    assert "knowledge benchmark: not reported" in format_capability_metrics(model)


def test_same_metric_with_different_protocols_keeps_resource_order():
    measurement_cls = getattr(android_pool, "CapabilityMeasurement", None)
    assert measurement_cls is not None, "CapabilityMeasurement is not implemented"
    shared = dict(
        metric="MMLU-Pro",
        artifact="test/model",
        mode="non-thinking",
        source="https://example.test/card",
    )
    cheaper = ModelSpec(
        model_id="test/cheap",
        size_mb=400,
        tier=1,
        capability_measurements=(
            measurement_cls(value=0.20, protocol="protocol-a", **shared),
        ),
    )
    stronger_but_larger = ModelSpec(
        model_id="test/large",
        size_mb=800,
        tier=1,
        capability_measurements=(
            measurement_cls(value=0.99, protocol="protocol-b", **shared),
        ),
    )

    with patch(
        "config.android_pool.filter_pool",
        return_value=[stronger_but_larger, cheaper],
    ):
        ranked = filter_pool_by_task(_hw(), "classification")

    assert ranked == [cheaper, stronger_but_larger]


def test_capability_sections_selects_only_requested_qwen35_sections():
    from config.model_capabilities import capability_sections

    selected = capability_sections(
        ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"]
    )

    assert "## Qwen/Qwen3.5-0.8B" in selected
    assert "## Qwen/Qwen3.5-4B" in selected
    assert "## Qwen/Qwen3.5-2B" not in selected
    assert "## Qwen/Qwen3-1.7B" not in selected
    assert "METRIC-COMPARABILITY CAVEAT" in selected


def test_task_sorting_falls_back_to_resource_order_for_incomparable_metrics():
    cheaper_pro = _model(size_mb=400, knowledge_score=0.20)
    larger_mmlu = _model(
        "Qwen/Qwen3-0.6B",
        size_mb=700,
        knowledge_metric="MMLU",
        knowledge_score=0.99,
        mmlu_redux=None,
        gsm8k=0.596,
    )

    with patch(
        "config.android_pool.filter_pool",
        return_value=[larger_mmlu, cheaper_pro],
    ):
        ranked = filter_pool_by_task(_hw(), "classification")

    assert ranked == [cheaper_pro, larger_mmlu]


def test_task_sorting_does_not_treat_unknown_gsm8k_as_zero():
    unknown_cheaper = _model(size_mb=400, gsm8k=None)
    known_larger = _model(
        "Qwen/Qwen3-1.7B",
        size_mb=700,
        gsm8k=0.754,
        knowledge_score=0.402,
        mmlu_redux=0.644,
    )

    with patch(
        "config.android_pool.filter_pool",
        return_value=[known_larger, unknown_cheaper],
    ):
        ranked = filter_pool_by_task(_hw(), "math_reasoning")

    assert ranked == [unknown_cheaper, known_larger]


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_orchestrator_choice_injects_sourced_qwen35_capability_contract():
    orchestrator_choice = _orchestrator_choice_module()

    candidate = _model()
    captured = {}

    def fake_create(*_args, **kwargs):
        captured.update(kwargs)
        return _text_response(candidate.model_id)

    state = {
        "description": "classify messages",
        "feasible_models": [candidate],
        "stop_threshold": 0.8,
        "task_type": "classification",
        "task_plan": {"task_name": "messages", "labels": ["a", "b"]},
        "hardware_constraints": _hw(),
    }
    with (
        patch("anthropic.Anthropic"),
        patch.object(
            orchestrator_choice,
            "tracked_anthropic_messages_create",
            side_effect=fake_create,
        ),
    ):
        out = orchestrator_choice.orchestrator_choice_node(state)

    assert out["selected_model"] is candidate
    _assert_metric_contract(
        captured["messages"][0]["content"],
        captured["system"],
    )


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_orchestrator_choice_selects_exact_quant_sibling():
    orchestrator_choice = _orchestrator_choice_module()
    siblings = [
        _model(quant=None, size_mb=1_818),
        _model(quant="Q8_0", size_mb=909),
        _model(quant="Q4_K_M", size_mb=500),
    ]
    wanted = siblings[1]
    captured = {}

    def fake_create(*_args, **kwargs):
        captured.update(kwargs)
        return _selector_response(wanted.selector)

    state = {
        "description": "classify messages",
        "feasible_models": siblings,
        "stop_threshold": 0.8,
        "task_type": "classification",
        "task_plan": {"task_name": "messages", "labels": ["a", "b"]},
        "hardware_constraints": _hw(),
    }
    with (
        patch("anthropic.Anthropic"),
        patch.object(
            orchestrator_choice,
            "tracked_anthropic_messages_create",
            side_effect=fake_create,
        ),
    ):
        out = orchestrator_choice.orchestrator_choice_node(state)

    assert out["selected_model"] is wanted
    prompt = captured["messages"][0]["content"]
    assert f"selector: {wanted.selector}" in prompt
    assert '"selector": "<exact selector from the list>"' in prompt


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_escalate_choice_injects_sourced_qwen35_capability_contract():
    from agent.nodes import escalate

    candidate = _model(tier=1)
    captured = {}

    def fake_create(*_args, **kwargs):
        captured.update(kwargs)
        return _text_response(candidate.model_id)

    with (
        patch("anthropic.Anthropic"),
        patch.object(
            escalate,
            "tracked_anthropic_messages_create",
            side_effect=fake_create,
        ),
    ):
        chosen = escalate._llm_choose_model(
            [candidate],
            "classification",
            {"task_name": "messages", "labels": ["a", "b"]},
            0.7,
        )

    assert chosen is candidate
    _assert_metric_contract(
        captured["messages"][0]["content"],
        captured["system"],
    )


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_escalate_choice_selects_exact_quant_sibling():
    from agent.nodes import escalate

    siblings = [
        _model(tier=1, quant=None, size_mb=1_818),
        _model(tier=1, quant="Q8_0", size_mb=909),
        _model(tier=1, quant="Q4_K_M", size_mb=500),
    ]
    wanted = siblings[1]

    with (
        patch("anthropic.Anthropic"),
        patch.object(
            escalate,
            "tracked_anthropic_messages_create",
            return_value=_selector_response(wanted.selector),
        ),
    ):
        chosen = escalate._llm_choose_model(
            siblings,
            "classification",
            {"task_name": "messages", "labels": ["a", "b"]},
            0.7,
        )

    assert chosen is wanted


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_escalate_failure_fallback_chooses_lowest_ram_quant_sibling():
    from agent.nodes import escalate

    siblings = [
        _model(tier=2, quant=None, size_mb=1_818),
        _model(tier=2, quant="Q8_0", size_mb=909),
        _model(tier=2, quant="Q4_K_M", size_mb=500),
    ]

    with patch("anthropic.Anthropic", side_effect=RuntimeError("offline")):
        chosen = escalate._llm_choose_model(
            siblings,
            "classification",
            {"task_name": "messages", "labels": ["a", "b"]},
            0.7,
            direction="up",
        )

    assert chosen.quant == "Q4_K_M"
    # Tie-break moved from modelled peak RAM to real on-disk size.
    assert chosen.size_mb == min(model.size_mb for model in siblings)


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_downward_choice_injects_sourced_qwen35_capability_contract():
    from agent.nodes import downward_probe, escalate

    candidate = _model(tier=1)
    current = _model(
        "Qwen/Qwen3.5-4B",
        size_mb=2_200,
        tier=2,
        knowledge_score=0.791,
        mmlu_redux=0.888,
    )
    captured = {}

    def fake_create(*_args, **kwargs):
        captured.update(kwargs)
        return _text_response(candidate.model_id)

    state = {
        "selected_model": current,
        "task_type": "classification",
        "task_plan": {"task_name": "messages", "labels": ["a", "b"]},
        "stop_threshold": 0.90,
        "best_score": 0.95,
        "best_weights_ref": "/current/checkpoint",
        "current_dataset_path": "/data.jsonl",
        "eval_set": MagicMock(),
        "hardware_constraints": _hw(),
    }
    failed_probe = EvalResult(
        f1=0.70,
        per_class={},
        failures=[],
    )
    with (
        patch("anthropic.Anthropic"),
        patch.object(
            escalate,
            "tracked_anthropic_messages_create",
            side_effect=fake_create,
        ),
        patch.object(downward_probe, "filter_pool", return_value=[candidate]),
        patch.object(
            downward_probe,
            "_train_and_eval",
            return_value=("/candidate/checkpoint", failed_probe),
        ),
    ):
        out = downward_probe.downward_probe_node(state)

    assert out["selected_model"] is current
    _assert_metric_contract(
        captured["messages"][0]["content"],
        captured["system"],
    )


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_downward_probe_reuses_shared_chooser_and_honors_exact_quant_selector():
    from agent.nodes import downward_probe, escalate

    siblings = [
        _model(tier=1, quant=None, size_mb=1_818),
        _model(tier=1, quant="Q8_0", size_mb=909),
        _model(tier=1, quant="Q4_K_M", size_mb=500),
    ]
    wanted = siblings[1]
    current = _model(
        "Qwen/Qwen3.5-4B",
        size_mb=2_200,
        tier=2,
        knowledge_score=0.791,
        mmlu_redux=0.888,
    )
    failed_probe = EvalResult(
        f1=0.70,
        per_class={},
        failures=[],
    )
    trained = []

    def fake_train_and_eval(model, *_args):
        trained.append(model)
        return "/candidate/checkpoint", failed_probe

    state = {
        "selected_model": current,
        "task_type": "classification",
        "task_plan": {"task_name": "messages", "labels": ["a", "b"]},
        "stop_threshold": 0.90,
        "best_score": 0.95,
        "best_weights_ref": "/current/checkpoint",
        "current_dataset_path": "/data.jsonl",
        "eval_set": MagicMock(),
        "hardware_constraints": _hw(),
    }
    with (
        patch("anthropic.Anthropic"),
        patch.object(
            escalate,
            "tracked_anthropic_messages_create",
            return_value=_selector_response(wanted.selector),
        ),
        patch.object(downward_probe, "filter_pool", return_value=siblings),
        patch.object(
            downward_probe,
            "_train_and_eval",
            side_effect=fake_train_and_eval,
        ),
    ):
        out = downward_probe.downward_probe_node(state)

    assert trained == [wanted]
    assert out["selected_model"] is current


def test_planner_pool_summary_uses_explicit_names_and_unknown_marker():
    from agent.task_planner import _pool_summary

    summary = _pool_summary([_model()])

    assert "MMLU-Pro: 29.7" in summary
    assert "MMLU-Redux: 48.5" in summary
    assert "GSM8K: not reported" in summary
    assert "mmlu=" not in summary.lower()
    assert "gsm8k=0.00" not in summary.lower()


def test_planner_pool_summary_accepts_model_with_no_measurements():
    from agent.task_planner import _pool_summary

    model = ModelSpec(
        model_id="test/unknown",
        size_mb=100,
        tier=0,
    )

    summary = _pool_summary([model])

    assert "GSM8K: not reported" in summary
    assert "knowledge benchmark: not reported" in summary
    assert "MMLU-Redux: not reported" in summary


def test_code_planner_and_model_choice_prompts_target_apps_introductory():
    from agent import task_planner
    from agent.nodes import escalate

    orchestrator_choice = _orchestrator_choice_module()
    planner_prompt = task_planner._PLANNER_PROMPT
    model_guidance = "\n".join(
        (
            escalate._BENCHMARK_HINT["code_generation"],
            orchestrator_choice._BENCHMARK_HINT["code_generation"],
        )
    )

    assert "APPS introductory" in planner_prompt
    assert "APPS introductory" in model_guidance
    assert "HumanEval/MBPP" not in planner_prompt
    assert "HumanEval/MBPP" not in model_guidance
