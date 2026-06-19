# tests/test_graph.py
from unittest.mock import patch, MagicMock
from android_pool import HardwareConstraints, ModelSpec
from agent.graph import build_graph


def test_graph_builds_without_error():
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_graph()
    node_names = set(graph.nodes.keys())
    expected = {
        "task_analysis", "eval_setup", "train",
        "evaluate", "iterate", "curate", "rollback", "escalate"
    }
    assert expected.issubset(node_names)


def _mock_eval_result(f1: float = 0.97):
    """Build a minimal EvalResult-like object for smoke tests."""
    result = MagicMock()
    result.f1 = f1
    result.per_class = {"spam": f1, "ham": 1.0 - f1 + 0.01}
    result.pos_score = f1
    result.neg_score = f1
    result.boundary_score = f1 - 0.02
    result.failures = []
    return result


def _base_model() -> ModelSpec:
    return ModelSpec(
        model_id="openbmb/MiniCPM4-0.5B",
        int4_size_mb=300,
        tier=1,
        tok_s_snapdragon_660=10.0,
        tok_s_snapdragon_778g=18.0,
        tok_s_snapdragon_8gen3=60.0,
        peak_memory_mb=380,
    )


def _spam_eval_set():
    """Build a minimal EvalSet for classification smoke tests."""
    from data.eval_set import EvalSet
    pos = [{"text": f"WINNER! Claim your prize now {i}", "label": "spam"} for i in range(40)]
    neg = [{"text": f"Hi, how are you doing {i}?", "label": "ham"} for i in range(40)]
    boundary = [{"text": f"Free offer click here {i}", "label": "ham"} for i in range(20)]
    return EvalSet(pos=pos, neg=neg, boundary=boundary, task_type="classification")


def test_graph_smoke_terminates():
    """
    Full-graph smoke test: patches external I/O so the graph runs to completion
    without GPU, network, or API calls.
    """
    from data.eval_set import EvalSet

    eval_set = _spam_eval_set()
    base_model = _base_model()
    mock_eval_result = _mock_eval_result(f1=0.97)

    train_examples = (
        [{"text": f"WIN a prize {i}", "label": "spam"} for i in range(50)]
        + [{"text": f"Hello friend {i}", "label": "ham"} for i in range(50)]
    )
    test_examples = (
        [{"text": f"FREE offer {i}", "label": "spam"} for i in range(30)]
        + [{"text": f"Hey there {i}", "label": "ham"} for i in range(30)]
    )

    # curate_node imports anthropic and config lazily inside the function body.
    # Mock both at their source so the lazy imports find the mocks.
    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = MagicMock()

    mock_config_module = MagicMock()
    mock_config_module.ANTHROPIC_API_KEY = "test-key"

    with (
        patch("data.loaders.sms_spam.download_sms_spam", return_value=(train_examples, test_examples)),
        patch("agent.nodes.eval_setup.build_eval_set", return_value=eval_set),
        patch("agent.nodes.curate.build_initial_curriculum", return_value=[{"text": "x", "label": "spam"}] * 150),
        patch("agent.nodes.curate.synthesize_hard_negatives", return_value=[{"text": "y", "label": "ham"}] * 50),
        patch("agent.nodes.curate.apply_quality_controls", side_effect=lambda d, task_type: d),
        patch.dict("sys.modules", {
            "anthropic": mock_anthropic_module,
            "config": mock_config_module,
        }),
        patch("agent.nodes.train.slm_train", return_value="artifacts/mock_checkpoint"),
        patch("eval.harness.infer_batch", return_value=["spam"] * 40 + ["ham"] * 40 + ["ham"] * 20),
        patch("agent.nodes.evaluate.CurationLog") as mock_log_cls,
        patch("agent.nodes.evaluate.theoretical_hardware_profile", return_value={"int4_size_mb": 300, "tier": 1}),
        patch("builtins.open", MagicMock()),
        patch("os.makedirs"),
    ):
        mock_log_cls.return_value.write_iteration = MagicMock()

        initial_state = {
            "description": "fine-tune on SMS Spam binary classification",
            "target_metric": "F1",
            "hardware_constraints": HardwareConstraints(
                storage_mb=800,
                memory_mb=1200,
                latency_ttft_ms=2000,
                power_watts=5.0,
                target_chip="snapdragon_778g",
            ),
            "task_type": "classification",
            "selected_model": None,
            "stop_threshold": 0.96,
            "train_examples": [],
            "eval_set": None,
            "current_dataset_path": None,
            "dataset_version": 0,
            "best_weights_ref": None,
            "best_score": 0.0,
            "iteration": 0,
            "scores": [],
            "dag": [],
            "consecutive_no_improvement": 0,
            "last_eval": None,
            "last_intervention": "",
            "next_action": "train",
            "messages": [],
            "_pending_weights_refs": None,
            "_pending_configs": None,
        }

        graph = build_graph()
        final_state = graph.invoke(initial_state)

    assert final_state["iteration"] > 0
    assert final_state["best_score"] > 0.0
    assert final_state["selected_model"] is not None
