import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")


def _state():
    return {
        "task_type": "math_reasoning",
        "selected_model": SimpleNamespace(label="test/model [bf16]"),
        "last_intervention": "data_rebuild",
        "current_dataset_path": None,
        "eval_set": MagicMock(),
        "train_examples": [{"prompt": "2+2?", "answer": "4"}],
        "last_eval": None,
        "llm_iterate_decision": None,
        "dataset_version": 0,
        "curriculum_size_target": 1,
        "task_plan": {"task_name": "gsm8k", "benchmark": "gsm8k"},
    }


def _gold():
    return [{"prompt": "2+2?", "answer": "4"}]


def _read_saved_rows(out):
    with open(out["current_dataset_path"], encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_curate_cot_uses_reachable_local_qwen_first_without_cloud(
    tmp_path,
    monkeypatch,
):
    from agent.nodes.curate import curate_node

    monkeypatch.chdir(tmp_path)
    local_calls = []

    def local_generate(prompt, temperature, max_tokens):
        local_calls.append((prompt, temperature, max_tokens))
        return "local Qwen3.6 reasoning"

    fallback = MagicMock()
    with (
        patch(
            "agent.nodes.curate.apply_quality_controls",
            side_effect=lambda rows, task_type: rows,
        ),
        patch("data.synth_client.is_available", return_value=True),
        patch(
            "data.synth_client.get_generate_fn",
            return_value=local_generate,
        ),
        patch(
            "agent.nodes.curate.get_cot_fallbacks",
            return_value=[(fallback, "gpt-4.1")],
        ),
        patch("config.config.SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    ):
        out = curate_node(_state())

    assert _read_saved_rows(out)[0]["cot_reasoning"] == "local Qwen3.6 reasoning"
    assert len(local_calls) == 1
    assert local_calls[0][1:] == (0.3, 512)
    fallback.chat.completions.create.assert_not_called()


def test_curate_cheap_mode_skips_cot_annotation(tmp_path, monkeypatch):
    from agent.nodes.curate import curate_node

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLM_CHEAP", "1")
    with (
        patch(
            "agent.nodes.curate.apply_quality_controls",
            side_effect=lambda rows, task_type: rows,
        ),
        patch("agent.nodes.curate.annotate_cot") as annotate,
        patch("agent.nodes.curate.get_cot_fallbacks") as fallbacks,
    ):
        out = curate_node(_state())

    annotate.assert_not_called()
    fallbacks.assert_not_called()
    assert "cot_reasoning" not in _read_saved_rows(out)[0]


def test_curate_uses_cloud_cot_fallback_when_local_synth_is_unavailable(
    tmp_path,
    monkeypatch,
):
    from agent.nodes.curate import curate_node

    monkeypatch.chdir(tmp_path)
    fallback = MagicMock()
    fallback.chat.completions.create.return_value.choices[
        0
    ].message.content = "cloud fallback reasoning"
    with (
        patch(
            "agent.nodes.curate.apply_quality_controls",
            side_effect=lambda rows, task_type: rows,
        ),
        patch("data.synth_client.is_available", return_value=False),
        patch("data.synth_client.get_generate_fn") as get_generate,
        patch(
            "agent.nodes.curate.get_cot_fallbacks",
            return_value=[(fallback, "gpt-4.1")],
        ),
        patch("config.config.SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    ):
        out = curate_node(_state())

    get_generate.assert_not_called()
    fallback.chat.completions.create.assert_called_once()
    assert _read_saved_rows(out)[0]["cot_reasoning"] == "cloud fallback reasoning"
