import os
from unittest.mock import patch


os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")


def _state():
    return {"task_type": "math_reasoning"}


def test_cot_uses_the_local_qwen_synth_teacher(monkeypatch):
    """The sole CoT teacher is the local Qwen3.6 synth endpoint — no cloud teacher exists.
    Each missing chain is authored locally at temperature 0.3 / 512 tokens."""
    from agent.nodes.curate import _annotate_generation_cot

    local_calls = []

    def local_generate(prompt, temperature, max_tokens):
        local_calls.append((prompt, temperature, max_tokens))
        return "local Qwen3.6 reasoning"

    with (
        patch("data.synth_client.is_available", return_value=True),
        patch("data.synth_client.get_generate_fn", return_value=local_generate),
        patch("config.config.SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    ):
        rows = _annotate_generation_cot(
            [{"prompt": "2+2?", "answer": "4"}], _state(), model_id="test/model [bf16]"
        )

    assert rows[0]["cot_reasoning"] == "local Qwen3.6 reasoning"
    assert len(local_calls) == 1
    assert local_calls[0][1:] == (0.3, 512)


def test_cot_is_skipped_in_cheap_mode(monkeypatch):
    from agent.nodes.curate import _annotate_generation_cot

    monkeypatch.setenv("SLM_CHEAP", "1")
    with patch("data.synth_client.get_generate_fn") as get_generate:
        rows = _annotate_generation_cot(
            [{"prompt": "2+2?", "answer": "4"}], _state(), model_id="test/model [bf16]"
        )

    get_generate.assert_not_called()
    assert "cot_reasoning" not in rows[0]


def test_cot_is_skipped_with_no_cloud_fallback_when_local_synth_unavailable(monkeypatch):
    """No cloud fallback: an unreachable local synth endpoint leaves rows CoT-less rather
    than reaching for DeepSeek/OpenAI (which no longer exist in the pipeline)."""
    from agent.nodes.curate import _annotate_generation_cot

    with (
        patch("data.synth_client.is_available", return_value=False),
        patch("data.synth_client.get_generate_fn") as get_generate,
        patch("config.config.SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    ):
        rows = _annotate_generation_cot(
            [{"prompt": "2+2?", "answer": "4"}], _state(), model_id="test/model [bf16]"
        )

    get_generate.assert_not_called()
    assert "cot_reasoning" not in rows[0]
