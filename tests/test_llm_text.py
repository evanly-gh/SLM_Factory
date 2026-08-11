"""B253: the orchestrator's answer is not always content[0].

When the model engages extended thinking the response is [ThinkingBlock, TextBlock]. Reading
index 0 either raises (no .text) or, at the sites that guarded with isinstance, yields "" —
which the model-choice parsers turned into an empty selector and a silent fallback to the
smallest candidate, at full API cost and with no error in the log.
"""
import os
from unittest.mock import MagicMock, patch

import anthropic

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

from agent.llm_text import describe_empty_text, response_text, truncated_by_output_budget


def _thinking_block(thinking: str = "weighing the candidates"):
    """A thinking block exposes .thinking, never .text — that is the whole bug."""
    block = MagicMock(spec=["type", "thinking"])
    block.type = "thinking"
    block.thinking = thinking
    return block


def _text_block(text: str):
    return anthropic.types.TextBlock(text=text, type="text")


def _response(blocks, stop_reason="end_turn"):
    return MagicMock(content=blocks, stop_reason=stop_reason)


def test_text_is_found_after_a_leading_thinking_block():
    resp = _response([_thinking_block(), _text_block('{"selector": "Qwen/Qwen3-1.7B@Q4_K_M"}')])
    assert response_text(resp) == '{"selector": "Qwen/Qwen3-1.7B@Q4_K_M"}'


def test_plain_text_response_is_unchanged():
    assert response_text(_response([_text_block("  hello  ")])) == "hello"


def test_multiple_text_blocks_are_concatenated_in_order():
    resp = _response([_thinking_block(), _text_block('{"a": 1,'), _text_block(' "b": 2}')])
    assert response_text(resp) == '{"a": 1, "b": 2}'


def test_thinking_only_response_yields_empty_text_and_a_diagnosis():
    """The exact shape observed in run 38303490: the thinking block consumed the whole
    output budget, so no text block was ever emitted."""
    resp = _response([_thinking_block()], stop_reason="max_tokens")

    assert response_text(resp) == ""
    assert truncated_by_output_budget(resp)
    diagnosis = describe_empty_text(resp)
    assert "thinking" in diagnosis
    assert "max_tokens" in diagnosis
    assert "raise max_tokens" in diagnosis


def test_empty_and_malformed_responses_do_not_raise():
    assert response_text(_response([])) == ""
    assert response_text(MagicMock(content=None)) == ""
    assert response_text(MagicMock(spec=[])) == ""


def test_escalate_uses_the_orchestrator_choice_behind_a_thinking_block():
    """End-to-end: escalate must honour the orchestrator instead of falling back."""
    from agent.nodes import escalate
    from config.android_pool import ANDROID_POOL

    candidates = [m for m in ANDROID_POOL if m.tier == 1]
    assert len(candidates) > 1
    wanted = candidates[-1]
    resp = _response([
        _thinking_block(),
        _text_block(f'{{"selector": "{wanted.selector}", "reason": "best task fit"}}'),
    ])

    with (
        patch("anthropic.Anthropic"),
        patch("agent.cost.tracked_anthropic_messages_create", return_value=resp),
        patch("agent.nodes.escalate.tracked_anthropic_messages_create", return_value=resp),
    ):
        chosen = escalate._llm_choose_model(
            candidates,
            "generation",
            {"task_name": "dialogue summarization", "labels": []},
            0.5706,
            log=lambda *_: None,
            direction="up",
        )

    assert chosen.selector == wanted.selector


def test_model_choice_calls_budget_for_thinking_tokens():
    """A 256-token cap is below what a thinking pass alone consumes (255 observed), so the
    call can only ever return an empty text block."""
    from agent.llm_text import MIN_THINKING_SAFE_MAX_TOKENS

    assert MIN_THINKING_SAFE_MAX_TOKENS >= 1024

    for path in (
        "agent/nodes/escalate.py",
        "agent/nodes/cold_start/model_selection/orchestrator_choice.py",
    ):
        source = open(path, encoding="utf-8").read()
        assert "max_tokens=256" not in source, path
        assert "resp.content[0]" not in source, path


def test_no_call_site_reads_the_first_content_block_directly():
    """Guard the whole family: reading content[0] is never correct."""
    import pathlib

    offenders = []
    for path in pathlib.Path(".").rglob("*.py"):
        parts = set(path.parts)
        if parts & {"tests", ".venv", ".venv_gpu", ".venv_vllm", "logs"}:
            continue
        # The helper itself quotes the broken pattern to explain why it exists.
        if path.name == "llm_text.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "content[0].text" in text or "resp.content[0]" in text:
            offenders.append(str(path))
    assert offenders == [], f"must use agent.llm_text.response_text: {offenders}"
