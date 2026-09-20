"""A generated row must satisfy the ANCHOR's tool signature, not one it invented for itself.

WHAT WENT WRONG (run 39361648)
    `_synthesize_new_correct` pinned the anchor's `tools` only when the teacher had omitted them
    (`if pinned in anchor and pinned not in row`). The generation prompt hands the teacher the
    anchor's whole JSON schema and asks for "EXACTLY this JSON schema (same keys, same value
    types)", so the teacher essentially always emitted its own `tools` — and the pin never fired.

    It then called a function from that invented list, and `verify_function_call_row` PASSED: the
    call is declared, by the row's own fabricated schema. Self-consistent and off-distribution.

    Measured: gold draws 1,158 distinct function names over 2,536 calls with the top ten at 6.1%;
    the generated corpus managed 787 over 3,230 with the top ten at 45.8% — 356 `convert_currency`,
    347 `get_crypto_price`, 302 `get_weather`. xLAM/BFCL is a long-tail benchmark about calling
    UNFAMILIAR APIs, so that trains a distribution the eval never measures.

    It also produced contradictions no check could catch: two kept rows carried the identical
    request "What are the current prices of Bitcoin and Ethereum?" against the same invented
    `get_crypto_price` with incompatible arguments — `{"coin_id": "bitcoin"}` and `{"symbol": "BTC"}`
    — each internally consistent with the schema it had invented for itself.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("EXA_API_KEY", "x")

from data.curriculum import _new_example_prompt, _synthesize_new_correct
from data.synth_verifiers import verify_function_call_row


def _anchor() -> dict:
    return {
        "text": "Find live giveaways for beta access",
        "answer": '[{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}]',
        "tools": [{
            "name": "live_giveaways_by_type",
            "parameters": {"type": "dict", "properties": {"type": {"type": "string"}},
                           "required": ["type"]},
        }],
    }


def _teacher_inventing_its_own_tools(prompt, temperature=0.7, max_tokens=None) -> str:
    """What the real teacher did: a self-consistent row built on a fabricated generic API."""
    return json.dumps({
        "text": "What is the weather in Paris?",
        "answer": '[{"name": "get_weather", "arguments": {"city": "Paris"}}]',
        "tools": [{
            "name": "get_weather",
            "parameters": {"type": "dict", "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        }],
    })


class TestTheAnchorsToolsWin:
    def test_the_invented_tools_are_overwritten(self):
        rows = _synthesize_new_correct(
            [_anchor()], task_description="t", n=1,
            generate_fn=_teacher_inventing_its_own_tools, verify_fn=None,
        )
        assert rows, "generation produced nothing"
        assert rows[0]["tools"] == _anchor()["tools"]
        assert rows[0]["tools"][0]["name"] == "live_giveaways_by_type"

    def test_a_row_calling_an_invented_function_is_then_rejected(self):
        """With the anchor's tools restored, `get_weather` is undeclared — which is exactly the
        check the eval scorer applies, so such a row would be unwinnable by construction."""
        rows = _synthesize_new_correct(
            [_anchor()], task_description="t", n=1,
            generate_fn=_teacher_inventing_its_own_tools,
            verify_fn=lambda row: verify_function_call_row(row)[0],
        )
        assert rows == []

    def test_the_rejection_names_the_undeclared_function(self):
        row = json.loads(_teacher_inventing_its_own_tools(""))
        row["tools"] = _anchor()["tools"]
        ok, reason = verify_function_call_row(row)
        assert not ok
        assert "undeclared function" in reason and "get_weather" in reason


class TestTheTeacherIsToldTheRuleUpFront:
    """Pinning without saying so would reject every row — the toolbench wipeout shape. The prompt
    has to state the constraint the verifier is about to enforce."""

    def test_the_prompt_names_the_allowed_functions(self):
        prompt = _new_example_prompt(_anchor(), "task", demos=[_anchor()])
        assert "live_giveaways_by_type" in prompt
        assert "FIXED" in prompt

    def test_the_prompt_forbids_inventing_a_different_name(self):
        prompt = _new_example_prompt(_anchor(), "task", demos=[_anchor()])
        assert "Do NOT invent a different function name" in prompt

    def test_it_points_the_variation_at_arguments_instead(self):
        """The teacher still has to produce diverse rows; the rule redirects that diversity to the
        request and the argument values rather than removing it."""
        prompt = _new_example_prompt(_anchor(), "task", demos=[_anchor()])
        assert "ARGUMENT VALUES" in prompt

    def test_a_task_without_tools_gets_no_rule(self):
        """calendar_json-style rows carry tools; gsm8k-style rows do not, and inventing a rule for
        them would put a contradictory instruction in the prompt."""
        prompt = _new_example_prompt({"text": "2+2?", "answer": "4"}, "task")
        assert "FIXED" not in prompt
