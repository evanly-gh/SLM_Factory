"""What the orchestrator is actually shown when it picks the starting model.

WHY THIS FILE EXISTS
    On run 38708719 the orchestrator chose a 2,382MB variant while explicitly reasoning that it
    was picking "the smallest model plausibly capable of the task". Both halves of that sentence
    were sincere; the prompt made them compatible.

    Three properties of the prompt were responsible, and each is pinned below.

    ORDER
        The candidate list came out in pool order, so following "pick the cheapest that can
        plausibly reach the goal" required holding eighteen sizes in mind while reading eighteen
        blocks of benchmark prose. Sorting the list smallest-first, and stating each entry's
        cheapness relative to the largest option, is not a hint — it is the difference between the
        instruction being followable and not.

    WHAT THE METRICS MEAN
        The capability numbers are general-knowledge benchmarks (MMLU-Pro, MMLU-Redux, GSM8K)
        measured on the BASE model, ZERO-SHOT. None of them measures the task being trained, and
        every candidate is about to be LoRA fine-tuned on thousands of in-domain rows — which
        routinely lets a smaller model beat a larger one's zero-shot number. Read as a capability
        ranking, they systematically overestimate how large a model the goal requires, so the
        prompt has to say so rather than leaving the obvious reading in place.

    WHAT A MISSING METRIC MEANS
        Unmeasured is not the same as incapable, and the asymmetry pushes the choice upward: the
        three `Qwen/Qwen3-0.6B` variants read "not reported" on every benchmark, which looks
        strictly worse than a 4B with a published MMLU-Pro score when in fact one has been measured
        and the other has not.

NO NETWORK
    The Anthropic client is stubbed and the prompt is captured from the call arguments. These tests
    are about what is SENT, so the reply only has to be well-formed enough for the node to finish.
"""
from __future__ import annotations

import os
import re
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

from config.android_pool import (  # noqa: E402
    ANDROID_POOL,
    HardwareConstraints,
    format_capability_metrics,
)

TASK = "xlam_bfcl"


def _feasible():
    """The whole real pool.

    Deliberately not a hand-built fixture: the properties under test are about the shape of the
    candidate BLOCK (ordering, ratios, which entries have no numbers), and a synthetic pool would
    let a change to the real one — a new variant with no published benchmarks, say — pass silently.
    """
    return list(ANDROID_POOL)


def _state(feasible):
    return {
        "description": "turn a user request into a JSON function call using the declared tools",
        "task": TASK,
        "task_plan": {"labels": []},
        "feasible_models": feasible,
        "stop_threshold": 0.80,
        "hardware_constraints": HardwareConstraints(
            storage_mb=8000, memory_mb=6000, latency_ttft_ms=3000,
        ),
    }


def _capture_prompt(state, *, reply_selector: str) -> str:
    """Run the node against a stubbed Anthropic client and return the prompt it sent."""
    import anthropic as _anth

    from agent.nodes.cold_start.model_selection.orchestrator_choice import (
        orchestrator_choice_node,
    )

    block = _anth.types.TextBlock(
        text='{"selector": "%s", "reason": "cheapest plausible"}' % reply_selector, type="text",
    )
    client = MagicMock()
    client.messages.create.return_value = MagicMock(content=[block])

    env = dict(os.environ)
    env.pop("SLM_FORCE_MODEL", None)
    with patch.dict(os.environ, env, clear=True), patch("anthropic.Anthropic", return_value=client):
        orchestrator_choice_node(state)

    assert client.messages.create.call_count == 1
    return client.messages.create.call_args.kwargs["messages"][0]["content"]


@pytest.fixture(scope="module")
def prompt() -> str:
    feasible = _feasible()
    return _capture_prompt(_state(feasible), reply_selector=feasible[0].selector)


def _candidate_sizes(prompt: str) -> list[int]:
    return [int(value) for value in re.findall(r"on-disk size: (\d+)MB", prompt)]


# --------------------------------------------------------------------------
# Order, and the cost of each option relative to the others
# --------------------------------------------------------------------------


def test_candidates_are_listed_smallest_first(prompt):
    """The instruction is "pick the cheapest that can plausibly reach the goal". A list in pool
    order makes following it a memory exercise across eighteen entries, which is how run 38708719
    picked a 2,382MB variant while reasoning that it had chosen the smallest capable one."""
    sizes = _candidate_sizes(prompt)
    assert len(sizes) == len(_feasible())
    assert sizes == sorted(sizes)


def test_every_candidate_states_its_tier_and_how_much_cheaper_it_is_than_the_largest(prompt):
    """The ratio is what makes the cost of going bigger legible. "2200MB" and "500MB" are two
    numbers to subtract; "4.4x cheaper than the largest option here" is the trade being decided."""
    feasible = _feasible()
    ratios = re.findall(r"([\d.]+)x cheaper than the largest option here", prompt)
    assert len(ratios) == len(feasible)

    # Tier, quant and size share one line per candidate, so a candidate whose tier drifted onto a
    # different line from its price would not match — which is the point: the reader has to be able
    # to see the two together.
    entries = re.findall(
        r"tier: (\d+)\s+quant: .+?on-disk size: (\d+)MB \(([\d.]+)x cheaper", prompt,
    )
    assert len(entries) == len(feasible)
    largest = max(model.size_mb for model in feasible)
    from config.android_pool import TIER_COUNT

    for tier, size, ratio in entries:
        # Bound read from the pool rather than hardcoded: tiers were renumbered 0-3 → 1-5 on
        # 2026-08-24 and this assertion is about the prompt rendering a REAL tier, not about any
        # particular numbering surviving.
        assert 1 <= int(tier) <= TIER_COUNT
        assert float(ratio) == pytest.approx(largest / max(int(size), 1), abs=0.05)

    # The largest option is 1.0x cheaper than itself, so the ratio is an anchor rather than a
    # separate scale the reader has to calibrate.
    assert float(min(ratios, key=float)) == pytest.approx(1.0, abs=0.05)


def test_every_feasible_variant_reaches_the_prompt(prompt):
    """Sorting and annotating the list must not drop candidates. A variant missing from the block
    cannot be chosen, and nothing downstream would report that it was never offered."""
    feasible = _feasible()
    for model in feasible:
        assert model.selector in prompt
    assert f"ALL {len(feasible)} feasible variants" in prompt


# --------------------------------------------------------------------------
# What a missing benchmark means
# --------------------------------------------------------------------------


def test_the_variants_with_no_published_benchmarks_are_named_as_unmeasured(prompt):
    """`Qwen/Qwen3-0.6B` reports no benchmark at all, on any of its three quant variants. Left
    unexplained, an all-"not reported" row reads as a worse model than a 4B with a published
    MMLU-Pro score, when in fact one has been measured and the other has not — an asymmetry that
    pushes every choice upward, i.e. toward more RAM for no evidence."""
    unmeasured = [model.selector for model in _feasible()
                  if not re.search(r"\d+\.\d", format_capability_metrics(model))]
    assert sorted(unmeasured) == [
        "Qwen/Qwen3-0.6B@Q4_K_M", "Qwen/Qwen3-0.6B@Q8_0", "Qwen/Qwen3-0.6B@bf16",
    ], "fixture assumption: the 0.6B variants are the pool's only unmeasured ones"

    sentence = next(line for line in prompt.splitlines()
                    if "NO published numbers" in line)
    assert f"{len(unmeasured)} candidate(s)" in sentence
    for selector in unmeasured:
        assert selector in sentence
    assert "Do not rank them last for lacking a score." in prompt


def test_nothing_is_called_unmeasured_when_every_candidate_has_a_score():
    """The claim has to be true of the list actually sent. A standing paragraph about unmeasured
    candidates, on a list where every candidate is measured, teaches the orchestrator to discount
    the ones that are."""
    measured = [model for model in _feasible()
                if re.search(r"\d+\.\d", format_capability_metrics(model))]
    assert measured
    prompt = _capture_prompt(_state(measured), reply_selector=measured[0].selector)
    assert "NO published numbers" not in prompt


# --------------------------------------------------------------------------
# What the metrics do and do not measure
# --------------------------------------------------------------------------


def test_the_prompt_says_the_capability_metrics_do_not_measure_this_task(prompt):
    """They are MMLU-Pro / MMLU-Redux / GSM8K. Presented without comment next to an accuracy goal
    for function calling, they read as a ranking for the task being trained, and the largest number
    wins — which is the reasoning that chose the 2,382MB variant."""
    assert "GENERAL-KNOWLEDGE benchmarks" in prompt
    assert f"None of them measures {TASK}" in prompt
    assert "do not treat the ordering as a ranking for this benchmark" in prompt


def test_the_prompt_says_fine_tuning_changes_the_picture(prompt):
    """The numbers are BASE-model, ZERO-SHOT, and every candidate is about to be LoRA fine-tuned on
    thousands of in-domain rows. Extrapolating from them therefore overestimates how large a model
    the goal needs — and the prompt says which DIRECTION the error runs in, because "these numbers
    are uncertain" alone does not tell the reader which way to lean."""
    assert "ZERO-SHOT" in prompt
    assert "will be LoRA fine-tuned" in prompt
    assert "systematically overestimates how large a model" in prompt


def test_the_prompt_says_falling_short_of_the_goal_today_is_not_disqualifying(prompt):
    """The run rebuilds data and retunes hyperparameters for many iterations. Without this, the
    zero-shot number is read as the deliverable and the choice becomes "which model already passes"
    — a question no on-device candidate answers, so the answer is always the biggest one."""
    assert "a model that starts short of the goal is not disqualified" in prompt


def test_the_choice_the_orchestrator_returns_is_the_one_selected():
    """The prompt work above is worthless if the reply is not honoured — and the fallback picks the
    smallest model, which would mask a broken selector round-trip in exactly the tests that assert
    small models are preferred."""
    from agent.nodes.cold_start.model_selection.orchestrator_choice import (
        orchestrator_choice_node,
    )
    import anthropic as _anth

    feasible = _feasible()
    wanted = max(feasible, key=lambda model: model.size_mb)
    state = _state(feasible)
    block = _anth.types.TextBlock(
        text='{"selector": "%s", "reason": "needs the capacity"}' % wanted.selector, type="text",
    )
    client = MagicMock()
    client.messages.create.return_value = MagicMock(content=[block])

    env = dict(os.environ)
    env.pop("SLM_FORCE_MODEL", None)
    with patch.dict(os.environ, env, clear=True), patch("anthropic.Anthropic", return_value=client):
        orchestrator_choice_node(state)

    assert state["selected_model"] is wanted
