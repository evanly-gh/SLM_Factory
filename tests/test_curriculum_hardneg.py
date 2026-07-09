import os
from unittest.mock import MagicMock

# data.curriculum -> config.config requires these env vars at import time.
os.environ.setdefault("EXA_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


def test_math_hard_negatives_not_trained_on_wrong_answers():
    """synthesize_hard_negatives for math_reasoning must not produce wrong answers as targets."""
    from data.curriculum import synthesize_hard_negatives

    examples = [
        {"prompt": "What is 2+2?", "answer": "4"},
        {"prompt": "What is 3+3?", "answer": "6"},
    ]

    results = synthesize_hard_negatives(
        examples, task_type="math_reasoning", n=2, anthropic_client=MagicMock()
    )

    # None of the results should have a "response" that is a synthesized wrong answer
    # (they should either be absent, equal to the gold answer, or the examples should
    #  not be augmented with negative wrong-answer pairs)
    for r in results:
        if "response" in r:
            # If there is a response, it must be the GOLD answer, not a synthetic wrong one
            gold = next((e["answer"] for e in examples if e["prompt"] == r.get("prompt", "")), None)
            if gold:
                assert r["response"] == gold, (
                    f"Math hard negative stored wrong answer '{r['response']}' as training target; "
                    f"gold was '{gold}'"
                )
    # Gold examples returned unchanged
    assert results == examples


def test_code_hard_negatives_not_stored_as_wrong_code():
    """code_generation hard negatives must not store broken code as the training target."""
    from data.curriculum import synthesize_hard_negatives

    examples = [{"prompt": "def add(a,b):", "answer": "    return a+b"}]

    results = synthesize_hard_negatives(
        examples, task_type="code_generation", n=1, anthropic_client=MagicMock()
    )

    for r in results:
        if "response" in r:
            gold = next((e["answer"] for e in examples if e["prompt"] == r.get("prompt", "")), None)
            if gold:
                assert r["response"] == gold, (
                    "Code hard negative stored broken code as training target"
                )
    assert results == examples
