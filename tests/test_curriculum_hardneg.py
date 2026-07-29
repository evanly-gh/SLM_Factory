import os
import inspect
from unittest.mock import MagicMock

import pytest

# data.curriculum -> config.config requires these env vars at import time.
os.environ.setdefault("EXA_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


def test_math_cot_fallback_order_is_deepseek_then_openai(monkeypatch):
    import config.config as config
    from data import curriculum

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "deep-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "open-key")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: kwargs)

    fallbacks = curriculum.get_cot_fallbacks("math_reasoning", "gsm8k")

    assert [model for _, model in fallbacks] == [
        config.TEACHER_MODEL_DEEPSEEK,
        config.TEACHER_MODEL_GPT,
    ]


def test_general_cot_fallback_order_is_openai_then_deepseek(monkeypatch):
    import config.config as config
    from data import curriculum

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "deep-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "open-key")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: kwargs)

    fallbacks = curriculum.get_cot_fallbacks("generation", "samsum")

    assert [model for _, model in fallbacks] == [
        config.TEACHER_MODEL_GPT,
        config.TEACHER_MODEL_DEEPSEEK,
    ]


def test_scienceqa_cot_fallback_order_is_deepseek_then_openai(monkeypatch):
    import config.config as config
    from data import curriculum

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "deep-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "open-key")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: kwargs)

    fallbacks = curriculum.get_cot_fallbacks("generation", "ScienceQA")

    assert [model for _, model in fallbacks] == [
        config.TEACHER_MODEL_DEEPSEEK,
        config.TEACHER_MODEL_GPT,
    ]


def test_arc_aliases_use_deepseek_before_openai(monkeypatch):
    import config.config as config
    from data import curriculum

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "deep-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "open-key")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: kwargs)

    for alias in ("ARC", "AI2 ARC", "ARC-Challenge", "ARC_C"):
        fallbacks = curriculum.get_cot_fallbacks("generation", alias)
        assert [model for _, model in fallbacks] == [
            config.TEACHER_MODEL_DEEPSEEK,
            config.TEACHER_MODEL_GPT,
        ], alias


def test_cot_benchmark_uses_task_name_when_benchmark_is_null():
    from agent.nodes.curate import _cot_benchmark

    assert _cot_benchmark({"benchmark": None, "task_name": "ScienceQA"}) == "ScienceQA"


def test_cot_fallback_builder_has_no_anthropic_backend():
    from data import curriculum

    source = inspect.getsource(curriculum.get_cot_fallbacks)
    assert "anthropic" not in source.lower()
    assert "ORCHESTRATOR_MODEL" not in source


def test_curate_cot_path_has_no_orchestrator_teacher():
    from agent.nodes import curate

    source = inspect.getsource(curate._annotate_generation_cot)
    assert "get_teacher_client" not in source
    assert "teacher_client" not in source
    assert "get_cot_fallbacks" in source


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


def test_apps_quality_controls_use_text_for_length_and_dedup():
    from data.curriculum import apply_quality_controls

    duplicate_a = {
        "text": "Return the sum of two integers",
        "answer": "print(sum(map(int, input().split())))",
        "label": "code_generation",
    }
    duplicate_b = {
        **duplicate_a,
        "text": "Return  the sum of two integers",
    }
    unique = {
        "text": "Reverse a string",
        "answer": "print(input()[::-1])",
        "label": "code_generation",
    }

    result = apply_quality_controls(
        [duplicate_a, duplicate_b, unique],
        task_type="code_generation",
    )

    assert result == [duplicate_a, unique]


def test_generation_hard_negatives_never_store_unverified_wrong_answers():
    from data.curriculum import synthesize_hard_negatives

    calls = []

    def gen(prompt, temperature, max_tokens):
        calls.append((prompt, temperature, max_tokens))
        return "plausible wrong answer"

    results = synthesize_hard_negatives(
        [{"prompt": "Question?", "response": "Correct"}],
        n=1,
        task_type="generation",
        generate_fn=gen,
        anthropic_client=None,
    )

    assert calls == []
    assert results == [{"prompt": "Question?", "response": "Correct"}]


def test_classification_hard_negatives_parallel_order_and_skip(monkeypatch):
    """Concurrent synthesis must preserve (gold, synthetic) pair order and skip per-call
    failures non-fatally (B-parallel: synthesis fired concurrently at the vLLM endpoint)."""
    from data.curriculum import synthesize_hard_negatives

    examples = [
        {"text": f"t{i}", "label": "joy" if i % 2 == 0 else "anger"} for i in range(6)
    ]

    def gen(prompt, temperature, max_tokens):
        # Simulate a single per-call failure on the example whose reference text is "t3".
        if "t3\n" in prompt or prompt.rstrip().endswith("t3"):
            raise RuntimeError("simulated per-call failure")
        return "SYNTH"

    monkeypatch.setenv("SLM_SYNTH_CONCURRENCY", "4")
    results = synthesize_hard_negatives(
        examples, n=6, task_type="classification", generate_fn=gen, source_label="synth:test"
    )

    # Gold examples all present, in original order.
    golds = [r for r in results if r.get("_source") != "synth:test"]
    assert golds == examples
    # 6 candidates, exactly one generation failed → 5 synthetics kept.
    synths = [r for r in results if r.get("_source") == "synth:test"]
    assert len(synths) == 5
    # Each surviving synthetic sits immediately after a gold example (contrastive pair).
    for i, r in enumerate(results):
        if r.get("_source") == "synth:test":
            assert i > 0 and results[i - 1].get("_source") != "synth:test"


def test_classification_synthesis_sequential_when_concurrency_1(monkeypatch):
    """SLM_SYNTH_CONCURRENCY=1 restores fully-sequential behavior (no threads)."""
    from data.curriculum import synthesize_hard_negatives

    examples = [{"text": f"t{i}", "label": "joy"} for i in range(3)]
    calls = []

    def gen(prompt, temperature, max_tokens):
        calls.append(prompt)
        return "SYNTH"

    monkeypatch.setenv("SLM_SYNTH_CONCURRENCY", "1")
    results = synthesize_hard_negatives(
        examples, n=3, task_type="classification", generate_fn=gen, source_label="synth:test"
    )
    assert len(calls) == 3
    assert len([r for r in results if r.get("_source") == "synth:test"]) == 3


def test_annotate_cot_uses_local_generate_fn(monkeypatch):
    """When a synth generate_fn is passed, the LOCAL Qwen3.6 model authors CoT (not a cloud
    teacher), preserves existing gold CoT, and requests 512 tokens per chain."""
    from data.curriculum import annotate_cot

    exs = [
        {"prompt": "What is 2+2?", "response": "4"},
        {"prompt": "Capital of France?", "response": "Paris"},
        {"prompt": "x", "response": "y", "cot_reasoning": "already-here"},
    ]
    calls = []

    def gen(prompt, temperature, max_tokens):
        calls.append((prompt, temperature, max_tokens))
        return "step-by-step reasoning"

    monkeypatch.setenv("SLM_SYNTH_CONCURRENCY", "4")
    out = annotate_cot(exs, generate_fn=gen, task_type="generation")

    assert out[0]["cot_reasoning"] == "step-by-step reasoning"
    assert out[1]["cot_reasoning"] == "step-by-step reasoning"
    assert out[2]["cot_reasoning"] == "already-here"   # gold CoT preserved, not regenerated
    assert len(calls) == 2                              # only the 2 that needed a CoT
    assert all(mt == 512 for (_, _, mt) in calls)


@pytest.mark.parametrize(
    ("task_type", "row", "gold", "task_label"),
    [
        (
            "code_generation",
            {
                "text": "Write an identity function.",
                "answer": "def identity(value):\n    return value",
                "label": "code_generation",
                "test_list": ["assert identity(3) == 3"],
                "test_imports": [],
            },
            "def identity(value):\n    return value",
            "code_generation",
        ),
        (
            "generation",
            {
                "text": "A: Meeting at 3.\nB: Confirmed.",
                "answer": "A and B confirm a meeting at 3.",
                "label": "generation",
            },
            "A and B confirm a meeting at 3.",
            "generation",
        ),
    ],
)
def test_bundle_schema_cot_prompt_uses_gold_answer_not_task_label(
    task_type,
    row,
    gold,
    task_label,
):
    from data.curriculum import annotate_cot

    prompts = []

    def generate(prompt, _temperature, _max_tokens):
        prompts.append(prompt)
        return "verified reasoning"

    out = annotate_cot([row], generate_fn=generate, task_type=task_type)

    assert out[0]["cot_reasoning"] == "verified reasoning"
    assert len(prompts) == 1
    assert gold in prompts[0]
    assert task_label not in prompts[0]


def test_qwen_cot_success_prevents_cloud_fallback():
    from data.curriculum import annotate_cot

    fallback = MagicMock()
    out = annotate_cot(
        [{"prompt": "p", "response": "r"}],
        generate_fn=lambda *_: "local reasoning",
        fallback_teachers=[(fallback, "gpt-4.1")],
    )

    assert out[0]["cot_reasoning"] == "local reasoning"
    fallback.chat.completions.create.assert_not_called()


def test_empty_qwen_cot_output_uses_first_fallback():
    from data.curriculum import annotate_cot

    fallback = MagicMock()
    fallback.chat.completions.create.return_value.choices[0].message.content = "cloud reasoning"
    out = annotate_cot(
        [{"prompt": "p", "response": "r"}],
        generate_fn=lambda *_: "",
        fallback_teachers=[(fallback, "deepseek-reasoner")],
    )

    assert out[0]["cot_reasoning"] == "cloud reasoning"


def test_failed_first_cot_fallback_advances_to_second():
    from data.curriculum import annotate_cot

    first, second = MagicMock(), MagicMock()
    first.chat.completions.create.side_effect = RuntimeError("down")
    second.chat.completions.create.return_value.choices[0].message.content = "second reasoning"

    def qwen_down(*_):
        raise RuntimeError("local down")

    out = annotate_cot(
        [{"prompt": "p", "response": "r"}],
        generate_fn=qwen_down,
        fallback_teachers=[(first, "deepseek-reasoner"), (second, "gpt-4.1")],
    )

    assert out[0]["cot_reasoning"] == "second reasoning"
    first.chat.completions.create.assert_called_once()
    second.chat.completions.create.assert_called_once()


def test_cloud_cot_fallback_concurrency_is_capped_at_16(monkeypatch):
    import threading
    import time
    from types import SimpleNamespace
    from data.curriculum import annotate_cot

    class TrackingCompletions:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def create(self, **_kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="cloud reasoning"))]
            )

    completions = TrackingCompletions()
    fallback = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    examples = [{"prompt": f"p{i}", "response": "r"} for i in range(40)]
    monkeypatch.setenv("SLM_SYNTH_CONCURRENCY", "32")

    out = annotate_cot(
        examples,
        generate_fn=lambda *_: "",
        fallback_teachers=[(fallback, "gpt-4.1")],
    )

    assert all(ex["cot_reasoning"] == "cloud reasoning" for ex in out)
    assert 1 < completions.max_active <= 16


def test_none_cot_is_treated_as_missing():
    from data.curriculum import annotate_cot

    out = annotate_cot(
        [{"prompt": "p", "response": "r", "cot_reasoning": None}],
        generate_fn=lambda *_: "generated reasoning",
    )

    assert out[0]["cot_reasoning"] == "generated reasoning"


def test_numeric_zero_answer_can_receive_cot():
    from data.curriculum import annotate_cot

    out = annotate_cot(
        [{"prompt": "What is zero?", "answer": 0}],
        generate_fn=lambda *_: "zero reasoning",
    )

    assert out[0]["cot_reasoning"] == "zero reasoning"


def test_annotate_cot_generate_fn_failure_is_nonfatal():
    """A per-call synth failure leaves the example CoT-less instead of crashing curate."""
    from data.curriculum import annotate_cot

    def gen(prompt, temperature, max_tokens):
        raise RuntimeError("synth endpoint down")

    out = annotate_cot([{"prompt": "a", "response": "b"}], generate_fn=gen, task_type="generation")
    assert out == [{"prompt": "a", "response": "b"}]    # unchanged, no crash


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


@pytest.mark.parametrize(
    "generated",
    [
        "not json",
        "{}",
        '{"text": "Aspirin may help.", "entities": []}',
        (
            '{"text": "No medicine is named here.", '
            '"entities": [{"text": "Aspirin", "type": "Chemical"}]}'
        ),
        (
            '{"text": "Aspirin may help.", '
            '"entities": [{"text": "Aspirin", "type": "Organization"}]}'
        ),
    ],
)
def test_ner_synthesis_discards_malformed_empty_or_invalid_entities(generated):
    from data.curriculum import synthesize_hard_negatives

    gold = {
        "text": "Aspirin may help.",
        "entities": [{"text": "Aspirin", "type": "Chemical"}],
    }

    results = synthesize_hard_negatives(
        [gold],
        n=1,
        task_type="NER",
        generate_fn=lambda *_args: generated,
    )

    assert results == [gold]


def test_ner_synthesis_keeps_only_present_spans_with_known_types():
    from data.curriculum import synthesize_hard_negatives

    gold = {
        "text": "Aspirin may help fever.",
        "entities": [
            {"text": "Aspirin", "type": "Chemical"},
            {"text": "fever", "type": "Disease"},
        ],
    }
    generated = (
        '{"text": "Fever may improve after Aspirin.", "entities": ['
        '{"text": "Aspirin", "type": "Chemical"}, '
        '{"text": "Fever", "type": "Disease"}]}'
    )

    results = synthesize_hard_negatives(
        [gold],
        n=1,
        task_type="NER",
        generate_fn=lambda *_args: generated,
        source_label="synth:test",
    )

    assert results == [
        gold,
        {
            "text": "Fever may improve after Aspirin.",
            "entities": [
                {"text": "Aspirin", "type": "Chemical"},
                {"text": "Fever", "type": "Disease"},
            ],
            "_source": "synth:test",
        },
    ]
