import json

import pytest

from training.lora_trainer import (
    CompletionOnlyDataCollator,
    _build_completion_only_rows,
    text_tokenizer,
)


class CharTokenizer:
    chat_template = "test-template"
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 3
    eos_token = "<eos>"

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.template_calls.append({
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        })
        prompt = f"<user>{messages[0]['content']}</user><assistant>"
        if add_generation_prompt:
            return prompt
        return f"{prompt}{messages[-1]['content']}</assistant>"

    def __call__(
        self,
        text,
        *,
        truncation,
        add_special_tokens,
    ):
        assert truncation is False
        assert add_special_tokens is True
        return {"input_ids": [ord(char) + 10 for char in text]}


# One case per SHAPE of training turn a task can declare. Two tasks naming the same builder is
# fine and expected (xlam and calendar both use `function_call_turn`); what matters is that each
# builder masks the prompt and trains the target.
TASK_CASES = [
    (
        "clinc150",
        {"text": "A joyful message", "label": "joy"},
        "joy",
    ),
    (
        "ner_bc5cdr",
        {
            "text": "Aspirin treats pain.",
            "entities": [{"text": "Aspirin", "type": "CHEMICAL"}],
        },
        '"CHEMICAL"',
    ),
    (
        "dialogsum",
        {
            "text": "Summarize this.",
            "answer": "A short summary.",
            "cot_reasoning": "Identify the key point.",
        },
        "<reasoning>\nIdentify the key point.",
    ),
    (
        "gsm8k",
        {
            "prompt": "What is 2 + 2?",
            "answer": "4",
            "cot_reasoning": "Add the two values.",
        },
        "<reasoning>\nAdd the two values.",
    ),
    (
        "xlam_bfcl",
        {
            "text": "What is the weather in Paris?",
            "answer": '[{"name": "get_weather", "arguments": {"city": "Paris"}}]',
            "tools": [{"name": "get_weather", "parameters": {"city": "string"}}],
        },
        '"get_weather"',
    ),
]


@pytest.mark.parametrize(("task", "example", "target_text"), TASK_CASES)
def test_completion_only_mask_trains_targets_not_prompts(
    task,
    example,
    target_text,
):
    tokenizer = CharTokenizer()

    row = _build_completion_only_rows(
        [example],
        tokenizer,
        task,
    )[0]
    batch = CompletionOnlyDataCollator(tokenizer.pad_token_id)([row])
    labels = batch["labels"][0].tolist()
    input_ids = batch["input_ids"][0].tolist()
    completion_mask = row["completion_mask"]

    prompt_length = completion_mask.index(1)
    assert labels[:prompt_length] == [-100] * prompt_length
    assert labels[prompt_length:] == input_ids[prompt_length:]
    assert target_text in row["text"]
    assert any(label != -100 for label in labels)
    assert [call["add_generation_prompt"] for call in tokenizer.template_calls] == [
        True,
        False,
    ]
    assert all(
        call["enable_thinking"] is False
        for call in tokenizer.template_calls
    )


def test_completion_only_collator_masks_padding_and_prompt_tokens():
    tokenizer = CharTokenizer()
    rows = _build_completion_only_rows(
        [
            {"text": "short", "label": "yes"},
            {"text": "a much longer prompt", "label": "no"},
        ],
        tokenizer,
        "clinc150",
    )

    batch = CompletionOnlyDataCollator(tokenizer.pad_token_id)(rows)

    for row_index, row in enumerate(rows):
        labels = batch["labels"][row_index].tolist()
        real_length = len(row["input_ids"])
        for index, is_completion in enumerate(row["completion_mask"]):
            expected = row["input_ids"][index] if is_completion else -100
            assert labels[index] == expected
        assert all(label == -100 for label in labels[real_length:])


def test_multimodal_text_only_uses_inner_tokenizer_completion_mask():
    class Qwen35Processor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    inner = CharTokenizer()
    processor = Qwen35Processor(inner)

    row = _build_completion_only_rows(
        [{"text": "happy", "label": "joy"}],
        text_tokenizer(processor),
        "clinc150",
    )[0]
    labels = CompletionOnlyDataCollator(inner.pad_token_id)([row])[
        "labels"
    ][0].tolist()

    assert labels[: row["completion_mask"].index(1)] == [-100] * row[
        "completion_mask"
    ].index(1)
    assert any(label != -100 for label in labels)
    assert json.loads(json.dumps(row["input_ids"])) == row["input_ids"]
