from data import curriculum


def _fake_gen(prompt, temperature=0.7, max_tokens=200):
    # A plausible math problem+answer JSON for the new-correct-example path.
    return '{"text": "What is 2+2?", "answer": "4", "cot_reasoning": "2+2=4"}'


def test_a_new_correct_task_produces_whole_new_pairs():
    seed = [{"text": "What is 1+1?", "answer": "2"}]
    out = curriculum.synthesize_examples(
        seed,
        task="gsm8k",
        n=1,
        generate_fn=_fake_gen,
        verify_fn=lambda row: row.get("answer") == "4",
    )
    assert len(out) == 1
    assert out[0]["answer"] == "4"  # correct, never a wrong answer as a positive target
    assert out[0].get("_source", "").startswith("synth:")


def test_a_new_gold_task_delegates_to_the_in_class_generator():
    seed = [{"text": "win a free prize now", "label": "spam"}]
    out = curriculum.synthesize_examples(
        seed,
        task="clinc150",
        n=1,
        generate_fn=lambda p, *a, **k: "call me about the meeting",
    )
    assert out and out[0].get("label") is not None


def test_verify_fn_filters_incorrect_rows():
    seed = [{"text": "q", "answer": "2"}]
    out = curriculum.synthesize_examples(
        seed,
        task="gsm8k",
        n=3,
        generate_fn=_fake_gen,
        verify_fn=lambda row: False,  # reject everything
    )
    assert out == []


def test_empty_inputs_return_empty():
    assert curriculum.synthesize_examples([], task="dialogsum", n=5,
                                          generate_fn=_fake_gen) == []
    assert curriculum.synthesize_examples([{"text": "x"}], task="dialogsum",
                                          n=0, generate_fn=_fake_gen) == []
