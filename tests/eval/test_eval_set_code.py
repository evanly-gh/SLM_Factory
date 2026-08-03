from data.eval_set import build_eval_set


def test_code_eval_uses_full_requested_800_row_budget():
    rows = [
        {
            "text": f"problem {index}",
            "input_output": {
                "inputs": [f"{index}\n"],
                "outputs": [f"{index}\n"],
            },
            "execution_mode": "stdin",
        }
        for index in range(800)
    ]

    eval_set = build_eval_set(
        rows,
        task_type="code_generation",
        target=800,
    )

    assert len(eval_set.all) == 800
