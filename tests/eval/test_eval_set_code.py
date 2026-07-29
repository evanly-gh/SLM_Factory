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
        n_pos=320,
        n_neg=320,
        n_boundary=160,
    )

    assert len(eval_set.pos) == 320
    assert len(eval_set.neg) == 320
    assert len(eval_set.boundary) == 160
    assert len(eval_set.all) == 800
