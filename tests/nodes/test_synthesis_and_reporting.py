"""Regressions from the xlam `single_model` run 38566712 (B291-B298) that still apply.

That run reached 0.8530 in 16 iterations and every one of its reports was, in some way, describing
something other than what happened. The four defects worth keeping under test are:

  B291  The `synthesize` strategy was a SILENT no-op on `function_call`. `synthesize_examples`
        dispatched on `("classification", "NER")` or a generation-family set, and `function_call` was
        in neither, so it fell through to a bare `return []`. The orchestrator chose synthesis for
        six of eight rebuilds, announced 250-500 rows each time against a healthy teacher endpoint,
        and got zero rows every time with nothing in the log naming a cause. The exact verifiers
        written for that exact path had therefore never once executed in production.

  B296  Every open-ended failure was reported as the constant confusion pair
        `gold_verifier -> incorrect`, whose count is the failure count the orchestrator already has.
        It then wrote paragraphs reasoning about that constant as though it were a class confusion.

  B298  The post-run accuracy chart drew ONE flat threshold line at the final value, so a run whose
        goal moved mid-flight was shown as if every iteration had been held to the last bar. Lowers
        were not recorded anywhere at all. And the trajectory table's Config column listed four
        hyperparameters no intervention can change.

The shape of the fix has changed — there is no dispatch set to be missing from, because each task
names its own behaviour — so these are written against the new design rather than ported verbatim.
"""
from __future__ import annotations

import json

import pytest

from data.eval_set import EvalSet

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]


def _call(city="Paris"):
    return json.dumps([{"name": "get_weather", "arguments": {"city": city}}])


# --------------------------------------------------------------------------
# B291 — the synthesis path that was dead
# --------------------------------------------------------------------------


def test_a_format_bound_task_actually_produces_rows(monkeypatch):
    """End to end on the path that returned `[]` for six consecutive rebuilds: generate, verify
    against the row's own declared tool schema, keep."""
    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")  # isolate generation + the programmatic verifier
    from data.curriculum import synthesize_examples
    from tasks import get_task

    anchors = [
        {"text": f"weather in city{i}?", "answer": _call(f"city{i}"), "tools": TOOLS}
        for i in range(6)
    ]

    def generate(_prompt, *_args, **_kwargs):
        return json.dumps({"text": "weather in Berlin?", "answer": _call("Berlin")})

    rows = synthesize_examples(
        anchors, task="xlam_bfcl", n=4, generate_fn=generate,
        verify_fn=get_task("xlam_bfcl").synth_verifier, log=None,
    )

    assert len(rows) == 4, "the format-bound synthesis path produced nothing"
    # `tools` is PINNED from the anchor rather than trusted to the teacher: without it the row cannot
    # be schema-checked at all, which is what made the verifier vacuous.
    assert all(row["tools"] == TOOLS for row in rows)


def test_a_generated_row_failing_the_exact_verifier_is_dropped(monkeypatch):
    """The verifier runs BEFORE any teacher call — it is free and cannot be fooled — so a row it
    rejects never costs one."""
    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")
    from data.curriculum import synthesize_examples
    from tasks import get_task

    def generate(_prompt, *_args, **_kwargs):
        # Calls a function the row does not declare — the check the eval scorer also applies, so a
        # row failing it is unwinnable by construction and would cap the ceiling below 1.0.
        return json.dumps({
            "text": "launch?",
            "answer": json.dumps([{"name": "launch_missiles", "arguments": {}}]),
        })

    rows = synthesize_examples(
        [{"text": "weather?", "answer": _call(), "tools": TOOLS}],
        task="xlam_bfcl", n=3, generate_fn=generate,
        verify_fn=get_task("xlam_bfcl").synth_verifier, log=None,
    )
    assert rows == []


def test_a_closed_label_space_task_generates_a_new_input_for_an_existing_class(monkeypatch):
    """The other row shape, DERIVED from the task rather than chosen by a channel. The generated row
    inherits a real anchor's label, so the target cannot be wrong and an out-of-vocabulary label is
    impossible by construction."""
    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")
    from data.curriculum import synthesize_examples

    anchors = [
        {"text": "move money to savings", "label": "transfer"},
        {"text": "how much is in checking", "label": "balance"},
    ]
    rows = synthesize_examples(
        anchors, task="clinc150", n=4,
        generate_fn=lambda *_a, **_k: "please shift funds across accounts", log=None,
    )

    assert rows
    assert {row["label"] for row in rows} <= {"transfer", "balance"}
    assert all(row["_provenance"] == "synthetic_positive" for row in rows)


def test_an_unknown_task_cannot_reach_synthesis_at_all():
    """It used to fall through to `return []`, so a rebuild announced 250-500 rows and produced none,
    silently. An unknown task now names every task that does exist."""
    from data.curriculum import synthesize_examples

    with pytest.raises(ValueError, match=r"unknown task 'not_a_task'.*registry holds"):
        synthesize_examples(
            [{"text": "t", "answer": "[]"}], task="not_a_task", n=5,
            generate_fn=lambda *_a, **_k: "{}", log=None,
        )


def test_a_failing_teacher_yields_fewer_rows_rather_than_raising(monkeypatch):
    """Non-fatal throughout: synthesis is one call per row and a rebuild must survive a bad one."""
    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")
    from data.curriculum import synthesize_examples

    def generate(*_args, **_kwargs):
        raise RuntimeError("teacher fell over")

    assert synthesize_examples(
        [{"text": "weather?", "answer": _call(), "tools": TOOLS}],
        task="xlam_bfcl", n=4, generate_fn=generate, log=None,
    ) == []


def test_the_row_count_reached_is_stated_in_the_log(monkeypatch):
    """The line that was missing. "Announced 400, kept 0" has to be readable, because a silent zero
    is indistinguishable from a mechanism that does not exist."""
    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")
    from data.curriculum import synthesize_examples

    logs: list[str] = []
    synthesize_examples(
        [{"text": "weather?", "answer": _call(), "tools": TOOLS}],
        task="xlam_bfcl", n=4, generate_fn=lambda *_a, **_k: "not json", log=logs.append,
    )
    joined = " ".join(logs)
    assert "requested 4" in joined and "kept 0" in joined


# --------------------------------------------------------------------------
# B296 — failure categories, not a constant
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("not json at all", "unparseable_output"),
    ('[{"name": "launch_missiles", "arguments": {}}]', "undeclared_function"),
    ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, '
     '{"name": "get_weather", "arguments": {"city": "Oslo"}}]', "wrong_call_count"),
    ('[{"name": "get_weather", "arguments": {"city": "Berlin"}}]', "wrong_arguments"),
])
def test_a_function_call_failure_names_its_own_cause(raw, expected):
    """Four categories calling for four different interventions, where there used to be one constant
    string: unparseable output is a format/template problem, an undeclared function is an
    unwinnable-row or prompt problem, the wrong count is a tool-selection problem, and wrong
    arguments is an extraction problem."""
    from eval.scorers.function_call import extract_predictions, score

    row = {"text": "q", "tools": TOOLS, "answer": _call("Paris")}
    eval_set = EvalSet(all=[row], task="xlam_bfcl")
    result = score(eval_set, extract_predictions([raw], eval_set))
    assert [failure["error_type"] for failure in result["failures"]] == [expected]


def test_the_test_report_carries_the_taxonomy_rather_than_one_bucket():
    """`build_test_report` reads `error_type`, so the confusion pairs the orchestrator reasons about
    name something the scorer measured. In run 38566712 they were pages about a constant."""
    from agent.nodes.test_agent import build_test_report
    from eval.harness import EvalResult

    failures = (
        [{"text": f"a{i}", "error_type": "wrong_arguments"} for i in range(7)]
        + [{"text": f"b{i}", "error_type": "unparseable_output"} for i in range(3)]
    )
    eval_set = EvalSet(
        all=[{"text": f"a{i}", "answer": "[]"} for i in range(7)]
            + [{"text": f"b{i}", "answer": "[]"} for i in range(3)],
        task="xlam_bfcl",
    )
    report = build_test_report(
        eval_set, EvalResult(f1=0.0, per_class={}, failures=failures),
        {"easy": ["a0"], "medium": ["a1"], "hard": ["b0"]}, 0.9, "xlam_bfcl",
    )

    golds = {pair["gold"]: pair["count"] for pair in report["confusion_pairs"]}
    assert golds == {"wrong_arguments": 7, "unparseable_output": 3}
    assert "gold_verifier" not in golds


def test_a_correct_prediction_produces_no_failure_record():
    from eval.scorers.function_call import extract_predictions, score

    row = {"text": "q", "tools": TOOLS, "answer": _call("Paris")}
    eval_set = EvalSet(all=[row], task="xlam_bfcl")
    result = score(eval_set, extract_predictions([_call("Paris")], eval_set))
    assert result["failures"] == []
    assert result["f1"] == 1.0


def test_a_category_can_be_recovered_from_a_failure_written_before_the_taxonomy_existed():
    """A resumed checkpoint's report must be as informative as a fresh one, so the category is
    recomputed rather than reported as "unknown"."""
    from eval.scorers.function_call import failure_category_of

    assert failure_category_of({
        "text": "q", "tools": TOOLS, "answer": _call("Paris"),
        "predicted": [{"name": "get_weather", "arguments": {"city": "Berlin"}}],
    }) == "wrong_arguments"


# --------------------------------------------------------------------------
# B298 — a moving goal, drawn as a moving goal
# --------------------------------------------------------------------------


def test_the_threshold_line_follows_its_per_iteration_values():
    """A run whose goal moved mid-flight was drawn as if every iteration had been held to the last
    bar, which makes early iterations look like failures against a target they never had."""
    from agent.run_graphics import _threshold_series

    records = [{"stop_threshold": value} for value in (0.90, 0.90, 0.86, 0.86, 0.92)]
    assert _threshold_series(records, 0.92) == [0.90, 0.90, 0.86, 0.86, 0.92]


def test_the_threshold_line_stays_continuous_across_older_nodes():
    """DAG nodes written before the field existed carry no value; the line must carry forward, never
    drop to zero."""
    from agent.run_graphics import _threshold_series

    records = [{"stop_threshold": None}, {"stop_threshold": 0.90}, {"stop_threshold": None}]
    assert _threshold_series(records, 0.88) == [0.88, 0.90, 0.90]


def test_no_threshold_line_is_drawn_when_nothing_was_recorded():
    """Better than a flat invented line, which is the bug."""
    from agent.run_graphics import _threshold_series

    assert _threshold_series([{"score": 0.5}, {"score": 0.6}], 0.9) is None


def test_lowering_the_goal_leaves_a_durable_record_not_just_a_log_line():
    """Only RAISES were audited. A lowered goal left one log line and nothing on the state, so a
    converged run could not afterwards be checked against the bar it was actually held to — and the
    accuracy chart had no per-iteration value to draw."""
    from unittest.mock import MagicMock, patch

    from agent.nodes.iterate import iterate_node

    model = MagicMock(model_id="unsloth/Qwen3-0.6B", quant=None, tier=0)
    state = {
        "selected_model": model,
        "scores": [0.62],
        "eval_history": [0.55, 0.62],
        "best_score": 0.62,
        "iteration": 6,
        "turn_budget": 1000,
        "stop_threshold": 0.90,
        "initial_stop_threshold": 0.80,
        "task": "gsm8k",
        "last_eval": None,
        "hw_gating_enabled": False,
        "consecutive_no_improvement": 0,
        "dag": [],
    }
    decision = {
        "intervention": "hyperparameter",
        "hypothesis": "the failures look like a model-capacity limit",
        "hyperparams": {"lora_rank": 16, "alpha_ratio": 2, "weight_decay": 0.01,
                        "learning_rate": 2e-4, "nr_epochs": 3},
        "threshold_adjustment": {"new_threshold": 0.84, "reason": "capacity limit"},
    }

    with patch("agent.nodes.iterate._llm_iterate", return_value=decision):
        out = iterate_node(state)

    assert out["stop_threshold"] == pytest.approx(0.84)
    lowers = out["threshold_lowers"]
    assert len(lowers) == 1
    assert lowers[0]["from"] == pytest.approx(0.90)
    assert lowers[0]["to"] == pytest.approx(0.84)
    assert lowers[0]["iteration"] == 6
    assert lowers[0]["reason"] == "capacity limit"


def test_the_goal_can_never_be_lowered_below_the_run_own_floor():
    """The floor is `initial_stop_threshold`. Without the clamp the orchestrator could lower the bar
    until whatever it had already achieved counted as convergence."""
    from unittest.mock import MagicMock, patch

    from agent.nodes.iterate import iterate_node

    model = MagicMock(model_id="unsloth/Qwen3-0.6B", quant=None, tier=0)
    state = {
        "selected_model": model,
        "scores": [0.40],
        "eval_history": [0.35, 0.40],
        "best_score": 0.40,
        "iteration": 6,
        "turn_budget": 1000,
        "stop_threshold": 0.90,
        "initial_stop_threshold": 0.80,
        "task": "gsm8k",
        "last_eval": None,
        "hw_gating_enabled": False,
        "consecutive_no_improvement": 0,
        "dag": [],
    }
    decision = {
        "intervention": "hyperparameter",
        "hypothesis": "lower the bar as far as it will go",
        "hyperparams": {"lora_rank": 16, "alpha_ratio": 2, "weight_decay": 0.01,
                        "learning_rate": 2e-4, "nr_epochs": 3},
        "threshold_adjustment": {"new_threshold": 0.10, "reason": "give up"},
    }

    with patch("agent.nodes.iterate._llm_iterate", return_value=decision):
        out = iterate_node(state)

    assert out["stop_threshold"] == pytest.approx(0.80)


def test_the_config_label_shows_only_what_an_intervention_can_change():
    """The Config column listed four hyperparameters no intervention can change, so two rows looked
    different when nothing about them had. Batch shape is derived by the trainer from the device and
    `lora_dropout` is fixed at 0.0."""
    from agent.nodes.train import _label_for

    config = {
        "lora_rank": 32, "lora_alpha": 128, "lora_dropout": 0.0, "weight_decay": 0.01,
        "learning_rate": 1e-4, "nr_epochs": 5, "micro_batch_size": 8,
        "gradient_accumulation_steps": 1, "effective_batch_size": 8,
    }
    label = _label_for(config)

    for token in ("r=", "a=", "wd=", "lr=", "ep="):
        assert token in label
    for token in ("drop=", "mb=", "ga=", "eb="):
        assert token not in label
    # Changing a tunable changes the label; changing derived batch shape does not.
    assert _label_for({**config, "lora_rank": 64}) != label
    assert _label_for({**config, "nr_epochs": 6}) != label
    assert _label_for({**config, "micro_batch_size": 4}) == label


def test_the_label_covers_exactly_the_tunable_set():
    """`alpha_ratio` appears as the derived `a=`; the other four appear by name. Five fields are
    tunable and only five."""
    from agent.nodes.iterate import _TUNABLE_HYPERPARAMS

    assert _TUNABLE_HYPERPARAMS == {
        "lora_rank", "alpha_ratio", "weight_decay", "learning_rate", "nr_epochs",
    }


def test_a_data_rebuild_row_is_not_annotated_with_a_carried_forward_config():
    """`[carry-fwd best]` is the DEFINITION of a non-hyperparameter intervention, already named in
    the adjacent column."""
    import inspect

    from agent.nodes import train

    assert "carry-fwd best" not in inspect.getsource(train)
