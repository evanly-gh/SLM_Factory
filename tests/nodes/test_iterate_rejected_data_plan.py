"""What `iterate` does when the orchestrator asks for data and its plan is refused (B312/B316).

WHY THIS FILE EXISTS
    On verification run 38658213 the orchestrator asked for a `data_rebuild` on five consecutive
    iterations and `normalize_data_rebuild_plan` refused every plan — the prompt showed a schema
    carrying `hypothesis` and `task`, the validator's accepted set did not list them. The refusal
    landed in the same `except` block as a network error, and that block substituted a fallback
    intervention: five `using test-agent suggestion: hyperparameter` lines, indistinguishable from an
    orchestrator that had wanted hyperparameters. No data intervention ran in the whole run and the
    score fell 0.789 → 0.690 while every surface reported an ordinary run.

    Nothing in the log said otherwise, and that is the part worth fixing. The recorded reasoning did
    not explain the run's own behaviour, which is the most misleading thing this node can produce.

WHAT CHANGED, AND WHY IT IS NOT A COUNTER
    The first version of this file pinned the fallback and asked only that it announce itself and
    count the refusals, stopping the run on the second. That kept a score-band rule running under the
    loop's name, and a score-band rule is a DIFFERENT ALGORITHM rather than a degraded version of the
    loop — the loop IS the orchestrator choosing an intervention from evidence. Reporting a rule's
    output as the loop's leaves a trajectory nobody can read afterwards: you cannot tell which
    iterations were reasoned and which were guessed.

    So `iterate` now raises `OrchestratorDecisionError` on the FIRST unusable decision. The B312
    scenario below is unchanged; only the correct outcome moved, from "substitutes a hyperparameter
    step" to "stops the run and says why" (B316).
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.data_rebuild import DATA_REBUILD_SCHEMA_VERSION, SURGICAL_SYNTHESIS
from agent.nodes.iterate import _wanted_data_rebuild, iterate_node


def _decision_error() -> type[BaseException]:
    """Resolve the class at CALL time rather than at import time.

    `tests/config/test_curation_config.py` reloads `agent.nodes.iterate` to re-read its environment
    defaults. Reload re-executes the module into its existing namespace, so this class is rebound to a
    NEW object while `iterate_node` raises the new one. A reference captured when this file was
    imported would match or not match purely according to the order pytest happened to choose.
    """
    from agent.nodes import iterate

    return iterate.OrchestratorDecisionError

# The literal message run 38658213 produced, five times. `hypothesis` and `task` are accepted now;
# what this file needs is a plan the validator still refuses, so the decision below carries
# `resample_fraction` — a field of the `resample` strategy deleted on 2026-08-19.
RUN_38658213_ERROR = (
    "data_rebuild has unknown field(s) ['hypothesis', 'task']; allowed: "
    "['pattern_hint', 'rows', 'schema_version', 'strategy', 'target_categories']"
)


def _model():
    model = MagicMock()
    model.model_id = "unsloth/Qwen3-0.6B"
    model.quant = None
    model.tier = 0
    # Pinned rather than left as a MagicMock, whose repr carries the object id: `_log` prefixes every
    # line with this, so an arbitrary run of digits and words there can satisfy an assertion about the
    # block's own content.
    model.label = "test/Model"
    return model


def _state():
    """A mid-run state below its goal, with room left on every budget.

    Deliberately NOT stagnant and well short of the eval cap: those routes escalate before the
    orchestrator is consulted at all, and this file is about the turn where it IS consulted.
    """
    return {
        "selected_model": _model(),
        "scores": [0.789],
        "eval_history": [0.72, 0.75, 0.789],
        "best_score": 0.789,
        "iteration": 3,
        "turn_budget": 1000,
        "stop_threshold": 0.90,
        "initial_stop_threshold": 0.90,
        "task": "xlam_bfcl",
        "last_eval": None,
        "hw_gating_enabled": False,
        "consecutive_no_improvement": 0,
        # The run's own shape: the test agent suggested hyperparameters, so the old fallback had
        # something plausible to substitute and the swap left no trace. Kept in the fixture so this
        # file still reproduces the exact state that made B312 invisible — the point is that a
        # plausible substitute being available no longer matters.
        "test_report": {
            "suggested_intervention": "hyperparameter",
            "diagnosis": "the hard bucket is under-fit",
        },
    }


def _unusable_data_decision():
    """An orchestrator decision asking for a data rebuild with a plan the validator refuses."""
    return {
        "intervention": "data_rebuild",
        "hypothesis": "argument construction fails on rows needing more than one call",
        "data_rebuild": {
            "schema_version": DATA_REBUILD_SCHEMA_VERSION,
            "strategy": SURGICAL_SYNTHESIS,
            "rows": 400,
            "target_categories": [{"category": "wrong_arguments", "count": 147}],
            "pattern_hint": "nested argument objects",
            "resample_fraction": 0.5,
        },
    }


@patch("agent.nodes.iterate._llm_iterate")
def test_a_refused_data_plan_stops_the_run(mock_llm):
    """The whole of B312 in one assertion: the turn does not happen at all.

    Run 38658213 needed five iterations and a GPU to notice; a raise costs one.
    """
    mock_llm.return_value = _unusable_data_decision()

    with pytest.raises(_decision_error()):
        iterate_node(_state())


@patch("agent.nodes.iterate._llm_iterate")
def test_no_hyperparameter_step_is_substituted_for_the_refused_data_plan(mock_llm):
    """The specific swap that made the trajectory unreadable must not happen even partially.

    The old fallback wrote `last_intervention="hyperparameter"` and routed to train, so the run's own
    record attributed a tuning decision to an orchestrator that had asked for data. Asserting on the
    state as well as on the raise catches a future fallback that sets the fields before giving up.
    """
    mock_llm.return_value = _unusable_data_decision()
    state = _state()

    with pytest.raises(_decision_error()):
        iterate_node(state)

    assert state.get("last_intervention") != "hyperparameter"
    assert state.get("next_action") not in ("train", "curate")


@patch("agent.nodes.iterate._llm_iterate")
def test_the_log_identifies_a_prompt_validator_schema_disagreement(mock_llm, capsys):
    """A refused DATA plan has a different cause from a malformed reply, and needs a different fix.

    A parse failure is a one-off; a plan the validator refuses means the orchestrator prompt and
    `normalize_data_rebuild_plan` disagree about the schema, which recurs on every call until someone
    changes one of them. The log has to say that, because the reader's next action depends on it.
    """
    mock_llm.return_value = _unusable_data_decision()

    with pytest.raises(_decision_error()):
        iterate_node(_state())

    log = capsys.readouterr().out
    assert "data_rebuild" in log
    assert "normalize_data_rebuild_plan" in log
    assert "schema" in log
    # Names the field that was actually refused, so the disagreement can be located rather than
    # searched for.
    assert "resample_fraction" in log


@patch("agent.nodes.iterate._llm_iterate")
def test_a_refused_hyperparameter_decision_is_not_diagnosed_as_a_schema_disagreement(mock_llm):
    """The diagnosis has to be specific to be worth printing.

    A rejected hyperparameter decision is an ordinary validation failure and still fatal, but naming
    `normalize_data_rebuild_plan` for it would send the reader to the wrong file.
    """
    mock_llm.return_value = {
        "intervention": "hyperparameter",
        "hypothesis": "the hard bucket is under-fit",
        "hyperparams": {"lora_rank": "sixteen"},
    }

    with pytest.raises(_decision_error()) as excinfo:
        iterate_node(_state())

    assert "normalize_data_rebuild_plan" not in str(excinfo.value)


def test_the_run_38658213_error_is_recognised_as_a_data_rebuild_request():
    """Read from the error text because the decision did not survive validation — that is what
    "rejected" means, and there is no parsed decision left to inspect."""
    assert _wanted_data_rebuild(ValueError(RUN_38658213_ERROR)) is True


def test_an_unrelated_failure_is_not_reported_as_a_refused_data_plan():
    """The diagnosis is only useful if it discriminates. A transient network error and a rejected
    hyperparameter config are both fatal now, but neither is a schema disagreement, and claiming one
    would point the reader at a file that is working."""
    assert _wanted_data_rebuild(RuntimeError("connection reset by peer")) is False
    assert _wanted_data_rebuild(
        ValueError("hyperparams.lora_rank must have integer JSON type")
    ) is False
