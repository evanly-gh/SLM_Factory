"""`iterate` has no fallback: an unusable orchestrator decision stops the run (B316).

WHY THIS FILE EXISTS
    `iterate_node` used to catch anything the orchestrator call or the validator raised and carry on
    with a substitute — the test agent's `suggested_intervention`, or failing that the score band from
    `apply_iteration_policy`. The intent was resilience. The effect was that a broken decision became
    an ordinary-looking iteration.

    A score-band rule is not a degraded version of this loop, it is a DIFFERENT ALGORITHM. The loop IS
    the orchestrator choosing an intervention from evidence, so running a rule and reporting the
    result under the same name makes the whole trajectory uninterpretable: afterwards you cannot tell
    which iterations were reasoned and which were guessed. Run 38658213 is the demonstration — five
    `data_rebuild` requests refused by a validator bug, five `using test-agent suggestion:
    hyperparameter` lines, a curriculum that never moved and a score that fell 0.789 → 0.690 while
    every surface in the system reported an ordinary run (B312).

    A run that cannot obtain a decision has nothing worth reporting, so it stops. These tests pin the
    three things that makes that useful rather than merely abrupt: it raises a NAMED error, the error
    carries the underlying cause, and the log block explains that the absence of a fallback is
    deliberate.

WHAT IS DELIBERATELY NOT HERE
    The B312 shape itself — a `data_rebuild` plan the validator refuses — lives in
    `test_iterate_rejected_data_plan.py`, together with the diagnosis that names the schema
    disagreement. This file is about the general contract for ANY unusable decision.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.nodes.iterate import apply_iteration_policy, iterate_node


def _exception_named(name: str) -> type[BaseException]:
    """Resolve an expected exception class at CALL time rather than at import time.

    `tests/config/test_curation_config.py` reloads `agent.nodes.iterate` to re-read its environment
    defaults. Reload re-executes the module into its existing namespace, so `OrchestratorDecisionError`
    is rebound to a NEW class object while `iterate_node` (which reads it from that namespace) raises
    the new one. A reference captured when this file was imported would therefore match or not match
    purely according to the order pytest happened to choose — which is randomised.
    """
    from agent import llm_errors
    from agent.nodes import iterate

    return {
        "OrchestratorDecisionError": iterate.OrchestratorDecisionError,
        "FatalLLMError": llm_errors.FatalLLMError,
    }[name]


def _decision_error() -> type[BaseException]:
    return _exception_named("OrchestratorDecisionError")


def _model():
    model = MagicMock()
    model.model_id = "unsloth/Qwen3-0.6B"
    model.quant = None
    model.tier = 0
    # A digit-free label, because `_log` prefixes every line with it and a MagicMock's default repr
    # carries its object id — which contains arbitrary digits and would let the assertions below that
    # look for an iteration number or a score pass on the prefix instead of on the block's content.
    model.label = "test/Model"
    return model


def _state(score=0.5, iteration=7):
    """A mid-run state that reaches the orchestrator call.

    The goal is pinned well above every score used here so the convergence route never fires, and the
    eval history is short and rising so neither stagnation nor the eval cap escalates first — both of
    those return before the orchestrator is consulted, which would make these tests vacuous.
    """
    return {
        "selected_model": _model(),
        "scores": [score],
        "eval_history": [score - 0.1, score - 0.05, score],
        "best_score": score,
        "iteration": iteration,
        "turn_budget": 1000,
        "stop_threshold": 0.99,
        "initial_stop_threshold": 0.99,
        "task": "clinc150",
        "last_eval": None,
        "hw_gating_enabled": False,
        "consecutive_no_improvement": 0,
        "test_report": {
            "suggested_intervention": "hyperparameter",
            "diagnosis": "the hard bucket is under-fit",
        },
    }


def _unsupported_intervention_decision():
    """A well-formed reply naming an intervention that does not exist.

    Rejected by `_validate_decision_json`, which is the failure `iterate_node` re-runs as defence in
    depth — so this reaches the fatal path through validation rather than through a raising stub.
    """
    return {
        "intervention": "resample",
        "hypothesis": "the curriculum needs re-drawing from the pool it came from",
    }


# --------------------------------------------------------------------------
# A decision that fails validation is fatal, and says what failed
# --------------------------------------------------------------------------


@patch("agent.nodes.iterate._llm_iterate")
def test_a_decision_that_fails_validation_raises(mock_llm):
    mock_llm.return_value = _unsupported_intervention_decision()

    with pytest.raises(_decision_error()):
        iterate_node(_state())


@patch("agent.nodes.iterate._llm_iterate")
def test_the_raised_error_names_the_underlying_failure(mock_llm):
    """The raise is what the runner reports, so it has to carry the diagnosis rather than only the
    fact that there was one. "could not obtain a decision" with the cause left in a log line above it
    is the shape that made B312 take a whole GPU run to notice."""
    mock_llm.return_value = _unsupported_intervention_decision()

    with pytest.raises(_decision_error()) as excinfo:
        iterate_node(_state())

    message = str(excinfo.value)
    assert "resample" in message
    assert "data_rebuild" in message and "hyperparameter" in message


@patch("agent.nodes.iterate._llm_iterate")
def test_the_underlying_exception_is_kept_as_the_cause(mock_llm):
    """Chained rather than replaced, so the traceback still shows where validation refused the
    decision. A bare raise loses the only pointer to the field that was wrong."""
    mock_llm.return_value = _unsupported_intervention_decision()

    with pytest.raises(_decision_error()) as excinfo:
        iterate_node(_state())

    assert isinstance(excinfo.value.__cause__, ValueError)


# --------------------------------------------------------------------------
# The fatal log block
# --------------------------------------------------------------------------


def _fatal_block(log: str) -> str:
    """Everything the node printed from the FATAL banner onwards."""
    assert "FATAL" in log, (
        "the fatal path printed no banner, so a reader scrolling the run log sees a bare traceback"
    )
    return log[log.index("FATAL"):]


@patch("agent.nodes.iterate._llm_iterate")
def test_the_fatal_block_says_the_absence_of_a_fallback_is_deliberate(mock_llm, capsys):
    """Otherwise the next reader restores the fallback.

    The line exists because "iterate raised" looks like an oversight, and the whole point of B316 is
    that stopping is the CHOSEN behaviour: a score-band rule run under this node's name is how B312
    stayed invisible for a whole run.
    """
    mock_llm.return_value = _unsupported_intervention_decision()

    with pytest.raises(_decision_error()):
        iterate_node(_state())

    block = _fatal_block(capsys.readouterr().out).lower()
    assert "no fallback" in block
    assert "score-band" in block


@patch("agent.nodes.iterate._llm_iterate")
def test_the_fatal_block_names_the_iteration_and_the_score(mock_llm, capsys):
    """A run log holds thousands of lines from many iterations. Without the iteration number the
    block cannot be tied to the decision input logged above it, which is the only way to see WHAT the
    orchestrator was asked before it produced something unusable."""
    mock_llm.return_value = _unsupported_intervention_decision()

    with pytest.raises(_decision_error()):
        iterate_node(_state(score=0.5, iteration=7))

    block = _fatal_block(capsys.readouterr().out)
    assert "7" in block, "the fatal block does not identify which iteration failed"
    assert "0.5" in block, "the fatal block does not record the score the decision was made against"


@patch("agent.nodes.iterate._llm_iterate")
def test_the_fatal_block_quotes_the_error_it_could_not_recover_from(mock_llm, capsys):
    mock_llm.return_value = _unsupported_intervention_decision()

    with pytest.raises(_decision_error()):
        iterate_node(_state())

    block = _fatal_block(capsys.readouterr().out)
    assert "ValueError" in block
    assert "resample" in block


# --------------------------------------------------------------------------
# There is no "resilient" branch left for anything
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected_name"),
    (
        (RuntimeError("Overloaded: please try again in a moment"), "OrchestratorDecisionError"),
        (ValueError("no parseable JSON object in LLM response"), "OrchestratorDecisionError"),
        # A transport failure is classified fatal by `raise_if_fatal` and leaves through THAT door
        # instead, one line earlier. Included so both exits are pinned: a run that stopped on the
        # transient-looking one but limped on for the genuinely broken one would be the worst of both.
        (TimeoutError("read timed out"), "FatalLLMError"),
    ),
    ids=["overloaded", "unparseable", "timeout"],
)
def test_no_exception_from_the_orchestrator_call_is_absorbed(exc, expected_name):
    """A transient-LOOKING failure gets the same treatment, and that is the deliberate part.

    The old policy was "a single fallback is reasonable for a timeout or a malformed reply". But one
    substituted iteration is still an iteration whose recorded reasoning never happened, and no later
    reader can tell which one it was. Neither error text nor error class buys a resilient branch here,
    because there is no longer a resilient branch to reach.
    """
    state = _state()

    with patch("agent.nodes.iterate._llm_iterate", side_effect=exc):
        with pytest.raises(_exception_named(expected_name)):
            iterate_node(state)

    assert state.get("next_action") not in ("train", "curate")


@pytest.mark.parametrize(
    "score",
    (0.50, 0.85, 0.97),
    ids=["data_band", "hyperparameter_band", "refinement_band"],
)
def test_the_score_band_rule_is_no_longer_a_source_of_interventions(score):
    """The band still labels the log line; it no longer decides anything.

    Parametrized across all three bands of `apply_iteration_policy` because the old fallback's choice
    depended on the score, so a partially-removed fallback would survive in exactly one band. Every
    band must now stop rather than route to train or curate.
    """
    assert apply_iteration_policy(score)["intervention"] in ("data_rebuild", "hyperparameter")
    state = _state(score=score)

    with patch(
        "agent.nodes.iterate._llm_iterate",
        side_effect=ValueError("hypothesis must be a non-empty string"),
    ):
        with pytest.raises(_decision_error()):
            iterate_node(state)

    assert state.get("next_action") not in ("train", "curate")
    assert not state.get("data_rebuild_plan")
