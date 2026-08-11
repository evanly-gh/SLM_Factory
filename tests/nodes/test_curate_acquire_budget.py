"""curate's `acquire` strategy must hand the paid-acquisition ledger a usable plan identity.

Regression for B220: curate passed a hardcoded ``plan_identity=""`` into
``mine_additional_real_rows``. That is inert until the local and benchmark candidates are all
rejected and mining falls through to a *paid* round, at which point
``reserve_paid_acquisition`` raises ``ValueError: plan_identity must be non-empty`` and kills
the run mid-curate.
"""
import inspect

from agent.nodes import curate as curate_module


def test_curate_source_never_passes_an_empty_plan_identity():
    source = inspect.getsource(curate_module.curate_node)

    assert 'plan_identity=""' not in source, (
        "curate must not hand an empty plan identity to the paid-acquisition ledger; "
        "reserve_paid_acquisition rejects it and the run dies mid-curate"
    )
    assert "plan_budget_identity" in source


def test_curate_passes_a_ledger_acceptable_identity_for_acquire(monkeypatch):
    """The value curate forwards must be non-empty and accepted by the real ledger guard."""
    from agent.data_rebuild import plan_budget_identity
    from data.acquisition_budget import reserve_paid_acquisition

    plan = {
        "schema_version": 2,
        "strategy": "acquire",
        "target_rows": 5000,
        "new_real_rows": 200,
        "max_acquire_rounds": 2,
    }
    identity = plan_budget_identity(plan)

    assert isinstance(identity, str) and identity.strip()
    # Exercise the exact guard that killed the run, against a throwaway ledger.
    reserve_paid_acquisition(
        plan_identity=identity,
        per_plan_limit=1,
        run_limit=1,
        path="/tmp/slm-b220-guard-check.jsonl",
    )
