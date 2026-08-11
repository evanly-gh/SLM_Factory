import importlib


def test_curriculum_floor_is_at_least_3000(monkeypatch):
    # Requirement: the lowest target dataset size is 3000 for any task, so the
    # per-task floor must be >= 3000. (The repo default may be set higher, e.g. 5000.)
    monkeypatch.delenv("SLM_CURRICULUM_FLOOR", raising=False)
    import config.config as cfg
    importlib.reload(cfg)
    assert cfg.CURRICULUM_SIZE_FLOOR >= 3000


def test_escalation_policy_defaults(monkeypatch):
    """Policy 2026-08-04: escalate after 15 non-improving evals, hard cap at 30 evals."""
    for var in (
        "SLM_STAGNATION_WINDOW",
        "SLM_STAGNATION_MIN_DELTA",
        "SLM_MAX_EVALS_BEFORE_ESCALATION",
    ):
        monkeypatch.delenv(var, raising=False)
    import agent.nodes.iterate as it
    importlib.reload(it)

    # ONE stagnation mechanism: 15 evals without a >2% gain.
    assert it.STAGNATION_WINDOW == 15
    assert it.STAGNATION_MIN_DELTA == 0.02
    # Unconditional ceiling.
    assert it.MAX_EVALS_BEFORE_ESCALATION == 30
    # The redundant consecutive-miss counter was removed (2026-08-05).
    assert not hasattr(it, "MAX_STALL_EVALS")
