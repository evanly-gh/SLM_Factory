import importlib


def test_curriculum_floor_is_at_least_3000(monkeypatch):
    # Requirement: the lowest target dataset size is 3000 for any task, so the
    # per-task floor must be >= 3000. (The repo default may be set higher, e.g. 5000.)
    monkeypatch.delenv("SLM_CURRICULUM_FLOOR", raising=False)
    import config.config as cfg
    importlib.reload(cfg)
    assert cfg.CURRICULUM_SIZE_FLOOR >= 3000


def test_max_stall_evals_default_is_20(monkeypatch):
    monkeypatch.delenv("SLM_MAX_STALL_EVALS", raising=False)
    import agent.nodes.iterate as it
    importlib.reload(it)
    assert it.MAX_STALL_EVALS == 20
