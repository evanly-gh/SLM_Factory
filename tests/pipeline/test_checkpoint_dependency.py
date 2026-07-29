import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_sqlite_checkpointer_dependency_is_locked_consistently():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert any(dep.startswith("langgraph-checkpoint-sqlite") for dep in dependencies)
    assert any(
        dep.startswith("langgraph-checkpoint-sqlite") for dep in requirements
    )
    assert 'name = "langgraph-checkpoint-sqlite"' in lock
