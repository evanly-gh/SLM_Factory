import sys
import os

# Ensure the project root is first on sys.path so that source packages
# (training, eval, agent, config, etc.) are importable from all test
# subdirectories. pyproject.toml also sets pythonpath=["."] for the same
# reason; this conftest guards against edge cases where that is insufficient.
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
