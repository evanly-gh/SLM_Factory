import sys
import os

# Ensure the project root is first on sys.path so that source packages
# (training, eval, agent, config, etc.) are importable from all test
# subdirectories. pyproject.toml also sets pythonpath=["."] for the same
# reason; this conftest guards against edge cases where that is insufficient.
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# No test may block on a real network wait. `curate` blocks for SLM_SYNTH_MIDRUN_WAIT_S
# (default 600s) waiting for the local synthesis endpoint to return before it stops the run;
# any test that simulates an unreachable endpoint would otherwise sit there for ten minutes.
# Setting the wait to 0 keeps the FAIL-vs-degrade decision under test while removing the sleep.
os.environ.setdefault("SLM_SYNTH_MIDRUN_WAIT_S", "0")
