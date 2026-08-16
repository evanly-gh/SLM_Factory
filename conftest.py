import sys
import os
import types

# Ensure the project root is first on sys.path so that source packages
# (training, eval, agent, config, etc.) are importable from all test
# subdirectories. pyproject.toml also sets pythonpath=["."] for the same
# reason; this conftest guards against edge cases where that is insufficient.
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- Windows dev-box compatibility (tests only) ---
# The pipeline runs on a Linux/SLURM cluster; several modules import `fcntl`
# (Unix-only file locking) at module load. On Windows there is no fcntl, so
# provide a no-op stub for the pytest process only. Locks become no-ops, which
# is safe for single-process unit tests. Never imported in production (cluster
# is Linux and has the real module).
if sys.platform == "win32" and "fcntl" not in sys.modules:
    try:
        import fcntl  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("fcntl")
        _stub.LOCK_EX = 2
        _stub.LOCK_SH = 1
        _stub.LOCK_UN = 8
        _stub.LOCK_NB = 4
        _stub.flock = lambda *args, **kwargs: None
        _stub.lockf = lambda *args, **kwargs: None
        _stub.fcntl = lambda *args, **kwargs: 0
        _stub.ioctl = lambda *args, **kwargs: 0
        sys.modules["fcntl"] = _stub

# config.config hard-requires these at import. On a dev box without real
# credentials, provide inert placeholders so pure-logic tests can import the
# package. Real keys (cluster/CI) take precedence via setdefault.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-placeholder")
os.environ.setdefault("EXA_API_KEY", "test-placeholder")

# No test may block on a real network wait. `curate` blocks for SLM_SYNTH_MIDRUN_WAIT_S
# (default 600s) waiting for the local synthesis endpoint to return before it stops the run;
# any test that simulates an unreachable endpoint would otherwise sit there for ten minutes.
# Setting the wait to 0 keeps the FAIL-vs-degrade decision under test while removing the sleep.
os.environ.setdefault("SLM_SYNTH_MIDRUN_WAIT_S", "0")

# Stretch goals are OFF by default under pytest. Every convergence path now asks the orchestrator
# whether to raise the accuracy goal, which is a live Anthropic call; the many existing tests that
# assert terminal routing on a threshold-clearing score would each make one. Tests that exercise
# raising turn it on explicitly and patch the call.
os.environ.setdefault("SLM_THRESHOLD_RAISE", "0")
