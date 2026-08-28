import sys
import os
import types

import pytest

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

# config.config hard-requires these at import, so the test process needs SOMETHING. It gets an inert
# placeholder, and it gets it whether or not a real key is present.
#
# This used to be `setdefault`, reasoning that "real keys (cluster/CI) take precedence". That is
# backwards for a test suite. Every paid call is supposed to be patched, but a test that forgets to
# patch one, or patches the wrong name, then spends real money and reaches the real network instead of
# failing — and only on the machine that HAS credentials, which is where it is least likely to be
# noticed. Overriding makes an unpatched call fail fast with an auth error, which is the outcome that
# gets fixed. Nothing under tests/ needs a working key.
os.environ["ANTHROPIC_API_KEY"] = "test-placeholder-not-a-real-key"
os.environ["EXA_API_KEY"] = "test-placeholder-not-a-real-key"

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

# --------------------------------------------------------------------------
# Heavyweight test modules, skipped unless asked for
# --------------------------------------------------------------------------
# Each of these files imports something large off /mmfs1 — `transformers`, `torch` in a subprocess,
# `matplotlib`, the dataset loaders — and this is a network filesystem, so a cold read costs tens of
# seconds while the same import is instant once the page cache is warm. Seven files were ~280s of a
# 320s suite; the remaining ~1,400 tests finish in about 40s between them.
#
# They are skipped at COLLECTION, not by a marker. A `@pytest.mark.slow` does not help: deselection
# happens after collection and collection imports the module, so the import is paid regardless.
# Measured — marking the single 96s test in test_train_eval_prefix_alignment.py moved 86s onto the next
# test in the same file, which had been running in 1.06s.
#
# Nothing here is badly written or testing the wrong thing, so they are skipped rather than deleted:
# they cover B290's train/serve template skew, the CUDA-worker traceback contract, crash-safe
# checkpoint resume, and the real dataset loaders. Run them with:
#
#     SLM_TEST_HEAVY=1 pytest tests/            # everything
#     pytest tests/training/test_cuda_isolation.py --heavy-ok    # one file
#
# A file named here directly on the command line still runs, so `pytest <that file>` behaves as
# expected and only a whole-suite sweep skips it.
_HEAVY_TEST_FILES = {
    "test_train_eval_prefix_alignment.py",   # ~87s: reads a cached HuggingFace tokenizer
    "test_local_judge.py",                   # ~40s: the judge's model plumbing and thread pool
    "test_cuda_isolation.py",                # ~40s: imports torch in a fresh subprocess
    "test_web_acquire_datasets.py",          # ~27s: real dataset loaders over NFS
    "test_cumulative_curriculum.py",         # ~27s: full curate passes including quality control
    "test_checkpoint_kill_resume_smoke.py",  # ~25s: spawns and kills real subprocesses
    "test_sqlite_authority.py",              # ~45s: same, several times over
    "test_run_graphics.py",                  # ~13s: imports matplotlib and renders every chart
    "test_synth_client_integration.py",      # ~32s: imports the openai client and serves real HTTP
}


def pytest_addoption(parser):
    parser.addoption(
        "--heavy-ok", action="store_true", default=False,
        help="collect the heavyweight test modules that a whole-suite run skips by default",
    )


def pytest_ignore_collect(collection_path, config):
    """Skip the heavyweight modules on a whole-suite run.

    Returning None rather than False for everything else leaves other plugins' opinions intact.
    """
    if os.environ.get("SLM_TEST_HEAVY") or config.getoption("--heavy-ok"):
        return None
    if collection_path.name not in _HEAVY_TEST_FILES:
        return None
    # Named explicitly on the command line, so the caller clearly wants it.
    for argument in config.invocation_params.args:
        if collection_path.name in str(argument):
            return None
    return True


@pytest.fixture(autouse=True)
def _clear_synthesis_rejection_state():
    """Reset the module-level synthesis rejection state between tests.

    `data.curriculum` keeps two module-level lists — the rejection REASONS fed back into the next
    generation prompt, and a bounded sample of the rejected ROWS for the audit archive. Both are
    correct in production, where a process is one run and the sample is cleared when curate takes it.
    In a test process they are shared mutable state: a synthesis test that produces rejections leaves
    them sitting there, and the next test to archive rows picks up the previous test's rejects and
    fails on a count it never created. That is a real failure mode of module-level state and worth
    neutralising here rather than in each test that happens to trip it.
    """
    from data import curriculum

    curriculum._RECENT_REJECTED_ROWS.clear()
    curriculum._RECENT_REJECTIONS.clear()
    curriculum._GENERATION_FAILURES.clear()
    yield
    curriculum._RECENT_REJECTED_ROWS.clear()
    curriculum._RECENT_REJECTIONS.clear()
    curriculum._GENERATION_FAILURES.clear()
