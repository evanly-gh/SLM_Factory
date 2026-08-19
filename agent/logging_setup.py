# agent/logging_setup.py
"""
Central noise control for the ML stack (transformers / unsloth / datasets / tqdm).

Why this exists
---------------
A single training/eval run emits thousands of lines of library chatter that bury the
signal we actually care about (per-step training loss, node decisions, scores):

  - "Accessing `is_flash_linear_attention_available` from `.models.<X>.image_processing_<X>`"
    Hundreds of transformers *deprecation-alias* warnings, one per image-processor module,
    triggered when Unsloth Zoo scans/patches every model module at import. Harmless.
  - "Loading weights: 0%|..." — a transformers shard-loading progress bar, once per model load.
  - "Unsloth: Tokenizing ["text"] (num_proc=25): ..." — a `datasets.map` progress bar.
  - "Both `max_new_tokens` (=256) and `max_length`(=40960) seem to have been set ..." — emitted
    once per generate() call (i.e. once per eval example).
  - assorted FutureWarning / DeprecationWarning (attention-mask API, etc.).

What is preserved
-----------------
The useful periodic training loss lines, e.g.
    {'loss': '1.499', 'grad_norm': '0.7413', 'learning_rate': '7.059e-05', 'epoch': '2.105'}
These come from the HF Trainer's PrinterCallback (active when `disable_tqdm=True`), which
prints via plain `print()` and is NOT gated by the transformers logging verbosity. So
silencing the loggers/progress bars removes the noise without hiding the loss.
"""
import contextlib
import os
import sys
import warnings

_ENV_APPLIED = False
_LIBS_QUIETED = False


def set_ml_env() -> None:
    """Set env vars that must exist BEFORE transformers/datasets/hf_hub are imported.

    transformers reads TRANSFORMERS_VERBOSITY at import time to pick the initial log
    level, so setting it here (from the top of the entrypoint, before any heavy import)
    is what actually suppresses the import-time `is_flash_linear_attention_available`
    alias spam emitted while Unsloth patches every model module.

    Idempotent and non-destructive: uses setdefault so an explicit override still wins.
    """
    global _ENV_APPLIED
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not _ENV_APPLIED:
        # These fire during import of transformers submodules; register them before the
        # first heavy import so the import-time deprecation aliases are filtered too.
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        _ENV_APPLIED = True


def quiet_ml_logging() -> None:
    """Silence library log + progress-bar noise. Idempotent.

    Call once after transformers/unsloth are importable (e.g. right after
    `from unsloth import FastLanguageModel`) so the library-level toggles
    (`disable_progress_bar`, `disable_progress_bars`) take effect for the current run.
    """
    global _LIBS_QUIETED
    set_ml_env()
    if _LIBS_QUIETED:
        return
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()  # kills the "Loading weights" bar
    except Exception:
        pass
    try:
        import datasets
        datasets.disable_progress_bars()   # kills the "Tokenizing [...]" bar
        try:
            datasets.utils.logging.set_verbosity_error()
        except Exception:
            pass
    except Exception:
        pass
    # accelerate emits the "Detected kernel version ... below the recommended minimum"
    # warning and offload notices — informational only; silence at ERROR.
    try:
        import logging
        for name in ("accelerate", "accelerate.utils.other", "accelerate.big_modeling"):
            logging.getLogger(name).setLevel(logging.ERROR)
    except Exception:
        pass
    _LIBS_QUIETED = True


@contextlib.contextmanager
def quiet_output():
    """Discard everything the enclosed block writes to stdout/stderr, at the fd level.

    `DATASETS_VERBOSITY=error` cannot hide the `datasets` loader noise, because the lines that
    matter are themselves logged at ERROR ("Failed to load JSON from file ... ArrowInvalid" is
    emitted once per shard of a malformed repo) and the download/split progress bars are raw
    tqdm writes rather than HF progress-bar API calls. Redirecting at the file-descriptor level
    is the only interception point that covers Python logging, tqdm and pyarrow's own C-level
    writes at once.

    Callers must already be capturing whatever they need to report: the acquisition path
    accumulates its own verdict lines in a list and replays them afterwards, so nothing of value
    goes through the descriptors being discarded here.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (ValueError, OSError):
        pass
    try:
        saved = (os.dup(1), os.dup(2))
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    except OSError:
        # No real descriptors to swap (embedded/pytest capture). Fall back to the Python-level
        # redirect, which still covers logging and tqdm's default stream.
        if saved is not None:
            for fd in saved:
                os.close(fd)
        saved = None
        os.close(devnull)
        with open(os.devnull, "w") as sink, \
                contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
        return
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (ValueError, OSError):
            pass
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)
