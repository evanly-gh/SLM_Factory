"""Make a run reproducible: same code, same data, same numbers.

WHY THIS EXISTS. Seeding alone was not enough and the gap was measured, not assumed. Probe job
40222458 trained the same 151-row file twice with identical arguments IN ONE PROCESS and produced
two different adapters (sha256 ``933f037f…`` against ``f6c9ba31…``), with per-step losses differing
in the third significant digit. LoRA init and data order were already deterministic — Unsloth's
``get_peft_model`` seeds itself and HuggingFace's ``Trainer`` seeds the sampler — but nothing asked
cuBLAS or cuDNN for a fixed reduction order, so bf16 accumulation followed whatever order the
scheduler happened to pick. On a 57-step fine-tune over 151 rows that is enough to move a score by
0.015, which is the same size as the per-tier effects the ablations are trying to measure.

FOUR LAYERS, AND ONLY THREE ARE FIXABLE HERE.

1. Kernel numerics — this module: deterministic algorithms, no cuDNN autotuning, fixed cuBLAS
   workspace.
2. Library RNG — this module seeds ``random``/``numpy``/``torch``; callers additionally pass the
   seed explicitly to the trainer and the adapter builder rather than relying on their defaults.
3. Teacher sampling — ``data/synth_client.py`` draws generation seeds from :func:`seed_sequence`,
   so synthetic rows stay diverse from one another while being identical across reruns.
4. The orchestrator's own decisions — NOT fixable. Claude is a remote service with no seed
   parameter, so two runs can choose different interventions from identical state. A run is
   reproducible in the sense that a fixed sequence of decisions produces fixed numbers; making the
   sequence itself fixed needs decision replay, which is a separate mechanism.

``CUBLAS_WORKSPACE_CONFIG`` must be set BEFORE the first cuBLAS handle is created, which is why
``tests/pipeline/_l40s_task_body.sh`` and ``training/cuda_worker.py`` both set it before importing
torch, and why :func:`enable_determinism` reports it as ineffective if CUDA is already up.
"""
from __future__ import annotations

import os
import random

SEED_ENV = "SLM_SEED"
MODE_ENV = "SLM_DETERMINISM"
CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
# `math` pins scaled-dot-product attention to the backend that does not accumulate with atomics.
# Set to anything else to leave PyTorch's backend selection alone.
SDPA_ENV = "SLM_DETERMINISM_SDPA"

# 3407 is Unsloth's own documented default for `get_peft_model(random_state=...)`. Adopting it as
# the project seed means turning this module on does not silently change the LoRA initialisation
# that every run before it used.
DEFAULT_SEED = 3407

# `:4096:8` is the setting PyTorch documents for deterministic cuBLAS GEMMs; it costs a little
# workspace memory and nothing else.
CUBLAS_DETERMINISTIC = ":4096:8"

_VALID_MODES = ("strict", "warn", "off")


def configured_seed() -> int:
    """The run's seed. Recorded in the resume fingerprint, so it cannot change under a resume."""
    raw = str(os.environ.get(SEED_ENV, "")).strip()
    if not raw:
        return DEFAULT_SEED
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{SEED_ENV}={raw!r} is not an integer. Refusing to guess a seed for a run whose "
            "whole purpose is being reproducible."
        ) from exc


def configured_mode() -> str:
    """``strict`` raises on a nondeterministic op, ``warn`` logs it, ``off`` disables the module.

    ``warn`` is the default because a raise lands mid-run: a 30-hour job should not die at hour 20
    on an op that has no deterministic implementation. The warnings appear in the run log, so an
    op that cannot be pinned is visible rather than silent.
    """
    mode = str(os.environ.get(MODE_ENV, "warn")).strip().lower() or "warn"
    if mode not in _VALID_MODES:
        raise ValueError(
            f"{MODE_ENV}={mode!r} is not one of {_VALID_MODES}. Refused at startup rather than "
            "leaving a run to discover it later."
        )
    return mode


def set_cublas_workspace_config() -> None:
    """Set the cuBLAS workspace env var. Safe to call repeatedly; never overrides an operator."""
    os.environ.setdefault(CUBLAS_ENV, CUBLAS_DETERMINISTIC)


def seed_everything(seed: int | None = None) -> int:
    """Seed ``random``, ``numpy`` and ``torch`` (including CUDA). Returns the seed used."""
    seed = configured_seed() if seed is None else int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def enable_determinism(*, log=print, label: str = "") -> dict:
    """Pin every source of run-to-run variation this process controls.

    Returns a record of what was applied, which callers log so a run's own output states the
    conditions its numbers were produced under.
    """
    mode = configured_mode()
    record = {"mode": mode, "seed": None, "cublas": os.environ.get(CUBLAS_ENV), "notes": []}
    if mode == "off":
        log(f"[determinism]{' ' + label if label else ''} DISABLED by {MODE_ENV}=off — "
            "this run's numbers are not reproducible")
        return record

    set_cublas_workspace_config()
    record["cublas"] = os.environ.get(CUBLAS_ENV)
    record["seed"] = seed_everything()

    try:
        import torch
    except Exception as exc:  # a CPU-only caller (report rebuild, tests) has nothing to pin
        record["notes"].append(f"torch unavailable: {exc}")
        return record

    # A cuBLAS handle created before the env var was read ignores it, and silently. Say so.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        record["notes"].append(
            "CUDA was already initialised when determinism was enabled, so "
            f"{CUBLAS_ENV} may not have taken effect in this process"
        )

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # autotuning picks kernels by timing, so by load
    except Exception as exc:
        record["notes"].append(f"cudnn flags unavailable: {exc}")

    try:
        torch.use_deterministic_algorithms(True, warn_only=(mode == "warn"))
    except Exception as exc:
        record["notes"].append(f"use_deterministic_algorithms failed: {exc}")

    # `use_deterministic_algorithms` governs PyTorch's OWN ops and nothing else. Probe 40253558
    # applied it, the seeds and the cuBLAS workspace, produced not one nondeterminism warning, and
    # still returned two different adapters — because the attention kernel is xformers' and the
    # fused paths are Unsloth's, neither of which that flag reaches. Flash and memory-efficient
    # attention accumulate their backward pass with atomics, so their reduction order follows the
    # scheduler; the math backend does not.
    record["sdpa"] = os.environ.get(SDPA_ENV, "math")
    if record["sdpa"] == "math":
        for name, enable in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        ):
            fn = getattr(torch.backends.cuda, name, None)
            if fn is None:
                record["notes"].append(f"torch.backends.cuda.{name} unavailable")
                continue
            try:
                fn(enable)
            except Exception as exc:
                record["notes"].append(f"{name}({enable}) failed: {exc}")

    record["torch"] = getattr(torch, "__version__", "?")
    log(f"[determinism]{' ' + label if label else ''} seed={record['seed']} mode={mode} "
        f"{CUBLAS_ENV}={record['cublas']} cudnn.deterministic=True cudnn.benchmark=False "
        f"sdpa={record.get('sdpa')} compile_disable={os.environ.get('UNSLOTH_COMPILE_DISABLE', '<unset>')} "
        f"torch={record.get('torch')}")
    for note in record["notes"]:
        log(f"[determinism]   NOTE: {note}")
    return record


def seed_sequence(namespace: str) -> "_SeedSequence":
    """A reproducible stream of per-request seeds for a sampling caller.

    A single fixed seed on every generation request would make the teacher return the SAME row for
    the same prompt, which destroys the diversity synthesis exists to provide. A counter keyed off
    the run seed keeps the rows different from each other and identical across reruns.
    """
    return _SeedSequence(configured_seed(), namespace)


class _SeedSequence:
    __slots__ = ("_base", "_namespace", "_n")

    def __init__(self, base: int, namespace: str) -> None:
        # Fold the namespace in so two callers sharing a run seed do not draw the same stream.
        # crc32 and not hash(): str hashing is salted per interpreter unless PYTHONHASHSEED was
        # set before the process started, so hash() would hand out a different stream per run —
        # the exact failure this module exists to remove.
        import zlib

        salt = zlib.crc32(namespace.encode("utf-8")) % 100_000
        self._base = (base + salt) % (2**31 - 1)
        self._namespace = namespace
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return (self._base + self._n) % (2**31 - 1)

    @property
    def drawn(self) -> int:
        return self._n
