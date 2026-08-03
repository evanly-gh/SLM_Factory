#!/usr/bin/env python
"""
SLM Factory — cold-start runner.

Runs the full agentic fine-tuning loop for a given task description.
The loop terminates when f(π) >= stop_threshold (calibrated by the planner
against published SOTA at the target model size) or the 1500-turn cold-start
budget is exhausted. No artificial step caps.

Usage:
    python run.py "fine tune a model for SMS spam detection on my Pixel 8 (8GB RAM)"
    python run.py --model "Qwen/Qwen3-1.7B@Q4_K_M" "your task description"

The task description should mention:
  - The target task (what the model should do)
  - The target device (name, RAM, storage) so hardware constraints can be resolved

Environment variables (all optional):
    SLM_FORCE_MODEL   exact variant selector (e.g. "Qwen/Qwen3-0.6B@Q4_K_M");
                      legacy bare IDs choose the lowest-peak-RAM sibling
"""
import argparse
import atexit
import datetime
import json
import os
import signal
import sys
import time
from pathlib import Path

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

# Enable cheap mode as EARLY as possible — before ANY `import config.config` or node import
# — so config.CHEAP_MODE (which makes Anthropic orchestration use Haiku) and the nodes'
# SLM_CHEAP checks all see it. The required local judge is unchanged. The full argparse
# below still declares --cheap for help/validation.
if "--cheap" in sys.argv:
    os.environ["SLM_CHEAP"] = "1"
# Keep this long-lived orchestrator free of model tensors. GPU-heavy operations run in
# disposable child processes whose exit guarantees complete CUDA-context cleanup.
os.environ.setdefault("SLM_CUDA_ISOLATION", "1")

# Parse before creating a run directory so --resume can reuse the original
# manifest, ledgers, SQLite thread, log, and artifacts.
parser = argparse.ArgumentParser(
    description="SLM Factory cold-start runner",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "description",
    nargs="?",
    default="",
    help="Natural-language task description (optional with --resume)",
)
parser.add_argument(
    "--model",
    default="",
    help="Force an exact model_id@bf16|Q8_0|Q4_K_M deployment selector",
)
parser.add_argument(
    "--cheap",
    action="store_true",
    help="Cheap mode: use Haiku for Anthropic orchestration and skip curate's "
    "hard-negative synthesis + CoT annotation (gold-only training). Keeps "
    "the local Qwen3.6 judge, agent intervention decisions, and Exa data "
    "research. Minimizes Claude spend on runs that may crash.",
)
parser.add_argument(
    "--resume",
    metavar="PATH",
    default="",
    help="Resume a checkpoint.json (or its containing run directory)",
)
args = parser.parse_args()

_configured_run_dir = os.environ.get("SLM_RUN_DIR", "").strip()
_resume_value = args.resume.strip() or os.environ.get("SLM_RESUME", "").strip()
if _resume_value.lower() in {"1", "true", "yes", "auto"}:
    if not _configured_run_dir:
        parser.error("SLM_RESUME=auto requires SLM_RUN_DIR")
    _resume_value = os.path.join(_configured_run_dir, "checkpoint.json")

_IS_RESUME = bool(_resume_value)
_resume_manifest = None
if _IS_RESUME:
    _resume_candidate = Path(_resume_value).expanduser().resolve()
    if _resume_candidate.is_dir():
        _resume_candidate = _resume_candidate / "checkpoint.json"
    CHECKPOINT_PATH = str(_resume_candidate)
    RUN_DIR = str(_resume_candidate.parent)
    _manifest_candidate = _resume_candidate.parent / "run-manifest.json"
    try:
        _resume_manifest = json.loads(
            _manifest_candidate.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"resume run manifest is missing or corrupt: {exc}")
    if not _resume_candidate.is_file():
        parser.error(f"resume checkpoint does not exist: {_resume_candidate}")
    TS = os.path.basename(RUN_DIR.rstrip(os.sep))
else:
    TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _RUN_UID = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    TS = f"{TS}_{_RUN_UID}"
    RUN_DIR = (
        os.path.abspath(os.path.expanduser(_configured_run_dir))
        if _configured_run_dir
        else os.path.join(PROJ, "logs", "runs", TS)
    )
    TS = os.path.basename(RUN_DIR.rstrip(os.sep))
    CHECKPOINT_PATH = os.path.join(RUN_DIR, "checkpoint.json")

description = args.description.strip()
if not description and _resume_manifest is not None:
    description = str(_resume_manifest.get("description") or "").strip()
if not description:
    parser.error("A task description is required for a fresh run.")

_requested_force_model = os.environ.get("SLM_FORCE_MODEL", "") or args.model
force_model = _requested_force_model
if not force_model and _resume_manifest is not None:
    force_model = str(_resume_manifest.get("force_model") or "")
if force_model:
    os.environ["SLM_FORCE_MODEL"] = force_model

# Quiet the ML-stack log flood BEFORE transformers/unsloth/datasets are imported anywhere
# (transformers reads TRANSFORMERS_VERBOSITY at import time). Keeps training-loss lines.
from agent.logging_setup import set_ml_env
set_ml_env()

from dotenv import load_dotenv
# override=True: .env is the source of truth for API keys. A SLURM job inherits the submitting
# shell's environment, so a stale/placeholder ANTHROPIC_API_KEY/EXA_API_KEY there would other-
# wise shadow the real .env key (load_dotenv defaults to override=False) → 401 auth errors.
# The SLURM run scripts never put API keys in the environment, so overriding is safe here.
load_dotenv(os.path.join(PROJ, ".env"), override=True)

# --------------------------------------------------------------------------
# Run directory + tee logger (set up before any imports that might print)
# --------------------------------------------------------------------------
from agent.checkpoint import prepare_fresh_run_directory

if _IS_RESUME:
    os.makedirs(os.path.join(RUN_DIR, "artifacts"), exist_ok=True)
else:
    prepare_fresh_run_directory(RUN_DIR)
os.environ["SLM_RUN_DIR"] = os.path.abspath(RUN_DIR)
os.environ["SLM_COST_EVENT_PATH"] = os.path.abspath(
    os.path.join(RUN_DIR, "cost-events.jsonl")
)
os.environ["SLM_TIMING_EVENT_PATH"] = os.path.abspath(
    os.path.join(RUN_DIR, "timing-events.jsonl")
)
CUR_LOG_PATH = os.path.abspath(
    os.path.join(RUN_DIR, "data-curation.md")
)
os.environ["SLM_CURATION_LOG_PATH"] = CUR_LOG_PATH
_LOGF = open(
    os.path.join(RUN_DIR, "run.log"),
    "a" if _IS_RESUME else "w",
    buffering=1,
)


# Substrings identifying high-noise library lines (Unsloth load banner, accelerate
# offload/kernel notices, remote-code prompts) that bury the useful pipeline logs. Any
# stdout line containing one of these is dropped by _Tee. The periodic training-loss
# lines and all [node] logs do NOT match and are preserved.
_NOISE_MARKERS = (
    "==((====))==",
    "\\   /|",
    "O^O/",
    '"-____-"',
    "🦥",
    "Free license: http://github.com/unslothai",
    "Unsloth: Fast downloading is enabled",
    "Unsloth: Will load",
    "Unsloth: Restored added_tokens_decoder",
    "Unsloth: Will smartly offload gradients",
    "Unsloth: Double buffering enabled",
    "Unsloth Zoo will now patch",
    "will patch your computer",
    "trust_remote_code` is True",
    "Are you certain you want to do remote code execution",
    "Detected kernel version",
    "Some parameters are on the meta device",
    "Unsloth: Padding-free",
    "Please restructure your imports with 'import unsloth'",
)


def _is_noise(line: str) -> bool:
    return any(m in line for m in _NOISE_MARKERS)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, d):
        # Drop library banner/offload noise line-by-line; keep everything else (loss lines,
        # node logs). Splitting on newlines handles multi-line writes without losing signal.
        if d and ("\n" in d or _is_noise(d)):
            kept = "".join(
                ln for ln in d.splitlines(keepends=True) if not _is_noise(ln)
            )
        else:
            kept = d
        if not kept:
            return
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.write(kept)
                s.flush()

    def flush(self):
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.flush()

    def isatty(self):
        return False

    def fileno(self):
        return sys.__stdout__.fileno() if sys.__stdout__ is not None else -1

    def __getattr__(self, name):
        return getattr(sys.__stdout__, name) if sys.__stdout__ is not None else None


sys.stdout = _Tee(sys.__stdout__, _LOGF)


def log(msg=""):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# --------------------------------------------------------------------------
# Process-safe observability — explicit provider wrappers in the call sites write
# to the shared paths above. Forked acquisition and spawned CUDA workers inherit
# those paths; no runner-only SDK monkey patches are needed.
# --------------------------------------------------------------------------
from agent.cost import LEDGER, install_cost_tracking
from agent.timing import (
    TIMINGS,
    TimingEvent,
    install_timing_tracking,
    record_timing_event,
)
from agent.observability import write_observability_artifacts

install_cost_tracking(os.environ["SLM_COST_EVENT_PATH"], required=True)
install_timing_tracking(os.environ["SLM_TIMING_EVENT_PATH"], required=True)

_RUN_STATUS = "initializing"
_RUN_EXIT_CODE = None
_RUN_REASON = None
_OBSERVABILITY_CLOSED = False
_OBSERVABILITY_RESULT = None


def _persist_observability():
    global _OBSERVABILITY_RESULT
    _OBSERVABILITY_RESULT = write_observability_artifacts(
        RUN_DIR,
        LEDGER,
        TIMINGS,
        run_status=_RUN_STATUS,
        exit_code=_RUN_EXIT_CODE,
        reason=_RUN_REASON,
    )
    return _OBSERVABILITY_RESULT


def _shutdown_observability():
    """Persist final artifacts, then restore stdout and close the tee exactly once."""
    global _OBSERVABILITY_CLOSED
    if _OBSERVABILITY_CLOSED:
        return _OBSERVABILITY_RESULT
    try:
        result = _persist_observability()
    except BaseException as exc:  # atexit must still restore stdout/log handles
        result = None
        if sys.__stderr__ is not None:
            sys.__stderr__.write(f"[observability] finalization failed: {exc}\n")
    finally:
        if isinstance(sys.stdout, _Tee):
            sys.stdout = sys.__stdout__
        if not _LOGF.closed:
            _LOGF.close()
        _OBSERVABILITY_CLOSED = True
    return result


atexit.register(_shutdown_observability)

# --------------------------------------------------------------------------
# Redirect node artifact paths to the per-run directory.
# Must happen before any node module caches ARTIFACTS_DIR at import time.
# --------------------------------------------------------------------------
import agent.nodes.curate as _curate_mod
import agent.nodes.train as _train_mod
import agent.nodes.downward_probe as _downward_probe_mod
import agent.nodes.cold_start.eval_setup as _eval_setup_mod

ART = os.path.join(RUN_DIR, "artifacts")
_curate_mod.ARTIFACTS_DIR = ART
_train_mod.ARTIFACTS_DIR = ART
_eval_setup_mod.ARTIFACTS_DIR = ART
_downward_probe_mod.ARTIFACTS_DIR = ART

# --------------------------------------------------------------------------
# Hardware research
# --------------------------------------------------------------------------
import config.config as config
from agent.checkpoint import (
    CHECKPOINT_FILENAME,
    RUN_MANIFEST_FILENAME,
    CheckpointError,
    CheckpointCompatibilityError,
    CheckpointCorruptError,
    atomic_write_json,
    atomic_write_text,
    checkpoint_compatibility,
    checkpoint_has_graph_progress,
    create_run_manifest,
    cumulative_wall_time_from_sqlite,
    inspect_sqlite_state,
    load_checkpoint,
    load_run_manifest,
    reconcile_checkpoint_from_sqlite,
    require_sqlite_authority,
    remaining_recursion_limit,
    resume_input_from_sqlite,
    runtime_config_snapshot,
    save_checkpoint,
    sqlite_checkpointer,
    stream_with_checkpoints,
)
from agent.graph import build_graph, graph_run_config
from agent.nodes.cold_start.hardware_research import research_device

_MODE = "cold_start"
_SEGMENT_START_TS = time.time()
_RUN_MANIFEST_PATH = os.path.join(RUN_DIR, RUN_MANIFEST_FILENAME)
_EFFECTIVE_CONFIG = runtime_config_snapshot(_MODE)
_COMPATIBILITY = checkpoint_compatibility(mode=_MODE)
if _IS_RESUME:
    _RUN_MANIFEST = load_run_manifest(
        _RUN_MANIFEST_PATH,
        expected_description=description,
        expected_force_model=force_model,
        expected_mode=_MODE,
        expected_compatibility=_COMPATIBILITY,
        expected_effective_config=_EFFECTIVE_CONFIG,
    )
    if os.path.abspath(CHECKPOINT_PATH) != os.path.abspath(
        _RUN_MANIFEST["checkpoint_path"]
    ):
        raise CheckpointError(
            "resume checkpoint path does not match the stable run manifest"
        )
    try:
        _RESTORED_CHECKPOINT = load_checkpoint(
            CHECKPOINT_PATH,
            expected_compatibility=_COMPATIBILITY,
            validate_artifact_paths=False,
        )
    except CheckpointCorruptError:
        if not os.path.isfile(_RUN_MANIFEST["sqlite_path"]):
            raise
        _RESTORED_CHECKPOINT = None
else:
    _RUN_MANIFEST = create_run_manifest(
        _RUN_MANIFEST_PATH,
        run_dir=RUN_DIR,
        description=description,
        force_model=force_model,
        mode=_MODE,
        compatibility=_COMPATIBILITY,
        effective_config=_EFFECTIVE_CONFIG,
    )
    _RESTORED_CHECKPOINT = save_checkpoint(
        CHECKPOINT_PATH,
        {"description": description, "_graph_steps": 0},
        thread_id=_RUN_MANIFEST["thread_id"],
        compatibility=_COMPATIBILITY,
        graph_steps=0,
        last_node=None,
        next_nodes=("__pregraph__",),
        cumulative_wall_time_s=0.0,
        status="initializing",
    )

THREAD_ID = _RUN_MANIFEST["thread_id"]
SQLITE_PATH = _RUN_MANIFEST["sqlite_path"]
if _IS_RESUME and _RESTORED_CHECKPOINT is not None:
    require_sqlite_authority(
        _RESTORED_CHECKPOINT,
        sqlite_path=SQLITE_PATH,
        expected_thread_id=THREAD_ID,
    )
_sqlite_startup_state = None
if _IS_RESUME and os.path.isfile(SQLITE_PATH):
    with sqlite_checkpointer(SQLITE_PATH) as _startup_saver:
        _startup_graph = build_graph(mode=_MODE, checkpointer=_startup_saver)
        _startup_config = graph_run_config(
            THREAD_ID,
            recursion_limit=config.MAX_TURNS_MAIN + 1,
        )
        _sqlite_startup_state = inspect_sqlite_state(
            _startup_graph,
            _startup_config,
        )
        if _sqlite_startup_state.exists:
            _previous_wall = cumulative_wall_time_from_sqlite(
                _RESTORED_CHECKPOINT or {"progress": {}},
                _sqlite_startup_state,
            )
            reconcile_checkpoint_from_sqlite(
                _startup_graph,
                config=_startup_config,
                checkpoint_path=CHECKPOINT_PATH,
                thread_id=THREAD_ID,
                compatibility=_COMPATIBILITY,
                cumulative_wall_time_s=_previous_wall,
            )
            _RESTORED_CHECKPOINT = load_checkpoint(
                CHECKPOINT_PATH,
                expected_compatibility=_COMPATIBILITY,
            )
if (
    _IS_RESUME
    and _RESTORED_CHECKPOINT is not None
    and checkpoint_has_graph_progress(_RESTORED_CHECKPOINT)
    and not (_sqlite_startup_state and _sqlite_startup_state.exists)
):
    raise CheckpointCompatibilityError(
        "checkpoint JSON records graph progress, but the expected SQLite "
        f"thread {THREAD_ID!r} has no authoritative state. Refusing to inject "
        "progressed JSON as fresh graph input."
    )
if _RESTORED_CHECKPOINT is None:
    raise CheckpointCorruptError(
        "JSON mirror is corrupt and SQLite contains no recoverable graph checkpoint"
    )
if _RESTORED_CHECKPOINT["thread_id"] != THREAD_ID:
    raise CheckpointError(
        "checkpoint thread_id does not match the stable run manifest"
    )
if not (_sqlite_startup_state and _sqlite_startup_state.exists):
    # Pre-graph JSON is authoritative only until the first SQLite generation.
    load_checkpoint(
        CHECKPOINT_PATH,
        expected_compatibility=_COMPATIBILITY,
    )
_GRAPH_ALREADY_TERMINAL = bool(
    _sqlite_startup_state and _sqlite_startup_state.terminal
)
_BASE_CUMULATIVE_WALL_S = float(
    ((_RESTORED_CHECKPOINT or {}).get("progress") or {}).get(
        "cumulative_wall_time_s", 0.0
    )
)
os.environ["SLM_RUN_ELAPSED_S"] = str(_BASE_CUMULATIVE_WALL_S)
os.environ["SLM_RUN_START_TS"] = str(_SEGMENT_START_TS)


class _SignalInterruption(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"received {signal.Signals(signum).name}")


def _request_checkpoint(signum, _frame):
    raise _SignalInterruption(signum)


_old_signal_handlers = {}
for _sig in (signal.SIGTERM, signal.SIGUSR1):
    _old_signal_handlers[_sig] = signal.getsignal(_sig)
    signal.signal(_sig, _request_checkpoint)


log(f"=== SLM Factory cold-start  (run {TS}) ===")
if _IS_RESUME:
    log(
        "RESUME: "
        f"last={_RESTORED_CHECKPOINT['progress'].get('last_node')} "
        f"next={_RESTORED_CHECKPOINT['progress'].get('next_nodes')} "
        f"steps={_RESTORED_CHECKPOINT['progress'].get('graph_steps')} "
        f"wall={_BASE_CUMULATIVE_WALL_S:.1f}s"
    )
log(f"task: {description}")
log(f"model selection strategy: {config.MODEL_SELECTION_STRATEGY}")
log(f"turn budget: {config.MAX_TURNS_MAIN}")
log(f"orchestrator model: {config.ORCHESTRATOR_MODEL}")
log(f"judge model: {config.JUDGE_MODEL}  |  judge endpoint: "
    f"{config.JUDGE_ENDPOINT or '<unset>'}  |  provider: local")
if config.CHEAP_MODE:
    log("CHEAP MODE ON: Haiku for Anthropic orchestration; local Qwen3.6 judge unchanged; "
        "hard-negative synthesis + CoT annotation skipped (gold-only). Agent intervention "
        "decisions + Exa data research KEPT.")
log("")
_PREGRAPH_RESTORED = bool(
    _IS_RESUME
    and _RESTORED_CHECKPOINT["state"].get("hardware_constraints") is not None
)
if _PREGRAPH_RESTORED:
    initial_state = _RESTORED_CHECKPOINT["state"]
    HW = initial_state.get("hardware_constraints")
    _device_path = os.path.join(RUN_DIR, "device_research.json")
    try:
        with open(_device_path, encoding="utf-8") as _device_file:
            hw_info = json.load(_device_file)
    except (OSError, json.JSONDecodeError):
        hw_info = {}
    record_timing_event(TimingEvent(
        kind="phase",
        name="checkpoint_restore",
        duration_ms=0.0,
        status="success",
        metadata={"pregraph_skipped": True},
    ))
    log("  ▶ checkpoint_restore (hardware research and graph prefix skipped)")
    log(f"      constraints: storage={HW.storage_mb}MB  memory={HW.memory_mb}MB  "
        f"latency_ttft={HW.latency_ttft_ms}ms  power={HW.power_watts}W  "
        f"chip={HW.target_chip}")
else:
    log("  ▶ hardware_research")
    _hw_t0 = time.time()
    _hw_cost_before = LEDGER.total_cost
    try:
        HW, hw_info = research_device(description, log=log)
    except BaseException as exc:
        _hw_dt = time.time() - _hw_t0
        record_timing_event(TimingEvent(
            kind="phase",
            name="hardware_research",
            duration_ms=_hw_dt * 1000,
            status="error",
            metadata={"error_type": type(exc).__name__, "error": str(exc)[:500]},
        ))
        _RUN_STATUS = "hardware_research_error"
        _RUN_REASON = f"{type(exc).__name__}: {exc}"
        _prior = ((_RESTORED_CHECKPOINT or {}).get("progress") or {})
        save_checkpoint(
            CHECKPOINT_PATH,
            {"description": description},
            thread_id=THREAD_ID,
            compatibility=_COMPATIBILITY,
            graph_steps=int(_prior.get("graph_steps", 0)),
            last_node=_prior.get("last_node"),
            next_nodes=("__pregraph__",),
            cumulative_wall_time_s=(
                _BASE_CUMULATIVE_WALL_S
                + max(0.0, time.time() - _SEGMENT_START_TS)
            ),
            status="hardware_research_error",
            error=_RUN_REASON,
        )
        raise
    _hw_dt = time.time() - _hw_t0
    _hw_dcost = LEDGER.total_cost - _hw_cost_before
    record_timing_event(TimingEvent(
        kind="phase",
        name="hardware_research",
        duration_ms=_hw_dt * 1000,
        status="success",
    ))

    log(f"      constraints: storage={HW.storage_mb}MB  memory={HW.memory_mb}MB  "
        f"latency_ttft={HW.latency_ttft_ms}ms  power={HW.power_watts}W  "
        f"chip={HW.target_chip}  [{_hw_dt:.1f}s  +${_hw_dcost:.4f}]")
    atomic_write_json(os.path.join(RUN_DIR, "device_research.json"), hw_info)
log("")

# --------------------------------------------------------------------------
# Initial state
# --------------------------------------------------------------------------
# Curated-benchmark pin: SLM_BENCHMARK_TASK selects one of the six deterministic loaders instead
# of the autonomous web_acquire path. eval_setup only consults that env on the NON-autonomous
# branch (task_plan is None), so we must start the run non-autonomous with the loader's task_type
# preset — otherwise task_analysis would plan a task and the env would be silently ignored.
_benchmark_task = (os.environ.get("SLM_BENCHMARK_TASK") or "").strip().lower()
_initial_autonomous = True
_initial_task_type = ""
if _benchmark_task:
    _bench_map = _eval_setup_mod.NAMED_BENCHMARK_TASK_TYPES
    if _benchmark_task not in _bench_map:
        raise SystemExit(
            f"SLM_BENCHMARK_TASK={_benchmark_task!r} is not a known benchmark; "
            f"choose one of {sorted(_bench_map)}"
        )
    _initial_task_type = _bench_map[_benchmark_task][0]
    _initial_autonomous = False
    os.environ["SLM_BENCHMARK_TASK"] = _benchmark_task  # normalized for eval_setup
    log(f"  benchmark: SLM_BENCHMARK_TASK={_benchmark_task} "
        f"(curated loader, task_type={_initial_task_type}, non-autonomous)")

fresh_initial_state = {
    "description": description,
    "target_metric": "F1",
    "hardware_constraints": HW,
    "task_type": _initial_task_type,
    "autonomous": _initial_autonomous,
    "task_plan": None,
    "selected_model": None,
    "feasible_models": [],
    # Both are overwritten by task_analysis._calibrate_stop_threshold before any training:
    # from the sourced registry, or parked unreachable and calibrated at the first evaluation.
    # DEFAULT_STOP_THRESHOLD survives only as the pre-graph placeholder.
    "stop_threshold": config.DEFAULT_STOP_THRESHOLD,
    "initial_stop_threshold": config.DEFAULT_STOP_THRESHOLD,
    "threshold_calibration": None,
    "train_examples": [],
    "eval_set": None,
    "data_source": None,
    "current_dataset_path": None,
    "dataset_version": 0,
    "data_rebuild_plan": None,
    "data_rebuild_plan_identity": None,
    "source_acquire_rounds_used": 0,
    "curation_log_path": CUR_LOG_PATH,
    "best_weights_ref": None,
    "best_score": 0.0,
    "lifetime_best_score": 0.0,
    "iteration": 0,
    "scores": [],
    "dag": [],
    "consecutive_no_improvement": 0,
    "downward_probe_done": False,
    "retained_gguf_paths": [],
    "last_eval": None,
    "last_curation": None,
    "last_intervention": "data_rebuild",
    "last_hypothesis": "",
    "llm_iterate_decision": None,
    "next_action": "train",
    "model_baselines": [],
    "quantize_enabled": False,
    "hw_gating_enabled": config.HW_GATING_ENABLED,
    "turn_budget": config.MAX_TURNS_MAIN,
    "_graph_steps": 0,
    "_wallclock_terminated_before": None,
    "_largest_first_phase": None,
    "escalation_history": [],
    "_pending_weights_refs": None,
    "_pending_training_outputs": None,
    "_pending_configs": None,
    # Redesign additions (B161+)
    "curriculum_size_target": config.CURRICULUM_SIZE_FLOOR,
    "eval_size_target": config.EVAL_SET_SIZE,
    "eval_source_ban": [],
    "data_sources": [],
    "data_source_usage": [],
    "eval_difficulty": None,
    "test_report": None,
    "downward_tiers_tried": [],
    "converged_model_ref": None,
    "downward_probe_history": {"origin": None, "attempts": []},
    "downward_probe_pending": None,
}
if not _PREGRAPH_RESTORED:
    initial_state = fresh_initial_state
else:
    restored_curation_log = initial_state.get("curation_log_path")
    if (
        restored_curation_log
        and os.path.abspath(restored_curation_log) != CUR_LOG_PATH
    ):
        raise CheckpointCompatibilityError(
            "resume curation_log_path drift: "
            f"stored={restored_curation_log!r}, expected={CUR_LOG_PATH!r}"
        )
    initial_state["curation_log_path"] = CUR_LOG_PATH

if force_model:
    os.environ["SLM_FORCE_MODEL"] = force_model
    log(f"  model override: {force_model}")

if not _PREGRAPH_RESTORED:
    save_checkpoint(
        CHECKPOINT_PATH,
        initial_state,
        thread_id=THREAD_ID,
        compatibility=_COMPATIBILITY,
        graph_steps=0,
        last_node=None,
        next_nodes=("task_analysis",),
        cumulative_wall_time_s=max(0.0, time.time() - _SEGMENT_START_TS),
        status="pregraph_complete",
    )

# --------------------------------------------------------------------------
# Compatibility shim (BUGS B111)
# --------------------------------------------------------------------------
# Several HF model repos ship custom modeling code (loaded via trust_remote_code)
# that still imports `is_torch_fx_available` from transformers.utils.import_utils —
# e.g. openbmb/MiniCPM4-0.5B's modeling_minicpm.py. transformers 5.5.0 removed that
# symbol, so the import crashes model load (ARC/MiniCPM) and the scaling-curve probe.
# Re-add it: torch.fx has existed since torch 1.8, so availability tracks torch itself.
import importlib.util as _ilu
import transformers.utils.import_utils as _tf_iu
if not hasattr(_tf_iu, "is_torch_fx_available"):
    def _is_torch_fx_available():
        return _ilu.find_spec("torch") is not None and _ilu.find_spec("torch.fx") is not None
    _tf_iu.is_torch_fx_available = _is_torch_fx_available
    # Some remote code imports it from the transformers top-level namespace too.
    import transformers as _tf
    if not hasattr(_tf, "is_torch_fx_available"):
        _tf.is_torch_fx_available = _is_torch_fx_available

# --------------------------------------------------------------------------
# Synthesis preflight (B161): synthesis is REQUIRED BY DEFAULT. Block here — BEFORE the
# curate/train/eval loop — until the local synth server is reachable, retrying up to
# SLM_SYNTH_WAIT_S. Do NOT progress the pipeline on a dead/absent endpoint; abort with a
# clear message instead (so a run can't silently go gold-only). Opt out with SLM_REQUIRE_SYNTH=0
# for a deliberate gold-only run.
# --------------------------------------------------------------------------
_synth_preflight_t0 = time.perf_counter()


def _exit_preflight(reason: str):
    global _RUN_STATUS, _RUN_EXIT_CODE, _RUN_REASON
    record_timing_event(TimingEvent(
        kind="phase",
        name="synth_preflight",
        duration_ms=(time.perf_counter() - _synth_preflight_t0) * 1000,
        status="error",
        metadata={"reason": reason},
    ))
    _RUN_STATUS = "preflight_error"
    _RUN_EXIT_CODE = 2
    _RUN_REASON = reason
    _progress = (
        ((_RESTORED_CHECKPOINT or {}).get("progress") or {})
        if _PREGRAPH_RESTORED
        else {}
    )
    save_checkpoint(
        CHECKPOINT_PATH,
        initial_state,
        thread_id=THREAD_ID,
        compatibility=_COMPATIBILITY,
        graph_steps=int(_progress.get("graph_steps", 0)),
        last_node=_progress.get("last_node"),
        next_nodes=tuple(_progress.get("next_nodes") or ("task_analysis",)),
        cumulative_wall_time_s=(
            _BASE_CUMULATIVE_WALL_S
            + max(0.0, time.time() - _SEGMENT_START_TS)
        ),
        status="preflight_error",
        error=reason,
    )
    raise SystemExit(2)


def _run_synth_preflight():
    if _GRAPH_ALREADY_TERMINAL:
        log("  ▶ synthesis preflight skipped: SQLite graph is already terminal")
        return
    if os.environ.get("SLM_REQUIRE_SYNTH", "1") == "0":
        return
    from data.synth_client import is_available as _synth_available
    endpoint = os.environ.get("SLM_SYNTH_ENDPOINT", "")
    if not endpoint:
        log("  !! SLM_REQUIRE_SYNTH=1 but SLM_SYNTH_ENDPOINT is unset — start scripts/serve_synth.slurm "
            "first (it writes logs/synth_endpoint.txt). Aborting.")
        _exit_preflight("SLM_REQUIRE_SYNTH=1 but SLM_SYNTH_ENDPOINT is unset")
    wait_s = float(os.environ.get("SLM_SYNTH_WAIT_S", "2400"))
    deadline = time.time() + wait_s
    attempt = 0
    log(f"  ▶ synthesis preflight: waiting for {endpoint} (up to {wait_s/60:.0f} min)")
    while True:
        attempt += 1
        if _synth_available(log=log):
            log(f"      synthesis server connected on attempt {attempt} — proceeding")
            return
        if time.time() >= deadline:
            log(f"  !! synthesis server {endpoint} never became reachable within {wait_s/60:.0f} min — "
                f"aborting (set SLM_REQUIRE_SYNTH=0 to allow gold-only).")
            _exit_preflight(
                f"synthesis server {endpoint} unavailable after {wait_s:.0f}s"
            )
        time.sleep(30)


try:
    _run_synth_preflight()
except (KeyboardInterrupt, _SignalInterruption) as exc:
    _exit_preflight(f"{type(exc).__name__}: {exc}")
record_timing_event(TimingEvent(
    kind="phase",
    name="synth_preflight",
    duration_ms=(time.perf_counter() - _synth_preflight_t0) * 1000,
    status="success",
    metadata={"required": os.environ.get("SLM_REQUIRE_SYNTH", "1") != "0"},
))

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
_RUN_STATUS = "running"

last_state = initial_state
t_start = time.time()
_prior_progress = ((_RESTORED_CHECKPOINT or {}).get("progress") or {})
GRAPH_STEPS = (
    _sqlite_startup_state.step
    if _sqlite_startup_state and _sqlite_startup_state.exists
    else int(initial_state.get("_graph_steps", 0) or 0)
)
_initial_graph_wall_s = (
    _BASE_CUMULATIVE_WALL_S
    + max(0.0, t_start - _SEGMENT_START_TS)
)
pipeline_error = None

try:
    if (
        _IS_RESUME
        and int(_prior_progress.get("graph_steps", 0)) > 0
        and not os.path.isfile(SQLITE_PATH)
    ):
        raise CheckpointError(
            "LangGraph SQLite checkpoint is missing; conditional routing cannot "
            "be reconstructed safely"
        )
    _remaining_steps = remaining_recursion_limit(
        GRAPH_STEPS,
        total=config.MAX_TURNS_MAIN,
    )
    _graph_config = graph_run_config(
        THREAD_ID,
        recursion_limit=_remaining_steps,
    )

    with sqlite_checkpointer(SQLITE_PATH) as _sqlite_saver:
        graph = build_graph(mode=_MODE, checkpointer=_sqlite_saver)
        _authoritative_before_stream = inspect_sqlite_state(
            graph,
            _graph_config,
        )
        if _authoritative_before_stream.exists:
            _authoritative_before_stream = reconcile_checkpoint_from_sqlite(
                graph,
                config=_graph_config,
                checkpoint_path=CHECKPOINT_PATH,
                thread_id=THREAD_ID,
                compatibility=_COMPATIBILITY,
                cumulative_wall_time_s=_initial_graph_wall_s,
            )
            last_state = _authoritative_before_stream.state
            GRAPH_STEPS = _authoritative_before_stream.step
        _graph_input = resume_input_from_sqlite(
            _authoritative_before_stream,
            initial_state,
        )

        # The topology is immutable for a compatible resume, but exporting it
        # again is harmless and keeps fresh-run behavior unchanged.
        try:
            mermaid_text = graph.get_graph().draw_mermaid()
            mermaid_path = os.path.join(RUN_DIR, "graph.mermaid")
            atomic_write_text(mermaid_path, mermaid_text)
            log(f"  Graph structure saved: {mermaid_path}")
        except Exception as e:
            log(f"  Could not export Mermaid graph: {e}")

        _segment_result = stream_with_checkpoints(
            graph,
            _graph_input,
            config=_graph_config,
            checkpoint_path=CHECKPOINT_PATH,
            thread_id=THREAD_ID,
            compatibility=_COMPATIBILITY,
            initial_graph_steps=GRAPH_STEPS,
            initial_wall_time_s=_initial_graph_wall_s,
            max_graph_steps=config.MAX_TURNS_MAIN,
        )
        last_state = _segment_result.state
        GRAPH_STEPS = _segment_result.graph_steps
except BaseException as exc:
    import traceback
    if isinstance(exc, KeyboardInterrupt):
        pipeline_error = KeyboardInterrupt("interrupted by user")
        log("\n  interrupted by user")
    else:
        pipeline_error = exc
        log(f"\n  !! exception in {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
    # stream_with_checkpoints may have completed additional nodes before the
    # interruption. Reload its atomic snapshot so failure summaries never
    # regress to this segment's stale input state.
    try:
        _failure_checkpoint = load_checkpoint(
            CHECKPOINT_PATH,
            expected_compatibility=_COMPATIBILITY,
        )
        last_state = _failure_checkpoint["state"]
        GRAPH_STEPS = int(
            _failure_checkpoint["progress"].get("graph_steps", GRAPH_STEPS)
        )
    except CheckpointError as _checkpoint_exc:
        log(f"  !! could not reload failure checkpoint: {_checkpoint_exc}")
    record_timing_event(TimingEvent(
        kind="graph_stream",
        name="stream",
        duration_ms=(time.time() - t_start) * 1000,
        status="error",
        metadata={"error_type": type(exc).__name__, "error": str(exc)[:500]},
    ))
finally:
    for _sig, _handler in _old_signal_handlers.items():
        signal.signal(_sig, _handler)

_segment_elapsed = max(0.0, time.time() - _SEGMENT_START_TS)
elapsed = _BASE_CUMULATIVE_WALL_S + _segment_elapsed
record_timing_event(TimingEvent(
    kind="run",
    name="graph_pipeline",
    duration_ms=(time.time() - t_start) * 1000,
    status="error" if pipeline_error else "success",
    metadata={
        "graph_steps": GRAPH_STEPS,
        "cumulative_wall_time_s": elapsed,
        "resumed": _IS_RESUME,
    },
))

# --------------------------------------------------------------------------
# Post-convergence on-device hardware verification (opt-in via SLM_HW_VERIFY_ON_DEVICE=1)
# --------------------------------------------------------------------------
# Measures the FINAL selected model on real hardware, writes hardware_eval.json,
# re-checks the four constraints against MEASURED values, and records the deployed
# model reference. Fully guarded: any failure logs and continues.
hardware_eval_report = None
_hw_verify_t0 = time.perf_counter()
if config.HW_VERIFY_ON_DEVICE:
    log("")
    log("  ▶ on-device hardware verification")
    try:
        final_model = last_state.get("selected_model")
        best_ref = last_state.get("best_weights_ref")
        if final_model is None or not best_ref:
            log("      [hw-verify] no converged model/weights — skipping")
        elif final_model.quant is None:
            log("      [hw-verify] final model is BF16 (no GGUF path) — skipping "
                "SmolChat/GGUF verification; set a quantized variant to enable")
        else:
            from training.cuda_isolation import merge_and_quantize
            from hardware_eval.on_device_eval import run_on_device_eval, result_to_dict
            from config.android_pool import check_hardware_constraints, all_constraints_pass

            mid_safe = final_model.model_id.replace("/", "_")
            gguf = merge_and_quantize(
                best_ref,
                os.path.join(ART, "merged", mid_safe, "final_verify"),
                os.path.join(ART, "gguf", mid_safe, "final_verify"),
                final_model.quant,
            )

            hw_result = run_on_device_eval(
                final_model, HW, gguf_path=gguf,
                backend=config.HW_ONDEVICE_BACKEND, log=log)
            hw_check = check_hardware_constraints(
                final_model, HW, measured=hw_result.to_measured())
            passed = all_constraints_pass(hw_check)

            hardware_eval_report = {
                "backend": config.HW_ONDEVICE_BACKEND,
                "model_id": final_model.model_id,
                "quant": final_model.quant,
                "gguf_path": gguf,
                "weights_ref": best_ref,
                "measured_success": hw_result.success,
                "measured_error": hw_result.error,
                "result": result_to_dict(hw_result),
                "constraint_check": hw_check,
                "all_constraints_pass": passed,
            }
            atomic_write_json(
                os.path.join(RUN_DIR, "hardware_eval.json"),
                hardware_eval_report,
            )

            if hw_result.success:
                log(f"      [hw-verify] {hw_result.eval_method}: "
                    f"TTFT={hw_result.ttft_ms}ms  tok/s={hw_result.tok_per_s}  "
                    f"peakRSS={hw_result.peak_memory_mb}MB  power={hw_result.avg_watts}W  "
                    f"→ constraints {'PASS' if passed else 'FAIL'}")
                # The verified artifact is recorded in hardware_eval.json above
                # (weights_ref + constraint_check + all_constraints_pass). The former
                # `deployed_model_ref` state field was production-mode-only and write-only;
                # it was removed with production mode on 2026-07-29.
            else:
                log(f"      [hw-verify] measurement failed: {hw_result.error}")
    except Exception as exc:
        import traceback
        log(f"      [hw-verify] skipped due to error: {exc}")
        log(traceback.format_exc())
record_timing_event(TimingEvent(
    kind="phase",
    name="on_device_verification",
    duration_ms=(time.perf_counter() - _hw_verify_t0) * 1000,
    status="success",
    metadata={"enabled": bool(config.HW_VERIFY_ON_DEVICE)},
))

# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
_RUN_STATUS = "pipeline_error" if pipeline_error else "finalizing"
_RUN_REASON = (
    f"{type(pipeline_error).__name__}: {pipeline_error}"
    if pipeline_error is not None
    else None
)
cost, timings = _persist_observability()

atomic_write_json(
    os.path.join(RUN_DIR, "dag.json"),
    last_state.get("dag", []),
)

_history_origin = (
    (last_state.get("downward_probe_history") or {}).get("origin") or {}
)
_selected_for_artifact = last_state.get("selected_model")
atomic_write_json(
    os.path.join(RUN_DIR, "scores.json"),
    {
        "scores": last_state.get("scores", []),
        "trajectory_selector": (
            _history_origin.get("selector")
            or (
                _selected_for_artifact.selector
                if _selected_for_artifact is not None
                else None
            )
        ),
        "final_selector": (
            _selected_for_artifact.selector
            if _selected_for_artifact is not None
            else None
        ),
        "best_score": last_state.get("best_score", 0.0),
        "stop_threshold": last_state.get("stop_threshold", config.DEFAULT_STOP_THRESHOLD),
        "iterations": last_state.get("iteration", 0),
        "converged": last_state.get("best_score", 0.0) >= last_state.get("stop_threshold", 1.0),
    },
)

atomic_write_json(
    os.path.join(RUN_DIR, "downward_probe_history.json"),
    last_state.get("downward_probe_history") or {},
)

if last_state.get("task_plan"):
    atomic_write_json(
        os.path.join(RUN_DIR, "task_plan.json"),
        last_state["task_plan"],
    )

# --------------------------------------------------------------------------
# Data-source provenance across the whole run (links + row counts). Aggregated
# from the per-build usage log, seeded with the initial eval_setup lineage so the
# first curriculum's sources are represented even though its rows predate tagging.
# --------------------------------------------------------------------------
from data.provenance import aggregate_data_sources, format_run_data_sources

_data_sources_agg = aggregate_data_sources(
    last_state.get("data_source_usage") or [],
    base_records=last_state.get("data_sources") or [],
)
atomic_write_json(
    os.path.join(RUN_DIR, "data_sources.json"),
    _data_sources_agg,
)

# --------------------------------------------------------------------------
# DAG summary (what was tried and how well it worked)
# --------------------------------------------------------------------------
dag = last_state.get("dag", [])
dag_lines = ["# DAG Summary — Training Attempts", ""]
dag_lines.append(f"{'Iter':>4}  {'Model':<30}  {'Score':>7}  {'Pruned':>6}  {'Config':<25}  {'Intervention'}")
dag_lines.append("-" * 100)
for node in dag:
    pruned_str = "✗" if node.get("pruned") else ""
    dag_lines.append(
        f"{node.get('iteration', '?'):>4}  "
        f"{node.get('model_id', '?'):<30}  "
        f"{node.get('score', 0):.4f}  "
        f"{pruned_str:>6}  "
        f"{node.get('best_config', '?'):<25}  "
        f"{node.get('intervention', '?')}"
    )
dag_lines.append("")
dag_text = "\n".join(dag_lines)
atomic_write_text(os.path.join(RUN_DIR, "dag_summary.txt"), dag_text)

# --------------------------------------------------------------------------
# Baseline vs fine-tuned summary
# --------------------------------------------------------------------------
baselines = last_state.get("model_baselines", [])
# Update the final model's best_finetuned_f1 with the current best score
if baselines:
    final_model_id = last_state.get("selected_model")
    if final_model_id:
        final_selector = final_model_id.selector
        for entry in baselines:
            if entry.get("selector", entry.get("model_id")) == final_selector:
                entry["best_finetuned_f1"] = max(
                    entry.get("best_finetuned_f1", 0.0),
                    last_state.get("best_score", 0.0),
                )

atomic_write_json(os.path.join(RUN_DIR, "baselines.json"), baselines)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
m = last_state.get("selected_model")
best = last_state.get("best_score", 0.0)
threshold = last_state.get("stop_threshold", config.DEFAULT_STOP_THRESHOLD)
converged = best >= threshold
# Name the metric explicitly. The comparison scalar is carried in a field called `f1`, but
# only classification and NER compute an F1 — math is exact match, code is an execution
# pass-rate, and open generation is a judge mean. Printing the real name keeps a run summary
# from being quoted as an F1 result for a task that never measured one.
from eval.harness import TASK_METRIC_NAMES

_metric_name = getattr(
    last_state.get("last_eval"),
    "metric",
    None,
) or TASK_METRIC_NAMES.get(last_state.get("task_type", ""), "f1")
log(f"score metric: {_metric_name} (carried in the EvalResult.f1 field)")
from agent.pipeline_status import (
    build_run_progression,
    format_downward_probe_history,
    outcome_text,
    process_exit_code,
    run_heading,
)
model_label = (
    m.selector
    if m is not None else "none"
)
progression = build_run_progression(last_state, baselines)
_final_progression = [
    entry for entry in progression
    if m is not None and entry.get("selector") == m.selector
]
_final_entry = _final_progression[-1] if _final_progression else {}
_final_scores = _final_entry.get("scores", [])

log("")
log(f"{'='*70}")
log(f"  {run_heading(pipeline_error)} — {elapsed:.1f}s")
log(f"{'='*70}")
log(f"  task_type : {last_state.get('task_type')}")
log(f"  model     : {m.selector if m else None}")
log(f"  iterations: {_final_entry.get('iterations', 0)}")
log(f"  best F1   : {best:.4f}  (threshold {threshold:.4f})  "
    f"{'✗ failed' if pipeline_error else ('✓ converged' if converged else '✗ budget exhausted')}")
lifetime_best = last_state.get("lifetime_best_score", 0.0) or 0.0
log(f"  lifetime best F1 across all tiers: {max(lifetime_best, best):.4f}")
log(f"  final-model trajectory: {[f'{x:.3f}' for x in _final_scores]}")

# --------------------------------------------------------------------------
# Convergence-speed metrics (for comparing model-selection strategies): how many steps
# the whole process took to settle on the final model. "Steps" = graph node executions;
# also report total train→eval iterations across ALL tiers and how many models were tried.
# --------------------------------------------------------------------------
_total_iters = sum(int(entry.get("iterations", 0)) for entry in progression)
_models_tried = len(progression)
log("")
log(f"  ── Convergence-speed metrics (strategy={config.MODEL_SELECTION_STRATEGY}) ──")
log(f"  total pipeline steps (graph node executions): {GRAPH_STEPS}")
log(f"  total train→eval iterations (all tiers)     : {_total_iters}")
log(f"  models tried                                : {_models_tried}")
log(f"  outcome                                     : "
    f"{outcome_text(converged, pipeline_error, model_label)}")

# --------------------------------------------------------------------------
# Full run progression across EVERY model/tier (not just the final one).
# build_run_progression also separates post-convergence downward attempts from the
# original model trajectory, preventing adopted-model relabeling.
# --------------------------------------------------------------------------
if len(progression) > 1:
    log("")
    log(f"  Full run progression ({len(progression)} ordered model attempts):")
    log(f"  {'Tier':>4}  {'Model':<32} {'Quant':<8} {'Iters':>5} {'Best':>7}  Trajectory")
    log(f"  {'-'*92}")
    for p in progression:
        traj = " → ".join(f"{x:.3f}" for x in p.get("scores", [])) or "(reset)"
        best_value = p.get("best_score")
        best_text = (
            f"{best_value:.4f}" if best_value is not None else "n/a"
        )
        log(f"  {str(p.get('tier','?')):>4}  {p.get('model_id','?'):<32} "
            f"{str(p.get('quant') or 'bf16'):<8} {p.get('iterations',0):>5} "
            f"{best_text:>7}  {traj}")

_downward_lines = format_downward_probe_history(
    last_state.get("downward_probe_history")
)
if _downward_lines:
    log("")
    for _line in _downward_lines:
        log(_line)

# Model Improvement Report — one row per VARIANT actually run (model_id + quant), so the
# on-device deployment format is explicit and a model that appears at multiple tiers as
# different quants is not collapsed into one row (B161 reporting request).
if progression:
    log("")
    log(f"  Model Improvement Report (per quantized variant):")
    log(f"  {'Tier':>4}  {'Model':<32} {'Quant':<8} {'Baseline':>9} {'Best FT':>9} {'Δ':>9}")
    log(f"  {'-'*80}")
    for p in progression:
        bl = p.get("baseline_f1")
        ft = p.get("best_score")
        bl_text = f"{bl:.4f}" if bl is not None else "n/a"
        ft_text = f"{ft:.4f}" if ft is not None else "n/a"
        delta_text = (
            f"{ft - bl:+.4f}"
            if bl is not None and ft is not None
            else "n/a"
        )
        log(f"  {str(p.get('tier','?')):>4}  {p.get('model_id','?'):<32} "
            f"{str(p.get('quant') or 'bf16'):<8} {bl_text:>9} "
            f"{ft_text:>9} {delta_text:>9}")

# DAG Traversal for EVERY model tested (not just the final one). Each escalation reset the
# DAG, so escalate_node stashed each model's per-iteration DAG in escalation_history; here
# we print all of them in order (B161 reporting request).
if progression:
    log("")
    log(f"  DAG Traversal (all {len(progression)} models):")
    for p in progression:
        pdag = p.get("dag") or []
        log("")
        log(f"  ── Tier {p.get('tier','?')}: {p.get('model_id','?')} "
            f"[{p.get('quant') or 'bf16'}]  ({len(pdag)} iterations) ──")
        if not pdag:
            log(f"     (no completed iterations recorded)")
            continue
        log(f"     {'Iter':>4}  {'Score':>7}  {'Pruned':>6}  {'Config':<25}  {'Intervention'}")
        for node in pdag:
            pruned_str = "✗" if node.get("pruned") else ""
            log(f"     {node.get('iteration','?'):>4}  {node.get('score',0):.4f}  "
                f"{pruned_str:>6}  {str(node.get('best_config','?')):<25}  "
                f"{node.get('intervention','?')}")

log("")
_anthropic_cost = cost["by_provider"].get("anthropic", {})
_exa_cost = cost["by_provider"].get("exa", {})
log(
    f"  cost: Claude {_anthropic_cost.get('calls', 0)} calls "
    f"({_anthropic_cost.get('input_tokens', 0)}→{_anthropic_cost.get('output_tokens', 0)} tok) "
    f"${_anthropic_cost.get('estimated_usd', 0.0):.4f}  |  "
    f"Exa {_exa_cost.get('calls', 0)} searches ${_exa_cost.get('estimated_usd', 0.0):.4f}  |  "
    f"total ${cost['total_cost_usd']:.4f}"
)
for _provider, _summary in cost["by_provider"].items():
    log(
        f"    cost provider={_provider} calls={_summary['calls']} "
        f"failures={_summary['failures']} latency={_summary['latency_ms'] / 1000:.1f}s "
        f"usd=${_summary['estimated_usd']:.6f}"
    )
log("")
log(format_run_data_sources(_data_sources_agg))
log(f"  (full provenance: {os.path.join(RUN_DIR, 'data_sources.json')})")

# Post-run summary graphics. Fail-safe: the run has already succeeded by this point, so a
# plotting error (or a missing matplotlib) must never change the outcome — log and move on.
try:
    from agent.run_graphics import generate_run_graphics

    _graphics = generate_run_graphics(
        RUN_DIR,
        state=last_state,
        baselines=baselines,
        stop_threshold=threshold,
    )
    if _graphics:
        log(f"  graphics: {os.path.dirname(str(_graphics[0]))} ({len(_graphics)} files)")
except Exception as _graphics_error:  # noqa: BLE001 — never let reporting break a finished run
    log(f"  graphics: skipped ({type(_graphics_error).__name__}: {_graphics_error})")

log(f"  logs: {RUN_DIR}")

_RUN_EXIT_CODE = process_exit_code(pipeline_error)
_RUN_STATUS = "pipeline_error" if pipeline_error else "completed"
atexit.unregister(_shutdown_observability)
_shutdown_observability()
raise SystemExit(_RUN_EXIT_CODE)
