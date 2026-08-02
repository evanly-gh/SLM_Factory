"""Crash-safe state checkpoints and LangGraph resume helpers."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent.state_codec import StateCodecError, decode_state, encode_state


CHECKPOINT_SCHEMA_VERSION = 3
CHECKPOINT_FILENAME = "checkpoint.json"
SQLITE_FILENAME = "langgraph.sqlite"
RUN_MANIFEST_FILENAME = "run-manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 3
ARTIFACT_MANIFEST_FILENAME = ".slm_artifact_manifest.json"
ARTIFACT_COMPLETE_FILENAME = ".slm_complete"
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base class for durable-checkpoint failures."""


class CheckpointCorruptError(CheckpointError):
    """Checkpoint bytes or schema are invalid."""


class CheckpointCompatibilityError(CheckpointError):
    """Checkpoint was created by an incompatible pipeline definition."""


class CheckpointArtifactError(CheckpointError):
    """A durable artifact referenced by state is missing or has drifted."""


class RecursionBudgetExhausted(CheckpointError):
    """The cumulative graph-step budget has already been consumed."""


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_pool_fingerprint(pool=None) -> str:
    """Fingerprint exact selectors and all runtime-relevant ModelSpec metadata."""
    if pool is None:
        from config.android_pool import ANDROID_POOL

        pool = ANDROID_POOL
    values = sorted((asdict(model) for model in pool), key=lambda row: (
        row["model_id"],
        row.get("quant") or "bf16",
    ))
    return _canonical_hash(values)


def graph_topology_fingerprint(mode: str) -> str:
    from agent.graph import graph_topology_descriptor

    return _canonical_hash(graph_topology_descriptor(mode))


def runtime_config_snapshot(mode: str) -> dict[str, Any]:
    """Return resume-sensitive settings while excluding secrets and host endpoints."""
    import config.config as config
    from agent.data_rebuild import (
        DATA_REBUILD_SCHEMA_VERSION,
        MAX_PAID_ACQUIRE_ROUNDS_PER_RUN,
    )
    from data.acquisition_budget import (
        ACQUISITION_LEDGER_SCHEMA_VERSION,
    )
    from training.hparams import (
        LORA_SEARCH_SPACE_VERSION,
        MAX_EFFECTIVE_BATCH_SIZE,
    )
    from training.lora_trainer import SFT_LOSS_CONTRACT_VERSION

    names = (
        "MODEL_SELECTION_STRATEGY",
        "MAX_TURNS_MAIN",
        "DEFAULT_STOP_THRESHOLD",
        "CURRICULUM_SIZE_FLOOR",
        "EVAL_SET_SIZE",
        "DATA_SIZE_CEILING",
        "MAX_WALLCLOCK_S",
        "QUANT_ACCURACY_EVAL",
        "HW_GATING_ENABLED",
        "HW_VERIFY_ON_DEVICE",
        "HW_ONDEVICE_BACKEND",
        "ORCHESTRATOR_MODEL",
        "TEACHER_MODEL_CLAUDE",
        "CHEAP_MODE",
        "ORCHESTRATOR_1M",
        "ANTHROPIC_BETAS",
        "SYNTH_MODEL",
        "JUDGE_MODEL",
        "JUDGE_ALLOW_REMOTE",
        "JUDGE_CACHE_PATH",
        "JUDGE_CONCURRENCY",
        "JUDGE_REQUEST_TIMEOUT_S",
        "LOCAL_DATASET_DIR",
        "HW_BATTERY_VOLTAGE_V",
    )
    snapshot = {
        "mode": mode,
        # A search-contract change must fail resume compatibility explicitly;
        # individual selected values remain in state/DAG and round-trip as JSON.
        "LORA_SEARCH_SPACE_VERSION": LORA_SEARCH_SPACE_VERSION,
        "LORA_MAX_EFFECTIVE_BATCH_SIZE": MAX_EFFECTIVE_BATCH_SIZE,
        "SFT_LOSS_CONTRACT_VERSION": SFT_LOSS_CONTRACT_VERSION,
        "DATA_REBUILD_SCHEMA_VERSION": DATA_REBUILD_SCHEMA_VERSION,
        "DATA_REBUILD_MAX_PAID_ROUNDS_PER_RUN": (
            MAX_PAID_ACQUIRE_ROUNDS_PER_RUN
        ),
        "ACQUISITION_BUDGET_SCHEMA_VERSION": (
            ACQUISITION_LEDGER_SCHEMA_VERSION
        ),
    }
    for name in names:
        if hasattr(config, name):
            snapshot[name] = getattr(config, name)
    snapshot["SLM_FORCE_MODEL"] = os.environ.get("SLM_FORCE_MODEL", "")
    snapshot["SLM_SHARED_DATASET_DIR"] = os.environ.get(
        "SLM_SHARED_DATASET_DIR", ""
    )
    env_defaults = {
        "SLM_STAGNATION_WINDOW": "50",
        "SLM_STAGNATION_MIN_DELTA": "0.02",
        "SLM_MAX_STALL_EVALS": "50",
        "SLM_MAX_SEQ_LENGTH": "4096",
        "SLM_EVAL_BATCH_SIZE": "",
        "SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION": "50",
        "SLM_EVAL_MAX_NEW_TOKENS_NER": "512",
        "SLM_EVAL_MAX_NEW_TOKENS_MATH": "512",
        "SLM_EVAL_MAX_NEW_TOKENS_GENERATION": "512",
        "SLM_EVAL_MAX_NEW_TOKENS_APPS": "1024",
        "SLM_CODE_EVAL_TIMEOUT_S": "3.0",
        "SLM_APPS_PROBLEM_TIMEOUT_S": "6.0",
        "SLM_CURATION_LOG_PATH": "",
        "SLM_AGENT_FIRST_DATASET_DISCOVERY": "0",
        "SLM_REQUIRE_SYNTH": "1",
        "SLM_SYNTH_WAIT_S": "2400",
        # GPU placement/profile throughput settings are intentionally omitted:
        # a resumed run may move between compatible GPU allocations.
        "SLM_CUDA_ISOLATION": "0",
        "SLM_EARLY_STOPPING": "1",
        "SLM_VAL_FRACTION": "0.12",
        "SLM_MIN_FOR_VAL": "60",
        "SLM_EVAL_STEPS": "20",
        "SLM_EARLY_STOP_PATIENCE": "3",
        "SLM_GGUF_GPU_LAYERS": "-1",
        "SLM_DIFFICULTY": "zeroshot",
        "SLM_PROBE_EPOCHS": "3",
        "SLM_PROBE_MAX_EXAMPLES": "300",
        "SLM_CURRICULUM_SIZE": "",
        "SLM_STOP_THRESHOLD": "",
        "SLM_DATA_DIR": "data_cache",
        "SLM_SMOLCHAT_PACKAGE": "io.shubham0204.smollmandroid",
    }
    snapshot.update({
        name: os.environ.get(name, default)
        for name, default in env_defaults.items()
    })
    return snapshot


def runtime_config_fingerprint(mode: str) -> str:
    return _canonical_hash(runtime_config_snapshot(mode))


def checkpoint_compatibility(
    *,
    mode: str,
    pool_fingerprint: str | None = None,
    topology_fingerprint: str | None = None,
    config_fingerprint: str | None = None,
) -> dict[str, str]:
    return {
        "mode": mode,
        "pool_fingerprint": pool_fingerprint or model_pool_fingerprint(),
        "topology_fingerprint": (
            topology_fingerprint or graph_topology_fingerprint(mode)
        ),
        "config_fingerprint": (
            config_fingerprint or runtime_config_fingerprint(mode)
        ),
    }


def stable_thread_id(run_dir: str | os.PathLike) -> str:
    """Derive a stable, non-secret LangGraph thread key from the run directory."""
    resolved = str(Path(run_dir).expanduser().resolve())
    return f"slm-{hashlib.sha256(resolved.encode()).hexdigest()[:24]}"


def _utc_now() -> str:
    import datetime

    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def create_run_manifest(
    path: str | os.PathLike,
    *,
    run_dir: str | os.PathLike,
    description: str,
    force_model: str,
    mode: str,
    compatibility: Mapping[str, str],
    effective_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise CheckpointCompatibilityError(
            f"refusing to overwrite existing run manifest: {destination}"
        )
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "run_dir": str(resolved_run_dir),
        "thread_id": stable_thread_id(resolved_run_dir),
        "mode": mode,
        "description": description,
        "force_model": force_model,
        "compatibility": dict(compatibility),
        "effective_config": dict(
            effective_config
            if effective_config is not None
            else runtime_config_snapshot(mode)
        ),
        "checkpoint_path": str(resolved_run_dir / CHECKPOINT_FILENAME),
        "sqlite_path": str(resolved_run_dir / SQLITE_FILENAME),
    }
    atomic_write_json(destination, payload)
    return payload


def load_run_manifest(
    path: str | os.PathLike,
    *,
    expected_description: str | None = None,
    expected_force_model: str | None = None,
    expected_mode: str = "cold_start",
    expected_compatibility: Mapping[str, Any] | None = None,
    expected_effective_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptError(
            f"run manifest is unreadable or corrupt at {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointCorruptError("run manifest root must be an object")
    if payload.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "run manifest schema is incompatible"
        )
    required = (
        "run_dir",
        "thread_id",
        "mode",
        "description",
        "compatibility",
        "effective_config",
        "checkpoint_path",
        "sqlite_path",
    )
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise CheckpointCorruptError(
            f"run manifest is missing required fields: {', '.join(missing)}"
        )
    if payload["mode"] != expected_mode:
        raise CheckpointCompatibilityError(
            f"run manifest mode drift: {payload['mode']!r} != {expected_mode!r}"
        )
    if (
        expected_description is not None
        and payload["description"] != expected_description
    ):
        raise CheckpointCompatibilityError(
            "run manifest description drift: "
            f"{payload['description']!r} != {expected_description!r}"
        )
    if (
        expected_force_model is not None
        and payload.get("force_model", "") != expected_force_model
    ):
        raise CheckpointCompatibilityError(
            "run manifest force_model drift: "
            f"{payload.get('force_model', '')!r} != {expected_force_model!r}"
        )
    expected_thread = stable_thread_id(payload["run_dir"])
    if payload["thread_id"] != expected_thread:
        raise CheckpointCompatibilityError(
            "run manifest thread_id does not match its stable run directory"
        )
    if not isinstance(payload["compatibility"], Mapping):
        raise CheckpointCorruptError("run manifest compatibility must be an object")
    if not isinstance(payload["effective_config"], Mapping):
        raise CheckpointCorruptError(
            "run manifest effective_config must be an object"
        )
    # Structural durability probes intentionally omit expected_effective_config:
    # they must not import API-key-requiring runtime config merely to decide
    # whether checkpoint bytes are complete. The runner loads .env, constructs
    # the current snapshot, and passes it here for strict resume drift checks.
    if expected_effective_config is not None:
        current_config = dict(expected_effective_config)
        drift = [
            (
                key,
                payload["effective_config"].get(key),
                current_config.get(key),
            )
            for key in sorted(
                set(payload["effective_config"]) | set(current_config)
            )
            if payload["effective_config"].get(key) != current_config.get(key)
        ]
        if drift:
            details = "; ".join(
                f"{key}: stored={stored!r}, expected={expected!r}"
                for key, stored, expected in drift
            )
            raise CheckpointCompatibilityError(
                f"run manifest effective config drift: {details}"
            )
    if expected_compatibility is not None:
        _validate_compatibility(
            payload["compatibility"],
            expected_compatibility,
        )
    return payload


def checkpoint_has_graph_progress(
    checkpoint: Mapping[str, Any],
) -> bool:
    """True once JSON records any graph generation or completed node."""
    progress = checkpoint.get("progress") or {}
    state = checkpoint.get("state") or {}
    try:
        graph_steps = int(progress.get("graph_steps", 0) or 0)
        state_steps = int(state.get("_graph_steps", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise CheckpointCorruptError(
            f"checkpoint graph progress is invalid: {exc}"
        ) from exc
    if graph_steps > 0 or state_steps > 0 or progress.get("last_node"):
        return True
    if (
        progress.get("sqlite_checkpoint_id")
        or progress.get("sqlite_generation")
        or progress.get("sqlite_step") is not None
        or progress.get("pending_tasks")
        or progress.get("pending_weights_refs")
        or progress.get("pending_configs")
    ):
        return True
    try:
        dataset_version = int(state.get("dataset_version", 0) or 0)
        iteration = int(state.get("iteration", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise CheckpointCorruptError(
            f"checkpoint state progress is invalid: {exc}"
        ) from exc
    if (
        state.get("task_type")
        or state.get("selected_model") is not None
        or state.get("eval_set") is not None
        or state.get("current_dataset_path")
        or dataset_version > 0
        or iteration > 0
        or state.get("scores")
        or state.get("dag")
        or state.get("_pending_weights_refs")
        or state.get("_pending_configs")
    ):
        return True
    next_nodes = set(progress.get("next_nodes") or ())
    pregraph_nodes = {"__pregraph__", "task_analysis"}
    return bool(next_nodes - pregraph_nodes)


def _sqlite_threads(path: Path) -> list[str]:
    import sqlite3

    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
        )
        try:
            rows = connection.execute(
                "SELECT DISTINCT thread_id FROM checkpoints "
                "ORDER BY thread_id"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        raise CheckpointCorruptError(
            f"LangGraph SQLite checkpoint is unreadable at {path}: {exc}"
        ) from exc
    return [str(row[0]) for row in rows]


def require_sqlite_authority(
    checkpoint: Mapping[str, Any],
    *,
    sqlite_path: str | os.PathLike,
    expected_thread_id: str,
) -> None:
    """Fail closed when progressed JSON lacks matching SQLite authority."""
    if not checkpoint_has_graph_progress(checkpoint):
        return
    path = Path(sqlite_path).expanduser().resolve()
    refusal = (
        " Refusing to inject progressed JSON as fresh graph input; "
        "restore the expected langgraph.sqlite or start a new run directory."
    )
    if not path.is_file():
        raise CheckpointCompatibilityError(
            f"checkpoint JSON records graph progress but SQLite database is "
            f"missing at {path}.{refusal}"
        )
    try:
        with sqlite_checkpointer(path) as saver:
            checkpoint_tuple = saver.get_tuple({
                "configurable": {
                    "thread_id": expected_thread_id,
                    "checkpoint_ns": "",
                }
            })
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointCorruptError(
            f"could not read authoritative SQLite state at {path}: {exc}"
        ) from exc
    if checkpoint_tuple is not None:
        return
    threads = _sqlite_threads(path)
    if not threads:
        raise CheckpointCompatibilityError(
            f"checkpoint JSON records graph progress but SQLite database "
            f"has no checkpoints at {path}.{refusal}"
        )
    raise CheckpointCompatibilityError(
        f"checkpoint JSON records graph progress for expected thread "
        f"{expected_thread_id!r}, but SQLite has no authoritative state for "
        f"that expected thread; found SQLite threads: {', '.join(threads)}."
        f"{refusal}"
    )


def durable_resume_available(run_dir: str | os.PathLike) -> bool:
    """True only when manifest/checkpoint parse and identify the same run."""
    root = Path(run_dir).expanduser()
    try:
        manifest = load_run_manifest(root / RUN_MANIFEST_FILENAME)
        checkpoint = _read_checkpoint(root / CHECKPOINT_FILENAME)
        if checkpoint["thread_id"] != manifest["thread_id"]:
            return False
        _validate_compatibility(
            checkpoint["compatibility"],
            manifest["compatibility"],
        )
        require_sqlite_authority(
            checkpoint,
            sqlite_path=manifest["sqlite_path"],
            expected_thread_id=manifest["thread_id"],
        )
    except (CheckpointError, OSError, TypeError, KeyError):
        return False
    return True


def prepare_fresh_run_directory(run_dir: str | os.PathLike) -> Path:
    """Create a fresh run directory or repair a known partial initialization.

    Unknown operator-owned content is never removed.
    """
    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        (root / "artifacts").mkdir(parents=True)
        return root
    if not root.is_dir():
        raise CheckpointCompatibilityError(
            f"run path exists but is not a directory: {root}"
        )
    if durable_resume_available(root):
        raise CheckpointCompatibilityError(
            f"durable run already exists at {root}; resume it instead"
        )
    manifest_path = root / RUN_MANIFEST_FILENAME
    checkpoint_path = root / CHECKPOINT_FILENAME
    if manifest_path.is_file() and checkpoint_path.is_file():
        try:
            manifest = load_run_manifest(manifest_path)
            checkpoint = _read_checkpoint(checkpoint_path)
        except CheckpointCorruptError:
            # Structurally incomplete startup files are repairable below.
            pass
        else:
            if checkpoint["thread_id"] != manifest["thread_id"]:
                raise CheckpointCompatibilityError(
                    "run manifest/checkpoint thread mismatch; refusing to "
                    f"clear partial run directory {root}"
                )
            _validate_compatibility(
                checkpoint["compatibility"],
                manifest["compatibility"],
            )
            # A valid progressed checkpoint with broken SQLite is not a fresh
            # partial initialization and must never be cleared/reseeded.
            require_sqlite_authority(
                checkpoint,
                sqlite_path=manifest["sqlite_path"],
                expected_thread_id=manifest["thread_id"],
            )
            raise CheckpointCompatibilityError(
                f"durable run already exists at {root}; resume it instead"
            )
    known_root_names = {
        "artifacts",
        "run.log",
        "cost-events.jsonl",
        "acquisition-reservations.jsonl",
        "timing-events.jsonl",
        "cost.json",
        "timings.json",
        RUN_MANIFEST_FILENAME,
        CHECKPOINT_FILENAME,
        SQLITE_FILENAME,
        f"{SQLITE_FILENAME}-wal",
        f"{SQLITE_FILENAME}-shm",
        "device_research.json",
    }
    unknown = [
        child.name for child in root.iterdir()
        if child.name not in known_root_names
    ]
    artifacts = root / "artifacts"
    if artifacts.is_dir():
        unsafe_artifacts = [
            str(child.relative_to(root))
            for child in artifacts.rglob("*")
            if child.is_file()
            and ".partial-" not in child.name
            and ".tmp." not in child.name
        ]
        unknown.extend(unsafe_artifacts)
    if unknown:
        raise CheckpointCompatibilityError(
            "refusing to repair partial run directory with unknown content: "
            + ", ".join(sorted(unknown))
        )
    shutil.rmtree(root)
    (root / "artifacts").mkdir(parents=True)
    return root


_SQLITE_TYPE_KEY = "__slm_checkpoint_type__"


def _sqlite_encode(value: Any) -> Any:
    """Replace application runtime objects before LangGraph serializes a blob."""
    from config.android_pool import HardwareConstraints, ModelSpec
    from data.eval_set import EvalSet
    from eval.harness import EvalResult
    from training.lora_trainer import TrainingOutput

    if isinstance(value, ModelSpec):
        return {_SQLITE_TYPE_KEY: "model_selector", "selector": value.selector}
    if isinstance(value, HardwareConstraints):
        return {_SQLITE_TYPE_KEY: "hardware_constraints", "value": asdict(value)}
    if isinstance(value, EvalSet):
        return {
            _SQLITE_TYPE_KEY: "eval_set",
            "value": {
                "pos": _sqlite_encode(value.pos),
                "neg": _sqlite_encode(value.neg),
                "boundary": _sqlite_encode(value.boundary),
                "task_type": value.task_type,
                "multi_label": value.multi_label,
                "schema": _sqlite_encode(value.schema),
                "multilingual": value.multilingual,
            },
        }
    if isinstance(value, EvalResult):
        return {
            _SQLITE_TYPE_KEY: "eval_result",
            "value": _sqlite_encode(asdict(value)),
        }
    if isinstance(value, TrainingOutput):
        return {
            _SQLITE_TYPE_KEY: "training_output",
            "weights_ref": value.weights_ref,
            "gguf_path": value.gguf_path,
        }
    if value is None or isinstance(value, (str, bool, int, float, bytes)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {
            key: _sqlite_encode(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sqlite_encode(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sqlite_encode(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return {
            _SQLITE_TYPE_KEY: "set",
            "frozen": isinstance(value, frozenset),
            "items": [_sqlite_encode(item) for item in value],
        }
    # LangGraph's serializer owns its internal task/channel classes. Passing
    # those through retains native resume semantics while application objects
    # remain strictly allow-listed above.
    module = type(value).__module__
    if module.startswith(("langgraph.", "langchain_core.")):
        return value
    raise TypeError(
        f"checkpoint contains a runtime object of type {type(value).__name__}"
    )


def _sqlite_decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_sqlite_decode(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sqlite_decode(item) for item in value)
    if not isinstance(value, Mapping):
        return value
    kind = value.get(_SQLITE_TYPE_KEY)
    if kind == "model_selector":
        from config.android_pool import ANDROID_POOL

        selector = value.get("selector")
        matches = [model for model in ANDROID_POOL if model.selector == selector]
        if len(matches) != 1:
            raise CheckpointCompatibilityError(
                f"SQLite checkpoint exact selector is incompatible: {selector!r}"
            )
        return matches[0]
    if kind == "hardware_constraints":
        from config.android_pool import HardwareConstraints

        return HardwareConstraints(**_sqlite_decode(value["value"]))
    if kind == "eval_set":
        from data.eval_set import EvalSet

        return EvalSet(**_sqlite_decode(value["value"]))
    if kind == "eval_result":
        from eval.harness import EvalResult

        return EvalResult(**_sqlite_decode(value["value"]))
    if kind == "training_output":
        from training.lora_trainer import TrainingOutput

        return TrainingOutput(
            weights_ref=value.get("weights_ref"),
            gguf_path=value.get("gguf_path"),
        )
    if kind == "set":
        items = [_sqlite_decode(item) for item in value.get("items", ())]
        return frozenset(items) if value.get("frozen") else set(items)
    return {key: _sqlite_decode(item) for key, item in value.items()}


class SafeCheckpointSerializer:
    """LangGraph serializer that never pickles application runtime objects."""

    def __init__(self):
        try:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        except ImportError as exc:  # pragma: no cover - base LangGraph is required
            raise RuntimeError("LangGraph checkpoint serialization is unavailable") from exc
        self._serializer = JsonPlusSerializer(pickle_fallback=False)

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return self._serializer.dumps_typed(_sqlite_encode(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return _sqlite_decode(self._serializer.loads_typed(data))


@contextmanager
def sqlite_checkpointer(path: str | os.PathLike):
    """Open the optional SQLite LangGraph saver with the safe state serializer."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "SQLite resume requires langgraph-checkpoint-sqlite; "
            "install the project's checkpoint dependency"
        ) from exc

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(destination)) as saver:
        saver.serde = SafeCheckpointSerializer()
        saver._slm_after_put = None
        saver._slm_after_writes = None
        original_put = saver.put
        original_put_writes = saver.put_writes

        def hooked_put(config, checkpoint, metadata, new_versions):
            saved_config = original_put(
                config,
                checkpoint,
                metadata,
                new_versions,
            )
            callback = saver._slm_after_put
            if callback is not None:
                callback(saved_config, checkpoint, metadata)
            return saved_config

        saver.put = hooked_put

        def hooked_put_writes(config, writes, task_id, task_path=""):
            result = original_put_writes(
                config,
                writes,
                task_id,
                task_path,
            )
            callback = saver._slm_after_writes
            if callback is not None:
                callback(config, writes, task_id, task_path)
            return result

        saver.put_writes = hooked_put_writes
        try:
            saver.setup()
        except __import__("sqlite3").DatabaseError as exc:
            raise CheckpointCorruptError(
                f"LangGraph SQLite checkpoint is corrupt at {destination}: {exc}"
            ) from exc
        yield saver


def atomic_write_json(path: str | os.PathLike, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(
    path: str | os.PathLike,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_artifact_manifest(
    directory: str | os.PathLike,
    *,
    artifact_type: str,
) -> dict[str, Any]:
    """Hash a completed directory once, then publish its completion marker."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise CheckpointArtifactError(
            f"artifact directory does not exist: {root}"
        )
    if artifact_type == "training_checkpoint":
        adapter_config = root / "adapter_config.json"
        if not adapter_config.is_file():
            raise CheckpointArtifactError(
                f"training checkpoint is missing adapter_config.json: {root}"
            )
        weight_names = (
            "adapter_model.safetensors",
            "adapter_model.bin",
        )
        weights = [root / name for name in weight_names if (root / name).is_file()]
        if not weights:
            raise CheckpointArtifactError(
                f"training checkpoint has no adapter weight file: {root}"
            )
        if any(path.stat().st_size <= 0 for path in weights):
            raise CheckpointArtifactError(
                f"training checkpoint contains an empty adapter weight file: {root}"
            )
    entries = []
    excluded = {ARTIFACT_MANIFEST_FILENAME, ARTIFACT_COMPLETE_FILENAME}
    for child in sorted(root.rglob("*")):
        if not child.is_file() or child.name in excluded:
            continue
        entries.append(
            {
                "path": str(child.relative_to(root)),
                "size": child.stat().st_size,
                "sha256": _sha256_file(child),
            }
        )
    if not entries:
        raise CheckpointArtifactError(f"artifact directory is empty: {root}")
    manifest = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "files": entries,
    }
    manifest_path = root / ARTIFACT_MANIFEST_FILENAME
    atomic_write_json(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)
    atomic_write_json(
        root / ARTIFACT_COMPLETE_FILENAME,
        {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "manifest": ARTIFACT_MANIFEST_FILENAME,
            "manifest_sha256": manifest_sha256,
        },
    )
    return manifest


def _validate_directory_manifest(
    path: Path,
    *,
    verify_contents: bool,
) -> tuple[str, dict[str, Any]]:
    manifest_path = path / ARTIFACT_MANIFEST_FILENAME
    marker_path = path / ARTIFACT_COMPLETE_FILENAME
    if not marker_path.is_file() or not manifest_path.is_file():
        raise CheckpointArtifactError(
            f"directory artifact lacks manifest/completion marker: {path}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointArtifactError(
            f"directory artifact manifest is corrupt at {path}: {exc}"
        ) from exc
    if (
        marker.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION
        or manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION
    ):
        raise CheckpointArtifactError(
            f"directory artifact manifest schema is incompatible: {path}"
        )
    manifest_sha256 = _sha256_file(manifest_path)
    if marker.get("manifest_sha256") != manifest_sha256:
        raise CheckpointArtifactError(
            f"directory artifact manifest hash drifted: {path}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise CheckpointArtifactError(
            f"directory artifact manifest has no files: {path}"
        )
    if verify_contents:
        for entry in files:
            relative = entry.get("path") if isinstance(entry, Mapping) else None
            if not isinstance(relative, str) or not relative:
                raise CheckpointArtifactError(
                    f"directory artifact manifest has an invalid path: {path}"
                )
            candidate = path / relative
            if not candidate.is_file():
                raise CheckpointArtifactError(
                    f"directory artifact file is missing: {candidate}"
                )
            if candidate.stat().st_size != int(entry.get("size", -1)):
                raise CheckpointArtifactError(
                    f"directory artifact file size drifted: {candidate}"
                )
            if _sha256_file(candidate) != entry.get("sha256"):
                raise CheckpointArtifactError(
                    f"directory artifact content hash drifted: {candidate}"
                )
    return manifest_sha256, manifest


def run_training_atomically(
    final_directory: str | os.PathLike,
    producer: Callable[[str], Any],
):
    """Run a trainer in a temp directory and atomically publish/reuse it."""
    from training.lora_trainer import TrainingOutput

    final = Path(final_directory).expanduser().resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    published_checkpoint = final / "final_checkpoint"
    if published_checkpoint.is_dir():
        try:
            _, manifest = _validate_directory_manifest(
                published_checkpoint,
                verify_contents=True,
            )
            if manifest.get("artifact_type") == "training_checkpoint":
                return TrainingOutput(str(published_checkpoint), None)
        except CheckpointArtifactError:
            pass
    for stale in final.parent.glob(f"{final.name}.partial-*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        else:
            stale.unlink(missing_ok=True)
    temporary = final.parent / (
        f"{final.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        output = producer(str(temporary))
        weights_ref = Path(output.weights_ref).expanduser().resolve()
        try:
            relative_weights = weights_ref.relative_to(temporary)
        except ValueError as exc:
            raise CheckpointArtifactError(
                f"trainer returned weights outside its atomic directory: "
                f"{weights_ref}"
            ) from exc
        write_artifact_manifest(
            weights_ref,
            artifact_type="training_checkpoint",
        )
        if final.exists():
            shutil.rmtree(final)
        os.replace(temporary, final)
        _fsync_directory(final.parent)
        published_weights = str(final / relative_weights)
        published_gguf = output.gguf_path
        if published_gguf:
            try:
                relative_gguf = Path(published_gguf).resolve().relative_to(
                    temporary
                )
                published_gguf = str(final / relative_gguf)
            except ValueError:
                pass
        if hasattr(output, "_replace"):
            return output._replace(
                weights_ref=published_weights,
                gguf_path=published_gguf,
            )
        return TrainingOutput(published_weights, published_gguf)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _looks_like_local_path(value: str) -> bool:
    return (
        os.path.isabs(value)
        or value.startswith(".")
        or value.startswith("artifacts/")
        or os.path.exists(value)
    )


def _artifact_values(state: Mapping[str, Any]):
    dataset = state.get("current_dataset_path")
    if isinstance(dataset, str) and dataset:
        yield "dataset", dataset

    for node in state.get("dag") or []:
        if not isinstance(node, Mapping):
            continue
        dataset_identity = (
            ((node.get("pi") or {}).get("D") or {})
            if isinstance(node.get("pi"), Mapping)
            else {}
        )
        path = dataset_identity.get("path")
        if isinstance(path, str) and path:
            version = dataset_identity.get("version")
            yield f"dag_dataset:v{version}", path

    best = state.get("best_weights_ref")
    if isinstance(best, str) and best and _looks_like_local_path(best):
        yield "best_weights", best

    pending = state.get("_pending_weights_refs") or {}
    if isinstance(pending, Mapping):
        for label, value in pending.items():
            if isinstance(value, str) and value and _looks_like_local_path(value):
                yield f"pending_weights:{label}", value


def capture_artifacts(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    seen = set()
    for role, raw_path in _artifact_values(state):
        path = Path(raw_path).expanduser()
        normalized = str(path.resolve())
        identity = (role, normalized)
        if identity in seen:
            continue
        seen.add(identity)
        if not path.exists():
            raise CheckpointArtifactError(
                f"{role} artifact does not exist: {raw_path}"
            )
        if path.is_file():
            kind = "file"
            fingerprint = _sha256_file(path)
            complete = True
        elif path.is_dir():
            kind = "directory"
            fingerprint, manifest = _validate_directory_manifest(
                path,
                verify_contents=False,
            )
            complete = True
        else:
            raise CheckpointArtifactError(
                f"{role} artifact is not a regular file/directory: {raw_path}"
            )
        artifacts.append(
            {
                "role": role,
                "path": normalized,
                "kind": kind,
                "fingerprint": fingerprint,
                "complete": complete,
                "artifact_type": (
                    manifest.get("artifact_type")
                    if kind == "directory"
                    else None
                ),
            }
        )
    return artifacts


def validate_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise CheckpointCorruptError("checkpoint artifacts must be a list")
    validated: dict[tuple[str, str], tuple[str, str | None]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise CheckpointCorruptError("checkpoint artifact entry must be an object")
        role = str(artifact.get("role") or "unknown")
        path = Path(str(artifact.get("path") or ""))
        if not path.exists():
            raise CheckpointArtifactError(f"{role} artifact is missing: {path}")
        kind = artifact.get("kind")
        cache_key = (str(kind), str(path.resolve()))
        if cache_key in validated:
            actual, actual_type = validated[cache_key]
        elif kind == "file" and path.is_file():
            actual = _sha256_file(path)
            actual_type = None
            validated[cache_key] = (actual, actual_type)
        elif kind == "directory" and path.is_dir():
            actual, manifest = _validate_directory_manifest(
                path,
                verify_contents=True,
            )
            actual_type = manifest.get("artifact_type")
            validated[cache_key] = (actual, actual_type)
        else:
            raise CheckpointArtifactError(
                f"{role} artifact changed kind or is invalid: {path}"
            )
        if actual_type != artifact.get("artifact_type"):
            raise CheckpointArtifactError(
                f"{role} artifact type drifted: {path}"
            )
        if actual != artifact.get("fingerprint"):
            raise CheckpointArtifactError(
                f"{role} artifact fingerprint drifted: {path}"
            )


def save_checkpoint(
    path: str | os.PathLike,
    state: Mapping[str, Any],
    *,
    thread_id: str,
    compatibility: Mapping[str, str],
    graph_steps: int,
    last_node: str | None,
    next_nodes: Sequence[str],
    cumulative_wall_time_s: float,
    status: str = "running",
    error: str | None = None,
    sqlite_checkpoint_id: str | None = None,
    sqlite_generation: str | None = None,
    sqlite_step: int | None = None,
    pending_tasks: Sequence[Mapping[str, Any]] = (),
    segment_started_at_epoch_s: float | None = None,
    segment_base_wall_time_s: float | None = None,
) -> dict[str, Any]:
    """Atomically publish one complete application checkpoint."""
    encoded_state = encode_state(state)
    artifacts = capture_artifacts(encoded_state)
    destination = Path(path)
    created_at = None
    previous_progress: Mapping[str, Any] = {}
    if destination.is_file():
        try:
            previous = json.loads(destination.read_text(encoding="utf-8"))
            created_at = previous.get("created_at")
            previous_progress = previous.get("progress") or {}
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    now = _utc_now()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": created_at or now,
        "updated_at": now,
        "thread_id": str(thread_id),
        "compatibility": dict(compatibility),
        "progress": {
            "graph_steps": int(graph_steps),
            "last_node": last_node,
            "next_nodes": list(next_nodes),
            "cumulative_wall_time_s": max(0.0, float(cumulative_wall_time_s)),
            "status": status,
            "error": error,
            "pending_weights_refs": encoded_state.get(
                "_pending_weights_refs"
            ),
            "pending_configs": encoded_state.get("_pending_configs"),
            "sqlite_checkpoint_id": sqlite_checkpoint_id,
            "sqlite_generation": sqlite_generation,
            "sqlite_step": sqlite_step,
            "pending_tasks": [dict(task) for task in pending_tasks],
            "segment_started_at_epoch_s": (
                segment_started_at_epoch_s
                if segment_started_at_epoch_s is not None
                else previous_progress.get("segment_started_at_epoch_s")
            ),
            "segment_base_wall_time_s": (
                segment_base_wall_time_s
                if segment_base_wall_time_s is not None
                else previous_progress.get("segment_base_wall_time_s")
            ),
        },
        "artifacts": artifacts,
        "state": encoded_state,
    }
    atomic_write_json(destination, payload)
    return payload


def _read_checkpoint(path: str | os.PathLike) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise CheckpointCorruptError(
            f"checkpoint cannot be read at {source}: {exc}"
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CheckpointCorruptError(
            f"checkpoint is not valid JSON at {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointCorruptError("checkpoint root must be a JSON object")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "checkpoint schema version is incompatible: "
            f"stored={payload.get('schema_version')!r}, "
            f"expected={CHECKPOINT_SCHEMA_VERSION}"
        )
    for key in ("thread_id", "compatibility", "progress", "artifacts", "state"):
        if key not in payload:
            raise CheckpointCorruptError(
                f"checkpoint is missing required field {key!r}"
            )
    return payload


def _validate_compatibility(
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    labels = {
        "mode": "mode",
        "pool_fingerprint": "pool fingerprint",
        "topology_fingerprint": "topology fingerprint",
        "config_fingerprint": "config fingerprint",
    }
    for key, label in labels.items():
        if key in expected and stored.get(key) != expected.get(key):
            raise CheckpointCompatibilityError(
                f"checkpoint {label} is incompatible: "
                f"stored={stored.get(key)!r}, expected={expected.get(key)!r}"
            )


def load_checkpoint(
    path: str | os.PathLike,
    *,
    expected_compatibility: Mapping[str, Any] | None = None,
    validate_artifact_paths: bool = True,
) -> dict[str, Any]:
    payload = _read_checkpoint(path)
    compatibility = payload["compatibility"]
    if not isinstance(compatibility, Mapping):
        raise CheckpointCorruptError("checkpoint compatibility must be an object")
    if expected_compatibility is not None:
        _validate_compatibility(compatibility, expected_compatibility)
    if validate_artifact_paths:
        validate_artifacts(payload["artifacts"])
    try:
        payload["state"] = decode_state(payload["state"])
    except StateCodecError as exc:
        if "selector" in str(exc):
            raise CheckpointCompatibilityError(
                f"checkpoint model selector is incompatible: {exc}"
            ) from exc
        raise CheckpointCorruptError(f"checkpoint state is invalid: {exc}") from exc
    progress = payload["progress"]
    if not isinstance(progress, Mapping):
        raise CheckpointCorruptError("checkpoint progress must be an object")
    return payload


def restore_or_initialize_state(
    resume_path: str | os.PathLike | None,
    *,
    fresh_state_factory: Callable[[], Mapping[str, Any]],
    expected_compatibility: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """Skip all pre-graph initialization when a checkpoint is supplied."""
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            expected_compatibility=expected_compatibility,
        )
        return checkpoint["state"], checkpoint, True
    return dict(fresh_state_factory()), None, False


def remaining_recursion_limit(completed_steps: int, *, total: int = 1500) -> int:
    """Return LangGraph's limit including its required terminal-check superstep."""
    remaining = int(total) - int(completed_steps)
    if remaining < 0:
        raise RecursionBudgetExhausted(
            f"cumulative LangGraph recursion budget of {total} steps is exhausted"
        )
    # LangGraph 1.2.9 raises when recursion_limit equals the number of node
    # supersteps, even if the final node routed to END. One extra superstep is
    # required to observe termination; it cannot execute another node.
    return remaining + 1


@dataclass(frozen=True)
class SqliteCheckpointState:
    exists: bool
    state: dict[str, Any]
    step: int
    checkpoint_id: str | None
    generation: str | None
    next_nodes: tuple[str, ...]
    pending_tasks: tuple[dict[str, Any], ...]
    last_node: str | None
    created_at: str | None

    @property
    def terminal(self) -> bool:
        return self.exists and not self.next_nodes and not self.pending_tasks


def _task_record(task: Any) -> dict[str, Any]:
    error = getattr(task, "error", None)
    result = getattr(task, "result", None)
    return {
        "id": str(getattr(task, "id", "") or ""),
        "name": str(getattr(task, "name", "") or ""),
        "error": str(error) if error else None,
        "completed": result is not None,
    }


def _last_completed_node(graph, snapshot: Any, step: int) -> str | None:
    if step <= 0:
        return None
    parent_config = getattr(snapshot, "parent_config", None)
    if not parent_config:
        return None
    try:
        parent = graph.get_state(parent_config)
    except Exception:
        return None
    candidates = [
        str(getattr(task, "name", "") or "")
        for task in (getattr(parent, "tasks", ()) or ())
        if str(getattr(task, "name", "") or "") not in {"", "__start__"}
    ]
    return candidates[0] if len(candidates) == 1 else None


def inspect_sqlite_state(graph, config: Mapping[str, Any]) -> SqliteCheckpointState:
    """Read the latest committed LangGraph checkpoint and pending tasks."""
    snapshot = graph.get_state(dict(config))
    metadata = getattr(snapshot, "metadata", None)
    snapshot_config = getattr(snapshot, "config", None) or {}
    configurable = (
        snapshot_config.get("configurable", {})
        if isinstance(snapshot_config, Mapping)
        else {}
    )
    checkpoint_id = configurable.get("checkpoint_id")
    exists = isinstance(metadata, Mapping) and bool(checkpoint_id)
    if not exists:
        return SqliteCheckpointState(
            exists=False,
            state={},
            step=0,
            checkpoint_id=None,
            generation=None,
            next_nodes=(),
            pending_tasks=(),
            last_node=None,
            created_at=None,
        )
    checkpoint_step = max(0, int(metadata.get("step", 0) or 0))
    values = dict(getattr(snapshot, "values", {}) or {})
    snapshot_tasks = tuple(getattr(snapshot, "tasks", ()) or ())
    completed_tasks = [
        task
        for task in snapshot_tasks
        if getattr(task, "result", None) is not None
        and str(getattr(task, "name", "") or "") != "__start__"
    ]
    for task in completed_tasks:
        result = getattr(task, "result", None)
        if isinstance(result, Mapping):
            values.update(result)
    state_steps = values.get("_graph_steps")
    step = max(
        checkpoint_step + len(completed_tasks),
        int(state_steps) if state_steps is not None else 0,
    )
    tasks = tuple(_task_record(task) for task in snapshot_tasks)
    pending_signature = [
        {
            "id": task["id"],
            "name": task["name"],
            "error": task["error"],
            "completed": task["completed"],
        }
        for task in tasks
        if task["completed"] or task["error"]
    ]
    generation = str(checkpoint_id)
    if pending_signature:
        generation = f"{generation}:{_canonical_hash(pending_signature)[:16]}"
    completed_names = [
        str(getattr(task, "name", "") or "")
        for task in completed_tasks
    ]
    last_node = (
        completed_names[0]
        if len(completed_names) == 1
        else _last_completed_node(graph, snapshot, checkpoint_step)
    )
    return SqliteCheckpointState(
        exists=True,
        state=values,
        step=step,
        checkpoint_id=str(checkpoint_id),
        generation=generation,
        next_nodes=tuple(getattr(snapshot, "next", ()) or ()),
        pending_tasks=tasks,
        last_node=last_node,
        created_at=getattr(snapshot, "created_at", None),
    )


def resume_input_from_sqlite(
    sqlite_state: SqliteCheckpointState,
    initial_state: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Resume pending/terminal SQLite state with None; seed only an empty thread."""
    return None if sqlite_state.exists else initial_state


def reconcile_checkpoint_from_sqlite(
    graph,
    *,
    config: Mapping[str, Any],
    checkpoint_path: str | os.PathLike,
    thread_id: str,
    compatibility: Mapping[str, str],
    cumulative_wall_time_s: float,
    status: str | None = None,
    error: str | None = None,
    segment_started_at_epoch_s: float | None = None,
    segment_base_wall_time_s: float | None = None,
) -> SqliteCheckpointState:
    """Atomically rebuild the JSON mirror from authoritative SQLite state."""
    authoritative = inspect_sqlite_state(graph, config)
    if not authoritative.exists:
        return authoritative
    task_errors = [task["error"] for task in authoritative.pending_tasks if task["error"]]
    mirror_status = status
    if mirror_status is None:
        mirror_status = (
            "completed"
            if authoritative.terminal
            else ("interrupted" if task_errors else "running")
        )
    save_checkpoint(
        checkpoint_path,
        authoritative.state,
        thread_id=thread_id,
        compatibility=compatibility,
        graph_steps=authoritative.step,
        last_node=authoritative.last_node,
        next_nodes=authoritative.next_nodes,
        cumulative_wall_time_s=cumulative_wall_time_s,
        status=mirror_status,
        error=error or (task_errors[0] if task_errors else None),
        sqlite_checkpoint_id=authoritative.checkpoint_id,
        sqlite_generation=authoritative.generation,
        sqlite_step=authoritative.step,
        pending_tasks=authoritative.pending_tasks,
        segment_started_at_epoch_s=segment_started_at_epoch_s,
        segment_base_wall_time_s=segment_base_wall_time_s,
    )
    return authoritative


def cumulative_wall_time_from_sqlite(
    json_checkpoint: Mapping[str, Any],
    sqlite_state: SqliteCheckpointState,
) -> float:
    """Recover active segment time when SQLite committed ahead of JSON."""
    progress = json_checkpoint.get("progress") or {}
    recorded = max(
        0.0,
        float(progress.get("cumulative_wall_time_s", 0.0) or 0.0),
    )
    started = progress.get("segment_started_at_epoch_s")
    base = progress.get("segment_base_wall_time_s")
    if started is None or base is None or not sqlite_state.created_at:
        return recorded
    try:
        import datetime

        committed_epoch = datetime.datetime.fromisoformat(
            str(sqlite_state.created_at).replace("Z", "+00:00")
        ).timestamp()
        estimated = float(base) + max(0.0, committed_epoch - float(started))
    except (TypeError, ValueError):
        return recorded
    return max(recorded, estimated)


@dataclass(frozen=True)
class SegmentResult:
    state: dict[str, Any]
    graph_steps: int
    last_node: str | None
    next_nodes: tuple[str, ...]
    cumulative_wall_time_s: float


def _delta_state(delta: Any, previous: Mapping[str, Any] | None):
    if not isinstance(delta, Mapping) or len(delta) != 1:
        raise CheckpointCorruptError(
            "LangGraph update must contain exactly one completed node"
        )
    node_name, update = next(iter(delta.items()))
    if not isinstance(node_name, str) or not isinstance(update, Mapping):
        raise CheckpointCorruptError("LangGraph update has an invalid node/state")
    if previous is None:
        state = dict(update)
    else:
        state = dict(previous)
        state.update(update)
    return node_name, state


def stream_with_checkpoints(
    graph,
    input_state: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any],
    checkpoint_path: str | os.PathLike,
    thread_id: str,
    compatibility: Mapping[str, str],
    initial_graph_steps: int,
    initial_wall_time_s: float,
    clock: Callable[[], float] = time.monotonic,
    on_node: Callable[[str, Mapping[str, Any]], None] | None = None,
    after_sqlite_commit: Callable[[SqliteCheckpointState], None] | None = None,
    max_graph_steps: int = 1500,
) -> SegmentResult:
    """Stream a graph segment and atomically snapshot every completed node.

    ``input_state=None`` is passed through unchanged on resume so LangGraph uses
    the persisted ``next`` tasks and conditional routing rather than re-entering
    the graph at its entry point.
    """
    started = clock()
    authoritative = inspect_sqlite_state(graph, config)
    graph_steps = (
        authoritative.step if authoritative.exists else int(initial_graph_steps)
    )
    last_node = authoritative.last_node if authoritative.exists else None
    next_nodes = authoritative.next_nodes if authoritative.exists else ()
    last_state = (
        dict(authoritative.state)
        if authoritative.exists
        else (dict(input_state) if input_state is not None else None)
    )
    segment_started_at_epoch_s = time.time()
    segment_base_wall_time_s = float(initial_wall_time_s)
    if last_state is not None:
        prior_progress: Mapping[str, Any] = {}
        if Path(checkpoint_path).is_file():
            try:
                prior_progress = _read_checkpoint(checkpoint_path).get(
                    "progress", {}
                )
            except CheckpointError:
                prior_progress = {}
        save_checkpoint(
            checkpoint_path,
            last_state,
            thread_id=thread_id,
            compatibility=compatibility,
            graph_steps=graph_steps,
            last_node=last_node or prior_progress.get("last_node"),
            next_nodes=(
                next_nodes
                or tuple(prior_progress.get("next_nodes") or ())
            ),
            cumulative_wall_time_s=initial_wall_time_s,
            status="running",
            sqlite_checkpoint_id=authoritative.checkpoint_id,
            sqlite_generation=authoritative.generation,
            sqlite_step=authoritative.step if authoritative.exists else None,
            pending_tasks=authoritative.pending_tasks,
            segment_started_at_epoch_s=segment_started_at_epoch_s,
            segment_base_wall_time_s=segment_base_wall_time_s,
        )
    saver = getattr(graph, "checkpointer", None)
    supports_commit_hook = (
        hasattr(saver, "_slm_after_put")
        and hasattr(saver, "_slm_after_writes")
    )
    previous_hook = getattr(saver, "_slm_after_put", None)
    previous_writes_hook = getattr(saver, "_slm_after_writes", None)
    mirrored_generation = authoritative.generation
    mirror_lock = threading.Lock()

    def committed_after_put(saved_config, _checkpoint, metadata):
        nonlocal authoritative, graph_steps, last_node, next_nodes
        nonlocal last_state, mirrored_generation
        with mirror_lock:
            if (metadata or {}).get("source") == "input":
                return
            # Read the exact generation just committed. Another saver thread may
            # already have published a newer generation; checkpoint_id pinning
            # prevents callbacks from mislabelling that newer state.
            committed = inspect_sqlite_state(graph, saved_config)
            if (
                not committed.exists
                or committed.generation == mirrored_generation
            ):
                return
            if after_sqlite_commit is not None:
                after_sqlite_commit(committed)
            cumulative = initial_wall_time_s + max(0.0, clock() - started)
            previous_step = graph_steps
            committed = reconcile_checkpoint_from_sqlite(
                graph,
                config=config,
                checkpoint_path=checkpoint_path,
                thread_id=thread_id,
                compatibility=compatibility,
                cumulative_wall_time_s=cumulative,
                segment_started_at_epoch_s=segment_started_at_epoch_s,
                segment_base_wall_time_s=segment_base_wall_time_s,
            )
            authoritative = committed
            graph_steps = committed.step
            last_node = committed.last_node
            next_nodes = committed.next_nodes
            last_state = committed.state
            mirrored_generation = committed.generation
            if (
                on_node is not None
                and committed.last_node is not None
                and committed.step > previous_step
            ):
                on_node(committed.last_node, committed.state)
            has_completed_pending = any(
                task.get("completed") for task in committed.pending_tasks
            )
            if (
                committed.step >= int(max_graph_steps)
                and committed.next_nodes
                and not has_completed_pending
            ):
                raise RecursionBudgetExhausted(
                    f"cumulative LangGraph recursion budget of "
                    f"{max_graph_steps} steps is exhausted"
                )

    if supports_commit_hook:
        saver._slm_after_put = committed_after_put
        saver._slm_after_writes = (
            lambda saved_config, _writes, _task_id, _task_path: (
                committed_after_put(
                    saved_config,
                    None,
                    {"source": "writes"},
                )
            )
        )

    try:
        if (
            authoritative.exists
            and authoritative.step >= int(max_graph_steps)
            and authoritative.next_nodes
            and not any(
                task.get("completed")
                for task in authoritative.pending_tasks
            )
        ):
            raise RecursionBudgetExhausted(
                f"cumulative LangGraph recursion budget of "
                f"{max_graph_steps} steps is exhausted"
            )
        for delta in graph.stream(
            input_state,
            stream_mode="updates",
            config=dict(config),
        ):
            if supports_commit_hook:
                continue
            # Lightweight graph doubles have no checkpointer hook. Retain the
            # legacy mirror path for unit tests, never for production SQLite.
            last_node, last_state = _delta_state(delta, last_state)
            graph_steps += 1
            snapshot = graph.get_state(dict(config))
            next_nodes = tuple(getattr(snapshot, "next", ()) or ())
            cumulative = initial_wall_time_s + max(0.0, clock() - started)
            save_checkpoint(
                checkpoint_path,
                last_state,
                thread_id=thread_id,
                compatibility=compatibility,
                graph_steps=graph_steps,
                last_node=last_node,
                next_nodes=next_nodes,
                cumulative_wall_time_s=cumulative,
            )
            if on_node is not None:
                on_node(last_node, last_state)
    except BaseException as exc:
        cumulative = initial_wall_time_s + max(0.0, clock() - started)
        reconciled = reconcile_checkpoint_from_sqlite(
            graph,
            config=config,
            checkpoint_path=checkpoint_path,
            thread_id=thread_id,
            compatibility=compatibility,
            cumulative_wall_time_s=cumulative,
            status="interrupted",
            error=f"{type(exc).__name__}: {exc}",
            segment_started_at_epoch_s=segment_started_at_epoch_s,
            segment_base_wall_time_s=segment_base_wall_time_s,
        )
        if reconciled.exists:
            authoritative = reconciled
        raise
    finally:
        if supports_commit_hook:
            saver._slm_after_put = previous_hook
            saver._slm_after_writes = previous_writes_hook

    cumulative = initial_wall_time_s + max(0.0, clock() - started)
    authoritative = reconcile_checkpoint_from_sqlite(
        graph,
        config=config,
        checkpoint_path=checkpoint_path,
        thread_id=thread_id,
        compatibility=compatibility,
        cumulative_wall_time_s=cumulative,
        segment_started_at_epoch_s=segment_started_at_epoch_s,
        segment_base_wall_time_s=segment_base_wall_time_s,
    )
    if authoritative.exists:
        last_state = authoritative.state
        graph_steps = authoritative.step
        last_node = authoritative.last_node
        next_nodes = authoritative.next_nodes
    elif last_state is None:
        last_state = {}
    return SegmentResult(
        state=last_state,
        graph_steps=graph_steps,
        last_node=last_node,
        next_nodes=next_nodes,
        cumulative_wall_time_s=cumulative,
    )
