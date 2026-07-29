"""Strict JSON codec for durable pipeline state.

Only restartable values are serialized.  Live training objects are deliberately
discarded; the paths/configuration needed by the following node are stored in
``_pending_weights_refs`` and ``_pending_configs``.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from config.android_pool import ANDROID_POOL, HardwareConstraints, ModelSpec
from data.eval_set import EvalSet
from eval.harness import EvalResult


class StateCodecError(ValueError):
    """Application state cannot be safely stored or restored."""


def _json_safe(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StateCodecError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        encoded = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StateCodecError(f"{path} has a non-string mapping key")
            encoded[key] = _json_safe(item, f"{path}.{key}")
        return encoded
    raise StateCodecError(
        f"{path} contains a runtime object of type {type(value).__name__}"
    )


def _encode_eval_set(value: EvalSet | None) -> dict | None:
    if value is None:
        return None
    return _json_safe(
        {
            "pos": value.pos,
            "neg": value.neg,
            "boundary": value.boundary,
            "task_type": value.task_type,
            "multi_label": value.multi_label,
            "schema": value.schema,
            "multilingual": value.multilingual,
        },
        "state.eval_set",
    )


def _encode_eval_result(value: EvalResult | None) -> dict | None:
    if value is None:
        return None
    return _json_safe(asdict(value), "state.last_eval")


def encode_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Encode an AgentState mapping without pickling live runtime objects."""
    if not isinstance(state, Mapping):
        raise StateCodecError("state must be a mapping")

    encoded: dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise StateCodecError("state has a non-string key")
        if key == "_pending_training_outputs":
            continue
        if key == "selected_model":
            if value is not None and not isinstance(value, ModelSpec):
                raise StateCodecError("state.selected_model is not a ModelSpec")
            encoded[key] = value.selector if value is not None else None
        elif key == "feasible_models":
            models = value or []
            if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
                raise StateCodecError("state.feasible_models must be a sequence")
            if not all(isinstance(model, ModelSpec) for model in models):
                raise StateCodecError(
                    "state.feasible_models contains a runtime object that is not ModelSpec"
                )
            encoded[key] = [model.selector for model in models]
        elif key == "hardware_constraints":
            if not isinstance(value, HardwareConstraints):
                raise StateCodecError(
                    "state.hardware_constraints is not HardwareConstraints"
                )
            encoded[key] = _json_safe(asdict(value), f"state.{key}")
        elif key == "eval_set":
            if value is not None and not isinstance(value, EvalSet):
                raise StateCodecError("state.eval_set is not an EvalSet")
            encoded[key] = _encode_eval_set(value)
        elif key == "last_eval":
            if value is not None and not isinstance(value, EvalResult):
                raise StateCodecError("state.last_eval is not an EvalResult")
            encoded[key] = _encode_eval_result(value)
        else:
            encoded[key] = _json_safe(value, f"state.{key}")
    return encoded


def _exact_model(selector: Any, model_pool: Sequence[ModelSpec]) -> ModelSpec | None:
    if selector is None:
        return None
    if not isinstance(selector, str) or "@" not in selector:
        raise StateCodecError(
            f"checkpoint requires an exact model selector, got {selector!r}"
        )
    matches = [model for model in model_pool if model.selector == selector]
    if len(matches) != 1:
        raise StateCodecError(
            f"exact model selector {selector!r} is absent or ambiguous in the current pool"
        )
    return matches[0]


def decode_state(
    payload: Mapping[str, Any],
    *,
    model_pool: Sequence[ModelSpec] = ANDROID_POOL,
) -> dict[str, Any]:
    """Rehydrate domain objects from a previously encoded state mapping."""
    if not isinstance(payload, Mapping):
        raise StateCodecError("encoded state must be a mapping")
    # Validate all ordinary values before constructing application objects.
    decoded = _json_safe(payload, "state")

    if "selected_model" in decoded:
        decoded["selected_model"] = _exact_model(
            decoded["selected_model"], model_pool
        )
    if "feasible_models" in decoded:
        selectors = decoded["feasible_models"] or []
        if not isinstance(selectors, list):
            raise StateCodecError("state.feasible_models must be a list")
        decoded["feasible_models"] = [
            _exact_model(selector, model_pool) for selector in selectors
        ]
    if "hardware_constraints" in decoded:
        value = decoded["hardware_constraints"]
        if not isinstance(value, dict):
            raise StateCodecError("state.hardware_constraints must be an object")
        try:
            decoded["hardware_constraints"] = HardwareConstraints(**value)
        except (TypeError, ValueError) as exc:
            raise StateCodecError(
                f"invalid hardware constraints in checkpoint: {exc}"
            ) from exc
    if "eval_set" in decoded and decoded["eval_set"] is not None:
        value = decoded["eval_set"]
        if not isinstance(value, dict):
            raise StateCodecError("state.eval_set must be an object")
        try:
            decoded["eval_set"] = EvalSet(**value)
        except (TypeError, ValueError) as exc:
            raise StateCodecError(f"invalid EvalSet in checkpoint: {exc}") from exc
    if "last_eval" in decoded and decoded["last_eval"] is not None:
        value = decoded["last_eval"]
        if not isinstance(value, dict):
            raise StateCodecError("state.last_eval must be an object")
        try:
            decoded["last_eval"] = EvalResult(**value)
        except (TypeError, ValueError) as exc:
            raise StateCodecError(f"invalid EvalResult in checkpoint: {exc}") from exc

    decoded["_pending_training_outputs"] = None
    return decoded


# Explicit names are useful to callers; short aliases keep the API ergonomic.
encode_agent_state = encode_state
decode_agent_state = decode_state
