"""Process-safe, provider-aware API cost observability.

Every instrumented call appends one :class:`CostEvent` to the JSONL path in
``SLM_COST_EVENT_PATH``.  Each append takes an OS file lock, so the runner, its
forked acquisition process, and disposable CUDA workers can safely share a
single ledger.  Provider SDK calls are wrapped explicitly at their call sites;
there is intentionally no global SDK monkey patch.
"""
from __future__ import annotations

import copy
import datetime as _datetime
import fcntl
import inspect
import ipaddress
import json
import os
import tempfile
import threading
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


COST_EVENT_PATH_ENV = "SLM_COST_EVENT_PATH"
PRICING_OVERRIDES_ENV = "SLM_PRICING_OVERRIDES"
PRICING_PATH_ENV = "SLM_PRICING_PATH"
OBSERVABILITY_REQUIRED_ENV = "SLM_OBSERVABILITY_REQUIRED"


class UnknownPricingWarning(UserWarning):
    """A paid model was used without a matching pricing entry."""


class UnknownPricingError(ValueError):
    """Strict pricing was requested for a paid model with no known rate."""


class MissingEventPathError(RuntimeError):
    """Pipeline instrumentation requires a shared cost event path."""

# Official public API rates, USD per million tokens, verified 2026-07-21.
# Sources:
#   Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
#   OpenAI:    https://developers.openai.com/api/docs/models/gpt-4.1
#   DeepSeek:  https://api-docs.deepseek.com/quick_start/pricing
_DEFAULT_PRICING = {
    "effective_date": "2026-07-21",
    "currency": "USD",
    "unit": "per_million_tokens",
    "sources": {
        "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
        "openai": "https://developers.openai.com/api/docs/models/gpt-4.1",
        "deepseek": "https://api-docs.deepseek.com/quick_start/pricing",
    },
    "models": {
        "claude-sonnet-5": {
            "provider": "anthropic",
            "input_per_mtok": 2.0,
            "output_per_mtok": 10.0,
            "cached_input_per_mtok": 0.20,
            "cache_write_5m_per_mtok": 2.50,
            "cache_write_1h_per_mtok": 4.0,
            "valid_through": "2026-08-31",
            "rate_note": "introductory rate through August 31, 2026",
        },
        "claude-opus-4-8": {
            "provider": "anthropic",
            "input_per_mtok": 5.0,
            "output_per_mtok": 25.0,
            "cached_input_per_mtok": 0.50,
            "cache_write_5m_per_mtok": 6.25,
            "cache_write_1h_per_mtok": 10.0,
        },
        "claude-sonnet-4-6": {
            "provider": "anthropic",
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
            "cached_input_per_mtok": 0.30,
            "cache_write_5m_per_mtok": 3.75,
            "cache_write_1h_per_mtok": 6.0,
        },
        "claude-haiku-4-5": {
            "provider": "anthropic",
            "input_per_mtok": 1.0,
            "output_per_mtok": 5.0,
            "cached_input_per_mtok": 0.10,
            "cache_write_5m_per_mtok": 1.25,
            "cache_write_1h_per_mtok": 2.0,
        },
        "gpt-4.1": {
            "provider": "openai",
            "input_per_mtok": 2.0,
            "output_per_mtok": 8.0,
            "cached_input_per_mtok": 0.50,
        },
        "deepseek-v4-flash": {
            "provider": "deepseek",
            "input_per_mtok": 0.14,
            "output_per_mtok": 0.28,
            "cached_input_per_mtok": 0.0028,
        },
    },
    # Exa exposes the authoritative cost on successful responses.  This is used
    # only by the legacy record_exa interface or older SDK responses without it.
    "exa_fallback_per_request": 0.005,
}

# Legacy names retained for callers/tests that imported these constants.
SONNET_INPUT_PER_M = 3.0
SONNET_OUTPUT_PER_M = 15.0
EXA_SEARCH_COST = 0.005

_append_thread_lock = threading.Lock()
_fallback_path_lock = threading.Lock()
_fallback_path: str | None = None


def _utc_timestamp() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _deep_merge(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def pricing_registry() -> dict:
    """Return a fresh pricing registry with optional JSON overrides applied.

    ``SLM_PRICING_PATH`` may point to a JSON file and
    ``SLM_PRICING_OVERRIDES`` may contain an inline JSON object.  Both are
    recursively merged over the dated defaults, with inline values winning.
    """
    registry = copy.deepcopy(_DEFAULT_PRICING)
    pricing_path = os.environ.get(PRICING_PATH_ENV)
    if pricing_path:
        with open(pricing_path, encoding="utf-8") as pricing_file:
            _deep_merge(registry, json.load(pricing_file))
    inline = os.environ.get(PRICING_OVERRIDES_ENV)
    if inline:
        _deep_merge(registry, json.loads(inline))
    return registry


def _pricing_key(provider: str, model: str, registry: dict) -> str | None:
    provider_l = (provider or "").lower()
    model_l = (model or "").lower()
    models = registry.get("models", {})
    if model_l in models:
        return model_l
    aliases = (
        ("claude-sonnet-5", "claude-sonnet-5"),
        ("claude-opus-4-8", "claude-opus-4-8"),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("claude-haiku-4-5", "claude-haiku-4-5"),
        ("deepseek-v4-flash", "deepseek-v4-flash"),
        # Until retirement, this alias has V4 Flash thinking-mode billing.
        ("deepseek-reasoner", "deepseek-v4-flash"),
        ("gpt-4.1", "gpt-4.1"),
    )
    for needle, key in aliases:
        if needle in model_l and key in models:
            return key
    # An exact provider-specific custom key can still be supplied by override.
    for key, rates in models.items():
        if (
            str(rates.get("provider", "")).lower() == provider_l
            and key.lower() == model_l
        ):
            return key
    return None


def pricing_status(provider: str, model: str) -> str:
    """Return ``known``, ``unknown``, or ``not_applicable`` for one model."""
    provider_l = (provider or "").lower()
    if provider_l == "local":
        return "not_applicable"
    if provider_l == "exa":
        return "provider_reported"
    return (
        "known"
        if _pricing_key(provider, model, pricing_registry()) is not None
        else "unknown"
    )


def estimate_cost_usd(
    provider: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    actual_usd: float | None = None,
    strict: bool = False,
) -> float:
    """Estimate one call from dated rates, or return a provider's actual cost."""
    if actual_usd is not None:
        return max(0.0, float(actual_usd))
    if (provider or "").lower() == "local":
        return 0.0
    registry = pricing_registry()
    key = _pricing_key(provider, model, registry)
    if key is None:
        message = (
            f"No pricing entry for paid provider={provider!r}, model={model!r}; "
            f"estimated_usd is unknown. Add an override via "
            f"{PRICING_OVERRIDES_ENV} or {PRICING_PATH_ENV}."
        )
        if strict:
            raise UnknownPricingError(message)
        warnings.warn(message, UnknownPricingWarning, stacklevel=2)
        return 0.0
    rates = registry["models"][key]
    amount = (
        max(0, int(input_tokens or 0)) * float(rates.get("input_per_mtok", 0.0))
        + max(0, int(output_tokens or 0))
        * float(rates.get("output_per_mtok", 0.0))
        + max(0, int(cache_read_tokens or 0))
        * float(rates.get("cached_input_per_mtok", 0.0))
        + max(0, int(cache_write_tokens or 0))
        * float(
            rates.get(
                "cache_write_5m_per_mtok", rates.get("input_per_mtok", 0.0)
            )
        )
        + max(0, int(cache_write_1h_tokens or 0))
        * float(
            rates.get(
                "cache_write_1h_per_mtok", rates.get("input_per_mtok", 0.0)
            )
        )
    ) / 1_000_000
    return float(amount)


@dataclass
class CostEvent:
    """One provider request, successful or failed."""

    provider: str
    model: str
    stage: str
    callsite: str = ""
    response_id: str | None = None
    request_id: str | None = None
    status: str = "success"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    latency_ms: float = 0.0
    estimated_usd: float = 0.0
    pricing_status: str = "provided"
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_timestamp)
    pid: int = field(default_factory=os.getpid)

    def to_dict(self) -> dict:
        payload = asdict(self)
        if not payload["cache_tokens"]:
            payload["cache_tokens"] = (
                payload["cache_read_tokens"]
                + payload["cache_write_tokens"]
                + payload["cache_write_1h_tokens"]
            )
        payload["latency_ms"] = round(float(payload["latency_ms"]), 3)
        payload["estimated_usd"] = round(float(payload["estimated_usd"]), 12)
        return payload


def _absolute_path(path: str | os.PathLike) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _resolve_event_path(
    path: str | os.PathLike | None = None,
    *,
    required: bool | None = None,
) -> str:
    if path is not None:
        return _absolute_path(path)
    configured = os.environ.get(COST_EVENT_PATH_ENV)
    if configured:
        return _absolute_path(configured)
    if required is None:
        required = os.environ.get(OBSERVABILITY_REQUIRED_ENV) == "1"
    if required:
        raise MissingEventPathError(
            f"{COST_EVENT_PATH_ENV} is required for pipeline instrumentation"
        )
    # Library/test callers outside run.py still get file-backed accounting.
    global _fallback_path
    with _fallback_path_lock:
        if _fallback_path is None:
            _fallback_path = os.path.join(
                tempfile.gettempdir(),
                f"slm-cost-{os.getpid()}-{uuid.uuid4().hex}.jsonl",
            )
        os.environ.setdefault(COST_EVENT_PATH_ENV, _fallback_path)
        return _fallback_path


def _append_jsonl(path: str, payload: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    # A process-level flock protects forked/spawned writers.  The local lock also
    # serializes threads in high-concurrency CoT annotation.
    with _append_thread_lock:
        with open(path, "a", encoding="utf-8") as output:
            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
            try:
                output.write(encoded + "\n")
                output.flush()
            finally:
                fcntl.flock(output.fileno(), fcntl.LOCK_UN)


def _cost_log_line(event: CostEvent) -> str:
    cache_tokens = (
        event.cache_tokens
        or event.cache_read_tokens
        + event.cache_write_tokens
        + event.cache_write_1h_tokens
    )
    return (
        f"[cost] status={event.status} provider={event.provider} "
        f"model={event.model or '<unknown>'} stage={event.stage} "
        f"tokens={event.input_tokens}->{event.output_tokens} cache={cache_tokens} "
        f"latency={event.latency_ms:.1f}ms usd=${event.estimated_usd:.8f} "
        f"pricing={event.pricing_status}"
    )


def record_cost_event(
    event: CostEvent, path: str | os.PathLike | None = None
) -> CostEvent:
    """Append and log one event."""
    _append_jsonl(_resolve_event_path(path), event.to_dict())
    print(_cost_log_line(event), flush=True)
    return event


def install_cost_tracking(
    event_path: str | os.PathLike | None = None,
    *,
    required: bool | None = None,
) -> "CostLedger":
    """Configure this process to contribute to a shared event file.

    Child processes normally inherit the path.  Calling this again in a spawned
    CUDA worker is harmless and ensures the destination exists before dispatch.
    """
    if required is not None:
        os.environ[OBSERVABILITY_REQUIRED_ENV] = "1" if required else "0"
    resolved = _resolve_event_path(event_path, required=required)
    os.environ[COST_EVENT_PATH_ENV] = resolved
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    Path(resolved).touch(exist_ok=True)
    return LEDGER


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        try:
            value = getattr(obj, name)
        except (AttributeError, TypeError):
            continue
        if value is not None:
            return value
    return default


def _nested(obj: Any, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current = obj
        found = True
        for name in path:
            current = _value(current, name, default=None)
            if current is None:
                found = False
                break
        if found:
            return current
    return None


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _infer_callsite() -> str:
    for frame in inspect.stack()[2:]:
        module = frame.frame.f_globals.get("__name__", "")
        if module != __name__:
            return f"{module}.{frame.function}:{frame.lineno}"
    return "<unknown>"


def _response_id(response: Any) -> str | None:
    value = _value(response, "id", "request_id", "requestId")
    if value is None:
        value = _nested(
            response,
            ("response_metadata", "id"),
            ("response_metadata", "request_id"),
        )
    return str(value) if value is not None else None


def _exception_request_id(exc: BaseException) -> str | None:
    value = _value(exc, "request_id", "requestId")
    if value is None:
        value = _nested(exc, ("response", "headers", "request-id"))
    return str(value) if value is not None else None


def _anthropic_usage(response: Any) -> dict:
    usage = _value(response, "usage")
    if usage is None:
        usage = _value(response, "usage_metadata")
    if usage is None:
        usage = _nested(response, ("response_metadata", "usage"))
    details = _value(usage, "input_token_details", "input_tokens_details", default={})
    cache_creation = _value(usage, "cache_creation", default=None)
    cache_write_5m_raw = _value(
        cache_creation, "ephemeral_5m_input_tokens", default=None
    )
    cache_write_1h_raw = _value(
        cache_creation, "ephemeral_1h_input_tokens", default=None
    )
    has_duration_breakdown = (
        cache_write_5m_raw is not None or cache_write_1h_raw is not None
    )
    if has_duration_breakdown:
        # The top-level cache_creation_input_tokens is the TOTAL of these
        # duration buckets. Never fall back to that total merely because the
        # 5-minute bucket is zero; doing so double-charges pure 1-hour writes.
        cache_write_5m = _integer(cache_write_5m_raw)
        cache_write_1h = _integer(cache_write_1h_raw)
    else:
        total_creation = _value(
            usage,
            "cache_creation_input_tokens",
            default=_value(details, "cache_creation", default=cache_creation or 0),
        )
        cache_write_5m = _integer(total_creation)
        cache_write_1h = 0
    cache_read = _integer(
        _value(
            usage,
            "cache_read_input_tokens",
            default=_value(details, "cache_read", default=0),
        )
    )
    return {
        "input_tokens": _integer(
            _value(usage, "input_tokens", "prompt_tokens", default=0)
        ),
        "output_tokens": _integer(
            _value(usage, "output_tokens", "completion_tokens", default=0)
        ),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write_5m,
        "cache_write_1h_tokens": cache_write_1h,
    }


def _event_for_anthropic_response(
    response: Any,
    *,
    model: str,
    stage: str,
    callsite: str,
    latency_ms: float,
    metadata: dict,
) -> CostEvent:
    usage = _anthropic_usage(response)
    model_pricing_status = pricing_status("anthropic", model)
    return CostEvent(
        provider="anthropic",
        model=model,
        stage=stage,
        callsite=callsite,
        response_id=_response_id(response),
        status="success",
        latency_ms=latency_ms,
        estimated_usd=estimate_cost_usd("anthropic", model, **usage),
        pricing_status=model_pricing_status,
        metadata=metadata,
        **usage,
    )


def tracked_anthropic_messages_create(
    messages_api: Any,
    *args,
    stage: str,
    event_path: str | os.PathLike | None = None,
    callsite: str | None = None,
    **kwargs,
) -> Any:
    """Call ``Anthropic.messages.create`` and append exactly one event."""
    model = str(kwargs.get("model", ""))
    site = callsite or _infer_callsite()
    started = time.perf_counter()
    try:
        response = messages_api.create(*args, **kwargs)
    except BaseException as exc:
        latency = (time.perf_counter() - started) * 1000
        record_cost_event(
            CostEvent(
                provider="anthropic",
                model=model,
                stage=stage,
                callsite=site,
                request_id=_exception_request_id(exc),
                status="error",
                latency_ms=latency,
                pricing_status=pricing_status("anthropic", model),
                metadata={
                    "operation": "messages.create",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            ),
            path=event_path,
        )
        raise
    latency = (time.perf_counter() - started) * 1000
    record_cost_event(
        _event_for_anthropic_response(
            response,
            model=model,
            stage=stage,
            callsite=site,
            latency_ms=latency,
            metadata={"operation": "messages.create"},
        ),
        path=event_path,
    )
    return response


def tracked_chat_anthropic_invoke(
    llm: Any,
    messages: Any,
    *,
    stage: str,
    model: str,
    event_path: str | os.PathLike | None = None,
    callsite: str | None = None,
    **kwargs,
) -> Any:
    """Call one LangChain ChatAnthropic round without double-counting it."""
    site = callsite or _infer_callsite()
    started = time.perf_counter()
    try:
        response = llm.invoke(messages, **kwargs)
    except BaseException as exc:
        latency = (time.perf_counter() - started) * 1000
        record_cost_event(
            CostEvent(
                provider="anthropic",
                model=model,
                stage=stage,
                callsite=site,
                request_id=_exception_request_id(exc),
                status="error",
                latency_ms=latency,
                pricing_status=pricing_status("anthropic", model),
                metadata={
                    "operation": "ChatAnthropic.invoke",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            ),
            path=event_path,
        )
        raise
    latency = (time.perf_counter() - started) * 1000
    response_model = _nested(response, ("response_metadata", "model_name"))
    record_cost_event(
        _event_for_anthropic_response(
            response,
            model=str(response_model or model),
            stage=stage,
            callsite=site,
            latency_ms=latency,
            metadata={"operation": "ChatAnthropic.invoke"},
        ),
        path=event_path,
    )
    return response


def _client_base_url(client: Any) -> str:
    base_url = _value(client, "base_url")
    if base_url is None:
        base_url = _nested(client, ("_client", "base_url"))
    return str(base_url or "")


def _is_private_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    host = hostname.strip("[]").lower()
    if host in {"localhost", "0.0.0.0"} or host.endswith(".local"):
        return True
    try:
        return bool(ipaddress.ip_address(host).is_private)
    except ValueError:
        # Bare SLURM node names (no DNS suffix) are internal endpoints.
        return "." not in host


def openai_provider(client: Any, model: str, provider: str | None = None) -> str:
    """Classify an OpenAI-compatible client, prioritizing local-vLLM safety."""
    model_l = (model or "").lower()
    base_url = _client_base_url(client)
    parsed = urlparse(base_url if "://" in base_url else f"//{base_url}")
    host = parsed.hostname
    configured_local = os.environ.get("SLM_SYNTH_ENDPOINT", "")
    if (
        (provider or "").lower() == "local"
        or "qwen" in model_l
        or "vllm" in model_l
        or _is_private_host(host)
        or (configured_local and base_url.rstrip("/") == configured_local.rstrip("/"))
    ):
        return "local"
    if "deepseek" in model_l or (host and "deepseek.com" in host):
        return "deepseek"
    if host and "openai.com" in host:
        return "openai"
    return (provider or "openai").lower()


def _openai_usage(response: Any, provider: str) -> dict:
    usage = _value(response, "usage", default={})
    prompt_total = _integer(
        _value(usage, "prompt_tokens", "input_tokens", default=0)
    )
    output_tokens = _integer(
        _value(usage, "completion_tokens", "output_tokens", default=0)
    )
    details = _value(
        usage, "prompt_tokens_details", "input_tokens_details", default={}
    )
    cached = _integer(
        _value(
            usage,
            "prompt_cache_hit_tokens",
            default=_value(details, "cached_tokens", default=0),
        )
    )
    misses = _value(usage, "prompt_cache_miss_tokens", default=None)
    if misses is None:
        input_tokens = max(0, prompt_total - cached)
    else:
        input_tokens = _integer(misses)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cached,
        "cache_write_tokens": 0,
        "cache_write_1h_tokens": 0,
    }


def tracked_openai_chat_create(
    client: Any,
    *args,
    stage: str,
    event_path: str | os.PathLike | None = None,
    callsite: str | None = None,
    provider: str | None = None,
    **kwargs,
) -> Any:
    """Track an OpenAI-compatible chat call, including local vLLM at $0."""
    model = str(kwargs.get("model", ""))
    resolved_provider = openai_provider(client, model, provider)
    site = callsite or _infer_callsite()
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(*args, **kwargs)
    except BaseException as exc:
        latency = (time.perf_counter() - started) * 1000
        record_cost_event(
            CostEvent(
                provider=resolved_provider,
                model=model,
                stage=stage,
                callsite=site,
                request_id=_exception_request_id(exc),
                status="error",
                latency_ms=latency,
                pricing_status=pricing_status(resolved_provider, model),
                metadata={
                    "operation": "chat.completions.create",
                    "base_url": _client_base_url(client),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            ),
            path=event_path,
        )
        raise
    latency = (time.perf_counter() - started) * 1000
    usage = _openai_usage(response, resolved_provider)
    model_pricing_status = pricing_status(resolved_provider, model)
    record_cost_event(
        CostEvent(
            provider=resolved_provider,
            model=model,
            stage=stage,
            callsite=site,
            response_id=_response_id(response),
            status="success",
            latency_ms=latency,
            estimated_usd=estimate_cost_usd(
                resolved_provider, model, **usage
            ),
            pricing_status=model_pricing_status,
            metadata={
                "operation": "chat.completions.create",
                "base_url": _client_base_url(client),
            },
            **usage,
        ),
        path=event_path,
    )
    return response


def tracked_local_call(
    call: Callable[..., Any],
    *args,
    stage: str,
    model: str,
    operation: str,
    event_path: str | os.PathLike | None = None,
    callsite: str | None = None,
    **kwargs,
) -> Any:
    """Track a non-chat local service call (for example vLLM preflight)."""
    site = callsite or _infer_callsite()
    started = time.perf_counter()
    try:
        response = call(*args, **kwargs)
    except BaseException as exc:
        record_cost_event(
            CostEvent(
                provider="local",
                model=model,
                stage=stage,
                callsite=site,
                status="error",
                latency_ms=(time.perf_counter() - started) * 1000,
                pricing_status="not_applicable",
                metadata={
                    "operation": operation,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            ),
            path=event_path,
        )
        raise
    record_cost_event(
        CostEvent(
            provider="local",
            model=model,
            stage=stage,
            callsite=site,
            response_id=_response_id(response),
            status="success",
            latency_ms=(time.perf_counter() - started) * 1000,
            estimated_usd=0.0,
            pricing_status="not_applicable",
            metadata={"operation": operation},
        ),
        path=event_path,
    )
    return response


def _exa_actual_cost(response: Any) -> float | None:
    value = _nested(
        response,
        ("costDollars", "total"),
        ("cost_dollars", "total"),
    )
    return _number(value)


def tracked_exa_call(
    call: Callable[..., Any],
    *args,
    stage: str,
    model: str,
    event_path: str | os.PathLike | None = None,
    callsite: str | None = None,
    **kwargs,
) -> Any:
    """Track one Exa request and prefer ``costDollars.total`` over estimates."""
    site = callsite or _infer_callsite()
    started = time.perf_counter()
    try:
        response = call(*args, **kwargs)
    except BaseException as exc:
        latency = (time.perf_counter() - started) * 1000
        record_cost_event(
            CostEvent(
                provider="exa",
                model=model,
                stage=stage,
                callsite=site,
                request_id=_exception_request_id(exc),
                status="error",
                latency_ms=latency,
                pricing_status="provider_reported",
                metadata={
                    "operation": getattr(call, "__name__", "exa_call"),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            ),
            path=event_path,
        )
        raise
    latency = (time.perf_counter() - started) * 1000
    actual = _exa_actual_cost(response)
    estimated = (
        actual
        if actual is not None
        else float(pricing_registry().get("exa_fallback_per_request", 0.0))
    )
    record_cost_event(
        CostEvent(
            provider="exa",
            model=model,
            stage=stage,
            callsite=site,
            response_id=_response_id(response),
            status="success",
            latency_ms=latency,
            estimated_usd=estimated,
            pricing_status="actual" if actual is not None else "fallback",
            metadata={
                "operation": getattr(call, "__name__", "exa_call"),
                "actual_cost": actual is not None,
            },
        ),
        path=event_path,
    )
    return response


def _empty_summary() -> dict:
    return {
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_tokens": 0,
        "latency_ms": 0.0,
        "estimated_usd": 0.0,
        "unknown_pricing_calls": 0,
    }


def _add_summary(summary: dict, event: dict) -> None:
    summary["calls"] += 1
    if event.get("status") == "success":
        summary["successes"] += 1
    else:
        summary["failures"] += 1
    summary["input_tokens"] += _integer(event.get("input_tokens"))
    summary["output_tokens"] += _integer(event.get("output_tokens"))
    summary["cache_tokens"] += _integer(
        event.get("cache_tokens")
        or _integer(event.get("cache_read_tokens"))
        + _integer(event.get("cache_write_tokens"))
        + _integer(event.get("cache_write_1h_tokens"))
    )
    summary["latency_ms"] += float(event.get("latency_ms") or 0.0)
    summary["estimated_usd"] += float(event.get("estimated_usd") or 0.0)
    if event.get("pricing_status") == "unknown":
        summary["unknown_pricing_calls"] += 1


def _round_summary(summary: dict) -> dict:
    rounded = dict(summary)
    rounded["latency_ms"] = round(float(rounded["latency_ms"]), 3)
    rounded["estimated_usd"] = round(float(rounded["estimated_usd"]), 8)
    return rounded


class CostLedger:
    """Read/aggregate an append-only event file.

    ``event_path`` is optional so the legacy process-wide ``LEDGER`` follows the
    environment path configured by run.py and inherited by child processes.
    """

    def __init__(self, event_path: str | os.PathLike | None = None):
        self._event_path = (
            _absolute_path(event_path) if event_path is not None else None
        )

    @property
    def event_path(self) -> str:
        return _resolve_event_path(self._event_path)

    def append(self, event: CostEvent) -> CostEvent:
        return record_cost_event(event, path=self.event_path)

    def events(self) -> list[dict]:
        path = self.event_path
        if not os.path.exists(path):
            return []
        parsed: list[dict] = []
        with open(path, encoding="utf-8") as source:
            fcntl.flock(source.fileno(), fcntl.LOCK_SH)
            try:
                for line in source:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        parsed.append(value)
            finally:
                fcntl.flock(source.fileno(), fcntl.LOCK_UN)
        return parsed

    def reset(self) -> None:
        """Compatibility helper for tests/new runs; production ledgers are append-only."""
        path = self.event_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with _append_thread_lock:
            with open(path, "w", encoding="utf-8") as output:
                fcntl.flock(output.fileno(), fcntl.LOCK_EX)
                fcntl.flock(output.fileno(), fcntl.LOCK_UN)

    def record_anthropic(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-sonnet-4-6",
        stage: str = "legacy",
    ) -> None:
        self.append(
            CostEvent(
                provider="anthropic",
                model=model,
                stage=stage,
                callsite="CostLedger.record_anthropic",
                input_tokens=_integer(input_tokens),
                output_tokens=_integer(output_tokens),
                estimated_usd=estimate_cost_usd(
                    "anthropic",
                    model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                pricing_status=pricing_status("anthropic", model),
            )
        )

    def record_exa(self, n: int = 1, stage: str = "legacy") -> None:
        for _ in range(max(0, int(n))):
            self.append(
                CostEvent(
                    provider="exa",
                    model="search",
                    stage=stage,
                    callsite="CostLedger.record_exa",
                    estimated_usd=float(
                        pricing_registry().get("exa_fallback_per_request", 0.0)
                    ),
                    pricing_status="fallback",
                    metadata={"actual_cost": False},
                )
            )

    def snapshot(self) -> dict:
        events = self.events()
        total = _empty_summary()
        by_provider: dict[str, dict] = {}
        by_model: dict[str, dict] = {}
        by_stage: dict[str, dict] = {}
        by_provider_model_stage: dict[tuple[str, str, str], dict] = {}
        for event in events:
            provider = str(event.get("provider") or "unknown")
            model = str(event.get("model") or "unknown")
            stage = str(event.get("stage") or "unknown")
            _add_summary(total, event)
            _add_summary(by_provider.setdefault(provider, _empty_summary()), event)
            _add_summary(by_model.setdefault(model, _empty_summary()), event)
            _add_summary(by_stage.setdefault(stage, _empty_summary()), event)
            key = (provider, model, stage)
            _add_summary(
                by_provider_model_stage.setdefault(key, _empty_summary()), event
            )

        anthropic = by_provider.get("anthropic", _empty_summary())
        exa = by_provider.get("exa", _empty_summary())
        compounds = [
            {
                "provider": provider,
                "model": model,
                "stage": stage,
                **_round_summary(summary),
            }
            for (provider, model, stage), summary in sorted(
                by_provider_model_stage.items()
            )
        ]
        unknown_pricing = [
            {"provider": provider, "model": model, "stage": stage}
            for provider, model, stage in sorted(
                {
                    (
                        str(event.get("provider") or "unknown"),
                        str(event.get("model") or "unknown"),
                        str(event.get("stage") or "unknown"),
                    )
                    for event in events
                    if event.get("pricing_status") == "unknown"
                }
            )
        ]
        return {
            "schema_version": 2,
            "event_path": self.event_path,
            "pricing": pricing_registry(),
            "total_calls": total["calls"],
            "successful_calls": total["successes"],
            "failed_calls": total["failures"],
            "paid_calls": sum(
                summary["calls"]
                for provider, summary in by_provider.items()
                if provider != "local"
            ),
            "input_tokens": total["input_tokens"],
            "output_tokens": total["output_tokens"],
            "cache_tokens": total["cache_tokens"],
            "total_latency_ms": round(total["latency_ms"], 3),
            "total_cost_usd": round(total["estimated_usd"], 8),
            "pricing_complete": total["unknown_pricing_calls"] == 0,
            "unknown_pricing_calls": total["unknown_pricing_calls"],
            "unknown_pricing": unknown_pricing,
            "by_provider": {
                key: _round_summary(value)
                for key, value in sorted(by_provider.items())
            },
            "by_model": {
                key: _round_summary(value)
                for key, value in sorted(by_model.items())
            },
            "by_stage": {
                key: _round_summary(value)
                for key, value in sorted(by_stage.items())
            },
            "by_provider_model_stage": compounds,
            # Backward-compatible top-level fields used by the existing runner.
            "anthropic_calls": anthropic["calls"],
            "anthropic_cost_usd": round(anthropic["estimated_usd"], 8),
            "exa_calls": exa["calls"],
            "exa_cost_usd": round(exa["estimated_usd"], 8),
        }

    @property
    def total_cost(self) -> float:
        return float(self.snapshot()["total_cost_usd"])

    @property
    def anthropic_cost(self) -> float:
        return float(self.snapshot()["anthropic_cost_usd"])

    @property
    def exa_cost(self) -> float:
        return float(self.snapshot()["exa_cost_usd"])

    @property
    def anthropic_calls(self) -> int:
        return int(self.snapshot()["anthropic_calls"])

    @property
    def exa_calls(self) -> int:
        return int(self.snapshot()["exa_calls"])

    @property
    def input_tokens(self) -> int:
        return int(self.snapshot()["input_tokens"])

    @property
    def output_tokens(self) -> int:
        return int(self.snapshot()["output_tokens"])


LEDGER = CostLedger()
