"""
Local synthesis client (B161): hard-negative / rare-class synthesis via a LOCAL vLLM
OpenAI-compatible endpoint serving Qwen3.6-35B-A3B (config.SYNTH_ENDPOINT), instead of the
Claude orchestrator API.

Why local: (1) no Claude cost, (2) fully reproducible + contamination-safe (a model we own),
(3) the 35B is strong enough for contrastive example generation. If the endpoint is not
configured or unreachable, callers get None and degrade gracefully (gold-only) — they must
NEVER silently fall back to Claude for synthesis.

The generate fn runs the server's model in NON-THINKING mode (chat_template_kwargs
enable_thinking=False) for fast, direct output, at the card's recommended instruct sampling.
"""
import logging
from agent.cost import tracked_local_call, tracked_openai_chat_create

logger = logging.getLogger(__name__)

# Provenance label attached to every synthesized example's _source (data lineage).
SYNTH_SOURCE_LABEL = "synth:vllm"


def _endpoint_config():
    try:
        from config.config import SYNTH_ENDPOINT, SYNTH_MODEL, SYNTH_API_KEY
    except Exception:
        return None, None, None
    return SYNTH_ENDPOINT, SYNTH_MODEL, SYNTH_API_KEY


def _make_client(timeout: float):
    """OpenAI client for the LOCAL endpoint with proxy DISABLED (trust_env=False). Compute
    nodes often export HTTP(S)_PROXY for internet access, which a Squid proxy then applies to
    the internal node:port request and returns a 5xx error page (B161 crash). trust_env=False
    makes httpx ignore proxy/no_proxy/ca env so internal node-to-node HTTP works directly."""
    from openai import OpenAI
    import httpx
    endpoint, model, api_key = _endpoint_config()
    http_client = httpx.Client(trust_env=False, timeout=timeout)
    return OpenAI(base_url=endpoint, api_key=api_key or "EMPTY", timeout=timeout,
                  http_client=http_client), model


def is_available(timeout: float = 8.0, log=print) -> bool:
    """True iff the endpoint responds and serves the configured exact model id."""
    endpoint, model, api_key = _endpoint_config()
    if not endpoint or not model:
        return False
    try:
        client, model = _make_client(timeout)
        response = tracked_local_call(
            client.models.list,
            stage="synth_preflight",
            model=model,
            operation="models.list",
        )
        served_models = {
            str(item.id)
            for item in (getattr(response, "data", None) or [])
            if getattr(item, "id", None)
        }
        if model not in served_models:
            available = ", ".join(sorted(served_models)) or "(none)"
            log(
                f"      [synth] endpoint {endpoint} is reachable but configured "
                f"model {model!r} is not served (available: {available}); "
                "synthesis will be skipped"
            )
            return False
        log(f"      [synth] endpoint {endpoint} reachable (model={model})")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"      [synth] endpoint {endpoint} not reachable ({str(e)[:80]}); "
            f"synthesis will be skipped (gold-only)")
        return False


class SynthesisUnavailableError(RuntimeError):
    """The local synthesis endpoint did not come back within the allowed wait.

    Raised instead of degrading to gold-only. A plan that declared
    ``targeted_synth_positive`` and then silently produced zero synthetic rows is a
    strategy attribution lie: the DAG records the strategy the orchestrator chose while
    the dataset contains none of its output. Stopping is the honest outcome — the run
    checkpoints, the endpoint gets restarted, and the run resumes.
    """


# Mid-run wait. Deliberately shorter than the startup preflight (SLM_SYNTH_WAIT_S, 40 min
# default): at startup nothing has been spent yet, but mid-run every minute of blocking is
# charged against the aggregate wall-clock guard that has to leave room to write summaries.
# 10 minutes covers a vLLM restart or a transient node blip without eating a training slot.
MIDRUN_WAIT_ENV = "SLM_SYNTH_MIDRUN_WAIT_S"
DEFAULT_MIDRUN_WAIT_S = 600.0
DEFAULT_POLL_INTERVAL_S = 15.0


def _midrun_wait_s() -> float:
    import os

    try:
        return max(0.0, float(os.environ.get(MIDRUN_WAIT_ENV, DEFAULT_MIDRUN_WAIT_S)))
    except (TypeError, ValueError):
        return DEFAULT_MIDRUN_WAIT_S


def _wallclock_remaining_s() -> float | None:
    """Seconds left in the run's aggregate wall-clock budget, or None when unbounded."""
    try:
        from agent.nodes.iterate import (
            _WALLCLOCK_BUDGET_S,
            _wallclock_elapsed_s,
        )
    except Exception:  # noqa: BLE001 - never let observability break synthesis
        return None
    if not _WALLCLOCK_BUDGET_S or _WALLCLOCK_BUDGET_S <= 0:
        return None
    return max(0.0, float(_WALLCLOCK_BUDGET_S) - _wallclock_elapsed_s())


def wait_until_available(
    timeout_s: float | None = None,
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    probe_timeout: float = 8.0,
    log=print,
    sleep=None,
) -> bool:
    """Block until the synthesis endpoint serves the configured model, or the wait expires.

    Returns True as soon as it is reachable. Never raises for an unreachable endpoint —
    callers decide whether that is fatal.

    The wait is clamped to the remaining aggregate wall-clock budget so blocking here can
    never consume the reserve the run needs to checkpoint and write its summary. If the
    budget is already spent, this returns immediately rather than sleeping into a hard kill.
    """
    import time as _time

    sleeper = sleep or _time.sleep
    if timeout_s is None:
        timeout_s = _midrun_wait_s()

    remaining_budget = _wallclock_remaining_s()
    if remaining_budget is not None:
        # Keep a 5-minute reserve for graceful termination.
        usable = max(0.0, remaining_budget - 300.0)
        if usable < timeout_s:
            log(
                f"      [synth] clamping endpoint wait {timeout_s:.0f}s → {usable:.0f}s "
                "to preserve the wall-clock reserve for checkpoint/summary"
            )
            timeout_s = usable

    if is_available(timeout=probe_timeout, log=log):
        return True
    if timeout_s <= 0:
        return False

    deadline = _time.monotonic() + timeout_s
    attempt = 1
    log(
        f"      [synth] endpoint unavailable — blocking up to {timeout_s / 60:.1f} min "
        f"for it to return (polling every {poll_interval_s:.0f}s). "
        "Synthesis is required; the run will stop rather than silently go gold-only."
    )
    while _time.monotonic() < deadline:
        sleeper(min(poll_interval_s, max(0.0, deadline - _time.monotonic())))
        attempt += 1
        if is_available(timeout=probe_timeout, log=log):
            log(f"      [synth] endpoint recovered on attempt {attempt} — continuing")
            return True
    return False


def get_generate_fn(log=print, request_timeout: float = 120.0):
    """Return a `generate(prompt, temperature=0.7, max_tokens=200) -> str` backed by the
    local vLLM endpoint, or None if unavailable. The fn raises on per-call failure so the
    caller can count/skip; it never falls back to Claude."""
    endpoint, model, api_key = _endpoint_config()
    if not endpoint:
        return None
    try:
        client, model = _make_client(request_timeout)
    except Exception:
        return None

    def generate(prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        resp = tracked_openai_chat_create(
            client,
            stage="local_synthesis",
            provider="local",
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.80,
            extra_body={
                "top_k": 20,
                # Non-thinking mode → direct answer, no <think> preamble to strip.
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return (resp.choices[0].message.content or "").strip()

    log(f"      [synth] using local synthesis endpoint {endpoint} (model={model})")
    return generate
