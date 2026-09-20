"""
Synthesis client (B161): training-example synthesis and CoT annotation via an OpenAI-compatible
teacher endpoint (config.SYNTH_ENDPOINT), instead of the Claude orchestrator API.

TWO BACKENDS, ONE CLIENT
    LOCAL (default) — a vLLM server co-located with the run, serving Qwen3.6-35B-A3B. Why local:
    (1) no Claude cost, (2) fully reproducible + contamination-safe (a model we own), (3) the 35B
    is strong enough for in-distribution example generation.

    API (config.SYNTH_API_MODE) — the DeepSeek API. No vLLM server is launched, so no GPU is
    reserved for the teacher and there is no 40-minute server warmup. Rows generated this way
    carry a `synth:deepseek` teacher label: their provenance is a model we do not own.

    Neither backend may EVER silently fall back to Claude for synthesis. If the endpoint is not
    configured or unreachable, callers get None and degrade gracefully (gold-only).

The generate fn runs the local model in NON-THINKING mode (chat_template_kwargs
enable_thinking=False) for fast, direct output, at the card's recommended instruct sampling. Those
are vLLM-specific `extra_body` fields and DeepSeek rejects unknown fields outright, so they are
sent only on the local path.
"""
import logging
import os

from agent.cost import tracked_local_call, tracked_openai_chat_create

logger = logging.getLogger(__name__)

# Provenance label attached to every synthesized example's `_teacher` field (data lineage).
SYNTH_SOURCE_LABEL = "synth:vllm"
SYNTH_API_SOURCE_LABEL = "synth:deepseek"


def _api_mode() -> bool:
    """Whether the teacher is a hosted API rather than the co-located vLLM server.

    Read through config rather than os.environ so a test that monkeypatches `config.SYNTH_API_MODE`
    moves this module with it — the endpoint constants it pairs with are resolved at config import
    and cannot be changed by setting the environment variable afterwards.
    """
    try:
        from config.config import SYNTH_API_MODE
    except Exception:  # noqa: BLE001 — config may be unimportable in a bare unit test
        return False
    return bool(SYNTH_API_MODE)


def synth_source_label() -> str:
    """Which teacher produced a row, for data lineage. See `_api_mode`."""
    return SYNTH_API_SOURCE_LABEL if _api_mode() else SYNTH_SOURCE_LABEL


def _cost_provider() -> str:
    """Provider tag for the cost ledger: local vLLM is $0, DeepSeek is priced per token."""
    if not _api_mode():
        return "local"
    from config.config import SYNTH_API_PROVIDER

    return SYNTH_API_PROVIDER


def _endpoint_config():
    try:
        from config.config import SYNTH_ENDPOINT, SYNTH_MODEL, SYNTH_API_KEY
    except Exception:
        return None, None, None
    return SYNTH_ENDPOINT, SYNTH_MODEL, SYNTH_API_KEY


def _connection_pool_size() -> int:
    """Keep-alive slots to hold open, sized to the synthesis fan-out.

    Synthesis fans out over a thread pool of SLM_SYNTH_CONCURRENCY workers sharing one client.
    httpx keeps only 20 connections alive by default, so once concurrency exceeds that the
    surplus workers pay a fresh TCP handshake on every single row. Holding one slot per worker
    (plus headroom) makes the pool a non-event instead of a per-call tax.
    """
    import os
    try:
        concurrency = int(os.environ.get("SLM_SYNTH_CONCURRENCY", "16"))
    except (TypeError, ValueError):
        concurrency = 16
    return max(32, concurrency + 8)


def _make_client(timeout: float):
    """OpenAI client for the configured teacher endpoint.

    `trust_env` is INVERTED between the two backends and both directions are load-bearing:

      * LOCAL — proxy DISABLED. Compute nodes often export HTTP(S)_PROXY for internet access, which
        a Squid proxy then applies to the internal node:port request and returns a 5xx error page
        (B161 crash). trust_env=False makes httpx ignore proxy/no_proxy/ca env so internal
        node-to-node HTTP works directly.
      * API — proxy ENABLED. Reaching api.deepseek.com from a compute node REQUIRES exactly the
        proxy the local path has to avoid, so trust_env=False here would make every call time out.
    """
    from openai import OpenAI
    import httpx
    endpoint, model, api_key = _endpoint_config()
    pool = _connection_pool_size()
    http_client = httpx.Client(
        trust_env=_api_mode(),
        timeout=timeout,
        limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool),
    )
    return OpenAI(base_url=endpoint, api_key=api_key or "EMPTY", timeout=timeout,
                  http_client=http_client), model


def is_available(timeout: float = 8.0, log=print) -> bool:
    """True iff the endpoint responds and serves the configured model id.

    A hosted API lists models it can route to rather than the one process it loaded, so the check
    is a membership test either way. The one relaxation for API mode: an endpoint that returns an
    EMPTY list is accepted with a warning, because `/models` is optional in the OpenAI spec and
    refusing there would mean a working key is reported as an unreachable teacher. A non-empty list
    that omits the configured model is still a hard no — that is a typo in the model name, and
    finding out at the first synthesis call instead of at preflight helps nobody.

    REPORTS THE PROBE, NEVER THE CONSEQUENCE. This is one probe of a possibly-unstarted server,
    and it does not know what its caller will do with a False — `wait_until_available` polls it
    every 15s for up to 40 minutes and the run then proceeds WITH synthesis. It used to log
    "synthesis will be skipped (gold-only)" on every failed probe, so a run waiting normally for
    vLLM to finish loading a 35B model printed that line repeatedly, directly beneath its caller's
    own "Synthesis is required; the run will stop rather than silently go gold-only". Two
    contradictory claims about the same moment, and the wrong one was the one repeated. Observed on
    run 39881378, where it read as a degraded run and was not one. A log line that misreports state
    costs whatever the reader decides on it, which here is a cancelled healthy run.
    """
    endpoint, model, api_key = _endpoint_config()
    if not endpoint or not model:
        return False
    api = _api_mode()
    try:
        client, model = _make_client(timeout)
        response = tracked_local_call(
            client.models.list,
            stage="synth_preflight",
            model=model,
            operation="models.list",
            provider=_cost_provider(),
        )
        served_models = {
            str(item.id)
            for item in (getattr(response, "data", None) or [])
            if getattr(item, "id", None)
        }
        if not served_models and api:
            log(f"      [synth] API endpoint {endpoint} reachable but lists no models; "
                f"proceeding with configured model={model}")
            return True
        if model not in served_models:
            available = ", ".join(sorted(served_models)) or "(none)"
            # A served-model mismatch IS terminal — polling cannot fix a wrong model name — but
            # saying so is the caller's job, and `wait_until_available` reports it once at the end.
            log(
                f"      [synth] endpoint {endpoint} is reachable but configured "
                f"model {model!r} is not served (available: {available})"
            )
            return False
        log(f"      [synth] {'API' if api else 'endpoint'} {endpoint} reachable (model={model})")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"      [synth] endpoint {endpoint} not reachable yet ({str(e)[:80]})")
        return False


class SynthesisUnavailableError(RuntimeError):
    """The local synthesis endpoint did not come back within the allowed wait.

    Retained for compatibility. As of the 2026-07-31 redesign, curate no longer raises this
    on a missing endpoint: the ``synthesize`` strategy degrades GRACEFULLY (fewer rows, an honest
    ``allocation_fallbacks`` entry, gold rows from the train pool for the remainder) rather than
    stopping the run. Synthesis is ungated across task types and score bands.
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

    # A hosted API has no cold start and no restart to wait through. If it is unreachable the
    # cause is a bad key, a dead network route or an outage, and none of those resolve by polling
    # a compute node's clock for ten minutes — so probe once and report the answer.
    if _api_mode():
        return is_available(timeout=probe_timeout, log=log)

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


def _sampling_extra_body(api: bool, provider: str, *, log=print) -> dict | None:
    """Provider-specific request fields, chiefly: TURN THINKING OFF.

    Both teachers reason by default and neither should here. Synthesis asks for one training row in
    a strict output format; a reasoning trace in front of it is tokens we pay for and then throw
    away, and the row still has to survive an exact verifier afterwards.

    The two teachers spell the switch completely differently, and the old code only knew the local
    spelling:

        local vLLM   chat_template_kwargs={"enable_thinking": False}   (a vLLM extension)
        DeepSeek     thinking={"type": "disabled"}                     (a DeepSeek API field)

    `chat_template_kwargs` and `top_k` are vLLM extensions, not OpenAI-spec parameters, and sending
    them to DeepSeek 400s every call — which is why API mode used to send `extra_body=None` and
    therefore never disabled thinking at all.

    WHAT THAT COST, measured on run 39562029 (ner_bc5cdr, deepseek-v4-flash). 2,480 calls produced
    10.95M output tokens for ~1,440 rows — about 7,100 output tokens per generated row, for what is
    a sentence plus a short JSON list. 92% of the arm's $22.63 went to synthesis at $0.0048 a row.
    DeepSeek's docs are explicit that reasoning tokens count inside the completion budget, so a
    call can return HTTP 200 with NO final answer once `max_tokens` is spent on thinking — which is
    also the most likely explanation for that run's format_valid=0.7040 against the local teacher's
    1.0000. Both the bill and the format failures point at the same missing flag.

    Overridable with SLM_SYNTH_API_THINKING=enabled for a task that genuinely wants a reasoning
    teacher, and because a future model may rename or drop the field; an unknown value is refused
    here rather than sent and 400'd mid-run.
    """
    if not api:
        return {
            "top_k": 20,
            # Non-thinking mode → direct answer, no <think> preamble to strip.
            "chat_template_kwargs": {"enable_thinking": False},
        }
    if provider != "deepseek":
        # An OpenAI-compatible endpoint we have not characterised. Sending a DeepSeek-specific
        # field to it is how the local flags broke API mode in the first place.
        return None
    mode = str(os.environ.get("SLM_SYNTH_API_THINKING", "disabled")).strip().lower()
    if mode not in {"disabled", "enabled"}:
        raise ValueError(
            f"SLM_SYNTH_API_THINKING must be 'disabled' or 'enabled', got {mode!r}"
        )
    if mode == "enabled":
        log("      [synth] DeepSeek thinking mode is ENABLED by SLM_SYNTH_API_THINKING — "
            "reasoning tokens are billed inside the completion budget and can consume max_tokens "
            "before a final answer is produced")
    return {"thinking": {"type": mode}}


def judge_extra_body(api_mode: bool, *, log=print) -> dict | None:
    """The same thinking-off request fields for the eval judge, minus the sampling controls.

    The judge does not want `top_k`: it is scoring, not generating, and its temperature/top-p are
    set deliberately by the caller. What it does want is exactly what synthesis wants — no
    reasoning trace — and it was getting it locally and not on the API for the same reason.

    Lives beside the synthesis version so a future change to how thinking is disabled cannot be
    applied to one caller and forgotten on the other.
    """
    if api_mode:
        # Resolved from config rather than hardcoded to "deepseek", so pointing the teacher at
        # another OpenAI-compatible provider stops sending a DeepSeek-only field rather than
        # 400ing every judge call.
        try:
            from config.config import SYNTH_API_PROVIDER as provider
        except Exception:  # noqa: BLE001 — config may be unimportable in a bare unit test
            provider = "deepseek"
    else:
        provider = "local"
    body = _sampling_extra_body(api_mode, provider, log=log)
    if body is None:
        return None
    return {k: v for k, v in body.items() if k != "top_k"}


def get_generate_fn(log=print, request_timeout: float = 120.0):
    """Return a `generate(prompt, temperature=0.7, max_tokens=None) -> str` backed by the
    configured teacher endpoint, or None if unavailable. The fn raises on per-call failure so the
    caller can count/skip; it never falls back to Claude."""
    endpoint, model, api_key = _endpoint_config()
    if not endpoint:
        return None
    api = _api_mode()
    try:
        client, model = _make_client(request_timeout)
    except Exception:
        return None

    provider = _cost_provider()
    stage = "api_synthesis" if api else "local_synthesis"
    extra_body = _sampling_extra_body(api, provider, log=log)

    # Synthesis samples at temperature 0.7, so without a seed the curriculum itself differs
    # between two runs of identical code — the single largest reproducibility hole in the
    # pipeline, and a bigger one than the kernel nondeterminism probe 40222458 found. A COUNTER
    # rather than one fixed seed: a constant seed would return the same row for the same prompt
    # and collapse the diversity synthesis exists to provide. vLLM honours per-request `seed`;
    # the DeepSeek API does not, so API-mode runs stay unreproducible and say so below.
    from training.determinism import seed_sequence

    seeds = seed_sequence(f"synth:{model}")

    def generate(prompt: str, temperature: float = 0.7, max_tokens: int | None = None) -> str:
        # None, not 200. A caller that omits the argument used to silently receive 200 output
        # tokens, which is where the habit of guessing this number came from; it now receives as
        # much as the served context can return for its prompt.
        if max_tokens is None:
            from config.token_budget import output_budget

            max_tokens = output_budget(prompt)
        kwargs = {}
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        if not api:
            kwargs["seed"] = seeds.next()
        resp = tracked_openai_chat_create(
            client,
            stage=stage,
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.80,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

    log(f"      [synth] using {'DeepSeek API' if api else 'local'} synthesis endpoint "
        f"{endpoint} (model={model})")
    if api:
        log("      [synth] WARNING: the API teacher accepts no seed parameter, so the generated "
            "rows in this run are NOT reproducible. Use the local teacher for a paper run.")
    else:
        log(f"      [synth] generation seeds drawn from SLM_SEED (stream 'synth:{model}'), so "
            "rerunning this configuration regenerates the same rows")
    return generate
