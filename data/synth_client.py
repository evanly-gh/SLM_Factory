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
