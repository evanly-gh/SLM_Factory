# agent/llm_errors.py
"""
Classify LLM API errors so the pipeline can FAIL FAST on unrecoverable ones instead of
silently degrading to score-band fallbacks for the rest of a multi-hour run.

The distinction (B144):
  - FATAL (billing / auth / quota / permission): every subsequent call will also fail, so
    continuing produces a garbage run driven entirely by fallbacks. Raise a clear, actionable
    error and stop.
  - TRANSIENT (timeout, rate limit, overloaded, malformed response): a single fallback is
    reasonable; the run can continue.
"""

_FATAL_SUBSTRINGS = (
    "credit balance is too low",
    "billing",
    "insufficient_quota",
    "exceeded your current quota",
    "invalid x-api-key",
    "authentication",
    "permission",
    "account is not active",
)


class FatalLLMError(RuntimeError):
    """An unrecoverable LLM API error (billing/auth/quota). Stops the run cleanly."""


def is_fatal_llm_error(exc: BaseException) -> bool:
    """True if `exc` looks like a billing/auth/quota error that will recur on every call."""
    msg = str(exc).lower()
    # Anthropic/OpenAI 401/403 and 400-billing all surface these in the message text.
    if any(s in msg for s in _FATAL_SUBSTRINGS):
        return True
    name = type(exc).__name__.lower()
    return "authentication" in name or "permissiondenied" in name


def raise_if_fatal(exc: BaseException, where: str = "") -> None:
    """Re-raise as FatalLLMError if `exc` is unrecoverable; otherwise return (caller may fall back)."""
    if is_fatal_llm_error(exc):
        prefix = f"[{where}] " if where else ""
        raise FatalLLMError(
            f"{prefix}Unrecoverable LLM API error — the run cannot continue meaningfully "
            f"(every orchestrator call would fail and the pipeline would silently degrade to "
            f"score-band fallbacks). Original error: {exc}"
        ) from exc
