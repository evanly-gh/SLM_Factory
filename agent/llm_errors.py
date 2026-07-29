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
    """An unrecoverable LLM API error. Stops the run cleanly."""


def is_api_transport_error(exc: BaseException) -> bool:
    """True if `exc` came from the Anthropic SDK / HTTP layer rather than our own parsing.

    The distinction that matters: did the API call itself FAIL (no usable response), or
    did it SUCCEED and return something we then rejected? The first means the
    orchestrator never got to decide; continuing would run the pipeline on a substituted
    heuristic while reporting it as an orchestrator-driven run. The second is a
    response-quality problem handled by the reask/validation path.
    """
    # Walk the module path rather than matching on message text: an anthropic.APIError
    # subclass is unambiguous, whereas message matching produces both false positives
    # and false negatives.
    for klass in type(exc).__mro__:
        module = getattr(klass, "__module__", "") or ""
        if module.startswith("anthropic") or module.startswith("httpx"):
            return True
    # Network-level failures that never reach the SDK's own exception types.
    return isinstance(exc, (ConnectionError, TimeoutError))


def is_fatal_llm_error(exc: BaseException) -> bool:
    """True if `exc` must stop the run.

    Policy (tightened 2026-07-28, at the user's direction): ANY genuine API failure is
    fatal, not just billing/auth/quota. Previously a timeout / 500 / overloaded / rate
    limit fell through to a hard-coded heuristic, and the run kept going while its logs
    still attributed each step to the orchestrator. A run whose decisions were silently
    made by a fallback ladder is not a result you can report, so it must stop instead.

    Deliberately NOT fatal: JSON-parse and schema-validation failures. Those mean the API
    answered fine and we rejected the content, which the reask path exists to handle.
    """
    if is_api_transport_error(exc):
        return True
    msg = str(exc).lower()
    if any(s in msg for s in _FATAL_SUBSTRINGS):
        return True
    name = type(exc).__name__.lower()
    return "authentication" in name or "permissiondenied" in name


def raise_if_fatal(exc: BaseException, where: str = "") -> None:
    """Re-raise as FatalLLMError if `exc` is unrecoverable; otherwise return (caller may fall back)."""
    if is_fatal_llm_error(exc):
        prefix = f"[{where}] " if where else ""
        raise FatalLLMError(
            f"{prefix}Claude API call failed — stopping the run rather than continuing on "
            f"substituted heuristics. The orchestrator never produced a decision here, so "
            f"any further iterations would be driven by fallback rules while still being "
            f"reported as an orchestrator-run. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
