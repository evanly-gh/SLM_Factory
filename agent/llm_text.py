# agent/llm_text.py
"""
Read the assistant's TEXT out of an Anthropic response, whatever else the model emitted.

Why this exists (B253): every call site used to read `resp.content[0].text`, which assumes
the first content block is the answer. It is not. When the orchestrator engages extended
thinking, the response is `[ThinkingBlock, TextBlock]` — the answer is at index 1, and a
ThinkingBlock has no `.text` at all. Reading index 0 therefore either raised AttributeError
or, at the sites that guarded with `isinstance(..., TextBlock) else ""`, silently produced an
empty string that downstream parsers turned into "no decision" and a fallback.

Thinking is decided by the model per request, not configured by us: the identical escalate
prompt returned zero thinking tokens on a short candidate list and 255 on the real one. So
any call can start returning a leading thinking block at any time, and reading index 0 is
never safe regardless of which model or max_tokens a call site uses.
"""

# A thinking block that runs to the token limit leaves NO text block behind, so callers get
# an empty string and cannot tell that apart from a model that answered with nothing. Budget
# generously on structured-output calls: unused output tokens are not billed.
MIN_THINKING_SAFE_MAX_TOKENS = 2048


def response_text(resp) -> str:
    """Concatenate every text block of an Anthropic response, in order.

    Non-text blocks (thinking, redacted_thinking, tool_use) are skipped rather than allowed
    to shadow the answer. Returns "" when the model produced no text at all — which, given a
    successful call, means the output budget was spent before it could answer.
    """
    parts = []
    for block in getattr(resp, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def truncated_by_output_budget(resp) -> bool:
    """True if the model stopped because it hit max_tokens rather than finishing its answer."""
    return getattr(resp, "stop_reason", None) == "max_tokens"


def describe_empty_text(resp) -> str:
    """One-line diagnosis for a response that carried no text, for the run log.

    Names the block types and the stop reason so "the orchestrator returned nothing" is
    actionable instead of a mystery.
    """
    blocks = [getattr(b, "type", "?") for b in (getattr(resp, "content", None) or [])]
    stop = getattr(resp, "stop_reason", None)
    detail = f"blocks={blocks or '[]'} stop_reason={stop!r}"
    if truncated_by_output_budget(resp):
        return (
            f"{detail} — the output budget was exhausted before any text was produced "
            f"(extended thinking consumed it); raise max_tokens for this call"
        )
    return detail
