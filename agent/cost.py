# agent/cost.py
"""
Tiny API cost ledger for the autonomous loop.

Tracks Claude (Anthropic) token usage and Exa search calls, and converts them to
an estimated USD cost. Prices follow the Pioneer Agent paper (arXiv:2604.09791v1,
Section 6.1): Claude Sonnet 4.6 at $3 / 1M input tokens and $15 / 1M output tokens.
Exa search is billed per request; the paper folds it into a ~$1-2/run line item,
so we use a per-search estimate that can be tuned.
"""
import threading

# --- pricing (USD) -------------------------------------------------------
SONNET_INPUT_PER_M = 3.0      # $/1M input tokens   (Sonnet 4.6)
SONNET_OUTPUT_PER_M = 15.0    # $/1M output tokens  (Sonnet 4.6)
EXA_SEARCH_COST = 0.005       # estimated $/search (neural/keyword) — tune as needed


class CostLedger:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.anthropic_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.exa_calls = 0

    def record_anthropic(self, input_tokens: int, output_tokens: int):
        with self._lock:
            self.anthropic_calls += 1
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)

    def record_exa(self, n: int = 1):
        with self._lock:
            self.exa_calls += n

    @property
    def anthropic_cost(self) -> float:
        return (self.input_tokens / 1e6) * SONNET_INPUT_PER_M + \
               (self.output_tokens / 1e6) * SONNET_OUTPUT_PER_M

    @property
    def exa_cost(self) -> float:
        return self.exa_calls * EXA_SEARCH_COST

    @property
    def total_cost(self) -> float:
        return self.anthropic_cost + self.exa_cost

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "anthropic_calls": self.anthropic_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "anthropic_cost_usd": round(self.anthropic_cost, 4),
                "exa_calls": self.exa_calls,
                "exa_cost_usd": round(self.exa_cost, 4),
                "total_cost_usd": round(self.total_cost, 4),
            }


# Process-wide singleton used by the cost-tracking client wrappers.
LEDGER = CostLedger()
