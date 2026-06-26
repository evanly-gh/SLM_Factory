# agent/context_manager.py
"""
Context Manager for sustaining long agentic runs (paper §2.1).

The paper's Context Manager "monitors the conversation state and selectively compacts
older turns while preserving key decisions, evaluation results, and dataset lineage."
The internal mechanism is proprietary and not disclosed.

Our implementation uses structured compaction of the data-curation.md log:
- The last N_RECENT iterations are kept in full detail
- Older iterations are compressed to a one-line summary each
- Key decisions (best score, model escalations, rollbacks) are always preserved
- The compressed version is what gets sent to the LLM in iterate_node;
  the full version stays on disk for lineage

This approach matches the research consensus (LangChain deep agents, JetBrains 2025,
Claude Code auto-compact): preserve recent context in full, summarize older turns,
keep structured decisions intact.

References:
- Paper §2.1: "selectively compacts older turns while preserving key decisions,
  evaluation results, and dataset lineage"
- Design doc §2.2: "Context Manager monitors conversation state"
- LangGraph context engineering: structured compaction with key field preservation
"""
import re


N_RECENT = 3


def compact_trajectory(full_trajectory: str, n_recent: int = N_RECENT) -> str:
    """
    Compact a data-curation.md trajectory for the LLM's context window.

    Keeps the last n_recent iterations in full detail. Older iterations are
    compressed to a one-line summary preserving: iteration number, score,
    intervention, and hypothesis. This prevents the LLM call in iterate_node
    from growing linearly with iteration count.

    The full uncompacted file stays on disk — this function only produces
    the version sent to the LLM.
    """
    if not full_trajectory.strip():
        return full_trajectory

    sections = re.split(r'(?=^## Iteration \d+)', full_trajectory, flags=re.MULTILINE)
    non_iteration = []
    iterations = []

    for section in sections:
        if section.strip().startswith("## Iteration"):
            iterations.append(section)
        else:
            non_iteration.append(section)

    if len(iterations) <= n_recent:
        return full_trajectory

    old = iterations[:-n_recent]
    recent = iterations[-n_recent:]

    summaries = []
    for section in old:
        summary = _extract_iteration_summary(section)
        summaries.append(summary)

    header = "".join(non_iteration).strip()
    compact_section = "\n## Compacted history (older iterations)\n" + "\n".join(summaries)
    recent_section = "\n\n## Recent iterations (full detail)\n" + "\n".join(recent)

    result = header + "\n" + compact_section + "\n" + recent_section
    return result


def _extract_iteration_summary(section: str) -> str:
    """Extract a one-line summary from a full iteration section."""
    iter_match = re.search(r'Iteration (\d+)', section)
    iter_num = iter_match.group(1) if iter_match else "?"

    score_match = re.search(r'f\(π_\d+\):\s*([\d.]+)', section)
    score = score_match.group(1) if score_match else "?"

    intervention_match = re.search(r'Next intervention:\s*(.+)', section)
    intervention = intervention_match.group(1).strip() if intervention_match else "?"

    hypothesis_match = re.search(r'Hypothesis:\s*(.+)', section)
    hypothesis = hypothesis_match.group(1).strip()[:100] if hypothesis_match else ""

    band_match = re.search(r'Score band:\s*(.+)', section)
    band = band_match.group(1).strip() if band_match else "?"

    model_match = re.search(r'Model:\s*(\S+)', section)
    model = model_match.group(1) if model_match else "?"

    return (
        f"- Iter {iter_num}: f(π)={score}, band={band}, "
        f"intervention={intervention}, model={model}"
        + (f" — {hypothesis}" if hypothesis else "")
    )


def estimate_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def should_compact(trajectory: str, max_tokens: int = 8000) -> bool:
    """Check if the trajectory exceeds the target token budget for LLM calls."""
    return estimate_token_count(trajectory) > max_tokens
