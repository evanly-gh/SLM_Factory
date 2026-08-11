# agent/run_memory.py
"""
Structured run memory for the orchestrator prompt.

Replaces the raw ``data-curation.md`` dump that used to be pasted into every iterate call.
That dump answered "what is in the log?"; this answers the question the orchestrator is
actually being asked — *what have I already tried, what worked, and what has failed since?*

Why the dump was inadequate (measured on slm-clinc150-cse-38155022):

- **Success and failure were indistinguishable.** Iteration 2 (a new best) and iteration 6
  (rolled back) rendered identically. Nothing marked the outcome or the delta, so "remember
  why it succeeded / why it failed" was not answerable from the context.
- **No notion of "since the last improvement".** Seventeen consecutive failures appeared as
  seventeen unrelated rows; the pattern — ``synthesize`` tried fifteen times and never once
  kept — was present but invisible.
- **The prose was truncated garbage.** Each older row was cut to 100 characters mid-word, and
  an empty hypothesis captured the following markdown heading (B238).

Source of truth is ``state["dag"]``, which is append-only and marks discarded attempts with
``pruned`` rather than deleting them, so rolled-back iterations remain visible here even though
``state["scores"]`` pops them. ``escalate``/``largest_first`` reset the DAG on a tier change,
which is intended: this memory is per-model, matching the escalation policy's own scope.

Nothing here is character-truncated. Bounding is by *how many attempts are described in full*,
never by cutting a hypothesis mid-sentence — that is the failure mode this module exists to end.
"""
from __future__ import annotations

from typing import Any


# Failures older than this are still counted and aggregated, just not narrated individually.
# Bounds the block without ever cutting an individual hypothesis.
MAX_DETAILED_FAILURES = 5


def _plan_strategy(node: dict) -> str:
    plan = ((node.get("pi") or {}).get("D") or {}).get("plan") or {}
    return str(plan.get("strategy") or "")


def _label(node: dict) -> str:
    """
    `data_rebuild/synthesize`, or plain `hyperparameter`.

    The sub-strategy is only meaningful for a data_rebuild. `pi.D.plan` persists across
    iterations, so a hyperparameter node still carries the plan from whichever rebuild last
    ran — rendering that as `hyperparameter/acquire` would attribute a data strategy to an
    iteration that never rebuilt the data. That is the same class of false-memory defect as
    B237 and would undo the fix.
    """
    intervention = str(node.get("intervention") or "?")
    if intervention != "data_rebuild":
        return intervention
    strategy = _plan_strategy(node)
    return f"{intervention}/{strategy}" if strategy else intervention


def _annotate(dag: list[dict]) -> list[dict]:
    """
    Walk the DAG in order, attaching the running best and the kept/discarded verdict.

    An attempt is KEPT when it set a new best at the moment it ran. That is decided here from
    the score sequence rather than read from `pruned`, because `pruned` is only written when a
    rollback actually fires; a non-improving attempt that was not rolled back is still not a
    thing the orchestrator should treat as progress.
    """
    annotated: list[dict] = []
    best = float("-inf")
    for node in dag:
        if not isinstance(node, dict):
            continue
        try:
            score = float(node.get("score"))
        except (TypeError, ValueError):
            continue
        previous_best = best
        kept = score > best
        if kept:
            best = score
        annotated.append({
            "iteration": node.get("iteration"),
            "label": _label(node),
            "score": score,
            "delta": score - previous_best if previous_best > float("-inf") else score,
            "kept": kept,
            "hypothesis": str(node.get("hypothesis") or "").strip(),
            "node": node,
        })
    return annotated


def _difficulty_line(node: dict) -> str:
    report = ((node.get("evaluation_state") or {}).get("test_report") or {})
    buckets = report.get("by_difficulty") or {}
    if not isinstance(buckets, dict) or not buckets:
        return ""
    parts = []
    for name in ("easy", "medium", "hard"):
        entry = buckets.get(name)
        if isinstance(entry, dict) and entry.get("accuracy") is not None:
            parts.append(f"{name}={float(entry['accuracy']):.3f}(n={entry.get('n', '?')})")
        elif isinstance(entry, (int, float)):
            parts.append(f"{name}={float(entry):.3f}")
    return "  by bucket: " + "  ".join(parts) if parts else ""


def _confusion_line(node: dict) -> str:
    report = ((node.get("evaluation_state") or {}).get("test_report") or {})
    pairs = [p for p in (report.get("confusion_pairs") or []) if isinstance(p, dict)]
    if not pairs:
        return ""
    top = sorted(pairs, key=lambda p: -int(p.get("count", 0) or 0))[:5]
    rendered = ", ".join(
        f"{p.get('gold')}->{p.get('predicted')} ({p.get('count')})" for p in top
    )
    return f"  top confusions: {rendered}"


def _most_recent_section(entry: dict) -> list[str]:
    node = entry["node"]
    verdict = "KEPT (new best)" if entry["kept"] else "ROLLED BACK"
    dataset = (node.get("pi") or {}).get("D") or {}
    lines = [
        "## MOST RECENT ITERATION (full detail)",
        f"  iteration {entry['iteration']} | {entry['label']} | "
        f"f(pi)={entry['score']:.4f} (delta {entry['delta']:+.4f}) | {verdict}",
    ]
    if dataset.get("version") is not None:
        lines.append(f"  dataset v{dataset.get('version')}")
    for line in (_difficulty_line(node), _confusion_line(node)):
        if line:
            lines.append(line)
    if entry["hypothesis"]:
        lines.append(f"  its stated reasoning: {entry['hypothesis']}")
    return lines


def _what_worked_section(entries: list[dict]) -> list[str]:
    kept = [e for e in entries if e["kept"]]
    lines = ["## WHAT WORKED (kept improvements, oldest first)"]
    if not kept:
        lines.append("  (nothing has improved on the baseline yet)")
        return lines
    for entry in kept:
        lines.append(
            f"  iter {entry['iteration']}  {entry['label']}  "
            f"-> {entry['score']:.4f} ({entry['delta']:+.4f})"
        )
        if entry["hypothesis"]:
            lines.append(f"      because: {entry['hypothesis']}")
    return lines


def _failed_since_section(entries: list[dict]) -> list[str]:
    last_kept = max((i for i, e in enumerate(entries) if e["kept"]), default=-1)
    failures = entries[last_kept + 1:]
    if not failures:
        return ["## FAILED SINCE THE LAST IMPROVEMENT", "  (none — the last attempt improved)"]

    lines = [f"## FAILED SINCE THE LAST IMPROVEMENT ({len(failures)} attempts, none kept)"]

    grouped: dict[str, list[float]] = {}
    for entry in failures:
        grouped.setdefault(entry["label"], []).append(entry["delta"])
    for label, deltas in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        mean = sum(deltas) / len(deltas)
        lines.append(
            f"  {label}  x{len(deltas)}   deltas {min(deltas):+.4f} .. {max(deltas):+.4f}   "
            f"mean {mean:+.4f}"
        )

    detailed = failures[-MAX_DETAILED_FAILURES:]
    if detailed:
        lines.append(f"  --- the {len(detailed)} most recent, in full ---")
        for entry in detailed:
            lines.append(
                f"  iter {entry['iteration']}  {entry['label']}  "
                f"{entry['score']:.4f} ({entry['delta']:+.4f})"
            )
            if entry["hypothesis"]:
                lines.append(f"      it predicted: {entry['hypothesis']}")
    if len(failures) > len(detailed):
        lines.append(
            f"  ({len(failures) - len(detailed)} older failure(s) counted in the totals above)"
        )

    dominant = max(grouped.items(), key=lambda kv: len(kv[1]))
    if len(dominant[1]) >= 3:
        lines.append(
            f"  => {dominant[0]} has been tried {len(dominant[1])}x since the last improvement "
            f"and has never once been kept. Prefer a DIFFERENT intervention type."
        )
    return lines


def _surgical_section(state: dict) -> list[str]:
    history = state.get("surgical_pair_history") or {}
    if not isinstance(history, dict) or not history:
        return []
    current = {
        f"{p.get('gold')}->{p.get('predicted')}": int(p.get("count", 0) or 0)
        for p in ((state.get("test_report") or {}).get("confusion_pairs") or [])
        if isinstance(p, dict)
    }
    lines = ["## SURGICAL SPEND (confusion pairs already targeted)"]
    for key, record in sorted(history.items()):
        if not isinstance(record, dict):
            continue
        before = record.get("count_when_targeted")
        now = current.get(key)
        if now is None:
            verdict = "no longer in the top confusions — RESOLVED"
            movement = f"count {before} -> absent"
        elif before is not None and now >= before:
            verdict = "EXHAUSTED, do not target again"
            movement = f"count {before} -> {now}"
        else:
            verdict = "improving"
            movement = f"count {before} -> {now}"
        lines.append(
            f"  {key}  targeted iter {record.get('iteration')} "
            f"({record.get('rows_generated')} rows), {movement} — {verdict}"
        )
    return lines


def build_run_memory(state: dict[str, Any]) -> str:
    """
    Render the orchestrator's memory of this model's attempts.

    Returns "" when there is no history yet, so the caller can fall back to its own
    "(no iterations logged yet)" placeholder.
    """
    dag = state.get("dag") or []
    entries = _annotate(dag if isinstance(dag, list) else [])
    if not entries:
        return ""

    blocks: list[list[str]] = [
        _most_recent_section(entries[-1]),
        _what_worked_section(entries),
        _failed_since_section(entries),
    ]
    surgical = _surgical_section(state)
    if surgical:
        blocks.append(surgical)

    return "\n\n".join("\n".join(block) for block in blocks)
