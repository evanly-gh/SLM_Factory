# data/curation_log.py
import hashlib
import os
from datetime import datetime
from agent.checkpoint import atomic_write_text
from eval.harness import EvalResult


class CurationLog:
    """Reads and writes data-curation.md — the agent's durable lineage artifact.

    This is the primary lineage artifact per paper §2.1. It survives context compaction
    cycles and can be re-read by the agent at any point. Records dataset versions,
    composition ratios, quality-control decisions, per-iteration eval results, and
    hardware constraint PASS/FAIL status (design doc §4.3).
    """

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = os.fspath(
            path
            if path is not None
            else os.environ.get(
                "SLM_CURATION_LOG_PATH",
                "data-curation.md",
            )
        )

    def write_iteration(
        self,
        iteration: int,
        task_type: str,
        dataset_version: str,
        n_gold: int,
        n_hard: int,
        label_dist: dict,
        config_a: str,
        config_b: str,
        best_config: str,
        eval_result: EvalResult,
        score_band: str,
        next_intervention: str,
        hypothesis: str,
        model_id: str,
        size_mb: int,
        tier: int,
        total_examples: int | None = None,
        n_hard_source: int = 0,
        n_hard_generated: int | None = None,
        rebuild_plan_identity: str = "",
        strategy_composition: list[dict] | None = None,
        source_novelty: dict | None = None,
        plan_yield: dict | None = None,
        source_usage: list[dict] | None = None,
        confusion_pairs: list[dict] | None = None,
        hardware_notes: str = "Phase 1: theoretical",
        hw_constraints: dict | None = None,
        entry_id: str | None = None,
    ) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        generated = n_hard if n_hard_generated is None else n_hard_generated
        actual_total = (
            total_examples
            if total_examples is not None
            else n_gold + n_hard_source + generated
        )
        ratio_total = actual_total if actual_total > 0 else 1

        # Format hardware PASS/FAIL lines (design doc §4.3)
        hw_lines = ""
        if hw_constraints:
            for key in ("storage", "memory", "latency", "power"):
                c = hw_constraints.get(key, {})
                status = "PASS" if c.get("pass", True) else "FAIL"
                if key == "storage":
                    hw_lines += f"- Storage: {c.get('value_mb', '?')}MB vs S_max={c.get('limit_mb', '?')}MB — {status}\n"
                elif key == "memory":
                    hw_lines += f"- Memory: {c.get('value_mb', '?')}MB vs M_max={c.get('limit_mb', '?')}MB — {status}\n"
                elif key == "latency":
                    hw_lines += f"- Latency: {c.get('estimated_ttft_ms', '?')}ms vs L_max={c.get('limit_ms', '?')}ms ({c.get('tok_s', '?')} tok/s on {c.get('chip', '?')}) — {status}\n"
                elif key == "power":
                    hw_lines += f"- Power: {c.get('note', 'not measured')} — {status}\n"

        # Data sources used this build — only when NEW external data was fetched (an entry
        # carries a url). Pure resample/synthesize builds have no linked source and omit it.
        data_sources_section = ""
        if source_usage and any(entry.get("url") for entry in source_usage if isinstance(entry, dict)):
            source_lines = []
            for entry in source_usage:
                if not isinstance(entry, dict):
                    continue
                where = entry.get("url") or entry.get("source", "?")
                novel = int(entry.get("novel_rows", 0) or 0)
                novel_text = f" (novel: {novel})" if novel else ""
                source_lines.append(
                    f"- {where} — {int(entry.get('rows', 0) or 0)} rows{novel_text}"
                )
            data_sources_section = (
                "\n### Data sources\n" + "\n".join(source_lines) + "\n"
            )

        # Aggregate confusion (top failure patterns)
        taxonomy_section = ""
        if confusion_pairs:
            lines = [
                "  - "
                f"{pair.get('gold', '?')}→{pair.get('predicted', '?')}: "
                f"{int(pair.get('count', 0) or 0)} failures"
                for pair in confusion_pairs[:10]
                if isinstance(pair, dict)
            ]
            taxonomy_section = (
                "\n### Aggregate confusion counts\n"
                + "\n".join(lines)
                + "\n"
            )
        elif eval_result.failures:
            taxonomy_section = (
                "\n### Aggregate failure report\n"
                f"  - {len(eval_result.failures)} failure row(s); raw eval "
                "content omitted by firewall\n"
            )

        marker = (
            hashlib.sha256(entry_id.encode("utf-8")).hexdigest()
            if entry_id
            else None
        )
        marker_text = (
            f"<!-- slm-curation-entry:{marker} -->"
            if marker
            else ""
        )
        entry = f"""
## Iteration {iteration} — {timestamp}

### Dataset
- Task type: {task_type}
- Version: {dataset_version}
- Total examples: {actual_total}
- Initial gold: {n_gold} ({n_gold / ratio_total * 100:.0f}%)
- Source anchors: {n_hard_source} ({n_hard_source / ratio_total * 100:.0f}%)
- Generated hard rows: {generated} ({generated / ratio_total * 100:.0f}%)
- Rebuild plan identity: {rebuild_plan_identity or "n/a"}
- Strategy composition: {strategy_composition or []}
- Source novelty: {source_novelty or {}}
- Plan yield: {plan_yield or {}}
- Distribution: {label_dist}
{data_sources_section}
### Training config (π_{iteration})
- Config A: {config_a}
- Config B: {config_b}
- Best config selected: {best_config}

### Eval results
- f(π_{iteration}): {eval_result.f1:.4f}
- Per class: {eval_result.per_class}
- Remaining failures: {len(eval_result.failures)}
{taxonomy_section}
### Iteration policy decision
- Score band: {score_band}
- Next intervention: {next_intervention}
- Hypothesis: {hypothesis}

### Hardware profile ({hardware_notes})
- Model: {model_id} | Weight size: {size_mb}MB | Tier: {tier}
{hw_lines if hw_lines else "- (Phase 1: theoretical estimates only)"}
---
{marker_text}
"""
        try:
            with open(self.path, encoding="utf-8") as source:
                existing = source.read()
        except FileNotFoundError:
            existing = ""
        if marker_text and marker_text in existing:
            return
        atomic_write_text(self.path, existing + entry)

    def read_latest(self) -> str:
        """Read the full data-curation.md contents."""
        try:
            with open(self.path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""
