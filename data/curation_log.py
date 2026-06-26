# data/curation_log.py
from datetime import datetime
from eval.harness import EvalResult


class CurationLog:
    """Reads and writes data-curation.md — the agent's durable lineage artifact.

    This is the primary lineage artifact per paper §2.1. It survives context compaction
    cycles and can be re-read by the agent at any point. Records dataset versions,
    composition ratios, quality-control decisions, per-iteration eval results, and
    hardware constraint PASS/FAIL status (design doc §4.3).
    """

    def __init__(self, path: str = "data-curation.md"):
        self.path = path

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
        int4_size_mb: int,
        tier: int,
        hardware_notes: str = "Phase 1: theoretical",
        hw_constraints: dict | None = None,
        failure_taxonomy: str = "",
    ) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        total = n_gold + n_hard if (n_gold + n_hard) > 0 else 1

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

        # Format failure taxonomy (top failure patterns)
        taxonomy_section = ""
        if failure_taxonomy:
            taxonomy_section = f"\n### Failure taxonomy\n{failure_taxonomy}\n"
        elif eval_result.failures:
            from collections import Counter
            patterns = Counter()
            for f in eval_result.failures[:50]:
                key = f"{f.get('label', '?')}→{f.get('predicted', '?')}"
                patterns[key] += 1
            lines = [f"  - {pat}: {cnt} failures" for pat, cnt in patterns.most_common(10)]
            taxonomy_section = "\n### Failure taxonomy (auto-generated)\n" + "\n".join(lines) + "\n"

        entry = f"""
## Iteration {iteration} — {timestamp}

### Dataset
- Task type: {task_type}
- Version: {dataset_version}
- Total examples: {n_gold + n_hard}
- Dgold: {n_gold} ({n_gold / total * 100:.0f}%)
- Dhard: {n_hard} ({n_hard / total * 100:.0f}%)
- Distribution: {label_dist}

### Training config (π_{iteration})
- Config A: {config_a}
- Config B: {config_b}
- Best config selected: {best_config}

### Eval results
- f(π_{iteration}): {eval_result.f1:.4f}
- Per class: {eval_result.per_class}
- Epos: {eval_result.pos_score:.4f} | Eneg: {eval_result.neg_score:.4f} | Eboundary: {eval_result.boundary_score:.4f}
- Remaining failures: {len(eval_result.failures)}
{taxonomy_section}
### Iteration policy decision
- Score band: {score_band}
- Next intervention: {next_intervention}
- Hypothesis: {hypothesis}

### Hardware profile ({hardware_notes})
- Model: {model_id} | INT4 size: {int4_size_mb}MB | Tier: {tier}
{hw_lines if hw_lines else "- (Phase 1: theoretical estimates only)"}
---
"""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry)

    def read_latest(self) -> str:
        """Read the full data-curation.md contents."""
        try:
            with open(self.path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""
