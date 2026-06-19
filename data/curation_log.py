# data/curation_log.py
from datetime import datetime
from eval.harness import EvalResult


class CurationLog:
    """Reads and writes data-curation.md — the agent's durable lineage artifact."""

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
    ) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        total = n_gold + n_hard if (n_gold + n_hard) > 0 else 1
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

### Iteration policy decision
- Score band: {score_band}
- Next intervention: {next_intervention}
- Hypothesis: {hypothesis}

### Hardware profile ({hardware_notes})
- Model: {model_id} | INT4 size: {int4_size_mb}MB | Tier: {tier}

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
